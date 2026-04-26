"""End-to-end numerical verification of the gfx1250 → gfx942 transpile
path against the simulator-captured gold corpus.

For every TensileLite gfx1250 code object in the gold corpus (cloned
from the `users/tgymnich/hsaco-runner` branch of rocm-hotswap-testing),
this tool:

  1. Reuses the `.inputs.npz` that was captured on the gfx1250 FFM
     simulator — the exact kernarg buffer and per-buffer contents that
     produced the gold outputs.
  2. Spawns one isolated child process per kernel. The child loads the
     .co through HIP; the Salmon LD_PRELOAD shim patches the ELF's
     e_flags so HIP accepts it, and the Salmon-enabled ROCR runtime
     either pushes it through the LLVM IR raiser
     (`HSA_HOTSWAP_IR_RAISER=1`, `--mode=salmon`, the default) or
     through the byte-level rewriter (`--mode=legacy`). `--mode=native`
     skips the intercept entirely, which is only useful if you build
     same-ISA .co files yourself.
  3. The child launches the kernel on real gfx942 hardware with the
     gold input buffers patched in, D2Hs every output, and diffs it
     against the gold `.outputs.npz` with per-dtype tolerances.
  4. The parent collects per-kernel JSON verdicts and renders an
     aggregate report.

Usage:

    python3 verify_transpile.py \\
      --corpus /path/to/rocm-hotswap-testing \\
      --rocr-build $HOME/rocm-systems/projects/rocr-runtime/build \\
      --salmon-intercept ../compare_correctness/libsalmon_intercept.so

Per-kernel isolation is deliberate — ROCR reads `HSA_HOTSWAP_IR_RAISER`
into a `static const char*` on first use and never re-reads it, and any
state leaking between launches would confuse the comparison. A hung or
crashing kernel (a real possibility on the transpiled path) is killed
by the per-child wall-clock timeout and recorded without blocking the
rest of the sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_DEFAULT_TOLERANCES = {
    "default": [0.0, 0.0],
    "f64":     [1e-8, 1e-6],
    "f32":     [1e-4, 1e-3],
    "f16":     [5e-3, 1e-2],
    "bf16":    [1e-3, 1e-2],
}

_HERE = Path(__file__).resolve().parent
_CHILD_WORKER = _HERE / "_child_worker.py"


@dataclass
class KernelCase:
    """One (.co, kernel) pair to verify."""
    co: Path
    inputs_npz: Path
    outputs_npz: Path
    kernel_name: str
    gold_smoke_ok: bool
    gold_smoke_error: str


@dataclass
class KernelVerdict:
    case: KernelCase
    status: str              # "match" / "mismatch" / "launch-error" / "crash" / "timeout"
    elapsed_s: float
    child_exit: int
    child_signal: int
    child_json: Optional[dict] = None
    stderr_tail: str = ""
    skipped_reason: str = ""


def _discover_cases(corpus: Path, include_failed: bool,
                    kernel_filter: Optional[re.Pattern]) -> tuple[list[KernelCase], list[KernelCase]]:
    """Walk `<corpus>/kernels/tensilelite_io/` and return (runnable, skipped)."""
    co_dir = corpus / "kernels" / "tensilelite"
    io_dir = corpus / "kernels" / "tensilelite_io"
    if not co_dir.is_dir():
        raise SystemExit(f"no such directory: {co_dir}")
    if not io_dir.is_dir():
        raise SystemExit(f"no such directory: {io_dir}")

    runnable: list[KernelCase] = []
    skipped: list[KernelCase] = []

    for sub in sorted(p for p in io_dir.iterdir() if p.is_dir()):
        co = co_dir / f"{sub.name}.co"
        if not co.exists():
            # Dangling I/O dir; surface as a skip so it's visible.
            skipped.append(KernelCase(
                co=co, inputs_npz=Path("<missing>"),
                outputs_npz=Path("<missing>"),
                kernel_name=sub.name,
                gold_smoke_ok=False,
                gold_smoke_error=f"no matching .co in {co_dir}",
            ))
            continue

        for inputs_npz in sorted(sub.glob("*.inputs.npz")):
            stem = inputs_npz.name[: -len(".inputs.npz")]
            outputs_npz = inputs_npz.with_name(f"{stem}.outputs.npz")
            if not outputs_npz.exists():
                skipped.append(KernelCase(
                    co=co, inputs_npz=inputs_npz, outputs_npz=outputs_npz,
                    kernel_name=stem, gold_smoke_ok=False,
                    gold_smoke_error=f"no matching .outputs.npz for {inputs_npz.name}",
                ))
                continue

            # Read just the outputs meta to learn the full kernel name
            # and the gold's smoke_ok bit. Lazy-import numpy here so
            # `python3 verify_transpile.py --help` still works on a
            # host without numpy installed.
            import numpy as np
            try:
                o = np.load(outputs_npz, allow_pickle=True)
                ometa = json.loads(str(o["_metadata_json"]))
            except Exception as exc:
                skipped.append(KernelCase(
                    co=co, inputs_npz=inputs_npz, outputs_npz=outputs_npz,
                    kernel_name=stem, gold_smoke_ok=False,
                    gold_smoke_error=f"failed to read outputs npz: {exc}",
                ))
                continue

            case = KernelCase(
                co=co,
                inputs_npz=inputs_npz,
                outputs_npz=outputs_npz,
                kernel_name=ometa.get("name", stem),
                gold_smoke_ok=bool(ometa.get("smoke_ok", False)),
                gold_smoke_error=str(ometa.get("smoke_error", "")),
            )

            if kernel_filter is not None and not kernel_filter.search(case.kernel_name):
                continue

            if not case.gold_smoke_ok and not include_failed:
                skipped.append(case)
                continue

            runnable.append(case)

    return runnable, skipped


def _build_child_env(args: argparse.Namespace, corpus: Path) -> dict[str, str]:
    env = os.environ.copy()

    rocr_lib = args.rocr_build / "rocr" / "lib"
    if not (rocr_lib / "libhsa-runtime64.so.1").exists():
        raise SystemExit(
            f"expected Salmon-enabled libhsa-runtime64.so.1 under "
            f"{rocr_lib}, not found (pass --rocr-build)"
        )

    # The Salmon-enabled libhsa-runtime must win the resolution race, so
    # prepend its directory to LD_LIBRARY_PATH.
    env["LD_LIBRARY_PATH"] = str(rocr_lib) + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )

    # `PYTHONPATH=<corpus>` so `from hsaco_runner import ...` works in
    # the child without mutating the corpus checkout.
    env["PYTHONPATH"] = str(corpus) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    # ROCm runtime discovery inside the child (hsaco_runner reads these).
    env["ROCM_PATH"] = str(args.rocm_path)

    # Mode-specific env: see compare_correctness README.
    #
    # In `native` mode we do NOT preload the intercept — we expect the
    # .co's ISA to already match the host GPU's ISA. Useful for
    # debugging the harness against a same-ISA build of the corpus.
    if args.mode != "native":
        preload = Path(args.salmon_intercept).resolve()
        if not preload.exists():
            raise SystemExit(
                f"salmon intercept library not found: {preload} "
                f"(pass --salmon-intercept)"
            )

        # For Python callers we must also preload, in order:
        #
        #   1. the Salmon-enabled libhsa-runtime64.so.1 — so its
        #      `rocr_salmon_patch_elf` symbol is globally resolvable
        #      when the intercept's init() dlsyms it via RTLD_DEFAULT.
        #      Without this the intercept aborts with "is Salmon-enabled
        #      libhsa-runtime64.so loaded?".
        #   2. libsalmon_intercept.so — the actual ELF e_flags rewriter.
        #   3. libamdhip64.so — so the intercept's
        #      `dlsym(RTLD_NEXT, "hipModuleLoadData")` can find the real
        #      symbol to forward to after patching. Without libamdhip64
        #      in the preload chain, the shim aborts with "no real
        #      hipModuleLoadData" the first time a module load comes in
        #      through `ctypes.CDLL(None).hipModuleLoadData` (our
        #      RTLD_DEFAULT routing in `_child_worker._load_module_via_intercept`).
        #      Order matters: libamdhip64 must come AFTER the intercept
        #      in the preload chain so the intercept's own wrapper wins
        #      the RTLD_DEFAULT search.
        salmon_rt = rocr_lib / "libhsa-runtime64.so.1"
        libhip = args.rocm_path / "lib" / "libamdhip64.so"

        for required in (salmon_rt, libhip):
            if not required.exists():
                raise SystemExit(f"expected library not found: {required}")

        parts = [str(salmon_rt), str(preload), str(libhip)]
        if env.get("LD_PRELOAD"):
            parts.append(env["LD_PRELOAD"])
        env["LD_PRELOAD"] = os.pathsep.join(parts)
        env["HSA_HOTSWAP_ISA_OVERRIDE"] = args.target_isa
        env["HSA_HOTSWAP_RULES"] = env.get("HSA_HOTSWAP_RULES", "/dev/null")
        if args.mode == "salmon":
            env["HSA_HOTSWAP_IR_RAISER"] = "1"
        else:
            env.pop("HSA_HOTSWAP_IR_RAISER", None)

    return env


def _run_case(case: KernelCase, args: argparse.Namespace,
              child_env: dict[str, str], tolerances_json: str) -> KernelVerdict:
    t0 = time.time()
    cmd = [
        sys.executable,
        str(_CHILD_WORKER),
        "--co", str(case.co),
        "--inputs-npz", str(case.inputs_npz),
        "--outputs-npz", str(case.outputs_npz),
        "--tolerances", tolerances_json,
    ]

    # We need the child's own process group so a hung launch can be
    # killed cleanly (otherwise a libhsa worker thread can prevent exit).
    proc = subprocess.Popen(
        cmd,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )

    try:
        stdout_b, stderr_b = proc.communicate(timeout=args.timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        # SIGKILL the whole process group.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout_b, stderr_b = proc.communicate()
        timed_out = True

    elapsed = time.time() - t0
    stdout = stdout_b.decode("utf-8", "replace") if stdout_b else ""
    stderr = stderr_b.decode("utf-8", "replace") if stderr_b else ""

    # The child emits exactly one JSON object as its very last stdout
    # line. Split and parse only that, so any print-debug noise above
    # does not break result parsing.
    child_json: Optional[dict] = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                child_json = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    exit_code = proc.returncode if proc.returncode is not None else -1
    sig = 0
    if exit_code < 0:
        sig = -exit_code
        exit_code = -1

    if timed_out:
        status = "timeout"
    elif sig != 0:
        status = "crash"
    elif child_json is None:
        status = "crash"  # child died without emitting a JSON line
    elif child_json.get("error"):
        status = "launch-error"
    elif child_json.get("all_outputs_ok"):
        status = "match"
    else:
        status = "mismatch"

    # Last ~200 stderr lines for the failure section of the report.
    # Was 40, but llc bug backtraces eat >30 lines by themselves and bury
    # the actual diagnostic (LLVM ERROR / Cannot select: <SDNode>) above
    # the backtrace, so we need a larger window to keep the signal.
    stderr_tail = "\n".join(stderr.splitlines()[-200:])

    return KernelVerdict(
        case=case,
        status=status,
        elapsed_s=elapsed,
        child_exit=exit_code,
        child_signal=sig,
        child_json=child_json,
        stderr_tail=stderr_tail,
    )


def _render_report(verdicts: list[KernelVerdict], skipped: list[KernelCase],
                   mode: str, target_isa: str) -> str:
    lines: list[str] = []
    lines.append(f"=== tensile_gold_verify  mode={mode}  target={target_isa} ===")
    lines.append(f"  runnable     : {len(verdicts)}")
    lines.append(f"  skipped      : {len(skipped)}")
    lines.append("")
    lines.append("  per-kernel grid (. = match, X = mismatch, E = error, T = timeout, C = crash):")
    lines.append("")

    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1

    # Grid rows: (co name, kernel name, status, elapsed)
    for v in verdicts:
        sigil = {
            "match": ".",
            "mismatch": "X",
            "launch-error": "E",
            "crash": "C",
            "timeout": "T",
        }.get(v.status, "?")
        lines.append(
            f"  [{sigil}] {v.elapsed_s:6.1f}s  {v.case.co.name}  ::  "
            f"{v.case.kernel_name}"
        )

    lines.append("")
    lines.append("=== Failures ===")
    n_fail = 0
    for v in verdicts:
        if v.status == "match":
            continue
        n_fail += 1
        lines.append("")
        lines.append(f"[{v.status}]  {v.case.co.name}  ::  {v.case.kernel_name}")
        if v.status == "timeout":
            lines.append(f"  timed out after {v.elapsed_s:.1f}s; SIGKILL sent")
        elif v.status == "crash":
            lines.append(f"  exit={v.child_exit}  signal={v.child_signal}")
        cj = v.child_json or {}
        if cj.get("error"):
            lines.append(f"  child error: {cj['error']}")
        for oname, ocmp in (cj.get("outputs") or {}).items():
            if ocmp.get("ok"):
                continue
            if "exact" in ocmp:
                lines.append(
                    f"  output {oname} [{ocmp.get('value_type', '?')}]: "
                    f"{ocmp['mismatches']}/{ocmp['n']} bytes differ "
                    f"(first idx {ocmp['first_diff_idx']}: "
                    f"got {ocmp['first_got']} want {ocmp['first_want']})"
                )
            elif "error" in ocmp:
                lines.append(
                    f"  output {oname}: {ocmp['error']}"
                )
            else:
                lines.append(
                    f"  output {oname} [{ocmp.get('value_type', '?')}]: "
                    f"{ocmp.get('mismatches', '?')}/{ocmp.get('n', '?')} "
                    f"mismatch  max|err|={ocmp.get('max_abs', float('nan')):.3g}  "
                    f"max_rel={ocmp.get('max_rel', float('nan')):.3g}  "
                    f"(atol={ocmp.get('atol')}, rtol={ocmp.get('rtol')})"
                )
                idx = ocmp.get("first_diff_idx", -1)
                if idx >= 0:
                    lines.append(
                        f"    first diff [{idx}]: "
                        f"got {ocmp.get('first_got')} "
                        f"want {ocmp.get('first_want')}"
                    )
        if v.stderr_tail:
            lines.append("  stderr tail:")
            for sl in v.stderr_tail.splitlines():
                lines.append(f"    {sl}")
    if n_fail == 0:
        lines.append("  (none)")

    lines.append("")
    lines.append("=== Summary ===")
    # Ordered output.
    for key in ("match", "mismatch", "launch-error", "crash", "timeout"):
        lines.append(f"  {key:<13s}: {counts.get(key, 0)}")
    lines.append(f"  {'skipped':<13s}: {len(skipped)}")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus", type=Path, required=True,
        help="Path to a checkout of rocm-hotswap-testing on the "
             "users/tgymnich/hsaco-runner branch.",
    )
    p.add_argument(
        "--rocr-build", type=Path,
        default=Path.home() / "rocm-systems" / "projects" / "rocr-runtime" / "build",
        help="Path to Salmon-enabled ROCR build (contains "
             "`rocr/lib/libhsa-runtime64.so.1`).",
    )
    p.add_argument(
        "--salmon-intercept", type=Path,
        default=_HERE.parent / "compare_correctness" / "libsalmon_intercept.so",
        help="Path to `libsalmon_intercept.so` (built by the "
             "compare_correctness tool's Makefile).",
    )
    p.add_argument(
        "--rocm-path", type=Path, default=Path("/opt/rocm-7.2.1"),
        help="ROCm install root (used by hsaco_runner to find "
             "libamdhip64.so).",
    )
    p.add_argument(
        "--mode", choices=("salmon", "legacy", "native"), default="salmon",
        help="`salmon` = ROCR hotswap + LLVM IR raiser (default); "
             "`legacy` = ROCR hotswap + byte-level rewriter; "
             "`native` = no intercept (expects same-ISA .co).",
    )
    p.add_argument(
        "--target-isa", default="gfx942",
        help="HSA_HOTSWAP_ISA_OVERRIDE value for the child process.",
    )
    p.add_argument(
        "--filter", default="",
        help="Regex on kernel name; only matching kernels are run.",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Stop after N kernels (0 = no limit).",
    )
    p.add_argument(
        "--timeout", type=float, default=60.0,
        help="Per-kernel wall-clock timeout in seconds (default 60).",
    )
    p.add_argument(
        "--include-failed", action="store_true",
        help="Also run kernels whose gold outputs failed the gfx1250 "
             "simulator smoke check (NaN / all-zero). Off by default "
             "because the gold is not trustworthy for those.",
    )
    p.add_argument(
        "--tolerances-json", default=json.dumps(_DEFAULT_TOLERANCES),
        help="JSON mapping AMDHSA value_type → [atol, rtol]. Must "
             "contain a 'default' key.",
    )
    p.add_argument(
        "--report-json", type=Path,
        help="Optional path to dump the full per-kernel JSON report.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print each verdict as it comes in (default: render the "
             "report at the end only).",
    )
    args = p.parse_args()

    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        raise SystemExit(f"--corpus does not point at a directory: {corpus}")

    # Parse & re-serialize tolerances to fail fast on JSON errors.
    tolerances = json.loads(args.tolerances_json)
    if "default" not in tolerances:
        raise SystemExit("--tolerances-json must contain a 'default' key")
    tolerances_json = json.dumps(tolerances)

    kernel_filter = re.compile(args.filter) if args.filter else None

    runnable, skipped = _discover_cases(corpus, args.include_failed, kernel_filter)
    if args.limit:
        runnable = runnable[: args.limit]

    print(f"Discovered {len(runnable)} runnable kernels, "
          f"{len(skipped)} skipped.", file=sys.stderr, flush=True)

    child_env = _build_child_env(args, corpus)

    verdicts: list[KernelVerdict] = []
    for i, case in enumerate(runnable, 1):
        v = _run_case(case, args, child_env, tolerances_json)
        verdicts.append(v)
        if args.verbose:
            cj = v.child_json or {}
            err = cj.get("error") or ""
            print(f"[{i:4d}/{len(runnable)}] {v.status:<13s} "
                  f"{v.elapsed_s:6.1f}s  {case.co.name}  "
                  f"{case.kernel_name}"
                  + (f"  -- {err}" if err else ""),
                  file=sys.stderr, flush=True)

    report = _render_report(verdicts, skipped, args.mode, args.target_isa)
    print(report)

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "mode": args.mode,
            "target_isa": args.target_isa,
            "runnable": len(runnable),
            "skipped": [
                {
                    "co": s.co.name,
                    "inputs_npz": s.inputs_npz.name,
                    "kernel": s.kernel_name,
                    "gold_smoke_ok": s.gold_smoke_ok,
                    "gold_smoke_error": s.gold_smoke_error,
                }
                for s in skipped
            ],
            "verdicts": [
                {
                    "co": v.case.co.name,
                    "kernel": v.case.kernel_name,
                    "status": v.status,
                    "elapsed_s": v.elapsed_s,
                    "child_exit": v.child_exit,
                    "child_signal": v.child_signal,
                    "child_json": v.child_json,
                    "stderr_tail": v.stderr_tail,
                }
                for v in verdicts
            ],
        }
        args.report_json.write_text(json.dumps(out, indent=2))
        print(f"\nWrote full report to {args.report_json}", file=sys.stderr)

    # Non-zero exit iff any runnable kernel did not match. A pre-run
    # skip (e.g. gold smoke_failed) is not an error.
    any_bad = any(v.status != "match" for v in verdicts)
    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
