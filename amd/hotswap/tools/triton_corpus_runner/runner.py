#!/usr/bin/env python3
"""Run upstream Triton-using scripts under each transpilation mode and
report which ones crash, hang, or produce wrong results.

This is the breadth-only counterpart to ``compare_correctness``: it
trades the per-launch numerical verdict for the ability to ingest
*any* Triton-using script (tutorial, pytest case, torch.compile
output dump, ...) without authoring a per-kernel shim.  Every script
is run three times and we record what happened to each run:

    native   no LD_PRELOAD, no target override.  Triton compiles for
             the actual gfx942 device — this is the baseline and
             tells us whether the script itself works on this
             hardware at all.
    legacy   LD_PRELOAD=libsalmon_intercept.so + ISA_OVERRIDE=gfx942
             + Triton forced to gfx1250 wave32.  ROCR's hotswap byte
             translator turns the gfx1250 .co into gfx942 at load
             time.  IR_RAISER unset.
    salmon   same as legacy + HSA_HOTSWAP_IR_RAISER=1.  ROCR's
             hotswap hook routes through the Salmon IR raiser
             instead of the byte translator.

Per (script, mode) we record one of:

    PASS         exit 0
    FAIL         exit non-zero, normal termination (e.g. assertion
                 failure inside the script — the script ran far
                 enough to compare results).
    CRASH        terminated by signal (SIGSEGV / SIGABRT / SIGILL...)
    HANG         exceeded ``--timeout`` and was SIGKILLed.
    SPAWN_FAIL   the subprocess couldn't even start (missing python,
                 missing libsalmon_intercept, etc.).

The interesting comparisons are columnar: a script that PASSes under
native but CRASHes / HANGs / FAILs under legacy or salmon is a real
transpilation gap; a script that already FAILs under native is a
problem with the script itself (or the local Triton install) and is
flagged as ``BASELINE_FAIL`` in the summary so it doesn't pollute
the salmon-coverage signal.

This tool is deliberately script-agnostic: it makes no attempt to
parse the script, infer its kernels, or interpret its output.  All
it knows is the exit status and the tail of stderr.  That's a much
weaker verdict than ``compare_correctness``'s per-output comparator,
but it's also much cheaper to gain coverage with — adding a new
script is just dropping it into a directory.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional


HERE = os.path.dirname(os.path.abspath(__file__))


# Defaults that match the in-tree Salmon-ROCR layout plus a self-contained
# rocm-7 PyTorch venv.  All of these can be overridden on the command line.
#
# We default to a sibling .venv-rocm7 created with the rocm-7 nightly torch
# wheel (which bundles a HIP ~7 runtime that's ABI-compatible with the
# Salmon-enabled libhsa).  The bundled triton-rocm in that wheel works as-is
# for our purposes — no in-tree Triton needed — so the default
# ``--triton-pythonpath`` is empty.  Users wanting to point at the in-tree
# Triton can override both via flags.
DEFAULT_TRITON_VENV = os.path.join(HERE, ".venv-rocm7")
DEFAULT_TRITON_PYTHONPATH = ""
DEFAULT_LIBSALMON = os.path.normpath(
    os.path.join(HERE, "..", "compare_correctness", "libsalmon_intercept.so")
)
# Salmon-enabled libhsa-runtime64.so.  Triton's Python process otherwise
# dlopens the system /opt/rocm libhsa, which has neither the hotswap hook
# nor the rocr_salmon_patch_elf symbol the intercept shim needs.  We
# LD_PRELOAD this so it (a) replaces the system copy in HIP's resolution
# and (b) loads with global scope so the shim's dlsym(RTLD_DEFAULT, ...)
# can find its symbol — the same trick compare_correctness's rpath does
# for its native binary, adapted to the Python case.
DEFAULT_LIBHSA = os.path.normpath(os.path.expanduser(
    "~/rocm-systems/projects/rocr-runtime/build/rocr/lib/libhsa-runtime64.so.1"
))
# A system libamdhip64.so to LD_PRELOAD.  Empty by default because the
# rocm-7 torch wheel ships a HIP runtime that tolerates the Salmon ROCR's
# multi-ISA agent enumeration directly — no shadowing needed.
#
# Set this (e.g. /opt/rocm-7.2.1/lib/libamdhip64.so.7) if you're pointing
# ``--triton-venv`` at an older torch wheel whose bundled HIP (~6.3)
# rejects multi-ISA agents with "agent has 2 ISAs but can only support a
# single ISA".  See amd_gpu_agent.cpp:188-191.
DEFAULT_LIBAMDHIP = ""
# An empty rules file silences the "failed to parse JSON" warning the
# hotswap hook emits when fed /dev/null, and is otherwise equivalent
# (no rules → fall back to ISA-override-driven defaults).
EMPTY_RULES = os.path.join(HERE, "_empty_rules.json")
DEFAULT_PULLED_TUTORIALS = os.path.normpath(
    os.path.join(
        HERE,
        "..",
        "compare_correctness",
        "kernels",
        "triton",
        "_corpus",
        "upstream",
    )
)


# We force gfx1250 wave32 because (a) that's the source ISA the
# Salmon transpiler expects to consume, matching what compare_correctness
# does, and (b) Triton's ``TRITON_OVERRIDE_ARCH`` env var alone leaves
# warp_size pinned to the device — which would yield a wave64-IR
# gfx1250 binary on gfx942 hardware, an inconsistent thing the
# transpiler isn't designed to ingest.
FORCED_TARGET = "gfx1250:32"


@dataclasses.dataclass
class ModeSpec:
    name: str
    description: str
    needs_preload: bool
    needs_force_target: bool
    needs_ir_raiser: bool
    # Sets ``HSA_SALMON_STRICT=1`` so the salmon transpiler promotes
    # known silent-miscompile sites (MODE-register writes, implicitarg.ptr
    # cross-arch lifts) to honest refusals instead of warn-and-continue.
    # See ``transpiler/pipeline.hpp::isStrictMode`` and the corpus
    # runner's ``INTEGRATION_GAP.md`` for the diagnosis behind each
    # promoted site. We only set this for the salmon mode because:
    #
    #   * native does not invoke the salmon transpiler at all.
    #   * legacy uses the byte-translator hotswap path which does not
    #     consult this flag.
    #   * compare_correctness and the gtest binary do not set it, so
    #     existing GPU tests stay on the warn-and-continue path and
    #     keep passing — the strict tightening is opt-in per caller.
    needs_strict_mode: bool = False


MODES: Dict[str, ModeSpec] = {
    "native": ModeSpec(
        name="native",
        description="gfx942 directly; no LD_PRELOAD, no target override",
        needs_preload=False,
        needs_force_target=False,
        needs_ir_raiser=False,
    ),
    "legacy": ModeSpec(
        name="legacy",
        description="gfx1250-Triton -> hotswap byte translator -> gfx942",
        needs_preload=True,
        needs_force_target=True,
        needs_ir_raiser=False,
    ),
    "salmon": ModeSpec(
        name="salmon",
        description="gfx1250-Triton -> hotswap Salmon IR raiser -> gfx942",
        needs_preload=True,
        needs_force_target=True,
        needs_ir_raiser=True,
        needs_strict_mode=True,
    ),
}


@dataclasses.dataclass
class RunResult:
    script: str
    mode: str
    verdict: str           # PASS / FAIL / CRASH / HANG / SPAWN_FAIL
    detail: str            # short human-readable detail for the report
    exit_code: Optional[int]
    signal_name: Optional[str]
    elapsed_s: float
    stderr_tail: str


# -- subprocess plumbing ----------------------------------------------------


def _build_env(
    mode: ModeSpec,
    triton_venv: str,
    triton_pythonpath: str,
    libsalmon: str,
    libhsa: str,
    libamdhip: str,
    stub_bench: bool,
    visible_devices: Optional[str],
) -> Dict[str, str]:
    """Construct the environment for one child run.  We start from
    the parent env and selectively override; that keeps things like
    HOME / DISPLAY / ROCM_PATH that the tutorial scripts may rely
    on."""
    env = dict(os.environ)

    # Make sure the bootstrap module is importable.
    env["PYTHONPATH"] = HERE + ":" + env.get("PYTHONPATH", "")

    # If the caller pointed us at an in-tree Triton, prepend it to
    # PYTHONPATH so it shadows whatever the venv ships.  Empty means
    # "use the venv's bundled Triton as-is" — the rocm-7 torch wheel
    # already bundles a working triton-rocm.
    if triton_pythonpath:
        env["PYTHONPATH"] = triton_pythonpath + ":" + env["PYTHONPATH"]

    # Avoid matplotlib trying to open a display in a tutorial that
    # imports it for benchmark plots.  Always Agg, regardless of the
    # parent's MPLBACKEND.
    env["MPLBACKEND"] = "Agg"

    # Don't let any inherited torch.compile cache mask first-run
    # crashes on the salmon path (which often go through fresh codegen).
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/triton-corpus-runner-inductor")

    # Defensive isolation: pin to a single GPU so any pathological
    # transpiled kernel that managed to wedge an SQ/CU only causes
    # an amdgpu queue reset on *one* device.  This box has 8 MI300X
    # shared with other users (talumbau, anush, yrathore, annier as
    # of writing); without a pin, Triton/torch see all 8 and a bad
    # launch could in principle force a reset on a GPU that someone
    # else is using.  ``HIP_VISIBLE_DEVICES`` is honoured by both HIP
    # and torch.  The user can override by exporting it themselves
    # (we only ``setdefault``) or via --gpu on the command line.
    if visible_devices is not None:
        env["HIP_VISIBLE_DEVICES"] = visible_devices
    else:
        env.setdefault("HIP_VISIBLE_DEVICES", "0")

    if mode.needs_preload:
        # Always-required pieces of the LD_PRELOAD chain.
        for path, kind, hint in (
            (libhsa, "Salmon-enabled libhsa-runtime64.so",
             "build the Salmon ROCR tree first or override --libhsa"),
            (libsalmon, "libsalmon_intercept.so",
             "build compare_correctness first or override --libsalmon"),
        ):
            if not os.path.exists(path):
                raise RuntimeError(
                    f"{kind} not found at {path}; {hint}"
                )

        # Optional: a system libamdhip64 to shadow torch's bundled HIP.
        # Only needed when the venv's torch ships a HIP older than the
        # Salmon ROCR's multi-ISA agent enumeration (see DEFAULT_LIBAMDHIP
        # comment).  Empty string means "trust the venv's bundled HIP".
        preload_chain: List[str] = []
        ldlib_dirs: List[str] = []
        if libamdhip:
            if not os.path.exists(libamdhip):
                raise RuntimeError(
                    f"--libamdhip {libamdhip!r} does not exist; pass an "
                    f"empty string to disable libamdhip preload"
                )
            preload_chain.append(libamdhip)
            ldlib_dirs.append(os.path.dirname(libamdhip))

        # Order matters.  libhsa next so its symbols (rocr_salmon_patch_elf,
        # the hotswap hook) are visible to the intercept shim's
        # dlsym(RTLD_DEFAULT, ...) lookups.  Then the intercept shim itself.
        preload_chain.extend([libhsa, libsalmon])
        ldlib_dirs.append(os.path.dirname(libhsa))

        env["LD_PRELOAD"] = ":".join(preload_chain)
        env["LD_LIBRARY_PATH"] = ":".join(
            ldlib_dirs + [env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(":")
        env.setdefault("HSA_HOTSWAP_ISA_OVERRIDE", "gfx942")
        env.setdefault("HSA_HOTSWAP_RULES", EMPTY_RULES)
    else:
        # Make sure native runs are *clean* — no inherited preload from
        # an interactive shell that already had one set.
        env.pop("LD_PRELOAD", None)
        env.pop("HSA_HOTSWAP_ISA_OVERRIDE", None)
        env.pop("HSA_HOTSWAP_IR_RAISER", None)
        env.pop("HSA_HOTSWAP_RULES", None)

    if mode.needs_ir_raiser:
        env["HSA_HOTSWAP_IR_RAISER"] = "1"
    elif mode.needs_preload:
        # Legacy: must NOT have IR_RAISER set, even if the parent shell did.
        env.pop("HSA_HOTSWAP_IR_RAISER", None)

    if mode.needs_strict_mode:
        env["HSA_SALMON_STRICT"] = "1"
    else:
        # Make sure non-strict modes are not contaminated by an inherited
        # value from an interactive shell that happened to export it.
        env.pop("HSA_SALMON_STRICT", None)

    if mode.needs_force_target:
        env["TRITON_CORPUS_FORCE_TARGET"] = FORCED_TARGET
    else:
        env.pop("TRITON_CORPUS_FORCE_TARGET", None)

    if stub_bench:
        env["TRITON_CORPUS_STUB_BENCH"] = "1"
    else:
        env.pop("TRITON_CORPUS_STUB_BENCH", None)

    return env


def _classify_exit(rc: Optional[int]) -> tuple[str, Optional[str]]:
    """Convert a Popen returncode (which may be a negative signal) into
    a (verdict, signal_name) pair."""
    if rc is None:
        return "HANG", None
    if rc < 0:
        # Killed by signal -N
        sig = -rc
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = f"signal {sig}"
        return "CRASH", name
    if rc == 0:
        return "PASS", None
    return "FAIL", None


def _tail(text: str, n: int = 1200) -> str:
    if len(text) <= n:
        return text
    return "...[truncated]\n" + text[-n:]


def _run_one(
    script: str,
    mode: ModeSpec,
    triton_venv: str,
    triton_pythonpath: str,
    libsalmon: str,
    libhsa: str,
    libamdhip: str,
    timeout_s: float,
    stub_bench: bool,
    visible_devices: Optional[str],
) -> RunResult:
    """Spawn one child process for one (script, mode) and wait."""
    env = _build_env(
        mode, triton_venv, triton_pythonpath, libsalmon, libhsa, libamdhip,
        stub_bench, visible_devices,
    )
    env["TRITON_CORPUS_SCRIPT"] = os.path.abspath(script)

    python = os.path.join(triton_venv, "bin", "python")
    if not os.path.exists(python):
        raise RuntimeError(
            f"Triton venv python not found at {python}; "
            f"override --triton-venv if your venv lives elsewhere"
        )

    cmd = [python, "-m", "_bootstrap"]

    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=HERE,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Put the child in its own process group so SIGKILL on
            # timeout reaches grandchild HIP/Triton helpers too.
            start_new_session=True,
        )
    except OSError as e:
        return RunResult(
            script=script, mode=mode.name,
            verdict="SPAWN_FAIL",
            detail=f"could not exec child: {e}",
            exit_code=None, signal_name=None,
            elapsed_s=0.0, stderr_tail="",
        )

    timed_out = False
    try:
        _stdout_b, stderr_b = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            _stdout_b, stderr_b = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            _stdout_b, stderr_b = b"", b""

    elapsed = time.monotonic() - t0
    stderr = (stderr_b or b"").decode("utf-8", errors="replace")

    if timed_out:
        verdict, sig_name = "HANG", None
        detail = f"exceeded timeout of {timeout_s:.0f}s; SIGKILLed"
    else:
        verdict, sig_name = _classify_exit(proc.returncode)
        if verdict == "CRASH":
            detail = f"killed by {sig_name}"
        elif verdict == "FAIL":
            detail = f"exit {proc.returncode}"
        else:
            detail = "ok"

    return RunResult(
        script=script,
        mode=mode.name,
        verdict=verdict,
        detail=detail,
        exit_code=proc.returncode if not timed_out else None,
        signal_name=sig_name,
        elapsed_s=elapsed,
        stderr_tail=_tail(stderr),
    )


# -- discovery + reporting --------------------------------------------------


def _discover_scripts(
    explicit_scripts: List[str],
    explicit_dirs: List[str],
    default_dir: str,
) -> List[str]:
    """Resolve the list of scripts to run.  Explicit ``--script`` wins;
    otherwise scan ``--scripts-dir`` (repeatable); otherwise fall
    back to the default pulled-tutorials directory."""
    paths: List[str] = []
    seen: set[str] = set()

    def _add(p: str) -> None:
        ap = os.path.abspath(p)
        if ap in seen:
            return
        seen.add(ap)
        paths.append(ap)

    for s in explicit_scripts:
        if not os.path.isfile(s):
            raise RuntimeError(f"--script {s!r}: not a file")
        _add(s)

    dirs = list(explicit_dirs)
    if not paths and not dirs:
        if not os.path.isdir(default_dir):
            raise RuntimeError(
                f"no --script / --scripts-dir given, and default "
                f"tutorial dir {default_dir!r} does not exist (did you "
                f"run kernels/triton/_corpus/pull.py?)"
            )
        dirs = [default_dir]

    for d in dirs:
        if not os.path.isdir(d):
            raise RuntimeError(f"--scripts-dir {d!r}: not a directory")
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py"):
                continue
            if fn.startswith("_"):
                continue
            _add(os.path.join(d, fn))

    if not paths:
        raise RuntimeError("no scripts to run")

    return paths


_VERDICT_ORDER = ["PASS", "FAIL", "CRASH", "HANG", "SPAWN_FAIL"]


def _short_label(script: str, base_dirs: List[str]) -> str:
    ap = os.path.abspath(script)
    for d in base_dirs:
        d = os.path.abspath(d)
        if ap.startswith(d + os.sep):
            return os.path.relpath(ap, d)
    return os.path.basename(ap)


def _print_grid(results: List[RunResult], mode_names: List[str],
                base_dirs: List[str]) -> None:
    by_script: Dict[str, Dict[str, RunResult]] = {}
    for r in results:
        by_script.setdefault(r.script, {})[r.mode] = r

    label_w = max(
        (len(_short_label(s, base_dirs)) for s in by_script),
        default=10,
    )
    label_w = max(label_w, len("script"))
    cell_w = max(
        (len(f"{r.verdict} ({r.detail[:18]})") for r in results),
        default=10,
    )
    cell_w = min(max(cell_w, max(len(m) for m in mode_names)), 32)

    print()
    print("=== Per-script grid ===")
    header = "  " + "script".ljust(label_w)
    for m in mode_names:
        header += "   " + m.ljust(cell_w)
    print(header)
    print("  " + "-" * label_w + ("   " + "-" * cell_w) * len(mode_names))
    for script in sorted(by_script):
        row = "  " + _short_label(script, base_dirs).ljust(label_w)
        for m in mode_names:
            r = by_script[script].get(m)
            if r is None:
                cell = "—"
            elif r.verdict == "PASS":
                cell = "PASS"
            else:
                cell = f"{r.verdict} ({r.detail})"
            row += "   " + cell[:cell_w].ljust(cell_w)
        print(row)


def _print_failures(results: List[RunResult], mode_names: List[str],
                    base_dirs: List[str]) -> None:
    by_mode: Dict[str, List[RunResult]] = {m: [] for m in mode_names}
    for r in results:
        if r.verdict != "PASS":
            by_mode.setdefault(r.mode, []).append(r)

    if not any(by_mode.values()):
        return

    print()
    print("=== Failures ===")
    for m in mode_names:
        runs = by_mode.get(m, [])
        if not runs:
            continue
        print(f"\n{m}")
        for r in runs:
            print(
                f"  {_short_label(r.script, base_dirs)}"
                f"  {r.verdict}  ({r.detail}, {r.elapsed_s:.1f}s)"
            )
            if r.stderr_tail.strip():
                for line in r.stderr_tail.rstrip().splitlines()[-12:]:
                    print(f"      | {line}")


def _print_summary(results: List[RunResult], mode_names: List[str]) -> None:
    counts: Dict[str, Dict[str, int]] = {
        v: {m: 0 for m in mode_names} for v in _VERDICT_ORDER
    }
    for r in results:
        counts[r.verdict][r.mode] += 1

    n_scripts = len({r.script for r in results})
    print()
    print(f"=== Summary ({n_scripts} scripts × {len(mode_names)} modes) ===")
    label_w = max(len(v) for v in _VERDICT_ORDER)
    header = "  " + " " * label_w
    for m in mode_names:
        header += "   " + m.rjust(8)
    print(header)
    for v in _VERDICT_ORDER:
        row = "  " + v.ljust(label_w)
        for m in mode_names:
            row += "   " + str(counts[v][m]).rjust(8)
        print(row)

    # Cross-mode salient observation: scripts that PASS native but not
    # legacy / salmon are real transpilation gaps; PASSes everywhere
    # are clean coverage.
    by_script: Dict[str, Dict[str, str]] = {}
    for r in results:
        by_script.setdefault(r.script, {})[r.mode] = r.verdict

    if "native" in mode_names:
        native_ok = [s for s, m in by_script.items() if m.get("native") == "PASS"]
        for transp in ("legacy", "salmon"):
            if transp not in mode_names:
                continue
            broke = [s for s in native_ok if by_script[s].get(transp) != "PASS"]
            print(
                f"\n  {transp}: {len(native_ok) - len(broke)}/{len(native_ok)} "
                f"native-PASS scripts also pass under {transp}"
            )
            if broke:
                for s in sorted(broke):
                    print(f"    - {os.path.basename(s)} "
                          f"({by_script[s].get(transp, '—')})")


# -- CLI --------------------------------------------------------------------


def _parse_modes(arg: str) -> List[str]:
    if arg.strip().lower() in ("all", ""):
        return ["native", "legacy", "salmon"]
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok not in MODES:
            raise argparse.ArgumentTypeError(
                f"unknown mode {tok!r}; valid: {sorted(MODES)}"
            )
        out.append(tok)
    if not out:
        raise argparse.ArgumentTypeError("at least one mode required")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--script", action="append", default=[],
        help="explicit script to run (repeatable)",
    )
    ap.add_argument(
        "--scripts-dir", action="append", default=[],
        help="directory of .py scripts to run, non-recursive (repeatable)",
    )
    ap.add_argument(
        "--modes", default="all", type=_parse_modes,
        help="comma-separated subset of {native,legacy,salmon}; "
             "default 'all'",
    )
    ap.add_argument(
        "--timeout", type=float, default=180.0,
        help="per-child wall-clock timeout in seconds (default 180)",
    )
    ap.add_argument(
        "--triton-venv", default=DEFAULT_TRITON_VENV,
        help=f"Triton venv root (default {DEFAULT_TRITON_VENV})",
    )
    ap.add_argument(
        "--triton-pythonpath", default=DEFAULT_TRITON_PYTHONPATH,
        help="colon-separated PYTHONPATH for the in-tree Triton",
    )
    ap.add_argument(
        "--libsalmon", default=DEFAULT_LIBSALMON,
        help=f"path to libsalmon_intercept.so "
             f"(default {DEFAULT_LIBSALMON})",
    )
    ap.add_argument(
        "--libhsa", default=DEFAULT_LIBHSA,
        help=f"path to the Salmon-enabled libhsa-runtime64.so.1 "
             f"(default {DEFAULT_LIBHSA})",
    )
    ap.add_argument(
        "--libamdhip", default=DEFAULT_LIBAMDHIP,
        help="path to a system libamdhip64.so.* to LD_PRELOAD ahead of "
             "the venv's bundled HIP.  Only needed when the venv ships a "
             "torch wheel whose HIP predates the Salmon ROCR's multi-ISA "
             "agent enumeration (e.g. torch wheels for ROCm <= 6.3). "
             "Empty (the default) means trust the venv's HIP — which is "
             "correct for the rocm-7 nightly torch wheel that the "
             "default --triton-venv is built against.",
    )
    ap.add_argument(
        "--no-stub-bench", action="store_true",
        help="don't stub triton.testing.do_bench / Mark.run; the "
             "scripts will run their full benchmark loops",
    )
    ap.add_argument(
        "--gpu", default=None,
        help="value for HIP_VISIBLE_DEVICES (single index like '0', or "
             "comma list).  Default: '0' if HIP_VISIBLE_DEVICES is not "
             "already in the environment, else inherit.  Pinning a single "
             "GPU localises any queue-reset blast radius on a "
             "shared box.  Pass '' to clear it (give Triton/torch all "
             "visible devices) — only do this on a machine you own.",
    )
    ap.add_argument(
        "--default-dir", default=DEFAULT_PULLED_TUTORIALS,
        help=f"fallback dir if no --script/--scripts-dir given "
             f"(default {DEFAULT_PULLED_TUTORIALS})",
    )
    ap.add_argument(
        "--json", default="",
        help="also write a machine-readable JSON record here",
    )
    args = ap.parse_args()

    try:
        scripts = _discover_scripts(
            args.script, args.scripts_dir, args.default_dir
        )
    except RuntimeError as e:
        print(f"discovery failed: {e}", file=sys.stderr)
        return 2

    base_dirs = list(args.scripts_dir) or [args.default_dir]
    print(f"triton corpus runner: {len(scripts)} script(s), "
          f"modes={args.modes}, timeout={args.timeout:.0f}s",
          file=sys.stderr)
    for s in scripts:
        print(f"  - {_short_label(s, base_dirs)}", file=sys.stderr)
    print(file=sys.stderr)

    results: List[RunResult] = []
    for s in scripts:
        for m in args.modes:
            mode = MODES[m]
            print(f"[run] {_short_label(s, base_dirs)} :: {m} ... ",
                  end="", flush=True, file=sys.stderr)
            r = _run_one(
                script=s, mode=mode,
                triton_venv=args.triton_venv,
                triton_pythonpath=args.triton_pythonpath,
                libsalmon=args.libsalmon,
                libhsa=args.libhsa,
                libamdhip=args.libamdhip,
                timeout_s=args.timeout,
                stub_bench=not args.no_stub_bench,
                visible_devices=args.gpu,
            )
            results.append(r)
            print(f"{r.verdict} ({r.detail}, {r.elapsed_s:.1f}s)",
                  file=sys.stderr)

    _print_grid(results, args.modes, base_dirs)
    _print_failures(results, args.modes, base_dirs)
    _print_summary(results, args.modes)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "generated": datetime.datetime.now().isoformat(),
                    "modes": args.modes,
                    "timeout_s": args.timeout,
                    "forced_target": FORCED_TARGET,
                    "results": [dataclasses.asdict(r) for r in results],
                },
                f,
                indent=2,
            )
            f.write("\n")
        print(f"\nWrote machine-readable record to {args.json}",
              file=sys.stderr)

    # Exit code: 0 if every native run passed (the harness itself
    # worked); 1 if even native is broken (we can't trust anything).
    # Transpilation failures are a *finding*, not a harness error —
    # mirroring compare_correctness's contract.
    if "native" in args.modes:
        native_ran = [r for r in results if r.mode == "native"]
        native_ok = [r for r in native_ran if r.verdict == "PASS"]
        if native_ran and not native_ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
