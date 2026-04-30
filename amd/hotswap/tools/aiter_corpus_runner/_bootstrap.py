"""Child-process bootstrap for the AITER corpus crash sweeper.

This module is imported by ``runner.py`` *inside* the child Python
process, before the user op-test script runs.  Its job is to:

  1. Set ``GPU_ARCHS`` *before* aiter is imported so AITER's loader
     picks the per-arch ``hsa/<gfx>/`` code-object directory we want
     to exercise (typically ``gfx950`` when running on a gfx942 box,
     so the resulting ``hipModuleLoad`` calls trip the hotswap hook
     and route through the Salmon transpiler).

  2. Monkey-patch ``aiter.test_common.checkAllclose`` so it actually
     raises on real failures instead of just logging.  The upstream
     implementation only ``logger.info``s a "failed!" line and
     returns the per-element mismatch percent (see test_common.py
     lines 274-307); a wrong transpiled kernel currently exits 0
     with no signal at all.  We preserve the original logging (so
     triage reads identically in stderr) and raise ``AssertionError``
     when the percent exceeds a configurable threshold, broken down
     by the ``msg=`` label the test passed (``"asm"``, ``"asm res"``,
     ``"res check"``, ...).  An ``atexit`` hook prints a per-label
     pass/fail summary so a single-script triage doesn't require
     parsing the whole stderr stream.

  3. Throttle ``aiter.test_common.perftest`` (and ``run_perftest``)
     to a single iteration so the 101-iter benchmark sweeps that
     decorate every AITER test don't dominate wall clock.  The
     correctness path is untouched; we just stop running the kernel
     a hundred times for a perf number we discard anyway.

Environment knobs the parent sets:

  AITER_CORPUS_FORCE_GFX     ``"gfx942"`` / ``"gfx950"`` / ...
                             Sets ``GPU_ARCHS`` so AITER picks the
                             matching ``hsa/<gfx>/`` directory.
                             Required.

  AITER_CORPUS_RNG_SEED      Integer seed fed to ``torch.manual_seed``
                             + ``torch.cuda.manual_seed_all`` +
                             ``random.seed`` + ``numpy.random.seed``
                             *before* the user op-test script runs.
                             Default ``"0"``.  Making this non-empty
                             means every mode (native / legacy /
                             salmon) sees identical random-input
                             tensors for a given script + args, which
                             is the invariant the cross-mode
                             comparison relies on: any per-label
                             ``checkAllclose`` delta between native
                             and salmon under this setup is
                             necessarily a transpilation signal, not
                             random noise from unseeded RNG.  Set
                             to the empty string to leave RNG alone
                             (older behaviour; only useful when
                             debugging RNG-sensitive numerical
                             issues in the kernel itself).

  AITER_CORPUS_STRICT_TOL    Float in ``[0.0, 1.0]``; mismatches
                             whose elementwise-fail-percent is
                             *strictly greater* than this raise
                             ``AssertionError``.  Default ``"0.05"``
                             matches AITER's own ``tol_err_ratio=0.05``
                             cutoff inside ``checkAllclose``
                             (test_common.py:429 where AITER logs the
                             ``failed!`` banner iff
                             ``percent > tol_err_ratio``).  Set lower
                             (e.g. ``"0.0"``) to catch the slightest
                             salmon-vs-native divergence when hunting
                             silent transpilation miscompiles.

  AITER_CORPUS_PERFTEST_ITERS  Integer; ignored on the correctness
                             path because we always run the kernel
                             exactly once (the ``perftest`` patch
                             discards iteration counts entirely).
                             Kept as an env knob so the runner can
                             surface it in its report header.

    AITER_CORPUS_SCRIPT        Absolute path to the user op-test
                               script to run via ``runpy``.  Required.

    AITER_CORPUS_SCRIPT_ARGS   shlex-quoted (single string) argv
                               items to append after the script path
                               on ``sys.argv``.  Optional.  Empty /
                               unset means the script sees no extra
                               args, i.e. its argparse defaults take
                               effect.

The bootstrap exits with the user script's natural exit status (or
the propagated exception's traceback).  Any failure inside the
bootstrap itself (env-var missing, monkey-patch failures) is loud
and non-zero — we never silently fall back to the un-patched path,
because that would defeat the entire point of the run.
"""
from __future__ import annotations

import atexit
import collections
import os
import runpy
import sys
import traceback
from typing import Any, Callable


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        print(
            f"[aiter_corpus_runner] required env var {name} is unset",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)
    return val


def _set_gpu_archs(gfx: str) -> None:
    """Force AITER's ``GPU_ARCHS`` to the value the runner picked.
    Must run before any ``import aiter`` so the cached ``AITER_ASM_DIR``
    and the generated ``asm_<module>_configs.hpp`` (via codegen.py's
    read of ``AITER_GPU_ARCHS``) are built from this value.

    Two shapes appear in practice, all forwarded verbatim:

      ``"gfx942"``           single-arch run on a gfx942 host where
                             no transpile mode is enabled (spoof off).
      ``"gfx942;gfx950"``    the canonical runner default on a gfx942
                             host — kept identical across native,
                             legacy, and salmon modes so the on-disk
                             JIT cache (keyed by hipcc build flags)
                             is byte-identical between runs.  The
                             native arch ensures host-side fat
                             binaries actually run on the real
                             device; the source-ISA arch ensures
                             (a) codegen.py emits config rows keyed
                             on gfx950 for the asm lookup, and
                             (b) hipcc produces gfx950 device code,
                             which survives being fed to
                             ``hipModuleLoadData`` where Salmon
                             intercepts it.

    Note: ``GPU_ARCHS`` only steers the *Python-side* AITER paths
    (chip_info.get_gfx, JIT build target selection, asm-availability
    guards in tests).  The companion ``AITER_GPU_ARCHS`` — which
    ``hsa/codegen.py`` reads directly — is pre-seeded to the same
    value below so the codegen subprocess doesn't rely on import
    order (AITER normally derives it lazily inside
    ``chip_info.get_gfx_list()``; if our child process hasn't yet
    touched that code path when ``os.system(blob_gen_cmd)`` fires,
    codegen would silently emit host-arch-only rows).  The C++
    asm-kernel loader keys on ``hipGetDeviceProperties().gcnArchName``
    instead — that's why the runner also LD_PRELOADs
    ``libaiter_arch_spoof.so`` for transpile modes; see runner.py
    and arch_spoof.cpp for the full rationale.
    """
    os.environ["GPU_ARCHS"] = gfx
    # Pre-seed AITER_GPU_ARCHS from the same value so hsa/codegen.py
    # (invoked as a subprocess via ``os.system(blob_gen_cmd)`` from
    # aiter/jit/core.py) sees the full multi-arch list regardless of
    # whether anything in the parent process has yet called
    # ``chip_info.get_gfx_list()`` (which normally derives
    # AITER_GPU_ARCHS lazily from GPU_ARCHS).  Without this pre-seed,
    # the codegen subprocess inherits whatever the caller's shell
    # happened to export — almost always nothing — and emits config
    # rows for only the host arch, which causes the
    # ``get_heuristic_kernel_*: cannot get heuristic kernel!
    # arch_id:gfx950`` abort when the spoofed asm loader asks for the
    # foreign-ISA row.  Setting both env vars up front matches AITER's
    # post-get_gfx_list invariant and is the minimum change that
    # survives import ordering.
    os.environ["AITER_GPU_ARCHS"] = gfx
    spoof = os.environ.get("AITER_CORPUS_SPOOF_ARCH", "")
    if spoof:
        print(
            f"[aiter_corpus_runner] forced GPU_ARCHS={gfx!r} "
            f"(AITER_GPU_ARCHS pre-seeded identically); "
            f"hipGetDeviceProperties().gcnArchName spoofed to {spoof!r} "
            f"by libaiter_arch_spoof.so (AITER's C++ asm loader will "
            f"read hsa/{spoof}/*.co)",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[aiter_corpus_runner] forced GPU_ARCHS={gfx!r} "
            f"(AITER_GPU_ARCHS pre-seeded identically); "
            f"AITER_CORPUS_SPOOF_ARCH unset (C++ asm loader will use "
            f"the device's real gcnArchName)",
            file=sys.stderr,
            flush=True,
        )


def _label_key(msg: Any) -> str:
    """Normalize a ``checkAllclose(msg=...)`` argument into a short
    triage label.  AITER tests pass either a perf-style banner
    (``"[perf] dim: (128,8192), ..."``) or a path-distinguishing tag
    (``"asm"``, ``"asm res"``, ``"res check"``).  We keep the first
    40 chars of whichever was passed; that's enough to keep the
    distinct path tags separate while truncating the perf banners
    so they don't blow out the final summary."""
    if msg is None:
        return "<no msg>"
    s = str(msg).strip()
    if not s:
        return "<empty msg>"
    return s[:40]


def _patch_check_allclose(strict_tol: float) -> None:
    """Wrap ``aiter.test_common.checkAllclose`` so it (a) still logs
    via the upstream implementation (preserving stderr triage), and
    (b) raises ``AssertionError`` when the returned mismatch percent
    is *strictly greater* than ``strict_tol``.  Per-label pass/fail
    counts are accumulated in module-global state and dumped via
    ``atexit``.

    We patch by name on the module (not by re-binding the imported
    symbol) because the test scripts do ``from aiter.test_common
    import checkAllclose`` only *after* this bootstrap has already
    rebound the attribute; their import then gets the wrapper.  See
    aiter/test_common.py:274 for the upstream signature."""
    import aiter.test_common as tc

    original = tc.checkAllclose
    counts: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"pass": 0, "fail": 0}
    )
    state = {"first_failure": None}

    def wrapper(a, b, rtol=1e-2, atol=1e-2, tol_err_ratio=0.05, msg="",
                printNum=8, printLog=True, **extra):
        # Match upstream's full keyword signature verbatim (see
        # aiter/test_common.py:399) and accept **extra so future
        # upstream additions don't surface as TypeError during the
        # sweep — an unknown kwarg will simply be forwarded to
        # ``original`` which is the right place to fail loudly if the
        # upstream signature actually changed.
        percent = original(
            a, b, rtol=rtol, atol=atol, tol_err_ratio=tol_err_ratio,
            msg=msg, printNum=printNum, printLog=printLog, **extra,
        )
        # The upstream returns either ``0`` (allclose) or the float
        # percent of mismatched elements.  Anything above our effective
        # threshold is a real failure under our contract.
        #
        # We deliberately do NOT coerce odd return shapes (``None``,
        # strings, tuples) into a "pass".  If upstream ever changes
        # its contract we want to hear about it immediately, via an
        # honest ``TypeError`` that aborts the run, rather than
        # silently recording every comparison as 0% mismatched.
        # Same reason we refuse to paper over the ``percent is None``
        # case with ``or 0``: a None here is a contract bug, not a
        # pass.
        if percent is None:
            raise TypeError(
                "[aiter_corpus_runner] aiter.test_common.checkAllclose "
                "returned None — upstream contract says it returns a "
                "float mismatch percent (0 for allclose).  Refusing "
                "to treat None as a pass; investigate the upstream "
                "change before re-running the corpus sweep."
            )
        pct = float(percent)

        # Strictest-wins composition: respect whichever of
        # ``--strict-tolerance`` and the test author's ``tol_err_ratio``
        # is tighter.  This keeps ``--strict-tolerance 0.0`` as a
        # universal "flag every numerical deviation" knob for hunting
        # salmon miscompiles, while still honoring a per-call
        # ``tol_err_ratio=0.01`` when the test author knew the kernel
        # was expected to be tight even though our global default is
        # AITER's own 0.05.
        effective_thresh = min(strict_tol, float(tol_err_ratio))
        label = _label_key(msg)
        if pct > effective_thresh:
            counts[label]["fail"] += 1
            if state["first_failure"] is None:
                state["first_failure"] = (label, pct, atol, rtol)
            raise AssertionError(
                f"[aiter_corpus_runner] checkAllclose label={label!r} "
                f"failed: {pct:.4%} of elements mismatched "
                f"(atol={atol}, rtol={rtol}, "
                f"threshold={effective_thresh} "
                f"[min(--strict-tolerance={strict_tol}, "
                f"caller_tol_err_ratio={tol_err_ratio})])"
            )
        counts[label]["pass"] += 1
        return percent

    tc.checkAllclose = wrapper

    def _summary() -> None:
        if not counts:
            return
        total_pass = sum(c["pass"] for c in counts.values())
        total_fail = sum(c["fail"] for c in counts.values())
        print(
            f"[aiter_corpus_runner] checkAllclose: "
            f"{total_pass} pass / {total_fail} fail "
            f"(threshold={strict_tol})",
            file=sys.stderr,
            flush=True,
        )
        # Per-label breakdown so triage can immediately tell
        # torch-vs-asm mismatches apart from torch-vs-ck mismatches
        # in the same script.
        for label in sorted(counts):
            c = counts[label]
            mark = " FAIL" if c["fail"] else ""
            print(
                f"[aiter_corpus_runner]   "
                f"{c['pass']:>4} pass / {c['fail']:>4} fail  "
                f"label={label!r}{mark}",
                file=sys.stderr,
                flush=True,
            )
        if state["first_failure"] is not None:
            label, pct, atol, rtol = state["first_failure"]
            print(
                f"[aiter_corpus_runner] first failure: label={label!r} "
                f"percent={pct:.4%} atol={atol} rtol={rtol}",
                file=sys.stderr,
                flush=True,
            )

    atexit.register(_summary)

    print(
        f"[aiter_corpus_runner] patched aiter.test_common.checkAllclose "
        f"(raise on percent > {strict_tol})",
        file=sys.stderr,
        flush=True,
    )


# Sentinel value returned in place of the 101-iter average wall-clock
# us.  Must be (a) a plain float (some tests destructure as
# ``(data, avg) = run_perftest(...)`` then format ``avg``); (b)
# nonzero (several tests compute ``avg_a / avg_b - 1`` for an uplift
# percent, which divisions-by-zero out if we return 0.0); (c) not so
# small that downstream ``f"{avg:5.1%}"`` formatting overflows.  1.0
# satisfies all three and makes any uplift line render as "0.0%",
# which is correct: we ran the kernel exactly once, no perf signal
# was collected.
_PERFTEST_FAKE_AVG_US = 1.0


def _patch_aiter_config_dir() -> None:
    """Redirect AITER's hardcoded ``/tmp/aiter_configs/`` config-merge
    directory to a per-user path so the runner works on shared hosts.

    Why this is needed: AITER's ``AITER_CONFIG.update_config_files``
    (aiter/jit/core.py) does ``Path("/tmp/aiter_configs/")`` with no
    env override and then writes merged tuned-config CSVs + FileBaton
    lockfiles into that directory.  On any multi-user box where a
    different user ran AITER first, that directory already exists
    owned by *them* with mode 0775 (group-writable, not
    world-writable) — so any subsequent user's first tuned-config
    merge dies with ``PermissionError: [Errno 13] Permission denied:
    '/tmp/aiter_configs/<name>.csv.lock'`` the moment AITER's
    ``FileBaton`` tries to ``os.open(..., O_CREAT | O_EXCL)`` the lock.

    The fix is a minimal wrapper around ``update_config_files`` that
    swaps ``pathlib.Path`` for a constructor that redirects exactly
    the hardcoded ``/tmp/aiter_configs/`` literal to a user-writable
    directory, leaving all other ``Path(...)`` calls untouched.  We
    don't modify the upstream AITER source — this is an import-time
    monkey-patch of a single method on the ``AITER_CONFIG`` class.

    The target directory comes from ``AITER_CORPUS_CONFIG_DIR`` (set
    by the runner); fallback is ``~/.cache/aiter_corpus_runner/``.
    Several AITER tests (test_gemm_a16w16, test_gemm_a8w8,
    test_gemm_a8w8_blockscale, ...) hit this path on first run and
    without the redirect every one of them fails native with a
    runner-induced ``PermissionError``, completely hiding any real
    AITER-on-gfx942 signal.
    """
    import pathlib
    import aiter.jit.core as core

    env_override = os.environ.get("AITER_CORPUS_CONFIG_DIR", "").strip()
    user_cfg_dir = env_override or os.path.expanduser(
        "~/.cache/aiter_corpus_runner/aiter_configs"
    )
    os.makedirs(user_cfg_dir, exist_ok=True)

    original_method = core.AITER_CONFIG.update_config_files
    real_Path = pathlib.Path

    # The target literal in aiter/jit/core.py.  Spelled with the
    # trailing slash because that's what AITER writes — we match
    # byte-identically to minimize the redirect surface.
    TARGET_LITERAL = "/tmp/aiter_configs/"

    def _redirecting_path(*args, **kwargs):
        if args and str(args[0]) == TARGET_LITERAL:
            return real_Path(user_cfg_dir, *args[1:], **kwargs)
        return real_Path(*args, **kwargs)

    def _wrapped(self, tuned_files, merge_name):
        pathlib.Path = _redirecting_path
        try:
            return original_method(self, tuned_files, merge_name)
        finally:
            pathlib.Path = real_Path

    core.AITER_CONFIG.update_config_files = _wrapped

    print(
        f"[aiter_corpus_runner] redirected AITER's hardcoded "
        f"{TARGET_LITERAL!r} -> {user_cfg_dir!r} "
        f"(AITER_CORPUS_CONFIG_DIR)",
        file=sys.stderr,
        flush=True,
    )


def _start_heartbeat_thread() -> None:
    """Spawn a daemon thread that writes a heartbeat line to stderr
    every ``_HEARTBEAT_SEC`` seconds for the lifetime of the child.

    Why this is needed: AITER's JIT pipeline has *several* silent
    phases the runner's idle-output watchdog (``--idle-timeout``)
    would otherwise SIGKILL prematurely under parallel ``--jobs N``:
      * ``FileBaton.wait()`` (serialises concurrent builds of the
        same module) spins in ``while os.path.exists(lock):
        time.sleep(0.2)`` with no output at all until the peer
        releases the lock.
      * ``_jit_compile`` → ``ninja`` / ``hipcc`` can spend tens of
        seconds linking large CK ops after the last compile-unit
        print lands, again with nothing on our stderr pipe.
      * ``dlopen`` of a freshly-built 100+ MB .so on a cold NFS /
        ext4 page cache can take many seconds with no Python-level
        activity.
    A heartbeat thread is the simplest way to make all three visible
    to the runner's watchdog without monkey-patching every silent
    call site individually.
    It also serves as a liveness probe: if the child is genuinely
    wedged at the process level (hard deadlock, GPU reset,
    ``SIGSTOP``), the Python interpreter stops running threads and
    the heartbeats stop — which is exactly when the runner's
    ``--idle-timeout`` should fire.  The thread is the shortest
    possible path to "quiet in our stderr == no Python progress".

    Principled choice: this is a detection aid, not a lock-bypass
    shim.  We do NOT touch the baton semantics, we do NOT pretend a
    lock is released, we do NOT catch any exception from the worker
    thread.  We just emit periodic breadcrumbs so the watchdog
    classifies "waiting for a peer" distinctly from "genuinely
    dead".
    """
    import threading
    import time

    _HEARTBEAT_SEC = 5.0

    start = time.monotonic()
    stop_event = threading.Event()

    def _beat():
        # Use direct ``os.write`` to fd 2 rather than ``print(...,
        # file=sys.stderr)`` because some AITER / ROCm helpers install
        # their own stderr wrappers and we want the heartbeat to
        # survive those unchanged.  fd 2 is the same fd the runner's
        # pump thread is reading from, so a write here directly
        # advances ``last_activity`` in the parent.
        while not stop_event.wait(_HEARTBEAT_SEC):
            msg = (
                f"[aiter_corpus_runner] heartbeat "
                f"(+{time.monotonic() - start:.0f}s)\n"
            )
            try:
                os.write(2, msg.encode())
            except OSError:
                # Parent closed its end of the pipe — we're being
                # torn down.  Let the thread exit naturally on the
                # next iteration when ``stop_event`` is set; until
                # then, there's nothing useful we can do on a
                # broken fd 2.
                return

    t = threading.Thread(
        target=_beat, name="aiter_corpus_runner-heartbeat",
        daemon=True,
    )
    t.start()

    # Expose the stop event so an orderly shutdown (e.g. atexit) can
    # silence the thread cleanly.  We don't register the stop here —
    # the heartbeat running until process exit is fine and the
    # daemon=True thread is reaped at interpreter shutdown.
    globals()["_HEARTBEAT_STOP_EVENT"] = stop_event

    print(
        f"[aiter_corpus_runner] heartbeat thread started "
        f"({_HEARTBEAT_SEC:.0f}s interval) so the parent's "
        f"--idle-timeout doesn't kill us during silent JIT "
        f"compile / baton-wait / dlopen phases",
        file=sys.stderr,
        flush=True,
    )


def _patch_perftest() -> None:
    """Replace ``aiter.test_common.perftest`` and ``run_perftest`` so
    the decorated callable is invoked exactly once and a fixed
    ``avg=_PERFTEST_FAKE_AVG_US`` is returned.  AITER tests
    typically destructure ``(data, avg) = run_torch(...)``; ``data``
    is the only thing the correctness path consumes.  Keeping
    ``avg`` as a plain float (not a pandas Series, as the unpatched
    ``get_trace_perf`` returns) matches the unpacking shape every
    op_test relies on."""
    import aiter.test_common as tc

    def perftest(num_iters: int = 101,
                 num_warmup: int = 2,
                 testGraph: bool = False,
                 num_rotate_args: int = 0,
                 needTrace: bool = False) -> Callable:
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                data = func(*args, **kwargs)
                return data, _PERFTEST_FAKE_AVG_US
            return wrapper
        return decorator

    def run_perftest(func: Callable,
                     *args,
                     num_iters: int = 101,
                     num_warmup: int = 2,
                     testGraph: bool = False,
                     num_rotate_args: int = 0,
                     needTrace: bool = False,
                     **kwargs):
        return func(*args, **kwargs), _PERFTEST_FAKE_AVG_US

    tc.perftest = perftest
    tc.run_perftest = run_perftest

    print(
        f"[aiter_corpus_runner] patched aiter.test_common.perftest / "
        f"run_perftest (single-call, avg={_PERFTEST_FAKE_AVG_US})",
        file=sys.stderr,
        flush=True,
    )


def _seed_rng(seed_env: str) -> None:
    """Seed every RNG we know AITER's op-tests touch so the random
    inputs they build are deterministic across (native, legacy,
    salmon) invocations.  Called after monkey-patching but before
    ``runpy`` fires the user script.

    The seed itself comes from ``AITER_CORPUS_RNG_SEED`` (int).  An
    empty or unset value means "leave RNG alone" — we prefer the
    default of always seeding because the canonical asm-path
    smoke test (``test_moeTopkSoftmax.py``) uses ``torch.randn``
    at tight tolerances and unseeded inputs otherwise cause
    mode-dependent flakiness in sibling tests within the same
    script.  Seeding makes ``native`` either deterministically pass
    or deterministically fail a given label, which means a
    ``native != salmon`` verdict is always a real transpilation
    signal rather than RNG jitter.

    Importing torch here is safe: AITER's top-level ``import aiter``
    already imports torch, and the user op-test does the same — we
    just force the import a few lines earlier to seed it, which
    also conveniently warms up the CUDA context before the test's
    tensor allocations start."""
    if not seed_env:
        print(
            "[aiter_corpus_runner] RNG seeding disabled "
            "(AITER_CORPUS_RNG_SEED is empty); native/legacy/salmon "
            "numerical comparisons may be flaky",
            file=sys.stderr,
            flush=True,
        )
        return
    seed = int(seed_env)
    import random
    random.seed(seed)
    # numpy is an AITER hard dep — if it's missing the op-test would
    # fail on its own import line anyway, but we refuse to silently
    # skip numpy RNG seeding here: an ImportError at this point
    # means the environment is mis-configured and every downstream
    # AITER test will produce garbage.  Surface it now so the user
    # sees a clean "numpy is required" message instead of a confusing
    # numerical mismatch in the first test that happens to use
    # ``np.random``.
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # ``manual_seed_all`` is the documented way to seed every
        # visible CUDA/HIP device; critically, it also reseeds the
        # default generator torch uses when a tensor is explicitly
        # placed on cuda via ``device="cuda"`` (which every AITER
        # op-test does at the top of the file via
        # ``torch.set_default_device("cuda")``).
        torch.cuda.manual_seed_all(seed)
    print(
        f"[aiter_corpus_runner] seeded torch/numpy/random with "
        f"{seed} (AITER_CORPUS_RNG_SEED); identical inputs across "
        f"native/legacy/salmon for this script + args",
        file=sys.stderr,
        flush=True,
    )


def _run_user_script(script: str) -> int:
    """Execute the script as if invoked directly so its
    ``if __name__ == '__main__':`` block fires (and any top-level
    test loops run too).  Make sure the script's own directory is
    on ``sys.path`` so it can import sibling modules.

    AITER's op_tests are mostly argparse-driven scripts that read
    their own ``sys.argv``; if we leave it pointing at the
    ``-m _bootstrap`` invocation, every test trips on an unknown
    arg.  Reset it to look exactly like ``python <script> [args]``
    would, where ``[args]`` is whatever the parent passed via the
    shlex-quoted ``AITER_CORPUS_SCRIPT_ARGS`` envelope."""
    import shlex
    script_dir = os.path.dirname(os.path.abspath(script))
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    extra = os.environ.get("AITER_CORPUS_SCRIPT_ARGS", "")
    extra_args = shlex.split(extra) if extra else []
    sys.argv = [script, *extra_args]

    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (
            0 if e.code is None else 1
        )
        return code
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


def main() -> int:
    gfx = _require_env("AITER_CORPUS_FORCE_GFX")
    _set_gpu_archs(gfx)

    script = _require_env("AITER_CORPUS_SCRIPT")

    strict_tol = float(os.environ.get("AITER_CORPUS_STRICT_TOL", "0.05"))

    # Import + patch must happen *before* runpy fires the user
    # script.  Importing aiter at this point also gives the user a
    # clean stderr line if the install is broken (rather than a
    # confusing failure five seconds into the test).
    try:
        _patch_check_allclose(strict_tol)
        _patch_perftest()
        _patch_aiter_config_dir()
        _start_heartbeat_thread()
    except BaseException:
        traceback.print_exc()
        print(
            "[aiter_corpus_runner] bootstrap patch failed — refusing "
            "to run the user script with an un-patched checkAllclose",
            file=sys.stderr,
            flush=True,
        )
        return 2

    # Seed *after* aiter has been imported-and-patched (both patches
    # above do ``import aiter.test_common``, which pulls in torch
    # transitively).  Seeding here means the user script's very
    # first ``torch.randn`` call sees a generator primed to the
    # runner-controlled seed.
    seed_env = os.environ.get("AITER_CORPUS_RNG_SEED", "0").strip()
    try:
        _seed_rng(seed_env)
    except BaseException:
        traceback.print_exc()
        print(
            "[aiter_corpus_runner] RNG seeding failed — refusing to "
            "run with non-deterministic inputs because the runner's "
            "cross-mode comparison relies on identical tensors",
            file=sys.stderr,
            flush=True,
        )
        return 2

    return _run_user_script(script)


if __name__ == "__main__":
    sys.exit(main())
