"""Child-process bootstrap for the Triton corpus crash sweeper.

This module is imported by ``runner.py`` *inside* the child Python
process, before the user script runs.  Its job is to:

  1. Force Triton to compile for gfx1250 wave32 even though we're
     executing on gfx942 hardware.  Without this the parent's
     ``LD_PRELOAD=libsalmon_intercept.so`` shim is wired up but
     pointless, because Triton emits gfx942 directly and the hotswap
     hook never engages.

  2. Optionally stub out ``triton.testing.do_bench`` and the
     ``perf_report`` benchmark decorator so tutorial scripts don't
     burn minutes on benchmark loops we don't care about — we only
     care about whether the kernel launched at all and whether the
     tutorial's own correctness check passed.

  3. Run the user script via ``runpy.run_path(..., run_name='__main__')``
     so its ``if __name__ == "__main__":`` block fires.

The "what target am I compiling for" question lives in
``HIPDriver.get_current_target``; Triton has a built-in env-var
override (``TRITON_OVERRIDE_ARCH``) that handles the *arch* string
but still pulls ``warp_size`` from the actual device.  On gfx942
that means ``TRITON_OVERRIDE_ARCH=gfx1250`` produces ``("hip",
"gfx1250", 64)``, i.e. wave64 IR for a wave32 arch — which is wrong
for what the salmon transpilation path expects to consume.  We fix
that by replacing the method outright.

Environment knobs the parent sets:

  TRITON_CORPUS_FORCE_TARGET  ``arch[:warp]``  (e.g. ``gfx1250:32``).
                              If unset, no patch is applied — useful
                              for the ``native`` mode where Triton
                              should compile for the actual device.
  TRITON_CORPUS_STUB_BENCH    ``"1"`` to no-op ``do_bench`` /
                              ``perf_report``.  Recommended for
                              tutorial-style scripts.
  TRITON_CORPUS_SCRIPT        absolute path to the user script to run.

The bootstrap exits with the user script's natural exit status (or
the propagated exception's traceback).  Any failure inside the
bootstrap itself (import errors, monkey-patch failures) is loud and
non-zero — we never silently fall back to the un-patched compile
path, because that would defeat the entire point of the run.
"""
from __future__ import annotations

import os
import runpy
import sys
import traceback


def _clear_hip_sticky_error() -> None:
    """Call ``hipGetLastError`` on the LD_PRELOAD'd libamdhip64 to
    drain any sticky error left behind by Triton's HIPUtils probing.

    Triton's HIPUtils (loaded the first time ``driver.active.utils``
    is touched) probes ``hipGetProcAddress`` for symbols like
    ``hipModuleGetFunction`` against a hard-coded HIP version, and
    on the rocm-7.2.1 HIP runtime we LD_PRELOAD some of those probes
    return ``hipErrorInvalidValue``.  HIP latches that into the
    per-thread sticky last-error state, and the very next torch
    op (``torch.rand`` etc.) re-raises it as
    ``torch.AcceleratorError: HIP error: invalid argument``.  We
    drain it here so the user script's torch path starts clean.
    """
    import ctypes
    try:
        hip = ctypes.CDLL("libamdhip64.so.7", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        # Library not loaded under that soname — fall back to the
        # versionless name; one of these two has to be available
        # because the parent LD_PRELOAD'd a libamdhip64.so.* by
        # absolute path before we got here.
        hip = ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
    err = hip.hipGetLastError()
    print(
        f"[triton_corpus_runner] drained HIP sticky error "
        f"(hipGetLastError = {err})",
        file=sys.stderr,
        flush=True,
    )


def _force_triton_target(spec: str) -> None:
    """Replace ``HIPDriver.get_current_target`` so it returns the
    requested ``GPUTarget``.  ``spec`` is ``"<arch>"`` or
    ``"<arch>:<warp_size>"`` (warp size defaults to 32 because every
    arch we'd want to override to today is wave32).

    Also wrap ``HIPDriver.__init__`` to drain HIP's sticky last-error
    state at the end of init — see ``_clear_hip_sticky_error``.
    """
    if ":" in spec:
        arch, warp_str = spec.split(":", 1)
        warp = int(warp_str)
    else:
        arch, warp = spec, 32

    from triton.backends.amd.driver import HIPDriver
    from triton.backends.compiler import GPUTarget

    forced = GPUTarget("hip", arch, warp)

    def _get_current_target(self):  # noqa: ANN001
        return forced

    HIPDriver.get_current_target = _get_current_target

    _orig_init = HIPDriver.__init__

    def _patched_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        _orig_init(self, *args, **kwargs)
        _clear_hip_sticky_error()

    HIPDriver.__init__ = _patched_init
    print(
        f"[triton_corpus_runner] forced Triton target = {forced}",
        file=sys.stderr,
        flush=True,
    )


def _stub_triton_benchmarks() -> None:
    """No-op ``triton.testing.do_bench`` and the ``perf_report``
    decorator so tutorials don't spend the whole wall-clock budget
    running a benchmark grid we're discarding anyway.

    Correctness assertions (``torch.testing.assert_close`` and the
    like) are untouched — that's exactly the signal we want to keep.
    """
    import triton.testing as tt

    def _noop_bench(*args, **kwargs):
        return 0.0

    tt.do_bench = _noop_bench
    if hasattr(tt, "do_bench_cudagraph"):
        tt.do_bench_cudagraph = _noop_bench

    if hasattr(tt, "Mark"):
        _orig_run = tt.Mark.run

        def _noop_run(self, *args, **kwargs):
            # Skip the benchmark sweep entirely.  The decorator itself
            # is harmless; only .run() does the time-consuming work.
            return None

        tt.Mark.run = _noop_run

    print(
        "[triton_corpus_runner] stubbed triton.testing.do_bench / Mark.run",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    target = os.environ.get("TRITON_CORPUS_FORCE_TARGET", "").strip()
    if target:
        _force_triton_target(target)

    if os.environ.get("TRITON_CORPUS_STUB_BENCH") == "1":
        _stub_triton_benchmarks()

    script = os.environ.get("TRITON_CORPUS_SCRIPT", "").strip()
    if not script:
        print(
            "[triton_corpus_runner] TRITON_CORPUS_SCRIPT not set",
            file=sys.stderr,
        )
        return 2

    # Run the script as if invoked directly (i.e. fire its
    # ``if __name__ == '__main__':`` block).  Make sure the script's
    # own directory is on sys.path so it can import sibling modules.
    script_dir = os.path.dirname(os.path.abspath(script))
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        # Propagate the script's intended exit code.
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        return code
    except BaseException:
        # Print a real traceback (not a swallowed exception) so the
        # parent's stderr capture has actionable detail.
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
