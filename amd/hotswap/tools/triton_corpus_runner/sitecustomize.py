"""Process-wide Triton target override for runner-launched Python children.

Python imports ``sitecustomize`` automatically when it is importable on
``PYTHONPATH``. ``runner.py`` and ``compare_gpt_oss_sglang.py`` put this
directory on ``PYTHONPATH`` before launching SGLang, so this hook also runs in
SGLang's spawned scheduler/detokenizer processes. That matters because the
runtime-JIT kernels are compiled inside those children, not just in the parent
process that imports ``_bootstrap``.

The regular ``TRITON_OVERRIDE_ARCH`` variable is not enough here: it changes
the arch string but leaves the warp size from the physical gfx942 device. For
the Salmon source path we need a real gfx1250 wave32 ``GPUTarget``.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys


_PATCHED = False


def _append_proof_event(event: dict) -> None:
    path = os.environ.get("HSA_SALMON_PROOF_LOG", "").strip()
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    except OSError as exc:
        print(
            f"[triton_corpus_sitecustomize] failed to write proof event: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _clear_hip_sticky_error() -> None:
    try:
        hip = ctypes.CDLL("libamdhip64.so.7", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        try:
            hip = ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
        except OSError:
            return
    err = hip.hipGetLastError()
    print(
        f"[triton_corpus_sitecustomize] drained HIP sticky error "
        f"(hipGetLastError = {err})",
        file=sys.stderr,
        flush=True,
    )


def _patch_triton_target() -> None:
    global _PATCHED
    if _PATCHED:
        return
    spec = os.environ.get("TRITON_CORPUS_FORCE_TARGET", "").strip()
    if not spec:
        return
    if ":" in spec:
        arch, warp_str = spec.split(":", 1)
        warp = int(warp_str)
    else:
        arch, warp = spec, 32

    try:
        from triton.backends.amd.driver import HIPDriver
        from triton.backends.compiler import GPUTarget
    except Exception as exc:
        print(
            f"[triton_corpus_sitecustomize] cannot patch Triton target yet: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return

    forced = GPUTarget("hip", arch, warp)

    def _get_current_target(self):  # noqa: ANN001
        return forced

    HIPDriver.get_current_target = _get_current_target

    orig_init = HIPDriver.__init__

    def _patched_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        orig_init(self, *args, **kwargs)
        _clear_hip_sticky_error()

    HIPDriver.__init__ = _patched_init
    _PATCHED = True
    _append_proof_event(
        {
            "event": "triton_forced_target",
            "source": "sitecustomize",
            "pid": os.getpid(),
            "arch": arch,
            "warp_size": warp,
            "target_repr": repr(forced),
        }
    )
    print(
        f"[triton_corpus_sitecustomize] forced Triton target = {forced}",
        file=sys.stderr,
        flush=True,
    )


_patch_triton_target()
