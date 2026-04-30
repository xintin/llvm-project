"""Dump the gfx1250 wave32 .hsaco that Triton's *JIT runtime* produces
for ``add_kernel``, by intercepting at the CompiledKernel layer and
exiting before the launcher is ever called.

Calls ``add_kernel.run(...)`` which is the JIT entry point that handles
specialization, caching, and compilation — but we monkey-patch
``triton.compiler.compile`` to capture the artefact and abort the JIT
before launch.

Output goes to /tmp/jit_addk_gfx1250.hsaco and can be diffed against
/tmp/aotpath_addk_gfx1250.hsaco to see whether AOT-compile-API and
JIT-compile-API produce the same binary for the same kernel + constexprs
+ target.
"""
from __future__ import annotations

import os
import sys

import torch
import triton
from triton.compiler import compiler as _tc

EXTRACTED_DIR = os.path.expanduser(
    "~/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/hotswap/"
    "transpiler/tools/compare_correctness/kernels/triton/_corpus/extracted"
)
sys.path.insert(0, EXTRACTED_DIR)
from add_kernel import add_kernel  # noqa: E402


_real_compile = _tc.compile
_captured_path = "/tmp/jit_addk_gfx1250.hsaco"


class _Done(SystemExit):
    """Exit after capturing the artefact — we don't want to launch."""


def _capture(src, target=None, options=None):
    print(f"[capture-jit] triton.compile target={target!r}", flush=True)
    print(f"[capture-jit]   src.fn={src.fn.__name__} "
          f"signature={src.signature} constexprs={src.constexprs} "
          f"attrs={src.attrs}", flush=True)
    compiled = _real_compile(src, target=target, options=options)
    hsaco = compiled.asm.get("hsaco")
    if isinstance(hsaco, str):
        with open(hsaco, "rb") as f:
            data = f.read()
    else:
        data = bytes(hsaco)
    with open(_captured_path, "wb") as f:
        f.write(data)
    print(f"[capture-jit] wrote {_captured_path} ({len(data)} bytes)",
          flush=True)
    print(f"[capture-jit] CompiledKernel.metadata="
          f"{getattr(compiled, 'metadata', None)}", flush=True)
    raise _Done(0)


_tc.compile = _capture


def main() -> int:
    if not torch.cuda.is_available():
        print("[capture-jit] no GPU visible to torch", file=sys.stderr)
        return 2

    n_elements = 1024
    BLOCK_SIZE = 1024
    num_warps = 4
    device = torch.device("cuda:0")

    x = torch.randn(n_elements, device=device, dtype=torch.float32)
    y = torch.randn(n_elements, device=device, dtype=torch.float32)
    output = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)

    add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
