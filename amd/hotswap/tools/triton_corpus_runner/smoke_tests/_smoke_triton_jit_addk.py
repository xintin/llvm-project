"""Smoke test #2: drive the SAME ``add_kernel`` Triton source that
``compare_correctness corpus_add_fp32`` uses (literally the same
``_corpus/extracted/add_kernel.py`` file), but through Triton's JIT
path instead of through ``aot_compile.py``.

Strips out everything in upstream tutorial 01-vector-add.py that isn't
the kernel itself: no ``torch.testing.assert_close``, no tensor
``__repr__``, no benchmarks.  Just allocate three buffers, launch
the kernel through Triton, copy back, and check element-by-element on
host with ``numpy``.

If this PASSes in salmon mode but ``01-vector-add.py`` SIGSEGVs:
    The bug is in the upstream tutorial's torch wrapping (likely the
    tensor ``__repr__`` path stepping on memory the kernel touched).

If this CRASHes the same way 01-vector-add.py does:
    The bug is specifically in Triton's JIT path producing a binary
    that the salmon transpile + launch chain cannot handle correctly,
    even though the AOT-compiled .gfx1250.co for the same kernel
    works fine via compare_correctness.  Diff the JIT-emitted .co
    against ``kernels/build/corpus_add_fp32.gfx1250.co`` to find what
    differs.
"""
from __future__ import annotations

import os
import sys

import torch
import triton

EXTRACTED_DIR = os.path.expanduser(
    "~/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/hotswap/"
    "transpiler/tools/compare_correctness/kernels/triton/_corpus/extracted"
)
if EXTRACTED_DIR not in sys.path:
    sys.path.insert(0, EXTRACTED_DIR)

# This is the ``add_kernel`` extracted from triton-tutorial-01-vector-add by
# pull.py — exactly the source ``corpus_add_fp32`` (which we just confirmed
# passes 4/4 match through compare_correctness in salmon mode) consumes.
from add_kernel import add_kernel  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("[smoke-jit] no GPU visible to torch", file=sys.stderr)
        return 2

    device = torch.device("cuda:0")
    print(f"[smoke-jit] device: {torch.cuda.get_device_name(device)}",
          flush=True)

    # Match the corpus_add_fp32 recipe exactly: BLOCK_SIZE=1024, num_warps=4,
    # n_elements=1024 (the smallest of its default shape sweep).  Inputs
    # in [-1, 1] so add stays bit-exact across wave sizes.
    n_elements = 1024
    BLOCK_SIZE = 1024
    num_warps = 4

    torch.manual_seed(0)
    x = (torch.rand(n_elements, device=device, dtype=torch.float32) * 2 - 1)
    y = (torch.rand(n_elements, device=device, dtype=torch.float32) * 2 - 1)
    output = torch.empty_like(x)

    # 1D launch grid; ceil_div(n_elements, BLOCK_SIZE) — same expression
    # the recipe uses.  For n_elements=1024 and BLOCK_SIZE=1024 this is 1.
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)
    print(f"[smoke-jit] launching add_kernel grid={grid} "
          f"BLOCK_SIZE={BLOCK_SIZE} num_warps={num_warps} "
          f"n={n_elements}", flush=True)

    add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    # Synchronise so the kernel is guaranteed done before we touch
    # output's data.  Important: do NOT use ``output.cpu()`` here
    # without a sync first — the salmon path's failure on
    # 01-vector-add.py was exactly during the first host-side touch.
    torch.cuda.synchronize(device)
    print("[smoke-jit] sync ok", flush=True)

    # Move to host via .numpy() through .cpu() — explicit, no implicit
    # __repr__ that would trigger torch's internal repr path.
    out_h = output.detach().to("cpu").numpy()
    x_h = x.detach().to("cpu").numpy()
    y_h = y.detach().to("cpu").numpy()
    expected = x_h + y_h

    # Compare element-by-element on host with numpy.
    import numpy as np
    diff = np.abs(out_h - expected)
    max_diff = float(diff.max())
    n_mismatch = int((diff > 1e-6).sum())

    print(f"[smoke-jit] result: max_abs_diff={max_diff:.2e} "
          f"n_mismatch_above_1e-6={n_mismatch}/{n_elements}",
          flush=True)
    print(f"[smoke-jit] first 4 expected: {expected[:4]}", flush=True)
    print(f"[smoke-jit] first 4 actual  : {out_h[:4]}", flush=True)

    if n_mismatch != 0:
        print("[smoke-jit] FAIL: mismatch above tolerance", file=sys.stderr)
        return 1
    print("[smoke-jit] PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
