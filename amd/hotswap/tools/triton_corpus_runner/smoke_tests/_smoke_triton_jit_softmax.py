"""Smoke test: Triton-JIT softmax kernel from extracted corpus.

Exercises `softmax_kernel` through the same runner/JIT path as upstream
tutorial kernels and checks host-side numerical correctness.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import triton

EXTRACTED_DIR = os.path.expanduser(
    "~/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/hotswap/"
    "transpiler/tools/compare_correctness/kernels/triton/_corpus/extracted"
)
if EXTRACTED_DIR not in sys.path:
    sys.path.insert(0, EXTRACTED_DIR)

from softmax_kernel import softmax_kernel  # noqa: E402


def _reference_softmax(x: np.ndarray) -> np.ndarray:
    x_shifted = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x_shifted)
    return ex / ex.sum(axis=1, keepdims=True)


def main() -> int:
    if not torch.cuda.is_available():
        print("[smoke-jit-softmax] no GPU visible to torch", file=sys.stderr)
        return 2

    device = torch.device("cuda:0")
    print(f"[smoke-jit-softmax] device: {torch.cuda.get_device_name(device)}", flush=True)

    n_rows = 16
    n_cols = 128
    block_size = triton.next_power_of_2(n_cols)
    num_stages = 4
    num_warps = 4

    torch.manual_seed(0)
    x = torch.randn((n_rows, n_cols), device=device, dtype=torch.float32)
    y = torch.empty_like(x)

    grid = (n_rows, 1, 1)
    print(
        f"[smoke-jit-softmax] launching softmax_kernel grid={grid} "
        f"rows={n_rows} cols={n_cols} BLOCK_SIZE={block_size}",
        flush=True,
    )
    softmax_kernel[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=block_size,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    torch.cuda.synchronize(device)

    out_h = y.detach().to("cpu").numpy()
    x_h = x.detach().to("cpu").numpy()
    ref_h = _reference_softmax(x_h)
    finite_out = np.isfinite(out_h)
    finite_ref = np.isfinite(ref_h)
    finite_both = finite_out & finite_ref
    non_finite = int((~finite_both).sum())

    diff = np.abs(out_h - ref_h)
    max_diff = float(np.nanmax(diff)) if finite_both.any() else float("inf")
    mismatches = int((np.nan_to_num(diff, nan=np.inf, posinf=np.inf, neginf=np.inf) > 5e-3).sum())
    print(
        f"[smoke-jit-softmax] result: max_abs_diff={max_diff:.3e} "
        f"n_mismatch_above_5e-3={mismatches}/{out_h.size} "
        f"n_non_finite={non_finite}",
        flush=True,
    )
    if non_finite != 0 or mismatches != 0:
        print(
            "[smoke-jit-softmax] FAIL: non-finite values and/or mismatch above tolerance",
            file=sys.stderr,
        )
        return 1
    print("[smoke-jit-softmax] PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
