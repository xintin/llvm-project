#!/usr/bin/env python3
"""Focused repro for GPT-OSS MXFP4 decomposition upcast.

The SGLang decomposition path calls ``upcast_from_mxfp(..., axis=-1)`` on
GPT-OSS expert weights before running BF16 ``matmul_ogs``. This script isolates
that Triton kernel so native gfx942 and forced gfx1250->gfx942 Salmon runs can
be compared without loading the full model.
"""

from __future__ import annotations

import argparse
import importlib.util

import torch
import triton
from triton_kernels.numerics_details.mxfp import (
    upcast_from_mxfp,
    upcast_from_mxfp_torch,
)


def _print_environment() -> None:
    triton_kernels_spec = importlib.util.find_spec("triton_kernels")
    print(f"triton_version {getattr(triton, '__version__', 'unknown')}", flush=True)
    print(f"triton_file {triton.__file__}", flush=True)
    print(
        "triton_kernels_path "
        f"{triton_kernels_spec.origin if triton_kernels_spec else 'not found'}",
        flush=True,
    )
    print(f"torch_version {torch.__version__}", flush=True)
    print(f"torch_hip {torch.version.hip}", flush=True)
    print(f"target {triton.runtime.driver.active.get_current_target()}", flush=True)


def _make_inputs(
    experts: int,
    rows_per_expert: int,
    logical_k: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logical_k % 32 != 0:
        raise ValueError(f"logical_k must be divisible by 32, got {logical_k}")
    if logical_k % 2 != 0:
        raise ValueError(f"logical_k must be even for packed MXFP4, got {logical_k}")

    packed_k = logical_k // 2
    scale_k = logical_k // 32
    values = (
        torch.arange(experts * rows_per_expert * packed_k, device=device, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(experts, rows_per_expert, packed_k)
    )
    scales = (
        torch.arange(experts * rows_per_expert * scale_k, device=device, dtype=torch.int64)
        .remainder(8)
        .add(124)
        .to(torch.uint8)
        .reshape(experts, rows_per_expert, scale_k)
    )
    return values, scales


def run_case(experts: int, rows_per_expert: int, logical_k: int) -> None:
    device = "cuda"
    print(
        f"case experts={experts} rows_per_expert={rows_per_expert} logical_k={logical_k}",
        flush=True,
    )
    values, scales = _make_inputs(experts, rows_per_expert, logical_k, device)
    actual = upcast_from_mxfp(values, scales, torch.bfloat16, axis=-1)
    torch.cuda.synchronize()
    expected = upcast_from_mxfp_torch(values, scales, torch.bfloat16, axis=-1)
    torch.cuda.synchronize()
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: actual={actual.shape} expected={expected.shape}")
    mismatch = actual != expected
    mismatch_count = int(mismatch.sum().item())
    if mismatch_count:
        coords = mismatch.nonzero()[:16].cpu().tolist()
        print(f"mismatch_count={mismatch_count}", flush=True)
        for coord in coords:
            idx = tuple(coord)
            print(
                f"mismatch idx={idx} actual={actual[idx].item()} expected={expected[idx].item()}",
                flush=True,
            )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(
        f"PASS shape={tuple(actual.shape)} checksum={actual.float().sum().item():.6e}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--rows-per-expert", type=int, default=128)
    parser.add_argument("--logical-k", type=int, default=2880)
    parser.add_argument(
        "--gpt-oss-w13-shape",
        action="store_true",
        help="Use the decomposed GPT-OSS w13 upcast shape: 32 x 5760 x 2880.",
    )
    args = parser.parse_args()
    _print_environment()
    if args.gpt_oss_w13_shape:
        run_case(experts=32, rows_per_expert=5760, logical_k=2880)
    else:
        run_case(args.experts, args.rows_per_expert, args.logical_k)


if __name__ == "__main__":
    main()
