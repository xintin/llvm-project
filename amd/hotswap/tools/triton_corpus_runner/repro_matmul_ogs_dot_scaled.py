#!/usr/bin/env python3
"""Focused repro for GPT-OSS/SGLang matmul_ogs gfx1250 codegen.

Run with this directory on PYTHONPATH so ``sitecustomize.py`` can force the
Triton target:

  PYTHONPATH="$PWD:/home/nithin/triton/python/triton_kernels" \
  TRITON_CORPUS_FORCE_TARGET=gfx1250:32 \
  TRITON_CACHE_DIR=/tmp/triton_matmul_ogs_repro \
  TRITON_ALWAYS_COMPILE=1 \
  ./.venv-sglang-rocm720/bin/python repro_matmul_ogs_dot_scaled.py

Default mode reproduces Triton's mixed bf16 x e2m1 ``tt.dot_scaled`` failure.
``--decomposed-bf16`` uses dequantized BF16 weights to prove codegen gets past
that failure; without Salmon preload it is expected to stop later at HIP load
with "no kernel image is available".
"""

from __future__ import annotations

import argparse
import importlib.util

import torch
import triton
from triton_kernels.matmul_ogs import (
    FlexCtx,
    FusedActivation,
    GatherIndx,
    PrecisionConfig,
    RoutingData,
    matmul_ogs,
)
from triton_kernels.numerics import InFlexData
from triton_kernels.specialize import FnSpecs
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details import layout


def _print_environment() -> None:
    triton_kernels_spec = importlib.util.find_spec("triton_kernels")
    print(f"triton_version {getattr(triton, '__version__', 'unknown')}", flush=True)
    print(f"triton_file {triton.__file__}", flush=True)
    print(f"triton_kernels_path {triton_kernels_spec.origin if triton_kernels_spec else 'not found'}", flush=True)
    print(f"torch_version {torch.__version__}", flush=True)
    print(f"torch_hip {torch.version.hip}", flush=True)
    print(f"torch_cuda {torch.version.cuda}", flush=True)
    print(f"target {triton.runtime.driver.active.get_current_target()}", flush=True)


def _activation() -> FusedActivation:
    return FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (1.0, 7.0),
    )


def run_mxfp4() -> None:
    experts, tokens, topk, k_dim, n_dim = 32, 512, 4, 2880, 5760
    rows = tokens * topk
    device = "cuda"

    x = torch.empty((tokens, k_dim), device=device, dtype=torch.bfloat16)
    w_packed = torch.empty((experts, n_dim, k_dim // 2), device=device, dtype=torch.uint8)
    w_scale_raw = torch.empty((experts, n_dim, k_dim // 32), device=device, dtype=torch.uint8)

    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=1, num_warps=8
    )
    w = convert_layout(
        wrap_torch_tensor(w_packed.transpose(-2, -1), dtype=FP4),
        value_layout,
        **value_layout_opts,
    )
    w_scale = convert_layout(
        wrap_torch_tensor(w_scale_raw.transpose(-2, -1)),
        scale_layout,
        **scale_layout_opts,
    )
    precision = PrecisionConfig(weight_scale=w_scale, flex_ctx=FlexCtx(rhs_data=InFlexData()))

    src = (torch.arange(rows, device=device, dtype=torch.int32) % tokens).contiguous()
    dst = torch.arange(rows, device=device, dtype=torch.int32)
    routing = RoutingData(
        gate_scal=None,
        expt_hist=None,
        n_expts_tot=experts,
        n_expts_act=topk,
        expected_tokens_per_expt=64,
    )
    bias = torch.empty((experts, n_dim), device=device, dtype=torch.float32)
    y = torch.empty((rows, n_dim // 2), device=device, dtype=torch.bfloat16)

    matmul_ogs(
        x,
        w,
        bias,
        routing_data=routing,
        gather_indx=GatherIndx(src, dst),
        precision_config=precision,
        fused_activation=_activation(),
        y=y,
    )


def run_decomposed_bf16() -> None:
    experts, rows, k_dim, n_dim = 1, 64, 2880, 5760
    device = "cuda"
    x = torch.zeros((rows, k_dim), device=device, dtype=torch.bfloat16)
    w = torch.zeros((experts, k_dim, n_dim), device=device, dtype=torch.bfloat16)
    bias = torch.zeros((experts, n_dim), device=device, dtype=torch.float32)
    y = torch.empty((rows, n_dim // 2), device=device, dtype=torch.bfloat16)
    routing = RoutingData(
        gate_scal=None,
        expt_hist=None,
        n_expts_tot=experts,
        n_expts_act=1,
        expected_tokens_per_expt=rows,
    )
    matmul_ogs(
        x,
        w,
        bias,
        routing_data=routing,
        precision_config=PrecisionConfig(),
        fused_activation=_activation(),
        y=y,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposed-bf16", action="store_true")
    args = parser.parse_args()
    _print_environment()
    if args.decomposed_bf16:
        run_decomposed_bf16()
    else:
        run_mxfp4()


if __name__ == "__main__":
    main()
