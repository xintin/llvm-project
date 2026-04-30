"""AOT correctness recipe for GPT-OSS MXFP4 -> BF16 upcast.

This mirrors the lane-position-sensitive fallback path in
``triton_kernels.numerics_details.mxfp_details._upcast_from_mxfp`` without
using TensorDescriptor arguments, so the existing compare_correctness sidecar
schema can dispatch it directly.  The real GPT-OSS decomposition path exposed
that WaveNative must not suppress C5 equality predicates: ``workitem.id.x``-
derived ``icmp eq`` sites gate sign selection for packed e2m1 values.

Expected Salmon outcome after the classifier fix is a loud raise/load refusal
with ``workitem.id.x-predicate-chain-classifier``, not a native-vs-Salmon
numeric mismatch.
"""

import triton
import triton.language as tl


_BLOCK_OUT_DIM = 64
_BLOCK_QUANT_DIM = 128
_QUANT_DIM = 2880
_PACKED_PER_ROW = _QUANT_DIM // 2
_SCALE_PER_ROW = _QUANT_DIM // 32


@triton.jit
def upcast_mxfp4_bf16_kernel(
    out_ptr,
    mx_tensor_ptr,
    mx_scale_ptr,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    PACKED_PER_ROW: tl.constexpr,
    SCALE_PER_ROW: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % 32 == 0)

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)
    offs_packed = tl.arange(0, BLOCK_SIZE_QUANT_DIM // 2)[None, :].to(tl.int64)
    offs_scale = tl.arange(0, BLOCK_SIZE_QUANT_DIM // 32)[None, :].to(tl.int64)

    start_outer = outer_block * BLOCK_SIZE_OUT_DIM
    start_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_packed = start_quant // 2
    start_scale = start_quant // 32

    row = start_outer + offs_outer
    packed_col = start_packed + offs_packed
    scale_col = start_scale + offs_scale

    packed_mask = (row < outer_dim) & (packed_col < quant_dim // 2)
    packed = tl.load(
        mx_tensor_ptr + row * PACKED_PER_ROW + packed_col,
        mask=packed_mask,
        other=0,
    )

    # e2m1 unpack, matching triton_kernels' non-Hopper branch.
    dst_bias: tl.constexpr = 127
    dst_0p5: tl.constexpr = 16128
    dst_m_bits: tl.constexpr = 7
    em0 = packed & 0x07
    em1 = packed & 0x70
    x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | (
        (packed & 0x08).to(tl.uint16) << 12
    )
    x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | (
        (packed & 0x80).to(tl.uint16) << 8
    )
    x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
    x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
    x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
    x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)

    values = tl.interleave(x0, x1).to(tl.bfloat16, bitcast=True)
    values = values.reshape(
        [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 32, 32]
    )

    scale_mask = (row < outer_dim) & (scale_col < tl.cdiv(quant_dim, 32))
    scale = tl.load(
        mx_scale_ptr + row * SCALE_PER_ROW + scale_col,
        mask=scale_mask,
        other=0,
    )
    bf16_scale = (scale.to(tl.uint16) << 7).to(tl.bfloat16, bitcast=True)
    bf16_scale = bf16_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 32, 1])

    out = values * bf16_scale
    out = tl.where(scale.reshape(bf16_scale.shape) == 0xFF, float("nan"), out)
    out = out.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])

    offs_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    out_col = start_quant + offs_quant
    out_mask = (row < outer_dim) & (out_col < quant_dim)
    tl.store(out_ptr + row * quant_dim + out_col, out, mask=out_mask)


RECIPES = [
    {
        "name": "upcast_mxfp4_bf16",
        "kernel_fn": upcast_mxfp4_bf16_kernel,
        "kernel_symbol": "upcast_mxfp4_bf16_kernel",
        "signature": {
            "out_ptr": "*bf16",
            "mx_tensor_ptr": "*u8",
            "mx_scale_ptr": "*u8",
            "outer_dim": "i32",
            "quant_dim": "i32",
        },
        "constexprs": {
            "BLOCK_SIZE_OUT_DIM": _BLOCK_OUT_DIM,
            "BLOCK_SIZE_QUANT_DIM": _BLOCK_QUANT_DIM,
            "PACKED_PER_ROW": _PACKED_PER_ROW,
            "SCALE_PER_ROW": _SCALE_PER_ROW,
        },
        "harness_constants": {
            "QUANT_DIM": _QUANT_DIM,
        },
        "scalar_args": {
            "outer_dim": "N_ROWS",
            "quant_dim": "QUANT_DIM",
        },
        "num_warps": 4,
        "shape_dim": "N_ROWS",
        "default_shapes": [128, 5760],
        "grid": {
            "x": "ceil_div(N_ROWS, BLOCK_SIZE_OUT_DIM)",
            "y": "ceil_div(QUANT_DIM, BLOCK_SIZE_QUANT_DIM)",
            "z": "1",
        },
        "inputs": [
            {"name": "mx_tensor_ptr", "dtype": "u8", "elems": "N_ROWS * PACKED_PER_ROW"},
            {"name": "mx_scale_ptr", "dtype": "u8", "elems": "N_ROWS * SCALE_PER_ROW"},
        ],
        "outputs": [
            {"name": "out_ptr", "dtype": "bf16", "elems": "N_ROWS * QUANT_DIM"},
        ],
        "comparator": {"kind": "abs", "tol": 0.0},
    }
]
