"""Tier-1 primitive: fp16 GEMM at 16x16x16 tiles — forces `ds_swizzle_b32`.

What it is
==========

Companion to ``matmul_fp16`` with a smaller tile size.  Same kernel
semantics (C = A × B, fp16 × fp16 → fp16 with fp32 accumulator),
different WMMA fragment layout:

- ``matmul_fp16`` at 32×32×32 with num_warps=1 / num_stages=1 emits
  ``v_permlane16_swap_b32 × 8`` for the WMMA-fragment → DotOperand
  shuffle — the attn_fwd softmax variant.
- ``matmul_fp16_16x16`` here at 16×16×16 with num_warps=1 /
  num_stages=1 emits ``ds_swizzle_b32 × 2`` for the same logical
  step — the ``_sum_bitmatrix_rows`` variant.

Between this recipe and ``matmul_fp16``, both C2-hard primitives
``gpt-oss-derisking.md §7.2`` audited as "deferred until a
``tl.dot`` recipe lands" are now covered.

What forces `ds_swizzle_b32` specifically
=========================================

The choice is purely the tile size: at ``BLOCK_M = BLOCK_N =
BLOCK_K = 16``, the WMMA 16×16 fragment is already in the right
layout for one wave-local sub-reduction but needs a
``ds_swizzle_b32`` for the specific bit-pattern rearrangement.  At
32×32 the fragment spills across half-wave boundaries and a
``v_permlane16_swap_b32`` is emitted instead.  Empirically verified
via the pre-authoring tile-size probe: BM=BN=BK=16 with
num_warps=1 and num_stages=1 produces exactly ``ds_swizzle × 2``
and ``v_wmma × 1`` with no other cross-lane ops.

Harness schema notes
====================

Same single-swept-dim structure as ``matmul_fp16``: square ``M × M``
GEMM with ``N = K = M`` pinned via ``scalar_args``, multiples of
``BLOCK_M = 16`` to avoid the trailing-tile mask path.  The default
sweep ``[16, 32, 64, 128, 256]`` goes a bit further on the small
end (16 is a single tile, the minimum legal size for this recipe)
than ``matmul_fp16`` does because 16-tile kernels stay cheap
even at M=256.

``num_warps = 1`` and ``num_stages = 1`` have the same load-bearing
role as in ``matmul_fp16``: multiple warps would split the tile
across waves and eliminate the cross-fragment shuffle; num_stages
≥ 2 would hide it behind LDS double-buffering.  See the companion
recipe for the full reasoning.
"""
import triton
import triton.language as tl


@triton.jit
def matmul_kernel_16x16(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


RECIPES = [
    {
        "name": "matmul_fp16_16x16",
        "kernel_fn": matmul_kernel_16x16,
        "kernel_symbol": "matmul_kernel_16x16",
        "signature": {
            "A":         "*fp16",
            "B":         "*fp16",
            "C":         "*fp16",
            "M":         "i32",
            "N":         "i32",
            "K":         "i32",
            "stride_am": "i32",
            "stride_ak": "i32",
            "stride_bk": "i32",
            "stride_bn": "i32",
            "stride_cm": "i32",
            "stride_cn": "i32",
        },
        "constexprs": {
            "BLOCK_M": 16,
            "BLOCK_N": 16,
            "BLOCK_K": 16,
        },
        "num_warps":  1,
        "num_stages": 1,
        "shape_dim": "M",
        "default_shapes": [16, 32, 64, 128, 256],
        "grid": {
            "x": "ceil_div(M, BLOCK_M)",
            "y": "ceil_div(M, BLOCK_N)",
            "z": "1",
        },
        "scalar_args": {
            "N":         "M",
            "K":         "M",
            "stride_am": "M",
            "stride_ak": "1",
            "stride_bk": "M",
            "stride_bn": "1",
            "stride_cm": "M",
            "stride_cn": "1",
        },
        "inputs": [
            {"name": "A", "dtype": "fp16", "elems": "M * M",
             "range_lo": -1.0, "range_hi": 1.0},
            {"name": "B", "dtype": "fp16", "elems": "M * M",
             "range_lo": -1.0, "range_hi": 1.0},
        ],
        "outputs": [
            {"name": "C", "dtype": "fp16", "elems": "M * M"},
        ],
        # Same tolerance reasoning as matmul_fp16.  At the smaller
        # tile size the worst-case accumulator depth is shorter, so
        # this tolerance has even more headroom against rounding
        # drift than the companion recipe.
        "comparator": {"kind": "rel-rms", "tol": 1e-2},
    }
]
