"""Tier-1 primitive: fp16 × fp16 → fp16 GEMM with fp32 accumulator.

What it is
==========

A minimal hand-authored tiled matrix multiply that mirrors the upstream
Triton tutorial-03 ``matmul_kernel`` and the production MoE expert
GEMM shape in ``triton_kernels.matmul_ogs``.  The kernel computes:

    C[M, N]  =  A[M, K]  ·  B[K, N]

per-tile, accumulating ``BLOCK_M × BLOCK_N × BLOCK_K`` products in
fp32 across the K dimension, then casting back to fp16 for the store.

Why this is a hand-authored recipe (not a ``corpus_*`` shim)
============================================================

The upstream tutorial kernel in ``_corpus/extracted/matmul_kernel.py``
has two blockers for direct AOT shimming:

1. It is wrapped in ``@triton.autotune(configs=..., key=[...])`` —
   autotune is fundamentally a JIT concept and does not compose with
   the harness's AOT path (we need to fix one tile config at compile
   time).
2. Its ``ACTIVATION: tl.constexpr`` is a *string* constexpr; our
   harness schema accepts only integer constexprs today.

Rather than extend the schema for a string constexpr whose only
value GPT-OSS uses is ``""`` (no activation), this recipe fixes a
single tile config, drops the activation branch, and drops the
``GROUP_SIZE_M`` L2-cache-grouping pid-reorder that is a performance
optimisation unrelated to correctness.  What's left is the same
fp16 × fp16 → fp16 accumulate-in-fp32 shape the upstream tutorial
and ``matmul_ogs`` both compile to.

Why this matters for GPT-OSS
============================

Four of the 15 GPT-OSS Triton kernels are MoE expert GEMMs
(``_matmul_ogs`` × 4) and three of four are outcome (a) — wave-size
clean — per ``gpt-oss-derisking.md §5``.  The fourth uses only a
bounds-check ``s_and_saveexec_b32``.  So the correctness risk on
this shape is **not** wave-size translation; it's the ``tl.dot``
lowering itself (MFMA on gfx942 vs WMMA on gfx1250) plus whatever
gfx11+ ISA details surface.

What the gfx1250 build is supposed to emit
==========================================

Empirically, a fp16 ``tl.dot`` with ``BM = BN = BK = 32`` on gfx1250
wave32 emits eight ``v_permlane16_swap_b32`` instructions (the
WMMA-fragment → DotOperand layout shuffle).  This is the **exact
same cross-lane primitive** ``_attn_fwd``'s softmax row-reduce uses
(`gpt-oss-derisking.md §7.2`), which is why the canary set had
``permlane16_swap`` in its "deliberately not covered today" list
— without a ``tl.dot`` recipe, no idiomatic Triton program emits it.
With this recipe wired, the canary section should demote that
deferred item.

``ds_swizzle_b32`` (the other C2-hard primitive on the deferred
list) emerges at a *different* tile size — the 16×16×16 variant,
which is the companion recipe ``matmul_fp16_16x16`` — so the two
recipes together close the both deferred items.

Harness schema notes
====================

Swept shape dim is ``M``, with ``N`` and ``K`` pinned equal to ``M``
via ``scalar_args`` — a square ``M × M`` GEMM, simplest thing that
hits every code path in the kernel.  Strides are all derivable
expressions: ``stride_am = K = M``, ``stride_ak = 1``,
``stride_bk = N = M``, ``stride_bn = 1``, ``stride_cm = N = M``,
``stride_cn = 1``.

Default shapes stay on multiples of ``BLOCK_M = 32`` (32, 64, 128,
256, 512) so there is no trailing-tile mask path to exercise; the
kernel's code path under test is the dense-tile body plus the
across-K accumulation loop.  Maximum K for the default sweep is
512, so the worst-case fp32 accumulator sees ~ 512 products of
``[-1, 1] × [-1, 1]``; the output magnitude stays bounded and well
within fp16 range after the final cast.

Input range ``[-1, 1]`` keeps the accumulator well-behaved and the
comparator's job straightforward: wave-size reduction-order drift
along K produces < 10 ULPs of relative error per output element on
fp32, which gets further compressed by the fp16 cast; ``rel-rms``
with ``tol = 1e-2`` covers that comfortably while catching any
gross miscompile (e.g. a broken permlane16_swap collapsing the WMMA
fragment shuffle would produce errors of order 1.0).
"""
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
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
        "name": "matmul_fp16",
        "kernel_fn": matmul_kernel,
        "kernel_symbol": "matmul_kernel",
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
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_K": 32,
        },
        # num_warps = 1 is load-bearing here: with num_warps >= 2,
        # Triton splits the 32 x 32 tile across multiple waves, each
        # wave handles its own 16 x 16 WMMA sub-tile independently,
        # and no cross-fragment shuffle is emitted — eliminating the
        # v_permlane16_swap the recipe is designed to pin down.
        # num_warps = 1 keeps the whole tile on one wave and forces
        # the WMMA-fragment -> DotOperand layout shuffle that is the
        # salmon-testing point of this recipe (empirically 8x on
        # gfx1250, verified with llvm-objdump against the built
        # .gfx1250.co).
        "num_warps": 1,
        # num_stages = 1 is also load-bearing: Triton's default
        # (num_stages = 2 on the AMD backend) turns the K-loop into
        # a double-buffered pipeline that stages each iteration's
        # loads through LDS, and the WMMA-fragment -> DotOperand
        # shuffle then rides on those LDS writes rather than on a
        # cross-lane primitive — so the .gfx1250.co ends up with
        # twice the v_wmma count and zero v_permlane16_swap.  With
        # num_stages = 1 the loop is un-pipelined and the shuffle
        # has to go cross-lane, which is what forces the 8x
        # v_permlane16_swap emission that is the salmon-testing
        # point of this recipe.
        "num_stages": 1,
        "shape_dim": "M",
        # Square M = N = K, sweep on M.  All multiples of BLOCK_M so
        # the trailing-tile mask path is never taken.
        "default_shapes": [32, 64, 128, 256, 512],
        "grid": {
            # N and K are pinned equal to M via scalar_args, but the
            # harness's expression evaluator sees only
            # shape_dim + constexprs + harness_constants (scalar_args
            # are consumed by dispatch, not by the evaluator), so we
            # use M directly in the grid expression.
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
        # fp32 accumulator over up to 512 fp16 products of values in
        # [-1, 1]: accumulated error is a few ULPs per output element,
        # compressed by the fp16 cast.  rel-rms with 1e-2 covers
        # wave-size reduction-order drift along K; a gross miscompile
        # in the permlane16_swap lowering (the deferred canary this
        # recipe pins down) produces errors of order 1.0, dwarfing
        # the tolerance.
        "comparator": {"kind": "rel-rms", "tol": 1e-2},
    }
]
