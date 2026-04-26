"""GPT-OSS primitive: Root-Mean-Square Normalization (RMSNorm).

What it is
==========

RMSNorm is the normalization primitive GPT-OSS uses in every
transformer block (replacing the more traditional LayerNorm).  The
forward pass is:

    rms = sqrt(mean(x ** 2) + eps)
    y   = (x / rms) * w

per row of a ``(M, N)`` input matrix, with a per-feature weight
``w`` of shape ``(N,)``.  There is no bias and no mean subtraction —
that's the whole difference from LayerNorm, and it matters for
performance because the reduction is half the work.

Why this is in the `compare_correctness` corpus
==============================================

RMSNorm was flagged as **the first open coverage gap** in
``hotswap/docs/gpt-oss-derisking.md §2.4``: the scope-discovery
capture (15 GPT-OSS Triton kernels) contains no norm kernel, but
RMSNorm is demonstrably on the model's forward path.  This is the
minimal hand-authored stand-in that lets salmon verification
proceed without waiting for a full GPT-OSS capture pass to
re-emit the real kernel.  When a live RMSNorm `.hsaco` lands from
scope-discovery, it should produce the same cross-lane IR shape
as this kernel — `tl.sum` over the feature axis (DPP + permlanex16
on gfx1250) — so a regression caught here is a regression that would
have caught the captured kernel too.

Why fp32
========

GPT-OSS runs attention in bf16 and MoE in mxfp4, but the `compare_correctness` harness today supports fp16 / bf16 / fp32 / fp64 /
i32 / i64 only; fp8 / mxfp4 dtypes are future schema work.  A
fp32 RMSNorm keeps us in the "full precision, realistic IR shape"
regime — the cross-lane primitives Triton emits for the reduction
are identical in fp32 and bf16 builds, so a gfx1250 fp32 RMSNorm
exercises the same salmon translation path the bf16 production
kernel hits, minus the dtype-conversion prologue / epilogue.  A
bf16 variant can be added alongside once the harness gains bf16
input support on the salmon column (today's `vecadd_f16` is the
only fp16-family recipe).

Status note (current salmon verdict: WRONG on every shape)
==========================================================

Salmon miscompiles this recipe at every shape with the signature
``ref / actual ≈ 0.55`` — consistent with the sum-of-squares being
~3.3x too large.  The disassembly shows ``v_add_f32_dpp × 4``,
``v_permlanex16_b32 × 1``, and ``v_pk_mul_f32 × 8``; of those,
``v_add_f32_dpp`` + ``v_permlanex16_b32`` are confirmed handled
correctly by ``canary_dpp_compound_add_fp32``.  The **delta** that
separates this (WRONG) from the passing compound-add canary
(MATCH) is the ``v_pk_mul_f32`` (VOP3P packed-fp32 multiply)
Triton uses for the ``x * x`` step — plus the sqrt + reciprocal
sequence feeding off the broadcast reduction result.

A previous revision of this docstring attributed the failure to a
missing gfx11+ ``v_*_num_*`` opcode family.  That attribution has
been refuted: (a) the apparent ``v_min_num_f64`` in the
disassembly is actually a ``v_pk_mul_f32`` 64-bit VOP3P
instruction that ``llvm-objdump`` incorrectly splits into two
32-bit "instructions" (one undecoded ``.long``, one incidental
``v_min_num_f64_e32`` match); (b) ``raise_cli`` successfully
raises all 389 / 389 instructions without reporting any unsupported
opcode; (c) the ``v_*_num_*`` gfx12 mnemonics are LLVM aliases for
the ``V_MAX_F32`` / ``V_MIN_F32`` / ``V_MED3_F32`` pseudos that the
raiser already handles.

The correct triage playbook is therefore:

1. Diff the raised IR of this recipe against
   ``canary_dpp_compound_add_fp32`` to pinpoint the semantic
   difference — specifically around the ``v_pk_mul_f32`` handler in
   ``handle_valu_vop3p.cpp`` and how the broadcast of the reduction
   result is routed into the subsequent ``v_pk_mul_f32 * x`` uses.
2. Use ``raise_cli --write-hsaco`` to produce a standalone gfx942
   HSACO from the raised IR and run it directly on gfx942 hardware
   (bypassing the hotswap runtime path); confirms whether the bug
   is in the raised IR itself vs downstream in the hotswap hook.

Harness schema notes
====================

Same shape-and-stride idiom as ``corpus_layernorm_fp32``:

- ``shape_dim = N`` (the feature / reduction width); ``M`` (row
  count) is fixed via ``harness_constants`` because the harness
  sweeps one named dim.
- ``stride = N`` (row-major, no padding) goes in ``scalar_args`` as
  an expression; ``eps = 1e-6`` goes in as a float literal.
- Single output ``Y`` of shape ``(M, N)`` with a ``rel-rms``
  comparator: wave-size reduction-order drift dominates pointwise
  relative error for the elements near the row-mean-square boundary;
  the RMS norm stays far below the tolerance under any correct
  lowering, and a broken lowering (e.g. a ``ds_bpermute`` collapse
  in the sum-of-squares) produces RMS error on the order of the
  per-row standard deviation — well above tolerance.

``BLOCK_SIZE = 1024`` covers the default sweep (``N <= 1024``) with
a single pass.  Larger ``N`` would need the streaming-over-BLOCK_SIZE
pattern corpus_layernorm uses; out of scope for this recipe (add a
second recipe named ``rmsnorm_fp32_streaming`` if / when we need it).
"""
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    X, Y, W,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    X_row = X + pid * stride
    Y_row = Y + pid * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_row + cols, mask=mask, other=0.0)
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N
    rms_inv = 1.0 / tl.sqrt(mean_sq + eps)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = x * rms_inv * w
    tl.store(Y_row + cols, y, mask=mask)


RECIPES = [
    {
        "name": "rmsnorm_fp32",
        "kernel_fn": rmsnorm_kernel,
        "kernel_symbol": "rmsnorm_kernel",
        "signature": {
            "X":      "*fp32",
            "Y":      "*fp32",
            "W":      "*fp32",
            "stride": "i32",
            "N":      "i32",
            "eps":    "fp32",
        },
        "constexprs": {
            "BLOCK_SIZE": 1024,
        },
        "harness_constants": {
            "M": 16,
        },
        "num_warps": 1,
        "shape_dim": "N",
        "default_shapes": [128, 256, 512, 1024],
        "grid": {
            "x": "M",
            "y": "1",
            "z": "1",
        },
        "scalar_args": {
            "stride": "N",
            "eps":    1e-6,
        },
        "inputs": [
            {"name": "X", "dtype": "fp32", "elems": "M * N",
             "range_lo": -1.0, "range_hi": 1.0},
            {"name": "W", "dtype": "fp32", "elems": "N",
             "range_lo":  0.5, "range_hi": 1.5},
        ],
        "outputs": [
            {"name": "Y", "dtype": "fp32", "elems": "M * N"},
        ],
        # Wave-size reduction-order drift on the sum-of-squares
        # produces a few ULPs of pointwise error after the sqrt /
        # reciprocal; RMS norm stays well below 1e-4 under any
        # correct lowering.  A broken reduction (bpermute collapse,
        # DPP drop) produces RMS error on the order of the per-row
        # standard deviation — orders of magnitude above tolerance.
        "comparator": {"kind": "rel-rms", "tol": 1e-3},
    }
]
