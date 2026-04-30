"""GPT-OSS primitive: Rotary Position Embedding (RoPE, half-rotation form).

What it is
==========

RoPE is the positional encoding GPT-OSS applies to the Q and K
tensors before attention.  For each position ``t`` and each feature
pair ``(i, i + HEAD_DIM/2)``, the two-dim rotation is:

    y[t, i]              =  x[t, i] * cos(theta[t, i])
                           - x[t, i + HEAD_DIM/2] * sin(theta[t, i])
    y[t, i + HEAD_DIM/2] =  x[t, i + HEAD_DIM/2] * cos(theta[t, i])
                           + x[t, i] * sin(theta[t, i])

where ``theta[t, i]`` depends on position and feature index (for
GPT-OSS, additionally scaled by YaRN interpolation between short
and long context base frequencies).  This recipe takes ``cos`` /
``sin`` as **precomputed inputs** — the GPT-OSS driver would
materialise them once per forward pass — so the kernel itself is
just the elementwise rotation arithmetic.  That keeps the test
focused on the retargeting question (does salmon preserve
strided-load + elementwise-fma semantics end-to-end?) rather than
on the libdevice ``cos`` / ``sin`` call path, which is already
covered by ``corpus_asin_fp32``.

The scope-discovery capture in ``hotswap/docs/gpt-oss-derisking.md
§2.4`` flagged RoPE as "applied inline in ``_attn_fwd``, transitively
covered but unconfirmed at source" — so this is the first
*standalone* RoPE test in the harness, independent of whether we
eventually capture GPT-OSS's fused attention kernel.

Why fp32
========

GPT-OSS runs attention in bf16 but RoPE is typically computed in
fp32 (to avoid accumulated rotation error over long contexts) and
cast back.  An fp32 recipe therefore matches the precision GPT-OSS
actually uses for the rotation arithmetic, even though the
surrounding kernel is bf16.

Harness schema notes
====================

Swept shape dim is ``N_POS`` (number of positions / rows).
``HEAD_DIM`` is constexpr and fixed at 64 — GPT-OSS-20B has
``HEAD_DIM = 64`` (GPT-OSS-120B is also 64 per head, but uses GQA
with more key-value heads; the per-head width stays 64).
``HEAD_DIM = 64`` means half-rotation width is 32, which maps
cleanly to one thread per feature on wave32 (one wave per position)
and one thread per feature with the upper 32 lanes idle on wave64.

No cross-lane primitives are emitted (pure elementwise math); the
expected instruction mix on gfx1250 is strided ``global_load_b32``
for X, ``global_load_b32`` for cos / sin, ``v_fma_f32`` /
``v_mul_f32`` for the rotation, and ``global_store_b32`` for Y.
The interest here is the strided access pattern (two halves of X
per position, one cos / sin row per position) and whether salmon
preserves it correctly across the gfx1250 → gfx942 lowering.
"""
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    X, COS, SIN, Y,
    N_POS,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    HALF: tl.constexpr = HEAD_DIM // 2
    offs = tl.arange(0, HALF)
    cos = tl.load(COS + pid * HALF + offs)
    sin = tl.load(SIN + pid * HALF + offs)
    x1 = tl.load(X + pid * HEAD_DIM + offs)
    x2 = tl.load(X + pid * HEAD_DIM + HALF + offs)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    tl.store(Y + pid * HEAD_DIM + offs, y1)
    tl.store(Y + pid * HEAD_DIM + HALF + offs, y2)


RECIPES = [
    {
        "name": "rope_fp32",
        "kernel_fn": rope_kernel,
        "kernel_symbol": "rope_kernel",
        "signature": {
            "X":     "*fp32",
            "COS":   "*fp32",
            "SIN":   "*fp32",
            "Y":     "*fp32",
            "N_POS": "i32",
        },
        "constexprs": {
            "HEAD_DIM": 64,
        },
        "num_warps": 1,
        "shape_dim": "N_POS",
        "default_shapes": [8, 32, 128, 512],
        "grid": {
            "x": "N_POS",
            "y": "1",
            "z": "1",
        },
        "inputs": [
            {"name": "X",   "dtype": "fp32", "elems": "N_POS * HEAD_DIM",
             "range_lo": -1.0, "range_hi": 1.0},
            # cos / sin tables of "real" trig values live in [-1, 1].
            # Random uniform in that range is not actually a valid
            # (cos, sin) pair (no unit-circle constraint), but that
            # is fine for correctness testing: the recipe is the
            # kernel, and the kernel's semantics are defined for any
            # input pair, not only for (cos θ, sin θ) on the unit
            # circle.  A regression will show up independent of
            # whether the inputs are unit-circle-consistent.
            {"name": "COS", "dtype": "fp32", "elems": "N_POS * HEAD_DIM / 2",
             "range_lo": -1.0, "range_hi": 1.0},
            {"name": "SIN", "dtype": "fp32", "elems": "N_POS * HEAD_DIM / 2",
             "range_lo": -1.0, "range_hi": 1.0},
        ],
        "outputs": [
            {"name": "Y", "dtype": "fp32", "elems": "N_POS * HEAD_DIM"},
        ],
        # Two fma and two multiply-subtract per element.  With inputs
        # in [-1, 1], the output stays in roughly [-2, 2] and any
        # correct elementwise lowering produces the same bits up to
        # the fma ULP budget.  rel 1e-5 covers the expected
        # rounding drift; a broken strided-load (the likely failure
        # mode for this recipe, going on the swiglu precedent) would
        # produce errors many orders of magnitude above that.
        "comparator": {"kind": "rel", "tol": 1e-5},
    }
]
