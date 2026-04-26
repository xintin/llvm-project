"""Canary for `v_permlanex16_b32` emission on wave32 (C2-hard).

What it targets
===============

The P2 / P4 items in ``wave-size-translation.md §5.3`` — the
``permlane*`` intrinsic lifts.  Both ``v_permlanex16_b32`` (half-wave
read from the other half) and ``v_permlane16_swap_b32`` (half-wave
swap between the two halves) are C2-hard primitives covered by the
same rewrite class, though the instructions themselves are distinct.
Per ``gpt-oss-derisking.md §7.2`` the GPT-OSS corpus hits both
variants: ``_attn_fwd`` and ``_bitmatrix_metadata_compute_stage2``
use ``v_permlane16_swap_b32``; ``_bitmatrix_metadata_compute_stage1``
uses ``v_permlanex16_b32``.

This canary pins down the ``v_permlanex16_b32`` variant.  A separate
canary for the ``_swap`` variant would need a ``tl.dot`` kernel,
because ``tl.dot`` is the only idiomatic Triton primitive that emits
``_swap`` today (empirically verified during authoring — a plain
``tl.max`` reduction at any tile size lowers to ``permlanex16`` in
Triton 3.7).

How it forces the instruction
=============================

One program per row, ``ROW = 64`` fp32 columns, one wave per program.
On gfx1250 (wave32) this is 32 lanes holding 2 elements each: intra-
lane pair-reduce first, then a 32-lane reduction tree.  Triton's
lowering of that tree walks ``v_mov_b32_dpp`` for distances 1, 2, 4,
8 and then emits one ``v_permlanex16_b32`` for the final distance-16
step.  On gfx942 (wave64) the entire reduction fits inside a single
wave's DPP range, so no permlane is needed and the native gold is
pure-DPP — that asymmetry is what makes this a salmon-testing canary
(the gfx1250 path has a lowering that gfx942 does not emit directly;
salmon has to re-lower it).

``tl.max`` is chosen over ``tl.sum`` because max is exact and
order-independent at every precision — there is no reduction-order
noise across wave sizes, so any disagreement is a real miscompile
rather than floating-point drift.  That lets us keep the tolerance
at zero and catch even single-bit errors.

Harness schema notes
====================

Swept shape dim is ``N_ROWS``; ``ROW`` is a constexpr so Triton can
fold the row extent into the lowering.  The input range ``[-10, 10]``
is wider than the other canaries on purpose: a broken permlane lift
that collapses the reduction to a single lane's value would produce
an output on the order of one element, whereas the true max of 64
iid uniform draws from ``[-10, 10]`` clusters near ``+10`` — the gap
between the two is >> tolerance, so the regression signal is loud.
"""
import triton
import triton.language as tl


@triton.jit
def canary_permlanex16_rowmax_kernel(
    x_ptr, y_ptr,
    N_ROWS,
    ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, ROW)
    x = tl.load(x_ptr + pid * ROW + offs)
    y = tl.max(x, axis=0)
    tl.store(y_ptr + pid, y)


RECIPES = [
    {
        "name": "canary_permlanex16_rowmax_fp32",
        "kernel_fn": canary_permlanex16_rowmax_kernel,
        "kernel_symbol": "canary_permlanex16_rowmax_kernel",
        "signature": {
            "x_ptr":  "*fp32",
            "y_ptr":  "*fp32",
            "N_ROWS": "i32",
        },
        "constexprs": {
            "ROW": 64,
        },
        # num_warps = 1 gives us one wave per program, which is what
        # makes the "gfx1250 needs permlanex16, gfx942 needs only DPP"
        # asymmetry clean.  More warps would spread the row over
        # multiple waves and drag LDS into the lowering — a different
        # C-class (not the one this canary is pinning down).
        "num_warps": 1,
        "shape_dim": "N_ROWS",
        "default_shapes": [32, 128, 1024, 8192],
        "grid": {
            "x": "N_ROWS",
            "y": "1",
            "z": "1",
        },
        "inputs": [
            {"name": "x_ptr", "dtype": "fp32", "elems": "N_ROWS * ROW",
             "range_lo": -10.0, "range_hi": 10.0},
        ],
        "outputs": [
            {"name": "y_ptr", "dtype": "fp32", "elems": "N_ROWS"},
        ],
        # max is exact and order-independent across any correct
        # reduction tree; a tolerance of zero is the right default.
        # A broken permlane lift shows up as the output being one
        # lane's value instead of the row max — differences on the
        # order of whole elements, dwarfing any tolerance we'd pick.
        "comparator": {"kind": "abs", "tol": 0.0},
    }
]
