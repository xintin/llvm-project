"""Canary for compound-DPP (`v_add_f32_dpp`) emission on wave32 (C2-DPP).

What it targets
===============

Companion to ``canary_dpp_reduce_fp32``, covering the *other* DPP
opcode family that GPT-OSS reaches.  The opcode split matters at the
raiser boundary:

- ``canary_dpp_reduce_fp32``   forces ``v_mov_b32_dpp`` (move-only,
  the DPP-modifier goes to the move itself).  Empirically passes
  under salmon at every shape tried.
- ``canary_dpp_compound_add_fp32`` (this one) forces
  ``v_add_f32_dpp`` — a compound DPP where the add *and* the DPP
  modifier live on the same opcode.  This is the closest Triton
  analog to the ``v_add_nc_u32_dpp`` pattern in GPT-OSS's
  ``_bitmatrix_metadata_compute_stage1`` and represents the second
  of the two main DPP handler paths the raiser has to get right.

``gpt-oss-derisking.md §7.3`` calls P5 (the DPP intrinsic lift) the
largest outstanding correctness risk.  The §5.3 text is based on the
assumption that the raiser canonicalises the DPP modifier away.  If
this canary also passes, the real scope of P5 is narrower than
"every DPP silently miscompiles" — it's specifically the patterns
not hit by either this canary or the ``v_mov_b32_dpp`` one.

Note on wave-dependent DPP modifiers
====================================

Pre-authoring probe established that gfx1250 (RDNA4) emits **no**
wave-size-dependent DPP modifiers: ``bcast15`` / ``bcast31`` /
``wave_shl`` / ``wave_shr`` / ``row_share`` were gfx9/10 patterns
and are not used by the gfx12 lowering path.  All DPP modifiers
Triton emits on gfx1250 (``row_shl:n`` / ``row_shr:n`` / ``row_xmask``
/ ``quad_perm`` / ``dpp8``) operate within 16-lane rows or 4-lane
quads and are wave-invariant *by ISA construction*.  A dedicated
``canary_dpp_bcast15`` is therefore not possible on gfx1250 — there
is nothing to canary against.  The raiser work for P5 only has to
correctly lift the wave-invariant DPP opcodes this kernel and
``canary_dpp_reduce_fp32`` emit.

How it forces the instruction
=============================

One program per row, ``ROW = 32`` fp32 columns, one wave per program.
Each row's mean is computed via ``tl.sum`` and subtracted from every
element: a realistic "center the row around zero" prep step used by
layernorm-family kernels.  The reduction's tree stages use
``v_add_f32_dpp`` (4 per kernel, empirically confirmed via probe);
the final distance-16 exchange uses ``v_permlanex16_b32`` (1 per
kernel, inherited from the full-wave reduction).

The permlanex16 overlap means a regression on the ``permlanex16``
path would also make this canary fail — but
``canary_permlanex16_rowmax_fp32`` covers that primitive directly
and more cheaply (single instruction emission, no compound DPP to
compete), so in practice the attribution question when this canary
fails is "does the permlanex16 canary also fail?" — if yes, it's
permlanex16's fault; if no, it's the compound-DPP path.

Harness schema notes
====================

Swept shape dim is ``N_ROWS``.  ``ROW = 32`` matches ``N_ROWS * ROW``
elements fitting cleanly at every sweep size.  Input range
``[-1, 1]`` keeps the per-row sum bounded (no overflow risk at
``ROW = 32``) and makes the mean well-conditioned — a broken DPP
lift shows up as the output equalling the input (subtraction of
zero instead of the mean), which for input range ``[-1, 1]``
differs from the correct answer by the input's magnitude (well
above any sane tolerance).
"""
import triton
import triton.language as tl


@triton.jit
def canary_dpp_compound_add_kernel(
    x_ptr, y_ptr,
    N_ROWS,
    ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, ROW)
    x = tl.load(x_ptr + pid * ROW + offs)
    s = tl.sum(x, axis=0)
    mean = s / ROW
    y = x - mean
    tl.store(y_ptr + pid * ROW + offs, y)


RECIPES = [
    {
        "name": "canary_dpp_compound_add_fp32",
        "kernel_fn": canary_dpp_compound_add_kernel,
        "kernel_symbol": "canary_dpp_compound_add_kernel",
        "signature": {
            "x_ptr":  "*fp32",
            "y_ptr":  "*fp32",
            "N_ROWS": "i32",
        },
        "constexprs": {
            "ROW": 32,
        },
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
             "range_lo": -1.0, "range_hi": 1.0},
        ],
        "outputs": [
            {"name": "y_ptr", "dtype": "fp32", "elems": "N_ROWS * ROW"},
        ],
        # Per-row mean subtract: gold and candidate both compute the
        # same reduction in the same order up to DPP-tree shape
        # differences.  A few ULPs is the realistic noise floor for
        # fp32 sums over 32 elements in [-1, 1].  A regression (DPP
        # drop) shows up as every element being off by the full row
        # mean — orders of magnitude above tolerance, so the
        # regression signal is loud even with headroom.
        "comparator": {"kind": "rel-rms", "tol": 1e-4},
    }
]
