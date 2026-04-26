"""Canary for DPP-lane-shuffle emission on wave32 (C2-DPP).

What it targets
===============

The P5 item in ``wave-size-translation.md §5.3`` — the
``update_dpp`` intrinsic lift.  ``raiser.cpp`` currently canonicalises
away the ``_dpp`` modifier on lifted instructions without re-emitting
it as ``llvm.amdgcn.update.dpp`` (see ``gpt-oss-derisking.md §7.3``
for the open-risk write-up).  This is the single largest outstanding
correctness risk in the GPT-OSS corpus: 5 of 15 kernels today are
silently miscompiled on DPP (``_bitmatrix_metadata_compute_stage{1,2}``,
``_downcast_to_mxfp``, ``_sum_bitmatrix_rows``, ``_topk_forward``).

The repro those GPT-OSS kernels would hit at scale is a small-width
cross-lane reduction: Triton's reduction lowering on gfx1250 uses
``v_mov_b32_dpp`` / ``v_add_f32_dpp`` chains for any reduction whose
axis width is ``<= 16`` — narrower than a wave, wider than a single
lane, and therefore in the DPP sweet spot.  The wider reductions
(≥ 32 lanes on wave32) already exercise ``v_permlane16_swap`` via
canary_permlane_rowmax; this recipe deliberately stays in the DPP
regime so the two canaries pin down distinct lowerings.

How it forces the instruction
=============================

Load a 1D tile of ``BLOCK_SIZE`` fp32 elements, reshape it to
``(BLOCK_SIZE // GROUP, GROUP)``, and reduce along the inner GROUP
axis.  With ``GROUP = 8`` and ``BLOCK_SIZE = 128`` on gfx1250,
Triton emits a 3-stage DPP reduction tree (8 -> 4 -> 2 -> 1) using
``v_mov_b32_dpp`` + ``v_add_f32`` pairs.  On gfx942 the same shape
lowers to within-wave reductions that do not need any DPP lift —
again, the point: the gfx942 native run is the gold and salmon's job
is to re-lower the gfx1250 DPP path into whatever gfx942 wants.

Harness schema notes
====================

Single swept shape dim ``N`` (total elements).  ``GROUP`` is a
constexpr so Triton can fold the reduction tree at compile time;
``BLOCK_SIZE`` is the per-program tile.  Both must be powers of two,
and ``N`` must be a multiple of ``BLOCK_SIZE`` so there is never a
ragged tile — we deliberately do not want masked-DPP code here, the
dense path is what GPT-OSS's DPP use-sites hit.  The output buffer
is ``N / GROUP`` elements because each program emits
``BLOCK_SIZE / GROUP`` partial sums.

fp32 inputs are drawn from ``[-1, 1]``, summed over 8 elements, so
the output stays in ``[-8, 8]``.  Two correct implementations that
sum in different orders produce results that differ by at most a
few ULPs — ``rel`` with ``tol = 1e-5`` accepts that ordering drift
but loudly catches the "all lanes read their own value" failure
mode characteristic of a broken DPP lift (which collapses the sum
to a single-element passthrough).
"""
import triton
import triton.language as tl


@triton.jit
def canary_dpp_reduce_kernel(
    x_ptr, y_ptr,
    N,
    GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    x = tl.load(x_ptr + block_start + offs)
    x2 = tl.reshape(x, (BLOCK_SIZE // GROUP, GROUP))
    y = tl.sum(x2, axis=1)
    out_len: tl.constexpr = BLOCK_SIZE // GROUP
    out_offs = tl.arange(0, out_len)
    out_start = pid * out_len
    tl.store(y_ptr + out_start + out_offs, y)


RECIPES = [
    {
        "name": "canary_dpp_reduce_fp32",
        "kernel_fn": canary_dpp_reduce_kernel,
        "kernel_symbol": "canary_dpp_reduce_kernel",
        "signature": {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "N":     "i32",
        },
        "constexprs": {
            "GROUP": 8,
            "BLOCK_SIZE": 128,
        },
        "num_warps": 1,
        "shape_dim": "N",
        # All multiples of BLOCK_SIZE = 128, so the kernel never hits
        # its trailing-mask path and the ONLY work it does is the
        # small-group reduction that we care about.
        "default_shapes": [128, 1024, 8192, 65536],
        "grid": {
            "x": "ceil_div(N, BLOCK_SIZE)",
            "y": "1",
            "z": "1",
        },
        "inputs": [
            {"name": "x_ptr", "dtype": "fp32", "elems": "N",
             "range_lo": -1.0, "range_hi": 1.0},
        ],
        "outputs": [
            {"name": "y_ptr", "dtype": "fp32", "elems": "N / GROUP"},
        ],
        # GROUP=8 fp32 partial sums in [-1,1] tree-reduced over two
        # correct lowerings differ by at most a few ULPs.  Tolerance
        # is set to catch the DPP-lift regression (output = input[0])
        # which dwarfs this by orders of magnitude.
        "comparator": {"kind": "rel", "tol": 1e-5},
    }
]
