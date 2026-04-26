"""Canary for `ds_bpermute_b32` emission on wave32 (C2-bpermute).

What it targets
===============

The C2-bpermute path through the raiser (the P1 rewrite in
``wave-size-translation.md §5.3``, landed in commit ``d9bfd99626``).
``handle_ds.cpp`` already lifts ``ds_bpermute_b32`` through
``llvm.amdgcn.ds.bpermute``; this canary runs that handler end-to-end
through the cross-widening pipeline and checks for numerical parity
with the gold.

Salmon verdict evolution
========================

The canary's verdict has shifted through three principled states
as the wave-size machinery hardened:

1. Pre-VOPD-fix (through commit ``bd04c268e7``): **WRONG on every
   shape**.  The residual miscompile was an interaction between
   modulo-replication and Kogge-Stone scan predication — Triton's
   cumsum lowering computes its ``if lane_id >= 2^s`` scan guards
   on ``workitem.id.x()`` rather than the source-wave ``mbcnt``
   lane id.  Under cross-widening the upper-half lanes'
   ``workitem.id.x()`` values fell outside what the MODREP "each
   replica reproduces the source wave" model captured.

2. Post-VOPD-fix (commit ``bd04c268e7`` — "fix VOPD
   v_dual_cndmask_b32 SGPR-condition handling"):
   ``salmon match`` on every shape.  The fix re-shaped the
   compiler's lowering so the ``tid``-based predicate is
   materialised through ``ballot.i64`` + ``s_and_saveexec_b32``
   before the cross-lane ``ds_bpermute`` — a chain
   ``rewrite_cross_lane_divergent.cpp`` already traces as
   wave-aware.  The "match" was numerically correct, but the
   phantom-lane contract this kernel runs under (``num_warps=1``
   on wave32 source → ``max_flat_workgroup_size = 32`` in the
   emitted HSACO → under-filled wave64 on the gfx942 target)
   leaves the ``WaveNativeProjection`` soundness argument
   unprovable — the pass was a numerical coincidence of the
   specific Triton-emitted scan shape, not a universal guarantee.

3. Post-phantom-lane-rule: **EXIT=2** on every shape, principled
   refusal via ``c5_predicate_chain_classifier.cpp``'s phantom-
   lane arm.  The rule (added alongside
   ``canary_bitmatrix_composite`` which empirically falsified
   the unconditional WaveNative suppression) narrows the
   classifier's ``waveNative`` arm to also refuse when
   ``max_flat_workgroup_size < targetWaveSize``.  The
   wave32-source cumsum shape is in that regime, so the
   numerical-coincidence pass graduates to a principled refusal.
   The refusal diagnostic names both the phantom-lane regime
   (``max_flat_workgroup_size=32 < target wavefront width 64``)
   and the C5 shape (``icmp ugt against compile-time constant K
   within (0, W_s-1=31]``).

Regression contract: the EXIT=2 state is the current principled
verdict.  A drift back to ``match`` or ``WRONG`` is informative:

* ``EXIT=2`` -> ``match``  ==  somebody added a handler that
  genuinely models lane positions across phantom target lanes
  (e.g. a ``max_flat_workgroup_size``-widening pass that
  promotes the HSACO's WG bound to a multiple of the target
  wavefront width).  Retire the EXIT=2 expectation; document.
* ``EXIT=2`` -> ``WRONG``  ==  the phantom-lane arm regressed.
  Investigate ``c5_predicate_chain_classifier.cpp``'s
  ``phantomLaneGuaranteed`` and the ``maxFlatWorkgroupSize``
  thread-through in ``raiser.cpp``.

Keeping this canary as an ongoing signal for further scan-shape
investigation — the principled refusal is the correct outcome
for the current handler surface, but a future rewrite that
widens the safety domain (beyond the raise-time refusal) would
let the kernel transition back to ``match`` on a principled
basis.

How it forces the instruction
=============================

Triton lowers ``tl.cumsum`` (prefix sum, a.k.a. scan) on a wave-
sized axis via a Kogge-Stone-style tree whose inter-lane exchange
steps are ``ds_bpermute_b32`` on gfx1250 wave32.  Empirically, a
``tl.cumsum`` over an axis of length 32 emits 5 ``ds_bpermute_b32``
(one per scan stage: distances 1, 2, 4, 8, 16); length 128 emits 20
(four scan passes of 5 stages each).  This is a much more reliable
trigger than ``tl.trans`` or a permuted load — both of those get
optimised into address arithmetic instead of a cross-lane move, which
silently sidesteps the bpermute path.

Scan was empirically selected over a handful of other candidates
(``tl.trans``, ``tl.flip``, ``tl.reduce`` with a custom combinator,
``tl.dot``): only ``tl.cumsum`` and ``tl.dot`` emit bpermute in
Triton 3.7, and ``tl.cumsum`` is the simpler of the two because it
does not drag in the WMMA / MFMA intrinsic selection path.

``BLOCK_SIZE = 128`` is chosen because it sweeps across the wave
boundary (wave32 holds 32 lanes of 4 elements each; wave64 holds
64 lanes of 2 elements each) — so the scan on gfx1250 uses cross-
lane ``ds_bpermute_b32`` while on gfx942 it can stay within a single
wave using DPP.  That asymmetry is the point: the gfx1250 build
exercises exactly the lowering salmon has to translate.

Harness schema notes
====================

Single swept shape dim ``N`` (total elements).  ``N`` must be a
multiple of ``BLOCK_SIZE`` so no program hits a ragged trailing tile
— the scan is per-program (each tile's prefix sums stand alone), so
masking the scan input with zeros would change the answer and the
CPU-unreadable gfx942 gold would diverge from what the kernel
computes for the same shape.

``rel-rms`` comparator with ``tol = 1e-3``: the scan's reduction
tree on wave32 (Kogge-Stone across 32 lanes) is a different shape
to the wave64 tree (Kogge-Stone across 64 lanes), so floating-point
associativity gives slightly different per-element results.  The
RMS norm stays very small (``< 1e-5`` for fp32 inputs in ``[-1,1]``
scanned across 128 elements) under any correct lowering, so the
tolerance is three orders of magnitude below the expected noise
floor and more than five orders below what the current cross-
widening bug produces (``max|err|`` in the range 4.2 → 21.9,
climbing roughly as ``sqrt(N/BLOCK_SIZE)`` — consistent with
partial-scan-tree drift rather than a total collapse).
"""
import triton
import triton.language as tl


@triton.jit
def canary_bpermute_scan_kernel(
    x_ptr, y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    x = tl.load(x_ptr + block_start + offs)
    y = tl.cumsum(x, axis=0)
    tl.store(y_ptr + block_start + offs, y)


RECIPES = [
    {
        "name": "canary_bpermute_scan_fp32",
        "kernel_fn": canary_bpermute_scan_kernel,
        "kernel_symbol": "canary_bpermute_scan_kernel",
        "signature": {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "N":     "i32",
        },
        "constexprs": {
            "BLOCK_SIZE": 128,
        },
        "num_warps": 1,
        "shape_dim": "N",
        # Multiples of BLOCK_SIZE = 128 so the kernel never hits a
        # ragged tile.  Masking a scan's input would change the
        # mathematical answer, not just the out-of-bounds reads, so
        # "no ragged" here is a load-bearing constraint rather than
        # a convenience.
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
            {"name": "y_ptr", "dtype": "fp32", "elems": "N"},
        ],
        # Scan-tree shape differs between wave32 and wave64, so
        # pointwise fp32 results differ by a handful of ULPs in the
        # late elements of each per-program block.  RMS norm stays
        # well below 1e-5 under any correct lowering; 1e-3 gives
        # headroom against the expected ordering drift while still
        # catching the bpermute-miscompile failure mode (every lane
        # reading its own value), which produces RMS error
        # proportional to the per-block partial sum magnitude.
        "comparator": {"kind": "rel-rms", "tol": 1e-3},
    }
]
