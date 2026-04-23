// Post-PromoteMemToReg pass that rewrites the Triton-on-gfx1250
// `v_permlane16_swap_b32` emission when BOTH VGPR operands are
// initialised from the same seed SSA value.
//
// STATUS: TRANSITIONAL (2026-04-23).  Remove when either of the
// two conditions below is met — identical to the sibling
// `rewrite_permlane16_xor3_partner` pass, documented here once
// to keep the two TRANSITIONAL bridges' removal criteria
// together.
//
//   (a) AMD publishes gfx1250 ISA documentation disambiguating
//       the silicon semantic of `v_permlane16_swap_b32` when
//       `vdst_in` and `src0_in` hold identical per-lane values.
//       If the silicon's second output (`src0_out`) is NOT the
//       partner-read of `vdst_in` (which would be the
//       gfx950-documented symmetric cross-wire) but some other
//       value — e.g. `src0_in` unchanged, or `vdst_in`
//       unchanged — move the fix UP the stack: update
//       `emitPermLaneSwapEmulation` in
//       `handle_valu_cross_lane.cpp` to emit the gfx1250-correct
//       semantic (guarded by a source-ISA check).  This pass
//       then becomes dead code (harmless — the fingerprint
//       still matches, substituting an already-correct seed for
//       itself).
//
//   (b) Triton's gfx1250 codegen changes to stop emitting
//       `v_permlane16_swap_b32` with `v_dual_mov`-initialised
//       same-seed operands (e.g. switches to `ds_swizzle_b32
//       swap:16`, the single-output gfx942 analogue).  The
//       fingerprint stops appearing in lifted IR and the pass
//       becomes dead code.
//
// Under either condition we also delete the `--enable-/--disable-
// permlane16-swap-selfpreserve` raise_cli flag and the
// `enablePermLane16SwapSelfPreserveRewrite` parameter threaded
// through `raiser.hpp` + `pipeline.hpp`.
//
// Regression pins:
//   * `canary_tl_sort_fp32` / `canary_tl_topk_fp32` /
//     `canary_tl_topk_bf16` / `canary_tl_topk_bf16_nw1` /
//     `topk_forward_bf16` / `topk_forward_bisect_m2_strict` —
//     all 6 recipes graduate from WRONG to match under this
//     rewrite.  Removing it regresses them.
//   * `canary_tl_sort_fp32_deterministic` — continues to match
//     (this rewrite subsumes `rewrite_permlane16_xor3_partner`
//     for that shape; both fire, the later one is a no-op).
//   * `Gfx1250Gpu.Permlane16Swap{,Wave32,Wave32WaveNative}` and
//     `Gfx1250Gpu.BitonicCross16Probe` — pin that the swap's
//     standalone semantic is unaffected when `vdst_in` and
//     `src0_in` are DIFFERENT SSA values.  The fingerprint only
//     matches when both trace (via SPE active-arm phi walks) to
//     the same root SSA value; the probes use distinct inputs
//     (`vdst_in[L] = L`, `src0_in[L] = 1000 + L`) and don't
//     trigger the rewrite.
//
// Background
// ==========
//
// Triton's `tl.sort(dim=1, descending=True)` and `tl.topk`
// codegen for gfx1250, at the cross-16 stage of a bitonic
// merge, emits a 3-instruction sequence that initialises BOTH
// operands of `v_permlane16_swap_b32` from the same seed VGPR:
//
//   v_dual_mov_b32 v_a, v_c :: v_dual_mov_b32 v_b, v_c
//   v_permlane16_swap_b32 v_a, v_b
//   <composition on v_a, v_b, v_c>
//
// The composition varies by shape (encountered in the corpus):
//
//   tl.sort(deterministic-input):
//     v_xor3_b32 v_a, v_a, v_b, v_c
//   tl.sort(random-input):
//     v_xor_b32 v_a, v_b, v_a    ; inner xor
//     v_xor_b32 v_a, v_a, v_c    ; outer xor (= xor3 split)
//   tl.topk (max reduction):
//     v_max_num_f32 v_d, v_b, v_b
//     v_max_num_f32 v_a, v_a, v_a
//     v_max_num_f32 v_a, v_a, v_d
//
// Under gfx950-documented SYMMETRIC cross-wire semantics
//   new_vdst[L]      = src0_in[L XOR 16]
//   new_src0_out[L]  = vdst_in[L XOR 16]
// and the Triton initialiser `v_a_in == v_b_in == v_c`, BOTH
// outputs end up holding `partner_v_c`.  The downstream
// composition then has TWO copies of `partner_v_c` plus `v_c`
// (self) as its only inputs — no access to the "per-pair
// self-value" the algorithm wants.  Every composition above
// degenerates:
//
//   xor3(partner, partner, self)         = self   (want partner)
//   xor(xor(partner, partner), self)     = self   (want partner)
//   max(max(partner, partner),
//       max(partner, partner))           = partner (want max(self, partner))
//
// None produce the algorithm-expected value, so all 6 recipes
// miscompile.
//
// The pre-existing sibling rewrite (`rewrite_permlane16_xor3
// _partner`) catches the first composition (fused `v_xor3_b32`)
// by walking the outer xor's structure.  It does NOT catch the
// split-xor or max-based shapes because the inner expression
// isn't a single `xor3` BinaryOperator.  This pass fixes all
// three (and any future Triton composition around the same
// same-seed idiom) at the ROOT — the `v_permlane16_swap_b32`
// emission itself — instead of pattern-matching each
// composition downstream.
//
// Fix shape
// ---------
//
// `emitPermLaneSwapEmulation` emits the two bpermute calls
// together:
//
//   %pls16_addr         = ...           ; (lane_id XOR 16) << 2
//   %pls16_new_vdst     = call bpermute(%pls16_addr, %src0_in)
//   %pls16_new_src0_out = call bpermute(%pls16_addr, %vdst_in)
//   ... SPE wrappers around each write ...
//
// This pass finds every matching pair of adjacent bpermute
// calls where:
//   1. Both calls share the same address SSA value.
//   2. Their data arguments (`src0_in` and `vdst_in`), walked
//      backward through SPE `spe_do` active-arm phi edges,
//      reach the same root SSA value.
//
// When the pair matches, RAUWs the `%pls16_new_src0_out`
// call's result with the shared seed SSA value.  The downstream
// SPE phi now has shape:
//
//   %vgpr<src0_reg>.N = phi [ %seed, %spe_doN ],
//                           [ %vgpr<src0_reg>.N-1, %spe_skipN-1 ]
//
// so the active lanes see `seed` (self) and inactive lanes
// preserve — the asymmetric-output semantic.  `new_vdst`
// continues to carry `partner_seed` through the untouched first
// bpermute call.  The downstream composition now has BOTH
// `seed` and `partner_seed` available in distinct SSA values
// and every idiom above produces the algorithm-expected output:
//
//   xor3(partner, self, self)                    = partner ✓
//   xor(xor(partner, self), self)                = partner ✓
//   max(max(self, self), max(partner, partner))  = max(self, partner) ✓
//
// The unused dead `bpermute(addr, vdst_in)` call is left in
// place (it's `IntrConvergent`; DCE may or may not remove it,
// but an unused LDS round-trip is harmless).
//
// Narrow-fingerprint argument
// ---------------------------
//
// The pair structure + shared-address + shared-data-root
// conjunction is a fingerprint salmon emits ONLY from
// `emitPermLaneSwapEmulation` when `vdstReg`'s and
// `src0Reg`'s last stored value was the same SSA (the Triton
// `v_dual_mov v_a, v_c :: v_dual_mov v_b, v_c` initialiser).
// No other salmon path emits two bpermute calls with matching
// address AND matching data-root.  Non-Triton kernels that use
// `v_permlane16_swap_b32` with distinct inputs per operand
// (the P4 probe kernels, the AITER kernels' permlane32 siblings
// through a different opcode) don't trigger it.
//
// Composition with the xor3-partner rewrite
// -----------------------------------------
//
// When both rewrites are enabled, this pass runs FIRST (it
// operates on a broader / earlier shape) and the xor3-partner
// rewrite either becomes a no-op on sites this pass handled
// (both substitute `partner_seed` into the final result — no
// change if this one already did) or fires independently on
// xor3 sites where the data-root equivalence check fails (the
// pre-existing pass's fingerprint is phrased on the outer xor,
// not on the two bpermute inputs, so it's possible but unlikely
// for the two to disagree).  Disabling either pass audits the
// pre-rewrite shape of the sites that pass alone would catch.

#pragma once

#include "llvm/IR/Function.h"

namespace transpiler {

struct Permlane16SwapSelfPreserveRewriteReport {
  unsigned matchedSites = 0;
};

Permlane16SwapSelfPreserveRewriteReport
rewritePermLane16SwapSelfPreserve(llvm::Function &F);

} // namespace transpiler
