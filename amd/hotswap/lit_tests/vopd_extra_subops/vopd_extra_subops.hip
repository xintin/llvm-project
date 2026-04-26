// ARCH: gfx1250
//
// Lit fixture for the VOPD sub-operations and operand shapes that
// were unhandled before the supporting handler change in
// transpiler/handle_vopd.cpp:
//
//   1. v_dual_mov_b32 X-half with an f32 inline-constant literal
//      ("1.0", "0.5", etc.) — `readVOPDSrc` previously only parsed
//      integer literals through `StringRef::getAsInteger`, so any
//      kernel where the AMDGPU printer surfaced an inline-constant
//      pool entry as its decimal float spelling was rejected at
//      VOPD decomposition time.  Triton's
//      `attn_fwd_pipelined_kernel` GEMM hits this exact pattern.
//   2. v_dual_ashrrev_i32 Y-half — used by Triton scoring/masked-
//      compaction kernels for sign-extending the high bit of a
//      32-bit lane index for branch-free clamp/select sequences.
//   3. v_dual_max_i32 Y-half — used by Triton scaling kernels
//      (`upcast_from_mxfp`) to clamp a signed exponent against zero.
//   4. v_dual_mov_b32 with a `ttmp<N>` source — Triton-emitted SPE
//      preludes use `v_dual_mov_b32 vX, ttmp9` to copy the
//      workgroup-id-X value the raiser seeded into `regs.ttmp[9]`
//      into a VGPR for downstream per-lane use.  The X-half of a
//      canary (`canary_permlanex16_rowmax_fp32`) and Triton's
//      softmax kernel both emit this pattern.  Before the
//      supporting handler change, `readVOPDSrc` rejected
//      `ttmp9` at parse time (doesn't start with `v`, doesn't start
//      with `s`, doesn't parse as a literal) and `parseVOPDHalf`
//      returned false, aborting the whole kernel's raise with
//      `VOPD decomposition failed`.
//   5. v_dual_mov_b32 with a `vcc_lo` source — ~8 sites across the
//      current corpus use it as the wave-mask bitmap in SPE
//      fixtures.  Routes through `readVCCAsWaveMask` so the
//      wave-projection layer sees a principled VCC-as-scalar read
//      (which the downstream writelane/readlane-rewrite safety net
//      in rewrite_cross_lane_divergent.cpp can then classify
//      against the §5.6.3 use-chain rules).
//
// Inline asm is the cleanest way to pin all three shapes in a
// single fixture.  We use a paired VOPD instruction per asm block
// so the assembler emits a real VOPD encoding (rather than two
// single-issue VOPs the auto-pairer might or might not bundle).
// The kernel below pins one pair per shape:
//
//   * Pair A: `v_dual_mov_b32 vX, 1.0 :: v_dual_ashrrev_i32 vY,
//     31, vZ`  — covers the float-literal MOV and the AShr.
//   * Pair B: `v_dual_mov_b32 vX, v? :: v_dual_max_i32 vY, 0,
//     vZ`     — covers the smax intrinsic dispatch.
//   * Pair C: `v_dual_mov_b32 vX, ttmp9 :: v_dual_mov_b32 vY, s0`
//     — covers the TTMP-source branch (plus an s<N> SGPR partner to
//     exercise the pair-halves-disagree register-class combo that
//     the actual Triton kernels emit).  The SGPR Y-half is the
//     `s0` case we already handle via the existing SGPR branch;
//     pairing it with the TTMP X-half is deliberate — it matches
//     the shape Triton emits and ensures both halves can succeed
//     within a single VOPD pair.
//   * Pair D: `v_dual_mov_b32 vX, vcc_lo :: v_dual_mov_b32 vY, v?`
//     — covers the VCC-as-scalar source branch.
//
// All branches share the same `parseVOPDHalf` skeleton; if any one
// of the float-literal / AShr / smax / TTMP / VCC handler paths
// regresses, exactly one of the corresponding `lift` checks below
// will fail.

#include <hip/hip_runtime.h>

extern "C" __global__ void
vopd_extra_subops_kernel(int *in_ptr, int *out_ptr) {
  int v_in = in_ptr[threadIdx.x];

  unsigned int mov_lit = 0;
  int ashr_out = 0;
  asm volatile(
      "v_dual_mov_b32 %[d_mov], 1.0 :: v_dual_ashrrev_i32 "
      "%[d_ashr], 31, %[s]"
      : [d_mov] "=v"(mov_lit), [d_ashr] "=v"(ashr_out)
      : [s] "v"(v_in));

  int v_a = in_ptr[(threadIdx.x + 1) & 31];
  int v_b = in_ptr[(threadIdx.x + 2) & 31];
  int max_out = 0;
  int mov_reg = 0;
  asm volatile(
      "v_dual_mov_b32 %[d_mov], %[s_mov] :: v_dual_max_i32 "
      "%[d_max], 0, %[s_max]"
      : [d_mov] "=v"(mov_reg), [d_max] "=v"(max_out)
      : [s_mov] "v"(v_a), [s_max] "v"(v_b));

  // Pair C: `v_dual_mov_b32 vX, ttmp9 :: v_dual_mov_b32 vY, s0`.
  // `ttmp9` is seeded with the workgroup-id-X by the SPE prelude
  // (see raiser.cpp Phase 4); `s0` is the kernarg pointer-low
  // register in the standard SGPR user ABI.  Reading both through
  // a VOPD pair pins the TTMP source branch (new) alongside the
  // SGPR branch (pre-existing) — if either path regresses, exactly
  // one CHECK below will fail.
  //
  // The destinations are stored back through normal VGPRs; we
  // funnel both halves into the output buffer so the compiler
  // can't drop them as dead.
  int ttmp_out = 0;
  int sgpr_out = 0;
  asm volatile(
      "v_dual_mov_b32 %[d_ttmp], ttmp9 :: v_dual_mov_b32 "
      "%[d_sgpr], s0"
      : [d_ttmp] "=v"(ttmp_out), [d_sgpr] "=v"(sgpr_out));

  // Pair D: `v_dual_mov_b32 vX, vcc_lo :: v_dual_mov_b32 vY, v?`.
  // We set VCC up via a scalar compare so the VCC-source MOV reads
  // a known wave-mask rather than an undefined register.  On wave32
  // `vcc_lo` IS the whole VCC bitmask; we route through
  // `readVCCAsWaveMask` in the handler so the wave-projection
  // layer classifies it properly rather than width-slicing a
  // VCC.hi-less assumption.
  unsigned int vcc_mask = 0;
  int mov_reg2 = 0;
  int v_c = in_ptr[(threadIdx.x + 3) & 31];
  asm volatile(
      "v_cmp_eq_u32_e64 vcc_lo, %[s], 0\n\t"
      "v_dual_mov_b32 %[d_vcc], vcc_lo :: v_dual_mov_b32 "
      "%[d_mov], %[s_mov]"
      : [d_vcc] "=v"(vcc_mask), [d_mov] "=v"(mov_reg2)
      : [s] "v"(v_in), [s_mov] "v"(v_c)
      : "vcc_lo");

  out_ptr[threadIdx.x * 8 + 0] = (int)mov_lit;
  out_ptr[threadIdx.x * 8 + 1] = ashr_out;
  out_ptr[threadIdx.x * 8 + 2] = max_out;
  out_ptr[threadIdx.x * 8 + 3] = mov_reg;
  out_ptr[threadIdx.x * 8 + 4] = ttmp_out;
  out_ptr[threadIdx.x * 8 + 5] = sgpr_out;
  out_ptr[threadIdx.x * 8 + 6] = (int)vcc_mask;
  out_ptr[threadIdx.x * 8 + 7] = mov_reg2;
}
