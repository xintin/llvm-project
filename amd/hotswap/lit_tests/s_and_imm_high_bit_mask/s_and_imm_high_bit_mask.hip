// ARCH: gfx1250
//
// Regression fence for `readOpExecWidth`'s immediate-operand path on
// wave32 sources when the literal's high bit is set (bit pattern
// `>= 0x80000000`).
//
// Pattern exercised: a SOP2 wave-mask algebra with an immediate
// source whose bit pattern is not representable as a signed i32 —
// `s_and_b32 sN, sM, 0xFFFF0000` (Triton's "high-half upper short
// extraction" idiom, seen in `_bitmatrix_metadata_compute_stage2`).
// The topk commit `96cbb53e20` routed non-register SOP2 sources
// through `tryGetSrcWaveMaskI1` → `srcExecWidth` → `readOpExecWidth`
// so shadow propagation survives the immediate step; that path
// then tripped APInt's signed-range assertion on the
// `ConstantInt::getSigned(i32, 0xFFFF0000)` call, because the
// `int64_t` container decodes the MC literal as `+4294901760`
// (outside the signed i32 range `[-2^31, 2^31 - 1]`).
//
// The fix treats the raw `int64_t` as an unsigned bit pattern —
// mirroring `readOp32`'s existing contract — so any 32-bit
// literal decodes into an LLVM i32 constant regardless of its
// high-bit state.
//
// The companion `.ll` asserts:
//   * the lift succeeds (no signal_6 crash);
//   * the immediate materialises as the signed-equivalent i32
//     constant `-65536` (LLVM's canonical print of `0xFFFF0000`),
//     confirming we reached the fixed immediate-path rather than
//     some alternative routing.
//
// Distinct from `s_xor_imm_mask_shadow`: that fixture uses literal
// `-1` = `0xFFFFFFFF`, which MC decodes as `int64_t(-1)` (inside
// the signed i32 range) — so it does NOT exercise the bit-pattern
// failure this test pins. This is the minimal fixture that does.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_and_imm_high_bit_mask_kernel(int *out,
                                                           const float *in) {
  int lane = (int)threadIdx.x;
  float val = in[lane];
  int result = 0;
  // Produce a wave-mask i1 (`v_cmp_*`) into s2, AND it with a
  // high-bit-set literal, and consume the SGPR via v_cndmask_b32_e64
  // — the full `tryGetSrcWaveMaskI1` path over the immediate
  // source must materialise `ConstantInt::get(i32, 0xFFFF0000)`
  // without tripping APInt's signed-range assertion.
  asm volatile(
      "v_cmp_ge_f32_e64 s2, |%[v]|, 0.5\n\t"
      "s_and_b32 s2, s2, 0xFFFF0000\n\t"
      "v_cndmask_b32_e64 %[r], 0, 1, s2\n\t"
      : [r] "=v"(result)
      : [v] "v"(val)
      : "s2");
  out[lane] = result;
}
