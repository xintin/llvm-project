// ARCH: gfx1250
//
// Minimal fixture for SOP2 immediate-mask shadow propagation:
//   v_cmp_* -> s_xor_b32 sN, sN, -1 -> v_cndmask ... sN
//
// This is the exact pattern fixed in handle_sop2.cpp where non-register SOP2
// sources (immediates / exprs) must still produce per-lane i1 via
// srcExecWidth+extractLaneBitFromWaveMask so derived-mask shadow propagation
// remains valid.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_xor_imm_mask_shadow_kernel(float *out,
                                                         const float *in) {
  int lane = (int)threadIdx.x;
  float val = in[lane];
  float result = 0.0f;
  asm volatile(
      "v_cmp_ge_f32_e64 s2, |%[v]|, 0.5\n\t"
      "s_xor_b32 s2, s2, -1\n\t"
      "v_cndmask_b32_e64 %[r], -1.0, 1.0, s2\n\t"
      : [r] "=v"(result)
      : [v] "v"(val)
      : "s2");
  out[lane] = result;
}

