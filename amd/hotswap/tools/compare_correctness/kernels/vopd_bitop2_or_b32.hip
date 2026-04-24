// Minimal VOPD bitop2 probe: force `v_dual_bitop2_b32 ... bitop3:0x54`.
//
// This targets the same instruction family seen in failing tl.sort/tl.topk
// motifs, but strips the kernel down to one dual-issue packet so we can pin
// semantic drift to this opcode path.
//
// Expected per-bit behavior for bitop2 TTBL 0x54 with src0=8 is:
//   dst = src1 | 8
//
// We pair the Y-half bitop with an X-half mov to ensure VOPD packet emission.

#include <hip/hip_runtime.h>

extern "C" __global__ void vopd_bitop2_or_b32(const uint32_t *in,
                                               uint32_t *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  uint32_t x = in[tid];
#if defined(__gfx1250__)
  uint32_t x_half_dummy = 0;
  uint32_t y_half_result = 0;
  asm volatile(
      "v_dual_mov_b32 %0, 0 :: v_dual_bitop2_b32 %1, 8, %2 bitop3:0x54\n\t"
      : "=&v"(x_half_dummy), "=&v"(y_half_result)
      : "v"(x)
      : "memory");
  (void)x_half_dummy;
  out[tid] = y_half_result;
#else
  out[tid] = (x | 8u);
#endif
}
