// Bit-field insert via v_bfi_b32.
//
// Target handler: V_BFI_B32 (handle_valu.cpp).  `v_bfi_b32 dst, s0, s1, s2`
// computes `(s0 & s1) | (~s0 & s2)` — insert bits from `s1` where the
// `s0` mask is 1, otherwise take them from `s2`.  It is emitted by
// libdevice's asin implementation (see kernels/triton/corpus_asin_fp32.py),
// which exposes the real-world coverage gap:
//
//   kernels/triton/corpus_asin_fp32 (salmon)
//     EXIT=2  ...  unsupported instruction: v_bfi_b32
//
// This HIP probe reduces the gap to a one-instruction reproducer so the
// salmon handler can be landed independently of the rest of the libdevice
// asin flow.  Force the opcode with inline asm — hipcc otherwise tends to
// lower the ternary mask-and-merge pattern to and/andn2/or, and we want
// this recipe to pin `v_bfi_b32` specifically.
//
// Output shape: 1:1.  Inputs are three u32 arrays giving the per-lane mask,
// "one" source, and "zero" source (the gfx ISA names).  Output is the
// merged bit pattern.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_bfi_b32(const uint32_t *mask,
                                     const uint32_t *one_src,
                                     const uint32_t *zero_src,
                                     uint32_t *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t mi = mask[i], oi = one_src[i], zi = zero_src[i];
  uint32_t r;
  asm volatile("v_bfi_b32 %0, %1, %2, %3"
               : "=v"(r)
               : "v"(mi), "v"(oi), "v"(zi));
  out[i] = r;
}
