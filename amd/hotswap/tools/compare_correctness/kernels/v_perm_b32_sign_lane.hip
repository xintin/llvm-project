// Minimal v_perm_b32 sign-lane probe.
//
// The real TensorDescriptor `_upcast_from_mxfp` kernel builds the BF16 sign
// lanes for two packed FP4 values with:
//
//   low_sign  = tensor & 0x08
//   high_sign = tensor & 0x80
//   packed    = v_perm_b32(high_sign, low_sign, 0x05040100)
//   result    = v_pk_lshlrev_b16(0x0008000c, packed)
//
// Hardware `v_perm_b32` selectors 0..3 read the low 32-bit half of the
// constructed 64-bit value.  For the instruction form, that low half is src1,
// not src0.  A source-order inversion copies the high FP4 sign into the low
// BF16 lane, matching the sparse sign flips seen in the GPT-OSS MXFP upcast.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void v_perm_b32_sign_lane(const uint32_t *in,
                                                uint32_t *out, int n) {
  // The compare recipe launches this kernel with a fixed 256-thread block.
  // Avoid `blockDim.x` so the gfx1250 code object does not read the unresolved
  // implicit-arg block before reaching the instruction sequence under test.
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;

  uint32_t x = in[i] & 0xffu;
  uint32_t low_sign = x & 0x08u;
  uint32_t high_sign = x & 0x80u;
  uint32_t packed;
  uint32_t result;

#if defined(__gfx1250__)
  asm volatile(
      "v_perm_b32 %[packed], %[hi], %[lo], 0x05040100\n\t"
      "v_pk_lshlrev_b16 %[result], 0x0008000c, %[packed]\n\t"
      : [packed] "=&v"(packed), [result] "=&v"(result)
      : [hi] "v"(high_sign), [lo] "v"(low_sign));
#else
  packed = (high_sign << 16) | low_sign;
  uint32_t low = (packed & 0x0000ffffu) << 12;
  uint32_t high = (packed & 0xffff0000u) << 8;
  result = (high & 0xffff0000u) | (low & 0x0000ffffu);
#endif

  out[i] = result;
}
