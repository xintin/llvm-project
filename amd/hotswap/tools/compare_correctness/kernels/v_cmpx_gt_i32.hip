// Predicated store via v_cmpx_gt_i32.
//
// Target handler: V_CMPX_GT_I32 (handle_valu.cpp).  `v_cmpx_gt_i32_e64 s0, s1`
// is the compare-and-write-exec form — it sets exec lanes to `s0 > s1`
// (signed compare) in one step.  This pattern appears in `lane_swap` and
// in any HIP source that uses a simple if-then predicated store; the
// salmon path currently fails because it has no lowering for the opcode:
//
//   kernels/lane_swap (salmon)
//     EXIT=2  ...  unsupported instruction:  v_cmpx_gt_i32
//
// The probe deliberately uses data-dependent inputs (not lane position)
// so the wave-size classifier accepts the kernel — we want to exercise
// the *lowering*, not re-trip the lane-position-dependent-EXEC refuser.
// On gfx1250 the predicate is forced with inline asm; on gfx942 native
// the same observable behaviour is produced with a plain branch so the
// CPU reference agrees.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_cmpx_gt_i32(const int32_t *a, const int32_t *b,
                                         uint32_t *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  int32_t ai = a[tid], bi = b[tid];
  uint32_t *addr = out + tid;
  uint32_t val = 1;
#if defined(__gfx1250__)
  // v_cmpx_gt_i32_e64 sets exec_lo[lane] = (ai > bi).  The subsequent
  // global_store_b32 only fires on surviving lanes.  We then restore
  // exec_lo to all-ones so the rest of the kernel (if any) runs with
  // the original mask.  Mask source is purely data-dependent, not
  // lane-position-dependent.
  asm volatile(
      "v_cmpx_gt_i32_e64 %[ai], %[bi]\n\t"
      "global_store_b32 %[addr], %[val], off\n\t"
      "s_wait_storecnt 0\n\t"
      "s_mov_b32 exec_lo, -1\n\t"
      :
      : [ai] "v"(ai), [bi] "v"(bi), [addr] "v"(addr), [val] "v"(val)
      : "exec", "memory");
#else
  if (ai > bi) *addr = val;
#endif
}
