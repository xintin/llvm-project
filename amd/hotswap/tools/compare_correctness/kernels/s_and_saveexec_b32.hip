// Exec save/restore via s_and_saveexec_b32 (forced with inline asm).
//
// Target handler: S_AND_SAVEEXEC_B32 (handle_sop1.cpp) and its
// siblings (`s_saveexec_b32`, `s_or_saveexec_b32`, ...).  These are
// the scalar exec-manipulation opcodes emitted on gfx1250 to
// implement non-uniform branches.  `block_sum_shfl` tripped the gap:
//
//   kernels/block_sum_shfl (salmon)
//     EXIT=2  ...  unsupported instruction: s_[and_]saveexec_b32
//
// We force the opcode with inline asm because the current gfx1250
// compiler chooses the decomposed form
//   s_mov_b32  sX, exec_lo
//   v_cmpx_*   ...
//   s_or_b32   exec_lo, exec_lo, sX
// for simple branches.  That decomposition exposes its own
// wave-size-translation bug (covered by the v_cmpx_gt_i32 probe); the
// fused `s_and_saveexec_b32` opcode needs independent coverage.
//
// Semantics exercised: `s_and_saveexec_b32 sdst, src` sets
//   sdst     = exec_lo            (the saved mask)
//   exec_lo  = exec_lo & src      (the narrowed active mask)
// We feed a block-uniform `src` via `readfirstlane` so the mask is
// SGPR-resident, execute a data-dependent body under the narrowed
// exec, then restore exec_lo from the saved scalar.  Observable
// output is per-lane: active-under-narrowed-mask lanes receive
// body(pred[i]), masked-off lanes stay at the 0 sentinel.  The CPU
// reference applies the same bit-wise lane-mask semantics.
//
// On gfx942 native we emulate the observable output in plain C++ so
// the harness's native column stays meaningful.

#include <hip/hip_runtime.h>

static __device__ __forceinline__ uint32_t body(uint32_t p) {
  uint32_t acc = p;
  acc ^= 0xDEADBEEFu;
  acc = acc * 2654435761u;  // Knuth multiplicative hash (32-bit)
  acc ^= acc >> 13;
  return acc;
}

extern "C" __global__ void s_and_saveexec_b32(const uint32_t *mask_in,
                                              const uint32_t *pred,
                                              uint32_t *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  uint32_t p = pred[tid];
  // Narrow `out` mask lives per-block and is SGPR-resident.
  uint32_t mask = __builtin_amdgcn_readfirstlane(mask_in[blockIdx.x]);
#if defined(__gfx1250__)
  uint32_t saved;
  uint32_t hashed = body(p);
  uint32_t *addr = out + tid;
  asm volatile(
      "s_and_saveexec_b32 %[saved], %[mask]\n\t"
      "global_store_b32 %[addr], %[hashed], off\n\t"
      "s_wait_storecnt 0\n\t"
      "s_mov_b32 exec_lo, %[saved]\n\t"
      : [saved] "=&s"(saved)
      : [mask] "s"(mask), [addr] "v"(addr), [hashed] "v"(hashed)
      : "exec", "memory");
#else
  // gfx942 native (wave64) fallback: reproduce the same lane-mask
  // gating observable from the gfx1250 asm path.  Each thread's
  // wave-relative lane position, clamped to 32 (the bit-width of the
  // exec_lo mask on wave32), decides whether the store fires.
  int waveLane32 = threadIdx.x & 31;
  if ((mask >> waveLane32) & 1u) {
    out[tid] = body(p);
  }
#endif
}
