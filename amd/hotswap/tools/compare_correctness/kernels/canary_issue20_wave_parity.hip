// Canary for issue #20: "Lane ID leak on warp specialization due to hard
// coded wave size".
//
// The source pattern is the exact shape from the issue report:
//
//   v_readfirstlane_b32 sTmp, vThreadId
//   s_lshr_b32          sTmp, sTmp, 5
//   s_bitcmp1_b32       sTmp, 0
//   s_cbranch_scc1      oddPath
//
// On gfx1250 wave32, `threadIdx.x >> 5` is the source wave id inside the
// workgroup, so a 128-thread block takes alternating even/odd paths:
// lanes 0..31 even, 32..63 odd, 64..95 even, 96..127 odd.  A native gfx942
// wave64 compile has no source-wave notion: its two target waves see lane-0
// thread ids 0 and 64, both even after `>> 5`, so the odd paths disappear.
// Salmon must preserve the wave32 source semantics while running on wave64.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__(128) void
canary_issue20_wave_parity(uint32_t *__restrict__ out) {
  uint32_t localTid = threadIdx.x;
  uint32_t globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t first = 0;
  uint32_t waveId = 0;
  uint32_t base = 0;
  const uint32_t oddBase = 0xB0000000u;
  const uint32_t evenBase = 0xA0000000u;

  asm volatile("v_readfirstlane_b32 %0, %1" : "=s"(first) : "v"(localTid));
  asm volatile("s_lshr_b32 %0, %1, 5" : "=s"(waveId) : "s"(first));
  asm volatile("s_bitcmp1_b32 %1, 0\n\t"
               "s_cbranch_scc1 1f\n\t"
               "s_mov_b32 %0, %3\n\t"
               "s_branch 2f\n"
               "1:\n\t"
               "s_mov_b32 %0, %2\n"
               "2:"
               : "=s"(base)
               : "s"(waveId), "s"(oddBase), "s"(evenBase));

  out[globalTid] = base | globalTid;
}
