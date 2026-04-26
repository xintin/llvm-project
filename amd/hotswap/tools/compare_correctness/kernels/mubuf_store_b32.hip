// Minimal reproducer: force a raw buffer_store_b32 on gfx942 and
// gfx1250 so the compare_correctness harness exercises the MUBUF store
// path in isolation from every other kernel feature.
//
// Rationale (2026-04-20, salmon kerneldex triage):
//
//   The Triton softmax corpus kernel (softmax_gfx1250.hsaco, used by
//   Gfx1250Gpu.Softmax) stores through raw buffer_store_b32 and
//   silently produces no output on the salmon path. The Triton add
//   kernel (corpus_add_fp32) uses global_store_b32 and passes. That
//   two-point comparison pointed at MUBUF, but softmax also does
//   cross-warp reductions, LDS traffic, permlanex16 emulation, and a
//   dropped s_setreg_imm32_b32 HWREG(MODE,25,1) at entry, so the
//   MUBUF axis was never isolated.
//
//   This kernel is the isolation. It does:
//     1. One ordinary global_load (drives the read side from a clean
//        path that we know works, so we can't blame the read half).
//     2. One compute op (*2.0f + 1.0f) to make mismatches obvious
//        instead of both sides reading zero.
//     3. One explicit buffer_store_b32 via
//        __builtin_amdgcn_raw_buffer_store_b32 against an SRD we
//        hand-build with __builtin_amdgcn_make_buffer_rsrc.
//
//   If salmon produces correct output on this kernel, MUBUF storing
//   alone is not the bug and the softmax failure is upstream of the
//   store (cross-wave reduction, MODE-register drop, or
//   WaveNativeProjection doubling the participating lanes on the
//   reduction path). If salmon produces zeros or no writes, MUBUF *is*
//   the bug and buildMubufSRD / the raw.buffer.store lowering in
//   handle_mubuf.cpp is the site to fix.
//
// Wired into the harness as makeMubufStoreB32Recipe() in
// compare_correctness.cpp and registered in KERNELS in the Makefile.
// Run with:
//
//   LD_PRELOAD=./libsalmon_intercept.so \
//     ./compare_correctness --recipe=mubuf_store_b32
//
// The harness runs the same .co three ways (native gfx942, legacy
// gfx1250, salmon-translated gfx1250→gfx942) against a CPU reference
// computing out[i] = in[i] * 2 + 1.

#include <hip/hip_runtime.h>

// __amdgpu_buffer_rsrc_t is clang's opaque buffer-resource type; at IR
// it is a <4 x i32>, and the SRD built by __builtin_amdgcn_make_buffer_rsrc
// is laid out per the target ISA's V# format (gfx1250 vs gfx942),
// i.e. clang (not the transpiler) chooses the right stride/flag
// placement for the target. This is exactly what makes this kernel a
// clean A/B: on gfx942 native the SRD is gfx942-shaped, on gfx1250
// source it is gfx1250-shaped, and the transpiler's job is to
// translate the *instruction* path while ending up with an equivalent
// V# on the target.
//
// make_buffer_rsrc(base, stride, num_records, flags):
//   base        = pointer (dword0..1 of V# = base[47:0])
//   stride      = i16 record stride (0 for a raw buffer)
//   num_records = i32 record count; we use 0x00FFFFFF so the hardware
//                 bounds check never trips.
//   flags       = i32 format/cachepol flags; 0 for a raw buffer.

extern "C" __global__ void mubuf_store_b32(const float *__restrict__ in,
                                           float *__restrict__ out,
                                           int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  __amdgpu_buffer_rsrc_t rsrc = __builtin_amdgcn_make_buffer_rsrc(
      (void *)out, (short)0, 0x00FFFFFF, 0);

  float v = in[tid] * 2.0f + 1.0f;
  unsigned int bits = __builtin_bit_cast(unsigned int, v);
  __builtin_amdgcn_raw_buffer_store_b32(bits, rsrc, tid * 4, 0, 0);
}
