// ARCH: gfx1250
//
// Regression fixture for issue #21: SMEM handler ignored the constant
// byte delta accumulated on the kernarg-segment-ptr SGPR before a
// downstream `s_load_b*`. Every Tensile UniversalArgs kernel (the
// full gfx1250 BF16/BF8/FP8/I32/MFMA Tensile library) was silently
// miscompiling through this path — the raiser read kernarg bytes
// from offset 0 instead of offset 16, corrupting the downstream
// tensor-descriptor SGPRs and faulting on the first
// `tensor_load_to_lds`.
//
// The fix (transpiler/raise_context.hpp +
// transpiler/raiser.cpp +  transpiler/handle_smem.cpp):
//
//   1. Post-handler hook in raiser.cpp tracks a 64-bit const delta
//      on the kernarg-pair SGPRs for the canonical `s_add_u32 lo +
//      s_addc_u32 hi, 0` pattern. See `RaiseContext::KernargPtrDelta`
//      for the full state machine.
//   2. `handle_smem.cpp` threads that delta through
//      `extractKernargDword`, so a shifted kernarg load correctly
//      resolves to `delta + imm_offset` instead of `imm_offset`
//      alone.
//   3. Any kernarg-pair mutation the tracker cannot describe as a
//      single 64-bit const delta (non-canonical add/addc pairing,
//      cross-BB merge, overwrite via another base) invalidates the
//      tracker — handle_smem.cpp then refuses the kernarg-slot fast
//      path loudly rather than silently miscompiling.
//
// THE CANONICAL SHAPE:
//
//   s_add_u32  s0, s0, 0x10    ; kernarg ptr += 16 (skip preamble)
//   s_addc_u32 s1, s1, 0       ; propagate carry
//   s_load_b64 s[2:3], s[0:1], 0   ; load kernarg[16:24] = arg4 ptr
//
// This fixture uses inline asm to materialise exactly that shape.
// The inline asm adds 16 to the kernarg pointer s[0:1], then issues
// an `s_load_b64 s[2:3]` at offset 0 — which should reach kernarg
// byte 16, which is the start of `arg4` (the output pointer, the
// only global_buffer in the signature).
//
// HOW THE FIXTURE EXERCISES THE PATH:
//
// Signature: four i32 scalars + one global-buffer pointer.
//
//     kernel(i32 arg0, i32 arg1, i32 arg2, i32 arg3, uint *out)
//
// AMDGPU kernarg layout for this kernel:
//
//     offset  0, size  4 : arg0 (i32)
//     offset  4, size  4 : arg1 (i32)
//     offset  8, size  4 : arg2 (i32)
//     offset 12, size  4 : arg3 (i32)
//     offset 16, size  8 : arg4 (ptr addrspace(1))
//
// The shifted-then-loaded `s_load_b64 s[2:3], s[0:1], 0` reaches
// kernarg bytes [16, 24) — i.e. the low+high dwords of `arg4`. The
// correct IR routes `s2 <- ka_lo(arg4)` and `s3 <- ka_hi(arg4)`.
// The pre-fix buggy IR routed `s2 <- arg0` (kernarg byte 0) and
// `s3 <- arg1` (kernarg byte 4).
//
// Side note: the inline asm clobbers s[0:1] (the kernarg pointer),
// so the downstream hipcc-generated stores against s[0:1] are NOT
// runtime-correct. The fixture does not care: it is a FileCheck
// lift test that pins the IR shape of `s2/s3 = ka_lo/ka_hi(arg4)`.
// A runtime-correct version would need an explicit load-then-
// restore pattern, which would obscure the canonical
// `s_add_u32/s_addc_u32/s_load_b64` shape we want to pin.
//
// MAINTENANCE: any change to the delta tracker in raiser.cpp, to
// the `KernargPtrDelta` struct, or to the SMEM kernarg-load branch
// in handle_smem.cpp must keep this fixture's
// `%vgpr*.0 = phi i32 [ %ka_{lo,hi}*, %... ]` pin green.

#include <hip/hip_runtime.h>

extern "C" __global__ void
smem_kernarg_const_delta_kernel(unsigned int arg0, unsigned int arg1,
                                unsigned int arg2, unsigned int arg3,
                                unsigned int *out) {
  unsigned int lo, hi;
  // Shift kernarg ptr s[0:1] past the first 16 bytes via the
  // canonical S_ADD_U32 / S_ADDC_U32 pair, then issue an s_load_b64
  // at imm offset 0 — which should reach kernarg byte 16 (the low
  // dword of `out`).
  //
  // Register-named source and destination force the disassembly to
  // land on exactly the `s_add_u32 s0,s0,0x10 ; s_addc_u32 s1,s1,0 ;
  // s_load_b64 s[2:3], s[0:1], 0` shape the const-delta tracker
  // exists to recognise.
  asm volatile("s_add_u32  s0, s0, 0x10\n\t"
               "s_addc_u32 s1, s1, 0\n\t"
               "s_load_b64 s[2:3], s[0:1], 0\n\t"
               "s_wait_kmcnt 0\n\t"
               "s_mov_b32 %[lo], s2\n\t"
               "s_mov_b32 %[hi], s3\n\t"
               : [lo] "=s"(lo), [hi] "=s"(hi)
               :
               : "s0", "s1", "s2", "s3", "memory");

  unsigned int tid = threadIdx.x;
  out[tid * 2 + 0] = lo;
  out[tid * 2 + 1] = hi;

  // Reference every scalar kernarg so front-end DCE does not strip
  // them from the AMDGPU kernarg layout. Without this the by_value
  // slot decomposition collapses and the shifted-kernarg load has
  // no scalar args ahead of it to displace past.
  out[tid * 2 + 2] = arg0 + arg1 + arg2 + arg3;
}
