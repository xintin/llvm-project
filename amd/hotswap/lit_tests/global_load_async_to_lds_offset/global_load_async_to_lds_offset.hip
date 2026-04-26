// ARCH: gfx1250
//
// Lit fixture for FLAT `global_load_async_to_lds_b32` with a non-zero
// `flat_offset` immediate.  The sibling fixture in
// `global_load_async_to_lds/` pins the common zero-offset b32/b64/b128
// shape; this one exists solely to exercise the cross-target GEP arm in
// `handle_flat.cpp`.
//
// === Why this kernel must NEVER be dispatched ===
//
// The clang builtin lowers to the gfx1250-only FLAT async global -> LDS
// DMA instruction.  The `.co` is test input for the raiser; running it on
// a non-gfx1250 device would be a hardware fault, not a meaningful test.
//
// === ISA offset contract ===
//
// `instruction_manual.pdf §13.6.10` says:
//
//   dsaddr = LDS_BASE.b32 + VGPR[laneId][VDST.u32] + INST_OFFSET.b32;
//   memaddr = ADDR;
//   // Address computed the same as for other GLOBAL instructions
//
// and `programming_manual.pdf §4.9.9.1` expands the GLOBAL async-load
// addressing as:
//
//   LDS[VGPR[VDST][lane] + byte + INST_OFFSET] =
//   GLOBAL_MEMORY[VGPR[VADDR][lane] + INST_OFFSET + byte];
//
// That is the source-of-truth reason `flat_offset` must be materialised
// on BOTH pointers in the gfx1250 -> gfx942 synchronous emulation: one
// i8-GEP for `memaddr`, one i8-GEP for `dsaddr`.
//
// === Builtin signature (BuiltinsAMDGPU.td:1095-1102) ===
//
//   void __builtin_amdgcn_global_load_async_to_lds_b32(
//       global int *gaddr, local int *laddr, int offset, int cpol);

#include <hip/hip_runtime.h>

extern "C" __global__ void
global_load_async_to_lds_offset_kernel(int *gptr32) {
  __shared__ int lds32[64];

  int tid = (int)threadIdx.x;
  __builtin_amdgcn_global_load_async_to_lds_b32(
      __builtin_addressof(gptr32[tid]),
      __builtin_addressof(lds32[tid]),
      /*offset=*/16, /*cpol=*/0);
}
