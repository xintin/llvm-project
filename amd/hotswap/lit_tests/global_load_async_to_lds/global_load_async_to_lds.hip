// ARCH: gfx1250
//
// Lit fixture for FLAT `global_load_async_to_lds_b{8,32,64,128}` —
// the gfx1250-only async global → LDS DMA family. Exercises all four
// widths in one kernel so the cross-target emulation's access type
// switch is pinned directly rather than by inspection.
//
// === Why this kernel must NEVER be dispatched ===
//
// The clang builtins
// `__builtin_amdgcn_global_load_async_to_lds_b{8,32,64,128}` lower to
// the FLAT VFLAT 0x60-0x62 reals, which only execute on gfx1250
// hardware. The fixture exists exclusively to compile a `.co` file
// that the raiser can disassemble and lift; running the kernel on
// any non-gfx1250 device would surface as a hardware exception.
//
// === Companion lit tests ===
//
// `global_load_async_to_lds.ll` (this directory) — pins the
// cross-target (gfx1250 → gfx942) loud-refusal contract. The
// handler in `handle_flat.cpp` refuses via
// `RaiseFailure::unsupportedShape` because gfx942 has no
// asynccnt unit and no `amdgcn.global.load.async.to.lds.b*`
// intrinsic; a synthesised `global_load + ds_write` pair would
// alter the wave's memory-ordering and asynccnt observable
// state, which gfx1250 producers (tensilelite f8 / bf16 GEMMs,
// triton block-pipelined matmul kernels) rely on for software
// pipelining.
//
// `global_load_async_to_lds_same_target.ll` (this directory) —
// pins the same-target (gfx1250 → gfx1250) intrinsic-emit shape:
// `inttoptr i32` → `ptr addrspace(3)` for the per-lane VGPR_32
// LDS base, the canonical
// `call void @llvm.amdgcn.global.load.async.to.lds.b{8,32,64,128}`,
// and the `i32 0 / i32 0` (offset, cpol) immediates that match
// the corpus encoding (every observed instance disassembles
// with `off`-style flat_offset = 0 and cpol = 0).
//
// === Builtin signatures (BuiltinsAMDGPU.td:1095-1102) ===
//
//   void __builtin_amdgcn_global_load_async_to_lds_b8(
//       global  char  *gaddr,
//       local   char  *laddr,
//       int           offset,
//       int           cpol);
//   void __builtin_amdgcn_global_load_async_to_lds_b32(
//       global  int   *gaddr,
//       local   int   *laddr,
//       int           offset,
//       int           cpol);
//   void __builtin_amdgcn_global_load_async_to_lds_b64(
//       global  v2i32 *gaddr, local v2i32 *laddr, int, int);
//   void __builtin_amdgcn_global_load_async_to_lds_b128(
//       global  v4i32 *gaddr, local v4i32 *laddr, int, int);

#include <hip/hip_runtime.h>

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));

extern "C" __global__ void
global_load_async_to_lds_kernel(int *gptr32,
                                  v2i *gptr64,
                                  v4i *gptr128,
                                  char *gptr8) {
  __shared__ int   lds32 [64];
  __shared__ v2i   lds64 [64];
  __shared__ v4i   lds128[64];
  __shared__ char  lds8  [64];

  int tid = (int)threadIdx.x;
  __builtin_amdgcn_global_load_async_to_lds_b32(
      __builtin_addressof(gptr32[tid]),
      __builtin_addressof(lds32[tid]),
      /*offset=*/0, /*cpol=*/0);
  __builtin_amdgcn_global_load_async_to_lds_b64(
      __builtin_addressof(gptr64[tid]),
      __builtin_addressof(lds64[tid]),
      /*offset=*/0, /*cpol=*/0);
  __builtin_amdgcn_global_load_async_to_lds_b128(
      __builtin_addressof(gptr128[tid]),
      __builtin_addressof(lds128[tid]),
      /*offset=*/0, /*cpol=*/0);
  __builtin_amdgcn_global_load_async_to_lds_b8(
      __builtin_addressof(gptr8[tid]),
      __builtin_addressof(lds8[tid]),
      /*offset=*/0, /*cpol=*/0);
}
