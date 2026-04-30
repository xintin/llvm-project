#ifndef HOTSWAP_TRANSPILER_KERNARG_LAYOUT_HPP
#define HOTSWAP_TRANSPILER_KERNARG_LAYOUT_HPP

#include "code_object_utils.hpp"
#include "llvm/ADT/ArrayRef.h"

namespace transpiler {

// Source-kernel kernarg-segment metadata shared by the raiser and any
// helper that needs to reason about the segment without dereferencing
// it (e.g. the kernel-entry preloaded-SGPR seeding loop).
//
// The raiser itself no longer indexes the segment slot-by-slot:
// kernarg loads lift to GEP+load against `amdgcn_kernarg_segment_ptr`
// and the AMDGPU backend handles the ABI lowering. This struct keeps
// only the two pieces of metadata that survive that move.
struct KernargLayout {
  // Byte offset (within the source ABI's flat kernarg-segment view)
  // where the implicit-arg block begins. `handle_smem.cpp` consults
  // this to reroute SMEM loads at offsets >= implicitArgsBase through
  // `amdgcn_implicitarg_ptr` instead of the kernarg-segment pointer:
  // the source kernel's flat view is layout-correct for the source
  // ABI, but the lifted target kernel reaches implicit args via a
  // separate runtime pointer, so the offset must be rebased to
  // `byteOffset - implicitArgsBase`.
  int implicitArgsBase = 0;
  // Total kernarg segment size in bytes, copied from the kernel
  // descriptor's `.kernarg_segment_size`. Informational; the lifted
  // kernel's `Function` parameter list drives the backend's
  // `kernarg_segment_size` calculation in the output KD.
  int kernargSegmentSize = 0;
};

enum class PreloadedHiddenKernargDword {
  NotHidden,
  HiddenBlockCountX,
  HiddenBlockCountY,
  HiddenBlockCountZ,
  UnsupportedHidden,
};

// Classify a kernarg-preload dword that lands on a hidden metadata slot.
// Hardware preloads hidden args exactly like user args; treating them as
// padding would turn runtime-provided values (e.g. Triton's block count for
// `tl.num_programs`) into undef. Unsupported hidden kinds must refuse loudly.
PreloadedHiddenKernargDword classifyPreloadedHiddenKernargDword(
    llvm::ArrayRef<KernelArgMeta> args, int byteOffset);

} // namespace transpiler

#endif
