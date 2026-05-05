#ifndef HOTSWAP_TRANSPILER_KERNARG_LAYOUT_HPP
#define HOTSWAP_TRANSPILER_KERNARG_LAYOUT_HPP

#include "code_object_utils.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

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
  // Source metadata argument layout, including hidden_* entries. Used to
  // synthesize source-ABI hidden values without depending on target-runtime
  // implicit-arg layout.
  llvm::ArrayRef<KernelArgMeta> args;
  // Total kernarg segment size in bytes, copied from the kernel
  // descriptor's `.kernarg_segment_size`. Informational; the lifted
  // kernel's `Function` parameter list drives the backend's
  // `kernarg_segment_size` calculation in the output KD.
  int kernargSegmentSize = 0;
};

enum class SourceHiddenArgKind {
  None,
  HiddenBlockCountX,
  HiddenBlockCountY,
  HiddenBlockCountZ,
  HiddenGroupSizeX,
  HiddenGroupSizeY,
  HiddenGroupSizeZ,
  HiddenRemainderX,
  HiddenRemainderY,
  HiddenRemainderZ,
  HiddenGridDims,
  UnsupportedHidden,
};

struct SourceHiddenArgByte {
  SourceHiddenArgKind kind = SourceHiddenArgKind::None;
  llvm::StringRef valueKind;
  int argOffset = 0;
  int byteOffset = 0;

  bool matched() const { return kind != SourceHiddenArgKind::None; }
  unsigned byteIndexInArg() const {
    return static_cast<unsigned>(byteOffset - argOffset);
  }
};

// Resolve a byte offset in the source ABI's flat kernarg/hidden-arg metadata
// view.  Known source hidden args are later synthesized from dispatch state;
// unsupported hidden args must refuse instead of falling back to target
// implicitarg layout.
SourceHiddenArgByte classifySourceHiddenArgByte(
    llvm::ArrayRef<KernelArgMeta> args, int byteOffset);

} // namespace transpiler

#endif
