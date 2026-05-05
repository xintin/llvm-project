#include "kernarg_layout.hpp"

#include "llvm/ADT/StringRef.h"

using namespace llvm;

namespace transpiler {

SourceHiddenArgByte classifySourceHiddenArgByte(ArrayRef<KernelArgMeta> args,
                                                int byteOffset) {
  for (const KernelArgMeta &arg : args) {
    if (byteOffset < arg.offset || byteOffset >= arg.offset + arg.size)
      continue;

    StringRef kind(arg.valueKind);
    if (!kind.starts_with("hidden_"))
      return {};

    SourceHiddenArgByte result;
    result.valueKind = kind;
    result.argOffset = arg.offset;
    result.byteOffset = byteOffset;
    if (kind == "hidden_block_count_x")
      result.kind = SourceHiddenArgKind::HiddenBlockCountX;
    else if (kind == "hidden_block_count_y")
      result.kind = SourceHiddenArgKind::HiddenBlockCountY;
    else if (kind == "hidden_block_count_z")
      result.kind = SourceHiddenArgKind::HiddenBlockCountZ;
    else if (kind == "hidden_group_size_x")
      result.kind = SourceHiddenArgKind::HiddenGroupSizeX;
    else if (kind == "hidden_group_size_y")
      result.kind = SourceHiddenArgKind::HiddenGroupSizeY;
    else if (kind == "hidden_group_size_z")
      result.kind = SourceHiddenArgKind::HiddenGroupSizeZ;
    else if (kind == "hidden_remainder_x")
      result.kind = SourceHiddenArgKind::HiddenRemainderX;
    else if (kind == "hidden_remainder_y")
      result.kind = SourceHiddenArgKind::HiddenRemainderY;
    else if (kind == "hidden_remainder_z")
      result.kind = SourceHiddenArgKind::HiddenRemainderZ;
    else if (kind == "hidden_grid_dims")
      result.kind = SourceHiddenArgKind::HiddenGridDims;
    else
      result.kind = SourceHiddenArgKind::UnsupportedHidden;
    return result;
  }
  return {};
}

} // namespace transpiler
