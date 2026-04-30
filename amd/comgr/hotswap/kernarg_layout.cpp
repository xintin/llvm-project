#include "kernarg_layout.hpp"

#include "llvm/ADT/StringRef.h"

using namespace llvm;

namespace transpiler {

PreloadedHiddenKernargDword classifyPreloadedHiddenKernargDword(
    ArrayRef<KernelArgMeta> args, int byteOffset) {
  for (const KernelArgMeta &arg : args) {
    if (arg.offset != byteOffset)
      continue;
    StringRef kind(arg.valueKind);
    if (!kind.starts_with("hidden_"))
      return PreloadedHiddenKernargDword::NotHidden;
    if (kind == "hidden_block_count_x")
      return PreloadedHiddenKernargDword::HiddenBlockCountX;
    if (kind == "hidden_block_count_y")
      return PreloadedHiddenKernargDword::HiddenBlockCountY;
    if (kind == "hidden_block_count_z")
      return PreloadedHiddenKernargDword::HiddenBlockCountZ;
    return PreloadedHiddenKernargDword::UnsupportedHidden;
  }
  return PreloadedHiddenKernargDword::NotHidden;
}

} // namespace transpiler
