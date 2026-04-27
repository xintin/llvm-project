#include "kernarg_layout.hpp"

#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <string>

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

llvm::Value *extractKernargDword(const KernargLayout &layout,
                                 llvm::IRBuilder<> &B,
                                 llvm::Function *F,
                                 int byteOffset,
                                 std::string *whyNot) {
  Type *i32Ty = B.getInt32Ty();
  Type *i64Ty = B.getInt64Ty();

  int loadEnd = byteOffset + 4;
  Value *assembled = ConstantInt::get(i32Ty, 0);
  unsigned knownMask = 0;
  bool sawOverlap = false;

  for (auto &p : layout.params) {
    int pEnd = p.byteOffset + p.byteSize;
    if (pEnd <= byteOffset || p.byteOffset >= loadEnd)
      continue;

    int relOff = byteOffset - p.byteOffset;
    int relLoadOff = p.byteOffset - byteOffset;
    Value *arg = F->getArg(p.paramIdx);
    Type *argTy = arg->getType();

    // Pointer / i64 slot: the low dword corresponds to relOff == 0,
    // the high dword to relOff == 4. Both source-ABI and target-ABI
    // place the low dword in the low SGPR (little-endian kernarg
    // buffer), so the lshr direction here is unambiguous.
    if ((p.isPointer || argTy == i64Ty) && byteOffset >= p.byteOffset &&
        loadEnd <= pEnd) {
      Value *asI64 =
          (argTy == i64Ty) ? arg : B.CreatePtrToInt(arg, i64Ty, "ka_p2i");
      if (relOff == 0)
        return B.CreateTrunc(asI64, i32Ty, "ka_lo");
      if (relOff == 4)
        return B.CreateTrunc(B.CreateLShr(asI64, 32, "ka_hi_shr"), i32Ty,
                             "ka_hi");
      if (whyNot) {
        *whyNot = "kernarg byte offset " + std::to_string(byteOffset) +
                 " has unsupported sub-offset " + std::to_string(relOff) +
                 " into 64-bit param idx " + std::to_string(p.paramIdx) +
                 " (param byteOffset=" + std::to_string(p.byteOffset) +
                 ", byteSize=" + std::to_string(p.byteSize) + ")";
      }
      return nullptr;
    }

    if (p.byteOffset < byteOffset || pEnd > loadEnd) {
      if (whyNot) {
        *whyNot = "kernarg byte offset " + std::to_string(byteOffset) +
                 " partially overlaps param idx " +
                 std::to_string(p.paramIdx) + " (param byteOffset=" +
                 std::to_string(p.byteOffset) + ", byteSize=" +
                 std::to_string(p.byteSize) +
                 "); cross-dword scalar extraction is unsupported";
      }
      return nullptr;
    }

    if (argTy == i32Ty && p.byteSize == 4 && relLoadOff == 0)
      return arg;

    unsigned widthBits = static_cast<unsigned>(p.byteSize * 8);
    unsigned shiftBits = static_cast<unsigned>(relLoadOff * 8);
    Value *asI32 = nullptr;
    if (argTy == i32Ty) {
      if (p.byteSize != 4) {
        if (whyNot) {
          *whyNot = "kernarg byte offset " + std::to_string(byteOffset) +
                   " hits i32 param idx " + std::to_string(p.paramIdx) +
                   " with metadata byteSize=" + std::to_string(p.byteSize) +
                   "; raiser bug";
        }
        return nullptr;
      }
      asI32 = arg;
    } else if (argTy->isIntegerTy(8) || argTy->isIntegerTy(16)) {
      if (argTy->getIntegerBitWidth() != widthBits) {
        if (whyNot) {
          *whyNot = "kernarg byte offset " + std::to_string(byteOffset) +
                   " hits narrow param idx " + std::to_string(p.paramIdx) +
                   " with mismatched IR bit width";
        }
        return nullptr;
      }
      asI32 = B.CreateZExt(arg, i32Ty, "ka_narrow_zext");
    } else {
      if (whyNot) {
        *whyNot = "kernarg byte offset " + std::to_string(byteOffset) +
                 " hits param idx " + std::to_string(p.paramIdx) +
                 " with unsupported IR arg type";
      }
      return nullptr;
    }

    if (shiftBits != 0)
      asI32 = B.CreateShl(asI32, shiftBits, "ka_narrow_shl");
    assembled = B.CreateOr(assembled, asI32, "ka_assembled");
    unsigned mask =
        widthBits == 32 ? 0xffffffffu : ((1u << widthBits) - 1u);
    knownMask |= mask << shiftBits;
    sawOverlap = true;
  }

  if (sawOverlap) {
    if (knownMask != 0xffffffffu) {
      Value *undefBits = B.CreateAnd(UndefValue::get(i32Ty),
                                     ConstantInt::get(i32Ty, ~knownMask),
                                     "ka_padding_undef");
      assembled = B.CreateOr(assembled, undefBits, "ka_with_padding");
    }
    return assembled;
  }

  // No named kernarg slot covers [byteOffset, byteOffset+4).
  //
  // If the dword still lies fully inside the declared kernarg segment,
  // this is alignment padding (either between named args or trailing at
  // the end of the segment). HSA leaves padding bytes unspecified, so the
  // LLVM-correct representation is `undef i32` rather than inventing a
  // concrete value.
  //
  // Example that requires this behavior: gfx1250 kernarg-preload can pull
  // contiguous dwords that pass through an ABI alignment hole between an
  // i32 arg and a later aligned pointer arg. Those bytes are still in the
  // segment and must be modeled as "any value".
  //
  // A dword starting exactly at segment_end is also treated as undef. LLVM's
  // AMDGPU backend can choose an aligned vector SMEM load over the final named
  // args and route the last dword into an unused SGPR; refusing that shape
  // regresses existing MFMA fixtures. Further overrun remains a hard error.
  if (layout.kernargSegmentSize > 0 &&
      (byteOffset + 4 <= layout.kernargSegmentSize ||
       byteOffset == layout.kernargSegmentSize))
    return UndefValue::get(i32Ty);

  if (whyNot) {
    int lastEnd = 0;
    for (auto &p : layout.params)
      lastEnd = std::max(lastEnd, p.byteOffset + p.byteSize);
    *whyNot = "kernarg byte offset " + std::to_string(byteOffset) +
             " is not covered by any explicit kernarg slot "
             "(last named arg ends at " + std::to_string(lastEnd) +
             ", kernarg_segment_size=" +
             std::to_string(layout.kernargSegmentSize) + ")";
  }
  return nullptr;
}

} // namespace transpiler
