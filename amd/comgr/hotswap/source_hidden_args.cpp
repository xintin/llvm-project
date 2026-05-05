#include "source_hidden_args.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace transpiler {
namespace {

// Offsets in the HSA AQL `hsa_kernel_dispatch_packet_t` as defined by the
// public HSA runtime header.  Do not use SI::KernelInputOffsets here: those are
// LLVM's kernel-input/implicit-buffer offsets (`NGROUPS`, `LOCAL_SIZE`), not the
// AQL dispatch-packet layout addressed by `llvm.amdgcn.dispatch.ptr`.
namespace DispatchPacket {
constexpr unsigned WorkgroupSizeXOffset = 4;
constexpr unsigned WorkgroupSizeYOffset = 6;
constexpr unsigned WorkgroupSizeZOffset = 8;
constexpr unsigned GridSizeXOffset = 12;
constexpr unsigned GridSizeYOffset = 16;
constexpr unsigned GridSizeZOffset = 20;

unsigned dispatchWorkgroupSizeOffset(unsigned Dim) {
  switch (Dim) {
  case 0:
    return WorkgroupSizeXOffset;
  case 1:
    return WorkgroupSizeYOffset;
  case 2:
    return WorkgroupSizeZOffset;
  default:
    report_fatal_error("invalid source hidden workgroup-size dimension");
  }
}

unsigned dispatchGridSizeOffset(unsigned Dim) {
  switch (Dim) {
  case 0:
    return GridSizeXOffset;
  case 1:
    return GridSizeYOffset;
  case 2:
    return GridSizeZOffset;
  default:
    report_fatal_error("invalid source hidden grid-size dimension");
  }
}
} // namespace DispatchPacket

Value *dispatchPtr(SourceHiddenArgContext &Ctx) {
  Function *DispatchPtrFn =
      Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::amdgcn_dispatch_ptr);
  return Ctx.B.CreateCall(DispatchPtrFn, {}, "dispatch_ptr");
}

Value *loadDispatchU16(SourceHiddenArgContext &Ctx, unsigned ByteOffset,
                       const Twine &Name) {
  Value *Ptr =
      Ctx.B.CreateConstInBoundsGEP1_32(Ctx.i8Ty, dispatchPtr(Ctx), ByteOffset);
  return Ctx.B.CreateZExt(
      Ctx.B.CreateLoad(Type::getInt16Ty(Ctx.C), Ptr, Name), Ctx.i32Ty,
      Name + "_zext");
}

Value *loadDispatchU32(SourceHiddenArgContext &Ctx, unsigned ByteOffset,
                       const Twine &Name) {
  Value *Ptr =
      Ctx.B.CreateConstInBoundsGEP1_32(Ctx.i8Ty, dispatchPtr(Ctx), ByteOffset);
  return Ctx.B.CreateLoad(Ctx.i32Ty, Ptr, Name);
}

Value *emitDispatchWorkgroupSize(SourceHiddenArgContext &Ctx, unsigned Dim) {
  return loadDispatchU16(Ctx, DispatchPacket::dispatchWorkgroupSizeOffset(Dim),
                         Twine("source_hidden_wg_size_") + Twine(Dim));
}

Value *emitDispatchGridSize(SourceHiddenArgContext &Ctx, unsigned Dim) {
  return loadDispatchU32(Ctx, DispatchPacket::dispatchGridSizeOffset(Dim),
                         Twine("source_hidden_grid_size_") + Twine(Dim));
}

Value *emitHiddenBlockCount(SourceHiddenArgContext &Ctx, unsigned Dim) {
  return Ctx.B.CreateUDiv(emitDispatchGridSize(Ctx, Dim),
                          emitDispatchWorkgroupSize(Ctx, Dim),
                          Twine("source_hidden_block_count_") + Twine(Dim));
}

Value *emitHiddenRemainder(SourceHiddenArgContext &Ctx, unsigned Dim) {
  return Ctx.B.CreateURem(emitDispatchGridSize(Ctx, Dim),
                          emitDispatchWorkgroupSize(Ctx, Dim),
                          Twine("source_hidden_remainder_") + Twine(Dim));
}

Value *emitGridDims(SourceHiddenArgContext &Ctx) {
  Value *GridY = emitDispatchGridSize(Ctx, 1);
  Value *GridZ = emitDispatchGridSize(Ctx, 2);
  Value *HasZ = Ctx.B.CreateICmpUGT(GridZ, Ctx.B.getInt32(1), "grid_has_z");
  Value *HasY = Ctx.B.CreateICmpUGT(GridY, Ctx.B.getInt32(1), "grid_has_y");
  return Ctx.B.CreateSelect(
      HasZ, Ctx.B.getInt32(3),
      Ctx.B.CreateSelect(HasY, Ctx.B.getInt32(2), Ctx.B.getInt32(1),
                         "grid_dims_y_or_x"),
      "source_hidden_grid_dims");
}

SourceHiddenArgValue unsupportedHiddenKind(StringRef Kind) {
  SourceHiddenArgValue Result;
  Result.Matched = true;
  Result.FailureDetail =
      (Twine("unsupported source hidden argument kind '") + Kind +
       "'; add explicit source-ABI synthesis instead of falling back to "
       "target implicitarg layout")
          .str();
  return Result;
}

SourceHiddenArgValue emitHiddenArgValue(SourceHiddenArgContext &Ctx,
                                        SourceHiddenArgKind Kind) {
  SourceHiddenArgValue Result;
  Result.Matched = true;
  if (Kind == SourceHiddenArgKind::HiddenBlockCountX)
    Result.Value = emitHiddenBlockCount(Ctx, 0);
  else if (Kind == SourceHiddenArgKind::HiddenBlockCountY)
    Result.Value = emitHiddenBlockCount(Ctx, 1);
  else if (Kind == SourceHiddenArgKind::HiddenBlockCountZ)
    Result.Value = emitHiddenBlockCount(Ctx, 2);
  else if (Kind == SourceHiddenArgKind::HiddenGroupSizeX)
    Result.Value = emitDispatchWorkgroupSize(Ctx, 0);
  else if (Kind == SourceHiddenArgKind::HiddenGroupSizeY)
    Result.Value = emitDispatchWorkgroupSize(Ctx, 1);
  else if (Kind == SourceHiddenArgKind::HiddenGroupSizeZ)
    Result.Value = emitDispatchWorkgroupSize(Ctx, 2);
  else if (Kind == SourceHiddenArgKind::HiddenRemainderX)
    Result.Value = emitHiddenRemainder(Ctx, 0);
  else if (Kind == SourceHiddenArgKind::HiddenRemainderY)
    Result.Value = emitHiddenRemainder(Ctx, 1);
  else if (Kind == SourceHiddenArgKind::HiddenRemainderZ)
    Result.Value = emitHiddenRemainder(Ctx, 2);
  else if (Kind == SourceHiddenArgKind::HiddenGridDims)
    Result.Value = emitGridDims(Ctx);
  else
    return unsupportedHiddenKind("<unknown>");
  return Result;
}

SourceHiddenArgValue emitSourceHiddenByte(SourceHiddenArgContext &Ctx,
                                          int ByteOffset) {
  SourceHiddenArgByte Byte = classifySourceHiddenArgByte(Ctx.Args, ByteOffset);
  if (!Byte.matched())
    return {};

  SourceHiddenArgValue Result = emitHiddenArgValue(Ctx, Byte.kind);
  if (!Result.Value && !Byte.valueKind.empty())
    Result.FailureDetail =
        (Twine("unsupported source hidden argument kind '") + Byte.valueKind +
         "'; add explicit source-ABI synthesis instead of falling back to "
         "target implicitarg layout")
            .str();
  if (!Result.Value)
    return Result;

  Value *Wide = Ctx.B.CreateZExtOrTrunc(Result.Value, Ctx.i64Ty, "hidden_wide");
  unsigned ByteInArg = Byte.byteIndexInArg();
  if (ByteInArg != 0)
    Wide = Ctx.B.CreateLShr(Wide, Ctx.B.getInt64(ByteInArg * 8),
                            "hidden_byte_shift");
  Result.Value = Ctx.B.CreateTrunc(Wide, Ctx.i8Ty, "source_hidden_byte");
  return Result;
}

} // namespace

SourceHiddenArgValue emitSourceHiddenInteger(SourceHiddenArgContext &Ctx,
                                             int ByteOffset,
                                             unsigned ByteWidth,
                                             bool IsSigned) {
  if (ByteWidth != 1 && ByteWidth != 2 && ByteWidth != 4)
    report_fatal_error("unsupported source hidden integer byte width");

  SourceHiddenArgValue Result;
  Value *Acc = Ctx.B.getInt32(0);
  for (unsigned I = 0; I < ByteWidth; ++I) {
    SourceHiddenArgValue Byte =
        emitSourceHiddenByte(Ctx, ByteOffset + static_cast<int>(I));
    if (!Byte.Matched) {
      if (I == 0)
        return {};
      Result.Matched = true;
      Result.FailureDetail =
          (Twine("source hidden dword at byte offset ") + Twine(ByteOffset) +
           " spans non-hidden byte " + Twine(ByteOffset + static_cast<int>(I)))
              .str();
      return Result;
    }
    if (!Byte.Value)
      return Byte;

    Result.Matched = true;
    Value *Part =
        Ctx.B.CreateZExt(Byte.Value, Ctx.i32Ty, "source_hidden_byte_zext");
    if (I != 0)
      Part = Ctx.B.CreateShl(Part, Ctx.B.getInt32(I * 8),
                             "source_hidden_byte_place");
    Acc = Ctx.B.CreateOr(Acc, Part, "source_hidden_dword");
  }
  if (IsSigned && ByteWidth < 4) {
    Type *NarrowTy = Type::getIntNTy(Ctx.C, ByteWidth * 8);
    Result.Value =
        Ctx.B.CreateSExt(Ctx.B.CreateTrunc(Acc, NarrowTy, "source_hidden_narrow"),
                         Ctx.i32Ty, "source_hidden_sext");
  } else {
    Result.Value = Acc;
  }
  return Result;
}

SourceHiddenArgValue emitSourceHiddenDword(SourceHiddenArgContext &Ctx,
                                           int ByteOffset) {
  return emitSourceHiddenInteger(Ctx, ByteOffset, /*ByteWidth=*/4,
                                 /*IsSigned=*/false);
}

} // namespace transpiler
