#include "mubuf_addr.hpp"

#include "Utils/AMDGPUBaseInfo.h" // AMDGPU::getNamedOperandIdx, AMDGPU::OpName::offset
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;

namespace transpiler {

namespace {

// Classify source operands of a MUBUF load/store into {srsrc, vaddr,
// soff, imm, vdata}. Keys on `ParsedReg::Kind` rather than position so
// MUBUF and VBUFFER encodings with different operand orders both
// route here. `isStore` controls the skip-first-VGPR rule: stores
// carry vdata (VGPR) ahead of vaddr (VGPR) in the operand list.
//
// `immOff` is read by name (`AMDGPU::OpName::offset`) rather than
// scanning for the first non-zero immediate. The greedy scan was
// unsound: MUBUF / VBUFFER carry both an `OpName::offset` byte
// immediate AND an `OpName::cpol` cache-policy immediate, and when
// the byte offset is zero the scan would silently grab cpol's value
// (e.g. `0x20` for `scope:SCOPE_DEV`) and use it as a per-lane voffset
// — every store/load with `offset:0 scope:SCOPE_*` would land 32 B
// past the intended address. The R1 lit regression guard issues
// exactly this shape (`buffer_store_b128 ... offset:0 scope:SCOPE_DEV`),
// so the fix is required for the BUFFER_STORE rewrite to actually
// hit the right address. CPol bits are otherwise still dropped (see
// the "Not refused here" note in handle_mubuf.cpp's refusal block,
// and §"Known limitations" of hotswap/docs/buffer-store-lowering.md).
struct MubufOps {
  ParsedReg srsrc;
  ParsedReg vaddr;
  ParsedReg soff;
  ParsedReg vdata;
  int64_t immOff = 0;
  bool haveSrsrc = false;
  bool haveVaddr = false;
  bool haveSoff = false;
  bool hasSwz = false;
};

MubufOps classifyMubufOps(const DecodedInst &di, OpResolver &op,
                           bool isStore) {
  MubufOps out;

  // Byte offset by name. Absent (-1) on encodings that don't carry it
  // (e.g. atomics with no immediate offset slot); the default `immOff
  // = 0` is correct for those — the caller layers in `voffset` from
  // `vaddr` independently.
  int offIdx =
      llvm::AMDGPU::getNamedOperandIdx(di.inst.getOpcode(),
                                       llvm::AMDGPU::OpName::offset);
  if (offIdx >= 0 &&
      static_cast<unsigned>(offIdx) < di.inst.getNumOperands() &&
      di.inst.getOperand(static_cast<unsigned>(offIdx)).isImm()) {
    out.immOff = di.inst.getOperand(static_cast<unsigned>(offIdx)).getImm();
  }

  int swzIdx =
      llvm::AMDGPU::getNamedOperandIdx(di.inst.getOpcode(),
                                       llvm::AMDGPU::OpName::swz);
  if (swzIdx >= 0 &&
      static_cast<unsigned>(swzIdx) < di.inst.getNumOperands() &&
      di.inst.getOperand(static_cast<unsigned>(swzIdx)).isImm()) {
    out.hasSwz = di.inst.getOperand(static_cast<unsigned>(swzIdx)).getImm() != 0;
  }

  int vgprSrcCount = 0;
  for (unsigned k = 0; k < op.nSrcs(); ++k) {
    if (!di.isReg(op.srcIdx(k)))
      continue;
    ParsedReg pr = op.srcReg(k);
    if (pr.kind == ParsedReg::SGPR && pr.baseIdx >= 0 && !out.haveSrsrc) {
      out.srsrc = pr;
      out.haveSrsrc = true;
    } else if (pr.kind == ParsedReg::VGPR) {
      vgprSrcCount++;
      // For stores, the first VGPR source is vdata (the stored value);
      // the second is the per-lane buffer offset (vaddr).
      if (isStore && vgprSrcCount == 1) {
        out.vdata = pr;
        continue;
      }
      if (!out.haveVaddr) {
        out.vaddr = pr;
        out.haveVaddr = true;
      }
    } else if (pr.kind == ParsedReg::SGPR && pr.baseIdx >= 0 &&
               !out.haveSoff) {
      out.soff = pr;
      out.haveSoff = true;
    }
  }
  return out;
}

// Read the four consecutive SGPR dwords of a MUBUF/VBUFFER SRSRC
// 128-bit tuple. Returns the raw source dwords; callers that only need
// the packaged raw-buffer descriptor should use `buildMubufSRD` below.
//
// No null-check on the returned Value*s: `AllocaRegFile::readReg32`
// already fails loudly on unhandled ParsedReg kinds and out-of-range
// SGPR indices, so SGPR reads are guaranteed to hand back a real
// Value*.
struct SRSRCDwords {
  Value *dw0;
  Value *dw1;
  Value *dw2;
  Value *dw3;
};

SRSRCDwords readSRSRCDwords(RaiseContext &ctx, ParsedReg srsrc) {
  Value *dw0 = ctx.regs.readReg32(ctx.B, srsrc);
  ParsedReg srsrc1 = srsrc; srsrc1.baseIdx = srsrc.baseIdx + 1;
  ParsedReg srsrc2 = srsrc; srsrc2.baseIdx = srsrc.baseIdx + 2;
  ParsedReg srsrc3 = srsrc; srsrc3.baseIdx = srsrc.baseIdx + 3;
  Value *dw1 = ctx.regs.readReg32(ctx.B, srsrc1);
  Value *dw2 = ctx.regs.readReg32(ctx.B, srsrc2);
  Value *dw3 = ctx.regs.readReg32(ctx.B, srsrc3);
  return {dw0, dw1, dw2, dw3};
}

bool constantI32(Value *v, uint32_t &out) {
  if (auto *ci = dyn_cast<ConstantInt>(v)) {
    out = static_cast<uint32_t>(ci->getZExtValue());
    return true;
  }
  return false;
}

// Build a gfx942-compatible raw buffer descriptor <4 x i32> from the
// source SRSRC dwords. Each word is routed through
// `amdgcn.readfirstlane` so it lands in an SGPR — the backend would
// otherwise emit a waterfall loop around the intrinsic call.
//
// dw1/dw2/dw3 are target-normalised rather than blindly copied from the
// source SGPRs. gfx1250 raw-pointer descriptors carry high descriptor
// marker bits alongside the base-high dword and use a compact
// NUM_RECORDS sentinel that is not gfx942's raw-buffer maximum, while
// gfx942 still needs a non-invalid DATA_FORMAT in RSRC3 for stores to
// commit.
//
// Why dw3 must not be zero on gfx942: empirically
// `buffer_store_dword` (and the other MUBUF raw-store flavours) on
// CDNA3 silently drops every lane's write when dw3's DATA_FORMAT
// field is BUF_DATA_FORMAT_INVALID (0). The ISA manual advertises
// these ops as untyped, but the MI300 MUBUF engine still checks
// DATA_FORMAT != 0 before committing the store. This manifests as
// Softmax (the only corpus kernel that reaches us through MUBUF
// stores — vecadd / add_fp32 use FLAT/GLOBAL) leaving its output
// buffer at sentinel after a salmon run, with no HIP error, no GPU
// fault, and no stderr output from the runtime.
//
// Bisected with kernels/build/mubuf_store_b32 + inline-asm V#
// probes (see hotswap/docs/triage-2026-04-20-softmax-matmul128.md):
//
//   dw3=0x00020000  MATCH   DATA_FORMAT=32 alone (minimum working)
//   dw3=0x00024000  MATCH   DATA_FORMAT=32 + NUM_FORMAT=UINT
//   dw3=0x00027000  MATCH   Triton's native-gfx942 shape (FORMAT_32 + FLOAT)
//   dw3=0x00007000  DROP    NUM_FORMAT=FLOAT alone, DATA_FORMAT=0
//   dw3=0x00000004  DROP    DST_SEL_X identity, DATA_FORMAT=0
//   dw3=0x00000000  DROP    (what we used to emit)
//
// Use Triton's native gfx942 raw-buffer sentinels, narrowly:
//
//   RSRC2 = 0x7ffffffe  NUM_RECORDS, the largest 4-byte-aligned byte
//                       bound used by native gfx942 Triton for raw
//                       pointer-derived descriptors.
//   RSRC3 = 0x00027000  FORMAT_32 + NUM_FORMAT_FLOAT.
//
// The source descriptor value 0x00ffffff is the gfx1250 raw-pointer
// "effectively unbounded" sentinel Triton emits for these JIT MUBUF
// descriptors. Passing it through to gfx942 bounds the buffer to 16 MiB,
// so GPT-OSS decode_attention._fwd_kernel_stage2's `Mid_O` loads for
// batches >= 64 were hardware-OOB and returned zero. Map that sentinel
// to gfx942's native raw-buffer maximum, but preserve every other
// NUM_RECORDS value exactly. That keeps finite source bounds finite
// instead of turning real source-OOB accesses into target in-bounds
// accesses. A true all-ones source value (0xffffffff, OOB disabled per
// the buffer-resource contract) is also preserved.
//
// The previous DATA_FORMAT-only RSRC3 value (0x00020000) is sufficient
// for the original dword-store probe, but native gfx942 Triton uses
// FORMAT_32 + NUM_FORMAT_FLOAT for the same raw store family; the raw
// intrinsics still move the explicitly typed payload bits without
// numeric conversion.
Value *buildMubufSRD(RaiseContext &ctx, const SRSRCDwords &dw) {
  constexpr uint32_t kGfx1250RawPointerWord1Bits = 0xfc000000u;
  constexpr uint32_t kGfx1250RawBufferMaxRecords = 0x00ffffffu;
  constexpr uint32_t kGfx942RawBufferMaxRecords = 0x7ffffffeu;
  constexpr uint32_t kGfx942RawBufferFormat32 = 0x00020000u;
  constexpr uint32_t kGfx942RawBufferFormat32Uint = 0x00024000u;
  constexpr uint32_t kGfx942RawBufferFormat32Float = 0x00027000u;

  Function *readfirstlane = Intrinsic::getOrInsertDeclaration(
      &ctx.M, Intrinsic::amdgcn_readfirstlane, {ctx.i32Ty});
  Value *dw1NonBaseBits =
      ctx.B.CreateAnd(dw.dw1, ConstantInt::get(ctx.i32Ty, 0xFFFF0000u),
                      "srd_dw1_nonbase_bits");
  Value *dw1HasOnlyBase =
      ctx.B.CreateICmpEQ(dw1NonBaseBits, ConstantInt::get(ctx.i32Ty, 0),
                         "srd_dw1_only_base");
  Value *dw1HasGfx125RawBits =
      ctx.B.CreateICmpEQ(dw1NonBaseBits,
                         ConstantInt::get(ctx.i32Ty, kGfx1250RawPointerWord1Bits),
                         "srd_dw1_gfx125_raw_bits");
  Value *dw1IsRawBase =
      ctx.B.CreateOr(dw1HasOnlyBase, dw1HasGfx125RawBits, "srd_dw1_raw_base");
  Value *dw3IsZero =
      ctx.B.CreateICmpEQ(dw.dw3, ConstantInt::get(ctx.i32Ty, 0), "srd_dw3_zero");
  Value *dw3IsFormat32 =
      ctx.B.CreateICmpEQ(dw.dw3, ConstantInt::get(ctx.i32Ty, kGfx942RawBufferFormat32),
                         "srd_dw3_format32");
  Value *dw3IsFormat32Uint =
      ctx.B.CreateICmpEQ(dw.dw3,
                         ConstantInt::get(ctx.i32Ty, kGfx942RawBufferFormat32Uint),
                         "srd_dw3_format32_uint");
  Value *dw3IsFormat32Float =
      ctx.B.CreateICmpEQ(dw.dw3,
                         ConstantInt::get(ctx.i32Ty, kGfx942RawBufferFormat32Float),
                         "srd_dw3_format32_float");
  Value *dw3IsRaw =
      ctx.B.CreateOr(ctx.B.CreateOr(dw3IsZero, dw3IsFormat32),
                     ctx.B.CreateOr(dw3IsFormat32Uint, dw3IsFormat32Float),
                     "srd_dw3_raw_shape");
  Value *rawPointerShape =
      ctx.B.CreateAnd(dw1IsRawBase, dw3IsRaw, "srd_raw_pointer_shape");

  uint32_t dw1Const = 0;
  if (constantI32(dw.dw1, dw1Const) && (dw1Const & 0xFFFF0000u) != 0 &&
      (dw1Const & 0xFFFF0000u) != kGfx1250RawPointerWord1Bits) {
    report_fatal_error("transpiler: MUBUF: unsupported non-raw SRSRC "
                       "descriptor: source word1 contains structured/"
                       "swizzled fields that the raw-buffer lowering cannot "
                       "preserve");
  }
  uint32_t dw3Const = 0;
  if (constantI32(dw.dw3, dw3Const) && dw3Const != 0 &&
      dw3Const != kGfx942RawBufferFormat32 &&
      dw3Const != kGfx942RawBufferFormat32Uint &&
      dw3Const != kGfx942RawBufferFormat32Float) {
    report_fatal_error("transpiler: MUBUF: unsupported non-raw SRSRC "
                       "descriptor: source word3 is not a raw-buffer "
                       "FORMAT_32 descriptor shape");
  }

  Value *cleanDw1 = ctx.B.CreateAnd(dw.dw1,
                                     ConstantInt::get(ctx.i32Ty, 0xFFFF));
  Value *srdW0 = ctx.B.CreateCall(readfirstlane, {dw.dw0}, "srd_w0");
  Value *srdW1 = ctx.B.CreateCall(readfirstlane, {cleanDw1}, "srd_w1");
  Value *sourceMax = ConstantInt::get(ctx.i32Ty, kGfx1250RawBufferMaxRecords);
  Value *targetMax = ConstantInt::get(ctx.i32Ty, kGfx942RawBufferMaxRecords);
  Value *isSourceMax = ctx.B.CreateICmpEQ(dw.dw2, sourceMax, "srd_is_gfx125_max");
  Value *shouldMapMax =
      ctx.B.CreateAnd(isSourceMax, rawPointerShape, "srd_map_gfx125_max");
  Value *mappedDw2 = ctx.B.CreateSelect(shouldMapMax, targetMax, dw.dw2,
                                        "srd_num_records");
  Value *srdW2 =
      ctx.B.CreateCall(readfirstlane, {mappedDw2}, "srd_w2");
  Value *word3 = ConstantInt::get(ctx.i32Ty, kGfx942RawBufferFormat32Float);
  Value *srd = UndefValue::get(FixedVectorType::get(ctx.i32Ty, 4));
  srd = ctx.B.CreateInsertElement(srd, srdW0, static_cast<uint64_t>(0));
  srd = ctx.B.CreateInsertElement(srd, srdW1, static_cast<uint64_t>(1));
  srd = ctx.B.CreateInsertElement(srd, srdW2, static_cast<uint64_t>(2));
  srd = ctx.B.CreateInsertElement(srd, word3, static_cast<uint64_t>(3));
  return srd;
}

} // namespace

MubufAddr decodeMubufAddr(RaiseContext &ctx, const DecodedInst &di,
                          OpResolver &op, bool isStore,
                          StringRef diagLabel) {
  MubufOps m = classifyMubufOps(di, op, isStore);

  if (!m.haveSrsrc) {
    std::string msg;
    raw_string_ostream os(msg);
    os << "transpiler: " << diagLabel << ": no SRSRC found for "
       << di.rawMnemonic;
    report_fatal_error(StringRef(os.str()));
  }

  MubufAddr out;
  if (m.hasSwz) {
    report_fatal_error(Twine("transpiler: ") + diagLabel +
                       ": swizzled buffer addressing is unsupported "
                       "by the raw-buffer lowering");
  }
  SRSRCDwords dw = readSRSRCDwords(ctx, m.srsrc);
  out.srd = buildMubufSRD(ctx, dw);
  out.stData = m.vdata;

  Value *voffset = ConstantInt::get(ctx.i32Ty, 0);
  if (m.haveVaddr)
    voffset = ctx.B.CreateAdd(voffset, ctx.regs.readReg32(ctx.B, m.vaddr));
  if (m.immOff != 0)
    voffset = ctx.B.CreateAdd(
        voffset,
        ConstantInt::get(ctx.i32Ty, static_cast<int32_t>(m.immOff)));
  out.voffset = voffset;

  out.soffset = m.haveSoff ? ctx.regs.readReg32(ctx.B, m.soff)
                           : ConstantInt::get(ctx.i32Ty, 0);
  out.auxFlags = ConstantInt::get(ctx.i32Ty, 0);
  return out;
}

MubufAtomicAddr decodeMubufAtomicAddr(RaiseContext &ctx,
                                       const DecodedInst & /*di*/,
                                       OpResolver &op,
                                       StringRef /*diagLabel*/) {
  // `readReg32` fatal's on out-of-range SGPR indices, so no null-check
  // is needed here — the loads either succeed or never return.
  ParsedReg srsrc = op.srcReg(0);
  Value *dw0 = ctx.regs.readReg32(ctx.B, srsrc);
  ParsedReg srsrc1 = srsrc; srsrc1.baseIdx = srsrc.baseIdx + 1;
  Value *dw1 = ctx.regs.readReg32(ctx.B, srsrc1);
  Value *lo = ctx.B.CreateZExt(dw0, ctx.i64Ty);
  Value *hi = ctx.B.CreateAnd(ctx.B.CreateZExt(dw1, ctx.i64Ty),
                               ConstantInt::get(ctx.i64Ty, 0xFFFF));
  Value *ptrInt = ctx.B.CreateOr(lo, ctx.B.CreateShl(hi, 32), "buf_base");
  MubufAtomicAddr out;
  out.ptr = ctx.B.CreateIntToPtr(ptrInt, PointerType::get(ctx.C, 0));
  return out;
}

} // namespace transpiler
