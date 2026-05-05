#include "handlers.hpp"

#include "canonical_op.hpp"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

using namespace llvm;

namespace transpiler {

namespace {

struct PendingVopdWrite {
  ParsedReg dst;
  Value *value = nullptr;
};

ParsedReg applyVopdVGPRMsb(const RaiseContext &ctx, ParsedReg pr,
                           unsigned slot) {
  if (pr.kind != ParsedReg::VGPR && pr.kind != ParsedReg::AGPR)
    return pr;

  pr.baseIdx += ((ctx.vgprMSBs >> (slot * 2)) & 0x3) * 256;
  return pr;
}

Value *readVopdVCCAsSource(RaiseContext &ctx) {
  if (ctx.projection.sourceWaveScopedLaneOps()) {
    Value *mask = ctx.regs.readVCCAsWaveMask(ctx.B, ctx.regs.execTy);
    Value *lo = ctx.B.CreateTrunc(mask, ctx.i32Ty, "vopd_vcc_lo_src");
    Value *hi = ctx.B.CreateTrunc(
        ctx.B.CreateLShr(mask, ctx.isa.waveSize), ctx.i32Ty,
        "vopd_vcc_hi_src");
    Value *lane = ctx.projection.emitLaneIdx(ctx.B);
    Value *upper = ctx.B.CreateICmpUGE(
        lane, ConstantInt::get(ctx.i32Ty, ctx.isa.waveSize),
        "vopd_vcc_upper_src_wave");
    return ctx.B.CreateSelect(upper, hi, lo, "vopd_vcc_src_wave_mask");
  }
  return ctx.regs.readVCCAsWaveMask(ctx.B, ctx.i32Ty);
}

Value *applyVopdSourceModifiers(RaiseContext &ctx, Value *v,
                                uint8_t modifiers) {
  if (modifiers == 0)
    return v;
  bool isI32 = v->getType() == ctx.i32Ty;
  bool isI64 = v->getType() == ctx.i64Ty;
  if (isI32)
    v = ctx.B.CreateBitCast(v, ctx.f32Ty);
  if (isI64)
    v = ctx.B.CreateBitCast(v, Type::getDoubleTy(ctx.C));
  if (modifiers & 2)
    v = ctx.B.CreateUnaryIntrinsic(Intrinsic::fabs, v, nullptr, "vopd_abs");
  if (modifiers & 1)
    v = ctx.B.CreateFNeg(v, "vopd_neg");
  if (isI32)
    v = ctx.B.CreateBitCast(v, ctx.i32Ty);
  if (isI64)
    v = ctx.B.CreateBitCast(v, ctx.i64Ty);
  return v;
}

Value *readVopdSource(RaiseContext &ctx, const DecodedInst::VopdSource &src,
                      unsigned srcSlot) {
  Value *v = nullptr;
  auto parsed = [&](ParsedReg::Kind kind) {
    ParsedReg pr;
    pr.kind = kind;
    pr.baseIdx = src.baseIdx;
    pr.width = src.width;
    return applyVopdVGPRMsb(ctx, pr, srcSlot);
  };
  switch (src.kind) {
  case DecodedInst::VopdSource::Kind::None:
    return nullptr;
  case DecodedInst::VopdSource::Kind::Imm:
    v = ConstantInt::get(ctx.i32Ty,
                         static_cast<uint32_t>(src.imm & 0xffffffffu));
    break;
  case DecodedInst::VopdSource::Kind::VGPR:
    v = ctx.regs.readReg32(ctx.B, parsed(ParsedReg::VGPR));
    break;
  case DecodedInst::VopdSource::Kind::AGPR:
    v = ctx.regs.readReg32(ctx.B, parsed(ParsedReg::AGPR));
    break;
  case DecodedInst::VopdSource::Kind::SGPR:
    v = ctx.regs.loadSGPR32(ctx.B, src.baseIdx);
    break;
  case DecodedInst::VopdSource::Kind::TTMP:
    v = ctx.B.CreateLoad(ctx.i32Ty, ctx.regs.ttmp[src.baseIdx], "vopd_ttmp");
    break;
  case DecodedInst::VopdSource::Kind::VCC:
    v = readVopdVCCAsSource(ctx);
    break;
  case DecodedInst::VopdSource::Kind::EXEC: {
    ParsedReg pr;
    pr.kind = ParsedReg::EXEC;
    pr.baseIdx = src.baseIdx;
    pr.width = src.width;
    v = ctx.regs.readReg32(ctx.B, pr);
    break;
  }
  case DecodedInst::VopdSource::Kind::SCC:
    v = ctx.B.CreateZExt(ctx.regs.loadSCC(ctx.B), ctx.i32Ty);
    break;
  case DecodedInst::VopdSource::Kind::M0:
    v = ctx.B.CreateLoad(ctx.i32Ty, ctx.regs.m0, "vopd_m0");
    break;
  }
  return applyVopdSourceModifiers(ctx, v, src.modifiers);
}

Value *readVopdSource64(RaiseContext &ctx, const DecodedInst::VopdSource &src,
                        unsigned srcSlot, const DecodedInst &di,
                        HandlerResult &hr) {
  Value *v = nullptr;
  auto parsed = [&](ParsedReg::Kind kind) {
    ParsedReg pr;
    pr.kind = kind;
    pr.baseIdx = src.baseIdx;
    pr.width = src.width;
    return applyVopdVGPRMsb(ctx, pr, srcSlot);
  };

  switch (src.kind) {
  case DecodedInst::VopdSource::Kind::None:
    return nullptr;
  case DecodedInst::VopdSource::Kind::Imm:
    v = ConstantInt::get(ctx.i64Ty, static_cast<uint64_t>(src.imm));
    break;
  case DecodedInst::VopdSource::Kind::VGPR:
    v = ctx.regs.readReg64(ctx.B, parsed(ParsedReg::VGPR));
    break;
  case DecodedInst::VopdSource::Kind::AGPR:
    v = ctx.regs.readReg64(ctx.B, parsed(ParsedReg::AGPR));
    break;
  case DecodedInst::VopdSource::Kind::SGPR:
    v = ctx.regs.loadSGPR64(ctx.B, src.baseIdx);
    break;
  default:
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOPD", "VOPD f64 component source is not a 64-bit scalar/vector source");
    return nullptr;
  }

  return applyVopdSourceModifiers(ctx, v, src.modifiers);
}

Value *readVopdCond(RaiseContext &ctx, const DecodedInst &di,
                    const DecodedInst::VopdSource &src, HandlerResult &hr) {
  if (src.kind != DecodedInst::VopdSource::Kind::VCC &&
      src.kind != DecodedInst::VopdSource::Kind::SGPR) {
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOPD", "VOPD cndmask explicit condition is neither VCC nor SGPR");
    return nullptr;
  }
  if (src.kind == DecodedInst::VopdSource::Kind::VCC)
    return ctx.regs.loadVCC(ctx.B);

  if (Value *freshCmp = ctx.lookupSgprWaveMaskI1(src.baseIdx))
    return freshCmp;

  Value *condVal = ctx.isa.isWave32()
                       ? ctx.regs.loadSGPR32(ctx.B, src.baseIdx)
                       : ctx.regs.loadSGPR64(ctx.B, src.baseIdx);
  Value *fallback = ctx.projection.extractLaneBitFromWaveMask(ctx.B, condVal);
  if (Value *shadowValid = ctx.loadSgprWaveMaskValid(src.baseIdx)) {
    Value *shadowExec = ctx.loadSgprWaveMaskExec(src.baseIdx);
    Value *shadowI1 = ctx.projection.extractLaneBitFromWaveMask(ctx.B,
                                                                 shadowExec);
    return ctx.B.CreateSelect(shadowValid, shadowI1, fallback,
                              "vopd_sgpr_mask_shadow_sel");
  }
  return fallback;
}

bool requireVopdSources(const DecodedInst::VopdHalf &half, unsigned n,
                        const DecodedInst &di, HandlerResult &hr) {
  if (half.numSrcs >= n)
    return true;
  hr.failure = RaiseFailure::unsupportedShape(
      di, "VOPD", "VOPD component has too few decoded sources");
  return false;
}

bool requireVopdRegWidth(const DecodedInst &di, const char *what,
                         unsigned width, unsigned minWidth,
                         HandlerResult &hr) {
  if (width >= minWidth)
    return true;
  hr.failure = RaiseFailure::unsupportedShape(
      di, "VOPD", (Twine("VOPD ") + what + " is narrower than " +
                   Twine(minWidth) + " dwords")
                      .str());
  return false;
}

bool lowerVopdHalf(RaiseContext &ctx, const DecodedInst &di,
                   const DecodedInst::VopdHalf &half,
                   SmallVectorImpl<PendingVopdWrite> &writes,
                   HandlerResult &hr) {
  ParsedReg dst = applyVopdVGPRMsb(
      ctx, ctx.parseReg(half.dstReg, /*mciOpIdx=*/-1),
      /*slot=*/3);
  auto queue = [&](Value *v) {
    // VOPD destination operands name the low VGPR slot even for 64-bit
    // components. A 64-bit commit writes [baseIdx, baseIdx+1] through the
    // register file helper below, so dst.width is not a reliable arity check.
    writes.push_back(PendingVopdWrite{dst, v});
    return true;
  };
  auto lowerBitOp3 = [&]() {
    if (!requireVopdSources(half, 2, di, hr)) return false;
    Value *a = readVopdSource(ctx, half.src[0], 0);
    Value *b = readVopdSource(ctx, half.src[1], 1);
    Value *c = ConstantInt::get(ctx.i32Ty, 0);
    Value *na = ctx.B.CreateNot(a);
    Value *nb = ctx.B.CreateNot(b);
    Value *nc = ctx.B.CreateNot(c);
    Value *minterms[8] = {
        ctx.B.CreateAnd(ctx.B.CreateAnd(na, nb), nc),
        ctx.B.CreateAnd(ctx.B.CreateAnd(na, nb), c),
        ctx.B.CreateAnd(ctx.B.CreateAnd(na, b), nc),
        ctx.B.CreateAnd(ctx.B.CreateAnd(na, b), c),
        ctx.B.CreateAnd(ctx.B.CreateAnd(a, nb), nc),
        ctx.B.CreateAnd(ctx.B.CreateAnd(a, nb), c),
        ctx.B.CreateAnd(ctx.B.CreateAnd(a, b), nc),
        ctx.B.CreateAnd(ctx.B.CreateAnd(a, b), c),
    };
    Value *result = ConstantInt::get(ctx.i32Ty, 0);
    for (int i = 0; i < 8; ++i)
      if (half.bitOp3 & (1u << i))
        result = ctx.B.CreateOr(result, minterms[i]);
    return queue(result);
  };

  // `v_dual_bitop2_b32` components carry an 8-bit `bitop3` truth table even
  // though LLVM's canonical component opcode may look like a simple V_AND /
  // V_OR / V_XOR. The immediate is semantic, not decoration: Triton's sort
  // uses values such as 0x14 (xor) and 0x40 (and) in this encoding.
  if (half.hasBitOp3)
    return lowerBitOp3();

  switch (half.canonOp) {
  case CanonicalOp::V_MOV_B32: {
    if (!requireVopdSources(half, 1, di, hr)) return false;
    return queue(readVopdSource(ctx, half.src[0], 0));
  }
  case CanonicalOp::V_CNDMASK_B32: {
    if (!requireVopdSources(half, 2, di, hr)) return false;
    Value *s0 = readVopdSource(ctx, half.src[0], 0);
    Value *s1 = readVopdSource(ctx, half.src[1], 1);
    Value *cond = half.numSrcs >= 3 ? readVopdCond(ctx, di, half.src[2], hr)
                                    : ctx.regs.loadVCC(ctx.B);
    if (!cond) return false;
    return queue(ctx.B.CreateSelect(cond, s1, s0, "vopd_cndmask"));
  }
  case CanonicalOp::V_ADD_F32:
  case CanonicalOp::V_MUL_F32:
  case CanonicalOp::V_SUB_F32:
  case CanonicalOp::V_SUBREV_F32: {
    if (!requireVopdSources(half, 2, di, hr)) return false;
    Value *s0 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[0], 0),
                                    ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[1], 1),
                                    ctx.f32Ty);
    Value *res = nullptr;
    if (half.canonOp == CanonicalOp::V_ADD_F32)
      res = ctx.B.CreateFAdd(s0, s1, "vopd_fadd");
    else if (half.canonOp == CanonicalOp::V_MUL_F32)
      res = ctx.B.CreateFMul(s0, s1, "vopd_fmul");
    else if (half.canonOp == CanonicalOp::V_SUBREV_F32)
      res = ctx.B.CreateFSub(s1, s0, "vopd_fsubrev");
    else
      res = ctx.B.CreateFSub(s0, s1, "vopd_fsub");
    return queue(ctx.B.CreateBitCast(res, ctx.i32Ty));
  }
  case CanonicalOp::V_FMAC_F32: {
    if (!requireVopdSources(half, 2, di, hr)) return false;
    Value *s0 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[0], 0),
                                    ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[1], 1),
                                    ctx.f32Ty);
    Value *acc = ctx.B.CreateBitCast(ctx.regs.readReg32(ctx.B, dst),
                                     ctx.f32Ty);
    Function *fmuladd =
        Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fmuladd,
                                          {ctx.f32Ty});
    return queue(ctx.B.CreateBitCast(
        ctx.B.CreateCall(fmuladd, {s0, s1, acc}, "vopd_fmac"), ctx.i32Ty));
  }
  case CanonicalOp::V_FMA_F32: {
    if (!requireVopdSources(half, 3, di, hr)) return false;
    Value *s0 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[0], 0),
                                    ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[1], 1),
                                    ctx.f32Ty);
    Value *s2 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[2], 2),
                                    ctx.f32Ty);
    Function *fma =
        Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma,
                                          {ctx.f32Ty});
    return queue(ctx.B.CreateBitCast(
        ctx.B.CreateCall(fma, {s0, s1, s2}, "vopd_fma"), ctx.i32Ty));
  }
  case CanonicalOp::V_MUL_F64:
  case CanonicalOp::V_ADD_F64:
  case CanonicalOp::V_MAX_NUM_F64:
  case CanonicalOp::V_MIN_NUM_F64:
  case CanonicalOp::V_FMA_F64: {
    unsigned numSrcs = half.canonOp == CanonicalOp::V_FMA_F64 ? 3 : 2;
    if (!requireVopdSources(half, numSrcs, di, hr)) return false;
    for (unsigned i = 0; i < numSrcs; ++i) {
      if (half.src[i].kind != DecodedInst::VopdSource::Kind::Imm &&
          !requireVopdRegWidth(di, "f64 source", half.src[i].width, 2, hr))
        return false;
    }
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    Value *s0 = readVopdSource64(ctx, half.src[0], 0, di, hr);
    if (!s0) return false;
    Value *s1 = readVopdSource64(ctx, half.src[1], 1, di, hr);
    if (!s1) return false;
    s0 = ctx.B.CreateBitCast(s0, f64Ty);
    s1 = ctx.B.CreateBitCast(s1, f64Ty);

    Value *res = nullptr;
    if (half.canonOp == CanonicalOp::V_MUL_F64) {
      res = ctx.B.CreateFMul(s0, s1, "vopd_fmul_f64");
    } else if (half.canonOp == CanonicalOp::V_ADD_F64) {
      res = ctx.B.CreateFAdd(s0, s1, "vopd_fadd_f64");
    } else if (half.canonOp == CanonicalOp::V_MAX_NUM_F64 ||
               half.canonOp == CanonicalOp::V_MIN_NUM_F64) {
      Intrinsic::ID id = half.canonOp == CanonicalOp::V_MAX_NUM_F64
                             ? Intrinsic::maxnum
                             : Intrinsic::minnum;
      Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, id, {f64Ty});
      const char *name = half.canonOp == CanonicalOp::V_MAX_NUM_F64
                             ? "vopd_fmaxnum_f64"
                             : "vopd_fminnum_f64";
      res = ctx.B.CreateCall(fn, {s0, s1}, name);
    } else {
      Value *s2 = readVopdSource64(ctx, half.src[2], 2, di, hr);
      if (!s2) return false;
      s2 = ctx.B.CreateBitCast(s2, f64Ty);
      Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma,
                                                        {f64Ty});
      res = ctx.B.CreateCall(fma, {s0, s1, s2}, "vopd_fma_f64");
    }
    return queue(ctx.B.CreateBitCast(res, ctx.i64Ty));
  }
  case CanonicalOp::V_FMAMK_F32:
  case CanonicalOp::V_FMAAK_F32: {
    if (!requireVopdSources(half, 3, di, hr)) return false;
    Value *s0 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[0], 0),
                                    ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[1], 1),
                                    ctx.f32Ty);
    // MADK VOPD encodings have only src0/vsrc1 register fields; the mandatory
    // literal occupies a logical source slot but not a VGPR-MSB slot.
    unsigned s2Slot = half.canonOp == CanonicalOp::V_FMAMK_F32 ? 1 : 2;
    Value *s2 =
        ctx.B.CreateBitCast(readVopdSource(ctx, half.src[2], s2Slot),
                            ctx.f32Ty);
    Function *fma =
        Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma,
                                          {ctx.f32Ty});
    const char *name =
        half.canonOp == CanonicalOp::V_FMAMK_F32 ? "vopd_fmamk" : "vopd_fmaak";
    return queue(ctx.B.CreateBitCast(
        ctx.B.CreateCall(fma, {s0, s1, s2}, name), ctx.i32Ty));
  }
  case CanonicalOp::V_ADD_NC_U32:
  case CanonicalOp::V_SUB_NC_U32:
  case CanonicalOp::V_SUBREV_NC_U32:
  case CanonicalOp::V_LSHLREV_B32:
  case CanonicalOp::V_LSHRREV_B32:
  case CanonicalOp::V_ASHRREV_I32:
  case CanonicalOp::V_AND_B32:
  case CanonicalOp::V_OR_B32:
  case CanonicalOp::V_XOR_B32: {
    if (!requireVopdSources(half, 2, di, hr)) return false;
    Value *s0 = readVopdSource(ctx, half.src[0], 0);
    Value *s1 = readVopdSource(ctx, half.src[1], 1);
    Value *res = nullptr;
    switch (half.canonOp) {
    case CanonicalOp::V_ADD_NC_U32:    res = ctx.B.CreateAdd(s0, s1, "vopd_add"); break;
    case CanonicalOp::V_SUB_NC_U32:    res = ctx.B.CreateSub(s0, s1, "vopd_sub"); break;
    case CanonicalOp::V_SUBREV_NC_U32: res = ctx.B.CreateSub(s1, s0, "vopd_subrev"); break;
    case CanonicalOp::V_LSHLREV_B32:   res = ctx.B.CreateShl(s1, s0, "vopd_shl"); break;
    case CanonicalOp::V_LSHRREV_B32:   res = ctx.B.CreateLShr(s1, s0, "vopd_lshr"); break;
    case CanonicalOp::V_ASHRREV_I32:   res = ctx.B.CreateAShr(s1, s0, "vopd_ashr"); break;
    case CanonicalOp::V_AND_B32:       res = ctx.B.CreateAnd(s0, s1, "vopd_and"); break;
    case CanonicalOp::V_OR_B32:        res = ctx.B.CreateOr(s0, s1, "vopd_or"); break;
    case CanonicalOp::V_XOR_B32:       res = ctx.B.CreateXor(s0, s1, "vopd_xor"); break;
    default: llvm_unreachable("filtered by outer switch");
    }
    return queue(res);
  }
  case CanonicalOp::V_MAX_I32:
  case CanonicalOp::V_MIN_I32:
  case CanonicalOp::V_MAX_U32:
  case CanonicalOp::V_MIN_U32: {
    if (!requireVopdSources(half, 2, di, hr)) return false;
    Value *s0 = readVopdSource(ctx, half.src[0], 0);
    Value *s1 = readVopdSource(ctx, half.src[1], 1);
    Intrinsic::ID id = Intrinsic::smax;
    if (half.canonOp == CanonicalOp::V_MIN_I32) id = Intrinsic::smin;
    if (half.canonOp == CanonicalOp::V_MAX_U32) id = Intrinsic::umax;
    if (half.canonOp == CanonicalOp::V_MIN_U32) id = Intrinsic::umin;
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, id, {ctx.i32Ty});
    const char *name = "vopd_smax";
    if (half.canonOp == CanonicalOp::V_MIN_I32) name = "vopd_smin";
    if (half.canonOp == CanonicalOp::V_MAX_U32) name = "vopd_umax";
    if (half.canonOp == CanonicalOp::V_MIN_U32) name = "vopd_umin";
    return queue(ctx.B.CreateCall(fn, {s0, s1}, name));
  }
  case CanonicalOp::V_MAX_F32:
  case CanonicalOp::V_MIN_F32: {
    if (!requireVopdSources(half, 2, di, hr)) return false;
    Value *s0 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[0], 0),
                                    ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[1], 1),
                                    ctx.f32Ty);
    Intrinsic::ID id =
        half.canonOp == CanonicalOp::V_MAX_F32 ? Intrinsic::maxnum : Intrinsic::minnum;
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, id, {ctx.f32Ty});
    const char *name =
        half.canonOp == CanonicalOp::V_MAX_F32 ? "vopd_fmax" : "vopd_fmin";
    return queue(ctx.B.CreateBitCast(
        ctx.B.CreateCall(fn, {s0, s1}, name), ctx.i32Ty));
  }
  case CanonicalOp::V_BITOP3_B32: {
    if (!half.hasBitOp3) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOPD", "VOPD bitop component missing bitop3 immediate");
      return false;
    }
    return lowerBitOp3();
  }
  default:
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOPD", "unhandled structural VOPD component CanonicalOp");
    return false;
  }
}

} // namespace

HandlerResult handleVOPD(RaiseContext &ctx, const DecodedInst &di,
                        OpResolver &op) {
  HandlerResult hr;
  (void)op;
  if (!di.hasVopd) {
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOPD", "VOPD instruction reached handler without sidecar");
    return hr;
  }

  SmallVector<PendingVopdWrite, 4> pendingVGPRWrites;
  bool xOk = lowerVopdHalf(ctx, di, di.vopd[AMDGPU::VOPD::ComponentIndex::X],
                           pendingVGPRWrites, hr);
  bool yOk = xOk && lowerVopdHalf(
                        ctx, di, di.vopd[AMDGPU::VOPD::ComponentIndex::Y],
                        pendingVGPRWrites, hr);
  if (!xOk || !yOk)
    return hr;

  // VOPD executes as a paired issue packet: both halves read pre-instruction
  // register state. Commit writes only after both halves are decoded/lifted.
  for (const auto &w : pendingVGPRWrites) {
    if (w.value->getType()->getPrimitiveSizeInBits() == 64)
      ctx.writeReg64(w.dst, w.value);
    else
      ctx.writeReg32(w.dst, w.value);
  }
  hr.handled = true;
  return hr;
}

} // namespace transpiler
