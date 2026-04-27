#include "handlers.hpp"

#include "semop.hpp"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
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

ParsedReg applyVopdVGPRMsb(const RaiseContext &ctx, ParsedReg pr,
                           unsigned slot) {
  if (pr.kind == ParsedReg::VGPR || pr.kind == ParsedReg::AGPR)
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
  if (isI32)
    v = ctx.B.CreateBitCast(v, ctx.f32Ty);
  if (modifiers & 2)
    v = ctx.B.CreateUnaryIntrinsic(Intrinsic::fabs, v, nullptr, "vopd_abs");
  if (modifiers & 1)
    v = ctx.B.CreateFNeg(v, "vopd_neg");
  if (isI32)
    v = ctx.B.CreateBitCast(v, ctx.i32Ty);
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

bool lowerVopdHalf(RaiseContext &ctx, const DecodedInst &di,
                   const DecodedInst::VopdHalf &half,
                   SmallVectorImpl<std::pair<ParsedReg, Value *>> &writes,
                   HandlerResult &hr) {
  ParsedReg dst = applyVopdVGPRMsb(
      ctx, ctx.parseReg(half.dstReg, AMDGPU::VOPD::Component::DST),
      /*slot=*/3);
  auto queue = [&](Value *v) {
    writes.emplace_back(dst, v);
    return true;
  };

  switch (half.semOp) {
  case SemOp::V_MOV_B32: {
    if (!requireVopdSources(half, 1, di, hr)) return false;
    return queue(readVopdSource(ctx, half.src[0], 0));
  }
  case SemOp::V_CNDMASK_B32: {
    if (!requireVopdSources(half, 2, di, hr)) return false;
    Value *s0 = readVopdSource(ctx, half.src[0], 0);
    Value *s1 = readVopdSource(ctx, half.src[1], 1);
    Value *cond = half.numSrcs >= 3 ? readVopdCond(ctx, di, half.src[2], hr)
                                    : ctx.regs.loadVCC(ctx.B);
    if (!cond) return false;
    return queue(ctx.B.CreateSelect(cond, s1, s0, "vopd_cndmask"));
  }
  case SemOp::V_ADD_F32:
  case SemOp::V_MUL_F32:
  case SemOp::V_SUB_F32:
  case SemOp::V_SUBREV_F32: {
    if (!requireVopdSources(half, 2, di, hr)) return false;
    Value *s0 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[0], 0),
                                    ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[1], 1),
                                    ctx.f32Ty);
    Value *res = nullptr;
    if (half.semOp == SemOp::V_ADD_F32)
      res = ctx.B.CreateFAdd(s0, s1, "vopd_fadd");
    else if (half.semOp == SemOp::V_MUL_F32)
      res = ctx.B.CreateFMul(s0, s1, "vopd_fmul");
    else if (half.semOp == SemOp::V_SUBREV_F32)
      res = ctx.B.CreateFSub(s1, s0, "vopd_fsubrev");
    else
      res = ctx.B.CreateFSub(s0, s1, "vopd_fsub");
    return queue(ctx.B.CreateBitCast(res, ctx.i32Ty));
  }
  case SemOp::V_FMAC_F32: {
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
  case SemOp::V_FMA_F32: {
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
  case SemOp::V_ADD_NC_U32:
  case SemOp::V_SUB_NC_U32:
  case SemOp::V_SUBREV_NC_U32:
  case SemOp::V_LSHLREV_B32:
  case SemOp::V_LSHRREV_B32:
  case SemOp::V_ASHRREV_I32:
  case SemOp::V_AND_B32:
  case SemOp::V_OR_B32:
  case SemOp::V_XOR_B32: {
    if (!requireVopdSources(half, 2, di, hr)) return false;
    Value *s0 = readVopdSource(ctx, half.src[0], 0);
    Value *s1 = readVopdSource(ctx, half.src[1], 1);
    Value *res = nullptr;
    switch (half.semOp) {
    case SemOp::V_ADD_NC_U32:    res = ctx.B.CreateAdd(s0, s1, "vopd_add"); break;
    case SemOp::V_SUB_NC_U32:    res = ctx.B.CreateSub(s0, s1, "vopd_sub"); break;
    case SemOp::V_SUBREV_NC_U32: res = ctx.B.CreateSub(s1, s0, "vopd_subrev"); break;
    case SemOp::V_LSHLREV_B32:   res = ctx.B.CreateShl(s1, s0, "vopd_shl"); break;
    case SemOp::V_LSHRREV_B32:   res = ctx.B.CreateLShr(s1, s0, "vopd_lshr"); break;
    case SemOp::V_ASHRREV_I32:   res = ctx.B.CreateAShr(s1, s0, "vopd_ashr"); break;
    case SemOp::V_AND_B32:       res = ctx.B.CreateAnd(s0, s1, "vopd_and"); break;
    case SemOp::V_OR_B32:        res = ctx.B.CreateOr(s0, s1, "vopd_or"); break;
    case SemOp::V_XOR_B32:       res = ctx.B.CreateXor(s0, s1, "vopd_xor"); break;
    default: llvm_unreachable("filtered by outer switch");
    }
    return queue(res);
  }
  case SemOp::V_MAX_I32:
  case SemOp::V_MIN_I32:
  case SemOp::V_MAX_U32:
  case SemOp::V_MIN_U32: {
    if (!requireVopdSources(half, 2, di, hr)) return false;
    Value *s0 = readVopdSource(ctx, half.src[0], 0);
    Value *s1 = readVopdSource(ctx, half.src[1], 1);
    Intrinsic::ID id = Intrinsic::smax;
    if (half.semOp == SemOp::V_MIN_I32) id = Intrinsic::smin;
    if (half.semOp == SemOp::V_MAX_U32) id = Intrinsic::umax;
    if (half.semOp == SemOp::V_MIN_U32) id = Intrinsic::umin;
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, id, {ctx.i32Ty});
    const char *name = "vopd_smax";
    if (half.semOp == SemOp::V_MIN_I32) name = "vopd_smin";
    if (half.semOp == SemOp::V_MAX_U32) name = "vopd_umax";
    if (half.semOp == SemOp::V_MIN_U32) name = "vopd_umin";
    return queue(ctx.B.CreateCall(fn, {s0, s1}, name));
  }
  case SemOp::V_MAX_F32:
  case SemOp::V_MIN_F32: {
    if (!requireVopdSources(half, 2, di, hr)) return false;
    Value *s0 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[0], 0),
                                    ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(readVopdSource(ctx, half.src[1], 1),
                                    ctx.f32Ty);
    Intrinsic::ID id =
        half.semOp == SemOp::V_MAX_F32 ? Intrinsic::maxnum : Intrinsic::minnum;
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, id, {ctx.f32Ty});
    const char *name =
        half.semOp == SemOp::V_MAX_F32 ? "vopd_fmax" : "vopd_fmin";
    return queue(ctx.B.CreateBitCast(
        ctx.B.CreateCall(fn, {s0, s1}, name), ctx.i32Ty));
  }
  case SemOp::V_BITOP3_B32: {
    if (!requireVopdSources(half, 2, di, hr)) return false;
    if (!half.hasBitOp3) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOPD", "VOPD bitop component missing bitop3 immediate");
      return false;
    }
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
  }
  default:
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOPD", "unhandled structural VOPD component SemOp");
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

  SmallVector<std::pair<ParsedReg, Value *>, 4> pendingVGPRWrites;
  bool xOk = lowerVopdHalf(ctx, di, di.vopd[AMDGPU::VOPD::ComponentIndex::X],
                           pendingVGPRWrites, hr);
  bool yOk = xOk && lowerVopdHalf(
                        ctx, di, di.vopd[AMDGPU::VOPD::ComponentIndex::Y],
                        pendingVGPRWrites, hr);
  if (!xOk || !yOk)
    return hr;

  // VOPD executes as a paired issue packet: both halves read pre-instruction
  // register state. Commit writes only after both halves are decoded/lifted.
  for (const auto &w : pendingVGPRWrites)
    ctx.writeReg32(w.first, w.second);
  hr.handled = true;
  return hr;
}

} // namespace transpiler
