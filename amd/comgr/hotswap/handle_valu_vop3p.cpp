#include "handle_valu_internal.hpp"

#include "canonical_op.hpp"
#include "wmma_lowering.hpp"

#include "Utils/AMDGPUBaseInfo.h" // AMDGPU::getNamedOperandIdx, AMDGPU::OpName
#include "SIDefines.h"            // SISrcMods::NEG
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

#include <climits>

using namespace llvm;

namespace transpiler {

namespace {

struct PackedSrcOptions {
  // Register operands in the packed-f32 family are VGPR pairs that should be
  // read as `<2 x elem>` directly. Packed-f16/i16 operands are one i32 VGPR
  // whose low/high halves are bitcast to `<2 x elem>`.
  bool RegisterSourceIsVector = false;
  // Packed-f32 immediates are scalar 32-bit literals broadcast to both lanes.
  // Packed-f16/i16 immediates are raw packed i32 payloads decoded by LLVM MC.
  bool ImmediateIsScalarBroadcast = false;
  // Floating-point packed families use NEG / NEG_HI as per-lane fneg bits.
  // Integer packed families reject those bits before calling the helper.
  bool ApplyFloatNeg = false;
  // IRBuilder base name used for temporary values from this source family.
  const char *Name = "pk_src";
};

StringRef diagnosticMnemonic(const DecodedInst &di) {
  return di.mnemonic.empty() ? StringRef(canonicalOpName(di.canonOp))
                             : StringRef(di.mnemonic);
}

bool readSourceMods(const DecodedInst &di, OpResolver &op, unsigned numSrcs,
                    unsigned allowedMods, unsigned mods[3],
                    HandlerResult &hr) {
  StringRef instrName = diagnosticMnemonic(di);
  if (op.nSrcs() < numSrcs) {
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOP3P", (instrName + " requires more source operands").str());
    return false;
  }

  for (unsigned i = 0; i < numSrcs; ++i) {
    unsigned modIdx = di.modMap[i];
    if (modIdx == UINT_MAX || !di.isImm(modIdx)) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          (instrName + " missing immediate srcN_modifiers operand").str());
      return false;
    }
    mods[i] = static_cast<unsigned>(di.getImm(modIdx));
    if ((mods[i] & ~allowedMods) != 0) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          (instrName + " has unsupported srcN_modifiers bits").str());
      return false;
    }
  }
  return true;
}

bool readPackedSrcMods(const DecodedInst &di, OpResolver &op, unsigned numSrcs,
                       unsigned allowedMods, unsigned mods[3],
                       HandlerResult &hr) {
  if (!readSourceMods(di, op, numSrcs, allowedMods, mods, hr))
    return false;

  StringRef instrName = diagnosticMnemonic(di);
  for (unsigned i = 0; i < numSrcs; ++i) {
    unsigned srcIdx = op.srcIdx(i);
    if (!di.isReg(srcIdx) && !di.isImm(srcIdx)) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          (instrName + " source is neither a register nor an immediate").str());
      return false;
    }
  }
  return true;
}

Value *readPacked2Src(RaiseContext &ctx, OpResolver &op, unsigned i,
                      Type *elemTy, unsigned mods,
                      const PackedSrcOptions &opts) {
  auto *vecTy = FixedVectorType::get(elemTy, 2);
  Value *natLo = nullptr;
  Value *natHi = nullptr;

  if (opts.RegisterSourceIsVector && op.isSrcReg(i)) {
    Value *vec = ctx.regs.readRegVec(ctx.B, op.srcReg(i), vecTy);
    natLo = ctx.B.CreateExtractElement(vec, static_cast<uint64_t>(0));
    natHi = ctx.B.CreateExtractElement(vec, static_cast<uint64_t>(1));
  } else if (opts.ImmediateIsScalarBroadcast && !op.isSrcReg(i)) {
    Value *scalar = ctx.B.CreateBitCast(op.src(i), elemTy);
    natLo = scalar;
    natHi = scalar;
  } else {
    Value *raw = op.src(i);
    if (raw->getType() != ctx.i32Ty)
      raw = ctx.B.CreateBitCast(raw, ctx.i32Ty);
    Value *vec = ctx.B.CreateBitCast(raw, vecTy, opts.Name);
    natLo = ctx.B.CreateExtractElement(vec, static_cast<uint64_t>(0));
    natHi = ctx.B.CreateExtractElement(vec, static_cast<uint64_t>(1));
  }

  Value *lo = (mods & SISrcMods::OP_SEL_0) ? natHi : natLo;
  Value *hi = (mods & SISrcMods::OP_SEL_1) ? natHi : natLo;

  if (opts.ApplyFloatNeg) {
    if (mods & SISrcMods::NEG)
      lo = ctx.B.CreateFNeg(lo, (Twine(opts.Name) + "_neg_lo").str());
    if (mods & SISrcMods::NEG_HI)
      hi = ctx.B.CreateFNeg(hi, (Twine(opts.Name) + "_neg_hi").str());
  }

  Value *r = UndefValue::get(vecTy);
  r = ctx.B.CreateInsertElement(r, lo, static_cast<uint64_t>(0));
  r = ctx.B.CreateInsertElement(r, hi, static_cast<uint64_t>(1));
  return r;
}

Value *applyF32InputMods(RaiseContext &ctx, Value *v, unsigned mods,
                         const Twine &name) {
  if (v->getType() != ctx.f32Ty)
    v = ctx.B.CreateBitCast(v, ctx.f32Ty);
  if (mods & SISrcMods::ABS)
    v = ctx.B.CreateUnaryIntrinsic(Intrinsic::fabs, v, nullptr,
                                   (name + "_abs").str());
  if (mods & SISrcMods::NEG)
    v = ctx.B.CreateFNeg(v, (name + "_neg").str());
  return v;
}

Value *readMixF32Src(RaiseContext &ctx, OpResolver &op, unsigned i,
                     Type *narrowTy, unsigned mods, StringRef cvtName) {
  Value *raw = op.src(i);
  if ((mods & SISrcMods::OP_SEL_1) == 0)
    return applyF32InputMods(ctx, raw, mods, "mix_full");

  if (raw->getType() == ctx.f32Ty)
    raw = ctx.B.CreateBitCast(raw, ctx.i32Ty);

  Value *bits = nullptr;
  bool isImmediateOperand = !op.isSrcReg(i);
  if (!isImmediateOperand && (mods & SISrcMods::OP_SEL_0))
    bits = ctx.B.CreateTrunc(ctx.B.CreateLShr(raw, 16),
                              Type::getInt16Ty(ctx.C));
  else
    bits = ctx.B.CreateTrunc(raw, Type::getInt16Ty(ctx.C));

  Value *narrowVal = ctx.B.CreateBitCast(bits, narrowTy);
  Value *extended = ctx.B.CreateFPExt(narrowVal, ctx.f32Ty, cvtName);
  return applyF32InputMods(ctx, extended, mods, cvtName);
}

// Read the C (accumulator) operand of a WMMA instruction, handling the
// three encoding shapes LLVM's AMDGPU backend emits:
//
//   * _twoaddr form: C is tied to D (same VGPR slot, no separate `src2`
//     operand on the disassembled line). `op.isSrcReg(2)` is TRUE and
//     `srcReg(2)` returns the D VGPR — we read the live VGPR value.
//   * _threeaddr form with a VGPR C: `isSrcReg(2)` TRUE and `srcReg(2)`
//     returns the explicit C VGPR. Same path as twoaddr — just a
//     different VGPR index.
//   * _threeaddr form with an inline-constant C: LLVM picks this
//     encoding whenever the accumulator source is a constant that fits
//     in the VOP3P src2 inline-constant table (the important case is
//     `C = 0`, which Clang emits for every fresh accumulator built from
//     a zero-initialised `v8f c = {0, ..., 0}`). Here `isSrcReg(2)` is
//     FALSE; we MUST materialise the inline constant directly.
//
// The previous fallback `srcC = dest` was silently wrong for the third
// case: reading the D VGPR before the WMMA writes to it surfaces
// whatever stale (or undef) bits happened to be in those 8 VGPR slots,
// which on a cold kernel is typically zero by accident for the first
// WMMA in a wave but nondeterministic for any subsequent WMMA whose
// D range was never explicitly zero-initialised by the SGPR/VGPR
// prologue. In the `wmma_parallel{2,4,16}` probes the second and
// later WMMAs land on fresh D VGPRs (v[24:31], v[32:39], ...) that
// the compiler skipped zeroing — precisely because it knew the
// threeaddr-imm-0 encoding would satisfy C.
//
// We handle only inline constant `0` today: it is the only src2 inline
// the AMDGPU backend actually emits for the WMMA family (Clang folds
// non-zero accumulator constants through a VGPR mov before the WMMA).
// Any other immediate surfaces as a structured `unsupportedShape`
// failure rather than silently miscompiling.
//
// On failure the helper populates `hr.failure` and returns nullptr; the
// caller must short-circuit.
llvm::Value *readWMMAAccumC(RaiseContext &ctx, const DecodedInst &di,
                             OpResolver &op, const ParsedReg &dest,
                             llvm::Type *cdIRTy, HandlerResult &hr) {
  if (op.nSrcs() < 3) {
    // No src2 operand on the instruction at all (e.g. a hypothetical
    // encoding with C implicitly zero and no disassembler-surfaced
    // slot). Safest to refuse — the caller expects to have read C.
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOP3P",
        "WMMA instruction has no src2 (accumulator) operand; "
        "cannot recover C input");
    return nullptr;
  }
  if (op.isSrcReg(2)) {
    ParsedReg srcC = op.srcReg(2);
    return ctx.regs.readRegVec(ctx.B, srcC, cdIRTy);
  }
  // Inline-constant src2. Today we only model `0`.
  unsigned srcIdx2 = op.srcIdx(2);
  if (!di.isImm(srcIdx2)) {
    // Could be a symbolic constant slot (e.g. SRC_EXEC_LO/HI, SRC_PC).
    // None of those are valid semantics for a WMMA accumulator; refuse.
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOP3P",
        "WMMA src2 is neither a register nor an immediate; no "
        "accumulator C input path is defined for this encoding");
    return nullptr;
  }
  int64_t immC = di.getImm(srcIdx2);
  if (immC == 0)
    return llvm::ConstantAggregateZero::get(cdIRTy);
  hr.failure = RaiseFailure::unsupportedShape(
      di, "VOP3P",
      "WMMA src2 inline-constant other than 0 is not yet modelled; "
      "extend readWMMAAccumC if a corpus kernel surfaces this");
  (void)dest;
  return nullptr;
}

} // namespace

HandlerResult handleVALU_VOP3P(RaiseContext &ctx, const DecodedInst &di,
                                OpResolver &op) {
  HandlerResult hr;
  CanonicalOp sop = di.canonOp;
  StringRef mn(di.mnemonic);

  switch (sop) {
  // ---- VOP3P packed ops (2x fp32 in 2 dwords) ----
  // Handle op_sel_hi, neg_lo, neg_hi modifiers.
  case CanonicalOp::V_PK_MOV_B32: {
    ctx.writeReg64(op.dst(), op.src64(0));
    hr.handled = true;
    return hr;
  }
  case CanonicalOp::V_PK_FMA_F16: {
    constexpr unsigned KnownPkF16Mods =
        SISrcMods::NEG | SISrcMods::NEG_HI | SISrcMods::OP_SEL_0 |
        SISrcMods::OP_SEL_1;
    unsigned mods[3] = {};
    if (!readPackedSrcMods(di, op, 3, KnownPkF16Mods, mods, hr))
      return hr;

    int clampIdx = AMDGPU::getNamedOperandIdx(di.inst.getOpcode(),
                                              AMDGPU::OpName::clamp);
    if (clampIdx < 0 || !di.isImm(static_cast<unsigned>(clampIdx))) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P", "v_pk_fma_f16 missing immediate clamp operand");
      return hr;
    }
    int64_t clampImm = di.getImm(static_cast<unsigned>(clampIdx));
    if (clampImm != 0 && clampImm != 1) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P", "v_pk_fma_f16 clamp operand is not 0 or 1");
      return hr;
    }

    auto *v2f16 = FixedVectorType::get(ctx.f16Ty, 2);
    PackedSrcOptions opts;
    opts.ApplyFloatNeg = true;
    opts.Name = "pk_f16_src";
    // VSrc_v2f16 immediates are decoded by LLVM MC as the raw 32-bit
    // packed source bits. Scalar f16 inline constants occupy the low half;
    // OP_SEL_1 controls whether the high result lane also reads that low
    // half, matching LLVM's own v_pk_fma_f16 patterns.
    Value *s0 = readPacked2Src(ctx, op, 0, ctx.f16Ty, mods[0], opts);
    Value *s1 = readPacked2Src(ctx, op, 1, ctx.f16Ty, mods[1], opts);
    Value *s2 = readPacked2Src(ctx, op, 2, ctx.f16Ty, mods[2], opts);
    Function *fmaFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::fma, {v2f16});
    Value *res = ctx.B.CreateCall(fmaFn, {s0, s1, s2}, "pk_fma_f16");

    if (clampImm != 0) {
      Function *maxFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::maxnum, {v2f16});
      Function *minFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::minnum, {v2f16});
      // AMDGPU clamp is [0, 1] after the arithmetic result; maxnum/minnum
      // gives the target-independent IR shape used elsewhere in Hotswap.
      Value *zero = ConstantVector::getSplat(
          ElementCount::getFixed(2), ConstantFP::get(ctx.f16Ty, 0.0));
      Value *one = ConstantVector::getSplat(
          ElementCount::getFixed(2), ConstantFP::get(ctx.f16Ty, 1.0));
      res = ctx.B.CreateCall(maxFn, {res, zero}, "pk_fma_f16_clamp_lo");
      res = ctx.B.CreateCall(minFn, {res, one}, "pk_fma_f16_clamp");
    }

    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(res, ctx.i32Ty, "pk_fma_f16_pack"));
    hr.handled = true;
    return hr;
  }
  case CanonicalOp::V_PK_ADD_F32:
  case CanonicalOp::V_PK_MUL_F32:
  case CanonicalOp::V_PK_FMA_F32:
  case CanonicalOp::V_PK_MAX_F32:
  case CanonicalOp::V_PK_MIN_F32: {
    auto *v2f32 = FixedVectorType::get(ctx.f32Ty, 2);

    constexpr unsigned KnownPkF32Mods =
        SISrcMods::NEG | SISrcMods::NEG_HI | SISrcMods::OP_SEL_0 |
        SISrcMods::OP_SEL_1;
    unsigned mods[3] = {};
    unsigned numSrcs = (sop == CanonicalOp::V_PK_FMA_F32) ? 3 : 2;
    if (!readPackedSrcMods(di, op, numSrcs, KnownPkF32Mods, mods, hr))
      return hr;

    // Read each source as <2 x f32>, applying source selection and negation
    // from LLVM's decoded srcN_modifiers operand.
    //
    // Two operand shapes are accepted:
    //   * Register (the common case): reads a 64-bit VGPR pair as
    //     `<2 x f32>`; lo/hi extract index the two packed lanes.
    //   * Immediate / inline literal: VOP3P encodes a single 32-bit
    //     literal per source slot which the hardware broadcasts to
    //     both packed lanes (the `op_sel_hi` modifier is ignored on
    //     scalar literals because there's only one element to choose).
    //     The swiglu tensilelite kernel exercises this path with
    //     `v_pk_add_f32 vN, vM, 0x...` where the literal is a packed
    //     bias constant.  We model it by reading the i32, bit-casting
    //     to f32, and constructing a 2-lane vector with both lanes
    //     equal to the literal.  `neg_lo` / `neg_hi` still apply per lane.
    PackedSrcOptions opts;
    opts.RegisterSourceIsVector = true;
    opts.ImmediateIsScalarBroadcast = true;
    opts.ApplyFloatNeg = true;
    opts.Name = "pk_f32_src";
    Value *s0 = readPacked2Src(ctx, op, 0, ctx.f32Ty, mods[0], opts);
    Value *s1 = readPacked2Src(ctx, op, 1, ctx.f32Ty, mods[1], opts);

    Value *res = nullptr;
    switch (sop) {
    case CanonicalOp::V_PK_ADD_F32:
      res = ctx.B.CreateFAdd(s0, s1, "pk_add");
      break;
    case CanonicalOp::V_PK_MUL_F32:
      res = ctx.B.CreateFMul(s0, s1, "pk_mul");
      break;
    case CanonicalOp::V_PK_MAX_F32: {
      Function *fn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::maxnum, {v2f32});
      res = ctx.B.CreateCall(fn, {s0, s1}, "pk_max");
      break;
    }
    case CanonicalOp::V_PK_MIN_F32: {
      Function *fn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::minnum, {v2f32});
      res = ctx.B.CreateCall(fn, {s0, s1}, "pk_min");
      break;
    }
    case CanonicalOp::V_PK_FMA_F32: {
      Value *s2 = readPacked2Src(ctx, op, 2, ctx.f32Ty, mods[2], opts);
      Function *fn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::fma, {v2f32});
      res = ctx.B.CreateCall(fn, {s0, s1, s2}, "pk_fma");
      break;
    }
    default: llvm_unreachable("filtered by outer switch");
    }
    ctx.writeRegVec(op.dst(), res);
    hr.handled = true;
    return hr;
  }

  // ---- VOP3P packed-pair `<2 x i16>` int ops ----
  // V_PK_ADD_U16 / V_PK_LSHLREV_B16. Operand profile is
  // VOP_V2I16_V2I16_V2I16: 32-bit dst / 32-bit src0 / 32-bit src1, each
  // bitcast to `<2 x i16>` for the lane-wise op and back to i32 for the
  // VGPR write-back. Shared handler shape; per-CanonicalOp dispatch picks the
  // IR opcode (`add` vs the reversed `clshl_rev_16` shape — see notes
  // on each case below). Inline literals encode a packed `<2 x i16>`
  // directly (lo i16 = bits[15:0], hi i16 = bits[31:16]); there is NO
  // broadcast analogue to the V_PK_F32 32-bit-element family because
  // the literal width matches the operand width here. Sibling
  // V_PK_LSHRREV_B16 / V_PK_ASHRREV_I16 / V_PK_SUB_U16 / V_PK_MUL_LO_U16
  // share this exact shape — one extra `case` + IR-opcode dispatch in
  // the inner switch and they're done — but they're held out per the
  // "no fallback / design what the corpus exercises" discipline.
  case CanonicalOp::V_PK_ADD_U16:
  case CanonicalOp::V_PK_LSHLREV_B16: {
    auto *i16Ty = Type::getInt16Ty(ctx.C);

    constexpr unsigned KnownPkI16Mods =
        SISrcMods::OP_SEL_0 | SISrcMods::OP_SEL_1;
    unsigned mods[3] = {};
    if (!readPackedSrcMods(di, op, 2, KnownPkI16Mods, mods, hr))
      return hr;

    PackedSrcOptions opts;
    opts.Name = "pk_i16_src";
    Value *s0 = readPacked2Src(ctx, op, 0, i16Ty, mods[0], opts);
    Value *s1 = readPacked2Src(ctx, op, 1, i16Ty, mods[1], opts);

    Value *res = nullptr;
    switch (sop) {
    case CanonicalOp::V_PK_ADD_U16:
      res = ctx.B.CreateAdd(s0, s1, "pk_add_u16");
      break;
    case CanonicalOp::V_PK_LSHLREV_B16: {
      // clshl_rev_16 SDAG: dst = src1 << (src0 & 15). Reversed-operand
      // convention (shift count is src0, value is src1) AND a hardware
      // clamp to the low 4 bits of the count. LLVM `shl` is poison for
      // shifts >= bitwidth, the hardware masks instead — emit the AND
      // explicitly so the LLVM semantics match the AMDGPU semantics for
      // every legal hardware input. For constant shift counts the
      // optimiser folds the AND away; for VGPR-sourced shift counts the
      // mask is mandatory to preserve the corpus shift semantics.
      Value *mask = ConstantVector::getSplat(
          ElementCount::getFixed(2),
          ConstantInt::get(i16Ty, 15));
      Value *amt = ctx.B.CreateAnd(s0, mask, "pk_lshlrev_amt");
      res = ctx.B.CreateShl(s1, amt, "pk_lshlrev");
      break;
    }
    default: llvm_unreachable("filtered by outer switch");
    }

    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(res, ctx.i32Ty, "pk_i16_pack"));
    hr.handled = true;
    return hr;
  }

  // ---- v_dot4_i32_iu8 ----
  //
  // Mixed signed/unsigned 4-byte dot product:
  //   dst = src2 + sum_{i=0..3} extA(src0.byte[i]) * extB(src1.byte[i])
  //
  // AMDGPU models the input signedness through the VOP3P source modifier
  // operands: SISrcMods::NEG set on src0/src1 means that source's packed bytes
  // are signed, otherwise they are unsigned. LLVM's `VOP3PModsNeg` pattern in
  // SIInstrInfo.td encodes the same contract. Lower to ordinary integer IR
  // rather than a target dot intrinsic so gfx1250 same-target and gfx942
  // cross-target paths share one verifier-clean semantic representation; the
  // backend may rediscover a dot instruction where legal. A future
  // target-native optimisation can route supporting targets through
  // `llvm.amdgcn.sudot4`, but that should not be required for correctness.
  case CanonicalOp::V_DOT4_I32_IU8: {
    int clampIdx = AMDGPU::getNamedOperandIdx(di.inst.getOpcode(),
                                              AMDGPU::OpName::clamp);
    bool clamp = false;
    if (clampIdx >= 0 && di.isImm(static_cast<unsigned>(clampIdx)))
      clamp = di.getImm(static_cast<unsigned>(clampIdx)) != 0;
    if (clampIdx >= 0 && !di.isImm(static_cast<unsigned>(clampIdx))) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P", "v_dot4_i32_iu8 clamp operand is not an immediate");
      return hr;
    }

    Value *src0 = op.src(0);
    Value *src1 = op.src(1);
    Value *acc = ctx.B.CreateSExt(op.src(2), ctx.i64Ty, "dot4_acc_wide");
    auto *i8Ty = Type::getInt8Ty(ctx.C);

    auto extendByte = [&](Value *packed, unsigned byteIdx,
                          bool isSigned) -> Value * {
      Value *shift = ConstantInt::get(ctx.i32Ty, byteIdx * 8);
      Value *lo = ctx.B.CreateTrunc(ctx.B.CreateLShr(packed, shift), i8Ty,
                                    "dot4_byte");
      return isSigned ? ctx.B.CreateSExt(lo, ctx.i64Ty, "dot4_sext")
                      : ctx.B.CreateZExt(lo, ctx.i64Ty, "dot4_zext");
    };

    const bool src0Signed = (op.srcMod(0) & SISrcMods::NEG) != 0;
    const bool src1Signed = (op.srcMod(1) & SISrcMods::NEG) != 0;
    for (unsigned i = 0; i < 4; ++i) {
      Value *a = extendByte(src0, i, src0Signed);
      Value *b = extendByte(src1, i, src1Signed);
      acc = ctx.B.CreateAdd(acc, ctx.B.CreateMul(a, b, "dot4_mul"),
                            "dot4_acc");
    }

    if (clamp) {
      Value *lo = ConstantInt::get(ctx.i64Ty, INT32_MIN);
      Value *hi = ConstantInt::get(ctx.i64Ty, INT32_MAX);
      acc = ctx.B.CreateSelect(ctx.B.CreateICmpSLT(acc, lo), lo, acc,
                               "dot4_clamp_lo");
      acc = ctx.B.CreateSelect(ctx.B.CreateICmpSGT(acc, hi), hi, acc,
                               "dot4_clamp");
    }

    ctx.writeReg32(op.dst(), ctx.B.CreateTrunc(acc, ctx.i32Ty, "dot4_i32"));
    hr.handled = true;
    return hr;
  }

  // ---- WMMA (gfx1250 RDNA4, VOP3P encoding) ----
  // 16x16xK WMMA family. Three K-families × accumulator-type
  // permutations covered today:
  //   * 16-bit elements, K=32, f32 acc (8 VGPRs of <16 x t> per A/B side):
  //       v_wmma_f32_16x16x32_f16,  v_wmma_f32_16x16x32_bf16
  //   * 8-bit elements,  K=64, f32 acc (8 VGPRs of <8 x i32> per A/B side):
  //       v_wmma_f32_16x16x64_<a>_<b>  for a,b ∈ {fp8, bf8}
  //   * 8-bit elements,  K=64, i32 acc (8 VGPRs of <8 x i32> per A/B side):
  //       v_wmma_i32_16x16x64_iu8  (signed/unsigned 8-bit integer GEMMs)
  //
  // All share the per-Wave32-lane A/B fragment shape (8 VGPRs, 32 bytes).
  // The C/D side is <8 x f32> for the f32-accumulator variants and
  // <8 x i32> for the IU8 integer-accumulator variant. The WMMA12
  // native-intrinsic path (when target supports it) and the gfx942
  // MFMA lowering path (`emitWMMAtoMFMA`, parameterised on
  // `WMMAInputType`) are uniform across the entire family — the local
  // A/B IR vector type + native-WMMA intrinsic ID + WMMAInputType +
  // accumulator IR type is the only delta between variants. "Design
  // the operation, not the opcode."
  //
  // Native WMMA12 intrinsic-call shapes split THREE ways:
  //   * 16-bit f32-acc: AMDGPUWmmaIntrinsicModsAllReuse — 8 args
  //       (A_mod, A, B_mod, B, C_mod, C, reuse_a, reuse_b)
  //   * 8-bit  f32-acc: AMDGPUWmmaIntrinsicModsC       — 6 args
  //       (A, B, C_mod, C, reuse_a, reuse_b)
  //   * 8-bit  i32-acc: AMDGPUWmmaIntrinsicModsABClamp — 8 args
  //       (A_mod, A, B_mod, B, C, reuse_a, reuse_b, clamp)
  // The MFMA fallback path is uniform across all three.
  // 16x16x4 WMMA (32-bit f32 A/B/C, gfx1250 VOP3P opcode 0x05D).
  // This handler stands alone from the K=32 / K=64 family below
  // because (a) the per-lane A/B fragment is `<2 x f32>` (only 2
  // dwords) instead of <16 x t> (16-bit) or <8 x i32> (8-bit), and
  // (b) `emitWMMAtoMFMA` is parameterised on 16-/8-bit element
  // packing and does not cover the K=4 f32 case.
  //
  // The native intrinsic `int_amdgcn_wmma_f32_16x16x4_f32` is
  // declared inside `AMDGPUWMMAIntrinsicsGFX1250` (gated by
  // `isGFX125xOnly` in IntrinsicsAMDGPU.td:4113-4114), so it is
  // strictly gfx1250-only — the gfx12 (RDNA4 base) WMMA family
  // (`AMDGPUWMMAIntrinsicsGFX12`, gated by `hasWMMA12` =
  // FeatureWMMA{128,256}bInsts) does NOT include it. Same-target
  // lift therefore gates on `ctx.targetIsa.hasTensorOps`
  // (FeatureGFX1250Insts), not `hasWMMA12`.
  //
  // Cross-target on gfx942 we lower to `mfma_f32_16x16x4f32` via the
  // dedicated `emitWMMAtoMFMA_F32_16x16x4` helper in
  // `wmma_lowering.cpp` — gfx942 has a direct K=4 MFMA equivalent
  // so the decomposition is 1 MFMA per Wave32 group (not 2 chained
  // like the K=32/K=64 path). The shared ds_bpermute redistribution
  // math is documented alongside the helper. Targets with neither
  // `hasTensorOps` nor `hasMFMA` (e.g. gfx12 RDNA4 base) get a
  // principled refusal — they have no K=4 f32 matrix path at all.
  case CanonicalOp::V_WMMA_F32_16x16x4_F32: {
    auto *abIRTy = FixedVectorType::get(ctx.f32Ty, 2);
    auto *cdIRTy = FixedVectorType::get(ctx.f32Ty, 8);

    ParsedReg dest = op.dst();
    ParsedReg srcA = op.srcReg(0), srcB = op.srcReg(1);

    Value *a = ctx.regs.readRegVec(ctx.B, srcA, abIRTy);
    Value *b = ctx.regs.readRegVec(ctx.B, srcB, abIRTy);
    Value *c = readWMMAAccumC(ctx, di, op, dest, cdIRTy, hr);
    if (!c)
      return hr;

    Value *result_val;
    if (ctx.targetIsa.hasTensorOps) {
      Function *wmmaFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_wmma_f32_16x16x4_f32, {cdIRTy, abIRTy});
      // AMDGPUWmmaIntrinsicModsC (6 args, see IntrinsicsAMDGPU.td):
      //   (A, B, C_mod, C, matrix_a_reuse, matrix_b_reuse)
      // C_mod is the i16 source-modifier bitfield (op_sel etc.) and
      // matrix_*_reuse are i1 flags. K=4 f32 WMMA has NO per-element
      // A_mod / B_mod slots (unlike the 16-/8-bit ModsAllReuse /
      // ModsABClamp classes used by the K=32 / K=64 family). The
      // gfx1250 corpus emits the instruction without those modifiers
      // set; defaulting to 0 / false matches what the disassembler
      // surfaces for the failing kernels.
      result_val = ctx.B.CreateCall(wmmaFn, {
          a, b,
          ConstantInt::get(Type::getInt16Ty(ctx.C), 0), c,
          ctx.B.getFalse(), ctx.B.getFalse()
      }, "wmma");
    } else if (ctx.targetIsa.hasMFMA) {
      // Same-shape gate as the K=32/K=64 case below.  The staged
      // strict.wwm-scoped MODREP lowering is verified correct for
      // minimal-repro kernels (isolated and K-loop-chained WMMAs)
      // but an unexplained residual divergence remains on the
      // Triton `matmul_fp16_16x16` kernel at M>=32 through
      // `compare_correctness`.  See the K=32/K=64 arm below for
      // the full discussion.  Gate stays in place until the
      // residual is pinned and the fix lands; the infrastructure
      // in `wave_projection.hpp` (`numSourceWavesPerTarget`,
      // `wrapAsWWMValue`) is LANDED additively.
      // K=4 f32 arm: previously conservatively refused under MODREP
      // when a multi-WMMA-per-K-iter pattern (permlane16_swap
      // presence) was detected.  The root cause turned out to be a
      // wrong-semantic lift of `v_permlane16_swap_b32` (symmetric
      // vs. ISA-asymmetric — see `handle_valu_cross_lane.cpp` and
      // matrix-translation.md §12.4.7), not a WMMA-lowering
      // problem, so with that fixed the MODREP MFMA lowering
      // handles both single- and multi-WMMA cases correctly.
      result_val = emitWMMAtoMFMA_F32_16x16x4(ctx, a, b, c);
    } else {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          "v_wmma_f32_16x16x4_f32 cross-target requires either "
          "hasTensorOps (native gfx1250 intrinsic "
          "int_amdgcn_wmma_f32_16x16x4_f32) or hasMFMA (gfx942 "
          "mfma_f32_16x16x4f32 decomposition); this target has "
          "neither — no K=4 f32 matrix path is available");
      return hr;
    }

    ctx.writeRegVec(dest, result_val);
    hr.handled = true;
    return hr;
  }

  case CanonicalOp::V_WMMA_F32_16x16x32_F16:
  case CanonicalOp::V_WMMA_F32_16x16x32_BF16:
  case CanonicalOp::V_WMMA_F32_16x16x64_FP8_FP8:
  case CanonicalOp::V_WMMA_F32_16x16x64_FP8_BF8:
  case CanonicalOp::V_WMMA_F32_16x16x64_BF8_FP8:
  case CanonicalOp::V_WMMA_F32_16x16x64_BF8_BF8:
  case CanonicalOp::V_WMMA_I32_16x16x64_IU8: {
    const bool isIU8 = (sop == CanonicalOp::V_WMMA_I32_16x16x64_IU8);
    const bool isFP8orBF8 =
        (sop == CanonicalOp::V_WMMA_F32_16x16x64_FP8_FP8) ||
        (sop == CanonicalOp::V_WMMA_F32_16x16x64_FP8_BF8) ||
        (sop == CanonicalOp::V_WMMA_F32_16x16x64_BF8_FP8) ||
        (sop == CanonicalOp::V_WMMA_F32_16x16x64_BF8_BF8);
    const bool is8bit = isIU8 || isFP8orBF8;
    const bool isBF16 = (sop == CanonicalOp::V_WMMA_F32_16x16x32_BF16);

    Type *abIRTy = nullptr;
    if (is8bit) {
      abIRTy = FixedVectorType::get(ctx.i32Ty, 8);
    } else {
      Type *elemTy = isBF16 ? Type::getBFloatTy(ctx.C)
                            : Type::getHalfTy(ctx.C);
      abIRTy = FixedVectorType::get(elemTy, 16);
    }
    Type *cdIRTy = isIU8 ? FixedVectorType::get(ctx.i32Ty, 8)
                         : FixedVectorType::get(ctx.f32Ty, 8);

    ParsedReg dest = op.dst();
    ParsedReg srcA = op.srcReg(0), srcB = op.srcReg(1);

    Value *a = ctx.regs.readRegVec(ctx.B, srcA, abIRTy);
    Value *b = ctx.regs.readRegVec(ctx.B, srcB, abIRTy);
    Value *c = readWMMAAccumC(ctx, di, op, dest, cdIRTy, hr);
    if (!c)
      return hr;

    auto wmmaInputType = [&]() -> WMMAInputType {
      switch (sop) {
      case CanonicalOp::V_WMMA_F32_16x16x32_F16:    return WMMAInputType::F16;
      case CanonicalOp::V_WMMA_F32_16x16x32_BF16:   return WMMAInputType::BF16;
      case CanonicalOp::V_WMMA_F32_16x16x64_FP8_FP8:return WMMAInputType::FP8_FP8;
      case CanonicalOp::V_WMMA_F32_16x16x64_FP8_BF8:return WMMAInputType::FP8_BF8;
      case CanonicalOp::V_WMMA_F32_16x16x64_BF8_FP8:return WMMAInputType::BF8_FP8;
      case CanonicalOp::V_WMMA_F32_16x16x64_BF8_BF8:return WMMAInputType::BF8_BF8;
      case CanonicalOp::V_WMMA_I32_16x16x64_IU8:    return WMMAInputType::IU8;
      default:
        report_fatal_error("transpiler: WMMA CanonicalOp not in dispatch table");
      }
    }();

    Value *result_val;
    // Native-intrinsic branch: the K=32 / K=64 WMMA intrinsics
    // (`int_amdgcn_wmma_f32_16x16x32_f16`, `..._f32_16x16x64_*`,
    // `..._i32_16x16x64_iu8`) live in `AMDGPUWMMAIntrinsicsGFX1250`
    // (IntrinsicsAMDGPU.td:4096 — the gfx1250-specific family),
    // NOT in `AMDGPUWMMAIntrinsicsGFX12` (IntrinsicsAMDGPU.td:3123 —
    // the gfx12 RDNA4 base family, which only covers K=16
    // `..._16x16x16_*`).  The gate therefore must be
    // `hasTensorOps` (`FeatureGFX1250Insts`), not `hasWMMA12`
    // (`FeatureWMMA{128,256}bInsts` — set only on the gfx12 base
    // subtargets that DON'T have K=32/K=64 hardware).  An earlier
    // version of this handler used `hasWMMA12` here, which made
    // gfx1250 same-target lifts of K=32/K=64 WMMA fall through to
    // the `emitWMMAtoMFMA` branch below and emit MFMA-intrinsic IR
    // the backend couldn't lower on gfx1250 (no MFMA hardware).
    // `BatchRaise.Gfx1250TestData` "succeeded" on that broken path
    // because the raise completed, even though any downstream
    // codegen attempt would have failed.  The WMMA taxonomy
    // and the K=4 f32 case above use the same structural fix
    // pattern (which got it right originally — this K=32/K=64
    // case was slower to catch up).
    if (ctx.targetIsa.hasTensorOps) {
      Intrinsic::ID wmmaId;
      switch (sop) {
      case CanonicalOp::V_WMMA_F32_16x16x32_F16:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x32_f16; break;
      case CanonicalOp::V_WMMA_F32_16x16x32_BF16:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x32_bf16; break;
      case CanonicalOp::V_WMMA_F32_16x16x64_FP8_FP8:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_fp8; break;
      case CanonicalOp::V_WMMA_F32_16x16x64_FP8_BF8:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_bf8; break;
      case CanonicalOp::V_WMMA_F32_16x16x64_BF8_FP8:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_fp8; break;
      case CanonicalOp::V_WMMA_F32_16x16x64_BF8_BF8:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_bf8; break;
      case CanonicalOp::V_WMMA_I32_16x16x64_IU8:
        wmmaId = Intrinsic::amdgcn_wmma_i32_16x16x64_iu8; break;
      default:
        report_fatal_error(
            "transpiler: WMMA CanonicalOp not in gfx1250 K=32/K=64 dispatch");
      }
      Function *wmmaFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, wmmaId, {cdIRTy, abIRTy});
      if (isIU8) {
        // AMDGPUWmmaIntrinsicModsABClamp:
        //   (A_mod, A, B_mod, B, C, reuse_a, reuse_b, clamp)
        // A_mod / B_mod carry the IU8 sign-vs-zero-extension knobs in
        // the gfx1250 ISA; we conservatively emit 0 (zero-extend, i.e.
        // unsigned interpretation) because the corpus IU8 GEMMs
        // observed so far never set the matching `neg_lo` bits. A
        // future loud refusal could be added if a corpus kernel ever
        // surfaces a non-zero A_mod / B_mod through the decoder.
        result_val = ctx.B.CreateCall(wmmaFn, {
            ctx.B.getFalse(), a,
            ctx.B.getFalse(), b,
            c,
            ctx.B.getFalse(), ctx.B.getFalse(),
            ctx.B.getFalse()
        }, "wmma");
      } else {
        // AMDGPUWmmaIntrinsicModsC: (A, B, C_mod, C, reuse_a, reuse_b)
        //
        // Both the 8-bit FP8/BF8 family and the 16-bit f16/bf16
        // family use this 6-arg shape in the gfx1250 intrinsic
        // set (`AMDGPUWMMAIntrinsicsGFX1250` — IntrinsicsAMDGPU.td
        // ~line 4098).  An earlier version of this handler split
        // the 16-bit family into an 8-arg `ModsAllReuse` shape
        // (A_mod, A, B_mod, B, C_mod, C, reuse_a, reuse_b) that
        // matches the gfx12 RDNA4 base WMMA family's intrinsics —
        // which do NOT include K=32/K=64 variants; those gfx12-base
        // intrinsics are K=16-only (`..._16x16x16_*` in
        // `AMDGPUWMMAIntrinsicsGFX12` ~line 3123).  That mismatched
        // arg-list did not surface under the old `hasWMMA12` gate
        // because `hasWMMA12` is never true on gfx1250 subtargets
        // (their feature set deliberately excludes
        // `FeatureWMMA{128,256}bInsts`), so the branch was
        // unreachable at runtime — but the principled fix in the
        // same commit (switching the gate to `hasTensorOps`) makes
        // the branch reachable, and the arg list must match.  See
        // the LLVM intrinsic-signature trailer further up in this
        // block comment for the full per-family taxonomy.
        result_val = ctx.B.CreateCall(wmmaFn, {
            a, b,
            ConstantInt::get(Type::getInt16Ty(ctx.C), 0), c,
            ctx.B.getFalse(), ctx.B.getFalse()
        }, "wmma");
      }
    } else if (ctx.targetIsa.hasMFMA) {
      // Same-shape gate as the K=4 f32 case above.  The staged
      // strict.wwm-scoped MODREP lowering is verified correct for
      // isolated and K-loop-chained WMMAs in
      // `lit_tests/wmma_phantom_lane_f16_chain/` (post-rebuild with
      // my test harness actually populating per-iter A/B data —
      // earlier runs claiming "K-loop failure" were a host-side
      // test-input miscount that populated the wrong stride; the
      // fixture itself matches bit-exact at n_iters ∈ {1,2,4,8}
      // under both MODREP phantom-lane and WaveNative).  Triton's
      // `matmul_fp16_16x16` at M>=32 still shows a WRONG-numerics
      // residual through `compare_correctness`, where the
      // backend's MIR-level operand folding (SIWholeQuadMode input
      // from ISEL) merges MFMA2's A and B into the same virtual
      // register due to a kernel-specific IR pattern (v_dual_mov
      // copying v[0:3] to v[64:67] combined with `v5=v4, v6=v4,
      // v7=v4` in the source ISA, making aDwords[4..7] all equal
      // to v4's value).  That operand folding is semantically
      // valid if the values are provably equal at runtime — which
      // they are, given the copies — so the MFMA itself computes
      // correctly.  The residual divergence vs the native gfx1250
      // run is under separate investigation and NOT a bug in this
      // lowering's contract.  Until that residual is pinned to a
      // concrete root cause, this gate stays in place rather than
      // ship a subtle wrong-numerics kernel.  Land the
      // infrastructure (numSourceWavesPerTarget, wrapAsWWMValue)
      // additively; flip this gate once the matmul_fp16_16x16
      // divergence is explained.
      // K=32/K=64 WMMA arm: previously refused under MODREP when
      // the kernel carried a `v_permlane16_swap_b32` (the
      // multi-WMMA-per-K-iter marker of Triton's `matmul_fp16`
      // pattern), because empirical lane-data dumps showed the
      // MFMA redistribution reading duplicated K subsets for
      // WMMA.A.
      //
      // MI400 Shader Programming Guide § V_PERMLANE16_SWAP_B32
      // pinned the real root cause: our permlane16_swap lift was
      // implementing the SYMMETRIC cross-16 swap, but the ISA
      // semantic is ASYMMETRIC ("lanes 0:15 of src0 and lanes
      // 16:31 of vdst swapped; lanes 16:31 of src0 and lanes 0:15
      // of vdst UNCHANGED").  The symmetric emulation
      // over-swapped the two "unchanged" halves, which corrupted
      // every matmul_fp16 input position and surfaced downstream
      // as the "+16 col shift" on mode-5 / "+4 bias" on mode-8.
      // With the asymmetric lift in `handle_valu_cross_lane.cpp`
      // (Session-8, matrix-translation.md §12.4.7) the MODREP
      // MFMA redistribution is correct for both single- and
      // multi-WMMA regimes, so this refusal is no longer needed.
      result_val = emitWMMAtoMFMA(ctx, a, b, c, wmmaInputType);
    } else {
      // Target has neither gfx1250 tensor ops (hasTensorOps, K=32
      // / K=64 WMMA native) nor MFMA (gfx942 CDNA3 et al., the
      // cross-target decomposition sink).  No path exists to
      // lower this opcode.  The pre-2026-04-22 handler fell
      // through to `emitWMMAtoMFMA` here and emitted MFMA-
      // intrinsic IR the target couldn't lower; the raise
      // "succeeded" but the backend then failed to codegen —
      // `BatchRaise.Gfx1250TestData` was structurally accepting
      // that broken-IR outcome, not catching it.  Refuse loudly
      // so coverage tooling surfaces targets whose WMMA K=32/
      // K=64 support is missing rather than pretending the lift
      // worked.
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          "v_wmma_*_16x16x{32,64}_* has no available lowering on "
          "the target ISA: the target does not provide gfx1250 "
          "tensor ops (`hasTensorOps` / `FeatureGFX1250Insts` for "
          "the native K=32/K=64 WMMA intrinsic family "
          "`AMDGPUWMMAIntrinsicsGFX1250`) nor MFMA (`hasMFMA` for "
          "the cross-target decomposition via "
          "`wmma_lowering.cpp::emitWMMAtoMFMA`).  No known safe "
          "lowering exists; refusing rather than emitting IR that "
          "cannot be lowered.");
      return hr;
    }

    ctx.writeRegVec(dest, result_val);
    hr.handled = true;
    return hr;
  }

  // 16x16x128 scaled WMMA, f8f6f4 mantissa-format family (gfx1250-only).
  //
  // 18 MC pseudos (`{f4,f6,f8} A × {f4,f6,f8} B × _twoaddr/_threeaddr`)
  // collapse onto this single CanonicalOp; the per-matrix vector width is
  // encoded by the opcode's `_fA_fB_w32_*` suffix (per
  // `WMMA_F8F6F4_Profiles` in VOP3PInstructions.td:1908) — f8 → 16
  // dwords, f6 → 12 dwords, f4 → 8 dwords. The in-family element
  // distinction (BF8 vs FP8 within f8; BF6 vs FP6 within f6) lives in
  // the `matrix_a_fmt` / `matrix_b_fmt` named-immediate operands
  // (`enum MatrixFMT`, SIDefines.h:1052-1058).
  //
  // Cross-target lowering paths for v_wmma_scale_f32_16x16x128_f8f6f4:
  //
  //   * gfx1250 (hasTensorOps): emit the native
  //     `int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4` intrinsic in place
  //     (14-arg fast path below).
  //   * gfx950 (hasGfx950Insts): rewrite to the gfx950 scaled MFMA via
  //     `emitWMMAScaleF8F6F4toMFMA` in `wmma_lowering.cpp`, which does
  //     the wave32->wave64 lane redistribution and lowers to
  //     `int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` (the gfx950 has
  //     a near-1:1 MFMA equivalent for this WMMA, in the same K=128
  //     F8F6F4 shape; only the wave size and per-lane fragment width
  //     differ, both of which the redistribution helper handles).
  //     Note `hasGfx950Insts` (NOT `hasMFMA`) -- gfx942 also has
  //     hasMFMA == true but lacks the scaled F8F6F4 family.
  //   * Otherwise (e.g. gfx942 -- has hasMFMA == true but not the scaled
  //     F8F6F4 family): refuse loudly.  gfx942's MFMA family stops at
  //     `mfma_f32_16x16x32_*` (non-scaled, K=32); a WMMA-scale
  //     decomposition into multiple gfx942 MFMAs plus a software
  //     scale-exponent application is *possible* but not implemented.
  case CanonicalOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4: {
    // Extract per-matrix dword count from the MC pseudo suffix
    // (`*_fA_fB_w32_{twoaddr,threeaddr}`). MCInstrInfo names the
    // pseudo verbatim from TableGen, so the suffix is the
    // authoritative source of A/B widths.
    auto fmtSuffixToDwords = [](StringRef tag) -> unsigned {
      if (tag == "f8") return 16;
      if (tag == "f6") return 12;
      if (tag == "f4") return 8;
      return 0;
    };
    StringRef pseudoName = ctx.mc.instrInfo->getName(di.inst.getOpcode());
    StringRef body = pseudoName;
    body.consume_front("V_WMMA_SCALE_F32_16X16X128_F8F6F4_");
    SmallVector<StringRef, 4> parts;
    body.split(parts, '_');
    if (parts.size() < 2) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          "v_wmma_scale_f32_16x16x128_f8f6f4: cannot parse fA_fB suffix from "
          "MC pseudo name");
      return hr;
    }
    unsigned aDwords = fmtSuffixToDwords(parts[0]);
    unsigned bDwords = fmtSuffixToDwords(parts[1]);
    if (aDwords == 0 || bDwords == 0) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          "v_wmma_scale_f32_16x16x128_f8f6f4: unrecognised mantissa-format "
          "tag in MC pseudo suffix (expected f4/f6/f8)");
      return hr;
    }

    auto *aTy = FixedVectorType::get(ctx.i32Ty, aDwords);
    auto *bTy = FixedVectorType::get(ctx.i32Ty, bDwords);
    auto *cdTy = FixedVectorType::get(ctx.f32Ty, 8);

    ParsedReg dest = op.dst();
    ParsedReg srcA = op.srcReg(0), srcB = op.srcReg(1);
    ParsedReg srcC = op.isSrcReg(2) ? op.srcReg(2) : dest;

    Value *a = ctx.regs.readRegVec(ctx.B, srcA, aTy);
    Value *b = ctx.regs.readRegVec(ctx.B, srcB, bTy);
    Value *c = ctx.regs.readRegVec(ctx.B, srcC, cdTy);

    // Read named-immediate / named-register operands. Using
    // `getNamedOperandIdx` instead of positional scan means any
    // future TableGen reshuffle of the scaled-WMMA Ins64 layout
    // flows in for free (mirrors the MFMA-scale handler in
    // handle_mfma.cpp:175-194).
    auto namedImm = [&](AMDGPU::OpName name) -> int64_t {
      int idx = AMDGPU::getNamedOperandIdx(di.inst.getOpcode(), name);
      if (idx < 0 || !di.isImm(idx)) return 0;
      return di.getImm(idx);
    };
    auto namedReg32 = [&](AMDGPU::OpName name) -> Value * {
      int idx = AMDGPU::getNamedOperandIdx(di.inst.getOpcode(), name);
      if (idx < 0 || !di.isReg(idx))
        return ConstantInt::get(ctx.i32Ty, 0);
      ParsedReg pr = ctx.parseReg(di.getReg(idx), idx);
      if (pr.kind == ParsedReg::OTHER || pr.kind == ParsedReg::NOREG)
        return ConstantInt::get(ctx.i32Ty, 0);
      return ctx.regs.readReg32(ctx.B, pr);
    };

    Value *matrixAFmt =
        ConstantInt::get(ctx.i32Ty, namedImm(AMDGPU::OpName::matrix_a_fmt));
    Value *matrixBFmt =
        ConstantInt::get(ctx.i32Ty, namedImm(AMDGPU::OpName::matrix_b_fmt));
    Value *cMod = ConstantInt::get(
        Type::getInt16Ty(ctx.C),
        namedImm(AMDGPU::OpName::src2_modifiers));
    Value *matrixAScale = ConstantInt::get(
        ctx.i32Ty, namedImm(AMDGPU::OpName::matrix_a_scale));
    Value *matrixAScaleFmt = ConstantInt::get(
        ctx.i32Ty, namedImm(AMDGPU::OpName::matrix_a_scale_fmt));
    Value *scaleSrc0 = namedReg32(AMDGPU::OpName::scale_src0);
    Value *matrixBScale = ConstantInt::get(
        ctx.i32Ty, namedImm(AMDGPU::OpName::matrix_b_scale));
    Value *matrixBScaleFmt = ConstantInt::get(
        ctx.i32Ty, namedImm(AMDGPU::OpName::matrix_b_scale_fmt));
    Value *scaleSrc1 = namedReg32(AMDGPU::OpName::scale_src1);
    Value *matrixAReuse = ConstantInt::get(
        Type::getInt1Ty(ctx.C),
        namedImm(AMDGPU::OpName::matrix_a_reuse));
    Value *matrixBReuse = ConstantInt::get(
        Type::getInt1Ty(ctx.C),
        namedImm(AMDGPU::OpName::matrix_b_reuse));

    Value *result_val = nullptr;

    if (ctx.targetIsa.hasTensorOps) {
      // Same-target gfx1250 -> gfx1250 fast path: native scaled WMMA.
      //
      // AMDGPUWmmaScaleIntrinsicModsC<i32>:
      //   (matrix_a_fmt, A, matrix_b_fmt, B, C_mod, C,
      //    matrix_a_scale, matrix_a_scale_fmt, scale_src0,
      //    matrix_b_scale, matrix_b_scale_fmt, scale_src1,
      //    matrix_a_reuse, matrix_b_reuse)
      // Overloaded on D, A, B element vector types.
      Function *wmmaFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_wmma_scale_f32_16x16x128_f8f6f4,
          {cdTy, aTy, bTy});
      result_val = ctx.B.CreateCall(wmmaFn,
                                    {matrixAFmt, a, matrixBFmt, b, cMod, c,
                                     matrixAScale, matrixAScaleFmt, scaleSrc0,
                                     matrixBScale, matrixBScaleFmt, scaleSrc1,
                                     matrixAReuse, matrixBReuse},
                                    "wmma_scale");
    } else if (ctx.targetIsa.hasGfx950Insts) {
      // Cross-target gfx1250 -> gfx950 path: WMMA-scale -> MFMA-scale.
      //
      // The gfx950 scaled F8F6F4 MFMA (introduced as part of the gfx950
      // MAI family in LLVM upstream) covers the same K=128 F8/F6/F4 shape
      // as the gfx1250 WMMA, so the lowering is a pure wave32->wave64
      // lane redistribution + intrinsic swap (no K-decomposition, no
      // software scale emulation).  Implementation in
      // `wmma_lowering.cpp::emitWMMAScaleF8F6F4toMFMA`.
      //
      // matrix_a_reuse / matrix_b_reuse are perf hints (not correctness)
      // and have no MFMA equivalent; the helper drops them.
      result_val = emitWMMAScaleF8F6F4toMFMA(
          ctx, a, b, c, matrixAFmt, matrixBFmt, cMod, matrixAScale,
          matrixAScaleFmt, scaleSrc0, matrixBScale, matrixBScaleFmt, scaleSrc1,
          aDwords, bDwords);
      // Reference reuse hints so -Wunused-variable doesn't flag them
      // when this branch is taken.  (The hints are extracted up-top and
      // the gfx1250 arm above passes them through, so they always have
      // a use at the IR level; this is purely about the compiler's view
      // of this scope.)
      (void)matrixAReuse;
      (void)matrixBReuse;
      if (!result_val) {
        hr.failure = RaiseFailure::unsupportedShape(
            di, "VOP3P",
            "emitWMMAScaleF8F6F4toMFMA refused this fragment width "
            "(aDwords/bDwords outside f8/f6/f4 set)");
        return hr;
      }
    } else {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          "v_wmma_scale_f32_16x16x128_f8f6f4 requires either hasTensorOps "
          "(gfx1250 native scaled WMMA, "
          "int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4) or hasGfx950Insts "
          "(gfx950 scaled-MFMA F8F6F4, "
          "int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4 via "
          "emitWMMAScaleF8F6F4toMFMA); this target has neither.  "
          "(gfx942 has hasMFMA == true but no scaled-F8F6F4 family; a "
          "software decomposition is not yet implemented.)");
      return hr;
    }

    ctx.writeRegVec(dest, result_val);
    hr.handled = true;
    return hr;
  }

  // ---- v_fma_mixlo_bf16: BF16-result mixed-precision FMA (VOP3P) ----
  //
  // LLVM's RDNA4 TableGen definition declares V_FMA_MIXLO_BF16 with
  // VOP_BF16_BF16_BF16_BF16 and FPDPRounding=1. The generated selection
  // patterns model it as:
  //
  //   fptrunc_bf16(llvm.fma.f32(cvt_f32(src0_part),
  //                             cvt_f32(src1_part),
  //                             cvt_f32(src2_part)))
  //
  // and the ISA family writes only the low 16 bits of vdst (the high
  // half is the tied vdst_in input). The source `*_part` selection
  // matches V_FMA_MIX_F32_BF16 below: op_sel_hi chooses narrow-bf16 vs
  // full-f32, and op_sel chooses the high half when a register source is
  // interpreted as narrow.
  case CanonicalOp::V_FMA_MIXLO_BF16: {
    if (op.nSrcs() < 3) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3P",
          "v_fma_mixlo_bf16 requires three explicit source operands");
      return hr;
    }

    bool clampResult = false;
    int clampIdx = AMDGPU::getNamedOperandIdx(di.inst.getOpcode(),
                                              AMDGPU::OpName::clamp);
    if (clampIdx >= 0) {
      if (!di.isImm(static_cast<unsigned>(clampIdx))) {
        hr.failure = RaiseFailure::unsupportedShape(
            di, "VOP3P",
            "v_fma_mixlo_bf16 clamp operand is not an immediate");
        return hr;
      }
      clampResult = di.getImm(static_cast<unsigned>(clampIdx)) != 0;
    }

    Type *bf16Ty = Type::getBFloatTy(ctx.C);
    Type *i16Ty = Type::getInt16Ty(ctx.C);

    constexpr unsigned KnownMixMods =
        SISrcMods::NEG | SISrcMods::ABS | SISrcMods::OP_SEL_0 |
        SISrcMods::OP_SEL_1;
    unsigned mods[3] = {};
    if (!readSourceMods(di, op, 3, KnownMixMods, mods, hr))
      return hr;

    Value *s0 = readMixF32Src(ctx, op, 0, bf16Ty, mods[0],
                              "mixlo_cvt_bf16");
    Value *s1 = readMixF32Src(ctx, op, 1, bf16Ty, mods[1],
                              "mixlo_cvt_bf16");
    Value *s2 = readMixF32Src(ctx, op, 2, bf16Ty, mods[2],
                              "mixlo_cvt_bf16");
    Function *fmaFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::fma, {ctx.f32Ty});
    Value *fma = ctx.B.CreateCall(fmaFn, {s0, s1, s2}, "fma_mixlo_bf16");
    Value *rounded = ctx.B.CreateFPTrunc(fma, bf16Ty, "fma_mixlo_bf16_round");
    if (clampResult) {
      // AMDGPUclamp clamps to [0, 1] and maps NaN to 0 (SIInstrInfo.td).
      // V_FMA_MIXLO_BF16 applies it after the destination BF16 rounding.
      Function *maxFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::maxnum, {bf16Ty});
      Function *minFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::minnum, {bf16Ty});
      rounded = ctx.B.CreateCall(
          minFn,
          {ctx.B.CreateCall(maxFn,
                            {rounded, ConstantFP::get(bf16Ty, 0.0)},
                            "fma_mixlo_bf16_clamp_lo"),
           ConstantFP::get(bf16Ty, 1.0)},
          "fma_mixlo_bf16_clamp");
    }
    Value *loBits =
        ctx.B.CreateZExt(ctx.B.CreateBitCast(rounded, i16Ty), ctx.i32Ty);

    ParsedReg dest = op.dst();
    Value *oldDest = ctx.regs.readReg32(ctx.B, dest);
    Value *oldHi = ctx.B.CreateAnd(
        oldDest, ConstantInt::get(ctx.i32Ty, 0xFFFF0000u),
        "fma_mixlo_bf16_old_hi");
    ctx.writeReg32(dest, ctx.B.CreateOr(oldHi, loBits,
                                        "fma_mixlo_bf16_pack"));
    hr.handled = true;
    return hr;
  }

  // ---- v_fma_mix_f32 / v_fma_mix_f32_bf16: mixed-precision FMA (VOP3P) ----
  //
  // dst = fma(cvt_f32(src0_part), cvt_f32(src1_part), cvt_f32(src2_part))
  //
  // Per-source selection is driven by the VOP3P op_sel / op_sel_hi
  // modifier pair carried in LLVM's decoded srcN_modifiers operands:
  //
  //   op_sel_hi[i]==0  -> source i is the full f32 VGPR
  //   op_sel_hi[i]==1  -> source i is the 16-bit half selected by
  //                       op_sel[i] (0=lo [15:0], 1=hi [31:16])
  //                       interpreted as the mnemonic's narrow type
  //                       (f16 for V_FMA_MIX_F32, bf16 for
  //                       V_FMA_MIX_F32_BF16), then fpext'd to f32.
  //
  // The BF16 variant does NOT need a cross-target refusal because
  // `fpext bfloat to float` is universally lowered (it's a shift-left-16
  // + bitcast on every AMDGPU target); only the narrow element type
  // switches.
  case CanonicalOp::V_FMA_MIX_F32:
  case CanonicalOp::V_FMA_MIX_F32_BF16: {
    Type *narrowTy = (sop == CanonicalOp::V_FMA_MIX_F32_BF16)
                         ? Type::getBFloatTy(ctx.C)
                         : ctx.f16Ty;
    const char *cvtName = (sop == CanonicalOp::V_FMA_MIX_F32_BF16)
                              ? "mix_cvt_bf16"
                              : "mix_cvt";

    constexpr unsigned KnownMixMods =
        SISrcMods::NEG | SISrcMods::ABS | SISrcMods::OP_SEL_0 |
        SISrcMods::OP_SEL_1;
    unsigned mods[3] = {};
    if (!readSourceMods(di, op, 3, KnownMixMods, mods, hr))
      return hr;

    // `OP_SEL_0` is a VGPR-half selector and only makes sense when the source
    // is a 32-bit VGPR that holds two packed 16-bit values. For immediates,
    // LLVM's AMDGPU disassembler pre-resolves narrow-width operands to the
    // 16-bit value in the low half of the MCOperand immediate; the helper
    // therefore ignores OP_SEL_0 for non-register narrow sources.
    Value *s0 = readMixF32Src(ctx, op, 0, narrowTy, mods[0], cvtName);
    Value *s1 = readMixF32Src(ctx, op, 1, narrowTy, mods[1], cvtName);
    Value *s2 = readMixF32Src(ctx, op, 2, narrowTy, mods[2], cvtName);
    Function *fmaFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::fma, {ctx.f32Ty});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateCall(fmaFn, {s0, s1, s2}, "fma_mix"),
                       ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- v_cndmask_b32 (VOP2 or VOP3 — srcMap skips modifiers) ----
  case CanonicalOp::V_CNDMASK_B32: {
    ParsedReg dest = op.dst();
    Value *src0 = op.src(0);
    Value *src1 = op.src(1);
    Value *cond = nullptr;
    if (op.nSrcs() >= 3 && di.isReg(op.srcIdx(2))) {
      ParsedReg condReg =
          ctx.parseReg(di.getReg(op.srcIdx(2)), op.srcIdx(2));
      if (condReg.kind == ParsedReg::SGPR) {
        // Preferred path: a V_CMP_*_e64 in the current BB wrote this
        // SGPR and no intervening scalar write has invalidated the
        // cached per-lane `i1`. Use the `i1` directly — it carries
        // the full target-hardware ballot without the cross-widening
        // narrow-write information loss (the SGPR itself holds only
        // the source-width-truncated 32-bit projection). See
        // hotswap/docs/sgpr-wave-mask-translation.md section 3.1 for
        // the full contract and
        // `RaiseContext::lastSgprWaveMaskI1` for the invariants that
        // make this lookup sound.
        if (Value *freshCmp = ctx.lookupSgprWaveMaskI1(condReg.baseIdx)) {
          cond = freshCmp;
        } else {
          // Fallback: no fresh V_CMP writer in this BB (or the cache
          // was invalidated by a scalar SGPR write, or we crossed a
          // BB boundary). Route through the projection's per-lane
          // extractor, mirroring `readVCCAsWaveMask`'s consumer
          // symmetry. This path is correct for same-wave and
          // modulo-replication same-width cases, and lossy only in
          // the documented wave32 -> wave64 cross-widening narrow-
          // write case (where recovering the upper-half lanes'
          // compare results is impossible from the 32-bit SGPR —
          // those bits were destroyed at the writer's truncate).
          Value *condVal = ctx.isa.isWave32()
                               ? ctx.regs.loadSGPR32(ctx.B, condReg.baseIdx)
                               : ctx.regs.loadSGPR64(ctx.B, condReg.baseIdx);
          Value *fallback =
              ctx.projection.extractLaneBitFromWaveMask(ctx.B, condVal);
          // Cross-BB path: prefer the memory-backed shadow if valid.
          // This avoids carrying non-dominating `i1` SSA values across
          // blocks while still preserving the full EXEC-width compare mask.
          if (Value *shadowValid = ctx.loadSgprWaveMaskValid(condReg.baseIdx)) {
            Value *shadowExec = ctx.loadSgprWaveMaskExec(condReg.baseIdx);
            Value *shadowI1 =
                ctx.projection.extractLaneBitFromWaveMask(ctx.B, shadowExec);
            cond = ctx.B.CreateSelect(shadowValid, shadowI1, fallback,
                                      "sgpr_mask_shadow_sel");
          } else {
            cond = fallback;
          }
        }
      } else {
        cond = ctx.regs.loadVCC(ctx.B);
      }
    }
    if (!cond) cond = ctx.regs.loadVCC(ctx.B);
    ctx.writeReg32(dest, ctx.B.CreateSelect(cond, src1, src0, "cndmask"));
    hr.handled = true;
    return hr;
  }

  default:
    break;
  }
  return hr;
}

} // namespace transpiler
