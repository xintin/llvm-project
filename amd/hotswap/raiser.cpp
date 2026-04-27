#include "raiser.hpp"
#include "amdgpu_formats.hpp"
#include "code_object_utils.hpp"
#include "decode.hpp"
#include "semop.hpp"
#include "isa_profile.hpp"
#include "decoded_inst.hpp"
#include "parsed_reg.hpp"

#include "mc_state.hpp"
#include "opcode_map.hpp"
#include "Utils/AMDGPUBaseInfo.h"
#include "reg_file.hpp"
#include "kernarg_layout.hpp"
#include "raise_context.hpp"
#include "sem_op_attrs.hpp"
#include "setpc_analysis.hpp"
#include "user_sgpr_layout.hpp"
#include "wave_projection.hpp"
#include "wave_size_obstruction.hpp"
#include "handlers.hpp"
#include "rewrite_cross_lane_divergent.hpp"
#include "c5_predicate_chain_classifier.hpp"
#include "tdm_runtime.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <map>
#include <utility>

#define DEBUG_TYPE "wave-projection"

using namespace llvm;

namespace transpiler {

namespace {

namespace HsaKernelDispatchPacket {
constexpr unsigned WorkgroupSizeXOffset = 4;
constexpr unsigned WorkgroupSizeYOffset = 6;
constexpr unsigned WorkgroupSizeZOffset = 8;
constexpr unsigned GridSizeXOffset = 12;
constexpr unsigned GridSizeYOffset = 16;
constexpr unsigned GridSizeZOffset = 20;

unsigned workgroupSizeOffset(unsigned dim) {
  switch (dim) {
  case 0:
    return WorkgroupSizeXOffset;
  case 1:
    return WorkgroupSizeYOffset;
  case 2:
    return WorkgroupSizeZOffset;
  default:
    report_fatal_error("invalid HSA dispatch-packet workgroup-size dimension");
  }
}

unsigned gridSizeOffset(unsigned dim) {
  switch (dim) {
  case 0:
    return GridSizeXOffset;
  case 1:
    return GridSizeYOffset;
  case 2:
    return GridSizeZOffset;
  default:
    report_fatal_error("invalid HSA dispatch-packet grid-size dimension");
  }
}
} // namespace HsaKernelDispatchPacket

enum class ThreadLoopDecision {
  NotApplicable,
  EligibleButGateOff,
  EligibleAndGateOn,
  Ineligible,
};

struct ThreadLoopDecisionResult {
  ThreadLoopDecision decision = ThreadLoopDecision::NotApplicable;
  std::string reason;
};

ThreadLoopDecisionResult decideThreadLoopFallback(unsigned sourceWaveSize,
                                                  unsigned targetWaveSize,
                                                  bool sgprForcedRefusal,
                                                  bool threadLoopEligible) {
  if (!sgprForcedRefusal)
    return {ThreadLoopDecision::NotApplicable, "no SGPR-forced refusal"};
  if (!threadLoopEligible) {
    return {ThreadLoopDecision::Ineligible,
            "SGPR-forced sink is outside the proven readlane/writelane -> "
            "explicit readfirstlane ThreadLoop class"};
  }
  if (targetWaveSize <= sourceWaveSize) {
    return {ThreadLoopDecision::Ineligible,
            "thread-loop fallback is cross-widen-only"};
  }
  if ((targetWaveSize % sourceWaveSize) != 0) {
    return {ThreadLoopDecision::Ineligible,
            "target wave size is not an integer multiple of source wave size"};
  }
  // Graduation gate for the narrow SGPR-forced post-raise refusal class.
  //
  // Objective trigger:
  //   * the SSA use-chain classifier has already refused a cross-widening
  //     writelane/readlane rewrite because the value flows into an explicit
  //     `llvm.amdgcn.readfirstlane` consumer; and
  //   * the target wave size is an integer multiple of the source wave size.
  //
  // This does not widen the rewrite allow-list. The original refusal remains
  // the proof obligation: only after the classifier names the proven
  // readfirstlane sink do we retry under ThreadLoopProjection, with the
  // rewrite disabled so source-wave-scoped readlane / writelane /
  // readfirstlane lowering owns the boundary. Other SGPR-forced sinks
  // (scalar memory operands, inline asm, unknown calls) still refuse loudly.
  constexpr bool kThreadLoopAutoActivateSgprForcedCrossWiden = true;
  if (kThreadLoopAutoActivateSgprForcedCrossWiden)
    return {ThreadLoopDecision::EligibleAndGateOn,
            "SGPR-forced cross-widen refusal is covered by ThreadLoopProjection"};
  return {ThreadLoopDecision::EligibleButGateOff,
          "eligible but graduation gate is off"};
}

// Tracks one narrow question while raising a single instruction:
//
//   "Did this instruction prove that the source-ABI kernarg SGPR pair no
//    longer denotes the sentinel-modeled entry kernarg pointer?"
//
// This is intentionally not a general dataflow engine. It is a local
// provenance update over the SGPR dword table stored in RaiseContext, paired
// with the older constant-delta tracker for the original kernarg pair. Unknown
// provenance remains loud: handle_smem.cpp refuses loads through s[ka:ka+1]
// unless this updater proves either `Kernarg + const delta` or `NonKernarg`.
class KernargSgprProvenanceUpdater {
  using PairKind = RaiseContext::KernargPtrDelta::BaseKind;
  using DwordProv = RaiseContext::SgprKernargProvenance;
  using DwordKind = RaiseContext::SgprKernargProvenance::Kind;

public:
  KernargSgprProvenanceUpdater(RaiseContext &Ctx, const DecodedInst &DI,
                               OpResolver &Op, int KaLo)
      : Ctx(Ctx), DI(DI), Op(Op), KaLo(KaLo), KaHi(KaLo + 1) {
    HasLogicalDst = computeLogicalDst(LogicalDst);
  }

  void update() {
    const bool writesLo = writesKernargDword(KaLo);
    const bool writesHi = writesKernargDword(KaHi) || hasWideDefAtKernargLo();

    updateOriginalKernargPair(writesLo, writesHi);
    updateGenericSgprProvenance();
  }

private:
  RaiseContext &Ctx;
  const DecodedInst &DI;
  OpResolver &Op;
  int KaLo = -1;
  int KaHi = -1;
  ParsedReg LogicalDst;
  bool HasLogicalDst = false;

  static bool semOpHasLogicalDst(SemOp Sop) {
    switch (Sop) {
    case SemOp::S_GETREG_B32:
    case SemOp::S_MOV_B32:
    case SemOp::S_MOV_B64:
    case SemOp::S_AND_B32:
    case SemOp::S_OR_B32:
    case SemOp::S_XOR_B32:
    case SemOp::S_ANDN2_B32:
    case SemOp::S_ORN2_B32:
    case SemOp::S_NAND_B32:
    case SemOp::S_NOR_B32:
    case SemOp::S_XNOR_B32:
    case SemOp::S_ADD_U32:
    case SemOp::S_ADDC_U32:
    case SemOp::S_SUB_U32:
    case SemOp::S_SUBB_U32:
    case SemOp::S_MUL_I32:
    case SemOp::S_MUL_HI_U32:
    case SemOp::S_MUL_HI_I32:
    case SemOp::S_LSHL_B32:
    case SemOp::S_LSHR_B32:
    case SemOp::S_ASHR_I32:
    case SemOp::S_BFE_U32:
    case SemOp::S_BFE_I32:
    case SemOp::S_BFM_B32:
    case SemOp::S_CSELECT_B32:
    case SemOp::S_NOT_B64:
    case SemOp::S_CMOV_B64:
    case SemOp::S_AND_B64:
    case SemOp::S_OR_B64:
    case SemOp::S_XOR_B64:
    case SemOp::S_ANDN2_B64:
    case SemOp::S_ORN2_B64:
    case SemOp::S_NAND_B64:
    case SemOp::S_NOR_B64:
    case SemOp::S_XNOR_B64:
    case SemOp::S_LSHL_B64:
    case SemOp::S_LSHR_B64:
    case SemOp::S_ASHR_I64:
    case SemOp::S_BFM_B64:
    case SemOp::S_CSELECT_B64:
    case SemOp::S_BITSET0_B64:
    case SemOp::S_BITSET1_B64:
    case SemOp::S_ADD_NC_U64:
    case SemOp::S_SUB_NC_U64:
    case SemOp::S_MUL_U64:
    case SemOp::S_LOAD_B32:
    case SemOp::S_LOAD_B64:
    case SemOp::S_LOAD_B96:
    case SemOp::S_LOAD_B128:
    case SemOp::S_LOAD_B256:
    case SemOp::S_LOAD_B512:
    case SemOp::S_LOAD_U8:
    case SemOp::S_LOAD_I8:
    case SemOp::S_LOAD_U16:
    case SemOp::S_LOAD_I16:
      return true;
    default:
      return false;
    }
  }

  static bool semOpWritesScalarPair(SemOp Sop) {
    switch (Sop) {
    case SemOp::S_MOV_B64:
    case SemOp::S_NOT_B64:
    case SemOp::S_CMOV_B64:
    case SemOp::S_AND_B64:
    case SemOp::S_OR_B64:
    case SemOp::S_XOR_B64:
    case SemOp::S_ANDN2_B64:
    case SemOp::S_ORN2_B64:
    case SemOp::S_NAND_B64:
    case SemOp::S_NOR_B64:
    case SemOp::S_XNOR_B64:
    case SemOp::S_LSHL_B64:
    case SemOp::S_LSHR_B64:
    case SemOp::S_ASHR_I64:
    case SemOp::S_BFM_B64:
    case SemOp::S_CSELECT_B64:
    case SemOp::S_BITSET0_B64:
    case SemOp::S_BITSET1_B64:
    case SemOp::S_ADD_NC_U64:
    case SemOp::S_SUB_NC_U64:
    case SemOp::S_MUL_U64:
      return true;
    default:
      return false;
    }
  }

  static bool isSmemLoadResult(SemOp Sop) {
    switch (Sop) {
    case SemOp::S_LOAD_B32:
    case SemOp::S_LOAD_B64:
    case SemOp::S_LOAD_B96:
    case SemOp::S_LOAD_B128:
    case SemOp::S_LOAD_B256:
    case SemOp::S_LOAD_B512:
    case SemOp::S_LOAD_U8:
    case SemOp::S_LOAD_I8:
    case SemOp::S_LOAD_U16:
    case SemOp::S_LOAD_I16:
      return true;
    default:
      return false;
    }
  }

  static int smemLoadDwords(SemOp Sop) {
    switch (Sop) {
    case SemOp::S_LOAD_B64:
      return 2;
    case SemOp::S_LOAD_B96:
      return 3;
    case SemOp::S_LOAD_B128:
      return 4;
    case SemOp::S_LOAD_B256:
      return 8;
    case SemOp::S_LOAD_B512:
      return 16;
    default:
      return 1;
    }
  }

  bool computeLogicalDst(ParsedReg &Out) {
    if (!semOpHasLogicalDst(DI.semOp))
      return false;
    Out = Op.dst();
    return true;
  }

  int semanticDefDwords(ParsedReg PR) const {
    if (PR.kind != ParsedReg::SGPR)
      return 0;
    if (PR.width >= 2)
      return PR.width;
    if (semOpWritesScalarPair(DI.semOp) || DI.semOp == SemOp::S_LOAD_B64)
      return 2;
    if (isSmemLoadResult(DI.semOp))
      return smemLoadDwords(DI.semOp);
    return 1;
  }

  bool writesKernargDword(int BaseIdx) {
    if (HasLogicalDst && LogicalDst.kind == ParsedReg::SGPR) {
      int Lo = LogicalDst.baseIdx;
      int Hi = LogicalDst.baseIdx + LogicalDst.width - 1;
      if (Lo <= BaseIdx && BaseIdx <= Hi)
        return true;
    }
    for (unsigned D = 0; D < DI.numDefs; ++D) {
      if (!DI.isReg(D))
        continue;
      ParsedReg PR = Ctx.parseReg(DI.getReg(D), D);
      if (PR.kind != ParsedReg::SGPR)
        continue;
      int Lo = PR.baseIdx;
      int Hi = PR.baseIdx + PR.width - 1;
      if (Lo <= BaseIdx && BaseIdx <= Hi)
        return true;
    }
    return false;
  }

  bool hasWideDefAtKernargLo() {
    bool DefStartsAtLo = false;
    if (HasLogicalDst && LogicalDst.kind == ParsedReg::SGPR &&
        LogicalDst.baseIdx == KaLo) {
      if (LogicalDst.width >= 2)
        return true;
      DefStartsAtLo = true;
    }
    for (unsigned D = 0; D < DI.numDefs; ++D) {
      if (!DI.isReg(D))
        continue;
      ParsedReg PR = Ctx.parseReg(DI.getReg(D), D);
      if (PR.kind == ParsedReg::SGPR && PR.baseIdx == KaLo) {
        if (PR.width >= 2)
          return true;
        DefStartsAtLo = true;
      }
    }
    if (!DefStartsAtLo)
      return false;
    return semOpWritesScalarPair(DI.semOp) ||
           DI.semOp == SemOp::S_LOAD_B64 || DI.semOp == SemOp::S_LOAD_B96 ||
           DI.semOp == SemOp::S_LOAD_B128 ||
           DI.semOp == SemOp::S_LOAD_B256 ||
           DI.semOp == SemOp::S_LOAD_B512;
  }

  bool isKernargHighCanonicalMask() {
    if (DI.semOp != SemOp::S_AND_B32 || DI.numSrcs < 2 ||
        DI.numDefs < 1 || !DI.isReg(0))
      return false;
    ParsedReg Dst = Ctx.parseReg(DI.getReg(0), 0);
    if (Dst.kind != ParsedReg::SGPR || Dst.baseIdx != KaHi)
      return false;

    unsigned S0Idx = DI.srcMap[0];
    unsigned S1Idx = DI.srcMap[1];
    auto SrcIsKernargHi = [&](unsigned Idx) {
      if (!DI.isReg(Idx))
        return false;
      ParsedReg Src = Ctx.parseReg(DI.getReg(Idx), Idx);
      return Src.kind == ParsedReg::SGPR && Src.baseIdx == KaHi;
    };
    auto SrcIsLow16Mask = [&](unsigned Idx) {
      return DI.isImm(Idx) &&
             static_cast<uint64_t>(DI.getImm(Idx)) == 0xffffu;
    };
    return (SrcIsKernargHi(S0Idx) && SrcIsLow16Mask(S1Idx)) ||
           (SrcIsKernargHi(S1Idx) && SrcIsLow16Mask(S0Idx));
  }

  void markKernargUnknown() {
    Ctx.kernargPtrDelta.baseKind = PairKind::Unknown;
    Ctx.kernargPtrDelta.delta = 0;
    Ctx.kernargPtrDelta.valid = false;
    Ctx.kernargPtrDelta.pendingLow = false;
    Ctx.kernargPtrDelta.pendingLowDelta = 0;
    Ctx.setKernargPairProvenance(PairKind::Unknown);
  }

  void markNonKernarg() {
    Ctx.kernargPtrDelta.baseKind = PairKind::NonKernarg;
    Ctx.kernargPtrDelta.delta = 0;
    Ctx.kernargPtrDelta.valid = false;
    Ctx.kernargPtrDelta.pendingLow = false;
    Ctx.kernargPtrDelta.pendingLowDelta = 0;
    Ctx.setKernargPairProvenance(PairKind::NonKernarg);
  }

  void markTrackedKernarg(int64_t Delta) {
    Ctx.kernargPtrDelta.baseKind = PairKind::Kernarg;
    Ctx.kernargPtrDelta.delta = Delta;
    Ctx.kernargPtrDelta.valid = true;
    Ctx.kernargPtrDelta.pendingLow = false;
    Ctx.kernargPtrDelta.pendingLowDelta = 0;
    Ctx.setKernargPairProvenance(PairKind::Kernarg, Delta);
  }

  bool sgprOverlapsKernargPair(ParsedReg PR) const {
    if (PR.kind != ParsedReg::SGPR)
      return false;
    int Lo = PR.baseIdx;
    int Hi = PR.baseIdx + PR.width - 1;
    return Lo <= KaHi && Hi >= KaLo;
  }

  DwordProv dwordProvenance(int Idx) const {
    if (Idx < 0 ||
        static_cast<size_t>(Idx) >= Ctx.sgprKernargProvenance.size())
      return {};
    return Ctx.sgprKernargProvenance[Idx];
  }

  bool regIsDefinitelyNonKernarg(ParsedReg PR) const {
    if (PR.kind != ParsedReg::SGPR)
      return true;
    int Width = PR.width;
    if (Width < 2 && semOpWritesScalarPair(DI.semOp))
      Width = 2;
    for (int I = 0; I < Width; ++I) {
      if (dwordProvenance(PR.baseIdx + I).kind != DwordKind::NonKernarg)
        return false;
    }
    return true;
  }

  bool allSourcesDefinitelyNonKernarg() {
    for (unsigned S = 0; S < DI.numSrcs; ++S) {
      unsigned Idx = DI.srcMap[S];
      if (!DI.isReg(Idx))
        continue;
      if (!regIsDefinitelyNonKernarg(Ctx.parseReg(DI.getReg(Idx), Idx)))
        return false;
    }
    return true;
  }

  void setDwordProv(int Idx, DwordProv P) {
    if (Idx < 0 ||
        static_cast<size_t>(Idx) >= Ctx.sgprKernargProvenance.size())
      return;
    Ctx.sgprKernargProvenance[Idx] = P;
  }

  void setDwordKind(int Idx, DwordKind Kind) {
    DwordProv P;
    P.kind = Kind;
    setDwordProv(Idx, P);
  }

  void setOriginalPairDwordKind(int Idx, DwordKind Kind) {
    setDwordKind(Idx, Kind);
  }

  void refreshOriginalPairFromDwords() {
    auto Lo = dwordProvenance(KaLo);
    auto Hi = dwordProvenance(KaHi);
    if (Lo.kind == DwordKind::NonKernarg &&
        Hi.kind == DwordKind::NonKernarg) {
      Ctx.kernargPtrDelta.baseKind = PairKind::NonKernarg;
      Ctx.kernargPtrDelta.valid = false;
      Ctx.kernargPtrDelta.pendingLow = false;
      Ctx.kernargPtrDelta.pendingLowDelta = 0;
      Ctx.kernargPtrDelta.delta = 0;
    }
  }

  bool smemLoadMaterializesFullPair(bool WritesLo, bool WritesHi) const {
    if (!(WritesLo && WritesHi))
      return false;
    switch (DI.semOp) {
    case SemOp::S_LOAD_B64:
    case SemOp::S_LOAD_B96:
    case SemOp::S_LOAD_B128:
    case SemOp::S_LOAD_B256:
    case SemOp::S_LOAD_B512:
      return true;
    default:
      return false;
    }
  }

  std::pair<bool, int64_t> classifyConstInPlaceAdd(int DstBase) {
    if (DI.numSrcs < 2)
      return {false, 0};
    unsigned S0Idx = DI.srcMap[0];
    unsigned S1Idx = DI.srcMap[1];
    const bool S0Reg = DI.isReg(S0Idx);
    const bool S1Reg = DI.isReg(S1Idx);
    const bool S0Imm = DI.isImm(S0Idx);
    const bool S1Imm = DI.isImm(S1Idx);

    auto RegMatchesDst = [&](unsigned Idx) {
      ParsedReg PR = Ctx.parseReg(DI.getReg(Idx), Idx);
      return PR.kind == ParsedReg::SGPR && PR.baseIdx == DstBase;
    };

    if (S0Reg && S1Imm && RegMatchesDst(S0Idx))
      return {true, DI.getImm(S1Idx)};
    if (S1Reg && S0Imm && RegMatchesDst(S1Idx))
      return {true, DI.getImm(S0Idx)};
    return {false, 0};
  }

  void updateOriginalKernargPair(bool WritesLo, bool WritesHi) {
    if (!(WritesLo || WritesHi)) {
      if (Ctx.kernargPtrDelta.valid && Ctx.kernargPtrDelta.pendingLow &&
          DI.defsSCC)
        markKernargUnknown();
      return;
    }

    const bool WritesFullPair = WritesLo && WritesHi;
    const bool FullIndependentOverwrite =
        WritesFullPair && allSourcesDefinitelyNonKernarg();
    if (FullIndependentOverwrite ||
        smemLoadMaterializesFullPair(WritesLo, WritesHi)) {
      markNonKernarg();
      return;
    }
    if (!WritesFullPair && allSourcesDefinitelyNonKernarg()) {
      Ctx.kernargPtrDelta.baseKind = PairKind::Unknown;
      Ctx.kernargPtrDelta.delta = 0;
      Ctx.kernargPtrDelta.valid = false;
      Ctx.kernargPtrDelta.pendingLow = false;
      Ctx.kernargPtrDelta.pendingLowDelta = 0;
      if (WritesLo)
        setOriginalPairDwordKind(KaLo, DwordKind::NonKernarg);
      if (WritesHi)
        setOriginalPairDwordKind(KaHi, DwordKind::NonKernarg);
      refreshOriginalPairFromDwords();
      return;
    }
    if (!Ctx.kernargPtrDelta.valid) {
      markKernargUnknown();
      return;
    }
    if (WritesHi && !WritesLo && isKernargHighCanonicalMask() &&
        !Ctx.kernargPtrDelta.pendingLow)
      return;
    if (WritesLo && !WritesHi && DI.semOp == SemOp::S_ADD_U32) {
      auto [Ok, Imm] = classifyConstInPlaceAdd(KaLo);
      if (!Ok) {
        markKernargUnknown();
        return;
      }
      if (Ctx.kernargPtrDelta.pendingLow) {
        markKernargUnknown();
        return;
      }
      Ctx.kernargPtrDelta.pendingLow = true;
      Ctx.kernargPtrDelta.pendingLowDelta = Imm;
      Ctx.setKernargPairProvenance(PairKind::Unknown);
      return;
    }
    if (WritesHi && !WritesLo && DI.semOp == SemOp::S_ADDC_U32 &&
        Ctx.kernargPtrDelta.pendingLow) {
      auto [Ok, Imm] = classifyConstInPlaceAdd(KaHi);
      if (Ok && Imm == 0) {
        markTrackedKernarg(Ctx.kernargPtrDelta.delta +
                           Ctx.kernargPtrDelta.pendingLowDelta);
      } else {
        markKernargUnknown();
      }
      return;
    }
    markKernargUnknown();
  }

  void setPairFromBaseKind(int Dst, PairKind Kind, int64_t Delta = 0) {
    switch (Kind) {
    case PairKind::Kernarg: {
      DwordProv Lo, Hi;
      Lo.kind = DwordKind::Kernarg;
      Lo.delta = Delta;
      Lo.subDword = 0;
      Hi.kind = DwordKind::Kernarg;
      Hi.delta = Delta;
      Hi.subDword = 1;
      setDwordProv(Dst, Lo);
      setDwordProv(Dst + 1, Hi);
      break;
    }
    case PairKind::NonKernarg:
      setDwordKind(Dst, DwordKind::NonKernarg);
      setDwordKind(Dst + 1, DwordKind::NonKernarg);
      break;
    case PairKind::Unknown:
      setDwordKind(Dst, DwordKind::Unknown);
      setDwordKind(Dst + 1, DwordKind::Unknown);
      break;
    }
  }

  std::pair<PairKind, int64_t> pairProvenanceFromReg(ParsedReg PR) const {
    if (PR.kind != ParsedReg::SGPR)
      return {PairKind::Unknown, 0};
    auto Lo = dwordProvenance(PR.baseIdx);
    auto Hi = dwordProvenance(PR.baseIdx + 1);
    if (Lo.kind == DwordKind::NonKernarg &&
        Hi.kind == DwordKind::NonKernarg)
      return {PairKind::NonKernarg, 0};
    if (Lo.kind == DwordKind::Kernarg && Hi.kind == DwordKind::Kernarg &&
        Lo.subDword == 0 && Hi.subDword == 1 && Lo.delta == Hi.delta)
      return {PairKind::Kernarg, Lo.delta};
    return {PairKind::Unknown, 0};
  }

  std::pair<PairKind, int64_t> pairProvenanceFromSrc0() {
    if (DI.numSrcs < 1)
      return {PairKind::Unknown, 0};
    unsigned Idx = DI.srcMap[0];
    if (!DI.isReg(Idx))
      return {PairKind::NonKernarg, 0};
    return pairProvenanceFromReg(Ctx.parseReg(DI.getReg(Idx), Idx));
  }

  DwordProv dwordProvenanceFromSrc0() {
    DwordProv P;
    if (DI.numSrcs < 1)
      return P;
    unsigned Idx = DI.srcMap[0];
    if (!DI.isReg(Idx)) {
      P.kind = DwordKind::NonKernarg;
      return P;
    }
    ParsedReg Src = Ctx.parseReg(DI.getReg(Idx), Idx);
    if (Src.kind != ParsedReg::SGPR) {
      P.kind = DwordKind::NonKernarg;
      return P;
    }
    return dwordProvenance(Src.baseIdx);
  }

  void updateGenericDefProvenance(ParsedReg Def) {
    int DefDwords = semanticDefDwords(Def);
    if (DefDwords == 0 || sgprOverlapsKernargPair(Def))
      return;

    if (isSmemLoadResult(DI.semOp)) {
      for (int I = 0; I < DefDwords; ++I)
        setDwordKind(Def.baseIdx + I, DwordKind::NonKernarg);
      return;
    }
    if (DI.semOp == SemOp::S_MOV_B64 && DefDwords >= 2) {
      auto [Kind, Delta] = pairProvenanceFromSrc0();
      setPairFromBaseKind(Def.baseIdx, Kind, Delta);
      return;
    }
    if (DI.semOp == SemOp::S_MOV_B32 && DefDwords == 1) {
      setDwordProv(Def.baseIdx, dwordProvenanceFromSrc0());
      return;
    }

    DwordKind Kind =
        allSourcesDefinitelyNonKernarg() ? DwordKind::NonKernarg
                                         : DwordKind::Unknown;
    for (int I = 0; I < DefDwords; ++I)
      setDwordKind(Def.baseIdx + I, Kind);
  }

  void updateGenericSgprProvenance() {
    for (unsigned D = 0; D < DI.numDefs; ++D) {
      if (DI.isReg(D))
        updateGenericDefProvenance(Ctx.parseReg(DI.getReg(D), D));
    }
    if (HasLogicalDst)
      updateGenericDefProvenance(LogicalDst);
  }
};

} // namespace

// parseReg, readOp32/64/ExecWidth, and OpResolver are in raise_context.hpp/cpp
// instructionWritesEXEC and the cross-wave gate live in wave_projection.hpp/cpp
// RaiseFailure + reasonString are in raise_failure.hpp/cpp

// ============================================================================
// Main raising function
// ============================================================================

static RaiseResult raiseToIRImpl(const std::vector<uint8_t> &textBytes,
                                 const std::string &sourceISA,
                                 const std::string &kernelName,
                                 const KernelMeta &meta,
                                 uint64_t kernelOffset,
                                 const std::string &compilationTargetISA,
                                 bool enableWritelaneRewrite,
                                 bool enableWaveNative,
                                 bool forceThreadLoopProjection,
                                 bool suppressC5ForThreadLoopRoute) {
  RaiseResult result;

  // NOTE. The `HSA_SALMON_WAVE_NATIVE=1` process-environment override
  // that lived here through the empirical graduation sweep (pre-
  // 2026-04-21) has been removed now that `enableWaveNative`
  // defaults to `true`. The override served one purpose — flipping
  // every call-site's projection without editing each caller —
  // which is no longer needed. Keeping it around would subtly
  // break the opt-OUT path: `--disable-wave-native` on
  // `raise_cli` (and `enableWaveNative=false` on programmatic
  // callers) are how lit fixtures and operators pin MODREP for
  // projection-specific debugging, and a silent env-var that
  // unconditionally flips to WaveNative would defeat that. If
  // future evidence needs a global toggle, add a proper
  // `PipelineConfig` field rather than re-introducing the env var.

  MCState mc;
  initMCState(mc, sourceISA);

  ISAProfile isa = ISAProfile::fromSubtarget(*mc.subtargetInfo);
  // When the caller does not specify a distinct compilation target we raise
  // in place and reuse the source profile; otherwise we spin up a throwaway
  // MCSubtargetInfo just to snapshot the target's feature bits.
  ISAProfile targetIsa = isa;
  std::unique_ptr<MCSubtargetInfo> targetSTI;
  if (!compilationTargetISA.empty()) {
    targetSTI = buildSubtargetInfo(*mc.target, compilationTargetISA);
    targetIsa = ISAProfile::fromSubtarget(*targetSTI);
  }

  // LLVMContext + common IR types are created here (earlier than they used
  // to be) so the WaveProjection has access to i32/i64 before the cross-
  // wave gate runs. The module is still created lazily in Phase 2 so
  // early-return paths (pre-translation aborts) don't leave behind a
  // half-built module.
  result.ctx = std::make_unique<LLVMContext>();
  LLVMContext &C = *result.ctx;
  auto *i32Ty = Type::getInt32Ty(C);
  auto *i64Ty = Type::getInt64Ty(C);

  // Projection choice.
  //
  // `ModuloReplicationProjection` is the long-standing default: it fans
  // each target lane onto `lane_id mod W_src` of the source EXEC mask
  // and truncates cross-wave ballots to source width. Correct under
  // the wave-size-obliviousness theorem (hotswap/docs/wave-size-
  // translation.md §6); insufficient for kernels whose WMMA → MFMA
  // redistribute / collect pipeline needs hardware EXEC = -1 on the
  // upper half of the Wave64 target (lanes 32..63 would otherwise
  // never update their MFMA destination VGPRs — see the file-header
  // comment in `wmma_lowering.cpp`).
  //
  // `WaveNativeProjection` is the opt-in alternative for wave32
  // source → wave64 target. Its `emitInitialExec` calls
  // `@llvm.amdgcn.init_whole_wave` at kernel entry to force hardware
  // EXEC = -1 for the whole kernel body while saving the original
  // per-lane active mask into the (widened) EXEC alloca; every VGPR
  // write / memory store / LDS op already routes through
  // `emitUnderExec`, which rematerialises the per-lane predicate at
  // each side-effect site. The direction gate inside the
  // `WaveNativeProjection` constructor enforces that this projection
  // is only instantiated when `isa.isWave32() && !targetIsa.isWave32()`
  // — other directions fatal-error loudly to prevent a decider bug
  // from silently picking an unsupported shape.
  //
  // Phantom-lane fallback to MODREP.  WaveNative's `init_whole_wave`
  // sets hardware EXEC = -1 and relies on SPE `emitUnderExec`
  // diamonds (gated by `saved_exec`) to keep inactive source lanes
  // from committing side effects.  That model is correct when every
  // target-wavefront lane has a source-kernel workitem — i.e. when
  // the HSACO's `max_flat_workgroup_size` is at least
  // `targetWaveSize` so every launch fills the target wave.  When
  // `max_flat_workgroup_size < targetWaveSize` (the phantom-lane
  // regime, e.g. Triton's `num_warps=1` kernels whose source WG is
  // 32 on wave32 compiled for a wave64 target), the "extra" target
  // lanes have no source workitem: their `workitem.id.x()` is their
  // hardware lane index (e.g. 32..63 for a 32-thread block on
  // wave64), their VGPRs hold undef / dispatcher state, and their
  // cross-lane ops (`ds_bpermute`, `ds_swizzle`, `permlane*`) read
  // from / contribute to actively-masked source lanes with
  // undef-derived values — producing addresses that fault on
  // subsequent SPE-gated loads (the active lane's pointer
  // arithmetic picks up undef data through a cross-lane op, then
  // the gated load fires with that poisoned address).  Empirically
  // surfaced by `compare_correctness`'s `matmul_fp16` /
  // `matmul_fp16_16x16` Triton recipes (HIP error 700 on every
  // shape under WaveNative; bumping `num_warps` to 2 fills the
  // target wavefront and eliminates the fault, confirming the
  // phantom-lane attribution).
  //
  // `ModuloReplicationProjection` leaves hardware EXEC at the
  // dispatcher's boot state (the source-wave-sized active mask,
  // with the target wave's upper lanes inactive) and uses
  // `lane_id mod W_src` to project the target mask onto the source
  // EXEC alloca.  Under MODREP, phantom lanes are hardware-inactive
  // for the entire kernel body — every ISA instruction (VALU,
  // cross-lane, memory, control flow) is HW-EXEC-masked — so
  // undef-VGPR contamination can't escape into active lanes.  The
  // trade-off is that MODREP cannot express WMMA → MFMA layout
  // transposes that need all 64 target lanes active (see
  // `wmma_lowering.cpp`); those kernels will refuse at lift time
  // rather than silently running wrong.  That's the principled
  // outcome for the phantom-lane regime.
  const bool phantomLaneRegime =
      meta.maxFlatWorkgroupSize > 0 &&
      static_cast<unsigned>(meta.maxFlatWorkgroupSize) < targetIsa.waveSize;
  const bool useThreadLoop = forceThreadLoopProjection;
  const bool useWaveNative = !useThreadLoop && enableWaveNative &&
                             isa.isWave32() && !targetIsa.isWave32() &&
                             !phantomLaneRegime;
  std::unique_ptr<WaveProjection> projectionPtr;
  if (useThreadLoop) {
    projectionPtr = std::make_unique<ThreadLoopProjection>(
        isa, targetIsa, i32Ty, i64Ty);
    errs() << "transpiler: kernel '" << kernelName
           << "' selected ThreadLoopProjection (analysis-triggered "
              "SGPR-forced cross-lane route; writelane/readlane rewrite "
              "disabled for this retry)\n";
  } else if (useWaveNative) {
    projectionPtr = std::make_unique<WaveNativeProjection>(isa, targetIsa,
                                                             i32Ty, i64Ty);
  } else {
    projectionPtr = std::make_unique<ModuloReplicationProjection>(
        isa, targetIsa, i32Ty, i64Ty);
  }
  WaveProjection &projection = *projectionPtr;

  if (!useThreadLoop && enableWaveNative && phantomLaneRegime && isa.isWave32() &&
      !targetIsa.isWave32()) {
    // Log the fallback so operators can trace which kernels moved to
    // MODREP and why.  A regression that silently flips WaveNative's
    // selection on a phantom-lane kernel would then (re-)produce the
    // HIP-700 miscompile this fallback guards against.
    errs() << "transpiler: kernel '" << kernelName
           << "' is in phantom-lane regime (max_flat_workgroup_size="
           << meta.maxFlatWorkgroupSize << " < target wavefront width="
           << targetIsa.waveSize
           << "); falling back to ModuloReplicationProjection even "
              "though enableWaveNative=true, so phantom target lanes "
              "stay hardware-inactive and their undef-VGPR state "
              "cannot contaminate active-lane pointer arithmetic via "
              "cross-lane ops. See the block comment above in "
              "`raiser.cpp` for the full rationale.\n";
  }

  // Build opcode → SemOp map from MCInstrInfo
  OpcodeMap opcMap;
  opcMap.build(*mc.instrInfo);

  // Fail loudly if any MFMA-format SemOp is missing a handler row. Cheap
  // startup walk that catches table drift before any kernel is lifted.
  verifyMFMACoverage(*mc.instrInfo, opcMap);

  // Startup invariant: every MC opcode that implicitly defines EXEC must
  // map to a SemOp that has `routesExecThroughStoreExec` set. Explicit-
  // operand EXEC writers (where EXEC is an operand value rather than a
  // TableGen def) stay the per-kernel Phase 1.5 gate's responsibility
  // since they depend on runtime operand values.
  verifyExecAttrCoverage(*mc.instrInfo, opcMap);

  // ==== Phase 1: Disassemble + identify block boundaries ====
  //
  // The decode loop (and its two LLVM-drift guards) lives in decode.cpp so
  // this function stays focused on IR emission. decodeKernel returns a
  // linearised instruction stream + the set of CFG block-start offsets.
  DecodeResult decoded =
      decodeKernel(mc, opcMap,
                   ArrayRef<uint8_t>(textBytes.data(), textBytes.size()),
                   kernelOffset);
  auto &insts = decoded.insts;
  auto &blockStarts = decoded.blockStarts;

  // ==== Phase 1.1: s_set_pc_i64 analysis ====
  //
  // Classify every s_set_pc_i64 site (Pattern A direct branch /
  // Pattern B subroutine return / Unresolvable) and discover the
  // extra basic-block leaders the indirect control-flow implies
  // (Pattern A targets + Pattern B return targets + the offset
  // immediately following each set-PC, which is otherwise unreachable
  // by linear fall-through). Merging the extra leaders into
  // `blockStarts` here is mandatory: Phase 3 only creates LLVM
  // BasicBlocks for offsets in this set, and the handler / call-site
  // rewrite both look up those BBs via `ctx.lookupBB`.
  // See setpc_analysis.hpp + semop.hpp's `S_SET_PC_I64` doc for the
  // analysis contract.
  SetPcAnalysis setpcAnalysis = analyseSetPC(insts, blockStarts, mc);
  for (uint64_t addr : setpcAnalysis.extraBlockStarts)
    blockStarts.insert(addr);

  result.totalCount = static_cast<int>(insts.size());

  {
    raw_string_ostream disOS(result.disasmText);
    for (const auto &di : insts) {
      disOS << format_hex_no_prefix(di.offset, 8) << ":  " << di.fullText
            << "\n";
    }
  }

  // ==== Phase 1.4: Cross-wave legacy diagnostic (LLVM_DEBUG) ====
  //
  // Kept as a fallback diagnostic under `-debug-only=wave-projection`;
  // the structured classifier in Phase 1.4.5 below is the primary
  // decision surface. See wave_projection.cpp for the text of the
  // legacy diagnostic.
  emitCrossWaveWarning(projection, mc, insts, sourceISA,
                       compilationTargetISA);

  // ==== Phase 1.4.5: Wave-size obstruction classifier
  // (hotswap/docs/wave-size-translation.md §7) ====
  //
  // The classifier walks the decoded instruction stream and tags every
  // site that violates the wave-size-obliviousness theorem (see
  // wave-size-translation.md §6 for the precise definition). The
  // decider then applies the 3-outcome procedure:
  //   (a) no obstructions, or every obstruction is covered by an
  //       implemented rewrite → emit modulo-replication.
  //   (b) at least one obstruction has a rewrite structurally
  //       recognised but not yet implemented (the "Pending rewrite"
  //       table in wave-size-translation.md §7) → refuse with a
  //       `CrossWaveShuffleRewritePending` diagnostic naming the P-item.
  //   (c) at least one obstruction has no rewrite in the decision
  //       procedure's unrewritable table → refuse with the kind-
  //       specific CrossWave* diagnostic (`CrossWaveLaneIdLeak`,
  //       `CrossWaveUnrewritableShuffle`, `CrossWaveReplicaRace`,
  //       `CrossWaveLanePredicatedExec`).
  //
  // Refusal diagnostics are written to `errs()` (user-visible) AND the
  // full per-site trace is routed through LLVM_DEBUG so operators can
  // inspect the oblivious/pass path under `-debug-only=wave-projection`
  // without recompiling.
  // Number of `WaveIdLiftScalarized` sites the classifier matched.
  // Needed after Phase 6.5 for the rewrite-pass safety net (see
  // below): when this is > 0, the rewrite pass is *expected* to have
  // rewritten at least one divergent writelane/readlane site; if it
  // rewrote zero, the oracle disagrees with the syntactic
  // classifier and we refuse post-raise rather than emit silently
  // unchanged IR that scalarises the divergent wave_id lift.
  unsigned classifierWaveIdLiftScalarizedSites = 0;
  {
    ObstructionReport report =
        buildObstructionReport(insts, mc, isa, targetIsa,
                               enableWritelaneRewrite);
    for (const auto &s : report.sites)
      if (s.kind == ObstructionKind::WaveIdLiftScalarized)
        ++classifierWaveIdLiftScalarizedSites;
    std::string trace = renderObstructionTrace(
        report, kernelName, sourceISA,
        compilationTargetISA.empty() ? sourceISA : compilationTargetISA,
        isa.waveSize, targetIsa.waveSize);
    LLVM_DEBUG(dbgs() << trace);
    if (report.hasUnrewritable() || report.hasPendingRewrite()) {
      RaiseFailure f = selectFailureFromReport(report);
      // The factory names the class in `format`; surface the full
      // trace in `detail` so raise_cli / batch_raise_test can carry
      // the per-site context forward without re-invoking the
      // classifier.
      if (!f.detail.empty())
        f.detail += "\n";
      f.detail += trace;
      // `format_hex(value, width)` prepends "0x" itself; do NOT add a
      // literal "0x" here or the output will read "0x0x...". Use
      // `format_hex_no_prefix` if a manual prefix is desired (the
      // trace-renderer below uses that variant).
      errs() << "transpiler: pre-translation abort: " << f.format
             << " on '" << f.mnemonic << "' at offset "
             << format_hex(f.offset, 1) << " \u2014 "
             << (report.firstUnrewritable()
                     ? "no rewrite in wave-size-translation.md "
                       "\u00a77's unrewritable table"
                     : "rewrite pending (wave-size-translation.md "
                       "\u00a77's pending-rewrite table)")
             << "\n"
             << trace;
      result.failure = std::move(f);
      return result;
    }
  }

  // ==== Phase 1.5: SPE A-level gate (EXEC-writer attribute check) ====
  //
  // SPE (SIMT Predicated Execution) is correct only when every runtime
  // change to EXEC either (a) propagates through the EXEC alloca via a
  // handler we have audited, or (b) follows the standard dataflow form
  // `exec = f(old_exec, sgprs, ...)` where `f` is a bitwise / shift /
  // move / compare-based scalar op — the IR's live EXEC value then
  // matches the hardware EXEC that the backend re-materialises when it
  // lowers our predicated-store diamonds back to v_cmpx / s_and_saveexec
  // pairs. Anything outside this set risks silently generating IR that
  // looks well-typed but diverges from hardware semantics.
  //
  // The allow-list lives as per-SemOp attributes in `sem_op_attrs.{hpp,
  // cpp}`; `verifyExecAttrCoverage` above already enforces it for
  // implicit-def EXEC writers at startup. This per-kernel scan covers
  // the remaining case: explicit-operand EXEC writers (e.g.
  // `s_mov_b32 exec_lo, s2`) where "writes EXEC" depends on the
  // runtime operand value rather than the MCInstrDesc alone.
  for (const DecodedInst &di : insts) {
    if (!instructionWritesEXEC(di, mc))
      continue;
    if (getSemOpAttrs(di.semOp).routesExecThroughStoreExec)
      continue;
    result.failure = RaiseFailure::speUnsafeExecWriter(di);
    errs() << "transpiler: pre-translation abort: '" << di.rawMnemonic
           << "' writes EXEC but its SemOp (" << semOpName(di.semOp)
           << ") is not marked routesExecThroughStoreExec. Auditing "
              "the handler path against SPE (lane-active predication "
              "assumption) is required before declaring the SemOp in "
              "the handler's get*Attrs() registration.\n";
    return result;
  }

  // ==== Phase 2: Build LLVM IR module + function ====
  // LLVMContext + i32/i64 were created earlier for the WaveProjection.
  result.module = std::make_unique<Module>("transpiler_module", C);
  Module &M = *result.module;
  M.setTargetTriple(Triple("amdgcn-amd-amdhsa"));

  TargetOptions opts;
  std::unique_ptr<TargetMachine> tm(mc.target->createTargetMachine(
      Triple("amdgcn-amd-amdhsa"),
      compilationTargetISA.empty() ? sourceISA : compilationTargetISA,
      "", opts, Reloc::PIC_));
  if (!tm) {
    errs() << "transpiler: Failed to create TargetMachine\n";
    result.failure = RaiseFailure::targetMachineCreationFailed();
    return result;
  }
  M.setDataLayout(tm->createDataLayout());

  auto *voidTy = Type::getVoidTy(C);
  auto *i1Ty = Type::getInt1Ty(C);
  auto *i8Ty = Type::getInt8Ty(C);
  auto *f32Ty = Type::getFloatTy(C);
  auto *ptrGlobalTy = PointerType::get(C, 1);

  // Build function signature dynamically from kernel metadata.
  //
  // The IR-level argument list must reproduce the source binary's
  // kernarg byte layout exactly: every byte the source reads from the
  // kernarg buffer at offset O must be reachable through some IR
  // argument anchored at that offset. The AMDGPU backend places kernel
  // arguments in the kernarg buffer by their natural alignment + size,
  // so as long as we emit the right type at the right cumulative
  // offset, the buffer layout matches the runtime's packing.
  //
  // Slot shapes emitted:
  //   * `global_buffer` (size==8) → ptr addrspace(1).
  //   * non-pointer `by_value` size==1/2 → i8/i16.
  //   * non-pointer `by_value` size==4 → i32.
  //   * non-pointer `by_value` size==8 → i64.
  //   * non-pointer `by_value` size > 8 (and divisible by 4, i.e. an
  //     aggregate kernarg like Triton's tensor-descriptor struct) is
  //     DECOMPOSED into one i32 slot per dword. Without this split the
  //     IR would carry a single i32 placeholder for the whole struct
  //     and codegen would only allocate 4 bytes for it — silently
  //     shifting every downstream arg's runtime byte offset and turning
  //     all kernarg loads past the struct into reads of garbage. The
  //     per-dword split also makes SMEM kernarg loads against the
  //     interior of the struct addressable through `extractKernargDword`
  //     in handle_smem.cpp without needing any aggregate-aware extract
  //     logic. Other odd sizes are refused loudly: they would require
  //     aggregate extraction with a non-dword tail that no current
  //     handler supports, and the no-fallback rule applies.
  //
  // Test back-reference: lit_tests/s_load_b96_kernarg/ pins the i32
  // slot signature this branch produces for a 16-byte by_value
  // aggregate; any change to the dword-decomposition logic must keep
  // that fixture's `(i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3, ptr
  // addrspace(1) %arg4)` signature green.
  SmallVector<Type *, 8> paramTypes;
  KernargLayout kernargs;
  int paramIdx = 0;
  for (auto &arg : meta.args) {
    if (arg.valueKind == "hidden_global_offset_x" ||
        arg.valueKind == "hidden_global_offset_y" ||
        arg.valueKind == "hidden_global_offset_z" ||
        arg.valueKind.rfind("hidden_", 0) == 0)
      continue;
    bool isPtr = (arg.valueKind == "global_buffer");
    if (isPtr) {
      if (arg.size != 8)
        report_fatal_error(
            Twine("transpiler: kernel '") + kernelName + "' arg '" +
            arg.name + "' is global_buffer but size=" +
            Twine(arg.size) + " (expected 8)");
      paramTypes.push_back(ptrGlobalTy);
      kernargs.params.push_back({arg.offset, 8, paramIdx, true});
      paramIdx++;
      continue;
    }
    if (arg.size == 1) {
      paramTypes.push_back(Type::getInt8Ty(C));
      kernargs.params.push_back({arg.offset, 1, paramIdx, false});
      paramIdx++;
      continue;
    }
    if (arg.size == 2) {
      paramTypes.push_back(Type::getInt16Ty(C));
      kernargs.params.push_back({arg.offset, 2, paramIdx, false});
      paramIdx++;
      continue;
    }
    if (arg.size == 4) {
      paramTypes.push_back(i32Ty);
      kernargs.params.push_back({arg.offset, 4, paramIdx, false});
      paramIdx++;
      continue;
    }
    if (arg.size == 8) {
      paramTypes.push_back(i64Ty);
      kernargs.params.push_back({arg.offset, 8, paramIdx, false});
      paramIdx++;
      continue;
    }
    if (arg.size > 0 && arg.size % 4 == 0) {
      int nDwords = arg.size / 4;
      for (int d = 0; d < nDwords; ++d) {
        paramTypes.push_back(i32Ty);
        kernargs.params.push_back(
            {arg.offset + d * 4, 4, paramIdx, false});
        paramIdx++;
      }
      continue;
    }
    report_fatal_error(
        Twine("transpiler: kernel '") + kernelName + "' arg '" +
        arg.name + "' has unsupported by_value size=" + Twine(arg.size) +
        " (expected 1, 2, 4, 8, or a positive multiple of 4); non-dword-tail "
        "aggregate kernarg extraction is not modelled and silent rounding is "
        "rejected by the no-fallback rule.");
  }
  kernargs.implicitArgsBase = meta.implicitArgsBase();
  kernargs.kernargSegmentSize = meta.kernargSegmentSize;

  auto *funcTy = FunctionType::get(voidTy, paramTypes, false);
  Function *F =
      Function::Create(funcTy, GlobalValue::ExternalLinkage, kernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);
  {
    // Pin the workgroup size to exactly what the source kernel declared, so
    // the backend lays out LDS / workitem IDs the same way the original
    // gfx1250 binary did.
    int maxWg = meta.maxFlatWorkgroupSize > 0 ? meta.maxFlatWorkgroupSize : 1024;
    F->addFnAttr("amdgpu-flat-work-group-size",
                  std::to_string(maxWg) + "," + std::to_string(maxWg));

    // Deliberately do NOT set "amdgpu-waves-per-eu".  Pinning occupancy
    // constrains register allocation and caused spurious VGPR spills for
    // wide kernels (e.g. the Triton 128x128 matmul on gfx942), which then
    // triggered memory faults because our raised IR is register-pressure
    // heavy compared to a from-source compile.  Letting the backend choose
    // occupancy freely keeps register pressure safe.
    // TODO(gfx1250→gfx942): revisit once the raiser emits tighter IR; we may
    // want to propagate the source kernel's waves-per-eu for parity.
  }

  // Propagate static LDS allocation from the source kernel descriptor.
  //
  // The raiser's `ds_write_b128` / `ds_load_b128` / `ds_bpermute` emit
  // pointer-arithmetic into `addrspace(3)` DIRECTLY (via `inttoptr i64
  // to ptr addrspace(3)`), without declaring an LDS `GlobalVariable`.
  // LLVM's AMDGPU backend derives `group_segment_fixed_size` from
  // addrspace(3) GlobalVariables plus the `amdgpu-lds-size` function
  // attribute (see `AMDGPUMachineFunctionInfo` — `LDSSizeRange.first`
  // is read from the attr), so a raised kernel that only manipulates
  // addrspace(3) via int-to-ptr conversion and never sets the attr
  // gets `group_segment_fixed_size: 0` in the emitted HSACO.  The
  // hardware then treats every LDS op as out-of-segment and returns
  // zero / drops writes.  This silently miscompiled every lifted
  // kernel with a non-trivial LDS round-trip, most visibly Triton's
  // `matmul_fp16` (mode-5 B-only-varying input returned all zeros
  // because the cross-thread LDS fragment shuffle read from an
  // uninitialised segment; see matrix-translation.md §12.4 for the
  // bisection).
  //
  // We mirror the source's `.group_segment_fixed_size` by setting the
  // per-function `amdgpu-lds-size` attribute in the source-declared
  // range.  The attribute takes "min,max" — we pass the same value
  // for both since the source's static size is known exactly.
  if (meta.groupSegmentFixedSize > 0) {
    std::string sizeStr = std::to_string(meta.groupSegmentFixedSize);
    F->addFnAttr("amdgpu-lds-size", sizeStr + "," + sizeStr);
  }

  for (int i = 0; i < paramIdx; i++)
    F->getArg(i)->setName("arg" + std::to_string(i));

  errs() << "transpiler: Kernel '" << kernelName << "' has " << paramIdx
         << " args (kernarg_segment_size=" << meta.kernargSegmentSize << ")\n";

  Function *fnWorkgroupIdX =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_x);
  Function *fnWorkitemIdX =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workitem_id_x);
  Function *fnWorkgroupIdY =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_y);
  // Build the source-ISA user-SGPR ABI from the kernel descriptor.
  // Phase 4 seeding and handler-side ABI-sensitive decoding (e.g.
  // handle_smem's kernarg-pointer detection) both key off this layout.
  UserSgprLayout userSgprLayout;
  std::string userSgprFailureDetail;
  if (!UserSgprLayout::tryFromKernelMeta(meta, isa, sourceISA, userSgprLayout,
                                         userSgprFailureDetail)) {
    result.failure = meta.hasKernelDescriptor
                         ? RaiseFailure::userSgprLayoutMismatch(
                               kernelName, userSgprFailureDetail)
                         : RaiseFailure::missingKernelDescriptor(kernelName);
    if (!userSgprFailureDetail.empty())
      errs() << userSgprFailureDetail << "\n";
    return result;
  }
  // ==== Phase 3: Create basic blocks ====
  std::map<uint64_t, BasicBlock *> offsetToBB;
  for (uint64_t addr : blockStarts)
    offsetToBB[addr] = BasicBlock::Create(C, "bb_0x" + utohexstr(addr - kernelOffset), F);

  // ==== Phase 4: Init entry registers ====
  IRBuilder<> B(offsetToBB[kernelOffset]);

  AllocaRegFile regs;
  regs.init(B, i32Ty, i1Ty, isa, *mc.regInfo, projection);

  // Seed kernel-entry SGPR state from the descriptor-derived user-SGPR ABI.
  //
  // Crucial invariant: never hardcode SGPR indices. Kernarg preload and
  // enable_sgpr_* toggles legally move the kernarg pointer and workgroup-id
  // SGPRs away from s[0:1]/s2/s3. Hardcoding those indices mis-seeds entry
  // state and turns real source values into undef reads on the JIT path.
  if (userSgprLayout.kernargSegmentPtrSgpr >= 0) {
    regs.storeSGPR64(
        B, userSgprLayout.kernargSegmentPtrSgpr,
        Constant::getNullValue(PointerType::get(C, 4)));
  }
  if (userSgprLayout.workgroupIdXSgpr >= 0) {
    regs.storeSGPR32(B, userSgprLayout.workgroupIdXSgpr,
                     B.CreateCall(fnWorkgroupIdX, {}, "wg_id_x"));
  }
  if (userSgprLayout.workgroupIdYSgpr >= 0) {
    regs.storeSGPR32(B, userSgprLayout.workgroupIdYSgpr,
                     B.CreateCall(fnWorkgroupIdY, {}, "wg_id_y"));
  }
  auto loadDispatchU16 = [&](Value *dispatchPtr, unsigned byteOffset,
                             const Twine &name) -> Value * {
    Value *p = B.CreateConstInBoundsGEP1_32(i8Ty, dispatchPtr, byteOffset);
    return B.CreateZExt(B.CreateLoad(Type::getInt16Ty(C), p, name), i32Ty,
                        name + "_zext");
  };
  auto loadDispatchU32 = [&](Value *dispatchPtr, unsigned byteOffset,
                             const Twine &name) -> Value * {
    Value *p = B.CreateConstInBoundsGEP1_32(i8Ty, dispatchPtr, byteOffset);
    return B.CreateLoad(i32Ty, p, name);
  };
  auto emitHiddenBlockCount = [&](unsigned dim) -> Value * {
    Function *dispatchPtrFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::amdgcn_dispatch_ptr);
    Value *dispatchPtr = B.CreateCall(dispatchPtrFn, {}, "dispatch_ptr");
    // HSA kernel dispatch packet layout: workgroup_size_{x,y,z} are u16 at
    // bytes 4/6/8 and grid_size_{x,y,z} are u32 at bytes 12/16/20. Triton's
    // hidden_block_count_* ABI wants gridDim, i.e. grid_size / workgroup_size.
    unsigned wgOffset = HsaKernelDispatchPacket::workgroupSizeOffset(dim);
    unsigned gridOffset = HsaKernelDispatchPacket::gridSizeOffset(dim);
    Value *wgSize = loadDispatchU16(dispatchPtr, wgOffset,
                                    Twine("dispatch_wg_size_") + Twine(dim));
    Value *gridSize = loadDispatchU32(dispatchPtr, gridOffset,
                                      Twine("dispatch_grid_size_") + Twine(dim));
    return B.CreateUDiv(gridSize, wgSize,
                        Twine("hidden_block_count_") + Twine(dim));
  };
  auto emitPreloadedHiddenKernargDword = [&](int byteOffset) -> Value * {
    switch (classifyPreloadedHiddenKernargDword(meta.args, byteOffset)) {
    case PreloadedHiddenKernargDword::NotHidden:
      return nullptr;
    case PreloadedHiddenKernargDword::HiddenBlockCountX:
      return emitHiddenBlockCount(/*dim=*/0);
    case PreloadedHiddenKernargDword::HiddenBlockCountY:
      return emitHiddenBlockCount(/*dim=*/1);
    case PreloadedHiddenKernargDword::HiddenBlockCountZ:
      return emitHiddenBlockCount(/*dim=*/2);
    case PreloadedHiddenKernargDword::UnsupportedHidden:
      report_fatal_error(Twine("transpiler: preloaded hidden kernarg at byte "
                               "offset ") +
                         Twine(byteOffset) +
                       " has no modeled entry-SGPR seed. Refusing instead "
                       "of treating a runtime-provided hidden value as "
                       "padding/undef.");
    }
    return nullptr;
  };
  // Kernarg preload SGPRs carry dwords copied by hardware from the kernarg
  // segment before kernel entry. Materialize the same dwords from the IR
  // function args so early source-SGPR reads observe the descriptor-declared
  // preload image.
  for (size_t sgprIdx = 0; sgprIdx < userSgprLayout.entries.size(); ++sgprIdx) {
    const auto &entry = userSgprLayout.entries[sgprIdx];
    if (entry.source != UserSgprLayout::Source::PreloadedKernarg)
      continue;
    std::string why;
    Value *dw = emitPreloadedHiddenKernargDword(entry.kernargByteOffset);
    if (!dw)
      dw = extractKernargDword(kernargs, B, F, entry.kernargByteOffset, &why);
    if (!dw) {
      report_fatal_error(Twine("transpiler: failed to seed preloaded kernarg SGPR s") +
                         Twine(static_cast<int>(sgprIdx)) + " at byte offset " +
                         Twine(entry.kernargByteOffset) + ": " + why);
    }
    regs.storeSGPR32(B, static_cast<int>(sgprIdx), dw);
  }
  // v0 = workitem_id_x
  regs.storeVGPR32(B, 0, B.CreateCall(fnWorkitemIdX, {}, "tid"));
  // Init VCC/SCC to false
  regs.storeVCC(B, ConstantInt::getFalse(i1Ty));
  regs.storeSCC(B, ConstantInt::getFalse(i1Ty));

  // On gfx12+ the hardware command processor uses TTMP registers for
  // workgroup scheduling (RDNA4+ / CDNA-next layout):
  //   ttmp7[15:0]  = workgroup_id_y  (low 16 bits)
  //   ttmp7[31:16] = workgroup_id_z  (high 16 bits; 0 when grid has no Z)
  //   ttmp8[29:25] = wave_id within workgroup (subgroup ID)
  //   ttmp9        = workgroup_id_x  (accelerated launch)
  // The packed-Y-and-Z layout in ttmp7 is from the AMDGPU backend's
  // `loadInputValue` path (see LLVM's `AMDGPULegalizerInfo.cpp` —
  // `WorkGroupIDY = ArgDescriptor::createRegister(TTMP7, 0xFFFFu)`,
  // `WorkGroupIDZ = ArgDescriptor::createRegister(TTMP7, 0xFFFF0000u)`).
  // Triton-generated gfx1250 kernels read the Y component via
  // `s_and_b32 sN, ttmp7, 0xffff` (e.g. matmul_fp16_16x16's `pid_n =
  // tl.program_id(1)` lowering), so a kernel raised without ttmp7
  // initialised always sees `workgroup_id_y == 0` — only the
  // leftmost column of workgroups in a 2D-grid kernel writes its
  // tile, and the right-side tiles stay at whatever the destination
  // memory held at dispatch (verified empirically: matmul_fp16_16x16
  // M=32 with an all-1s input shows cols 0..15 = correct 32.0,
  // cols 16..31 = poison-fill from the host's pre-launch memset).
  // gfx11 (RDNA3) passes these via SGPRs set up by the CP instead.
  if (AMDGPU::isGFX12Plus(*mc.subtargetInfo)) {
    B.CreateStore(B.CreateCall(fnWorkgroupIdX, {}, "ttmp9_wg_id"), regs.ttmp[9]);

    // ttmp7 = (workgroup_id_z << 16) | (workgroup_id_y & 0xFFFF).
    // We mask Y to 16 bits before shifting Z so a stray-high-bit Y
    // doesn't bleed into the Z field.  CAVEAT: upstream's mask is
    // conditional — `AMDGPULegalizerInfo::loadInputValue` uses `~0u`
    // on no-Z-grid entry-function kernels (letting a consumer that
    // reads ttmp7 unmasked see the FULL 32-bit workgroup_id_y, for
    // Y up to UINT_MAX).  Our unconditional 16-bit mask clips Y on
    // no-Z grids with Y >= 65536, which is a hypothetical silent
    // miscompile.  We have not observed a lifted kernel that does
    // this in practice — every Triton-emitted consumer I surveyed
    // reads via `s_and ttmp7, 0xffff` — but if a Y >= 65536 no-Z
    // kernel shows up we'll need to either thread `hasWorkGroupIDZ`
    // through `meta` and emit the conditional mask here, or switch
    // to the `~0u` mask and let `s_and ttmp7, 0xffff` consumers
    // tolerate the Z bits bleeding into their read (they already do
    // per the consumer pattern definition).
    Value *wgIdY = B.CreateCall(fnWorkgroupIdY, {}, "ttmp7_wg_id_y");
    Function *fnWorkgroupIdZ =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_z);
    Value *wgIdZ = B.CreateCall(fnWorkgroupIdZ, {}, "ttmp7_wg_id_z");
    Value *wgIdYLo = B.CreateAnd(wgIdY, B.getInt32(0xFFFF), "wg_id_y_lo16");
    Value *wgIdZHi = B.CreateShl(wgIdZ, B.getInt32(16), "wg_id_z_hi16");
    Value *ttmp7Val = B.CreateOr(wgIdYLo, wgIdZHi, "ttmp7_val");
    B.CreateStore(ttmp7Val, regs.ttmp[7]);

    // wave_id = workitem_id_x / wavefront_size (32 for gfx12)
    Value *tidForTtmp = B.CreateCall(fnWorkitemIdX, {}, "ttmp8_tid");
    Value *waveId = B.CreateLShr(tidForTtmp, B.getInt32(5), "wave_id_in_wg");
    Value *ttmp8Val = B.CreateShl(waveId, B.getInt32(25), "ttmp8_val");
    B.CreateStore(ttmp8Val, regs.ttmp[8]);
  }

  // ==== Phase 5: Raise each instruction ====

  auto *f16Ty = Type::getHalfTy(C);
  // `userSgprLayout` was built above before Phase 4 so entry SGPR seeding
  // and handler-side ABI decisions use the same descriptor-derived mapping.
  RaiseContext ctx{C, M, B, regs, projection, mc, isa, targetIsa, kernargs,
                   &userSgprLayout, F,
                   i1Ty, i8Ty, i32Ty, i64Ty, f32Ty, f16Ty,
                   ptrGlobalTy, offsetToBB};
  ctx.setpcAnalysis = &setpcAnalysis;
  ctx.sourcePrivateSegmentFixedSize =
      static_cast<uint32_t>(std::max(meta.privateSegmentFixedSize, 0));
  ctx.sourceComputePgmRsrc2 = meta.computePgmRsrc2;
  ctx.sourceKernelCodeProperties = meta.kernelCodeProperties;
  ctx.initializeSgprKernargProvenance();

  // ==== Phase 4.5: Kernarg-pair pristine-at-BB-entry dataflow ====
  //
  // Populates `ctx.kernargPristineBBs` with the set of BB-start
  // offsets where the kernarg-pointer SGPR pair is guaranteed to still
  // hold its kernel-entry value along EVERY CFG path that reaches
  // the BB. `handle_smem.cpp` consults this (via the tracker
  // `RaiseContext::KernargPtrDelta`) to decide whether to take the
  // kernarg-slot fast path or refuse loudly.
  //
  // This closes issue #21's silent miscompile in the Tensile
  // UniversalArgs shape (`s_add_u32 ka, ka, 0x10 ; s_addc_u32 ka+1,
  // ka+1, 0 ; s_load_b* s[sDST:…], s[ka:ka+1], imm`): the NORMAL-path
  // BB that contains the shift still enters pristine (its sole
  // predecessor is the entry BB, which never writes the pair), so
  // the post-handler hook below can stage the delta and the shifted
  // downstream loads correctly extract via `extractKernargDword(delta
  // + imm)` instead of `extractKernargDword(imm)`.
  //
  // The analysis is a forward dataflow to fixed point:
  //
  //   pristine[entryBB]       = true
  //   pristine[bb] (bb != e)  = AND over all preds p of bb:
  //                                 pristine[p] AND !writesPairInBB[p]
  //
  // Converges monotonically (a BB can only transition pristine ->
  // dirty, never back) in at most `#BBs` iterations. The CFG edges
  // are read straight from the decoded instruction stream:
  // unconditional / conditional branches add one or two target edges;
  // fall-through adds an edge to the next BB in address order when
  // the terminator is not an unconditional jump or `s_endpgm`.
  //
  // Skipped entirely when `kernargSegmentPtrSgpr < 0` (the kernel
  // descriptor disables the kernarg pointer — no pair to track); in
  // that case every BB is trivially "pristine" because the tracker
  // never fires in handle_smem.cpp. We still populate the set to
  // keep the resetKernargPtrDeltaAtBBBoundary code path uniform.
  {
    const int kaLo = userSgprLayout.kernargSegmentPtrSgpr;
    const int kaHi = (kaLo >= 0) ? (kaLo + 1) : -1;

    // Step 1: group instructions by their owning BB (largest
    // blockStart <= di.offset) and record whether any instruction in
    // each BB writes either kernarg-pair dword.
    //
    // `blockStarts` is an ascending std::set<uint64_t>, so
    // lower_bound / --upper_bound gives the BB a given instruction
    // belongs to in O(log N) per step.
    llvm::DenseMap<uint64_t, bool> writesPairInBB;
    llvm::DenseMap<uint64_t, uint64_t> bbOfInst; // inst offset -> BB start
    llvm::DenseMap<uint64_t, const DecodedInst *> lastInstInBB;
    auto isKernargHighCanonicalMask = [&](const DecodedInst &di) -> bool {
      if (di.semOp != SemOp::S_AND_B32 || di.numSrcs < 2 || di.numDefs < 1)
        return false;
      if (!di.isReg(0))
        return false;
      ParsedReg dst = ctx.parseReg(di.getReg(0), 0);
      if (dst.kind != ParsedReg::SGPR || dst.baseIdx != kaHi)
        return false;

      unsigned s0Idx = di.srcMap[0];
      unsigned s1Idx = di.srcMap[1];
      auto srcIsKernargHi = [&](unsigned idx) {
        if (!di.isReg(idx))
          return false;
        ParsedReg src = ctx.parseReg(di.getReg(idx), idx);
        return src.kind == ParsedReg::SGPR && src.baseIdx == kaHi;
      };
      auto srcIsLow16Mask = [&](unsigned idx) {
        return di.isImm(idx) &&
               static_cast<uint64_t>(di.getImm(idx)) == 0xffffu;
      };
      return (srcIsKernargHi(s0Idx) && srcIsLow16Mask(s1Idx)) ||
             (srcIsKernargHi(s1Idx) && srcIsLow16Mask(s0Idx));
    };
    for (uint64_t bs : blockStarts) {
      writesPairInBB.try_emplace(bs, false);
      lastInstInBB.try_emplace(bs, nullptr);
    }
    for (const DecodedInst &di : insts) {
      // Find the BB that owns this instruction: the largest entry in
      // blockStarts <= di.offset.
      auto it = blockStarts.upper_bound(di.offset);
      if (it == blockStarts.begin())
        continue; // before first BB — defensive, shouldn't happen.
      --it;
      uint64_t bb = *it;
      bbOfInst[di.offset] = bb;
      lastInstInBB[bb] = &di;
      if (kaLo < 0)
        continue;
      // Detect writes to either dword of the pair via the MCInst's
      // def operands (first `numDefs` operand slots). We only check
      // register operands; parseReg may give a non-SGPR kind for
      // TTMP-class defs etc., but the match condition below rejects
      // those.
      for (unsigned d = 0; d < di.numDefs; ++d) {
        if (!di.isReg(d))
          continue;
        ParsedReg pr = ctx.parseReg(di.getReg(d), d);
        if (pr.kind == ParsedReg::SGPR &&
            (pr.baseIdx == kaLo || pr.baseIdx == kaHi)) {
          if (pr.baseIdx == kaHi && isKernargHighCanonicalMask(di))
            continue;
          writesPairInBB[bb] = true;
          break;
        }
      }
    }

    // Step 2: build CFG edges. For each BB, determine its successors
    // (list of BB offsets) based on the last instruction:
    //   * unconditional branch: single target from immediate.
    //   * conditional branch: target from immediate + fall-through.
    //   * s_endpgm / unconditional terminator without a target: no
    //     successors.
    //   * anything else (including the trivial "reached fall-through
    //     without a branch"): fall-through to the next BB in address
    //     order.
    //
    // We conservatively add s_set_pc_i64 targets too, via the
    // setpcAnalysis machinery: `setpcAnalysis.chainTerminators` and
    // the extra block starts merged into `blockStarts` above already
    // capture those, but we must still add the *edges* here. For
    // this bug's scope, s_set_pc_i64 targets only matter if they
    // reach a BB that consumes the kernarg pair — none of the
    // observed Tensile / AITER shapes do — so we model s_set_pc_i64
    // conservatively as "target is reachable if ANY reachable
    // BB's terminator is an s_set_pc_i64 whose resolved return
    // address equals the target". Simpler: treat an unresolved
    // s_set_pc_i64 as a successor edge to every extraBlockStart
    // (over-approximation: marks potentially-pristine BBs as
    // dirty only when they legitimately are). For the Tensile
    // shape the s_set_pc_i64 machinery is not in play.
    llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t>> successors;
    for (uint64_t bs : blockStarts)
      successors.try_emplace(bs, llvm::SmallVector<uint64_t>{});

    auto nextBBStart = [&](uint64_t bs) -> uint64_t {
      auto it = blockStarts.upper_bound(bs);
      return (it == blockStarts.end()) ? 0 : *it;
    };
    auto branchTargetOf = [](const DecodedInst &di) -> llvm::SmallVector<uint64_t> {
      // Mirror `decode.cpp::collectBranchTargets`: a branch instruction's
      // immediate operands each encode a signed 16-bit PC-relative
      // offset (scaled by 4) from the instruction's successor address
      // (off + 4, per the AMDGPU encoding definition). Collect every
      // such target.
      llvm::SmallVector<uint64_t> out;
      const llvm::MCInst &inst = di.inst;
      for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
        if (!inst.getOperand(i).isImm())
          continue;
        int64_t raw = inst.getOperand(i).getImm();
        int64_t brOff = static_cast<int64_t>(
            static_cast<int16_t>(static_cast<uint16_t>(raw & 0xFFFF)));
        out.push_back(di.offset + 4 + brOff * 4);
      }
      return out;
    };

    for (uint64_t bs : blockStarts) {
      const DecodedInst *last = lastInstInBB[bs];
      if (!last) {
        // BB has no instructions (shouldn't happen but be defensive).
        uint64_t nxt = nextBBStart(bs);
        if (nxt != 0)
          successors[bs].push_back(nxt);
        continue;
      }
      // S_ENDPGM / unconditional terminator with no target: no
      // successors. `desc.isBarrier()` would be more precise, but
      // we approximate via the SemOp set that matches the observed
      // AMDGPU terminators. S_ENDPGM is always a barrier; unmatched
      // cases fall through below.
      if (last->semOp == SemOp::S_ENDPGM)
        continue;
      if (last->isBranch) {
        auto targets = branchTargetOf(*last);
        for (uint64_t t : targets)
          successors[bs].push_back(t);
        // Conditional branches also fall through to the next BB.
        if (last->isConditionalBranch) {
          uint64_t nxt = nextBBStart(bs);
          if (nxt != 0)
            successors[bs].push_back(nxt);
        }
        continue;
      }
      // Non-branch terminator (or no terminator at all in this BB
      // because the BB split landed mid-stream): fall through to the
      // next BB in address order.
      uint64_t nxt = nextBBStart(bs);
      if (nxt != 0)
        successors[bs].push_back(nxt);
    }

    // Invert to predecessors for the fixed-point pass.
    llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t>> predecessors;
    for (uint64_t bs : blockStarts)
      predecessors.try_emplace(bs, llvm::SmallVector<uint64_t>{});
    for (auto &[bb, succs] : successors) {
      for (uint64_t s : succs) {
        auto it = predecessors.find(s);
        if (it != predecessors.end())
          it->second.push_back(bb);
      }
    }

    // Provenance carry is safe only for the lexical fallthrough block whose
    // sole predecessor is the block we just finished emitting. Any branch
    // target that is not the immediate next block would require saving and
    // restoring predecessor-exit state; any merge would require phi-style
    // joins. Both remain loud/conservative via the normal BB-boundary reset.
    uint64_t prevBB = 0;
    bool havePrevBB = false;
    for (uint64_t bs : blockStarts) {
      if (havePrevBB) {
        const auto &preds = predecessors[bs];
        if (preds.size() == 1 && preds[0] == prevBB)
          ctx.sgprProvenanceFallthroughBBs.insert(bs);
      }
      prevBB = bs;
      havePrevBB = true;
    }

    // Step 3: iterative forward dataflow.
    //   pristine[entryBB]  = true
    //   pristine[bb]       = AND over preds p: pristine[p] AND
    //                         !writesPairInBB[p]
    //
    // This is a greatest-fixed-point problem: a write-free loop header with
    // a pristine preheader and a self-edge is pristine, because every trip
    // around the loop preserves the pair. Initialising non-entry blocks to
    // false permanently poisons such loops. Start optimistic, then
    // monotonically remove blocks when any predecessor is dirty or writes the
    // pair. Unreachable non-entry blocks (no preds) are removed.
    //
    // Skipped when `kaLo < 0` (no kernarg pointer enabled): mark
    // every BB pristine unconditionally to keep the call sites
    // uniform.
    llvm::DenseSet<uint64_t> &pristine = ctx.kernargPristineBBs;
    if (kaLo < 0) {
      for (uint64_t bs : blockStarts)
        pristine.insert(bs);
    } else {
      for (uint64_t bs : blockStarts)
        pristine.insert(bs);
      bool changed = true;
      unsigned iters = 0;
      const unsigned maxIters = static_cast<unsigned>(blockStarts.size()) + 2;
      while (changed && iters < maxIters) {
        changed = false;
        ++iters;
        for (uint64_t bs : blockStarts) {
          if (bs == kernelOffset)
            continue;
          const auto &preds = predecessors[bs];
          bool keepPristine = !preds.empty();
          if (keepPristine) {
            for (uint64_t p : preds) {
              if (!pristine.contains(p) || writesPairInBB.lookup(p)) {
                keepPristine = false;
                break;
              }
            }
          }
          if (!keepPristine && pristine.erase(bs))
            changed = true;
        }
      }
      if (iters >= maxIters && changed) {
        // Did not converge within the expected bound — fail loudly
        // per the project's no-silent-fallback rule. Practically
        // unreachable (forward-dataflow over a monotone lattice
        // converges in at most `#BBs` iters), but surface a
        // diagnostic rather than silently accept a possibly-wrong
        // pristine set.
        report_fatal_error(
            "transpiler: kernarg-pristine dataflow failed to converge "
            "within #BBs + 2 iterations; the CFG edge construction in "
            "raiser.cpp Phase 4.5 is out of sync with the decoder's "
            "branch-target collection.");
      }
    }
  }

  // Dominance-safe SGPR wave-mask shadow storage.
  // One EXEC-width mask + one scalar-valid bit per SGPR base index.
  // Consumers can combine `(valid ? shadow : fallback)` across BBs without
  // carrying non-dominating SSA values in `lastSgprWaveMaskI1`.
  ctx.sgprWaveMaskExecShadow.reserve(regs.sgpr.size());
  ctx.sgprWaveMaskValidShadow.reserve(regs.sgpr.size());
  for (unsigned i = 0; i < regs.sgpr.size(); ++i) {
    auto *maskA = B.CreateAlloca(regs.execTy, nullptr,
                                 "sgpr_mask_shadow_" + std::to_string(i));
    auto *validA = B.CreateAlloca(i1Ty, nullptr,
                                  "sgpr_mask_valid_" + std::to_string(i));
    B.CreateStore(ConstantInt::get(regs.execTy, 0), maskA);
    B.CreateStore(B.getFalse(), validA);
    ctx.sgprWaveMaskExecShadow.push_back(maskA);
    ctx.sgprWaveMaskValidShadow.push_back(validA);
  }

  // Wire the reg-file's EXEC-write invalidation hook to ctx's lane_active
  // memo. This catches every EXEC mutation — ctx.storeExec, the various
  // ctx.writeReg*(EXEC, …) wrappers, *and* the handful of handlers that
  // still call ctx.regs.storeExec / ctx.regs.writeRegExecWidth directly
  // (SAVEEXEC family in handle_sop1, V_CMPX in handle_valu). Without
  // this hook those direct paths would leave the memo pointing at a
  // pre-write `lane_active`, silently mispredicating subsequent
  // emitUnderExec diamonds.
  regs.onExecWritten = [&ctx] { ctx.resetLaneActiveCache(); };

  // Wire the reg-file's per-SGPR write invalidation hook to ctx's
  // V_CMP -> V_CNDMASK per-lane-i1 shadow map
  // (`lastSgprWaveMaskI1`). Fires on every `storeSGPR32 / storeSGPR64`
  // and therefore on every path that mutates an SGPR — including
  // handlers that bypass `writeReg32 / writeReg64` to call the
  // low-level stores directly (handle_smem's multi-dword load
  // splitting, handle_valu's SCC-flag SGPR writes, etc.). The V_CMP
  // wave-mask write path also fires this hook; the V_CMP handler
  // immediately re-populates the shadow with the per-lane `i1`
  // afterwards via `ctx.recordSgprWaveMaskI1`. See hotswap/docs/sgpr-
  // wave-mask-translation.md section 3.1 for the full contract.
  regs.onSgprWritten = [&ctx](int idx) { ctx.invalidateSgprWaveMaskI1(idx); };

  int raisedCount = 0;

  for (size_t instIdx = 0; instIdx < insts.size(); ++instIdx) {
    const DecodedInst &di = insts[instIdx];

    // Source-BB boundary handling uses `B.GetInsertBlock()` rather than a
    // tracked `currentBB` so that intra-handler CFG splits (emitUnderExec
    // diamonds under SPE) propagate correctly: fall-through must leave
    // from whatever block the builder is currently at — which is the
    // `spe_skip` tail when the last emission was wrapped — not from the
    // block that started the source instruction.
    auto bbIt = offsetToBB.find(di.offset);
    if (bbIt != offsetToBB.end() && bbIt->second != B.GetInsertBlock()) {
      BasicBlock *insertBB = B.GetInsertBlock();
      if (!insertBB->hasTerminator())
        B.CreateBr(bbIt->second);
      B.SetInsertPoint(bbIt->second);
      // LLVM's AMDGPULowerVGPREncoding pass resets VGPR MSB mode at every
      // basic-block boundary (both before terminators and at BB fall-through
      // exits).  Mirror that behaviour so we do not inherit stale MSB state
      // from a previous linear instruction that does not control-flow into
      // this BB.
      ctx.vgprMSBs = 0;
      // Drop the V_CMP -> V_CNDMASK per-lane-i1 shadow at every BB
      // transition. The cached `i1` SSA values dominate only the BB
      // they were emitted in; carrying them into a successor would
      // read an SSA value out of its dominance scope. A future
      // reaching-definitions pass on the raised IR could upgrade this
      // to a proper per-BB merge (see sgpr-wave-mask-translation.md
      // section 7 evolution path).
      ctx.clearSgprWaveMaskShadow();
      // Reset the kernarg-pair const-delta tracker for the new BB,
      // consulting the precomputed `kernargPristineBBs` dataflow
      // result (Phase 4.5 below) to decide whether the pair is
      // guaranteed-pristine on every incoming CFG edge. When it is,
      // the tracker enters `valid = true, delta = 0`; otherwise
      // `valid = false` and handle_smem.cpp refuses the kernarg-slot
      // fast path loudly for any kernarg-pair SMEM load in this BB.
      // See `RaiseContext::KernargPtrDelta` and
      // `resetKernargPtrDeltaAtBBBoundary` for the full contract.
      ctx.resetKernargPtrDeltaAtBBBoundary(di.offset);
    }

    ctx.computeVGPRAdjust(di);
    // Invalidate the SPE lane_active memoisation at every instruction
    // boundary. Any instruction is a potential EXEC writer (either through
    // our modeled SemOp allow-list, or through a path we haven't yet
    // covered), and emitLaneActiveBit is load-bearing for per-lane
    // predication correctness: reusing a stale lane_active from before an
    // EXEC write would silently mispredicate side effects. See
    // RaiseContext::resetLaneActiveCache in raise_context.hpp for the full
    // invalidation contract.
    ctx.resetLaneActiveCache();
    OpResolver op{ctx, di};

    // Dispatch to the format-specific handler by querying TSFlags (and
    // `AMDGPU::isVOPD` for the one encoding without a dedicated flag bit)
    // directly, rather than going through a hand-rolled FormatKind enum.
    // Check precedence mirrors LLVM's decoder:
    //   * VOPD first — it has no TSFlags bit; detect by named-operand id.
    //   * IsMAI before VOP3 — MFMA is a VOP3 subclass with its own handler.
    //   * DPP / SDWA / VOPC / VOP3P / VOP3 / VOP2 / VOP1 all route to
    //     handleVALU, so they're collapsed into one mask test; ordering
    //     within the VOP family is therefore irrelevant here.
    //   * Scalar / memory family bits are mutually exclusive.
    // `default: break;` semantics are preserved: anything without a matching
    // bit falls through with `hr.handled == false` and hits the unsupported-
    // instruction error path below.
    const uint64_t kVALU =
        SIInstrFlags::DPP | SIInstrFlags::SDWA | SIInstrFlags::VOP1 |
        SIInstrFlags::VOP2 | SIInstrFlags::VOP3 | SIInstrFlags::VOPC |
        SIInstrFlags::VOP3P;
    const uint64_t flags = di.tsFlags;
    const unsigned opc = di.inst.getOpcode();
    HandlerResult hr;
    if (AMDGPU::isVOPD(opc))
      hr = handleVOPD(ctx, di, op);
    else if (flags & SIInstrFlags::IsMAI)
      hr = handleMFMA(ctx, di, op);
    else if (flags & kVALU)
      hr = handleVALU(ctx, di, op);
    else if (flags & SIInstrFlags::SOPP)
      hr = handleSOPP(ctx, di, op);
    else if (flags & SIInstrFlags::SOPC)
      hr = handleSOPC(ctx, di, op);
    else if (flags & SIInstrFlags::SOP1)
      hr = handleSOP1(ctx, di, op);
    else if (flags & SIInstrFlags::SOP2)
      hr = handleSOP2(ctx, di, op);
    else if (flags & SIInstrFlags::SOPK)
      hr = handleSOPK(ctx, di, op);
    else if (flags & SIInstrFlags::SMRD)
      hr = handleSMEM(ctx, di, op);
    else if (flags & SIInstrFlags::FLAT)
      hr = handleFLAT(ctx, di, op);
    else if (flags & SIInstrFlags::MUBUF)
      hr = handleMUBUF(ctx, di, op);
    else if (flags & SIInstrFlags::DS)
      hr = handleDS(ctx, di, op);
    // VIMAGE TENSOR pseudo-instructions (`tensor_load_to_lds_d{2,4}`,
    // `tensor_store_from_lds_d{2,4}`, MIMGInstructions.td:2049-2113).
    // The pseudo extends `InstSI` directly and only sets `let VALU =
    // 1` and `let TENSOR_CNT = 1` (NOT `let VIMAGE = 1`), so the
    // `SIInstrFlags::VIMAGE` bit stays 0 on these. Dispatch on
    // `TENSOR_CNT` instead — the only other carrier of that bit is
    // `s_wait_tensorcnt` (SOPP), which is already claimed by the
    // SOPP arm above and never reaches this fallthrough. Routed
    // late because TENSOR ops are exclusive to the gfx1250
    // (`isGFX125xOnly`) generation and the handler's only contract
    // today is a cross-target loud refusal; the same gating applies
    // when the same-target intrinsic-emit path lands.
    else if (flags & SIInstrFlags::TENSOR_CNT)
      hr = handleVIMAGE(ctx, di, op);

    // Operand-read paths (`readOp32` / `readOp64`) cannot bail mid-
    // handler, so they record any unsupported-register failures into
    // `ctx.pendingFailure`. Promote that to the structured failure
    // *before* the `hr.handled` check — a handler that "succeeded"
    // by returning undef from a read is still an unraised kernel.
    if (ctx.pendingFailure.hasFailed()) {
      result.failure = std::move(ctx.pendingFailure);
      ctx.pendingFailure = RaiseFailure{};
      return result;
    }

    if (hr.handled) {
      if (di.defsSCC && !hr.sccHandled && hr.sccResult) {
        Value *zero = Constant::getNullValue(hr.sccResult->getType());
        ctx.regs.storeSCC(ctx.B, ctx.B.CreateICmpNE(hr.sccResult, zero));
      }
      if (di.defsEXEC)
        result.hasDivergentExec = true;
      // Pattern B call-site post-processing: if this s_add_co_ci_u32
      // is the high-half terminator of a getpc+add chain that feeds
      // a Pattern B `s_set_pc_i64` enumerated-dispatch cascade (i.e.
      // some downstream s_set_pc_i64 reads the same ret-pair this
      // chain populated), overwrite the ret-pair SGPR with the plain
      // i64 marker `resolvedReturnAddr` — i.e. the source-MC byte
      // offset of the BB this chain meant to return to. The
      // downstream cascade compares against the same offsets via
      // `icmp eq i64 %marker, <offset_k>` for each enumerated
      // target; when this predecessor's marker matches one of the
      // enumerated offsets, mem2reg + SCCP + InstCombine fold the
      // compare to `i1 true` across the phi join and SimplifyCFG
      // collapses the cmp+br cascade into a direct
      // `br label %BB_<offset>`. The SOP2 handler has already done
      // its (binary-PC-producing) arithmetic above; this commit
      // happens *after* and clobbers that result on purpose — that
      // value was an opaque runtime PC we never want to see
      // downstream.
      //
      // An earlier revision of this hook wrote
      // `ptrtoint(blockaddress(@kernel, %BB_returnAddr)) to i64`
      // here so the cascade could compare against a `blockaddress`
      // constant. That form survived mem2reg + SCCP unfolded in
      // irreducible tensilelite-shaped CFGs (the `storeSGPR64`
      // hi/lo split prevented the cross-phi fold), leaving a
      // `BlockAddress` SDNode alive into AMDGPU ISel, which has no
      // codegen pattern for it and aborts llc with
      //   `Cannot select: t1: i64 = BlockAddress<@kernel, %bb_N>`.
      // Using a plain integer marker keeps `BlockAddress` solely
      // as a direct-branch `label` operand (which DOES have a
      // codegen pattern), sidestepping the ISel crash entirely.
      // See setpc_analysis.hpp + semop.hpp's S_SET_PC_I64 doc +
      // `emitEnumeratedDispatch` in handle_sop1.cpp.
      if (di.semOp == SemOp::S_ADDC_U32) {
        auto it = setpcAnalysis.chainTerminators.find(di.offset);
        if (it != setpcAnalysis.chainTerminators.end()) {
          // Force the BB to exist so the downstream cascade's
          // direct branch has a destination; we don't use the
          // pointer here.
          (void)ctx.lookupBB(it->second.resolvedReturnAddr);
          Value *retMarker =
              ConstantInt::get(ctx.i64Ty, it->second.resolvedReturnAddr);
          ctx.regs.storeSGPR64(ctx.B,
                                static_cast<int>(it->second.retPairLowReg),
                                retMarker);
        }
      }

      // Kernarg-pointer const-delta tracker update. Single-BB, fires
      // only for the canonical `s_add_u32 ; s_addc_u32 (hi, hi, #0)`
      // 64-bit const-add pattern against the source-ABI kernarg-pair
      // SGPRs (Tensile UniversalArgs: shift kernarg ptr past a 16-byte
      // preamble before issuing the downstream static kernarg loads).
      // Everything else that touches either dword invalidates the
      // tracker; handle_smem.cpp then refuses the kernarg-slot fast
      // path loudly. See `RaiseContext::KernargPtrDelta` for the full
      // state machine and invariants, and issue #21 for the canonical
      // miscompile this fix surfaces.
      //
      // Placed here (in the generic post-handler hook rather than in
      // handle_sop2.cpp) so the update point is a single location
      // independent of which SemOp fires: adding a new kernarg-pair
      // mutator opcode (e.g. S_SUB_U32, S_MOV_B32) means updating
      // *this* switch, not each handler in turn, and invariants over
      // the whole MC stream (the "any other write clobbers SCC"
      // dependency of the pending-low state) are checkable against
      // `di.defsSCC` directly without plumbing hint flags through the
      // HandlerResult.
      if (ctx.userSgprLayout != nullptr) {
        const int kaLo = ctx.userSgprLayout->kernargSegmentPtrSgpr;
        if (kaLo >= 0)
          KernargSgprProvenanceUpdater(ctx, di, op, kaLo).update();
      }
      raisedCount++;
      continue;
    }

    // The handler either recognised the instruction but refused the
    // specific shape (hr.failure.reason != None), or no handler claimed
    // it at all — promote to `UnsupportedOpcode` and bucket by format.
    if (hr.failure.hasFailed()) {
      result.failure = std::move(hr.failure);
    } else {
      result.failure = RaiseFailure::unsupportedOpcode(
          di, formatName(di.tsFlags, di.inst.getOpcode()));
      errs() << "transpiler: Unsupported instruction: " << di.mnemonic
             << " (raw: " << di.rawMnemonic << ")"
             << " [format=" << result.failure.format << "]"
             << " at offset 0x" << format_hex(di.offset, 1) << "\n";
    }
    return result;
  }

  // Ensure all BBs have terminators
  for (auto &BB : *F) {
    if (!BB.hasTerminator()) {
      B.SetInsertPoint(&BB);
      B.CreateUnreachable();
    }
  }

  result.liftedCount = raisedCount;

  // ==== Phase 6: Promote allocas to SSA ====
  {
    DominatorTree DT(*F);
    AssumptionCache AC(*F);
    SmallVector<AllocaInst *, 512> allocas;
    regs.collectAllocas(allocas);
    ctx.collectSgprWaveMaskShadowAllocas(allocas);
    PromoteMemToReg(allocas, DT, &AC);
  }

  // (Former Phase 6.035 "permlane16-swap-selfpreserve" and Phase
  // 6.04 "permlane16-xor3-partner" rewrites were deleted after
  // the asymmetric `v_permlane16_swap_b32` lift landed — see
  // `handle_valu_cross_lane.cpp::emitPermLaneSwapEmulation` and
  // matrix-translation.md §12.4.7.  Both passes were transitional
  // bridges that compensated for the symmetric lift's
  // over-swap of the asymmetric-semantic's "unchanged" halves;
  // with the lift corrected, their fingerprints either no
  // longer match (xor3-partner) or actively corrupt the new
  // select shape (selfpreserve).)

  // ==== Phase 6.5: Cross-widen writelane/readlane rewrite ====
  //
  // Opt-in symmetric rewrite of `v_writelane_b32` / `v_readlane_b32`
  // sites under cross-widening. Disabled by default; the caller
  // (raise_cli's `--enable-writelane-rewrite`, PipelineConfig's
  // `enableWritelaneRewrite`) must ask for it explicitly. See
  // `rewrite_cross_lane_divergent.{hpp,cpp}` and
  // wave-size-translation.md §5.6.3 for the principled derivation,
  // and hotswap/docs/learnings.md for the asymmetric-rewrite bug
  // that motivated the symmetry-plus-use-chain design.
  //
  // Runs AFTER `PromoteMemToReg` by construction: the rewrite pass's
  // forward use-chain classifier needs post-mem2reg SSA so a
  // scratch-addrspace round-trip (load / store through an alloca) does
  // not obscure the fact that a writelane / readlane result eventually
  // reaches an SGPR-constrained consumer. No behavioural change on
  // same-wave / narrowing directions — the rewrite pass short-
  // circuits internally on `targetWaveSize <= sourceWaveSize`.
  //
  // Refusal path. If any writelane / readlane site's forward use chain
  // reaches an SGPR-forced consumer that the classifier cannot prove
  // safe (`s_buffer_load` rsrc, `s_sendmsg` message, `readfirstlane`,
  // addrspace(4) load, inline asm with `"s"` constraint, or any
  // unaudited intrinsic / instruction), the rewrite pass performs
  // zero rewrites and populates `report.sgprForcedDetail`. The raiser
  // surfaces that detail as a `crossWaveRewriteOracleDisagreement`
  // refusal — principled per the no-silent-miscompile contract:
  // rewriting the ds_bpermute output into an SGPR-forced consumer
  // would re-introduce `v_readfirstlane_b32` at the SGPR boundary and
  // recreate the source-wave collapse the rewrite exists to avoid.
  if (enableWritelaneRewrite) {
    // `tm.get()` threaded through so `rewriteCrossLaneDivergent` can
    // build a `UniformityAnalysis` against the compilation target
    // for the §5.6.3 "UA-backed readfirstlane allow-gate" classifier
    // refinement. See the rewrite's header comment for the contract
    // (nullable — null disables the gate and falls back to the
    // conservative pre-UA refusal behaviour).
    CrossLaneDivergentRewriteReport rewriteReport = rewriteCrossLaneDivergent(
        *F, isa.waveSize, targetIsa.waveSize, tm.get());

    if (rewriteReport.refusedSgprForced()) {
      ThreadLoopDecisionResult tlDecision = decideThreadLoopFallback(
          isa.waveSize, targetIsa.waveSize, /*sgprForcedRefusal=*/true,
          rewriteReport.sgprForcedThreadLoopEligible);
      if (!forceThreadLoopProjection &&
          tlDecision.decision == ThreadLoopDecision::EligibleAndGateOn) {
        errs() << "transpiler: post-raise fallback: retrying kernel '"
               << kernelName
               << "' under ThreadLoopProjection after SGPR-forced cross-lane "
                  "rewrite refusal (analysis-triggered, no user opt-in)\n";
        errs() << "transpiler: thread-loop fallback trigger: "
               << rewriteReport.sgprForcedDetail << "\n";
        return raiseToIRImpl(textBytes, sourceISA, kernelName, meta,
                             kernelOffset, compilationTargetISA,
                             /*enableWritelaneRewrite=*/false,
                             /*enableWaveNative=*/false,
                             /*forceThreadLoopProjection=*/true,
                             /*suppressC5ForThreadLoopRoute=*/true);
      }
      if (!forceThreadLoopProjection &&
          tlDecision.decision == ThreadLoopDecision::EligibleButGateOff) {
        errs() << "transpiler: thread-loop fallback candidate for kernel '"
               << kernelName << "' not activated: " << tlDecision.reason
               << ". Keeping principled loud refusal.\n";
      }
      if (!forceThreadLoopProjection &&
          tlDecision.decision == ThreadLoopDecision::Ineligible) {
        errs() << "transpiler: thread-loop fallback not eligible for kernel '"
               << kernelName << "': " << tlDecision.reason
               << ". Keeping principled loud refusal.\n";
      }
      RaiseFailure f = RaiseFailure::crossWaveRewriteOracleDisagreement(
          kernelName, rewriteReport.sgprForcedDetail);
      errs() << "transpiler: post-raise abort: " << f.format << " on '"
             << f.mnemonic << "' \u2014 " << f.detail << "\n";
      result.failure = std::move(f);
      return result;
    }

    // Unsupported `dpp_ctrl` on an i32 update.dpp site — the rewrite
    // family covers quad_perm / row_shl / row_shr today (all stay
    // within a single 16-lane row).  Any ctrl outside that set is
    // either wave-size-dependent (wave_* shifts / rotations) or
    // hasn't been audited yet (row_mirror / row_half_mirror /
    // row_share / row_xmask — expressible but no corpus demand
    // yet).  Refusing loudly surfaces the demand so the next
    // extension has a concrete test pointer.  See
    // `buildDppLaneMap` in rewrite_cross_lane_divergent.cpp for
    // the per-ctrl widening protocol.
    if (rewriteReport.refusedUnsupportedDpp()) {
      RaiseFailure f = RaiseFailure::crossWaveRewriteOracleDisagreement(
          kernelName, rewriteReport.unsupportedDppDetail);
      errs() << "transpiler: post-raise abort: " << f.format << " on '"
             << f.mnemonic << "' \u2014 " << f.detail << "\n";
      result.failure = std::move(f);
      return result;
    }

    // Second-order invariant: the syntactic Phase 1.4.5 classifier
    // matched `WaveIdLiftScalarized` iff the decoded instruction
    // stream contains at least one `v_writelane_b32` /
    // `v_readlane_b32`. Under the symmetry rule every such intrinsic
    // is rewritten (or the whole function refuses above), so a non-
    // zero classifier count MUST coincide with a non-zero count of
    // writelane + readlane rewrites specifically. Checking that
    // specific sum (not the grand total including `dppRewritten`)
    // matters: a kernel that emits DPP sites alongside missing
    // writelane / readlane would otherwise silently satisfy the
    // invariant via the DPP count, masking the handler-emission
    // regression this gate exists to catch.
    if (classifierWaveIdLiftScalarizedSites > 0 &&
        (rewriteReport.writelaneRewritten +
         rewriteReport.readlaneRewritten) == 0) {
      std::string msg;
      raw_string_ostream os(msg);
      os << "classifier matched WaveIdLiftScalarized on "
         << classifierWaveIdLiftScalarizedSites
         << " site(s) but rewriteCrossLaneDivergent rewrote 0 \u2014 the "
            "raised IR is missing the writelane/readlane intrinsic(s) "
            "that the decoded instruction stream contained. This is a "
            "handler-emission regression, not a classifier/rewrite "
            "disagreement. Refusing rather than risk a silent "
            "miscompile (see wave-size-translation.md \u00a75.6.3).";
      RaiseFailure f = RaiseFailure::crossWaveRewriteOracleDisagreement(
          kernelName, os.str());
      errs() << "transpiler: post-raise abort: " << f.format << " on '"
             << f.mnemonic << "' \u2014 " << f.detail << "\n";
      result.failure = std::move(f);
      return result;
    }
  }

  // ==== Phase 6.6: Cross-widen predicate-chain classifier (C5) ====
  //
  // Post-mem2reg classifier for the Class-5 predicate-chain class
  // documented in hotswap/docs/modrep-predicate-chain.md §5 (narrow-O1).
  // Walks every `@llvm.amdgcn.workitem.id.x()` call in the function and
  // refuses the lift if any call's forward use chain reaches an `icmp`
  // against a compile-time constant K in `(0, W_s - 1]` without being
  // AND-masked by `(W_s - 1)` first — i.e. a lane-position-scoped
  // predicate (`tid < 2^s`, `tid < W_s/2`, quad-level masks) that would
  // evaluate differently on target replica-1 lanes than source wave 0
  // under modulo-replication despite sharing the source EXEC bit.
  //
  // Intentionally narrow: Phase-2 IR inspection (modrep-predicate-chain.md
  // §5 O1) established that the broader "any unmasked tid → icmp →
  // side-effect refuses" rule would also refuse baselines
  // `vecadd_f16` / `rope_fp32` / `canary_dpp_compound_add_fp32` (their
  // IR has structurally identical shapes but with a dynamic kernarg as
  // the icmp constant, not a compile-time K). The compile-time-K-only
  // rule catches `canary_bpermute_scan_fp32`'s Kogge-Stone scan-stage
  // predicates (K ∈ {1, 3, 7, 15}) while leaving the baselines green.
  //
  // Runs AFTER Phase 6 `PromoteMemToReg` so scratch-addrspace round-trips
  // are gone and the forward use-chain classifier operates on clean SSA.
  // Runs AFTER the Phase 6.5 writelane/readlane rewrite so the chain sees
  // the post-rewrite shapes (relevant when a future iteration widens the
  // classifier to audit additional users). Direction gate inside
  // `classifyPredicateChain` short-circuits when
  // `targetWaveSize <= sourceWaveSize`.
  //
  // No companion rewrite today. The design doc's §5 O2 "tid AND (W_s-1)"
  // rewrite is deferred (§6.2 documents the semantic-incorrectness of
  // the norm-family failing recipes and are a no-op for sub-case-2
  // scan-shaped recipes). If a future design iteration adds a principled
  // rewrite, pair it with a `RewriteId` alongside
  // `ObstructionKind::WorkitemIdPredicateChain`.
  {
    // Pass the projection actually selected for this kernel, not the
    // user-facing enable flag. Phantom-lane kernels route to MODREP above;
    // the classifier then decides whether that MODREP instance can have an
    // active replica lane before turning an observed C5 site into a refusal.
    PredicateChainProjection predProjection =
        useThreadLoop ? PredicateChainProjection::ThreadLoop
                      : (useWaveNative
                             ? PredicateChainProjection::WaveNative
                             : PredicateChainProjection::ModuloReplication);
    PredicateChainClassifierReport predReport =
        classifyPredicateChain(*F, isa.waveSize, targetIsa.waveSize,
                                predProjection,
                                /*maxFlatWorkgroupSize=*/
                                meta.maxFlatWorkgroupSize > 0
                                    ? static_cast<unsigned>(
                                          meta.maxFlatWorkgroupSize)
                                    : 0u,
                                useThreadLoop &&
                                    suppressC5ForThreadLoopRoute);

    if (!predReport.refused && !predReport.observedSites.empty()) {
      result.c5SuppressedCount +=
          static_cast<int>(predReport.observedSites.size());
      if (result.c5SuppressionReason.empty())
        result.c5SuppressionReason = predReport.suppressionReason;
      const char *projectionName =
          predProjection == PredicateChainProjection::ThreadLoop
              ? "ThreadLoopProjection"
              : (predProjection == PredicateChainProjection::WaveNative
                     ? "WaveNativeProjection"
                     : "ModuloReplicationProjection");
      LLVM_DEBUG({
        dbgs() << "c5-predicate-chain: observed "
               << predReport.observedSites.size()
               << " C5-shape site(s) in '" << kernelName << "' under "
               << projectionName
               << " (refusal "
                  "suppressed per c5_predicate_chain_classifier.hpp "
                  "projection contract):\n";
        for (const std::string &site : predReport.observedSites)
          dbgs() << "  - " << site << "\n";
      });
    }

    if (predReport.refused) {
      RaiseFailure f = RaiseFailure::crossWavePredicateChain(
          kernelName, predReport.refusalDetail);
      errs() << "transpiler: pre-translation abort: " << f.format << " on '"
             << f.mnemonic << "' \u2014 " << f.detail << "\n";
      errs() << "  outcome: (c) refuse \u2014 "
                "WorkitemIdPredicateChain (\u00a73 Class 5"
             << (predReport.waveNativePhantomRefusal
                     ? " phantom-lane sub-case"
                     : "")
             << ")\n";
      result.failure = std::move(f);
      return result;
    }
  }

  // ==== Phase 6.7: Link TDM emulation runtime ====
  // The cross-target VIMAGE handler emits calls to
  // `salmon_tdm_load_to_lds` / `salmon_tdm_store_from_lds` (declared,
  // no body) when the compilation target lacks the gfx1250 TENSORcnt
  // unit. Link the embedded HIP-authored runtime bitcode in here so
  // `verifyModule` sees a self-contained module and `llc` resolves the
  // calls at codegen time. No-op when the handler did not emit any
  // helper calls.
  if (moduleUsesTDMRuntime(M)) {
    if (!linkTDMRuntime(M, compilationTargetISA)) {
      errs() << "transpiler: TDM runtime link failed for kernel '" << kernelName << "'\n";
      result.failure = RaiseFailure::irVerificationFailed("TDM runtime bitcode link failed");
      return result;
    }
  }

  // ==== Phase 7: Verify IR ====
  std::string verifyErr;
  raw_string_ostream verifyOS(verifyErr);
  if (verifyModule(M, &verifyOS)) {
    errs() << "transpiler: IR verification failed:\n" << verifyErr << "\n";
    result.failure = RaiseFailure::irVerificationFailed(verifyErr);
    return result;
  }

  {
    raw_string_ostream irOS(result.irText);
    M.print(irOS, nullptr);
  }

  result.usesScratchPrivateSegment = ctx.usesScratchPrivateSegment;
  result.sourcePrivateSegmentFixedSize = ctx.sourcePrivateSegmentFixedSize;
  result.success = true;
  return result;
}

RaiseResult raiseToIR(const std::vector<uint8_t> &textBytes,
                      const std::string &sourceISA,
                      const std::string &kernelName,
                      const KernelMeta &meta,
                      uint64_t kernelOffset,
                      const std::string &compilationTargetISA,
                      bool enableWritelaneRewrite,
                      bool enableWaveNative) {
  return raiseToIRImpl(textBytes, sourceISA, kernelName, meta, kernelOffset,
                       compilationTargetISA, enableWritelaneRewrite,
                       enableWaveNative,
                       /*forceThreadLoopProjection=*/false,
                       /*suppressC5ForThreadLoopRoute=*/false);
}

} // namespace transpiler
