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

#include "llvm/ADT/Twine.h"
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
#include <functional>
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

static bool isSemOpInRange(SemOp op, SemOp first, SemOp last) {
  auto v = static_cast<uint16_t>(op);
  return v >= static_cast<uint16_t>(first) &&
         v <= static_cast<uint16_t>(last);
}

static bool threadLoopUnsupportedWorkgroupMemoryOrBarrier(
    ArrayRef<DecodedInst> insts, std::string &detail) {
  for (const DecodedInst &di : insts) {
    StringRef kind;
    switch (di.semOp) {
    case SemOp::S_BARRIER:
    case SemOp::S_BARRIER_WAIT:
    case SemOp::S_BARRIER_SIGNAL:
      kind = "workgroup barrier";
      break;
    case SemOp::BUFFER_LOAD_DWORD_LDS:
    case SemOp::BUFFER_LOAD_DWORDX4_LDS:
    case SemOp::TENSOR_LOAD_TO_LDS:
    case SemOp::TENSOR_STORE_FROM_LDS:
    case SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B8:
    case SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B32:
    case SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B64:
    case SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B128:
      kind = "LDS access";
      break;
    default:
      if (isSemOpInRange(di.semOp, SemOp::DS_LOAD_TR16_B128,
                         SemOp::DS_SWIZZLE_B32))
        kind = "LDS access";
      break;
    }

    if (!kind.empty()) {
      detail = (Twine("ThreadLoopProjection is not yet safe for kernels "
                      "containing ") +
                kind + " (" + semOpName(di.semOp) + " at offset 0x" +
                Twine::utohexstr(di.offset) +
                "); barrier hoisting and LDS aliasing checks are still "
                "unimplemented, so refusing is safer than launching a "
                "translated kernel that can fault or miscompile.")
                   .str();
      return true;
    }
  }
  return false;
}

} // namespace

// parseReg, readOp32/64/ExecWidth, and OpResolver are in raise_context.hpp/cpp
// instructionWritesEXEC and the cross-wave gate live in wave_projection.hpp/cpp
// RaiseFailure + reasonString are in raise_failure.hpp/cpp

// ============================================================================
// Main raising function
// ============================================================================

static RaiseResult raiseToIRImpl(llvm::ArrayRef<uint8_t> textBytes,
                                 llvm::StringRef sourceISA,
                                 llvm::StringRef kernelName,
                                 const KernelMeta &meta,
                                 uint64_t kernelOffset,
                                 llvm::StringRef compilationTargetISA,
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
              "cross-widen route; writelane/readlane rewrite may be "
              "disabled by the retry caller)\n";
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
      // The factory names the class in `format`; surface the full trace in
      // `detail` so diagnostics can carry the per-site context forward without
      // re-invoking the classifier.
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

  // Build function signature: a single opaque
  // `ptr byref([N x i8]) align 16` placeholder whose only job is to
  // make the AMDGPU backend emit `kernarg_segment_size = N` and
  // `kernarg_segment_align = 16` in the lifted kernel's KD/metadata,
  // so the runtime's kernarg buffer reaches the kernel intact and
  // the metadata reports the AMDGPU ABI's 16-byte minimum.
  //
  // The handlers do NOT read this argument — kernarg loads lift to
  // GEP+load against `amdgcn_kernarg_segment_ptr` and let the AMDGPU
  // backend re-select `s_load_*` against the kernarg segment. The
  // typed source-ABI signature (ptr addrspace(1) / i32 / i64 / per-
  // dword aggregate split) is therefore unnecessary on the lifted
  // side.
  //
  // Why `byref` + `align`: AMDGPULowerKernelArguments consults the
  // `align` parameter attribute only for byref kernel args (see
  // `MaybeAlign ParamAlign = IsByRef ? Arg.getParamAlign() :
  // std::nullopt;` in LLVM's `AMDGPULowerKernelArguments.cpp`). For
  // a non-byref `[N x i8]` arg, the IR-level alignment is the
  // type's natural alignment (1 byte), and the YAML metadata's
  // `.kernarg_segment_align` field reports a smaller value than the
  // ABI's 16-byte minimum. Using `byref` with an explicit
  // `align(16)` lets the backend honour the alignment without
  // forcing a vector or padding type, and the byref semantics —
  // "pointer to an aggregate that's actually placed in the kernarg
  // segment" — match the placeholder's intent: a stable region of
  // `kernarg_segment_size` bytes that handlers don't need a typed
  // view of.
  //
  // AMDGPULowerKernelArguments skips load emission for arguments
  // that are `use_empty()` but still bumps the cumulative arg
  // offset, so the unused placeholder still contributes to
  // `kernarg_segment_size`.
  //
  // Test back-reference: every lit fixture under `lit_tests/` pins
  // either a `ptr addrspace(4)` GEP shape or an addrspace(1) global
  // GEP shape against the segment_ptr intrinsic — none of them rely
  // on the kernarg buffer being a typed Function argument list.
  SmallVector<Type *, 1> paramTypes;
  KernargLayout kernargs;
  int paramIdx = 0;
  Type *kernargByrefTy = nullptr;
  if (meta.kernargSegmentSize > 0) {
    kernargByrefTy =
        ArrayType::get(i8Ty, static_cast<uint64_t>(meta.kernargSegmentSize));
    paramTypes.push_back(PointerType::get(C, /*addrspace=*/4));
    paramIdx = 1;
  }
  kernargs.implicitArgsBase = meta.implicitArgsBase();
  kernargs.kernargSegmentSize = meta.kernargSegmentSize;

  auto *funcTy = FunctionType::get(voidTy, paramTypes, false);
  Function *F =
      Function::Create(funcTy, GlobalValue::ExternalLinkage, kernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);

  // Attach `byref([N x i8])` + `align(16)` to the placeholder kernarg
  // pointer. AMDGPULowerKernelArguments only honours param-align on
  // byref kernel args, so this combo is what gets the lifted KD's
  // kernarg-segment alignment to the AMDGPU ABI's 16-byte minimum
  // without forcing an aggregate / vector type for the parameter.
  if (kernargByrefTy != nullptr) {
    F->addParamAttr(0, Attribute::getWithByRefType(C, kernargByrefTy));
    F->addParamAttr(0, Attribute::getWithAlignment(C, Align(16)));
  }
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

    // The hotswap caller still launches with the source kernel's host-side
    // kernarg buffer.  Salmon materialises every source-visible value either
    // as a normal formal parameter, as source-ABI preloaded SGPR state seeded
    // explicitly in IR below, or as an intrinsic for architected dispatch
    // state.  Suppress backend-invented implicit kernarg slots so the emitted
    // descriptor keeps the source kernarg size instead of appending a
    // target-default hidden-arg block that the host never populated.
    F->addFnAttr("amdgpu-no-cluster-id-x");
    F->addFnAttr("amdgpu-no-cluster-id-y");
    F->addFnAttr("amdgpu-no-cluster-id-z");
    F->addFnAttr("amdgpu-no-completion-action");
    F->addFnAttr("amdgpu-no-default-queue");
    F->addFnAttr("amdgpu-no-dispatch-id");
    F->addFnAttr("amdgpu-no-dispatch-ptr");
    F->addFnAttr("amdgpu-no-heap-ptr");
    F->addFnAttr("amdgpu-no-hostcall-ptr");
    F->addFnAttr("amdgpu-no-implicitarg-ptr");
    F->addFnAttr("amdgpu-no-lds-kernel-id");
    F->addFnAttr("amdgpu-no-multigrid-sync-arg");
    F->addFnAttr("amdgpu-no-queue-ptr");
    F->addFnAttr("amdgpu-no-workitem-id-x");
    F->addFnAttr("amdgpu-no-workitem-id-y");
    F->addFnAttr("amdgpu-no-workitem-id-z");
    F->addFnAttr("uniform-work-group-size", "true");
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

  if (paramIdx > 0)
    F->getArg(0)->setName("kargs");

  errs() << "transpiler: Kernel '" << kernelName
         << "' kernarg_segment_size=" << meta.kernargSegmentSize << "\n";

  Function *fnWorkgroupIdX =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_x);
  Function *fnWorkgroupIdY =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_y);
  Function *fnKargPtr =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_kernarg_segment_ptr);
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
  // `blockStarts` is a std::set (see decode.hpp) so it iterates in
  // ascending source-address order, giving deterministic BB labels.
  // `offsetToBB` is a DenseMap and intentionally unordered; for the
  // thread-loop entry BB we need the lowest-address BB as InsertBefore
  // (so the entry sorts above the kernel body in IR), which we capture
  // explicitly during the create loop.
  llvm::DenseMap<uint64_t, BasicBlock *> offsetToBB;
  BasicBlock *firstBodyBB = nullptr;
  for (uint64_t addr : blockStarts) {
    BasicBlock *bb =
        BasicBlock::Create(C, "bb_0x" + utohexstr(addr - kernelOffset), F);
    offsetToBB[addr] = bb;
    if (!firstBodyBB)
      firstBodyBB = bb;
  }
  BasicBlock *entryBB = useThreadLoop
                            ? BasicBlock::Create(C, "entry", F, firstBodyBB)
                            : offsetToBB[kernelOffset];

  // ==== Phase 4: Init entry registers ====
  IRBuilder<> B(entryBB);

  AllocaRegFile regs;
  regs.init(B, i32Ty, i1Ty, isa, *mc.regInfo, projection);

  // Seed kernel-entry SGPR state from the descriptor-derived user-SGPR ABI.
  //
  // Crucial invariant: never hardcode SGPR indices. Kernarg preload and
  // enable_sgpr_* toggles legally move the kernarg pointer and workgroup-id
  // SGPRs away from s[0:1]/s2/s3. Hardcoding those indices mis-seeds entry
  // state and turns real source values into undef reads on the JIT path.
  //
  // Seed the kernarg pair with ptrtoint(amdgcn_kernarg_segment_ptr) so the
  // generic GEP+load path in handle_smem.cpp materialises kernarg SMEM
  // loads as real scalar loads (the backend selects s_load_* off the
  // addrspace(4) cast). storeSGPR64 ptrtoint-splits the pointer into two
  // i32 halves; loadSGPR64 reconstructs and the SMEM handler casts back
  // to ptr addrspace(4).
  if (userSgprLayout.kernargSegmentPtrSgpr >= 0) {
    regs.storeSGPR64(B, userSgprLayout.kernargSegmentPtrSgpr,
                     B.CreateCall(fnKargPtr, {}, "kernarg_ptr"));
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
  // segment before kernel entry. Materialize the same dwords by loading
  // through `amdgcn_kernarg_segment_ptr` so the AMDGPU backend handles the
  // ABI lowering uniformly: the GEP+load lowers back to `s_load_b32` (or a
  // hardware-preload SGPR read on gfx12+) against the kernarg segment, with
  // identical bytes to what the source kernel saw at entry.
  //
  // Hidden block counts (Triton's hidden_block_count_* ABI) still need
  // dispatch-packet synthesis since their values aren't stored in the
  // kernarg segment at all — only `emitPreloadedHiddenKernargDword` can
  // materialize them from `amdgcn_dispatch_ptr`.
  for (size_t sgprIdx = 0; sgprIdx < userSgprLayout.entries.size(); ++sgprIdx) {
    const auto &entry = userSgprLayout.entries[sgprIdx];
    if (entry.source != UserSgprLayout::Source::PreloadedKernarg)
      continue;
    Value *dw = emitPreloadedHiddenKernargDword(entry.kernargByteOffset);
    if (!dw) {
      Value *segPtr = B.CreateCall(fnKargPtr, {}, "preload_kernarg_ptr");
      Value *gep = B.CreateInBoundsGEP(
          i8Ty, segPtr, B.getInt64(entry.kernargByteOffset), "preload_gep");
      dw = B.CreateAlignedLoad(i32Ty, gep, Align(4), "preload_dw");
    }
    regs.storeSGPR32(B, static_cast<int>(sgprIdx), dw);
  }
  auto seedWorkitemX = [&](IRBuilder<> &SeedB) {
    regs.storeVGPR32(SeedB, 0, projection.emitWorkitemIdX(SeedB));
  };

  if (!useThreadLoop)
    seedWorkitemX(B);

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
  std::function<void(IRBuilder<> &)> seedTtmp8 = [](IRBuilder<> &) {};
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

    seedTtmp8 = [&](IRBuilder<> &SeedB) {
      // wave_id = workitem_id_x / wavefront_size (32 for gfx12)
      Value *tidForTtmp = projection.emitWorkitemIdX(SeedB);
      tidForTtmp->setName("ttmp8_tid");
      Value *waveId =
          SeedB.CreateLShr(tidForTtmp, SeedB.getInt32(5), "wave_id_in_wg");
      Value *ttmp8Val =
          SeedB.CreateShl(waveId, SeedB.getInt32(25), "ttmp8_val");
      SeedB.CreateStore(ttmp8Val, regs.ttmp[8]);
    };
    if (!useThreadLoop)
      seedTtmp8(B);
  }

  auto seedThreadLoopIterationState = [&](IRBuilder<> &SeedB) {
    for (auto *slot : regs.sgpr)
      SeedB.CreateStore(ConstantInt::get(i32Ty, 0), slot);
    for (auto *slot : regs.vgpr)
      SeedB.CreateStore(ConstantInt::get(i32Ty, 0), slot);
    for (auto *slot : regs.agpr)
      SeedB.CreateStore(ConstantInt::get(i32Ty, 0), slot);
    for (auto *slot : regs.ttmp)
      SeedB.CreateStore(ConstantInt::get(i32Ty, 0), slot);
    SeedB.CreateStore(ConstantInt::get(i32Ty, 0), regs.m0);
    SeedB.CreateStore(ConstantInt::get(i32Ty, 0), regs.flatScr[0]);
    SeedB.CreateStore(ConstantInt::get(i32Ty, 0), regs.flatScr[1]);

    // Mirror the entry-BB user-SGPR seeding above: the kernarg pair is
    // re-seeded with `amdgcn_kernarg_segment_ptr` so kernarg SMEM loads
    // inside the thread-loop iteration body lift through the same
    // GEP+load shape, and preloaded-kernarg SGPRs materialise their
    // dwords via the same intrinsic + GEP + i32 load. Hidden block
    // counts continue to flow through `emitPreloadedHiddenKernargDword`
    // (dispatch-packet synthesis, not in kernarg memory).
    if (userSgprLayout.kernargSegmentPtrSgpr >= 0) {
      regs.storeSGPR64(SeedB, userSgprLayout.kernargSegmentPtrSgpr,
                       SeedB.CreateCall(fnKargPtr, {}, "kernarg_ptr"));
    }
    if (userSgprLayout.workgroupIdXSgpr >= 0) {
      regs.storeSGPR32(SeedB, userSgprLayout.workgroupIdXSgpr,
                       SeedB.CreateCall(fnWorkgroupIdX, {}, "wg_id_x"));
    }
    if (userSgprLayout.workgroupIdYSgpr >= 0) {
      regs.storeSGPR32(SeedB, userSgprLayout.workgroupIdYSgpr,
                       SeedB.CreateCall(fnWorkgroupIdY, {}, "wg_id_y"));
    }
    for (size_t sgprIdx = 0; sgprIdx < userSgprLayout.entries.size();
         ++sgprIdx) {
      const auto &entry = userSgprLayout.entries[sgprIdx];
      if (entry.source != UserSgprLayout::Source::PreloadedKernarg)
        continue;
      Value *dw = emitPreloadedHiddenKernargDword(entry.kernargByteOffset);
      if (!dw) {
        Value *segPtr =
            SeedB.CreateCall(fnKargPtr, {}, "preload_kernarg_ptr");
        Value *gep = SeedB.CreateInBoundsGEP(
            i8Ty, segPtr, SeedB.getInt64(entry.kernargByteOffset),
            "preload_gep");
        dw = SeedB.CreateAlignedLoad(i32Ty, gep, Align(4), "preload_dw");
      }
      regs.storeSGPR32(SeedB, static_cast<int>(sgprIdx), dw);
    }

    if (AMDGPU::isGFX12Plus(*mc.subtargetInfo)) {
      SeedB.CreateStore(SeedB.CreateCall(fnWorkgroupIdX, {}, "ttmp9_wg_id"),
                        regs.ttmp[9]);
      Value *wgIdY = SeedB.CreateCall(fnWorkgroupIdY, {}, "ttmp7_wg_id_y");
      Function *fnWorkgroupIdZ = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::amdgcn_workgroup_id_z);
      Value *wgIdZ = SeedB.CreateCall(fnWorkgroupIdZ, {}, "ttmp7_wg_id_z");
      Value *wgIdYLo =
          SeedB.CreateAnd(wgIdY, SeedB.getInt32(0xFFFF), "wg_id_y_lo16");
      Value *wgIdZHi =
          SeedB.CreateShl(wgIdZ, SeedB.getInt32(16), "wg_id_z_hi16");
      Value *ttmp7Val = SeedB.CreateOr(wgIdYLo, wgIdZHi, "ttmp7_val");
      SeedB.CreateStore(ttmp7Val, regs.ttmp[7]);
      seedTtmp8(SeedB);
    }

    seedWorkitemX(SeedB);
    regs.storeVCC(SeedB, ConstantInt::getFalse(i1Ty));
    regs.storeSCC(SeedB, ConstantInt::getFalse(i1Ty));
    regs.storeExec(SeedB, projection.emitInitialExec(SeedB));
  };

  // ==== Phase 5: Raise each instruction ====

  auto *f16Ty = Type::getHalfTy(C);
  // `userSgprLayout` was built above before Phase 4 so entry SGPR seeding
  // and handler-side ABI decisions use the same descriptor-derived mapping.
  RaiseContext ctx{C, M, B, regs, projection, mc, isa, targetIsa, kernargs,
                   &userSgprLayout, F,
                   nullptr,
                   i1Ty, i8Ty, i32Ty, i64Ty, f32Ty, f16Ty,
                   ptrGlobalTy, offsetToBB};
  ctx.setpcAnalysis = &setpcAnalysis;
  ctx.sourcePrivateSegmentFixedSize =
      static_cast<uint32_t>(std::max(meta.privateSegmentFixedSize, 0));
  ctx.sourceComputePgmRsrc2 = meta.computePgmRsrc2;
  ctx.sourceKernelCodeProperties = meta.kernelCodeProperties;

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

  if (useThreadLoop) {
    auto *iterA = B.CreateAlloca(i32Ty, nullptr, "tl_iter_alloca");
    B.CreateStore(B.getInt32(0), iterA);
    static_cast<ThreadLoopProjection *>(projectionPtr.get())
        ->setIterationAlloca(iterA);

    BasicBlock *condBB = BasicBlock::Create(C, "tl_cond", F);
    BasicBlock *latchBB = BasicBlock::Create(C, "tl_latch", F);
    BasicBlock *doneBB = BasicBlock::Create(C, "tl_done", F);
    ctx.threadLoopLatch = latchBB;

    B.CreateBr(condBB);
    B.SetInsertPoint(condBB);

    Value *iter = B.CreateLoad(i32Ty, iterA, "tl_iter_val");
    Value *iterOk = B.CreateICmpULT(
        iter, B.getInt32(targetIsa.waveSize / isa.waveSize), "tl_iter_ok");
    Value *lane = projection.emitLaneIdx(B);
    Value *laneOk =
        B.CreateICmpULT(lane, B.getInt32(isa.waveSize), "tl_lane_ok");
    Value *enterBody = B.CreateAnd(iterOk, laneOk, "tl_enter_body");

    seedThreadLoopIterationState(B);
    for (auto *validA : ctx.sgprWaveMaskValidShadow)
      B.CreateStore(B.getFalse(), validA);

    B.CreateCondBr(enterBody, offsetToBB[kernelOffset], latchBB);

    B.SetInsertPoint(latchBB);
    Value *oldIter = B.CreateLoad(i32Ty, iterA, "tl_iter_old");
    Value *nextIter = B.CreateAdd(oldIter, B.getInt32(1), "tl_iter_next");
    B.CreateStore(nextIter, iterA);
    Value *more = B.CreateICmpULT(
        nextIter, B.getInt32(targetIsa.waveSize / isa.waveSize), "tl_more");
    B.CreateCondBr(more, condBB, doneBB);

    B.SetInsertPoint(doneBB);
    B.CreateRetVoid();
  }

  int raisedCount = 0;

  for (size_t instIdx = 0; instIdx < insts.size(); ++instIdx) {
    const DecodedInst &di = insts[instIdx];

    // If a terminator ended the recovered CFG path and the next decoded
    // instruction is not a known block leader, that instruction is unreachable
    // fallthrough bytes (often code after an unconditional branch). Do not emit
    // it into the already-terminated LLVM block.
    auto bbIt = offsetToBB.find(di.offset);
    if (B.GetInsertBlock()->hasTerminator() && bbIt == offsetToBB.end())
      continue;

    // Source-BB boundary handling uses `B.GetInsertBlock()` rather than a
    // tracked `currentBB` so that intra-handler CFG splits (emitUnderExec
    // diamonds under SPE) propagate correctly: fall-through must leave
    // from whatever block the builder is currently at — which is the
    // `spe_skip` tail when the last emission was wrapped — not from the
    // block that started the source instruction.
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
        std::string threadLoopUnsupportedDetail;
        if (threadLoopUnsupportedWorkgroupMemoryOrBarrier(
                insts, threadLoopUnsupportedDetail)) {
          errs() << "transpiler: thread-loop fallback not eligible for kernel '"
                 << kernelName << "': " << threadLoopUnsupportedDetail
                 << "\n";
          RaiseFailure f = RaiseFailure::crossWaveRewriteOracleDisagreement(
              kernelName, threadLoopUnsupportedDetail);
          errs() << "transpiler: post-raise abort: " << f.format << " on '"
                 << f.mnemonic << "' — " << f.detail << "\n";
          result.failure = std::move(f);
          return result;
        }
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
        for (llvm::StringRef site : predReport.observedSites)
          dbgs() << "  - " << site << "\n";
      });
    }

    if (predReport.refused) {
      auto hasMatrixOp = [&]() {
        const auto first =
            static_cast<uint16_t>(SemOp::V_MFMA_F32_16x16x128_F8F6F4);
        const auto last =
            static_cast<uint16_t>(SemOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4);
        for (const DecodedInst &inst : insts) {
          const auto op = static_cast<uint16_t>(inst.semOp);
          if (op >= first && op <= last)
            return true;
        }
        return false;
      };
      constexpr bool kEnableThreadLoopC5Retry = false;
      const bool canRetryThreadLoop =
          kEnableThreadLoopC5Retry &&
          predReport.waveNativeEqualityRefusal && !forceThreadLoopProjection &&
          targetIsa.waveSize > isa.waveSize &&
          (targetIsa.waveSize % isa.waveSize) == 0 && !hasMatrixOp();
      if (canRetryThreadLoop) {
        errs() << "transpiler: post-raise fallback: retrying kernel '"
               << kernelName
               << "' under ThreadLoopProjection after WaveNative C5 equality "
                  "refusal (analysis-triggered, no user opt-in)\n";
        errs() << "transpiler: thread-loop fallback trigger: "
               << predReport.refusalDetail << "\n";
        return raiseToIRImpl(textBytes, sourceISA, kernelName, meta,
                             kernelOffset, compilationTargetISA,
                             /*enableWritelaneRewrite=*/false,
                             /*enableWaveNative=*/false,
                             /*forceThreadLoopProjection=*/true,
                             /*suppressC5ForThreadLoopRoute=*/true);
      }
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

RaiseResult raiseToIR(llvm::ArrayRef<uint8_t> textBytes,
                      llvm::StringRef sourceISA,
                      llvm::StringRef kernelName,
                      const KernelMeta &meta,
                      uint64_t kernelOffset,
                      llvm::StringRef compilationTargetISA,
                      bool enableWritelaneRewrite,
                      bool enableWaveNative) {
  return raiseToIRImpl(textBytes, sourceISA, kernelName, meta, kernelOffset,
                       compilationTargetISA, enableWritelaneRewrite,
                       enableWaveNative,
                       /*forceThreadLoopProjection=*/false,
                       /*suppressC5ForThreadLoopRoute=*/false);
}

} // namespace transpiler
