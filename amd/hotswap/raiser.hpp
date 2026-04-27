#ifndef HOTSWAP_TRANSPILER_RAISER_HPP
#define HOTSWAP_TRANSPILER_RAISER_HPP

#include "code_object_utils.hpp"
#include "raise_failure.hpp"

#include <memory>
#include <string>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace transpiler {

struct RaiseResult {
  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<llvm::Module> module;
  int liftedCount = 0;
  int totalCount = 0;
  std::string irText;
  std::string disasmText;
  // C5 sites that were observed by the post-mem2reg classifier but accepted
  // under the selected projection's proof obligation (for example
  // single-source-wave MODREP with no active replica lanes). Propagated to
  // the loader proof log so successful translations still carry attribution.
  int c5SuppressedCount = 0;
  std::string c5SuppressionReason;
  // Structured failure description. `failure.reason == None` iff `success`.
  RaiseFailure failure;
  bool success = false;
  bool hasDivergentExec = false;
};

// `enableWritelaneRewrite` toggles the post-raise rewrite pass that
// replaces cross-widen-divergent `v_writelane_b32` / `v_readlane_b32`
// sites with a per-source-wave `select` / `ds_bpermute` pair (see
// `rewrite_cross_lane_divergent.{hpp,cpp}` and wave-size-translation.md
// §5.6.3).
//
// Default **on** as of the Triton-corpus graduation: the rewrite pass
// is correct-by-construction under the §5.6.3 symmetry contract
// (every cross-lane site rewritten together under cross-widening), and
// the reduction-bearing recipes in the compare_correctness Triton
// corpus (`corpus_layernorm_fp32`, `corpus_softmax_fp32`) are the
// corpus-wide bit-exactness evidence the graduation plan asked for —
// their salmon lane miscompiles silently under the pre-rewrite path
// (the `v_readlane_b32 ..., 31` fan-in idiom only extracts one of the
// two source-wave partials on wave64 target) and lands `gold` /
// `match` under the rewrite.  Callers that specifically need to pin
// the pre-rewrite path (lit fixtures that assert the `UNCHANGED` /
// `REFUSE` sibling contract) pass `false` explicitly.
//
// `enableWaveNative` toggles `WaveNativeProjection` for wave32 source
// → wave64 target cross-widening.
//
// Default **on** as of the WaveNative graduation (see
// hotswap/docs/modrep-predicate-chain.md §6 "Picked: WaveNative as
// default" for the empirical evidence). Under the default, the
// kernel body runs with hardware EXEC = -1 after an `@llvm.amdgcn.
// init_whole_wave` prologue captures the original per-lane active
// mask into the (widened) EXEC alloca. Each target lane has its
// own modeled-EXEC bit — source-wave L's EXEC_L and target-wave's
// lane L+W_s are independent, NOT MODREP replicas — so kernels
// compiled with `num_warps > 1` (swiglu_fp32, corpus_layernorm_fp32)
// correctly project each source wave onto its own target-wavefront
// half. The compare_correctness sweep post-graduation shows
// `swiglu_fp32` flipping from WRONG 4/4 to match 4/4 and
// `corpus_layernorm_fp32` partial-matching with much smaller
// residual error.
//
// MODREP (`ModuloReplicationProjection`) is retained both as an explicit
// opt-out via `--disable-wave-native` / `enableWaveNative=false` and as the
// raiser's phantom-lane fallback for statically sub-wave workgroups. It
// remains correct when no active target replica lanes exist, and the C5
// predicate-chain classifier (`c5_predicate_chain_classifier`) now makes that
// launch-regime proof explicit: MODREP C5 refuses when active replica lanes
// can exist, but accepts `0 < max_flat_workgroup_size <= sourceWaveSize`
// because upper target lanes remain hardware-inactive. Under the WaveNative
// default, every recipe in the compare_correctness Triton corpus that hits the
// C5 predicate-chain signature end-to-end MATCHes (canary_bpermute_scan_fp32,
// corpus_layernorm_fp32); see hotswap/docs/modrep-predicate-chain.md §6.4 for
// the orthogonal VOPD-cndmask / carry-chain SGPR-operand fixes that close
// those recipes independently of the predicate-chain class.
//
// No process-global env-var override. `HSA_SALMON_WAVE_NATIVE=1`
// was a transient test/debug hook during the graduation sweep
// and was removed when the default flipped — the env var
// silently overriding `enableWaveNative=false` would defeat the
// opt-out path that `--disable-wave-native` (raise_cli) and
// `enableWaveNative=false` (programmatic callers, including lit
// fixtures that pin MODREP-shape IR invariants) rely on. If a
// future need for a global toggle arises, add a proper
// `PipelineConfig` field.
RaiseResult raiseToIR(const std::vector<uint8_t> &textBytes,
                      const std::string &sourceISA,
                      const std::string &kernelName,
                      const KernelMeta &meta,
                      uint64_t kernelOffset = 0,
                      const std::string &compilationTargetISA = "",
                      bool enableWritelaneRewrite = true,
                      bool enableWaveNative = true);

} // namespace transpiler

#endif
