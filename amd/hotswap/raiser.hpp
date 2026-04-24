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
// MODREP (`ModuloReplicationProjection`) is retained but opt-in via
// `--disable-wave-native` (raise_cli) or `enableWaveNative=false`
// (pipeline callers). It remains correct on the narrow class of
// kernels where source-wave-0's semantics replicate cleanly across
// the target wave (pointwise kernels with no cross-wave state;
// kernels that pass G1's obstruction classifier and whose cross-
// lane ops stay within a single source-wave half). The C5
// predicate-chain classifier (`c5_predicate_chain_classifier`) is
// structurally MODREP-scoped (the refusal rationale is MODREP-
// replica-specific) and short-circuits under WaveNative. Under
// the WaveNative default, every recipe in the compare_correctness
// Triton corpus that hits the C5 predicate-chain signature
// end-to-end MATCHes (canary_bpermute_scan_fp32, corpus_layernorm_fp32);
// see hotswap/docs/modrep-predicate-chain.md §6.4 for the
// orthogonal VOPD-cndmask / carry-chain SGPR-operand fixes that
// close those recipes independently of the predicate-chain class.
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
//
// `enablePermLane16Xor3PartnerRewrite` toggles the
// `rewrite_permlane16_xor3_partner` pass that substitutes
// `partner_seed` for the xor3 result at Triton's gfx1250
// cross-16 bitonic-merge idiom.
//
// Default **off** as of 2026-04-23: the pass was a transitional
// bridge that compensated for the SYMMETRIC `v_permlane16_swap_
// b32` lift that lived in `handle_valu_cross_lane.cpp` prior to
// the matmul_fp16 fix.  With the lift corrected to match the
// MI400 Shader Programming Guide's ASYMMETRIC semantic (§ V_
// PERMLANE16_SWAP_B32), the xor3 composition downstream already
// produces `partner_seed` through the standard arithmetic
// (lanes 0..15 see `seed^partner^seed = partner`, lanes 16..31
// see `partner^seed^seed = partner` symmetrically) without
// needing the bpermute-level RAUW.  Default-on would actively
// miscompile any kernel whose two bpermutes feed a matching
// outer xor pattern — the rewrite erases correct IR.
//
// The pass is retained for audit / bisection only — callers
// that want to reproduce the pre-fix (buggy) behaviour can
// pass `true` explicitly, or raise_cli exposes
// `--enable-permlane16-xor3-partner`.
//
// `enablePermLane16SwapSelfPreserveRewrite` toggles the
// `rewrite_permlane16_swap_selfpreserve` pass that rewrites the
// second output of `emitPermLaneSwapEmulation` from
// `partner_seed` to `seed` when BOTH bpermute data arguments
// trace (via SPE active-arm phi walks) to the same root SSA
// value.  Same story as the xor3-partner sibling: transitional
// compensation for the symmetric lift.  The correct asymmetric
// lift already produces the self-preserving output for the
// lanes that should preserve (lanes 16..31 keep `src0_in`,
// lanes 0..15 keep `vdst_in`), so the rewrite's `RAUW →
// seedRoot` is redundant on one half of the wave and wrong on
// the other.  Default **off**; raise_cli opt-in:
// `--enable-permlane16-swap-selfpreserve`.
RaiseResult raiseToIR(const std::vector<uint8_t> &textBytes,
                      const std::string &sourceISA,
                      const std::string &kernelName,
                      const KernelMeta &meta,
                      uint64_t kernelOffset = 0,
                      const std::string &compilationTargetISA = "",
                      bool enableWritelaneRewrite = true,
                      bool enableWaveNative = true,
                      bool enablePermLane16Xor3PartnerRewrite = false,
                      bool enablePermLane16SwapSelfPreserveRewrite = false);

} // namespace transpiler

#endif
