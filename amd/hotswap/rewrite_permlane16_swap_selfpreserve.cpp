#include "rewrite_permlane16_swap_selfpreserve.hpp"
#include "spe_phi_walker.hpp"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace llvm;

namespace transpiler {

namespace {

// Walk `v` backward through SPE active-arm phi edges to the
// deepest reachable non-phi SSA value, or to a phi that doesn't
// match the SPE diamond (return that phi as-is).  See
// `spe_phi_walker.hpp` for the structural identification of
// what constitutes an SPE active arm.
//
// `visited` guards against cycles (shouldn't happen for the
// linear SPE diamond chains `emitUnderExec` emits, but defensive
// against unexpected shapes after loop-aware optimisations).
Value *walkSpeActiveArmToRoot(Value *v, unsigned maxSteps,
                               SmallPtrSetImpl<Value *> &visited) {
  for (unsigned i = 0; i < maxSteps; ++i) {
    if (!v || !visited.insert(v).second)
      return v;
    auto *phi = dyn_cast<PHINode>(v);
    if (!phi)
      return v;
    Value *active = spePhiActiveArm(phi);
    if (!active || active == phi)
      return v;
    v = active;
  }
  return v;
}

Value *walkSpeActiveArmToRoot(Value *v,
                               unsigned maxSteps = kSpePhiWalkMaxSteps) {
  SmallPtrSet<Value *, 8> visited;
  return walkSpeActiveArmToRoot(v, maxSteps, visited);
}

// Test whether `a` and `b`, walked backward through SPE active-
// arm phi edges, reach the same SSA root.  Returns true iff:
//
//   1. `a == b` directly (trivial case: same bpermute data arg).
//   2. `walkSpeActiveArmToRoot(a) == walkSpeActiveArmToRoot(b)`.
//   3. One of `{a, b}` is already equal to the other's walked
//      root (asymmetric: one side bypasses the SPE phi
//      wrapping, the other side goes through it).
//
// These three conditions cover the shapes
// `emitPermLaneSwapEmulation` produces when both `vdstReg`'s and
// `src0Reg`'s last-stored value was the same SSA: one or both
// of the readReg32 loads may become that SSA directly post-mem2reg
// (case 1/3) or wrapped in an SPE preserve-arm phi (case 2).
bool sharedRootViaSpe(Value *a, Value *b) {
  if (!a || !b)
    return false;
  if (a == b)
    return true;
  Value *rootA = walkSpeActiveArmToRoot(a);
  Value *rootB = walkSpeActiveArmToRoot(b);
  if (rootA == rootB)
    return true;
  if (rootA == b || rootB == a)
    return true;
  return false;
}

// Match the `emitPermLaneSwapEmulation`-emitted bpermute pair:
//
//   %addr = shl i32 %partner, 2   ; emitted once per site
//   %new_vdst     = call bpermute(%addr, %src0_in)
//   %new_src0_out = call bpermute(%addr, %vdst_in)
//
// When `%vdst_in` and `%src0_in` trace to the same SSA root
// (see `sharedRootViaSpe`), returns the pair so the caller can
// rewrite the second call's result.  Otherwise returns `{}`.
//
// Both calls must be in the same basic block — the emission
// site is linear in `handle_valu_cross_lane.cpp` (two adjacent
// `CreateCall`s before any `writeReg32` SPE wrapping) so the
// post-PromoteMemToReg IR keeps them co-located.
struct MatchedPair {
  CallInst *newVdst = nullptr;
  CallInst *newSrc0Out = nullptr;
  Value *seedRoot = nullptr;
};

MatchedPair tryMatchPair(CallInst *first, CallInst *second) {
  MatchedPair m;
  if (!first || !second || first == second)
    return m;
  Function *f0 = first->getCalledFunction();
  Function *f1 = second->getCalledFunction();
  if (!f0 || !f1 ||
      f0->getIntrinsicID() != Intrinsic::amdgcn_ds_bpermute ||
      f1->getIntrinsicID() != Intrinsic::amdgcn_ds_bpermute)
    return m;
  if (first->getParent() != second->getParent())
    return m;
  if (first->getArgOperand(0) != second->getArgOperand(0))
    return m;  // address must be SSA-identical
  Value *a = first->getArgOperand(1);
  Value *b = second->getArgOperand(1);
  if (!sharedRootViaSpe(a, b))
    return m;
  // Use the walked root as the seed.  Prefer whichever of {a, b}
  // is already non-phi — that's the "canonical" seed SSA as
  // seen by the handler that emitted the swap.  When both are
  // SPE phis, fall through to the walked root (equivalent under
  // `sharedRootViaSpe`).  Any of these choices are semantically
  // interchangeable per the shared-root invariant.
  Value *seedRoot = nullptr;
  if (!isa<PHINode>(a))
    seedRoot = a;
  else if (!isa<PHINode>(b))
    seedRoot = b;
  else
    seedRoot = walkSpeActiveArmToRoot(a);
  m.newVdst = first;
  m.newSrc0Out = second;
  m.seedRoot = seedRoot;
  return m;
}

} // namespace

Permlane16SwapSelfPreserveRewriteReport
rewritePermLane16SwapSelfPreserve(Function &F) {
  Permlane16SwapSelfPreserveRewriteReport report;
  bool dbg = std::getenv("SALMON_DEBUG_PERMLANE16_SWAP_SELFPRESERVE") != nullptr;

  // Two-phase rewrite (collect then mutate) so the iterator over
  // `instructions(F)` doesn't see the IR change underneath us.
  //
  // Strategy: walk every BB, track the most-recent bpermute call
  // in that BB.  When a second bpermute appears with matching
  // address, try to match as a pair.  Reset on any non-matching
  // pair so the same `prev` doesn't re-match against a third
  // unrelated bpermute later in the BB.
  //
  // `emitPermLaneSwapEmulation` emits the two calls linearly
  // with no intervening bpermutes, so the sliding window is
  // tight in practice.  The matching conjunction (same addr SSA
  // + same-root data SSA) is narrow enough that an unrelated
  // bpermute pair in the same BB would need BOTH to share an
  // address SSA with the target pair (unlikely: each
  // `emitPermLaneSwapEmulation` computes its own `%pls16_addr`
  // via a site-local `shl`) AND both their data args to resolve
  // to the same root — which would itself be the Triton idiom
  // signature, a correct match.
  SmallVector<MatchedPair> sites;
  for (BasicBlock &BB : F) {
    CallInst *prev = nullptr;
    for (Instruction &I : BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;
      Function *f = CI->getCalledFunction();
      if (!f || f->getIntrinsicID() != Intrinsic::amdgcn_ds_bpermute)
        continue;
      if (!prev) {
        prev = CI;
        continue;
      }
      MatchedPair m = tryMatchPair(prev, CI);
      if (m.newSrc0Out) {
        sites.push_back(m);
        // Don't let `prev` also match against a future third
        // bpermute; reset.
        prev = nullptr;
      } else {
        // Not a match — advance the sliding window.
        prev = CI;
      }
    }
  }

  for (const MatchedPair &site : sites) {
    if (dbg)
      errs() << "[permlane16-swap-selfpreserve] rewriting %"
             << (site.newSrc0Out->hasName()
                     ? site.newSrc0Out->getName()
                     : "<bpermute>")
             << " -> %"
             << (site.seedRoot->hasName() ? site.seedRoot->getName()
                                           : "<seed>")
             << " (new_vdst untouched: %"
             << (site.newVdst->hasName() ? site.newVdst->getName()
                                          : "<bpermute>")
             << ")\n";
    site.newSrc0Out->replaceAllUsesWith(site.seedRoot);
    // Erase the now-dead bpermute call.  `ds_bpermute` is
    // `IntrNoMem` + `IntrConvergent`; ADCE would eliminate it
    // eventually, but doing so explicitly here avoids leaving a
    // wave-coordinated LDS round-trip in the emitted HSACO that
    // has no semantic effect.  Safe because `replaceAllUsesWith`
    // above transferred every use to `seedRoot`, so the call is
    // guaranteed to have no remaining users at this point.
    assert(site.newSrc0Out->use_empty() &&
           "dead bpermute call still has users after RAUW");
    site.newSrc0Out->eraseFromParent();
    report.matchedSites++;
  }

  if (dbg)
    errs() << "[permlane16-swap-selfpreserve] " << report.matchedSites
           << " site(s) rewritten in '" << F.getName() << "'\n";

  return report;
}

} // namespace transpiler
