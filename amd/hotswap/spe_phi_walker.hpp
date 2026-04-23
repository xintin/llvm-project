// STRUCTURAL (name-independent) walker over phis produced by
// `RaiseContext::emitUnderExec` + `PromoteMemToReg`.
//
// Context
// =======
//
// `emitUnderExec` in `raise_context.cpp` emits an SPE diamond:
//
//   preBB:
//     %active = <lane_active i1>
//     br i1 %active, label %spe_do, label %spe_skip
//   spe_do:
//     <body that stores into an alloca>
//     br label %spe_skip
//   spe_skip:
//     <continuation>
//
// After Phase 6's `PromoteMemToReg`, every alloca store inside
// the `spe_do` block becomes a phi at the `spe_skip` merge:
//
//   spe_skip:
//     %vgprX.N = phi i32 [ %new_val, %spe_do ],
//                         [ %old_val, %preBB ]
//
// The ACTIVE arm is the incoming from `%spe_do` (the lane-
// active-only new value); the PRESERVE arm is the incoming from
// `%preBB` (the prior SSA version of the same register).
//
// Siblings passes (`rewrite_permlane16_xor3_partner`,
// `rewrite_permlane16_swap_selfpreserve`) both need to walk
// backward through these SPE phis to recover "what SSA value
// was actually stored by the active-lane side of the diamond"
// and compare across register writes.  Identifying the active
// arm by BB name prefix (`starts_with("spe_do")`) is brittle —
// it would silently mis-walk if `emitUnderExec` ever renamed
// the blocks.  This helper does the identification STRUCTURALLY:
//
//   A phi's incoming block `P` is an SPE active arm iff:
//     1. `P`'s terminator is an unconditional branch to the
//        phi's parent block (i.e. `P` is the `spe_do` block
//        falling through to `spe_skip`).
//     2. `P` has exactly ONE predecessor `Q` (the `preBB`).
//     3. `Q`'s terminator is a conditional branch whose two
//        successors are exactly `{P, phi->getParent()}` — the
//        `spe_do` / `spe_skip` pair of the SPE diamond.
//
// Any phi shape that does NOT match all three conditions is
// treated as "not an SPE phi" — the walker returns nullptr and
// callers fall back to their no-match path.
//
// Correctness argument
// --------------------
//
// The conjunction of (1) + (2) + (3) is the precise structural
// shape `emitUnderExec` + alloca-store + PromoteMemToReg
// produces.  Nothing ELSE in the raiser produces this shape:
//   * Regular basic-block joins in the lifted CFG don't have
//     the "the preserve-arm predecessor dominates the active-
//     arm predecessor via a conditional branch on i1" structure.
//   * Loop-header phis have a latch predecessor, not a
//     conditional-br-on-i1 preBB.
//   * `writeReg32(EXEC_LO)` invalidates the `laneActive` memo
//     (see `raise_context.cpp`) which keeps distinct
//     `emitUnderExec` calls structurally independent.
//
// If a future optimisation pass were to simplify the SPE
// diamond away (e.g. jump-threading collapsing the conditional
// br), the walker returns nullptr — a miss, not a misdirection.

#pragma once

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"

namespace transpiler {

// Return the active-arm incoming value of an SPE-shaped phi, or
// nullptr if the phi doesn't structurally match the SPE diamond.
inline llvm::Value *spePhiActiveArm(llvm::PHINode *phi) {
  if (!phi)
    return nullptr;
  llvm::BasicBlock *mergeBB = phi->getParent();
  for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
    llvm::BasicBlock *pred = phi->getIncomingBlock(i);
    if (!pred)
      continue;
    // (1) pred's terminator is an unconditional br to mergeBB.
    //     LLVM 23 splits the legacy `BranchInst` into distinct
    //     `UncondBrInst` / `CondBrInst` subclasses; we match
    //     against the new classes directly so the walker is
    //     forward-compatible with the deprecation.
    auto *predBr =
        llvm::dyn_cast<llvm::UncondBrInst>(pred->getTerminator());
    if (!predBr || predBr->getSuccessor(0) != mergeBB)
      continue;
    // (2) pred has exactly one predecessor.
    llvm::BasicBlock *preBB = pred->getSinglePredecessor();
    if (!preBB)
      continue;
    // (3) preBB's terminator is a conditional br to {pred, mergeBB}.
    auto *preBr =
        llvm::dyn_cast<llvm::CondBrInst>(preBB->getTerminator());
    if (!preBr)
      continue;
    llvm::BasicBlock *t = preBr->getSuccessor(0);
    llvm::BasicBlock *f = preBr->getSuccessor(1);
    bool spePair = (t == pred && f == mergeBB) || (f == pred && t == mergeBB);
    if (!spePair)
      continue;
    return phi->getIncomingValue(i);
  }
  return nullptr;
}

// Shared phi-walk depth.  Set at the value both sibling passes
// historically used (8 steps) — nested SPE wrappers stay well
// under this bound in practice; the deepest observed chain in
// the Triton corpus is 3 hops (bpermute → spe_phi →
// write-preserving-phi → seed).
constexpr unsigned kSpePhiWalkMaxSteps = 8;

} // namespace transpiler
