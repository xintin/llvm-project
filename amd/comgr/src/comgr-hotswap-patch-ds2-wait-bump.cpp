//===- comgr-hotswap-patch-ds2-wait-bump.cpp - DS2 s_wait_dscnt bump ------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Strong-symbol override for applyDS2WaitBump, the post-pass that
/// compensates `s_wait_dscnt` immediates for the extra DS operations
/// produced by the DS2 split patches that land alongside (e.g. PR #2281
/// for non-stride64 variants and PR #2212 for stride64 variants).
///
/// Each split DS2 instruction (e.g. `ds_load_2addr_b32 -> 2x ds_load_b32`)
/// introduces one additional outstanding DS op. A trailing
/// `s_wait_dscnt N` (with N > 0) then under-counts by the number of split
/// sites in its basic block, because the wait imm was set by the compiler
/// against the original (unsplit) instruction sequence. This pass walks
/// the decoded stream, accumulates the per-wait deficit in a
/// `DenseMap<wait_offset, WaitBumpEntry>`, and re-encodes each affected
/// wait once with the bumped immediate (clamped at the architectural
/// maximum).
///
/// Why a post-pass and not a per-DS2 bump:
///   The original implementation tried to bump the next wait inline as
///   each DS2 was patched. That approach reads from the cached
///   `Ctx.Decoded[I].Inst` per call, so when multiple DS2 sites precede
///   the same wait, every call writes `original + 1` instead of the
///   accumulated `original + K` -- only the last bump is observable.
///   yxsamliu flagged this in PR #2281 review thread r3148375185 and the
///   same shape on PR #2212. This whole-function pass is structurally
///   immune: each affected wait is re-encoded exactly once with the sum
///   of all preceding DS2 sites in its basic block.
///
/// Counter semantics:
///   `s_wait_dscnt N` waits until the dscnt counter is at most N. Each
///   split adds one outstanding DS op; bumping the imm by K accounts for
///   K splits between the previous wait/terminator and this wait.
///
///   Drain waits (`s_wait_dscnt 0`) are intentionally NOT bumped: the
///   drain semantic (wait until ZERO outstanding) is independent of how
///   many ops were issued, so it remains correct after splits. Bumping a
///   drain to `s_wait_dscnt K` would relax it, leaving K split halves
///   potentially in flight at the wave's exit from the wait -- the
///   opposite of what the compiler asked for.
///
/// Mnemonic-vs-opcode design choice:
///   The `s_wait_dscnt` and DS2 mnemonic comparisons here use string
///   match (`DI.Mnemonic == "..."` and an explicit `StringSwitch`)
///   rather than cached MC opcode indices in `LLVMState`. This keeps the
///   PR self-contained -- adding `SWaitDscntOpcode` plus the DS2 opcode
///   table would need a separate touch on `comgr-hotswap-llvm.cpp`. The
///   cached-primitive consolidation (`CachedInst` struct) is tracked in
///   #2253; this pass should migrate alongside the rest of the cached
///   instruction state when that lands.
///
/// Limitations:
///   - Within-BB only: the scan resets at every basic-block terminator
///     (branch / call / endpgm / set-PC), so a wait reachable only via
///     control flow from a DS2 in another block is not bumped.
///   - Unconditional bump per DS2 mnemonic match: we assume any DS2 site
///     paired with this rewriter has been split. If a DS2 site is NOT
///     split (e.g. neither #2281 nor #2212 is merged yet), the bump
///     over-counts; on those builds the DS2 is broken on A0 anyway, so
///     the over-count is irrelevant.
///   - Saturating clamp at MaxWaitDscnt (63 = 6-bit field); kernels that
///     would exceed this need an extra wait inserted, which is out of
///     scope for this post-pass.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

// MSVC has no weak-symbol support; the stub in comgr-hotswap-b0a0.cpp
// becomes a regular definition there and this file would emit a duplicate
// (LNK2005). Mirrored from comgr-hotswap-patch-inplace.cpp.
#if !defined(_MSC_VER)

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {

// Source of truth for the dscnt field width: gfx12 SOPP encoding declares
// a 6-bit immediate for `s_wait_dscnt` (see SIMM6 field on the `S_WAIT_*`
// records in `llvm/lib/Target/AMDGPU/SOPInstructions.td`). Values >63 are
// not representable; clamping here keeps the encoder from rejecting bumps
// on kernels with many DS2 sites between two waits.
namespace {
constexpr unsigned MaxWaitDscnt = 63;
} // namespace

bool isDS2Mnemonic(StringRef Mnemonic) {
  return StringSwitch<bool>(Mnemonic)
      .Case("ds_load_2addr_b32", true)
      .Case("ds_load_2addr_b64", true)
      .Case("ds_store_2addr_b32", true)
      .Case("ds_store_2addr_b64", true)
      .Case("ds_load_2addr_stride64_b32", true)
      .Case("ds_load_2addr_stride64_b64", true)
      .Case("ds_store_2addr_stride64_b32", true)
      .Case("ds_store_2addr_stride64_b64", true)
      .Default(false);
}

SmallVector<uint8_t> encodeWaitWithImm(const MCInst &Wait, unsigned NewImm,
                                       const LLVMState &LS,
                                       uint32_t ExpectedSize) {
  if (Wait.getNumOperands() == 0 || !Wait.getOperand(0).isImm())
    return {};
  if (NewImm > MaxWaitDscnt)
    NewImm = MaxWaitDscnt;

  MCInst BumpedInst = Wait;
  BumpedInst.getOperand(0).setImm(NewImm);

  SmallVector<char> Code;
  SmallVector<MCFixup> Fixups;
  LS.MCE->encodeInstruction(BumpedInst, Code, Fixups, *LS.STI);
  if (Code.size() != ExpectedSize)
    return {};
  return SmallVector<uint8_t>(Code.begin(), Code.end());
}

namespace {

/// Per-wait worklist entry produced by Pass 1 and consumed by Pass 2.
/// `Wait` points into `Ctx.Decoded`, which is fixed for the duration of
/// `applyGfx1250B0toA0Rules` (no reallocation), so the pointer stays
/// valid across the two passes.
struct WaitBumpEntry {
  const InternalDecodedInst *Wait = nullptr;
  unsigned Bump = 0;
};

/// True if \p Inst ends a basic block: any branch, call, return, set-PC,
/// or kernel terminator. Prefers `MCInstrDesc::isTerminator` (always
/// available) and `MCInstrAnalysis` predicates when the target supplies
/// one. The mnemonic fallback covers AMDGPU control-flow ops that may
/// not be tagged as terminators in MCInstrDesc.
bool isBasicBlockTerminator(const MCInst &Inst, StringRef Mnemonic,
                            const MCInstrInfo &MCII,
                            const MCInstrAnalysis *MIA) {
  const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
  if (Desc.isTerminator())
    return true;
  if (MIA && (MIA->isBranch(Inst) || MIA->isCall(Inst) || MIA->isReturn(Inst)))
    return true;
  return Mnemonic == "s_endpgm" || Mnemonic == "s_endpgm_saved" ||
         Mnemonic == "s_setpc_b64" || Mnemonic == "s_swappc_b64";
}

/// Wrap `encodeWaitWithImm` with the per-site validation, range guards,
/// and Ctx.Text write-back used by the production pass. Returns false on
/// any structural anomaly (no immediate, implausible imm value,
/// out-of-bounds offset, encoder failure), with a diagnostic logged in
/// each case.
[[nodiscard]] bool reencodeWaitWithBump(const InternalDecodedInst &WaitDI,
                                        unsigned Bump, PatchContext &Ctx) {
  if (WaitDI.Inst.getNumOperands() == 0 || !WaitDI.Inst.getOperand(0).isImm()) {
    log() << "hotswap: error: ds2-wait-bump: s_wait_dscnt at 0x"
          << utohexstr(WaitDI.Offset) << " has no immediate operand\n";
    return false;
  }

  int64_t OldImm = WaitDI.Inst.getOperand(0).getImm();
  // Defensive guard: the disassembler should never produce a negative or
  // out-of-range dscnt imm (6-bit field), but a future MC change could
  // wrap silently through the unsigned cast below. Surface it instead.
  if (OldImm < 0 || OldImm > static_cast<int64_t>(MaxWaitDscnt)) {
    log() << "hotswap: error: ds2-wait-bump: implausible imm " << OldImm
          << " at 0x" << utohexstr(WaitDI.Offset) << "\n";
    return false;
  }

  // Overflow-safe bounds: rearranged so neither side of the comparison
  // can wrap, even on adversarial input.
  if (WaitDI.Offset > Ctx.TextSize ||
      WaitDI.Size > Ctx.TextSize - WaitDI.Offset) {
    log() << "hotswap: error: ds2-wait-bump: out-of-bounds wait at 0x"
          << utohexstr(WaitDI.Offset) << "\n";
    return false;
  }

  unsigned NewImm = static_cast<unsigned>(OldImm) + Bump;
  if (NewImm > MaxWaitDscnt)
    NewImm = MaxWaitDscnt;

  SmallVector<uint8_t> Bytes =
      encodeWaitWithImm(WaitDI.Inst, NewImm, Ctx.LS, WaitDI.Size);
  if (Bytes.empty()) {
    log() << "hotswap: error: ds2-wait-bump: encode failed for s_wait_dscnt "
          << "at 0x" << utohexstr(WaitDI.Offset) << "\n";
    return false;
  }

  std::memcpy(Ctx.Text + WaitDI.Offset, Bytes.data(), Bytes.size());
  log() << "hotswap: ds2-wait-bump: s_wait_dscnt at 0x"
        << utohexstr(WaitDI.Offset) << ": " << OldImm << " -> " << NewImm
        << " (+" << Bump << ")\n";
  return true;
}

} // anonymous namespace

uint32_t applyDS2WaitBump(PatchContext &Ctx) {
  // Pass 1: walk decoded stream, accumulate per-wait deficits. Skips
  // drain waits (imm=0) by design (see file header).
  DenseMap<uint64_t, WaitBumpEntry> WaitBumps;
  unsigned PendingDS2 = 0;
  for (const InternalDecodedInst &DI : Ctx.Decoded) {
    if (DI.Mnemonic == "<unknown>")
      continue;
    if (DI.Mnemonic == "s_wait_dscnt") {
      if (PendingDS2 > 0 && DI.Inst.getNumOperands() > 0 &&
          DI.Inst.getOperand(0).isImm() &&
          DI.Inst.getOperand(0).getImm() != 0) {
        WaitBumpEntry &Entry = WaitBumps[DI.Offset];
        Entry.Wait = &DI;
        Entry.Bump += PendingDS2;
      }
      // The wait drains the dscnt counter for both 0 and non-zero imms,
      // so DS2 sites before this point can no longer affect any later
      // wait. Reset the pending counter unconditionally.
      PendingDS2 = 0;
      continue;
    }
    if (isDS2Mnemonic(DI.Mnemonic)) {
      ++PendingDS2;
      continue;
    }
    if (isBasicBlockTerminator(DI.Inst, DI.Mnemonic, *Ctx.LS.MCII,
                               Ctx.LS.MIA.get())) {
      PendingDS2 = 0;
    }
  }

  if (WaitBumps.empty())
    return 0;

  // Pass 2: re-encode each affected wait once with the accumulated bump.
  // Iteration order over the DenseMap is unstable, but each entry is
  // independent, so order doesn't affect the result.
  uint32_t Bumped = 0;
  for (const std::pair<const uint64_t, WaitBumpEntry> &KV : WaitBumps) {
    const WaitBumpEntry &Entry = KV.second;
    if (Entry.Bump == 0 || Entry.Wait == nullptr)
      continue;
    if (reencodeWaitWithBump(*Entry.Wait, Entry.Bump, Ctx))
      ++Bumped;
  }

  log() << "hotswap: ds2-wait-bump: scanned " << Ctx.Decoded.size()
        << " insts, bumped " << Bumped << "/" << WaitBumps.size()
        << " s_wait_dscnt sites\n";
  return Bumped;
}

} // namespace hotswap
} // namespace COMGR

#endif // !defined(_MSC_VER)
