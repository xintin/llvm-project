#include "handlers.hpp"
#include "pipeline.hpp" // isStrictMode()

#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

#define DEBUG_TYPE "transpiler"

using namespace llvm;

namespace transpiler {

namespace {

// Look up the SGPR index that holds the *low* dword of the source-ISA
// KernargSegmentPtr at kernel entry, via `ctx.userSgprLayout`.
//
// Used by the dword-granular S_LOAD_B* block and the narrow-SMEM
// (S_LOAD_U8/I8/U16/I16) block to gate the implicit-args reroute on
// "is this load's sbase the kernarg pair?" The previous implementation
// hardcoded `baseIdx == 0`, which broke as soon as a kernel enabled
// PrivateSegmentBuffer (4 dwords), DispatchPtr (2 dwords), or QueuePtr
// (2 dwords) ahead of the kernarg pointer in the canonical
// enable_sgpr_* order — the kernarg pair then slides up to s[2:3],
// s[6:7], s[8:9], etc.
//
// The layout object is the single source of truth for the source
// ISA's user-SGPR ABI. It is populated by
// `UserSgprLayout::fromKernelMeta` (which itself aborts loudly if the
// kernel descriptor is missing, so we never fall back to a guessed
// layout), and wired into `RaiseContext` before handler dispatch in
// raiser.cpp. A null pointer here therefore means the raiser failed
// to wire the layout into the context — a wiring bug, not a runtime
// condition — and we surface it with `report_fatal_error`.
//
// Returns -1 if KernargSegmentPtr is disabled in the KD (no corpus
// kernel today, but the caller must still guard against the `-1`
// match to avoid a false-positive "is kernarg" on negative sbase
// indices).
int getKernargPtrSgpr(RaiseContext &ctx) {
  if (ctx.userSgprLayout == nullptr)
    llvm::report_fatal_error(
        "transpiler: handle_smem: RaiseContext::userSgprLayout is null. "
        "The raiser must populate this before dispatching to handlers; "
        "missing wiring is a bug.");
  return ctx.userSgprLayout->kernargSegmentPtrSgpr;
}

} // namespace

HandlerResult handleSMEM(RaiseContext &ctx, const DecodedInst &di,
                       OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;

  if (sop == SemOp::S_LOAD_B32 || sop == SemOp::S_LOAD_B64 ||
      sop == SemOp::S_LOAD_B96 || sop == SemOp::S_LOAD_B128 ||
      sop == SemOp::S_LOAD_B256 || sop == SemOp::S_LOAD_B512) {
    int loadDwords = 1;
    switch (sop) {
    case SemOp::S_LOAD_B32:
      loadDwords = 1;
      break;
    case SemOp::S_LOAD_B64:
      loadDwords = 2;
      break;
    case SemOp::S_LOAD_B96:
      loadDwords = 3;
      break;
    case SemOp::S_LOAD_B128:
      loadDwords = 4;
      break;
    case SemOp::S_LOAD_B256:
      loadDwords = 8;
      break;
    case SemOp::S_LOAD_B512:
      loadDwords = 16;
      break;
    default:
      break;
    }
    int loadBytes = loadDwords * 4;

    ParsedReg dest = op.dst();
    ParsedReg base = op.srcReg(0);

    unsigned offIdx = op.srcIdx(1);
    bool immOffset = di.isImm(offIdx);
    int64_t byteOffset = immOffset ? op.srcImm(1) : 0;
    int kernargPtrSgpr = getKernargPtrSgpr(ctx);
    bool baseIsKernargPair =
        (base.kind == ParsedReg::SGPR && kernargPtrSgpr >= 0 &&
         base.baseIdx == kernargPtrSgpr);

    // Implicit-args reroute. AMDGPU separates the explicit kernarg
    // segment from the implicit-arg block: the latter is reachable via
    // `amdgcn_implicitarg_ptr`, not via offsets past the end of the
    // kernarg segment. A source kernel that issues
    // `s_load_b* sN, kernarg_pair, off` with `off >= implicitArgsBase`
    // is reading hidden args through the source-ABI flat layout; the
    // lifted kernel must materialise those bytes via the implicit-arg
    // pointer with the offset rebased to `off - implicitArgsBase`.
    //
    // Strict-mode refusal: in `HSA_SALMON_STRICT=1` the cross-arch
    // implicit-arg layout is not yet proven equivalent for every
    // `(source ISA, target ISA)` pair we lift between, so the
    // pipeline refuses to silently substitute a target-ABI implicit
    // arg for a source-ABI one. In permissive mode we trust the
    // ROCm convention that the layouts match (both gfx9-12 follow
    // the same `hidden_*` block).
    //
    // Gating: `baseIsKernargPair` (literal SGPR-index match against
    // the source-ABI kernarg pair) + `immOffset` + a positive
    // `implicitArgsBase`. We deliberately do NOT track whether the
    // pair has been mutated since entry: corpus shapes that overwrite
    // the pair (Triton/SGLang `s[0:1] = preloaded_ptr + wg_offset`,
    // Tensile UniversalArgs `+16` shift) only issue follow-up loads at
    // small offsets that fall well below `implicitArgsBase`, so the
    // gate is precise enough in practice.
    if (baseIsKernargPair && immOffset &&
        ctx.kernargs.implicitArgsBase > 0 &&
        byteOffset >= ctx.kernargs.implicitArgsBase) {
      if (isStrictMode()) {
        hr.failure = RaiseFailure::strictUnsafeLowering(
            di, "implicitarg.ptr",
            "cross-arch implicitarg.ptr lowering is unresolved: source "
            "implicit-arg offsets are being applied to the target runtime "
            "hidden-arg block");
        return hr;
      }
      Function *fnImplicitArgPtr = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_implicitarg_ptr);
      Value *implPtr =
          ctx.B.CreateCall(fnImplicitArgPtr, {}, "implicitarg_ptr");
      int64_t implOffset = byteOffset - ctx.kernargs.implicitArgsBase;
      Value *gep =
          (implOffset == 0)
              ? implPtr
              : ctx.B.CreateInBoundsGEP(ctx.i8Ty, implPtr,
                                         ctx.B.getInt64(implOffset),
                                         "impl_gep");
      for (int d = 0; d < loadDwords; d++) {
        Value *ep = (d == 0) ? gep
                             : ctx.B.CreateInBoundsGEP(
                                   ctx.i8Ty, gep, ctx.B.getInt64(d * 4));
        ctx.regs.storeSGPR32(ctx.B, dest.baseIdx + d,
                             ctx.B.CreateLoad(ctx.i32Ty, ep, "impl_load"));
      }
      hr.handled = true;
      return hr;
    }

    // Generic GEP+load against `addrspace(1)`. The AMDGPU backend
    // re-derives uniformity / addrspace-narrowing during lowering: a
    // load whose pointer is provably from `amdgcn_kernarg_segment_ptr`
    // is selected as `s_load_*` against the kernarg segment regardless
    // of the IR-level addrspace cast; for runtime-mutated bases (Triton/
    // SGLang `s[0:1] = preloaded_ptr + wg_offset`, Tensile HBMArgs
    // `s_load_b64 s[0:1], s[0:1], 0x10`, etc.) the backend keeps the
    // VMEM lowering. The lift no longer hand-picks the addrspace —
    // tracking pointer provenance at lift time was redundant with the
    // backend's own analysis.
    {
      Value *baseAddr = ctx.regs.loadSGPR64(ctx.B, base.baseIdx);
      Value *ptr = ctx.B.CreateIntToPtr(baseAddr, ctx.ptrGlobalTy);
      if (immOffset) {
        if (byteOffset != 0)
          ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, ctx.B.getInt64(byteOffset));
      } else {
        // gfx12+ SMEM: when the `scale_offset` (CPol::SCAL) bit is
        // set the SGPR offset is an element index, not a byte
        // offset — hardware multiplies it by the load's data-type
        // size before adding to sbase. Mirror that here (the
        // FLAT/GLOBAL counterparts in flat_addr.cpp do the same
        // against `elemBytes`). The "element size" for the scalar
        // dword family is the full load width: 4B for B32, 8B for
        // B64, 16B for B128, etc. — i.e. `loadBytes`. Ignoring the
        // scale produced a silent off-by-N* miscompile on
        // `mask[blockIdx.x]`-style uses of a uniform SGPR index.
        Value *regOff = ctx.B.CreateZExt(op.src(1), ctx.i64Ty, "smem_roff");
        if (di.hasScaleOffset)
          regOff = ctx.B.CreateMul(regOff,
                                   ConstantInt::get(ctx.i64Ty, loadBytes),
                                   "smem_roff_scaled");
        ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, regOff);
      }
      for (int d = 0; d < loadDwords; d++) {
        Value *ep = (d == 0) ? ptr
                             : ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr,
                                                       ctx.B.getInt64(d * 4));
        ctx.regs.storeSGPR32(ctx.B, dest.baseIdx + d,
                             ctx.B.CreateLoad(ctx.i32Ty, ep, "smem_load"));
      }
    }
    hr.handled = true;
    return hr;
  }

  // gfx12+ scalar narrow loads: s_load_{u8,i8,u16,i16}. These fetch 1 or 2
  // bytes from a uniform address materialised in an SGPR-pair base and
  // zero/sign-extend the result into a 32-bit SGPR. MC operand shape
  // matches the dword-granular s_load_* family (sbase + imm-or-sgpr
  // offset), so operand decoding mirrors the S_LOAD_B* block above.
  //
  // Design notes:
  //  * IR shape: `load iN, ptr addrspace(1) %p, align N` + `zext`/`sext`
  //    to i32 → `storeSGPR32`. No AMDGPU-specific intrinsic exists for
  //    narrow scalar loads; the backend's ISel matches the uniform-address
  //    pattern directly.
  //  * Same-target gfx1250 → gfx1250: the backend re-codegens to the
  //    original `s_load_u16` / `s_load_u8` / etc. (identity-preserving).
  //  * Cross-target gfx1250 → gfx942: the backend has no native narrow
  //    SMEM load, so it lowers to VMEM (`global_load_ushort` / ubyte).
  //    The lifted kernel stays correct — the value appears on every lane
  //    with the same content, matching the SMEM broadcast semantics —
  //    but the register class shifts SGPR→VGPR and the memory path
  //    shifts scalar-cache→vector-cache.
  //  * Alignment: explicit `Align(1)` for byte, `Align(2)` for halfword.
  //
  // Test back-reference: lit_tests/s_load_u16/ exercises the halfword
  // same-target happy path. The byte (u8/i8) and signed (i8/i16)
  // variants share this handler body.
  if (sop == SemOp::S_LOAD_U8 || sop == SemOp::S_LOAD_I8 ||
      sop == SemOp::S_LOAD_U16 || sop == SemOp::S_LOAD_I16) {
    bool isHalfWord =
        (sop == SemOp::S_LOAD_U16 || sop == SemOp::S_LOAD_I16);
    bool isSigned =
        (sop == SemOp::S_LOAD_I8 || sop == SemOp::S_LOAD_I16);
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Type *narrowTy = isHalfWord ? i16Ty : ctx.i8Ty;
    Align narrowAlign = Align(isHalfWord ? 2 : 1);
    const char *narrowLoadName = isHalfWord ? "smem_load_h" : "smem_load_b";
    const char *extName = isSigned ? "smem_load_sext" : "smem_load_zext";

    ParsedReg dest = op.dst();
    ParsedReg base = op.srcReg(0);

    Value *baseAddr = ctx.regs.loadSGPR64(ctx.B, base.baseIdx);
    Value *ptr = ctx.B.CreateIntToPtr(baseAddr, ctx.ptrGlobalTy);
    unsigned offIdx = op.srcIdx(1);
    if (di.isImm(offIdx)) {
      int64_t off = op.srcImm(1);
      if (off != 0)
        ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, ctx.B.getInt64(off));
    } else {
      // Narrow SMEM element size for `scale_offset`: 1B for byte,
      // 2B for halfword. Same SCAL-scales-the-SGPR-offset rule as
      // the dword family above.
      int narrowBytes = isHalfWord ? 2 : 1;
      Value *regOff = ctx.B.CreateZExt(op.src(1), ctx.i64Ty, "smem_nroff");
      if (di.hasScaleOffset && narrowBytes != 1)
        regOff = ctx.B.CreateMul(regOff,
                                 ConstantInt::get(ctx.i64Ty, narrowBytes),
                                 "smem_nroff_scaled");
      ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, regOff);
    }

    Value *narrow = ctx.B.CreateAlignedLoad(narrowTy, ptr, narrowAlign,
                                             narrowLoadName);
    Value *ext = isSigned
                     ? ctx.B.CreateSExt(narrow, ctx.i32Ty, extName)
                     : ctx.B.CreateZExt(narrow, ctx.i32Ty, extName);
    ctx.regs.storeSGPR32(ctx.B, dest.baseIdx, ext);
    hr.handled = true;
    return hr;
  }

  // s_store_* (scalar store through SGPR base + imm/sgpr offset).
  // MC operand layout: (sdata, sbase, soffset/imm, cpol).
  if (sop == SemOp::S_STORE_B32 || sop == SemOp::S_STORE_B64 ||
      sop == SemOp::S_STORE_B128) {
    int storeDwords = (sop == SemOp::S_STORE_B32)  ? 1
                      : (sop == SemOp::S_STORE_B64) ? 2
                                                    : 4;
    ParsedReg data = op.srcReg(0);
    ParsedReg base = op.srcReg(1);
    if (data.kind != ParsedReg::SGPR || base.kind != ParsedReg::SGPR) {
      llvm::errs() << "transpiler: " << di.mnemonic
                   << ": S_STORE expects SGPR data and base\n";
      hr.failure = RaiseFailure::unsupportedShape(
          di, "SMEM", "S_STORE expects SGPR data and base");
      return hr;
    }
    Value *baseAddr = ctx.regs.loadSGPR64(ctx.B, base.baseIdx);
    Value *ptr = ctx.B.CreateIntToPtr(baseAddr, ctx.ptrGlobalTy);
    int storeBytes = storeDwords * 4;
    if (op.nSrcs() >= 3) {
      unsigned offIdx = op.srcIdx(2);
      if (di.isImm(offIdx)) {
        int64_t off = op.srcImm(2);
        if (off != 0)
          ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, ctx.B.getInt64(off));
      } else if (di.isReg(offIdx)) {
        // Same `scale_offset` scaling as the S_LOAD path — the
        // SCAL bit multiplies the SGPR offset by the store's
        // data-type size (4/8/16B for B32/B64/B128).
        Value *regOff = ctx.B.CreateZExt(op.src(2), ctx.i64Ty, "smem_st_roff");
        if (di.hasScaleOffset)
          regOff = ctx.B.CreateMul(regOff,
                                   ConstantInt::get(ctx.i64Ty, storeBytes),
                                   "smem_st_roff_scaled");
        ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, regOff);
      }
    }
    for (int d = 0; d < storeDwords; d++) {
      Value *ep = (d == 0) ? ptr
                           : ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr,
                                                     ctx.B.getInt64(d * 4));
      Value *v = ctx.regs.loadSGPR32(ctx.B, data.baseIdx + d);
      ctx.B.CreateStore(v, ep);
    }
    hr.handled = true;
    return hr;
  }

  // SMEM dword atomics (returned-old-value / GLC=1 / `_RTN` form only):
  // operate on memory through an SGPR-pair base pointer with an IMM /
  // SGPR / SGPR_IMM offset, and publish the pre-modification value into
  // the sdst (== tied sdst_in) SGPR slot.
  //
  // Dispatch is keyed on the SemOp; each arm picks the `atomicrmw`
  // BinOp that matches the hardware's scalar-cache semantics exactly:
  //   S_ATOMIC_SWAP -> Xchg      (pure exchange)
  //   S_ATOMIC_DEC  -> UDecWrap  (AMDGPU HW wrap-at-zero decrement:
  //                               new = (old == 0 || old > src) ? src
  //                                                             : old - 1;
  //                               NOT `atomicrmw sub`, which would
  //                               silently underflow past zero)
  //
  // The returned-old-value write-back is the hot path: AITER split-k
  // reductions (bf16gemm_*_splitk_clean) key the "last workgroup runs
  // the epilogue" barrier on `old == 1`, so dropping the dst store
  // would silently break the branch that follows.
  //
  // Non-matching SemOps fall through the `default:` arm without the
  // `hr.handled` flip, so the raiser's Phase-5 unsupportedOpcode path
  // reports the missing lowering.
  AtomicRMWInst::BinOp rmwOp;
  switch (sop) {
  case SemOp::S_ATOMIC_SWAP:
    rmwOp = AtomicRMWInst::Xchg;
    break;
  case SemOp::S_ATOMIC_DEC:
    rmwOp = AtomicRMWInst::UDecWrap;
    break;
  default:
    return hr;
  }

  // Two disassembler shapes for the same SemOp, distinguished by the
  // instruction's GLC bit (encoding-level) and `di.numDefs`
  // (decode-level).  The TableGen source of truth is
  // `llvm/lib/Target/AMDGPU/SMInstructions.td`, class
  // `SM_Pseudo_Atomic<..., bit isRet, ...>`:
  //
  //     !if(isRet, (outs dataClass:$sdst), (outs))
  //     (ins dataClass:$sdata, baseClass:$sbase, <offset>, CPolTy:$cpol)
  //     let Constraints = !if(isRet, "$sdst = $sdata", "")
  //
  // i.e. both forms always carry `$sdata`, `$sbase`, the offset, and
  // `$cpol` in the `ins` list; only RTN adds `$sdst` in the `outs`
  // list and ties it to `$sdata` (which the MC layer then elides
  // from the operand-print list, leaving the decoded MCInst as):
  //
  //   RTN     (GLC=1, numDefs=1, isRet=1):
  //           `(sdst, sbase, offset, cpol)` — TableGen's tied
  //           `"$sdst = $sdata"` constraint elides the tied input
  //           from the operand list; the atomic's input value (xchg
  //           data for swap, wrap-threshold for dec) comes from the
  //           *pre-instruction* value of the dst SGPR; the post-
  //           instruction value is the returned `old`.  AITER's
  //           split-k barrier keys its "am I the last workgroup?"
  //           branch on `old == 1`, so this is the hot path for
  //           `bf16gemm_*_splitk_clean.co` lowering.
  //
  //   non-RTN (GLC=0, numDefs=0, isRet=0):
  //           `(sdata, sbase, offset, cpol)` — no dst at all; `sdata`
  //           stays as an explicit source operand.  The atomic runs
  //           and the returned `old` is dropped on the floor.  hipcc's
  //           inline-asm lowering of `s_atomic_dec %[rmw], %[ptr],
  //           %[off]` (no `_rtn` suffix in the mnemonic string)
  //           produces this shape even when the `"+s"(rmw)`
  //           constraint suggests a tied in/out, so the non-RTN arm
  //           has to exist for any HIP fixture that spells the
  //           instruction via inline asm.  See `lit_tests/s_atomic_dec/`
  //           for the canonical fixture (split-k barrier reproducer).
  //
  // Common to both arms: the atomic binop (`rmwOp` above), the base
  // pointer in `sbase` (SGPR pair), a `soffset` that is either an
  // inline imm or an SGPR element index scaled by the dword width
  // when the `scale_offset` (CPol::SCAL) bit is set, and the
  // explicit `Align(4)` / `AtomicOrdering::Monotonic`.  `cpol` is
  // not consumed by the lift (the GLC bit it carries is already
  // reflected in `di.numDefs`; the non-GLC CPol bits — DLC, SCOPE,
  // SCAL — are either already threaded through `di.hasScaleOffset`
  // or are cache-hint-only and thus lift-invariant).
  //
  // No SMEM atomic today has more than one def.  The assertion below
  // pins that invariant so a hypothetical future 2-def form (none
  // exists in any ISA the raiser targets) fails loudly rather than
  // silently taking the RTN arm and mis-decoding `op.dst()` /
  // `op.dst(1)`.
  assert(di.numDefs <= 1 &&
         "SMEM atomic with >1 defs is not a shape the lift recognises");
  ParsedReg base;
  unsigned offIdx;
  Value *data = nullptr;
  ParsedReg dataDst;  // Set only on the RTN arm.
  if (di.numDefs == 0) {
    // Non-RTN: (sdata, sbase, offset).  Read sdata as the atomic's
    // input value; the returned `old` is intentionally discarded
    // below (the HW already wrote the new value to memory, which is
    // the whole point of the non-RTN form's existence).
    base = op.srcReg(1);
    offIdx = op.srcIdx(2);
    data = ctx.regs.readReg32(ctx.B, op.srcReg(0));
  } else {
    // RTN: (sdst_tied, sbase, offset).  Read the pre-instruction
    // value of sdst as the atomic's input (this is the tied-input
    // slot TableGen's `"$sdst = $sdata"` constraint names), then
    // write the returned `old` back to the same SGPR after the
    // atomicrmw.
    dataDst = op.dst();
    base = op.srcReg(0);
    offIdx = op.srcIdx(1);
    data = ctx.regs.readReg32(ctx.B, dataDst);
  }

  Value *baseAddr = ctx.regs.loadSGPR64(ctx.B, base.baseIdx);
  Value *ptr = ctx.B.CreateIntToPtr(baseAddr, ctx.ptrGlobalTy);

  // Positional source index of the offset operand in OpResolver's
  // operand view: slot 1 for the RTN shape (sbase, offset), slot 2
  // for the non-RTN shape (sdata, sbase, offset).  Keeps the
  // imm/SGPR-offset arm routed through the generic `op.src()` reader
  // (which handles imm, SGPR, and any other source kind uniformly)
  // so the RTN path's IR is bit-identical to the pre-split handler.
  unsigned offSrcPos = (di.numDefs == 0 ? 2u : 1u);
  if (di.isImm(offIdx)) {
    int64_t off = di.getImm(offIdx);
    if (off != 0)
      ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, ctx.B.getInt64(off));
  } else {
    // Dword-width atomic, so the `scale_offset` (CPol::SCAL) bit
    // scales the SGPR element-index by 4 to recover the byte offset
    // — same rule as the other SMEM paths.
    Value *regOff = ctx.B.CreateZExt(op.src(offSrcPos), ctx.i64Ty,
                                      "smem_at_roff");
    if (di.hasScaleOffset)
      regOff = ctx.B.CreateMul(regOff, ConstantInt::get(ctx.i64Ty, 4),
                               "smem_at_roff_scaled");
    ptr = ctx.B.CreateInBoundsGEP(ctx.i8Ty, ptr, regOff);
  }

  // Pin `Align(4)` explicitly instead of letting the IRBuilder infer
  // ABI alignment. Matches the explicit-Align convention the narrow
  // SMEM load block above sets (see its "Alignment:" design bullet);
  // relying on ABI inference happens to produce `align 4` for i32 on
  // AS(1) today, but inferred alignment is fragile against future LLVM
  // default-alignment changes and masks pointer-alignment bugs from
  // callers.
  Value *old = ctx.B.CreateAtomicRMW(rmwOp, ptr, data, Align(4),
                                     AtomicOrdering::Monotonic);
  // RTN arm only: publish the pre-modification value to the tied
  // sdst SGPR.  The non-RTN arm has no write-back — the HW has
  // already committed the new value to memory, which is all that
  // form guarantees, and `old` is left as a dead SSA value that
  // LLVM's DCE will remove in the usual way.
  if (di.numDefs != 0)
    ctx.regs.storeSGPR32(ctx.B, dataDst.baseIdx, old);
  hr.handled = true;
  return hr;
}

} // namespace transpiler
