#include "handlers.hpp"
#include "mubuf_addr.hpp"

#include "semop.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstring>
#include <map>
#include <optional>
#include <tuple>

using namespace llvm;

namespace transpiler {
HandlerResult handleMUBUF(RaiseContext &ctx, const DecodedInst &di,
                        OpResolver &op) {
  HandlerResult hr;
  StringRef mn(di.mnemonic);
  SemOp sop = di.semOp;

  // d16Half encodes D16 partial-write target: 0 = full-write (regular
  // load), 1 = low half (`_D16`), 2 = high half (`_D16_HI`). The
  // partial-write loads zero/sign-extend the loaded sub-dword to i16
  // and merge into the named half of the destination VGPR, preserving
  // the other half. See BUFInstructions.td:1155-1177 (predicate
  // `D16PreservesUnusedBits`).
  auto mubufClassify = [](SemOp s)
      -> std::tuple<bool, bool, int, int, bool, bool, int> {
    // returns {isLoad, isStore, dwords, loadBits, isSubDword, isSigned, d16Half}
    switch (s) {
    case SemOp::BUFFER_LOAD_DWORD:    return {true, false, 1, 32, false, false, 0};
    case SemOp::BUFFER_LOAD_DWORDX2:  return {true, false, 2, 64, false, false, 0};
    case SemOp::BUFFER_LOAD_DWORDX3:  return {true, false, 3, 96, false, false, 0};
    case SemOp::BUFFER_LOAD_DWORDX4:  return {true, false, 4, 128, false, false, 0};
    case SemOp::BUFFER_LOAD_UBYTE:    return {true, false, 1, 8, true, false, 0};
    case SemOp::BUFFER_LOAD_SBYTE:    return {true, false, 1, 8, true, true, 0};
    case SemOp::BUFFER_LOAD_USHORT:   return {true, false, 1, 16, true, false, 0};
    case SemOp::BUFFER_LOAD_SSHORT:   return {true, false, 1, 16, true, true, 0};
    case SemOp::BUFFER_LOAD_SHORT_D16:     return {true, false, 1, 16, true, false, 1};
    case SemOp::BUFFER_LOAD_SHORT_D16_HI:  return {true, false, 1, 16, true, false, 2};
    case SemOp::BUFFER_LOAD_UBYTE_D16:     return {true, false, 1, 8,  true, false, 1};
    case SemOp::BUFFER_LOAD_UBYTE_D16_HI:  return {true, false, 1, 8,  true, false, 2};
    case SemOp::BUFFER_LOAD_SBYTE_D16:     return {true, false, 1, 8,  true, true,  1};
    case SemOp::BUFFER_LOAD_SBYTE_D16_HI:  return {true, false, 1, 8,  true, true,  2};
    case SemOp::BUFFER_STORE_DWORD:   return {false, true, 1, 32, false, false, 0};
    case SemOp::BUFFER_STORE_DWORDX2: return {false, true, 2, 64, false, false, 0};
    case SemOp::BUFFER_STORE_DWORDX3: return {false, true, 3, 96, false, false, 0};
    case SemOp::BUFFER_STORE_DWORDX4: return {false, true, 4, 128, false, false, 0};
    case SemOp::BUFFER_STORE_BYTE:    return {false, true, 1, 8, true, false, 0};
    case SemOp::BUFFER_STORE_SHORT:   return {false, true, 1, 16, true, false, 0};
    default: return {false, false, 0, 0, false, false, 0};
    }
  };
  auto [isLoad, isStore, dwords, loadBits, isSubDword, isBufSigned, d16Half] =
      mubufClassify(sop);
  if (isLoad || isStore) {
    // Use gfx942 buffer intrinsics directly. The hardware handles OOB:
    // loads return 0, stores are silently dropped. This avoids the
    // flat-memory lowering that requires conditional branches (which
    // break under LLVM -O1+ SIMT optimizations).
    MubufAddr mbuf = decodeMubufAddr(ctx, di, op, isStore, "MUBUF");
    // For loads, vdata is the dst; for stores it's the first VGPR src
    // (captured into mbuf.stData by the decoder).
    ParsedReg vdata = isStore ? mbuf.stData : op.dst(0);
    Value *srd = mbuf.srd;
    Value *voffset = mbuf.voffset;
    Value *soffset = mbuf.soffset;
    Value *auxFlags = mbuf.auxFlags;
    auto rawPtrBufferLoad = [&](Type *loadTy) -> Value * {
      Function *bufLd = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_raw_ptr_buffer_load, {loadTy});
      return ctx.B.CreateCall(
          bufLd, {mbuf.rawPtrRsrc, voffset, soffset, auxFlags},
          "buf_ld_rawptr");
    };

    if (isLoad) {
      if (isSubDword) {
        // Load the sub-dword datum and zero/sign-extend to i32. For
        // plain ushort/sbyte/etc. (`d16Half == 0`) we then write the
        // whole VGPR; for D16 partial-write loads we merge with the
        // prior dst (see comment block above mubufClassify).
        Type *memTy = (loadBits == 8) ? Type::getInt8Ty(ctx.C)
                                      : Type::getInt16Ty(ctx.C);
        Value *loaded = nullptr;
        if (ctx.targetIsa.waveSize > ctx.isa.waveSize) {
          loaded = rawPtrBufferLoad(memTy);
        } else {
          Function *bufLd = Intrinsic::getOrInsertDeclaration(
              &ctx.M, Intrinsic::amdgcn_raw_buffer_load, {memTy});
          loaded = ctx.B.CreateCall(bufLd,
              {srd, voffset, soffset, auxFlags}, "buf_ld");
        }
        if (d16Half == 0) {
          Value *ext = isBufSigned ? ctx.B.CreateSExt(loaded, ctx.i32Ty)
                                   : ctx.B.CreateZExt(loaded, ctx.i32Ty);
          ctx.writeReg32(vdata, ext);
        } else {
          // Partial-write: extend to i16 (sign for `_SBYTE_D16*`,
          // zero for `_UBYTE_D16*` / `_SHORT_D16*`), zext to i32 so
          // the high half of the i32 is exactly zero before merging.
          Value *ext16 = loaded;
          if (loadBits == 8) {
            ext16 = isBufSigned
                        ? ctx.B.CreateSExt(loaded, Type::getInt16Ty(ctx.C))
                        : ctx.B.CreateZExt(loaded, Type::getInt16Ty(ctx.C));
          }
          Value *ext32 = ctx.B.CreateZExt(ext16, ctx.i32Ty);
          Value *prior = ctx.regs.readReg32(ctx.B, vdata);
          Value *merged;
          if (d16Half == 1) {
            // _D16: place datum in lo 16, preserve hi 16 of prior.
            Value *priorHi =
                ctx.B.CreateAnd(prior, ConstantInt::get(ctx.i32Ty, 0xFFFF0000));
            merged = ctx.B.CreateOr(priorHi, ext32, "d16_lo_merge");
          } else {
            // _D16_HI: place datum in hi 16, preserve lo 16 of prior.
            Value *priorLo =
                ctx.B.CreateAnd(prior, ConstantInt::get(ctx.i32Ty, 0x0000FFFF));
            Value *shifted =
                ctx.B.CreateShl(ext32, ConstantInt::get(ctx.i32Ty, 16));
            merged = ctx.B.CreateOr(priorLo, shifted, "d16_hi_merge");
          }
          ctx.writeReg32(vdata, merged);
        }
      } else if (dwords == 1) {
        Value *loaded = nullptr;
        if (ctx.targetIsa.waveSize > ctx.isa.waveSize) {
          loaded = rawPtrBufferLoad(ctx.i32Ty);
        } else {
          Function *bufLd = Intrinsic::getOrInsertDeclaration(
              &ctx.M,
              Intrinsic::amdgcn_raw_buffer_load,
              {ctx.i32Ty});
          loaded = ctx.B.CreateCall(bufLd,
              {srd, voffset, soffset, auxFlags}, "buf_ld");
        }
        ctx.writeReg32(vdata, loaded);
      } else {
        auto *vecTy = FixedVectorType::get(ctx.i32Ty, dwords);
        Value *loaded = nullptr;
        if (ctx.targetIsa.waveSize > ctx.isa.waveSize) {
          loaded = rawPtrBufferLoad(vecTy);
        } else {
          Function *bufLd = Intrinsic::getOrInsertDeclaration(
              &ctx.M,
              Intrinsic::amdgcn_raw_buffer_load,
              {vecTy});
          loaded = ctx.B.CreateCall(bufLd,
              {srd, voffset, soffset, auxFlags}, "buf_ld");
        }
        ctx.writeRegVec(vdata, loaded);
      }
      hr.handled = true;
    return hr;
    }
    if (isStore) {
      // Use the gfx942 buffer-store intrinsic directly, exactly
      // mirroring the load path above. The hardware's BUFFER unit
      // handles OOB silently (the write is dropped when the per-lane
      // offset is >= num_records), so no software OOB sink is needed.
      //
      // The earlier implementation lowered every BUFFER_STORE_* to a
      // generic `store` against an `addrspacecast(alloca i32, addrspace(5))`
      // OOB sink (selected via `select i1 oob, sink, real`). That was
      // wrong on three independent axes:
      //
      //   1. Size mismatch. The sink alloca was always `i32` (4 B), but
      //      `BUFFER_STORE_DWORDX4` writes 16 B. For OOB lanes the
      //      flat_store_dwordx4 walked 12 B past the sink and into
      //      either the next per-thread scratch slot or unmapped
      //      scratch — root cause of the gfx1250 Triton vector-add
      //      SIGSEGV (R1).
      //
      //   2. Forced scratch enablement. Adding any `addrspace(5)`
      //      alloca that survives PromoteMemToReg makes the AMDGPU
      //      backend emit `.amdhsa_enable_private_segment 1` plus
      //      `.amdhsa_private_segment_fixed_size > 0`. Salmon's KD
      //      doesn't request `flat_scratch_init` (we model only the
      //      source ABI's user-SGPR set), so on entry FLAT_SCRATCH is
      //      undefined; any flat instruction touching the scratch
      //      aperture (including the OOB sink path) is a fault waiting
      //      to happen. Native gfx942 Triton emits `buffer_store_*`
      //      directly and reports `private_segment_fixed_size 0` /
      //      `enable_private_segment 0` — confirming hardware OOB
      //      handling is the right primitive.
      //
      //   3. Asymmetric with the load path. Loads already go through
      //      `amdgcn.raw.buffer.load` and rely on hardware OOB clamp.
      //      Routing stores through a software select+sink was an
      //      avoidable divergence whose only justification ("avoids
      //      flat-memory lowering with conditional branches breaking
      //      under -O1+ SIMT optimisations" — comment block above)
      //      doesn't apply when we use the buffer intrinsic itself.
      //
      // EXEC gating: like the load, the call is emitted unconditionally;
      // the hardware EXEC mask suppresses inactive-lane writes natively.
      // We still wrap in `emitUnderExec` so that within an
      // already-narrowed IR-level lane window (e.g. an inner if-then
      // branch handled earlier in the kernel) the store body is
      // dominated by the handler's lane-active diamond, keeping
      // EXEC and IR-level dominance in sync — same pattern other
      // store handlers (DS, scratch) use.
      Type *storeTy;
      Value *val;
      if (isSubDword) {
        storeTy = Type::getIntNTy(ctx.C, loadBits);
        val = ctx.B.CreateTrunc(ctx.regs.readReg32(ctx.B, vdata), storeTy);
      } else if (dwords == 1) {
        storeTy = ctx.i32Ty;
        val = ctx.regs.readReg32(ctx.B, vdata);
      } else {
        auto *vecTy = FixedVectorType::get(ctx.i32Ty, dwords);
        storeTy = vecTy;
        val = ctx.regs.readRegVec(ctx.B, vdata, vecTy);
      }
      Function *bufSt = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_raw_buffer_store, {storeTy});
      ctx.emitUnderExec([&] {
        ctx.B.CreateCall(bufSt, {val, srd, voffset, soffset, auxFlags});
      });
      hr.handled = true;
    return hr;
    }
  }

  // ---- Buffer load to LDS (buffer_load_dword lds, ...) ----
  // Data goes directly to LDS at M0 + vaddr, not to a VGPR.
  // Model as: tmp = raw_buffer_load; ds_write(LDS[M0], tmp)
  if (sop == SemOp::BUFFER_LOAD_DWORD_LDS ||
      sop == SemOp::BUFFER_LOAD_DWORDX2_LDS ||
      sop == SemOp::BUFFER_LOAD_DWORDX4_LDS) {
    int dwords = (sop == SemOp::BUFFER_LOAD_DWORDX4_LDS) ? 4
               : (sop == SemOp::BUFFER_LOAD_DWORDX2_LDS) ? 2 : 1;

    MubufAddr mbuf = decodeMubufAddr(ctx, di, op, /*isStore=*/false,
                                      "MUBUF_LDS");

    // Load from buffer into a temp value.
    Type *ldTy = (dwords == 1)
                     ? ctx.i32Ty
                     : FixedVectorType::get(ctx.i32Ty, dwords);
    Function *bufLd = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_raw_buffer_load, {ldTy});
    Value *loaded = ctx.B.CreateCall(
        bufLd, {mbuf.srd, mbuf.voffset, mbuf.soffset, mbuf.auxFlags},
        "lds_buf_ld");

    // Store to LDS at address from M0.
    ParsedReg m0Reg; m0Reg.kind = ParsedReg::M0; m0Reg.baseIdx = 0;
    Value *ldsAddr = ctx.regs.readReg32(ctx.B, m0Reg);
    auto *ldsPtrTy = PointerType::get(ctx.C, 3);
    Value *ldsPtr = ctx.B.CreateIntToPtr(ldsAddr, ldsPtrTy);
    ctx.emitUnderExec([&] { ctx.B.CreateStore(loaded, ldsPtr); });

    hr.handled = true;
    return hr;
  }

  // ---- Buffer atomics ----
  //
  // RTN / non-RTN operand shape note.  MUBUF buffer atomics put the
  // vdata register at operand 0 in BOTH the RTN (glc=1 / tied-def)
  // and non-RTN (glc=0 / pure-source) forms.  The difference is
  // whether operand 0 is also tied to the destination (RTN) or
  // only a source (non-RTN).  `decodeMubufAddr(..., isStore=true)`
  // treats the first VGPR source as vdata for both forms, and the
  // RTN-only write-back below is gated by `di.numDefs > 0`
  // (consistent with the assert just below this comment), which
  // correctly SKIPS for non-RTN.  The
  // two per-form invariants are pinned by
  // `lit_tests/buffer_atomic_swap_b32/` (RTN) +
  // `lit_tests/buffer_atomic_swap_b32_nortn/` (non-RTN) and the
  // cmpswap twins.
  if (sop >= SemOp::BUFFER_ATOMIC_ADD && sop <= SemOp::BUFFER_ATOMIC_PK_ADD_F16) {
    assert(((di.tsFlags & SIInstrFlags::IsAtomicRet) != 0) == (di.numDefs > 0) &&
           "buffer atomic: IsAtomicRet disagrees with numDefs");
    MubufAddr mbuf = decodeMubufAddr(ctx, di, op, /*isStore=*/true,
                                     "buffer_atomic");

    // `BUFFER_ATOMIC_CMPSWAP` is the one buffer atomic whose vdata is
    // a register PAIR carrying `{cmp, new}` rather than a single data
    // word.  Split it out before the single-word raw-buffer atomic
    // dispatch below. The MUBUF data register is the first VGPR source,
    // and the second word is read with a synthetic `baseIdx + 1`.
    if (sop == SemOp::BUFFER_ATOMIC_CMPSWAP) {
      ParsedReg dataPair = mbuf.stData;
      Value *cmpVal = ctx.regs.readReg32(ctx.B, dataPair);
      ParsedReg newReg = dataPair;
      newReg.baseIdx += 1;
      newReg.width = 1;
      Value *newVal = ctx.regs.readReg32(ctx.B, newReg);
      Function *casFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_raw_buffer_atomic_cmpswap, {ctx.i32Ty});
      ctx.emitUnderExec([&] {
        // Raw-buffer atomics preserve descriptor-relative addressing and
        // hardware OOB behavior. The intrinsic takes {new, cmp}, matching
        // LLVM's AMDGPU intrinsic contract for buffer cmpswap.
        Value *oldVal = ctx.B.CreateCall(
            casFn, {newVal, cmpVal, mbuf.srd, mbuf.voffset, mbuf.soffset,
                    mbuf.auxFlags},
            "buf_atomic_cmpswap");
        if (di.numDefs > 0)
          ctx.regs.writeReg32(ctx.B, op.dst(), oldVal);
      });
      hr.handled = true;
      return hr;
    }

    Value *data = ctx.regs.readReg32(ctx.B, mbuf.stData);

    Intrinsic::ID atomicIntrinsic = Intrinsic::not_intrinsic;
    Type *atomicTy = ctx.i32Ty;
    bool isFP = false;
    switch (sop) {
    case SemOp::BUFFER_ATOMIC_ADD:
      atomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_add;
      break;
    case SemOp::BUFFER_ATOMIC_SUB:
      atomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_sub;
      break;
    case SemOp::BUFFER_ATOMIC_AND:
      atomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_and;
      break;
    case SemOp::BUFFER_ATOMIC_OR:
      atomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_or;
      break;
    case SemOp::BUFFER_ATOMIC_XOR:
      atomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_xor;
      break;
    // `buffer_atomic_swap` — pure exchange.  Added in the same
    // commit as the CMPSWAP branch above to close the handler gap
    // that used to refuse both atomics at the `default:` arm.  The
    // RTN-form write-back (see the shared emit below) is what makes
    // SWAP semantically meaningful; a dropped result would reduce
    // `buffer_atomic_swap` to a plain store and lose the caller's
    // "old value" read, quietly miscompiling any CAS-loop or
    // lock-free shape that relies on it.
    case SemOp::BUFFER_ATOMIC_SWAP:
      atomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_swap;
      break;
    case SemOp::BUFFER_ATOMIC_ADD_F32:
      atomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_fadd;
      atomicTy = ctx.f32Ty;
      isFP = true;
      break;
    case SemOp::BUFFER_ATOMIC_PK_ADD_BF16:
      atomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_fadd;
      atomicTy = FixedVectorType::get(Type::getBFloatTy(ctx.C), 2);
      isFP = true;
      break;
    case SemOp::BUFFER_ATOMIC_PK_ADD_F16:
      atomicIntrinsic = Intrinsic::amdgcn_raw_buffer_atomic_fadd;
      atomicTy = FixedVectorType::get(Type::getHalfTy(ctx.C), 2);
      isFP = true;
      break;
    default:
      llvm::errs() << "transpiler: Unsupported buffer atomic: " << mn << "\n";
      hr.failure = RaiseFailure::unsupportedShape(di, "MUBUF",
                                                   "unsupported buffer atomic");
      return hr;
    }
    if (isFP) data = ctx.B.CreateBitCast(data, atomicTy);
    Function *atomicFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, atomicIntrinsic, {atomicTy});
    ctx.emitUnderExec([&] {
      Value *oldVal = ctx.B.CreateCall(
          atomicFn,
          {data, mbuf.srd, mbuf.voffset, mbuf.soffset, mbuf.auxFlags},
          "buf_atomic");
      // RTN-form write-back. The raw-buffer intrinsic returns the old
      // memory value just like the target ISA RTN form; when the source
      // is non-RTN, leaving the result unused lets the backend select
      // the no-return encoding.
      if (di.numDefs > 0) {
        Value *retVal = oldVal;
        if (isFP) retVal = ctx.B.CreateBitCast(retVal, ctx.i32Ty);
        ctx.regs.writeReg32(ctx.B, op.dst(), retVal);
      }
    });
    hr.handled = true;
    return hr;
  }
  return hr;
}

} // namespace transpiler
