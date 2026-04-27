#include "handlers.hpp"

#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;

namespace transpiler {

HandlerResult handleSOPP(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op) {
  (void)op;
  HandlerResult hr;
  SemOp sop = di.semOp;

  if (sop == SemOp::S_ENDPGM) {
    if (ctx.threadLoopLatch)
      ctx.B.CreateBr(ctx.threadLoopLatch);
    else
      ctx.B.CreateRetVoid();
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_BRANCH) {
    int64_t raw = di.getImm(0);
    int64_t brOff = static_cast<int64_t>(
        static_cast<int16_t>(static_cast<uint16_t>(raw & 0xFFFF)));
    ctx.B.CreateBr(ctx.lookupBB(di.offset + 4 + brOff * 4));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CBRANCH_EXECZ || sop == SemOp::S_CBRANCH_EXECNZ) {
    int64_t raw = di.getImm(0);
    int64_t brOff = static_cast<int64_t>(
        static_cast<int16_t>(static_cast<uint16_t>(raw & 0xFFFF)));
    uint64_t target = di.offset + 4 + brOff * 4;
    BasicBlock *targetBB = ctx.lookupBB(target);
    BasicBlock *fallthroughBB = ctx.lookupBB(di.offset + di.size);
    Value *execVal = ctx.regs.loadExec(ctx.B);
    Value *isZero = ctx.B.CreateICmpEQ(
        execVal, Constant::getNullValue(ctx.regs.execTy), "exec_is_zero");
    if (sop == SemOp::S_CBRANCH_EXECZ)
      ctx.B.CreateCondBr(isZero, targetBB, fallthroughBB);
    else
      ctx.B.CreateCondBr(ctx.B.CreateNot(isZero, "exec_nz"), targetBB,
                         fallthroughBB);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CBRANCH_SCC0 || sop == SemOp::S_CBRANCH_SCC1) {
    int64_t raw = di.getImm(0);
    int64_t brOff = static_cast<int64_t>(
        static_cast<int16_t>(static_cast<uint16_t>(raw & 0xFFFF)));
    uint64_t target = di.offset + 4 + brOff * 4;
    BasicBlock *targetBB = ctx.lookupBB(target);
    BasicBlock *fallthroughBB = ctx.lookupBB(di.offset + di.size);
    Value *sccV = ctx.regs.loadSCC(ctx.B);
    if (sop == SemOp::S_CBRANCH_SCC0)
      sccV = ctx.B.CreateNot(sccV, "not_scc");
    ctx.B.CreateCondBr(sccV, targetBB, fallthroughBB);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_CBRANCH_VCCNZ || sop == SemOp::S_CBRANCH_VCCZ) {
    int64_t raw = di.getImm(0);
    int64_t brOff = static_cast<int64_t>(
        static_cast<int16_t>(static_cast<uint16_t>(raw & 0xFFFF)));
    uint64_t target = di.offset + 4 + brOff * 4;
    BasicBlock *targetBB = ctx.lookupBB(target);
    BasicBlock *fallthroughBB = ctx.lookupBB(di.offset + di.size);
    Value *vccV = ctx.regs.loadVCC(ctx.B);
    if (sop == SemOp::S_CBRANCH_VCCZ)
      vccV = ctx.B.CreateNot(vccV, "not_vcc");
    ctx.B.CreateCondBr(vccV, targetBB, fallthroughBB);
    hr.handled = true;
    return hr;
  }
  // Barriers. GFX<12 uses a single `s_barrier`; GFX12+ splits it into a
  // separate signal and wait (both SOPP in this format). We model signal as
  // a no-op (the cross-wave rendezvous happens at the wait) and wait (or the
  // legacy unified barrier) as a full LLVM `amdgcn.s.barrier` call.
  if (sop == SemOp::S_BARRIER || sop == SemOp::S_BARRIER_WAIT) {
    Function *barrierFn =
        Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_s_barrier);
    ctx.B.CreateCall(barrierFn, {});
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_BARRIER_SIGNAL) {
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::S_SET_VGPR_MSB) {
    // Only the low 8 bits of the immediate carry runtime meaning; the high
    // 8 bits record the previous mode for compiler bookkeeping (see
    // AMDGPULowerVGPREncoding::setMode in LLVM).  The hardware ignores them.
    int64_t imm = di.getImm(0);
    ctx.vgprMSBs = static_cast<uint8_t>(imm & 0xFF);
    hr.handled = true;
    return hr;
  }

  // gfx1250 async-memory wait counters. Explicit arm (rather than
  // falling through to the generic SOPP no-op catch-all below) so
  // this handler's surface documents the async/tensor cross-target
  // correctness argument alongside the other SOPP branches.
  //
  // Both counters track work in dedicated gfx1250 hardware units
  // (`ASYNCcnt`, `TENSORcnt`; programming_manual.pdf §4.9.9 and
  // §6 respectively) that do not exist on gfx942.  The source DMAs
  // they gate are emulated as synchronous `load`+`store` chains on
  // the cross-target arm (see `handle_flat.cpp`'s
  // `GLOBAL_LOAD_ASYNC_TO_LDS_B*` handler and `handle_vimage.cpp`'s
  // refusal → future emulation for TENSOR ops), so by the time the
  // wait site is reached the underlying memory transfer has
  // already completed at the IR level.  IR dataflow from the
  // emulated `store` through subsequent LDS reads carries the
  // happens-before the native counter was enforcing; the backend
  // re-inserts the target-appropriate `s_waitcnt lgkmcnt(0)` on
  // gfx942 from that ordering constraint.
  //
  // On the same-target arm (gfx1250 → gfx1250) this branch is
  // still a no-op — like `S_WAITCNT` / `S_WAIT_LOADCNT` / the
  // other wait counters handled by the generic catch-all, the
  // async intrinsic's `IntrInaccessibleMemOrArgMemOnly`
  // annotation prevents reorder across the wait site and the
  // backend re-emits the native `s_wait_asynccnt` /
  // `s_wait_tensorcnt` from the IR's load/store dataflow.  The
  // two arms are therefore emission-identical; the explicit
  // branch is a documentation / audit anchor, not a dispatch
  // split.  See the `S_WAIT_ASYNCCNT` / `S_WAIT_TENSORCNT` SemOp
  // doc block in `semop.hpp` and `sync-translation.md §5.2.b`
  // for the full trade-off and the dependency on IR-level
  // ordering.
  if (sop == SemOp::S_WAIT_ASYNCCNT || sop == SemOp::S_WAIT_TENSORCNT) {
    hr.handled = true;
    return hr;
  }

  // All other SOPP instructions (waitcnt, nop, etc.) are no-ops
  hr.handled = true;
  return hr;
}

} // namespace transpiler
