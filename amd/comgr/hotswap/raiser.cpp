//===- raiser.cpp - Hotswap MC -> LLVM IR raiser ---------------------------===//
//
// Disassembles a kernel's ELF text section into a typed `DecodedInst` stream
// and builds an `llvm::Module` with a kernel function whose body is `ret void`.
// See `raiser.hpp` for the full raise pipeline (ELF ingestion -> decode ->
// per-format handlers -> post-raise analyses).
//
//===----------------------------------------------------------------------===//

#include "raiser.hpp"

#include "amdgpu_formats.hpp"
#include "decode.hpp"
#include "decoded_inst.hpp"
#include "mc_state.hpp"
#include "opcode_map.hpp"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

namespace transpiler {

RaiseResult raiseToIR(llvm::ArrayRef<uint8_t> textBytes,
                      llvm::StringRef sourceISA,
                      llvm::StringRef kernelName,
                      const KernelMeta & /*meta*/,
                      uint64_t kernelOffset,
                      llvm::StringRef compilationTargetISA) {
  using namespace llvm;
  RaiseResult result;

  // === Phase 1: MC stack + opcode canonicalisation ===
  MCState mc;
  initMCState(mc, sourceISA);
  OpcodeMap opcMap;
  opcMap.build(*mc.instrInfo);

  // === Phase 2: Disassemble kernel text section ===
  DecodeResult decoded = decodeKernel(mc, opcMap, textBytes, kernelOffset);
  result.totalCount = static_cast<int>(decoded.insts.size());

  // === Phase 3: Build LLVM IR module + function ===
  result.ctx = std::make_unique<LLVMContext>();
  LLVMContext &C = *result.ctx;
  result.module = std::make_unique<Module>("transpiler_module", C);
  Module &M = *result.module;
  M.setTargetTriple(Triple(kAMDGPUTriple));

  TargetOptions opts;
  std::unique_ptr<TargetMachine> tm(mc.target->createTargetMachine(
      Triple(kAMDGPUTriple),
      compilationTargetISA.empty() ? sourceISA : compilationTargetISA,
      "", opts, Reloc::PIC_));
  if (tm)
    M.setDataLayout(tm->createDataLayout());

  auto *funcTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F =
      Function::Create(funcTy, GlobalValue::ExternalLinkage, kernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);
  BasicBlock *entry = BasicBlock::Create(C, "entry", F);
  IRBuilder<> B(entry);
  B.CreateRetVoid();

  // === Phase 4: Bail on the first instruction ===
  // No per-format handlers are wired up yet, so any non-empty kernel body
  // surfaces as `RaiseFailure::unsupportedOpcode`. An empty kernel still
  // raises successfully.
  if (!decoded.insts.empty()) {
    const DecodedInst &di = decoded.insts.front();
    result.failure =
        RaiseFailure::unsupportedOpcode(di, formatName(di.tsFlags,
                                                        di.inst.getOpcode()));
    result.success = false;
    return result;
  }

  result.success = true;
  return result;
}

} // namespace transpiler
