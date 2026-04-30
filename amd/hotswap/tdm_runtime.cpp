#include "tdm_runtime.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

// SALMON_HAVE_TDM_RUNTIME is set to 1 by `CMakeLists.txt` when hipcc is
// available at configure time and the embedded bitcode blob is generated
// into `tdm_runtime_blob.cpp`. When unset (no hipcc), the linker would
// otherwise fail to resolve `tdm_runtime_bc_data` / `tdm_runtime_bc_size`
// — we provide a one-byte stub here so the build succeeds and
// `tdmRuntimeAvailable()` reports false at runtime.
#ifndef SALMON_HAVE_TDM_RUNTIME
#define SALMON_HAVE_TDM_RUNTIME 0
#endif

extern "C" {
#if SALMON_HAVE_TDM_RUNTIME
extern const unsigned char tdm_runtime_bc_data[];
extern const unsigned long tdm_runtime_bc_size;
#else
const unsigned char tdm_runtime_bc_data[1] = {0};
const unsigned long tdm_runtime_bc_size = 0;
#endif
}

namespace transpiler {

namespace {

llvm::FunctionType *tdmHelperFnTy(llvm::LLVMContext &C) {
  using namespace llvm;
  auto *i32 = Type::getInt32Ty(C);
  auto *v4i = FixedVectorType::get(i32, 4);
  auto *v8i = FixedVectorType::get(i32, 8);
  // Helper takes the four D# groups the walker consumes plus the source
  // wave size. The LLVM intrinsic's trailing <8 x i32> `grp4` is reserved,
  // and `i32 cpol` is a cache-policy immediate with no target-side encoding
  // in the helper path. See `tdm_runtime.hpp` and the matching handler-side
  // elision in `handle_vimage.cpp`.
  return FunctionType::get(Type::getVoidTy(C), {v4i, v8i, v4i, v4i, i32},
                           /*isVarArg=*/false);
}

bool inlineTDMHelperCallSites(llvm::Module &M, llvm::StringRef name) {
  llvm::Function *F = M.getFunction(name);
  if (!F || F->isDeclaration())
    return true;

  llvm::SmallVector<llvm::CallBase *, 8> calls;
  for (llvm::User *U : F->users()) {
    auto *CB = llvm::dyn_cast<llvm::CallBase>(U);
    if (!CB || CB->getCalledOperand()->stripPointerCasts() != F)
      continue;
    calls.push_back(CB);
  }

  for (llvm::CallBase *CB : calls) {
    llvm::InlineFunctionInfo IFI;
    llvm::InlineResult inlined = llvm::InlineFunction(*CB, IFI);
    if (!inlined.isSuccess()) {
      llvm::errs() << "transpiler: failed to inline TDM helper '" << name
                   << "': " << inlined.getFailureReason() << "\n";
      return false;
    }
  }
  return true;
}

} // namespace

llvm::FunctionCallee declareTDMLoad(llvm::Module &M) {
  return M.getOrInsertFunction(kTDMLoadSymbol, tdmHelperFnTy(M.getContext()));
}

llvm::FunctionCallee declareTDMStore(llvm::Module &M) {
  return M.getOrInsertFunction(kTDMStoreSymbol, tdmHelperFnTy(M.getContext()));
}

bool tdmRuntimeAvailable() { return tdm_runtime_bc_size > 0; }

bool moduleUsesTDMRuntime(const llvm::Module &M) {
  return M.getFunction(kTDMLoadSymbol) != nullptr ||
         M.getFunction(kTDMStoreSymbol) != nullptr;
}

bool linkTDMRuntime(llvm::Module &M, llvm::StringRef /*targetISA*/) {
  if (!tdmRuntimeAvailable()) {
    llvm::errs()
        << "transpiler: TDM runtime requested but unavailable "
           "(transpiler was built without hipcc — re-run CMake with "
           "hipcc on PATH to enable cross-target gfx1250->gfx942 TDM "
           "emulation).\n";
    return false;
  }

  auto buf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(tdm_runtime_bc_data),
                      tdm_runtime_bc_size),
      "salmon_tdm_runtime.bc",
      /*RequiresNullTerminator=*/false);

  auto modOrErr =
      llvm::parseBitcodeFile(buf->getMemBufferRef(), M.getContext());
  if (!modOrErr) {
    llvm::errs() << "transpiler: failed to parse embedded TDM bitcode: "
                 << llvm::toString(modOrErr.takeError()) << "\n";
    return false;
  }
  std::unique_ptr<llvm::Module> rt = std::move(*modOrErr);

  // Inherit the kernel module's triple/datalayout. The helper bitcode is
  // architecture-neutral at the IR level (no MFMA, no target-private
  // intrinsics beyond `mbcnt`/`atomicrmw add` on addrspace(3)) and we
  // want `llc -mcpu=<targetISA>` to drive lowering, not whatever default
  // triple hipcc baked into the .bc.
  rt->setTargetTriple(M.getTargetTriple());
  rt->setDataLayout(M.getDataLayout());

  if (llvm::Linker::linkModules(M, std::move(rt))) {
    llvm::errs() << "transpiler: failed to link TDM runtime into module '"
                 << M.getName() << "'\n";
    return false;
  }

  // The helper bitcode is a semantic lowering aid, not a device-call ABI
  // contract.  Leave direct helper calls out of the final kernel IR: otherwise
  // LLVM lowers the calls through a private-segment/dynamic-stack path that the
  // original gfx1250 kernel did not require, and large Triton TensorDescriptor
  // kernels can fault before any useful numerical comparison.
  if (!inlineTDMHelperCallSites(M, kTDMLoadSymbol) ||
      !inlineTDMHelperCallSites(M, kTDMStoreSymbol))
    return false;

  return true;
}

} // namespace transpiler
