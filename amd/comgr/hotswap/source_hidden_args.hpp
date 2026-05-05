#ifndef HOTSWAP_TRANSPILER_SOURCE_HIDDEN_ARGS_HPP
#define HOTSWAP_TRANSPILER_SOURCE_HIDDEN_ARGS_HPP

#include "code_object_utils.hpp"
#include "kernarg_layout.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"

#include <string>

namespace llvm {
class Function;
class IRBuilderBase;
class LLVMContext;
class Module;
class Type;
class Value;
} // namespace llvm

namespace transpiler {

struct SourceHiddenArgContext {
  llvm::LLVMContext &C;
  llvm::Module &M;
  llvm::IRBuilderBase &B;
  llvm::Type *i8Ty;
  llvm::Type *i32Ty;
  llvm::Type *i64Ty;
  llvm::ArrayRef<KernelArgMeta> Args;
};

struct SourceHiddenArgValue {
  // True when ByteOffset maps to a source metadata hidden_* field.
  bool Matched = false;
  // Non-null when a matched hidden field was lowered successfully.
  llvm::Value *Value = nullptr;
  // Non-empty when Matched is true and Value is null.
  std::string FailureDetail;
};

SourceHiddenArgValue emitSourceHiddenDword(SourceHiddenArgContext &Ctx,
                                           int ByteOffset);
SourceHiddenArgValue emitSourceHiddenInteger(SourceHiddenArgContext &Ctx,
                                             int ByteOffset,
                                             unsigned ByteWidth,
                                             bool IsSigned);

} // namespace transpiler

#endif
