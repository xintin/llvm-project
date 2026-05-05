#include "../source_hidden_args.hpp"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace llvm;
using transpiler::KernelArgMeta;
using transpiler::SourceHiddenArgContext;
using transpiler::SourceHiddenArgValue;
using transpiler::emitSourceHiddenInteger;

namespace {

KernelArgMeta makeArg(const char *Name, int Offset, int Size,
                      const char *ValueKind) {
  KernelArgMeta Arg;
  Arg.name = Name;
  Arg.offset = Offset;
  Arg.size = Size;
  Arg.valueKind = ValueKind;
  return Arg;
}

struct HiddenArgModule {
  LLVMContext C;
  Module M{"source-hidden-args-test", C};
  IRBuilder<> B{C};
  Function *F = nullptr;

  HiddenArgModule() {
    auto *FTy = FunctionType::get(Type::getVoidTy(C), {}, false);
    F = Function::Create(FTy, GlobalValue::ExternalLinkage, "kernel", M);
    F->setCallingConv(CallingConv::AMDGPU_KERNEL);
    BasicBlock *BB = BasicBlock::Create(C, "entry", F);
    B.SetInsertPoint(BB);
  }

  std::string str() {
    B.CreateRetVoid();
    std::string Out;
    raw_string_ostream OS(Out);
    M.print(OS, nullptr);
    return OS.str();
  }

  SourceHiddenArgContext context(ArrayRef<KernelArgMeta> Args) {
    return SourceHiddenArgContext{C, M, B, Type::getInt8Ty(C),
                                  Type::getInt32Ty(C), Type::getInt64Ty(C),
                                  Args};
  }
};

} // namespace

TEST(SourceHiddenArgs, GroupSizeXUsesAqlDispatchPacketOffset) {
  std::vector<KernelArgMeta> Args = {
      makeArg("group_x", 44, 2, "hidden_group_size_x"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx = HM.context(Args);

  SourceHiddenArgValue Value =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/44, /*ByteWidth=*/2,
                              /*IsSigned=*/false);

  ASSERT_TRUE(Value.Matched);
  ASSERT_NE(Value.Value, nullptr);
  EXPECT_TRUE(Value.FailureDetail.empty());

  std::string IR = HM.str();
  EXPECT_NE(IR.find("@llvm.amdgcn.dispatch.ptr"), std::string::npos);
  EXPECT_NE(IR.find("getelementptr inbounds i8, ptr addrspace(4) %dispatch_ptr, i32 4"),
            std::string::npos);
  EXPECT_EQ(IR.find("i32 24"), std::string::npos)
      << "SI::KernelInputOffsets::LOCAL_SIZE_X is not the AQL packet offset";
}

TEST(SourceHiddenArgs, BlockCountXUsesGridDividedByWorkgroupSize) {
  std::vector<KernelArgMeta> Args = {
      makeArg("blocks_x", 32, 4, "hidden_block_count_x"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx = HM.context(Args);

  SourceHiddenArgValue Value =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/32, /*ByteWidth=*/4,
                              /*IsSigned=*/false);

  ASSERT_TRUE(Value.Matched);
  ASSERT_NE(Value.Value, nullptr);

  std::string IR = HM.str();
  EXPECT_NE(IR.find("i32 4"), std::string::npos);
  EXPECT_NE(IR.find("i32 12"), std::string::npos);
  EXPECT_NE(IR.find("udiv i32"), std::string::npos);
}

TEST(SourceHiddenArgs, UnsupportedHiddenArgFailsLoudly) {
  std::vector<KernelArgMeta> Args = {
      makeArg("hostcall", 64, 8, "hidden_hostcall_buffer"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx = HM.context(Args);

  SourceHiddenArgValue Value =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/64, /*ByteWidth=*/4,
                              /*IsSigned=*/false);

  EXPECT_TRUE(Value.Matched);
  EXPECT_EQ(Value.Value, nullptr);
  EXPECT_NE(Value.FailureDetail.find("unsupported source hidden argument kind"),
            std::string::npos);
}

TEST(SourceHiddenArgs, NonHiddenOffsetDoesNotMatch) {
  std::vector<KernelArgMeta> Args = {
      makeArg("n", 24, 4, "by_value"),
  };
  HiddenArgModule HM;
  SourceHiddenArgContext Ctx = HM.context(Args);

  SourceHiddenArgValue Value =
      emitSourceHiddenInteger(Ctx, /*ByteOffset=*/24, /*ByteWidth=*/4,
                              /*IsSigned=*/false);

  EXPECT_FALSE(Value.Matched);
  EXPECT_EQ(Value.Value, nullptr);
  EXPECT_TRUE(Value.FailureDetail.empty());
}
