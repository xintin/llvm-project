#include "../kernarg_layout.hpp"

#include "gtest/gtest.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>
#include <vector>

using transpiler::KernelArgMeta;
using transpiler::KernargLayout;
using transpiler::PreloadedHiddenKernargDword;
using transpiler::classifyPreloadedHiddenKernargDword;
using transpiler::extractKernargBytesAsI32;
using transpiler::extractKernargDword;

namespace {
KernelArgMeta makeArg(const char *name, int offset, int size,
                      const char *valueKind) {
  KernelArgMeta arg;
  arg.name = name;
  arg.offset = offset;
  arg.size = size;
  arg.valueKind = valueKind;
  return arg;
}

struct KernargDwordFixture {
  llvm::LLVMContext C;
  llvm::Module M{"kernarg_layout_test", C};
  llvm::IRBuilder<> B{C};
  llvm::Function *F = nullptr;

  KernargDwordFixture(std::initializer_list<unsigned> argWidths) {
    std::vector<llvm::Type *> argTypes;
    for (unsigned width : argWidths)
      argTypes.push_back(llvm::Type::getIntNTy(C, width));
    auto *fnTy = llvm::FunctionType::get(llvm::Type::getVoidTy(C), argTypes,
                                         false);
    F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                               "kernel", M);
    auto *bb = llvm::BasicBlock::Create(C, "entry", F);
    B.SetInsertPoint(bb);
  }
};
} // namespace

TEST(KernargLayout, ClassifiesPreloadedHiddenBlockCounts) {
  std::vector<KernelArgMeta> args = {
      makeArg("out", 0, 8, "global_buffer"),
      makeArg("grid_x", 48, 4, "hidden_block_count_x"),
      makeArg("grid_y", 52, 4, "hidden_block_count_y"),
      makeArg("grid_z", 56, 4, "hidden_block_count_z"),
  };

  EXPECT_EQ(classifyPreloadedHiddenKernargDword(args, 48),
            PreloadedHiddenKernargDword::HiddenBlockCountX);
  EXPECT_EQ(classifyPreloadedHiddenKernargDword(args, 52),
            PreloadedHiddenKernargDword::HiddenBlockCountY);
  EXPECT_EQ(classifyPreloadedHiddenKernargDword(args, 56),
            PreloadedHiddenKernargDword::HiddenBlockCountZ);
}

TEST(KernargLayout, ClassifiesUnsupportedPreloadedHiddenKinds) {
  std::vector<KernelArgMeta> args = {
      makeArg("hostcall", 64, 8, "hidden_hostcall_buffer"),
  };

  EXPECT_EQ(classifyPreloadedHiddenKernargDword(args, 64),
            PreloadedHiddenKernargDword::UnsupportedHidden);
}

TEST(KernargLayout, NonHiddenAndMissingOffsetsAreNotHidden) {
  std::vector<KernelArgMeta> args = {
      makeArg("n", 24, 4, "by_value"),
  };

  EXPECT_EQ(classifyPreloadedHiddenKernargDword(args, 24),
            PreloadedHiddenKernargDword::NotHidden);
  EXPECT_EQ(classifyPreloadedHiddenKernargDword(args, 28),
            PreloadedHiddenKernargDword::NotHidden);
}

TEST(KernargLayout, ExtractsDwordFromI32SlotWithoutRepacking) {
  KernargDwordFixture ir({32});
  KernargLayout layout;
  layout.params.push_back({0, 4, 0, false});
  layout.kernargSegmentSize = 4;

  std::string why;
  llvm::Value *value = extractKernargDword(layout, ir.B, ir.F, 0, &why);

  ASSERT_NE(value, nullptr) << why;
  EXPECT_EQ(value, ir.F->getArg(0));
}

TEST(KernargLayout, ExtractsDwordContainingNarrowScalarAndPadding) {
  KernargDwordFixture ir({8});
  KernargLayout layout;
  layout.params.push_back({0, 1, 0, false});
  layout.kernargSegmentSize = 4;

  std::string why;
  llvm::Value *value = extractKernargDword(layout, ir.B, ir.F, 0, &why);

  ASSERT_NE(value, nullptr) << why;
  EXPECT_TRUE(value->getType()->isIntegerTy(32));
  EXPECT_NE(llvm::dyn_cast<llvm::Argument>(value), ir.F->getArg(0));
}

TEST(KernargLayout, RefusesNarrowScalarCrossingDwordBoundary) {
  KernargDwordFixture ir({16});
  KernargLayout layout;
  layout.params.push_back({3, 2, 0, false});
  layout.kernargSegmentSize = 8;

  std::string why;
  llvm::Value *value = extractKernargDword(layout, ir.B, ir.F, 0, &why);

  EXPECT_EQ(value, nullptr);
  EXPECT_NE(why.find("cross-dword scalar extraction"), std::string::npos);
}

TEST(KernargLayout, TreatsVectorLoadDwordAtSegmentEndAsUndefPadding) {
  KernargDwordFixture ir({});
  KernargLayout layout;
  layout.kernargSegmentSize = 36;

  std::string why;
  llvm::Value *value = extractKernargDword(layout, ir.B, ir.F, 36, &why);

  ASSERT_NE(value, nullptr) << why;
  EXPECT_TRUE(llvm::isa<llvm::UndefValue>(value));
}

TEST(KernargLayout, ExtractsUnalignedBytesAcrossDwordBoundary) {
  KernargDwordFixture ir({32, 32});
  KernargLayout layout;
  layout.params.push_back({0, 4, 0, false});
  layout.params.push_back({4, 4, 1, false});
  layout.kernargSegmentSize = 8;

  std::string why;
  llvm::Value *value =
      extractKernargBytesAsI32(layout, ir.B, ir.F, 2, 4, &why);

  ASSERT_NE(value, nullptr) << why;
  EXPECT_TRUE(value->getType()->isIntegerTy(32));

  std::string irText;
  llvm::raw_string_ostream os(irText);
  ir.M.print(os, nullptr);
  os.flush();
  EXPECT_NE(irText.find("%0"), std::string::npos)
      << "low bytes should come from first dword";
  EXPECT_NE(irText.find("%1"), std::string::npos)
      << "high bytes should come from second dword";
}

TEST(KernargLayout, RefusesUnsupportedByteLoadWidth) {
  KernargDwordFixture ir({32});
  KernargLayout layout;
  layout.params.push_back({0, 4, 0, false});
  layout.kernargSegmentSize = 4;

  std::string why;
  llvm::Value *value =
      extractKernargBytesAsI32(layout, ir.B, ir.F, 0, 3, &why);

  EXPECT_EQ(value, nullptr);
  EXPECT_NE(why.find("unsupported kernarg byte load width"), std::string::npos);
}
