#include "../kernarg_layout.hpp"

#include "gtest/gtest.h"

#include <vector>

using transpiler::KernelArgMeta;
using transpiler::PreloadedHiddenKernargDword;
using transpiler::classifyPreloadedHiddenKernargDword;

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
