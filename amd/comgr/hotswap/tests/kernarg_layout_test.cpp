#include "../kernarg_layout.hpp"

#include "gtest/gtest.h"

#include <vector>

using transpiler::KernelArgMeta;
using transpiler::SourceHiddenArgKind;
using transpiler::classifySourceHiddenArgByte;

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

TEST(KernargLayout, ClassifiesHiddenBlockCountsByByteContainment) {
  std::vector<KernelArgMeta> args = {
      makeArg("out", 0, 8, "global_buffer"),
      makeArg("grid_x", 48, 4, "hidden_block_count_x"),
      makeArg("grid_y", 52, 4, "hidden_block_count_y"),
      makeArg("grid_z", 56, 4, "hidden_block_count_z"),
  };

  auto x0 = classifySourceHiddenArgByte(args, 48);
  auto x3 = classifySourceHiddenArgByte(args, 51);
  auto y0 = classifySourceHiddenArgByte(args, 52);
  auto z0 = classifySourceHiddenArgByte(args, 56);

  EXPECT_EQ(x0.kind, SourceHiddenArgKind::HiddenBlockCountX);
  EXPECT_EQ(x0.byteIndexInArg(), 0u);
  EXPECT_EQ(x3.kind, SourceHiddenArgKind::HiddenBlockCountX);
  EXPECT_EQ(x3.byteIndexInArg(), 3u);
  EXPECT_EQ(y0.kind, SourceHiddenArgKind::HiddenBlockCountY);
  EXPECT_EQ(z0.kind, SourceHiddenArgKind::HiddenBlockCountZ);
}

TEST(KernargLayout, ClassifiesGroupSizeRemainderAndGridDims) {
  std::vector<KernelArgMeta> args = {
      makeArg("group_x", 44, 2, "hidden_group_size_x"),
      makeArg("rem_x", 50, 2, "hidden_remainder_x"),
      makeArg("grid_dims", 96, 2, "hidden_grid_dims"),
  };

  EXPECT_EQ(classifySourceHiddenArgByte(args, 44).kind,
            SourceHiddenArgKind::HiddenGroupSizeX);
  EXPECT_EQ(classifySourceHiddenArgByte(args, 50).kind,
            SourceHiddenArgKind::HiddenRemainderX);
  EXPECT_EQ(classifySourceHiddenArgByte(args, 96).kind,
            SourceHiddenArgKind::HiddenGridDims);
}

TEST(KernargLayout, ClassifiesUnsupportedHiddenKinds) {
  std::vector<KernelArgMeta> args = {
      makeArg("hostcall", 64, 8, "hidden_hostcall_buffer"),
  };

  EXPECT_EQ(classifySourceHiddenArgByte(args, 64).kind,
            SourceHiddenArgKind::UnsupportedHidden);
}

TEST(KernargLayout, NonHiddenAndMissingOffsetsAreNotHidden) {
  std::vector<KernelArgMeta> args = {
      makeArg("n", 24, 4, "by_value"),
  };

  EXPECT_FALSE(classifySourceHiddenArgByte(args, 24).matched());
  EXPECT_FALSE(classifySourceHiddenArgByte(args, 28).matched());
}
