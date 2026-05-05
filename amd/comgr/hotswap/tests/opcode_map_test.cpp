#include "../mc_state.hpp"
#include "../opcode_map.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

#include <mutex>

namespace {

void ensureAMDGPURegistered() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
  });
}

} // namespace

TEST(OpcodeMap, Gfx1250I64MinMaxRealOpcodesMapToSemOps) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MAX_I64_e64_gfx1250),
            transpiler::SemOp::V_MAX_I64);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MAX_U64_e64_gfx1250),
            transpiler::SemOp::V_MAX_U64);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MIN_I64_e64_gfx1250),
            transpiler::SemOp::V_MIN_I64);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MIN_U64_e64_gfx1250),
            transpiler::SemOp::V_MIN_U64);
}

TEST(OpcodeMap, Gfx1250AddMinRealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_ADD_MIN_U32_e64_gfx1250),
            transpiler::SemOp::V_ADD_MIN_U32);
}

TEST(OpcodeMap, Gfx1250Min3RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MIN3_U32_e64_gfx12),
            transpiler::SemOp::V_MIN3_U32);
}

TEST(OpcodeMap, Gfx1250Dot4I32IU8RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_DOT4_I32_IU8),
            transpiler::SemOp::V_DOT4_I32_IU8);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_DOT4_I32_IU8_gfx12),
            transpiler::SemOp::V_DOT4_I32_IU8);
}

TEST(OpcodeMap, Gfx1250MadI32I24RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MAD_I32_I24_e64_gfx12),
            transpiler::SemOp::V_MAD_I32_I24);
}
