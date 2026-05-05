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

// Empty map: every opcode should resolve to `CanonicalOp::Unknown` until
// handler patches start adding entries.
TEST(OpcodeMap, UnknownLookupBeforeBuild) {
  transpiler::OpcodeMap map;
  EXPECT_EQ(map.lookup(0), transpiler::CanonicalOp::Unknown);
  EXPECT_EQ(map.lookup(12345), transpiler::CanonicalOp::Unknown);
}

TEST(OpcodeMap, BuildOnGfx942IsBenign) {
  ensureAMDGPURegistered();
  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx942"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  // No handler patches have landed yet, so every MC opcode resolves to
  // `Unknown` — the raiser bails on the first decoded instruction.
  EXPECT_EQ(map.lookup(0), transpiler::CanonicalOp::Unknown);
}

TEST(OpcodeMap, Gfx1250AddMinRealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_ADD_MIN_U32_e64_gfx1250),
            transpiler::CanonicalOp::V_ADD_MIN_U32);
}

TEST(OpcodeMap, Gfx1250Min3RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MIN3_U32_e64_gfx12),
            transpiler::CanonicalOp::V_MIN3_U32);
}

TEST(OpcodeMap, Gfx1250Dot4I32IU8RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_DOT4_I32_IU8),
            transpiler::CanonicalOp::V_DOT4_I32_IU8);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_DOT4_I32_IU8_gfx12),
            transpiler::CanonicalOp::V_DOT4_I32_IU8);
}

TEST(OpcodeMap, Gfx1250PkFmaF16RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_PK_FMA_F16_gfx12),
            transpiler::CanonicalOp::V_PK_FMA_F16);
}

TEST(OpcodeMap, Gfx1250MadI32I24RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MAD_I32_I24_e64_gfx12),
            transpiler::CanonicalOp::V_MAD_I32_I24);
}
