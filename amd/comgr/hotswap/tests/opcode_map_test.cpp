#include "../mc_state.hpp"
#include "../opcode_map.hpp"

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
