#include "../code_object_utils.hpp"

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <string>

TEST(CodeObjectUtils, KernelSymbolOffsetMalformedElfReturnsError) {
  uint8_t garbage[] = {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};

  llvm::Expected<uint64_t> offset =
      transpiler::findKernelSymbolOffset(garbage, "missing_kernel");

  ASSERT_FALSE(static_cast<bool>(offset));
  std::string message = llvm::toString(offset.takeError());
  EXPECT_NE(message.find("findKernelSymbolOffset: Failed to parse ELF"),
            std::string::npos);
}
