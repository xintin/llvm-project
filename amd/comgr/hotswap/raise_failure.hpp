#ifndef HOTSWAP_TRANSPILER_RAISE_FAILURE_HPP
#define HOTSWAP_TRANSPILER_RAISE_FAILURE_HPP

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace transpiler {

struct DecodedInst;

// Structured reason for a raise failure. Lives in its own header so the
// handler layer (`raise_context.hpp`) can depend on failure values
// without pulling in `RaiseResult` and the rest of the top-level
// `raiser.hpp` interface.
enum class RaiseFailureReason : uint16_t {
  None = 0,
  UnsupportedOpcode,
};

const char *reasonString(RaiseFailureReason r);

struct RaiseFailure {
  RaiseFailureReason reason = RaiseFailureReason::None;
  // Offending instruction mnemonic (e.g. `global_store_dwordx4`).
  std::string mnemonic;
  // Encoding-format category (e.g. `VALU`, `FLAT`, `MUBUF`) — stable
  // bucketing key for the batch / corpus test summaries.
  std::string format;
  // Byte offset inside the disassembled text section, in host order.
  // Zero for failures not tied to a specific instruction.
  uint64_t offset = 0;
  // Optional human-readable context.
  std::string detail;

  bool hasFailed() const { return reason != RaiseFailureReason::None; }

  // Main loop: no handler claimed the CanonicalOp (either no TSFlags match
  // or every matching handler returned `handled=false` without
  // setting a more specific failure). `di` supplies the mnemonic /
  // offset; `format` is the human-readable encoding label.
  static RaiseFailure unsupportedOpcode(const DecodedInst &di,
                                        llvm::StringRef format);
};

} // namespace transpiler

#endif
