#include "raise_failure.hpp"

#include "decoded_inst.hpp"

namespace transpiler {

const char *reasonString(RaiseFailureReason r) {
  switch (r) {
  case RaiseFailureReason::None:
    return "None";
  case RaiseFailureReason::UnsupportedOpcode:
    return "UnsupportedOpcode";
  }
  return "UnknownRaiseFailureReason";
}

RaiseFailure RaiseFailure::unsupportedOpcode(const DecodedInst &di,
                                             llvm::StringRef format) {
  RaiseFailure f;
  f.reason = RaiseFailureReason::UnsupportedOpcode;
  f.mnemonic = di.mnemonic;
  f.format = format.str();
  f.offset = di.offset;
  return f;
}

} // namespace transpiler
