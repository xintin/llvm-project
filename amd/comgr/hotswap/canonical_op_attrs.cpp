#include "canonical_op_attrs.hpp"

#include "opcode_map.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::EXEC, EXEC_LO, EXEC_HI
#include "Utils/AMDGPUBaseInfo.h"            // AMDGPU::mc2PseudoReg

#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <iterator>

using namespace llvm;

namespace transpiler {

namespace {

// Materialise the attribute table once at first-use. Aggregates every
// per-handler registration declared in `canonical_op_attrs.hpp`. The
// `Meyers singleton`-style static local sidesteps any cross-TU static
// initialisation ordering fuss — the table is built lazily, not at
// dynamic-init time.
class AttrTable {
public:
  AttrTable() {
    auto ingest = [&](ArrayRef<CanonicalOpAttrSpec> specs) {
      for (const CanonicalOpAttrSpec &s : specs) {
        auto idx = static_cast<std::size_t>(s.op);
        assert(idx < std::size(entries_) &&
               "CanonicalOp out of range; bump entries_ to CanonicalOp_COUNT");
        entries_[idx] = s.attrs;
      }
    };
    ingest(getHandlerSOP1Attrs());
    ingest(getHandlerSOP2Attrs());
    ingest(getHandlerVALU_VcmpAttrs());
  }

  const CanonicalOpAttrs &operator[](CanonicalOp op) const {
    auto idx = static_cast<std::size_t>(op);
    assert(idx < std::size(entries_) && "CanonicalOp out of range");
    return entries_[idx];
  }

private:
  CanonicalOpAttrs entries_[static_cast<std::size_t>(CanonicalOp::CanonicalOp_COUNT)] = {};
};

const AttrTable &theTable() {
  static const AttrTable t;
  return t;
}

static bool descImplicitlyDefinesEXEC(const MCInstrDesc &desc) {
  for (MCPhysReg r : desc.implicit_defs()) {
    MCRegister reg = AMDGPU::mc2PseudoReg(r);
    if (reg == AMDGPU::EXEC || reg == AMDGPU::EXEC_LO ||
        reg == AMDGPU::EXEC_HI)
      return true;
  }
  return false;
}

} // namespace

const CanonicalOpAttrs &getCanonicalOpAttrs(CanonicalOp op) { return theTable()[op]; }

void verifyExecAttrCoverage(const MCInstrInfo &MCII, const OpcodeMap &opcMap) {
  for (unsigned mc = 0, end = MCII.getNumOpcodes(); mc < end; ++mc) {
    const MCInstrDesc &desc = MCII.get(mc);
    if (!descImplicitlyDefinesEXEC(desc))
      continue;
    CanonicalOp sop = opcMap.lookup(mc);
    if (sop == CanonicalOp::Unknown)
      continue; // covered by the generic unsupported-opcode path
    if (getCanonicalOpAttrs(sop).routesExecThroughStoreExec)
      continue;
    report_fatal_error(Twine("transpiler: MC opcode ") + MCII.getName(mc) +
                       " (#" + Twine(mc) +
                       ") declares EXEC as an implicit def but its CanonicalOp "
                       "(" + canonicalOpName(sop) +
                       ") is not marked routesExecThroughStoreExec. Audit "
                       "the handler's EXEC write path against SPE before "
                       "declaring the CanonicalOp in that handler's "
                       "get*Attrs() registration.");
  }
}

} // namespace transpiler
