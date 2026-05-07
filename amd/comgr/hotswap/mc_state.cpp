#include "mc_state.hpp"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;

namespace transpiler {

const char kAMDGPUTriple[] = "amdgcn-amd-amdhsa";

std::unique_ptr<MCSubtargetInfo>
buildSubtargetInfo(const Target &target, StringRef isa) {
  Triple triple(kAMDGPUTriple);
  std::unique_ptr<MCSubtargetInfo> sti(
      target.createMCSubtargetInfo(triple, isa, ""));
  if (!sti)
    report_fatal_error(Twine("transpiler: failed to create MCSubtargetInfo "
                             "for ISA '") +
                       isa + "'");
  return sti;
}

bool initMCState(MCState &state, StringRef targetISA) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();

  Triple triple(kAMDGPUTriple);
  std::string error;
  state.target = TargetRegistry::lookupTarget(triple, error);
  if (!state.target)
    report_fatal_error(Twine("transpiler: Target lookup for '") +
                       kAMDGPUTriple + "' failed: " + error);

  state.instrInfo.reset(state.target->createMCInstrInfo());
  state.regInfo.reset(state.target->createMCRegInfo(triple));
  state.subtargetInfo = buildSubtargetInfo(*state.target, targetISA);
  state.asmInfo.reset(state.target->createMCAsmInfo(
      *state.regInfo, triple, MCTargetOptions()));
  state.ctx = std::make_unique<MCContext>(triple, *state.asmInfo,
                                         *state.regInfo,
                                         *state.subtargetInfo);
  // Defensive consistency with the legacy hotswap path
  // (see `hotswap.cpp` / `hotswap/transpiler.cpp`'s companion
  // `initInlineSourceManager` calls): the MCContext ctor defaults
  // `SourceMgr *Mgr = nullptr`, so any MC-layer diagnostic that
  // reaches `MCContext::reportCommon` or `MCContext::diagnose`
  // with a valid SMLoc and no SrcMgr trips the
  // `llvm_unreachable("Either SourceMgr should be available")`
  // abort at `llvm/lib/MC/MCContext.cpp:1093` / `:1120`.
  //
  // Hotswap's IR-raise pipeline doesn't currently exercise the MC
  // assembler (codegen runs through `llc`/`lld` on lifted IR, not
  // through this MCContext), so the abort doesn't fire on hotswap
  // today.  But the disassembler here can emit diagnostics on
  // malformed instruction bytes, and any future reuse of this
  // MCContext for an MC emission path (e.g. an assembly-based
  // post-rewrite pass or a new cross-widening lowering that
  // goes through MC) would hit the same abort.  Attaching an
  // inline SourceMgr here keeps the failure mode graceful for
  // both current and future callers — the cost is one pointer
  // and one default-constructed SourceMgr per MCState.
  state.ctx->initInlineSourceManager();
  state.disasm.reset(
      state.target->createMCDisassembler(*state.subtargetInfo, *state.ctx));
  state.printer.reset(state.target->createMCInstPrinter(
      triple, 0, *state.asmInfo, *state.instrInfo, *state.regInfo));
  state.printer->setPrintImmHex(true);
  return true;
}

std::string getMnemonic(const MCState &mc, const MCInst &inst) {
  std::string s;
  raw_string_ostream os(s);
  mc.printer->printInst(&inst, 0, "", *mc.subtargetInfo, os);
  StringRef sr(s);
  sr = sr.ltrim();
  return sr.split('\t').first.split(' ').first.str();
}

StringRef stripEncoding(StringRef mn) {
  for (StringRef suffix : {"_e32", "_e64", "_vi"})
    if (mn.ends_with(suffix))
      return mn.drop_back(suffix.size());
  return mn;
}

} // namespace transpiler
