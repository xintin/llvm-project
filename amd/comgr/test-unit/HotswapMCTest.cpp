//===- HotswapMCTest.cpp - Unit tests for HotSwap LLVM MC layer -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for the hotswap MC/LLVM infrastructure in comgr-hotswap-llvm.cpp:
/// initLLVM construction, LLVMState::encodeSBranch, assembleSingleInst /
/// decodeTextSection round-trip, applyMnemonicSwap, applyByteReplace, and
/// checkVgprOverlap.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "comgr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

#include <cstring>
#include <mutex>

using namespace COMGR;
using namespace COMGR::hotswap;

// --------------------------------------------------------------------------
// Test-only stub definition of COMGR::ensureLLVMInitialized.
//
// hotswap::initLLVM() calls COMGR::ensureLLVMInitialized() (normally defined
// in comgr.cpp) to register the AMDGPU target. The production definition
// lives in libamd_comgr, which we don't want to link into the unit-test
// binary (it drags in the full Comgr compiler pipeline). Providing this
// stub here keeps the test binary minimal while matching the production
// registration behaviour for the target components we exercise.
//
// Stubbing is safe because this translation unit is linked into
// HotswapMCTests only, never into libamd_comgr.
// --------------------------------------------------------------------------
namespace COMGR {
void ensureLLVMInitialized() {
  static std::once_flag Once;
  std::call_once(Once, []() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUTarget();
  });
}
} // namespace COMGR

// Build a TargetIdentifier for the gfx1250 test subtarget without features --
// production callers go through parseTargetIdentifier; here we populate
// directly so the tests stay self-contained.
static TargetIdentifier makeGfx1250Ident() {
  TargetIdentifier TI;
  TI.Arch = "amdgcn";
  TI.Vendor = "amd";
  TI.OS = "amdhsa";
  TI.Environ = "";
  TI.Processor = "gfx1250";
  return TI;
}

// Helper: decode the little-endian 32-bit dword at \p Bytes.
static uint32_t readDword(const uint8_t *Bytes) {
  uint32_t V;
  std::memcpy(&V, Bytes, sizeof(V));
  return V;
}

// -- initLLVM ----------------------------------------------------------------

TEST(InitLLVM, ValidGfx1250) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_EQ(S.Cpu, "gfx1250");
  EXPECT_NE(S.Target, nullptr);
  ASSERT_NE(S.MCII, nullptr);
  EXPECT_LT(S.SBranchOpcode, S.MCII->getNumOpcodes());
  EXPECT_EQ(S.SNopBytes.size(), MinInstSize);
}

TEST(InitLLVM, EmptyProcessorFails) {
  TargetIdentifier TI = makeGfx1250Ident();
  TI.Processor = "";
  LLVMState S = initLLVM(TI);
  EXPECT_FALSE(S.Valid);
}

TEST(InitLLVM, UnknownProcessorFails) {
  TargetIdentifier TI = makeGfx1250Ident();
  TI.Processor = "gfxbogus";
  LLVMState S = initLLVM(TI);
  EXPECT_FALSE(S.Valid);
}

// -- LLVMState::encodeSBranch -------------------------------------------------
//
// Exact byte checks are avoided here -- tblgen encodings can be reshuffled
// across LLVM versions. Instead we assert the structural invariants that
// downstream callers rely on: the encoded delta round-trips to the expected
// simm16 field, the size is MinInstSize, and out-of-range / unaligned deltas
// are rejected.

TEST(EncodeSBranch, ForwardBranchRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  uint8_t Out[MinInstSize] = {};
  // s_branch SIMM16 -> PC += (SIMM16 + 1) * 4; From=0, To=8 => SIMM16=1.
  ASSERT_TRUE(S.encodeSBranch(0, 8, Out));
  uint32_t Encoded = readDword(Out);
  EXPECT_EQ(static_cast<uint16_t>(Encoded & 0xFFFFu), 1u);
}

TEST(EncodeSBranch, BackwardBranchRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  uint8_t Out[MinInstSize] = {};
  // From=16, To=0 => delta=-5 dwords.
  ASSERT_TRUE(S.encodeSBranch(16, 0, Out));
  uint32_t Encoded = readDword(Out);
  EXPECT_EQ(static_cast<int16_t>(Encoded & 0xFFFFu), -5);
}

TEST(EncodeSBranch, ZeroOffsetBranch) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  uint8_t Out[MinInstSize] = {};
  // PC advance of MinInstSize: SIMM16 should be 0.
  ASSERT_TRUE(S.encodeSBranch(0, MinInstSize, Out));
  EXPECT_EQ(readDword(Out) & 0xFFFFu, 0u);
}

TEST(EncodeSBranch, UnalignedDeltaFails) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  uint8_t Out[MinInstSize] = {};
  EXPECT_FALSE(S.encodeSBranch(0, 7, Out));
}

TEST(EncodeSBranch, OutOfRangeFails) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  uint8_t Out[MinInstSize] = {};
  EXPECT_FALSE(S.encodeSBranch(0, 500000, Out));
}

TEST(EncodeSBranch, FailsOnInvalidState) {
  LLVMState S; // default-constructed, Valid = false
  uint8_t Out[MinInstSize] = {};
  EXPECT_FALSE(S.encodeSBranch(0, 8, Out));
}

// -- assembleSingleInst / decodeTextSection round-trip ------------------------

TEST(AssembleDecode, SNopRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst("s_nop 0", S);
  ASSERT_EQ(Bytes.size(), MinInstSize);
  // Must match the pre-encoded bytes cached in LLVMState at init time.
  EXPECT_EQ(llvm::ArrayRef<uint8_t>(Bytes),
            llvm::ArrayRef<uint8_t>(S.SNopBytes));

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  EXPECT_EQ(Decoded[0].Size, MinInstSize);
  EXPECT_EQ(Decoded[0].Mnemonic, "s_nop");
}

TEST(AssembleDecode, RejectsGarbageAsm) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst("not_a_real_op", S);
  EXPECT_TRUE(Bytes.empty());
}

// -- applyByteReplace ---------------------------------------------------------

TEST(ApplyByteReplace, PadsWithSNop) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // 8 bytes of zeroed "text", simulate replacing the first 8 bytes with a
  // 4-byte rule and expecting the remainder to be padded with s_nop.
  uint8_t Text[8] = {};
  RewriteRule Rule;
  Rule.ReplaceBytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());
  ASSERT_TRUE(applyByteReplace(Rule, /*InstOffset=*/0, /*InstSize=*/8, Text,
                               sizeof(Text), S));
  // Both halves should be s_nop bytes now.
  EXPECT_EQ(std::memcmp(Text, S.SNopBytes.data(), MinInstSize), 0);
  EXPECT_EQ(std::memcmp(Text + MinInstSize, S.SNopBytes.data(), MinInstSize),
            0);
}

TEST(ApplyByteReplace, RejectsOutOfBounds) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  uint8_t Text[4] = {};
  RewriteRule Rule;
  Rule.ReplaceBytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());
  // InstOffset+InstSize (8) exceeds TextSize (4).
  EXPECT_FALSE(applyByteReplace(Rule, /*InstOffset=*/0, /*InstSize=*/8, Text,
                                sizeof(Text), S));
}

// -- checkVgprOverlap ---------------------------------------------------------
//
// checkVgprOverlap checks whether any register operand of a "WMMA-like"
// MCInst overlaps the destination (operand 0) of a "VALU-like" MCInst.
// We drive it with real MCInsts produced by assembling + decoding simple
// AMDGPU instructions so the register operands are populated the way the
// production code sees them.

// Assemble \p Asm and decode the first resulting MCInst. Aborts the test if
// either step fails, so callers can rely on the return value being populated.
static llvm::MCInst assembleOne(llvm::StringRef Asm, const LLVMState &S) {
  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, S);
  EXPECT_FALSE(Bytes.empty()) << "failed to assemble: " << Asm.str();
  std::vector<InternalDecodedInst> Decoded;
  EXPECT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded))
      << "failed to decode: " << Asm.str();
  EXPECT_EQ(Decoded.size(), 1u) << "expected one inst for: " << Asm.str();
  return Decoded.empty() ? llvm::MCInst() : Decoded[0].Inst;
}

TEST(CheckVgprOverlap, DetectsDirectOverlap) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  // Wmma-like inst references v5 and v10; Valu-like inst writes v10.
  llvm::MCInst Wmma = assembleOne("v_mov_b32 v5, v10", S);
  llvm::MCInst Valu = assembleOne("v_mov_b32 v10, v20", S);
  EXPECT_TRUE(checkVgprOverlap(Wmma, Valu, *S.MRI));
}

TEST(CheckVgprOverlap, NoOverlapForDisjointVgprs) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  // Wmma-like inst references v0, v1; Valu-like inst writes v10.
  llvm::MCInst Wmma = assembleOne("v_mov_b32 v0, v1", S);
  llvm::MCInst Valu = assembleOne("v_mov_b32 v10, v20", S);
  EXPECT_FALSE(checkVgprOverlap(Wmma, Valu, *S.MRI));
}

TEST(CheckVgprOverlap, HandlesEmptyValuInst) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::MCInst Wmma = assembleOne("v_mov_b32 v0, v1", S);
  llvm::MCInst Empty; // no operands
  EXPECT_FALSE(checkVgprOverlap(Wmma, Empty, *S.MRI));
}

// -- buildTrampoline ----------------------------------------------------------
//
// buildTrampoline assembles one or more asm lines and appends a branch-back
// s_branch to the instruction immediately following the original site. We
// verify the size / structure of the result rather than the exact bytes
// (which are target-specific and captured separately in the encodeSBranch /
// SNopBytes tests).

TEST(BuildTrampoline, AppendsBranchBackAfterAssembledAsm) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::string AsmLine = "s_nop 0";
  std::vector<std::string> AsmLines = {AsmLine};
  constexpr uint64_t OriginalOffset = 0;
  constexpr uint32_t OriginalSize = MinInstSize;
  constexpr uint64_t TrampolineTextOffset = 0x1000;

  Trampoline T = buildTrampoline(AsmLines, OriginalOffset, OriginalSize,
                                 TrampolineTextOffset, S);

  EXPECT_EQ(T.OriginalOffset, OriginalOffset);
  EXPECT_EQ(T.OriginalSize, OriginalSize);
  // One assembled inst (s_nop 0, 4 bytes) + one branch-back (4 bytes).
  ASSERT_EQ(T.Bytes.size(), 2u * MinInstSize);
  // The first MinInstSize bytes should match the cached s_nop encoding.
  EXPECT_EQ(std::memcmp(T.Bytes.data(), S.SNopBytes.data(), MinInstSize), 0);
}

TEST(BuildTrampoline, EmptyOnBadAsm) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<std::string> AsmLines = {"this_is_not_a_valid_instruction"};
  Trampoline T = buildTrampoline(AsmLines, /*OriginalOffset=*/0,
                                 /*OriginalSize=*/MinInstSize,
                                 /*TrampolineTextOffset=*/0x1000, S);
  EXPECT_TRUE(T.Bytes.empty());
}

// -- isDS2Mnemonic ------------------------------------------------------------
//
// Predicate that classifies decoded mnemonics as DS2 instructions for the
// applyDS2WaitBump post-pass. The tests pin the StringSwitch enumeration so
// adding (or accidentally renaming) a variant trips a runtime check
// immediately. Hostile-prefix near-misses guard against the
// `starts_with`/`contains` regressions flagged in PR #2293 review.

TEST(IsDS2Mnemonic, NonStride64VariantsMatch) {
  EXPECT_TRUE(isDS2Mnemonic("ds_load_2addr_b32"));
  EXPECT_TRUE(isDS2Mnemonic("ds_load_2addr_b64"));
  EXPECT_TRUE(isDS2Mnemonic("ds_store_2addr_b32"));
  EXPECT_TRUE(isDS2Mnemonic("ds_store_2addr_b64"));
}

TEST(IsDS2Mnemonic, Stride64VariantsMatch) {
  EXPECT_TRUE(isDS2Mnemonic("ds_load_2addr_stride64_b32"));
  EXPECT_TRUE(isDS2Mnemonic("ds_load_2addr_stride64_b64"));
  EXPECT_TRUE(isDS2Mnemonic("ds_store_2addr_stride64_b32"));
  EXPECT_TRUE(isDS2Mnemonic("ds_store_2addr_stride64_b64"));
}

TEST(IsDS2Mnemonic, NearMissesDoNotMatch) {
  EXPECT_FALSE(isDS2Mnemonic(""));
  // Single-address DS variants share the prefix but are not split-targets.
  EXPECT_FALSE(isDS2Mnemonic("ds_load_b32"));
  EXPECT_FALSE(isDS2Mnemonic("ds_store_b64"));
  // Truncated mnemonic (no width suffix).
  EXPECT_FALSE(isDS2Mnemonic("ds_load_2addr"));
  EXPECT_FALSE(isDS2Mnemonic("ds_load_2addr_stride64"));
  // Wrong width suffix not actually emitted.
  EXPECT_FALSE(isDS2Mnemonic("ds_load_2addr_b16"));
  // Suffix collision: catches a regression to `starts_with` matching.
  EXPECT_FALSE(isDS2Mnemonic("ds_load_2addr_b32_extra"));
  // Prefix collision: catches a regression to `contains` matching.
  EXPECT_FALSE(isDS2Mnemonic("v_ds_load_2addr_b32"));
  EXPECT_FALSE(isDS2Mnemonic("prefix_ds_load_2addr_b32"));
  // Trailing whitespace: catches a regression where getMnemonic() stops
  // calling rtrim() on the AsmStrs entry.
  EXPECT_FALSE(isDS2Mnemonic("ds_load_2addr_b32 "));
  // Case sensitivity (assembly mnemonics are lowercase).
  EXPECT_FALSE(isDS2Mnemonic("DS_LOAD_2ADDR_B32"));
  // Non-DS instructions.
  EXPECT_FALSE(isDS2Mnemonic("s_wait_dscnt"));
  EXPECT_FALSE(isDS2Mnemonic("v_nop"));
  // Stride32 instead of stride64 (typo guard).
  EXPECT_FALSE(isDS2Mnemonic("ds_load_2addr_stride32_b32"));
}

// -- encodeWaitWithImm --------------------------------------------------------
//
// Pure encode primitive. We feed it a real `s_wait_dscnt` MCInst built via
// the asm parser, ask for a different immediate, and round-trip the bytes
// back through the disassembler to verify the operand made it to the
// encoded form. Avoids byte-exact checks (which would couple the test to
// gfx1250-specific SOPP encoding) and mirrors the approach used by the
// EncodeSBranch / AssembleDecode tests above.

// Decode a single instruction from \p Bytes and return its decoded MCInst,
// asserting size + mnemonic match the expectations. Returns a default
// MCInst on any decode failure so callers can EXPECT_* on the return.
static llvm::MCInst decodeOne(llvm::ArrayRef<uint8_t> Bytes,
                              llvm::StringRef ExpectedMnemonic,
                              const LLVMState &S) {
  std::vector<InternalDecodedInst> Decoded;
  EXPECT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  EXPECT_EQ(Decoded.size(), 1u);
  if (Decoded.size() != 1)
    return llvm::MCInst();
  EXPECT_EQ(Decoded[0].Mnemonic, ExpectedMnemonic.str());
  return Decoded[0].Inst;
}

TEST(EncodeWaitWithImm, RoundTripsBumpedImm) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Original = assembleSingleInst("s_wait_dscnt 5", S);
  ASSERT_FALSE(Original.empty());
  llvm::MCInst Wait = decodeOne(Original, "s_wait_dscnt", S);

  llvm::SmallVector<uint8_t> Bumped = encodeWaitWithImm(
      Wait, /*NewImm=*/7, S, /*ExpectedSize=*/Original.size());
  ASSERT_EQ(Bumped.size(), Original.size());

  llvm::MCInst Decoded = decodeOne(Bumped, "s_wait_dscnt", S);
  ASSERT_GE(Decoded.getNumOperands(), 1u);
  ASSERT_TRUE(Decoded.getOperand(0).isImm());
  EXPECT_EQ(Decoded.getOperand(0).getImm(), 7);
}

TEST(EncodeWaitWithImm, ClampsAtMax) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Original = assembleSingleInst("s_wait_dscnt 0", S);
  ASSERT_FALSE(Original.empty());
  llvm::MCInst Wait = decodeOne(Original, "s_wait_dscnt", S);

  // Ask for a value past the 6-bit field; encoder must clamp at 63.
  llvm::SmallVector<uint8_t> Bumped =
      encodeWaitWithImm(Wait, /*NewImm=*/100, S, Original.size());
  ASSERT_EQ(Bumped.size(), Original.size());

  llvm::MCInst Decoded = decodeOne(Bumped, "s_wait_dscnt", S);
  ASSERT_TRUE(Decoded.getOperand(0).isImm());
  EXPECT_EQ(Decoded.getOperand(0).getImm(), 63);
}

TEST(EncodeWaitWithImm, EncodesZeroImm) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Original = assembleSingleInst("s_wait_dscnt 5", S);
  ASSERT_FALSE(Original.empty());
  llvm::MCInst Wait = decodeOne(Original, "s_wait_dscnt", S);

  llvm::SmallVector<uint8_t> Zero =
      encodeWaitWithImm(Wait, /*NewImm=*/0, S, Original.size());
  ASSERT_EQ(Zero.size(), Original.size());
  llvm::MCInst Decoded = decodeOne(Zero, "s_wait_dscnt", S);
  ASSERT_TRUE(Decoded.getOperand(0).isImm());
  EXPECT_EQ(Decoded.getOperand(0).getImm(), 0);
}

TEST(EncodeWaitWithImm, RejectsNonImmOperand) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // Construct an MCInst whose operand 0 is a register, not an immediate.
  // encodeWaitWithImm must refuse rather than corrupting the encoding.
  llvm::MCInst Bogus;
  Bogus.setOpcode(0); // any opcode; we only inspect operand 0
  Bogus.addOperand(llvm::MCOperand::createReg(1));
  EXPECT_TRUE(
      encodeWaitWithImm(Bogus, /*NewImm=*/3, S, /*ExpectedSize=*/4).empty());

  llvm::MCInst Empty;
  EXPECT_TRUE(
      encodeWaitWithImm(Empty, /*NewImm=*/3, S, /*ExpectedSize=*/4).empty());
}

TEST(EncodeWaitWithImm, RejectsSizeMismatch) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Original = assembleSingleInst("s_wait_dscnt 0", S);
  ASSERT_FALSE(Original.empty());
  llvm::MCInst Wait = decodeOne(Original, "s_wait_dscnt", S);

  // Lying about the expected size must produce empty (not a partial
  // write). Pick a clearly-wrong size to make the intent obvious.
  EXPECT_TRUE(encodeWaitWithImm(Wait, /*NewImm=*/3, S,
                                /*ExpectedSize=*/Original.size() + 4)
                  .empty());
}
