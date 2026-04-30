#ifndef HOTSWAP_TRANSPILER_MUBUF_ADDR_HPP
#define HOTSWAP_TRANSPILER_MUBUF_ADDR_HPP

#include "decoded_inst.hpp"
#include "parsed_reg.hpp"
#include "raise_context.hpp"

#include "llvm/IR/Value.h"

namespace transpiler {

// Decoded addressing shape of a MUBUF / VBUFFER load, store, or atomic.
//
// `srd` is a <4 x i32> raw buffer descriptor built from the SRSRC
// 128-bit SGPR tuple. `rawPtrRsrc` is the same resource expressed as
// LLVM's addrspace(8) buffer-resource pointer via
// `llvm.amdgcn.make.buffer.rsrc`; cross-widening MUBUF loads use that
// form so LLVM preserves raw-pointer buffer semantics instead of
// consuming Salmon-synthesised descriptor dwords. `voffset` is the per-lane i32 byte offset
// (vaddr + imm). `soffset` is the SGPR byte offset (defaults to 0).
// `auxFlags` is always a zero i32 — the raw buffer intrinsics take it
// but today's raiser doesn't surface cpol/th/scope (see the
// "auxFlags is hard-coded to 0" item in
// hotswap/docs/buffer-store-lowering.md §"Known limitations").
//
// For stores and atomics, `stData` is the VGPR source carrying the data.
// For loads it's a default-constructed ParsedReg (kind == OTHER).
//
// Historical note: an earlier revision of this struct also exposed
// the raw SRSRC dwords (`dw0` / `dw1` / `dw2`) for the flat-store
// OOB-sink pattern in handle_mubuf.cpp.  That pattern was removed
// when the BUFFER_STORE handler was rewritten to use
// `llvm.amdgcn.raw.buffer.store` directly (R1 fix; see
// hotswap/docs/buffer-store-lowering.md), so the dword fields became
// dead — every consumer now goes through `srd`.  The fields were
// removed in the same cleanup pass that added the lit regression
// guard (lit_tests/buffer_store_no_scratch_alloca/).
struct MubufAddr {
  llvm::Value *srd = nullptr;
  llvm::Value *rawPtrRsrc = nullptr;
  llvm::Value *voffset = nullptr;
  llvm::Value *soffset = nullptr;
  llvm::Value *auxFlags = nullptr;
  ParsedReg stData;
};

// Decode a MUBUF / VBUFFER load, store, or atomic's addressing operands into a
// fully-materialised `MubufAddr`. Recognises the LLVM-MC operand
// layout:
//
//   load:  [dst(vdata), srsrc(SGPR4), vaddr(VGPR), soff(SGPR?), imm?, cpol?]
//   store/atomic:
//         [srsrc(SGPR4), vdata(VGPR), vaddr(VGPR), soff(SGPR?), imm?, cpol?]
//
// Both MUBUF and VBUFFER encodings share this shape — the operand
// order inside an encoding can vary between LLVM versions so the
// classifier keys on `ParsedReg::Kind` rather than position.
//
// Fails loudly via `report_fatal_error` on unrecognised shapes (no
// SRSRC found, or cannot read SRSRC dwords). `diagLabel` names the
// caller for the error message (e.g. `"MUBUF"` or `"MUBUF_LDS"`).
MubufAddr decodeMubufAddr(RaiseContext &ctx, const DecodedInst &di,
                          OpResolver &op, bool isStore,
                          llvm::StringRef diagLabel);

} // namespace transpiler

#endif
