; RUN: %raise_cli %global_load_async_to_lds_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=global_load_async_to_lds_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Lift fixture for FLAT `global_load_async_to_lds_b{8,32,64,128}` on the
; CROSS-TARGET arm (gfx1250 -> gfx942). Pins the synchronous per-lane
; emulation in transpiler/handle_flat.cpp under
; `SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128}` when
; `ctx.targetIsa.hasTensorOps` is false. Companion fixture to
; `global_load_async_to_lds_same_target.ll`, which pins the
; same-target (gfx1250 -> gfx1250) intrinsic-emit shape.
;
; === Why this is an emulation, not the prior loud refusal ===
;
; Pre-2026-04-24 (see the git-log entry accompanying this fixture
; change), the cross-target arm refused the lift loudly with a
; `RaiseFailure::unsupportedShape` diagnostic that cited the
; gfx1250-only asynccnt unit and the gated `amdgcn.global.load.async.
; to.lds.b*` intrinsic. That posture was correct in the sense of
; avoiding silent miscompile, but it blocked every `matmul_ogs_*`
; MoE expert-GEMM kernel in the GPT-OSS surface (4 kernels,
; runtime-dominant in inference) end-to-end on gfx942. Every one
; of those kernels compiled without errors up to the first
; async-DMA instruction, then refused.
;
; The DOCUMENTED semantic trade-off (spelled out verbatim on the
; SemOp doc block in `semop.hpp`): the synchronous `load` + `store`
; chain produces bit-identical per-lane LDS state to what the source
; async DMA produces AFTER the companion `s_wait_asynccnt 0`. The
; ONLY information lost is the **pipelining overlap** between the
; async DMA and unrelated VMEM / LDS ops in the wave's own
; instruction stream; on gfx942 the emulation serialises those.
; This is a throughput regression, not a correctness regression.
; Kernels that depend on observable effects under a partially-
; elapsed asynccnt (e.g. a hand-written pipeliner polling asynccnt
; state) are NOT in the GPT-OSS corpus and remain out of scope;
; they would have to be hand-written to produce instructions that
; LLVM IR cannot express anyway.
;
; === ISA pragma reference (verbatim) ===
;
; From `hotswap/docs/manuals/instruction_manual.pdf §13.6.{9,10,11,12}`
; (MI400 Shader Instruction Set, sp3), identical across all four
; widths except the LDS-store byte count:
;
;   pragma "vector" do
;     dsaddr  = LDS_BASE.b32 + VGPR[laneId][VDST.u32] + INST_OFFSET.b32;
;     memaddr = ADDR;   // CalcGlobalAddr(VADDR, SADDR, IOFFSET)
;     LDS[dsaddr].bN = MEM[memaddr].bN   (N = 8 / 32 / 64 / 128)
;   endpragma
;
; Two key observations the emulation relies on:
;
;   1. `pragma "vector" do` runs the body per-active-lane; the
;      raiser wraps the `load` + `store` pair in `emitUnderExec` so
;      inactive lanes skip the whole memory round-trip.
;
;   2. `INST_OFFSET` applies to BOTH the LDS address and the global
;      address.  For this HIP fixture all offsets are zero, so the
;      GEP chain is elided.  The handler has a non-zero-offset
;      branch that GEPs the offset onto both pointers via
;      `CreateGEP(i8Ty, ptr, i64 offset)`; a future fixture
;      exercising non-zero flat_offset would pin that branch.
;
; === Per-width emission ===
;
; The HIP fixture compiles the three clang builtins
; `__builtin_amdgcn_global_load_async_to_lds_b{32,64,128}` into
; three SADDR-form instructions with `scale_offset` enabled (the
; assembler emits `scale_offset` because the per-lane VGPR offset
; is the thread-id times the access element size).  The emulation
; materialises `scale_offset` as `mul i64 %voff_zext, N` for
; N = 4 / 8 / 16 respectively (see `decode.cpp::decodeScaleOffset`
; for how the cpol bit becomes `DecodedInst::hasScaleOffset`).
;
; The LDS-base VGPR holds the per-lane i32 address (lds_base +
; tid*elemBytes on this fixture); the emulation casts it to
; `ptr addrspace(3)` via `inttoptr i32`, matching the same-target
; arm's LDS-pointer shape (see
; `global_load_async_to_lds_same_target.ll`) so IR-shape reviewers
; can visually diff the two arms at the LDS-base cast.
;
; Access type per width — larger widths are lifted as vectors of
; i32 rather than a single `iN`, mirroring
; `GLOBAL_LOAD_DWORDX{2,3,4}`'s handling of the same aggregate
; shape; the backend picks `global_load_{dword,dwordx2,dwordx4}` /
; `ds_store_{b32,b64,b128}` on gfx942 from the aligned-load
; attribute below:
;
;   b32  : i32
;   b64  : <2 x i32>
;   b128 : <4 x i32>
;
; The natural alignment (accessBytes = 4 / 8 / 16) is attached to
; both the load and store so the backend's memop-alignment-derived
; codegen picks the right dword / dwordx2 / dwordx4 opcode.

; ----- b32 ----- (first load in the HIP kernel)

; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 4
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gload{{[0-9]*}} = load i32, ptr addrspace(1) %{{[0-9]+}}, align 4
; IR: store i32 %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 4

; ----- b64 ----- (second load)

; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 8
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gload{{[0-9]*}} = load <2 x i32>, ptr addrspace(1) %{{[0-9]+}}, align 8
; IR: store <2 x i32> %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 8

; ----- b128 ----- (third load)

; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 16
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gload{{[0-9]*}} = load <4 x i32>, ptr addrspace(1) %{{[0-9]+}}, align 16
; IR: store <4 x i32> %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 16

; ----- Negative assertions -----
;
; The emulation MUST NOT emit the native intrinsic on the cross-
; target arm (the intrinsic is gfx1250-only and would fail isel on
; gfx942).  A regression that accidentally took the same-target
; path on a non-gfx1250 target would surface here as an IR shape
; containing the intrinsic name.

; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b8
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b32
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b64
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b128
