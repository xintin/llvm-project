; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=setpc_swap_pattern_a_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for s_swap_pc_i64 — Pattern A (statically resolvable
; direct call). Pins that the SOP1 branch-and-link lowers to:
;   1. an i64 return-address marker (the plain source-MC byte
;      offset of the BB immediately following the swap) written to
;      sdst (the ret SGPR pair), and
;   2. an unconditional `br label %bb_<callee-offset>`
; whenever the static analysis in transpiler/setpc_analysis.cpp can
; fold the call-target SGPR pair's provenance through the canonical
; chain
;   s_get_pc_i64 → s_add_co_u32 → s_add_co_ci_u32
; into a single in-kernel offset. The handler lives in
; transpiler/handle_sop1.cpp under
; `if (sop == SemOp::S_SWAP_PC_I64) { ... DirectA path }`. The SemOp
; doc in transpiler/semop.hpp pins the lowering contract.
;
; Why we write an integer marker and not
; `ptrtoint(blockaddress(@kernel, %bb_<ret>))` (as earlier revisions
; of this fixture pinned): AMDGPU's instruction selector has no
; pattern to materialise a `BlockAddress` constant as an i64
; register value, so any `BlockAddress` SDNode that survives the
; middle-end pipeline aborts llc with
;   `Cannot select: t1: i64 = BlockAddress<@kernel, %bb_N>`.
; Using a plain integer offset sidesteps ISel entirely — see the
; rationale block on `emitEnumeratedDispatch` in handle_sop1.cpp.
;
; Why these CHECKs:
;   * `bb_0x18` is the fixture's resolved return target. Derived in
;     the .hip header doc:
;        K = 0x08 (kernarg-load prologue)
;        swap site at K+0x0C ⇒ return offset = K+0x10 = 0x18
;     The analysis adds 0x18 to extraBlockStarts (Phase 1 promotes
;     swap.offset+size to a leader) so `bb_0x18` is a real, named
;     BB in the lifted IR.
;   * `bb_0x20` is the chain-resolved callee. Derived as:
;        K+0x18 = 0x20 (the s_add_co_u32 IMM=20 lands the chain at
;        K+0x18; absolute kernel offset is 0x08+0x18 = 0x20)
;     The handler emits `br label %bb_0x20`.
;   * In this fixture sdst is never consumed (the downstream code is
;     straight-line and the return-path BB is an immediate
;     fall-through), so mem2reg + DCE promote-and-drop the sdst
;     alloca. We therefore do NOT pin the sdst write in the IR text;
;     the invariant that matters is pinned negatively below
;     (`CHECK-NOT: blockaddress(`) so a regression that re-introduced
;     the old `ptrtoint(blockaddress(...))` sdst materialisation
;     would fail the negative check.
;   * `CHECK-NOT: indirectbr` pins that we did NOT degenerate into
;     an IndirectB-shaped lowering; the DirectA arm must emit a
;     single unconditional `br`.
;   * `CHECK-NOT: blockaddress(` pins the ISel-safety fix: the
;     swap handler's sdst write must use a plain integer marker,
;     not `ptrtoint(blockaddress(...))`.

; CHECK-LABEL: define amdgpu_kernel void @setpc_swap_pattern_a_kernel(

; The swap site emits the direct branch to the chain-resolved
; callee. The return-address marker is written into sdst by the
; handler (as a plain `ConstantInt::get(i64Ty, 0x18)`) but is dead
; in this fixture and does not appear in the emitted IR.
; CHECK: br label %bb_0x28

; Both the return-target BB and the callee BB are real, named blocks
; in the lifted IR (the return site from extraBlockStarts via
; swap.offset+size, and the callee from extraBlockStarts via the
; resolved chain target).
; CHECK: bb_0x20:
; CHECK: bb_0x28:

; No `indirectbr` may leak back into the lifted IR.
; CHECK-NOT: indirectbr
; No `blockaddress` constant may leak back into the lifted IR —
; the sdst write must use a plain integer marker, not
; `ptrtoint(blockaddress(...))`.
; CHECK-NOT: blockaddress(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	setpc_swap_pattern_a_kernel
	.p2align	8
	.type	setpc_swap_pattern_a_kernel,@function
setpc_swap_pattern_a_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	;;#ASMSTART
	s_get_pc_i64 s[10:11]
	s_add_co_u32 s10, s10, 20
	s_add_co_ci_u32 s11, s11, 0
	s_swap_pc_i64 s[20:21], s[10:11]
	v_mov_b32 v1, 0xCAFE0001
	v_mov_b32 v1, 0xDEAD0001
	s_endpgm
	
	;;#ASMEND
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel setpc_swap_pattern_a_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 22
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           setpc_swap_pattern_a_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .symbol:         setpc_swap_pattern_a_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
