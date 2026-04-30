; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=setpc_pattern_a_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for s_set_pc_i64 — Pattern A (statically resolvable
; intra-kernel direct branch). Pins that the SOP1 indirect set-PC
; lowers to a plain `br label %BB_target` when the static analysis
; in transpiler/setpc_analysis.cpp can fold the source SGPR pair's
; provenance through the canonical chain
;   s_get_pc_i64 → s_add_co_u32 → s_add_co_ci_u32
; into a single in-kernel offset.
;
; The handler lives in transpiler/handle_sop1.cpp under
; `if (sop == SemOp::S_SET_PC_I64) { ... case DirectA: }`. The
; SemOp doc in transpiler/semop.hpp pins the lowering contract.
;
; Why these CHECKs:
;   * `bb_0x18` is the fixture's chain-resolved target. It is
;     derived in the .hip header doc:
;        s_get_pc_i64 records s[10:11] = (di.offset + di.size)
;                                      = 0x08 + 4 = 0x0C (kernel-relative)
;        s_add_co_u32 with imm=12 ⇒ s[10:11] = 0x0C + 12 = 0x18
;     The analysis adds 0x18 to extraBlockStarts; the raiser names
;     basic blocks `bb_0x<hex offset>`.
;   * The `br label %bb_0x18` is the Direct-A lowering. A Pattern B
;     regression would emit `indirectbr ptr ..., [`; an Unresolvable
;     regression would refuse and never produce IR; a silent stub
;     regression would emit `unreachable` instead of `br`.
;   * `bb_0x18:` confirms the target block actually exists in the
;     emitted IR (catches a regression where the chain target gets
;     promoted to a leader but no block is materialised).

; CHECK-LABEL: define amdgpu_kernel void @setpc_pattern_a_kernel(

; The s_set_pc_i64 site lowers to a direct br to the chain-resolved
; offset. CHECK-NOT pins that we did NOT degenerate to indirectbr or
; unreachable in the same function (a Pattern B / Unresolvable
; regression would surface as one of those).
; CHECK: br label %bb_0x20
; CHECK-NOT: indirectbr ptr
; CHECK-NOT: unreachable

; The branch target is a real, named block.
; CHECK: bb_0x20:

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	setpc_pattern_a_kernel
	.p2align	8
	.type	setpc_pattern_a_kernel,@function
setpc_pattern_a_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	;;#ASMSTART
	s_get_pc_i64 s[10:11]
	s_add_co_u32 s10, s10, 12
	s_add_co_ci_u32 s11, s11, 0
	s_set_pc_i64 s[10:11]
	v_mov_b32 v1, 0xDEAD0001
	
	;;#ASMEND
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel setpc_pattern_a_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 12
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
    .name:           setpc_pattern_a_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         setpc_pattern_a_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
