; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_cmp_eq_u64_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for the gfx8+ SOPC s_cmp_eq_u64 (and its sibling
; s_cmp_lg_u64 — both are SOPC_CMP_64 in SOPInstructions.td and
; share the same handler-block shape). See SemOp::S_CMP_EQ_U64 in
; transpiler/semop.hpp; the matching handler block in
; transpiler/handle_sopc.cpp under `if (sop == SemOp::S_CMP_EQ_U64)`;
; and the SOPC mapping in transpiler/opcode_map.cpp.
;
; INVARIANTS PINNED:
;
;   1. The kernel signature carries the i64 source-operand arguments
;      directly (`i64 %arg1, i64 %arg2`); no narrowing to i32. This
;      pins the typical SOPC_CMP_64 corpus shape.
;
;   2. The lift emits a single `icmp eq i64`. The handler reads
;      both operands via `op.src64(...)` (i64 SGPR-pair reads,
;      since SOPC_CMP_64 takes two 64-bit operands) and emits
;      `CreateICmpEQ`. A regression that narrowed the operands to
;      i32 — a common shortcut against 64-bit SGPR pairs — would
;      break the corpus's per-thread-mask compares used in
;      tensilelite gemm dispatch. The `scmp64` value-name is the
;      canonical breadcrumb (mirrors `scmp` for the 32-bit
;      compares).
;
;   3. The compare result drives `s_cselect_b32` via SCC: the lift
;      stores the i1 to SCC (the storeSCC path), and the cselect
;      reads it back as the predicate of `select i1 %scmp64, i32 1,
;      i32 0`. This pin verifies the SCC-writeback path end-to-end.
;
;   4. NO 32-bit compare on the source operands — a regression to
;      a pair-of-i32 lift would emit `icmp eq i32` against the
;      i64-wide source.

; CHECK-LABEL: define amdgpu_kernel void @s_cmp_eq_u64_kernel(
; CHECK-SAME: ptr addrspace(1) %arg0
; CHECK-SAME: i64 %arg1
; CHECK-SAME: i64 %arg2

; The s_cmp_eq_u64 lift. The handler-emitted value-name `scmp64`
; appears verbatim, and the operand types are i64.
; CHECK: %scmp64 = icmp eq i64 %{{[^,]+}}, %{{[^,]+}}

; The s_cselect_b32 immediately after the compare reads SCC. The
; lift surfaces this as `select i1 %scmp64, i32 1, i32 0` — pins
; the SCC-writeback chain end-to-end.
; CHECK: select i1 %scmp64, i32 1, i32 0

; Negative pin: no narrowing of the 64-bit operands to i32.
; CHECK-NOT: %scmp64 = icmp eq i32

; Negative pin: no signed-compare regression (the SOPC_CMP_64
; family is unsigned-only per SOPInstructions.td).
; CHECK-NOT: %scmp64 = icmp sgt
; CHECK-NOT: %scmp64 = icmp sge
; CHECK-NOT: %scmp64 = icmp slt
; CHECK-NOT: %scmp64 = icmp sle

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_cmp_eq_u64_kernel
	.p2align	8
	.type	s_cmp_eq_u64_kernel,@function
s_cmp_eq_u64_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x2
	s_load_b32 s8, s[0:1], 0x24
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x10
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s9, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s8, s8, 0xffff
	s_cmp_eq_u32 s9, 0
	s_cselect_b32 s0, ttmp9, s1
	v_mad_u32 v0, s0, s8, v0
	;;#ASMSTART
	s_cmp_eq_u64 s[6:7], s[2:3]
	s_cselect_b32 s0, 1, 0
	
	;;#ASMEND
	v_mov_b32_e32 v1, s0
	global_store_b32 v0, v1, s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_cmp_eq_u64_kernel
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 10
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
      - { .offset:         8, .size:           8, .value_kind:     by_value }
      - { .offset:         16, .size:           8, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .max_flat_workgroup_size: 1024
    .name:           s_cmp_eq_u64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         s_cmp_eq_u64_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
