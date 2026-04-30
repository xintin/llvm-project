; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_sub_nc_u64_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for s_sub_nc_u64. Pins that the SOP2 64-bit no-carry
; subtract lowers to a single `sub i64`. The handler lives in
; transpiler/handle_sop2.cpp under
; `if (sop == SemOp::S_SUB_NC_U64) { ... }`; the SemOp lives in
; transpiler/semop.hpp under SOP2.
;
; The "nc" suffix matters: the gfx12 form intentionally does NOT
; update SCC (see SOPInstructions.td around line 661 — `S_SUB_U64`
; is defined outside the surrounding `let Defs = [SCC]` block). A
; regression that emits a `usub.with.overflow` intrinsic (or any
; SCC-writing variant) would defeat that, so we negative-CHECK
; both shapes.

; CHECK-LABEL: define amdgpu_kernel void @s_sub_nc_u64_kernel(

; The lifted body must contain a single i64 sub — the handler's
; `ssub64` value-name is the canonical breadcrumb (mirrors the
; `sadd64` / `smul64` siblings in the same handler file).
; CHECK: sub {{.*}}i64 %{{[^,]+}}, %{{[^,]+}}

; Negative checks: no SCC-defining intrinsic and no narrower-width
; sub for this kernel. If the handler ever shrinks to 32-bit halves
; or grows an overflow-tracked variant, this fires.
; CHECK-NOT: @llvm.usub.with.overflow
; CHECK-NOT: sub {{.*}}i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_sub_nc_u64_kernel
	.p2align	8
	.type	s_sub_nc_u64_kernel,@function
s_sub_nc_u64_kernel:
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
	v_mad_u32 v2, s0, s8, v0
	;;#ASMSTART
	s_sub_nc_u64 s[0:1], s[6:7], s[2:3]
	
	;;#ASMEND
	v_mov_b64_e32 v[0:1], s[0:1]
	global_store_b64 v2, v[0:1], s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_sub_nc_u64_kernel
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
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
    .name:           s_sub_nc_u64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         s_sub_nc_u64_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
