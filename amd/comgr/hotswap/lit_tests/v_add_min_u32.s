; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_add_min_u32_kernel 2>/dev/null | %FileCheck %s --check-prefix=DEFAULT
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_add_min_u32_clamp_kernel 2>/dev/null | %FileCheck %s --check-prefix=CLAMP
;
; Lift test for gfx1250 v_add_min_u32. This pins the rocBLAS blocker shape
; seen in kerneldex: scalar source + literal -1 + VGPR source. The semantic
; identity comes from VOP3Instructions.td:
;   dst = umin(uaddsat(src0, src1), src2)
;
; LLVM also exposes llvm.amdgcn.add.min.u32 with an explicit clamp operand, but
; that target intrinsic does not lower on gfx942. The handler deliberately uses
; generic LLVM IR (`llvm.uadd.sat.i32` + `llvm.umin.i32`) because LLVM has a
; generated pattern from that shape back to V_ADD_MIN_U32 when the target
; supports the opcode, and the same IR remains legal when it does not.
;
; The second RUN line pins clamp=1 acceptance. Integer clamp limits the result
; to the operation type's representable range; for U32 add-min the generic
; unsigned-saturating-add plus unsigned-min form is already in that range.
;
; DEFAULT-LABEL: define amdgpu_kernel void @v_add_min_u32_kernel(
; DEFAULT: %vadd_min_sum{{[0-9]*}} = call i32 @llvm.uadd.sat.i32(i32 %{{[^,]+}}, i32 -1)
; DEFAULT: %vadd_min{{[0-9]*}} = call i32 @llvm.umin.i32(i32 %vadd_min_sum{{[0-9]*}}, i32 %{{[^)]+}})
; DEFAULT-NOT: call {{.*}}@llvm.amdgcn.add.min.u32
; DEFAULT-NOT: icmp slt
;
; CLAMP-LABEL: define amdgpu_kernel void @v_add_min_u32_clamp_kernel(
; CLAMP: %vadd_min_sum{{[0-9]*}} = call i32 @llvm.uadd.sat.i32(i32 {{[^,]+}}, i32 {{[^,]+}})
; CLAMP: %vadd_min{{[0-9]*}} = call i32 @llvm.umin.i32(i32 %vadd_min_sum{{[0-9]*}}, i32 {{[^)]+}})
; CLAMP-NOT: call {{.*}}@llvm.amdgcn.add.min.u32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_add_min_u32_kernel
	.p2align	8
	.type	v_add_min_u32_kernel,@function
v_add_min_u32_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b32 s2, s[0:1], 0x1c
	s_bfe_u32 s3, ttmp6, 0x4000c
	s_and_b32 s4, ttmp6, 15
	s_add_co_i32 s3, s3, 1
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s3, ttmp9, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s4, s4, s3
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s3, ttmp9, s4
	v_mad_u32 v3, s3, s2, v0
	s_load_b128 s[0:3], s[0:1], 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v0, v3, 1, v3
	v_ashrrev_i32_e32 v1, 31, v0
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[2:3]
	global_load_b96 v[0:2], v[0:1], off
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_add_min_u32 v0, s2, -1, v0

	;;#ASMEND
	global_store_b32 v3, v0, s[0:1] scale_offset
	s_endpgm
	.globl	v_add_min_u32_clamp_kernel
	.p2align	8
	.type	v_add_min_u32_clamp_kernel,@function
v_add_min_u32_clamp_kernel:
	v_mov_b32_e32 v0, 1
	v_mov_b32_e32 v1, 2
	v_mov_b32_e32 v2, 3
	v_add_min_u32 v0, v0, v1, v2 clamp
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_add_min_u32_kernel
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 6
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_add_min_u32_clamp_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 8
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
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           v_add_min_u32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         v_add_min_u32_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_add_min_u32_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_add_min_u32_clamp_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
