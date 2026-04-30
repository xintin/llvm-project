; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=user_sgpr_count_32_gfx125_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; GPT-OSS/SGLang RoPE regression pin: gfx125 widens
; compute_pgm_rsrc2.USER_SGPR_COUNT to 6 bits. A descriptor with
; kernarg_segment_ptr (2 SGPRs) plus 30 kernarg-preload dwords is valid and
; encodes the count as 32. Decoding it through the older 5-bit field reads zero
; and falsely reports a KD inconsistency.

; CHECK-NOT: user-sgpr-layout-mismatch
; CHECK-LABEL: define amdgpu_kernel void @user_sgpr_count_32_gfx125_kernel(
; CHECK-SAME: i32 %arg0
; CHECK-SAME: i32 %arg29

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	user_sgpr_count_32_gfx125_kernel
	.p2align	8
	.type	user_sgpr_count_32_gfx125_kernel,@function
user_sgpr_count_32_gfx125_kernel:
; %bb.0:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel user_sgpr_count_32_gfx125_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 120
		.amdhsa_user_sgpr_count 32
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_kernarg_preload_length 30
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 35
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
      - { .offset: 0, .size: 120, .value_kind: by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 120
    .max_flat_workgroup_size: 32
    .name: user_sgpr_count_32_gfx125_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 35
    .symbol: user_sgpr_count_32_gfx125_kernel.kd
    .vgpr_count: 1
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
