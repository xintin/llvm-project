; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=v_mad_i32_i24_kernel 2>/dev/null | %FileCheck %s --check-prefix=DEFAULT
; RUN: %raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=v_mad_i32_i24_clamp_kernel 2>/dev/null | %FileCheck %s --check-prefix=CLAMP
; RUN: %raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=v_mad_u32_u24_clamp_kernel 2>/dev/null | %FileCheck %s --check-prefix=UCLAMP
;
; Lift test for signed 24-bit multiply-add:
;   dst = signext24(src0) * signext24(src1) + src2
; The unclamped form wraps naturally in i32 IR. The clamp form computes in i64
; and saturates to signed i32 range before truncating back to the destination.

; DEFAULT-LABEL: define amdgpu_kernel void @v_mad_i32_i24_kernel(
; DEFAULT: shl i32 %{{[^,]+}}, 8
; DEFAULT: ashr i32 %{{[^,]+}}, 8
; DEFAULT: %mad_i24_mul{{[0-9]*}} = mul i32
; DEFAULT: %mad_i24{{[0-9]*}} = add i32 %mad_i24_mul{{[0-9]*}},
; DEFAULT-NOT: %mad_i24_clamp

; CLAMP-LABEL: define amdgpu_kernel void @v_mad_i32_i24_clamp_kernel(
; CLAMP: sext i32 %{{[^ ]+}} to i64
; CLAMP: %mad_i24_mul_wide{{[0-9]*}} = mul i64
; CLAMP: %mad_i24_wide{{[0-9]*}} = add i64
; CLAMP: icmp slt i64 %{{[^,]+}}, -2147483648
; CLAMP: select i1 %{{[^,]+}}, i64 -2147483648
; CLAMP: icmp sgt i64 %{{[^,]+}}, 2147483647
; CLAMP: select i1 %{{[^,]+}}, i64 2147483647
; CLAMP: trunc i64 %mad_i24_clamp{{[0-9]*}} to i32

; UCLAMP-LABEL: define amdgpu_kernel void @v_mad_u32_u24_clamp_kernel(
; UCLAMP: and i32 %{{[^,]+}}, 16777215
; UCLAMP: zext i32 %{{[^ ]+}} to i64
; UCLAMP: %mad_u24_mul_wide{{[0-9]*}} = mul i64
; UCLAMP: %mad_u24_wide{{[0-9]*}} = add i64
; UCLAMP: icmp ugt i64 %{{[^,]+}}, 4294967295
; UCLAMP: select i1 %{{[^,]+}}, i64 4294967295
; UCLAMP: trunc i64 %mad_u24_clamp{{[0-9]*}} to i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_mad_i32_i24_kernel
	.p2align	8
	.type	v_mad_i32_i24_kernel,@function
v_mad_i32_i24_kernel:
	s_mov_b64 s[6:7], s[0:1]
	s_load_b128 s[0:3], s[6:7], 0x0
	s_load_b96 s[4:6], s[6:7], 0x10
	s_wait_kmcnt 0
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	v_mov_b32_e32 v2, s6
	v_mad_i32_i24 v3, v0, v1, v2
	v_mov_b32_e32 v4, 0
	global_store_b32 v4, v3, s[0:1] scale_offset
	s_endpgm

	.globl	v_mad_i32_i24_clamp_kernel
	.p2align	8
	.type	v_mad_i32_i24_clamp_kernel,@function
v_mad_i32_i24_clamp_kernel:
	s_mov_b64 s[6:7], s[0:1]
	s_load_b128 s[0:3], s[6:7], 0x0
	s_load_b96 s[4:6], s[6:7], 0x10
	s_wait_kmcnt 0
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	v_mov_b32_e32 v2, s6
	v_mad_i32_i24 v3, v0, v1, v2 clamp
	v_mov_b32_e32 v4, 0
	global_store_b32 v4, v3, s[0:1] scale_offset
	s_endpgm

	.globl	v_mad_u32_u24_clamp_kernel
	.p2align	8
	.type	v_mad_u32_u24_clamp_kernel,@function
v_mad_u32_u24_clamp_kernel:
	s_mov_b64 s[6:7], s[0:1]
	s_load_b128 s[0:3], s[6:7], 0x0
	s_load_b96 s[4:6], s[6:7], 0x10
	s_wait_kmcnt 0
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	v_mov_b32_e32 v2, s6
	v_mad_u32_u24 v3, v0, v1, v2 clamp
	v_mov_b32_e32 v4, 0
	global_store_b32 v4, v3, s[0:1] scale_offset
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_mad_i32_i24_kernel
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_mad_i32_i24_clamp_kernel
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_mad_u32_u24_clamp_kernel
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
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
      - { .offset:         8, .size:           4, .value_kind:     by_value }
      - { .offset:        12, .size:           4, .value_kind:     by_value }
      - { .offset:        16, .size:           4, .value_kind:     by_value }
      - { .offset:        20, .size:           4, .value_kind:     by_value }
      - { .offset:        24, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .max_flat_workgroup_size: 1024
    .name:           v_mad_i32_i24_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_mad_i32_i24_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .offset:         8, .size:           4, .value_kind:     by_value }
      - { .offset:        12, .size:           4, .value_kind:     by_value }
      - { .offset:        16, .size:           4, .value_kind:     by_value }
      - { .offset:        20, .size:           4, .value_kind:     by_value }
      - { .offset:        24, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .max_flat_workgroup_size: 1024
    .name:           v_mad_i32_i24_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_mad_i32_i24_clamp_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .offset:         8, .size:           4, .value_kind:     by_value }
      - { .offset:        12, .size:           4, .value_kind:     by_value }
      - { .offset:        16, .size:           4, .value_kind:     by_value }
      - { .offset:        20, .size:           4, .value_kind:     by_value }
      - { .offset:        24, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .max_flat_workgroup_size: 1024
    .name:           v_mad_u32_u24_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_mad_u32_u24_clamp_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
