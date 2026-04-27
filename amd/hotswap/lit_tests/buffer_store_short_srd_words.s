; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=buffer_store_short_sentinel_srd_kernel 2>/dev/null | %FileCheck %s --check-prefix=SENTINEL
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=buffer_store_short_finite_srd_kernel 2>/dev/null | %FileCheck %s --check-prefix=FINITE
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=buffer_store_short_allones_srd_kernel 2>/dev/null | %FileCheck %s --check-prefix=ALLONES
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=buffer_store_short_ambiguous_srd_kernel 2>/dev/null | %FileCheck %s --check-prefix=AMBIG
;
; GPT-OSS decode_attention._fwd_kernel_stage2 stores its bf16 output through
; gfx12 `buffer_store_b16`.  On gfx942 the raw-buffer descriptor's RSRC2/RSRC3
; must be target-normalised, but not by widening every source bound:
;
;   * gfx1250 Triton's raw-pointer max sentinel, RSRC2=0x00ffffff, maps to
;     gfx942's native raw-buffer max sentinel, 0x7ffffffe when the descriptor
;     has raw-pointer shape, including gfx1250's high word1 marker bits.
;   * finite NUM_RECORDS values stay finite, so source-OOB accesses still
;     zero/drop on the target instead of becoming real memory accesses.
;   * true all-ones NUM_RECORDS remains all-ones / OOB-disabled.
;   * ambiguous constant non-raw descriptor shapes do not map the sentinel.
;   * RSRC3 uses gfx942's native FORMAT_32 + NUM_FORMAT_FLOAT shape, 0x27000.

; SENTINEL-LABEL: define amdgpu_kernel void @buffer_store_short_sentinel_srd_kernel(
; SENTINEL: icmp eq i32 {{.*}}16777215
; SENTINEL: select i1 {{.*}}, i32 2147483646, i32 16777215
; SENTINEL-NOT: insertelement <4 x i32> {{.*}}, i32 131072, i64 3
; SENTINEL: insertelement <4 x i32> {{.*}}, i32 159744, i64 3
; SENTINEL: call void @llvm.amdgcn.raw.buffer.store.i16(

; FINITE-LABEL: define amdgpu_kernel void @buffer_store_short_finite_srd_kernel(
; FINITE: icmp eq i32 {{.*}}16777215
; FINITE: select i1 {{.*}}, i32 2147483646, i32 4096
; FINITE-NOT: insertelement <4 x i32> {{.*}}, i32 131072, i64 3
; FINITE: insertelement <4 x i32> {{.*}}, i32 159744, i64 3
; FINITE: call void @llvm.amdgcn.raw.buffer.store.i16(

; ALLONES-LABEL: define amdgpu_kernel void @buffer_store_short_allones_srd_kernel(
; ALLONES: icmp eq i32 {{.*}}16777215
; ALLONES: select i1 {{.*}}, i32 2147483646, i32 {{(-1|4294967295)}}
; ALLONES-NOT: insertelement <4 x i32> {{.*}}, i32 131072, i64 3
; ALLONES: insertelement <4 x i32> {{.*}}, i32 159744, i64 3
; ALLONES: call void @llvm.amdgcn.raw.buffer.store.i16(

; AMBIG-LABEL: define amdgpu_kernel void @buffer_store_short_ambiguous_srd_kernel(
; AMBIG: icmp eq i32 1, 0
; AMBIG: icmp eq i32 1, 131072
; AMBIG: icmp eq i32 1, 147456
; AMBIG: icmp eq i32 1, 159744
; AMBIG: select i1 {{.*}}, i32 2147483646, i32 16777215
; AMBIG-NOT: select i1 true, i32 2147483646, i32 16777215
; AMBIG: call void @llvm.amdgcn.raw.buffer.store.i16(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl	buffer_store_short_sentinel_srd_kernel
	.p2align	8
	.type	buffer_store_short_sentinel_srd_kernel,@function
buffer_store_short_sentinel_srd_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0x1234
	s_or_b32 s1, s1, 0xfc000000
	s_mov_b32 s3, 0
	s_mov_b32 s2, 0xffffff
	s_wait_kmcnt 0x0
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

	.globl	buffer_store_short_finite_srd_kernel
	.p2align	8
	.type	buffer_store_short_finite_srd_kernel,@function
buffer_store_short_finite_srd_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0x1234
	s_mov_b32 s3, 0
	s_mov_b32 s2, 4096
	s_wait_kmcnt 0x0
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

	.globl	buffer_store_short_allones_srd_kernel
	.p2align	8
	.type	buffer_store_short_allones_srd_kernel,@function
buffer_store_short_allones_srd_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0x1234
	s_mov_b32 s3, 0
	s_mov_b32 s2, -1
	s_wait_kmcnt 0x0
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

	.globl	buffer_store_short_ambiguous_srd_kernel
	.p2align	8
	.type	buffer_store_short_ambiguous_srd_kernel,@function
buffer_store_short_ambiguous_srd_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0x1234
	s_mov_b32 s3, 1
	s_mov_b32 s2, 0xffffff
	s_wait_kmcnt 0x0
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_store_short_sentinel_srd_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

	.amdhsa_kernel buffer_store_short_finite_srd_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

	.amdhsa_kernel buffer_store_short_allones_srd_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

	.amdhsa_kernel buffer_store_short_ambiguous_srd_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
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
    .name:           buffer_store_short_sentinel_srd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_short_sentinel_srd_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           buffer_store_short_finite_srd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_short_finite_srd_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           buffer_store_short_allones_srd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_short_allones_srd_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           buffer_store_short_ambiguous_srd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_short_ambiguous_srd_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
