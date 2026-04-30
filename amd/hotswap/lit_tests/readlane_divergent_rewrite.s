; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --enable-writelane-rewrite \
; RUN:     --emit-ir=readlane_divergent_rewrite_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=REWRITE
;
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --disable-writelane-rewrite \
; RUN:     --emit-ir=readlane_divergent_rewrite_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=UNCHANGED
;
; Divergence-triggered `v_readlane_b32` rewrite contract.
;
; REWRITE path (--enable-writelane-rewrite on):
;   * `rewriteCrossLaneDivergent` replaces the
;     `@llvm.amdgcn.readlane` call with a `ds_bpermute` whose selector
;     is `((lane_id & ~(W_s-1)) | lane_idx) << 2`. The lane_id helper
;     is the canonical `mbcnt_lo(-1, 0); mbcnt_hi(-1, prev)` pair built
;     once at the entry block.
;   * Asserts the four stable signatures, all keyed on
;     rewriter-specific `cwd_*` variable names (disambiguates from the
;     raiser's SPE-wrapper lane_id construction around writelane /
;     readlane primitive sites):
;       (a) the `cwd_lane_id_lo` + `cwd_lane_id` two-step mbcnt pair,
;       (b) the `cwd_rl_selector` byte-offset (shl by 2),
;       (c) the `cwd_readlane_rewritten` ds_bpermute call,
;       (d) the absence of any `cwd_writelane_rewritten` sibling
;           (write half must NOT fire on this fixture).
;
; UNCHANGED path (flag off): `@llvm.amdgcn.readlane` survives; the
; rewriter-emitted `cwd_*` values are absent. Commit 1's default-off
; invariant. The raiser's own `ds_bpermute` lowerings for other
; opcodes (e.g. the dedicated DS_BPERMUTE handler exercised by
; `ds_bpermute_b32`) remain unrelated to this contract; we key the
; negative assertion on the rewriter's `cwd_readlane_rewritten` name
; rather than on `ds_bpermute` to keep the fixture robust against
; corpus shifts.

; REWRITE-LABEL: define amdgpu_kernel void @readlane_divergent_rewrite_kernel(
; REWRITE: %cwd_lane_id_lo = call i32 @llvm.amdgcn.mbcnt.lo
; REWRITE: %cwd_lane_id = call i32 @llvm.amdgcn.mbcnt.hi
; REWRITE: %cwd_rl_selector = shl
; REWRITE: %cwd_readlane_rewritten = call i32 @llvm.amdgcn.ds.bpermute
; REWRITE-NOT: cwd_writelane_rewritten

; UNCHANGED-LABEL: define amdgpu_kernel void @readlane_divergent_rewrite_kernel(
; UNCHANGED: call i32 @llvm.amdgcn.readlane
; UNCHANGED-NOT: cwd_lane_id_lo
; UNCHANGED-NOT: cwd_readlane_rewritten

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	readlane_divergent_rewrite_kernel
	.p2align	8
	.type	readlane_divergent_rewrite_kernel,@function
readlane_divergent_rewrite_kernel:      ; @readlane_divergent_rewrite_kernel
; %bb.0:
	s_load_b32 s2, s[0:1], 0x14
	s_bfe_u32 s3, ttmp6, 0x4000c
	s_wait_xcnt 0x0
	s_load_b64 s[0:1], s[0:1], 0x0
	s_add_co_i32 s3, s3, 1
	s_and_b32 s4, ttmp6, 15
	s_mul_i32 s3, ttmp9, s3
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s4, s4, s3
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s3, ttmp9, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_u32 v1, s3, s2, v0
	;;#ASMSTART
	v_readlane_b32 s2, v1, 0
	
	;;#ASMEND
	v_mov_b32_e32 v0, s2
	global_store_b32 v1, v0, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel readlane_divergent_rewrite_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           readlane_divergent_rewrite_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         readlane_divergent_rewrite_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
