; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=c2_dpp_quad_perm_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; The P5 rewrite (DPP modifier intrinsic lift) has landed — see
; the DPP row of hotswap/docs/wave-size-translation.md §5.3.  This
; test was originally a refuse-loud fixture asserting
; `cross-wave-shuffle-rewrite-pending`, then flipped to a positive
; faithful-lift fixture asserting `llvm.amdgcn.update.dpp.i32` in
; the emitted IR.
;
; A subsequent pass (`rewrite_cross_lane_divergent.cpp`) brought
; DPP under the same cross-widening rewrite invariant as
; writelane / readlane / permlane16 / permlanex16: under cross-
; widening (wave32 -> wave64) every cross-lane primitive is
; rewritten to a `ds_bpermute + select` shape whose correctness
; depends only on ds_bpermute's explicit per-lane read semantics
; (stable across gfx9+) rather than on target ISA and source ISA
; sharing the same bank_mask / row_mask interpretation.  This
; test's source kernel targets gfx1250 (wave32) and is raised
; against gfx942 (wave64), so the rewrite fires.
;
; The DPP modifier values in the fixture are
; `quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1` —
; which encode to `dpp_ctrl = 0xB1 = 177`, `row_mask = 0xF = 15`,
; `bank_mask = 0xF = 15`, `bound_ctrl = true`.  The
; `quad_perm:[1,0,3,2]` pattern is a within-4-lane swap pair: lane
; 0 reads lane 1, lane 1 reads lane 0, lane 2 reads lane 3, lane 3
; reads lane 2.  In the rewrite, `withinRow = lane & 0xF`,
; `quadBase = withinRow & ~3`, `quadWithin = withinRow & 3`,
; `selector = (0xB1 >> (quadWithin * 2)) & 3`; lane 0 decodes
; `selector = 0xB1 & 3 = 1`, lane 1 decodes `(0xB1 >> 2) & 3 = 0`,
; etc.  srcLaneAbs = rowBase | (quadBase | selector).
;
; Since both row_mask and bank_mask are 0xF (every target lane
; active), the per-lane mask-gating select collapses to the
; dpp-value select (see the `rowMaskImm == 0xF && bankMaskImm ==
; 0xF` shortcut in rewrite_cross_lane_divergent.cpp).  And since
; quad_perm is always in-range (every 2-bit selector resolves to
; a valid lane within the quad), the `inRange` predicate folds to
; `i1 true` and the bperm result flows directly through.  What
; survives in the IR is: lane-topology mbcnt / and / or / shl
; scaffolding plus one `ds_bpermute` call.

; CHECK-LABEL: define amdgpu_kernel void @c2_dpp_quad_perm_kernel(

; The faithful-lift `update.dpp` call MUST NOT appear in the
; emitted IR under cross-widening — the rewrite replaces every
; i32 update.dpp with a ds_bpermute + select chain.  Only the
; intrinsic declaration remains (inserted by other passes); the
; rewrite's `CreateCall` of `@llvm.amdgcn.ds.bpermute` does not
; re-insert `update.dpp`.
; CHECK-NOT: call i32 @llvm.amdgcn.update.dpp.i32(

; The rewritten shape is built from lane-topology primitives:
; within-row, quad-base, quad-within, 2-bit-selector, row-base.
; These name the per-lane mapping for quad_perm:[1,0,3,2].
; CHECK-DAG: %cwd_dpp_within_row = and i32 %{{.+}}, 15
; CHECK-DAG: %cwd_dpp_row_base = and i32 %{{.+}}, -16
; CHECK-DAG: %cwd_dpp_quad_base = and i32 %cwd_dpp_within_row, -4
; CHECK-DAG: %cwd_dpp_quad_within = and i32 %cwd_dpp_within_row, 3
; CHECK-DAG: %cwd_dpp_quad_shift = shl i32 %cwd_dpp_quad_within, 1
; CHECK-DAG: lshr i32 177, %cwd_dpp_quad_shift
; CHECK-DAG: %cwd_dpp_quad_sel = and i32 %{{.+}}, 3
; CHECK-DAG: %cwd_dpp_quad_src = or i32 %cwd_dpp_quad_base, %cwd_dpp_quad_sel
; CHECK-DAG: %cwd_dpp_src_abs = or i32 %cwd_dpp_row_base, %{{.+}}
; CHECK-DAG: %cwd_dpp_selector = shl i32 %cwd_dpp_src_abs, 2

; The ds_bpermute call carries the per-lane selector and the
; source value unchanged.  The declaration below is the only
; `amdgcn.update.dpp.i32` symbol that can remain — it is inserted
; by passes unrelated to the rewrite and is safely unused.
; CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 %cwd_dpp_selector, i32 %{{[^,]+}})

; Declaration of the bpermute intrinsic (emitted by the rewrite
; when it inserts the call).  Overloaded on i32 by fixed-shape.
; CHECK: declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c2_dpp_quad_perm_kernel
	.p2align	8
	.type	c2_dpp_quad_perm_kernel,@function
c2_dpp_quad_perm_kernel:                ; @c2_dpp_quad_perm_kernel
; %bb.0:
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x14
	s_load_b64 s[2:3], s[0:1], 0x0
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s0, ttmp9, s1
	v_mad_u32 v0, s0, s4, v0
	global_load_b32 v1, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_mov_b32_dpp v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	
	;;#ASMEND
	global_store_b32 v0, v1, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c2_dpp_quad_perm_kernel
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
    .name:           c2_dpp_quad_perm_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c2_dpp_quad_perm_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
