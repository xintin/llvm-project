; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=c2_permlane16_kernel 2>/dev/null | %FileCheck %s
;
; The P2 rewrite (permlane16 / permlanex16 lift) has landed — see
; the permlane16/permlanex16 row of hotswap/docs/wave-size-
; translation.md §5.3. The classifier's LaneGroupShuffle site
; accepts
; V_PERMLANE16_B32 and V_PERMLANEX16_B32 as outcome (b) because
; `handle_valu_cross_lane.cpp` emulates both through
; `llvm.amdgcn.ds.bpermute`.
;
; Why the emulation rather than `llvm.amdgcn.permlane16` directly?
; `v_permlane16_b32` is an RDNA/gfx10+ instruction and is absent on
; CDNA (gfx942). Emitting the native intrinsic fails LLVM isel with
; "Cannot select: intrinsic %llvm.amdgcn.permlanex16" when the
; target is gfx942. `ds_bpermute_b32` is available on gfx8+, so
; emulating via bpermute keeps the lowering target-independent.
;
; This test asserts:
;   1. The raise succeeds (the classifier's LaneGroupShuffle site is
;      `[implemented]` for the non-swap variants).
;   2. The emitted IR contains a call to `llvm.amdgcn.ds.bpermute`
;      — one per `v_permlane16` / `v_permlanex16` input, though
;      CSE may merge identical calls if the fixture happens to
;      produce the same byte address.
;   3. The intrinsic is declared.
;
; We do NOT check the byte-address computation structurally (the
; per-lane lane-id + selector-nibble arithmetic surrounding the
; call) because that's codified in the handler's source-level
; comment block, and structural IR checks would be brittle across
; LLVM SSA-renumbering changes.

; CHECK-LABEL: define amdgpu_kernel void @c2_permlane16_kernel(

; The permlanex16 emulation's signature property is the group-swap
; XOR by 0x10 (= 16) on the computed group-base before the byte-
; address shift feeds into ds_bpermute. Matching the constant 16 in
; a `xor` instruction that then feeds (directly or indirectly) into
; the bpermute's index operand is the minimal check that pins the
; cross-16-lane-group semantics without asserting on SSA names.
;
; permlane16 (non-swap) must NOT have a `xor %..., 16` anywhere in
; its byte-address chain — FileCheck-NOT would be brittle here
; since the later permlanex16 DOES carry the xor, so we instead rely
; on the `ds_bpermute` call order: the first bpermute (permlane16)
; precedes the xor (which belongs to permlanex16's chain).

; First bpermute: the permlane16 (non-swap) call. Its byte-address
; chain has no `xor …, 16`.
; CHECK:      %{{permlane16_emu[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(

; After the first permlane16 emits, the permlanex16 group-swap
; xor with 0x10 (= 16) must appear somewhere before the second
; ds_bpermute call consumes its byte-address operand.
; CHECK:      xor i32 %{{[^,]+}}, 16

; Second bpermute: the permlanex16 call.
; CHECK:      %{{permlanex16_emu[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(

; The intrinsic declaration must be present.
; CHECK:      declare {{.*}}i32 @llvm.amdgcn.ds.bpermute(i32, i32)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c2_permlane16_kernel
	.p2align	8
	.type	c2_permlane16_kernel,@function
c2_permlane16_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
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
	v_mad_u32 v2, s0, s4, v0
	global_load_b32 v0, v2, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_permlane16_b32  v0,  v0, 0x76543210, 0x76543210 op_sel:[1,0]
	v_permlanex16_b32 v1, v0, 0x76543210, 0x76543210 op_sel:[1,0]
	
	;;#ASMEND
	v_add_nc_u32_e32 v0, v0, v1
	global_store_b32 v2, v0, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c2_permlane16_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
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
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           c2_permlane16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c2_permlane16_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
