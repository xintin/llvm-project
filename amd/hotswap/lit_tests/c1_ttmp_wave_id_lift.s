; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=c1_ttmp_wave_id_lift_kernel 2>/dev/null | %FileCheck %s --check-prefix=IR
;
; Regression-fence for the Class-1 "wave_id lift" rescue
; (hotswap/docs/wave-size-translation.md §5.6.2, §6 Class-1).
;
; The canonical HIP-emitted pattern `s_bfe_u32 sDST, ttmp8, 0x50019`
; extracts `wave_id_in_workgroup` from bits [29:25] of ttmp8. Under
; cross-widening the generic BFE handler + SGPR alloca round-trip
; loses per-source-wave divergence (the formally-scalar BFE → SGPR
; shape collapses to `readfirstlane` during codegen, so all target
; lanes read one lane's wave_id). Before the lift this surfaced as
; a checkerboard miscompile in every Tensile / rocBLAS matmul (each
; Wave64 target wave covers two source waves but wrote to only one
; tile).
;
; The rescue in `handle_sop2.cpp` detects the exact
; `(src0 == ttmp8, src1 == imm 0x50019)` shape in the `S_BFE_U32`
; handler and emits the architectural expression directly from
; `@llvm.amdgcn.workitem.id.x` — a divergent-leaf intrinsic the
; backend keeps in a VGPR per lane. Downstream consumers of the
; destination SGPR see a divergent VGPR, so per-source-wave
; tile-offset arithmetic stays correct across cross-widening.
;
; We assert:
;   1) The raise succeeds (no %not; the RUN line errors the test
;      if raise_cli returns non-zero).
;   2) The kernel body is present (CHECK-LABEL anchors on it).
;   3) The lifted IR carries the canonical three-instruction shape
;      — `@llvm.amdgcn.workitem.id.x`, `lshr i32 ..., 5`
;      (log2(W_src=32) = 5), `and i32 ..., 31` — with 31 as the
;      five-bit mask from the 0x50019 immediate's width=5 field.
;      The three are asserted in decoded order; intermediate names
;      are regex-matched (%"..." or bare identifiers) to stay
;      stable across LLVM IR-printer versions.
;
; Paired with the lit_tests/c1_lane_id_leak v_mbcnt_hi fixture,
; which pins the *unrewritable* Class-1 leak (refuses). Together
; they pin both outcomes for Class-1 obstructions: rescue
; (this fixture, `isCanonicalWaveIdBfe == true` in
; `wave_size_obstruction.cpp` skips the refusal) vs. refuse
; (c1_lane_id_leak, no rescue path on paper).

; IR-LABEL: define amdgpu_kernel void @c1_ttmp_wave_id_lift_kernel(
; IR: call i32 @llvm.amdgcn.workitem.id.x()
; IR: lshr i32 {{.*}}, 5
; IR: and i32 {{.*}}, 31

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c1_ttmp_wave_id_lift_kernel
	.p2align	8
	.type	c1_ttmp_wave_id_lift_kernel,@function
c1_ttmp_wave_id_lift_kernel:            ; @c1_ttmp_wave_id_lift_kernel
; %bb.0:
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
	v_mad_u32 v0, s0, s4, v0
	;;#ASMSTART
	s_bfe_u32 s0, ttmp8, 0x50019
	
	;;#ASMEND
	v_mov_b32_e32 v1, s0
	global_store_b32 v0, v1, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c1_ttmp_wave_id_lift_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
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
    .name:           c1_ttmp_wave_id_lift_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c1_ttmp_wave_id_lift_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
