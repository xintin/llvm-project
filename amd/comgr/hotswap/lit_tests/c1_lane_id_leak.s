; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=c1_lane_id_leak_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR
;
; hotswap/docs/wave-size-translation.md §6 Class 1:
; v_mbcnt_hi_u32_b32 on a wave32 source binary leaks the absolute
; target-hardware lane position into an observable value whenever
; the raised IR runs on a wider target wave. No rewrite in §7's
; unrewritable table recovers the original wave32 semantics (the
; source never named "lane_id mod W_s" as a distinct quantity), so
; the only correct outcome is the (c) refusal branch of §7's
; 3-outcome decision procedure.
;
; The classifier must flag the v_mbcnt_hi site at raise time and
; abort with the stable diagnostic substrings asserted below.
; Matching on substrings rather than the full sentence keeps the
; test resilient to future rewordings.
;
; We also assert `%not` inverts the exit code, so the test fails
; loudly if the abort silently degrades into a warning.

; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-lane-id-leak
; STDERR-SAME: v_mbcnt_hi

; The per-site trace emitted after the abort line names the
; ObstructionKind in human-readable form and includes the
; Class 1..4 cross-reference (wave-size-translation.md §6)
; parenthetically. We key on stable substrings only.
; STDERR: MbcntHiLaneIdLeak
; STDERR-SAME: Class 1
; STDERR: outcome: (c) refuse

; The raise_cli wrapper reports the failure once more in its
; kerneldex-style format so coverage tooling bucket on it.
; STDERR: raise_cli: kernel 'c1_lane_id_leak_kernel' failed to raise:
; STDERR-SAME: v_mbcnt_hi

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c1_lane_id_leak_kernel
	.p2align	8
	.type	c1_lane_id_leak_kernel,@function
c1_lane_id_leak_kernel:
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
	v_mad_u32 v1, s0, s4, v0
	;;#ASMSTART
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	
	;;#ASMEND
	global_store_b32 v1, v0, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c1_lane_id_leak_kernel
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
    .name:           c1_lane_id_leak_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c1_lane_id_leak_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
