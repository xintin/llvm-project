; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=c2_ds_swizzle_unsafe_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR
;
; Negative case for the P6 lift after the audit follow-up that
; widened the safe set to all four valid swizzle-mode envelopes
; (QUAD_PERM / BITMASK_PERM / FFT_MODE / ROTATE_MODE). What remains
; refused is the RESERVED top-nibble envelope (top nibble in
; {0x9, 0xA, 0xB, 0xD, 0xF}) where AMDGPU SIDefines.h `Swizzle::
; EncBits` assigns no semantics — hardware behavior is undefined and
; a silent lift would map the source's imm to whatever the wave64
; backend happens to do.
;
; This complements c2_ds_swizzle.ll (positive case for the
; BITMASK_PERM/SWAP-1 corpus pattern) by pinning the safe/unsafe
; boundary exactly at the RESERVED envelope.

; The pre-translation abort uses the same
; cross-wave-shuffle-rewrite-pending failure as the original P6
; refusal — only the cause changed: previously every imm that wasn't
; QUAD_PERM/BITMASK_PERM was refused; now the gate fires only for the
; RESERVED top-nibble envelope.
; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-shuffle-rewrite-pending
; STDERR-SAME: ds_swizzle

; Classifier trace must surface the DsSwizzle site with `[pending]`
; and a detail line citing the RESERVED top-nibble (one of the
; broader "not a valid swizzle encoding" refusal categories). The
; exact imm value (0x9000) is asserted to pin the imm-extraction
; path against silent truncation.
; STDERR: DsSwizzle
; STDERR-SAME: Class 2
; STDERR: rewrite: P6
; STDERR-SAME: pending
; STDERR: detail: ds_swizzle_b32 imm 0x9000
; STDERR-SAME: not a valid swizzle encoding
; STDERR-SAME: RESERVED top-nibble
; STDERR-SAME: AMDGPU hardware semantics undefined
; STDERR: outcome: (c) refuse

; Final raise_cli refusal line — the `failed to raise` message must
; cite ds_swizzle by mnemonic so corpus-sweep tooling can bucket the
; failure correctly.
; STDERR: raise_cli: kernel 'c2_ds_swizzle_unsafe_kernel' failed to raise:
; STDERR-SAME: ds_swizzle

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c2_ds_swizzle_unsafe_kernel
	.p2align	8
	.type	c2_ds_swizzle_unsafe_kernel,@function
c2_ds_swizzle_unsafe_kernel:
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
	global_load_b32 v0, v1, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	ds_swizzle_b32 v0, v0 offset:0x9000
	s_wait_dscnt 0
	
	;;#ASMEND
	global_store_b32 v1, v0, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c2_ds_swizzle_unsafe_kernel
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
    .name:           c2_ds_swizzle_unsafe_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c2_ds_swizzle_unsafe_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
