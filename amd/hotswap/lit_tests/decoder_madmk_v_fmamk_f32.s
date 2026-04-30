; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=decoder_madmk_v_fmamk_f32_kernel 2>/dev/null | %FileCheck %s
;
; Decoder regression test for VOP2 MADMK form. Pins that the
; `driftCheckSrcN` MADMK exception (src0Idx < immIdx < src1Idx)
; survives. Without the exception, raise_cli would abort with
; "srcMap disagrees with OpName::srcN table for v_fmamk_f32" and
; this fixture would never produce IR for FileCheck to match.
;
; The decoder fix landed in commit ab25d0257a after the GPT-OSS
; `attn_fwd` corpus kernel hit the strict-check abort. This
; synthetic fixture isolates the regression detection from the
; corpus state — a future change that re-introduces the strict
; check would fail this lit test even if the corpus sweep wasn't
; run.

; CHECK-LABEL: define amdgpu_kernel void @decoder_madmk_v_fmamk_f32_kernel(

; The lifted IR must contain an FMA call with three float operands.
; The K=π literal (encoding 0x40490FDB) lands at the middle operand
; slot per the V_FMAMK_F32 handler convention (`srcF(0)=src0,
; srcF(1)=K, srcF(2)=src2` → `fma(s0, k, s2)`).
;
; We assert the IR shape (three float operands) and that the K
; literal appears SOMEWHERE in the IR, allowing for either of LLVM
; IR's two valid f32-constant renderings:
;   * decimal/hex-int form like `0x40490FDB` or `3.141593e+00`
;   * hex-double form like `0x400921FB60000000` (the f64 expansion
;     of pi-as-f32)
; LLVM has historically rendered f32 constants in either form
; depending on whether the value is exactly representable as a
; short decimal — pinning a single form would couple the test to
; the IR printer's heuristic. The shape check + K-literal-anywhere
; check is sufficient to catch:
;   * decoder regressions (no FMA call at all → fails CHECK)
;   * MADMK srcMap reordering (K imm in wrong operand slot →
;     fails the K-literal check below if K stops being inline)
;   * imm extraction truncation/sign-extension (K bit pattern
;     wrong → fails the K-literal check below)
;
; Match the FMA call with the K=π literal at the middle (second)
; operand slot — that's where the V_FMAMK_F32 handler routes the
; K-imm per its `srcF(0)=src0, srcF(1)=K, srcF(2)=src2` convention
; (handle_valu.cpp). The literal can render in either of LLVM IR's
; valid f32-constant forms:
;   * hex-int form like `0x40490FDB`
;   * f64-expanded hex-double form like `0x400921FB60000000` (the
;     f64 expansion of pi-as-f32; LLVM's IR printer chooses this
;     when the value isn't exactly representable as a short
;     decimal — which is the case for pi)
; A `3\.14159` decimal-float form is also accepted in case a future
; LLVM defaults to that. Pinning a single form would couple the
; test to the IR printer's heuristic; matching any of the three
; correct renderings is sufficient to catch:
;   * decoder regressions (no FMA call at all → CHECK fails)
;   * MADMK srcMap reordering (K imm in wrong operand slot → the
;     literal wouldn't be in the second-operand position → CHECK
;     fails because the constraint is positional)
;   * imm extraction truncation/sign-extension (K bit pattern wrong
;     → none of the three alternatives match → CHECK fails)
;
; CHECK: call {{.*}}float @llvm.fma.f32(float {{.*}}, float {{0x40490FDB|0x400921FB60000000|3\.14159[0-9]+}}, float {{.*}})

; The intrinsic declaration must be present.
; CHECK: declare {{.*}}float @llvm.fma.f32(float, float, float)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	decoder_madmk_v_fmamk_f32_kernel
	.p2align	8
	.type	decoder_madmk_v_fmamk_f32_kernel,@function
decoder_madmk_v_fmamk_f32_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b32 s8, s[0:1], 0x24
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_and_b32 s9, ttmp6, 15
	s_add_co_i32 s10, s2, 1
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x10
	s_wait_xcnt 0x0
	s_mul_i32 s0, ttmp9, s10
	s_getreg_b32 s1, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s9, s9, s0
	s_wait_kmcnt 0x0
	s_and_b32 s0, s8, 0xffff
	s_cmp_eq_u32 s1, 0
	s_cselect_b32 s1, ttmp9, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_u32 v2, s1, s0, v0
	s_clause 0x1
	global_load_b32 v0, v2, s[6:7] scale_offset
	global_load_b32 v1, v2, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_fmamk_f32 v0, v0, 0x40490fdb, v1
	
	;;#ASMEND
	global_store_b32 v2, v0, s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel decoder_madmk_v_fmamk_f32_kernel
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 11
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
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
      - { .address_space:  global, .offset:         16, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .max_flat_workgroup_size: 1024
    .name:           decoder_madmk_v_fmamk_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .symbol:         decoder_madmk_v_fmamk_f32_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
