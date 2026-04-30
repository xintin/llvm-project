; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_div_scale_f32_literal_numer_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift fixture for `v_div_scale_f32`'s literal-numerator encoding —
; the shape hipcc's AMDGPU backend emits for any `1.0 / x` fdiv
; expansion, where the numerator rides through src2 as an inline
; `1.0` constant rather than a register.
;
; The three-arm decoder in `handle_valu.cpp::V_DIV_SCALE_F32`
; distinguishes:
;   * (n, d, n)  with src0 == src2 → scale NUMERATOR    → flag=true.
;   * (d, d, n)  with src0 == src1 → scale DENOMINATOR  → flag=false.
;   * otherwise                    → loud refusal (unsupported shape).
;
; Equality is at the OPERAND level — register identity for register
; operands, immediate-bit-pattern equality for inline constants.
; Before this contract was in place, the handler's register-only
; gate fell through for literal-carried numerators and decoded both
; scale calls as `(src0, src1, false)`; the downstream div_fmas /
; div_fixup chain then collapsed to `1.0` bit-exact (observable as
; layer-norm's rstd reading `0x3f800000` regardless of input).
;
; This fixture pins four stable anchors:
;
;   (a) the POSITIVE shape: a scale-DENOM call whose numer is the
;       literal `1.000000e+00` and whose denom is a VGPR-sourced
;       value — the canonical `(1.0, %x, false)` lifted triple;
;   (b) the POSITIVE shape's scale-NUMER sibling in the same pair —
;       the canonical `(1.0, %x, true)` — asserting both arms of
;       the expanded IEEE fdiv are rewritten consistently;
;   (c) a NEGATIVE assertion against the pre-fix shape: neither
;       scale call should carry the numer as a VGPR duplicated into
;       src0 (i.e. `div.scale.f32(float %<vgpr>, float %<vgpr>, ...)` —
;       the (d, d, ...) form before my audit landed); and
;   (d) the resulting `div.fixup.f32` uses the canonical `1.0`
;       numerator (as the 3rd arg of the fixup), which is how the
;       AMDGPU backend surfaces the fdiv's numer after the Newton
;       iteration — a secondary sanity check that the chain stays
;       coherent end-to-end rather than just locally-correct on the
;       scale calls.
;
; Matching is substring-based on stable LLVM-IR landmarks so the
; fixture survives unrelated SSA-name churn in the raiser.

; CHECK-LABEL: define amdgpu_kernel void @v_div_scale_f32_literal_numer_kernel(

; (a) Scale DENOMINATOR arm of `1.0 / x`: canonical (1.0, vgpr, false).
; CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float 1.000000e+00, float %{{[^,]+}}, i1 false)

; (b) Scale NUMERATOR arm of the same fdiv: canonical (1.0, vgpr, true).
; CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float 1.000000e+00, float %{{[^,]+}}, i1 true)

; (d) The Newton-iteration fixup emits the `1.0` numerator in the
;     third arg position.  This anchors the chain's coherence: if a
;     future rewrite drops the literal-numer audit on the scale
;     calls but somehow keeps the fixup call, (a) and (b) fire and
;     (d) silently passes.  Pairing them guards against isolated
;     regressions on either side.
; CHECK: call float @llvm.amdgcn.div.fixup.f32(float %{{[^,]+}}, float %{{[^,]+}}, float 1.000000e+00)

; (c) Negative: in this fixture the only fdiv is `1.0 / sqrt(x)`,
;     so EVERY `div.scale.f32` call the kernel raises MUST carry
;     the literal `1.000000e+00` as its first argument.  The pre-
;     fix bug made both scale calls route the numerator through a
;     register (`(d, d, ...)` for scale-denom and `(n, d, false)`-
;     without-the-true-flag for scale-numer — the exact failure
;     mode the `(c)` negative assertion needs to rule out is "the
;     first argument to `div.scale.f32` is a register" for any call
;     site in this kernel).  The back-reference shape `float %[[X]],
;     float %[[X]]` that an earlier revision of this fixture used
;     would NOT catch the specific buggy IR this fixture guards
;     against — the pre-fix handler emitted `(%1414, %1415, ...)`
;     with two DIFFERENT SSA names that both `bitcast`ed the same
;     underlying VGPR value, so the two operand SSA names did not
;     literally match.  The stronger assertion below forbids ANY
;     register-first-arg `div.scale.f32` anywhere in this fixture's
;     IR — the correct regression guard for the literal-numer audit
;     in `handle_valu.cpp::V_DIV_SCALE_F32`.  If the fixture ever
;     grows a non-`1.0/x` divide (e.g. adds an `a/b` operation), this
;     assertion needs to be re-scoped (CHECK-NOT on the specific
;     kernel region, not the whole function).
; CHECK-NOT: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{[^,)]+}},

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_div_scale_f32_literal_numer_kernel
	.p2align	8
	.type	v_div_scale_f32_literal_numer_kernel,@function
v_div_scale_f32_literal_numer_kernel:   ; @v_div_scale_f32_literal_numer_kernel
; %bb.0:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_kmcnt 0x0
	flat_load_b32 v1, v0, s[6:7] scale_offset scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v2, 0x4f800000, v1
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v1, v1, v2, vcc_lo
	v_sqrt_f32_e32 v2, v1
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_nc_u32 v3, -1, v2 :: v_dual_add_nc_u32 v4, 1, v2
	v_fma_f32 v5, -v3, v2, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ge_f32_e64 s0, 0, v5
	v_dual_fma_f32 v6, -v4, v2, v1 :: v_dual_cndmask_b32 v2, v2, v3, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_lt_f32_e64 s0, 0, v6
	v_cndmask_b32_e64 v2, v2, v4, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v3, 0x37800000, v2
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo
	v_cmp_class_f32_e64 vcc_lo, v1, 0x260
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v1, v2, v1, vcc_lo
	v_div_scale_f32 v2, null, v1, v1, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v3, v2
	v_nop
	v_fma_f32 v4, -v2, v3, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v4, v3
	v_div_scale_f32 v4, vcc_lo, 1.0, v1, 1.0
	v_mul_f32_e32 v5, v4, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v2, v5, v4
	v_fmac_f32_e32 v5, v6, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v2, -v2, v5, v4
	v_div_fmas_f32 v2, v2, v3, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v1, v2, v1, 1.0
	global_store_b32 v0, v1, s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_div_scale_f32_literal_numer_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 7
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 3
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
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_div_scale_f32_literal_numer_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         v_div_scale_f32_literal_numer_kernel.kd
    .vgpr_count:     7
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
