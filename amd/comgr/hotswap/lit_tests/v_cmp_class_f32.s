; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_cmp_class_f32_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for v_cmp_class_f32_e64 (VOPC, floating-point IEEE class
; predicate). Pins the dual contract that completes the parser-level
; gap closure in `parseVCmpPseudoName`:
;
;   1. The MC pseudo `V_CMP_CLASS_F32_e64` parses as the class
;      special-case (not an FCmp predicate compare); the handler in
;      `handle_valu_vcmp.cpp` takes the `if (m->isClass)` branch
;      and emits `llvm.amdgcn.class.f32`, NOT a CreateFCmp.
;   2. The wave-mask write-back to the SGPR-pair destination shares
;      the same `ballot` + `trunc` discipline as the V_CMP/V_CMPX
;      predicate compares (the reused `cmp` -> `vcmp_ballot` tail
;      in handle_valu_vcmp.cpp). The cross-wave wave32 -> wave64
;      direction here pins the `i64 -> i32` truncation step.
;
; The mask is the literal `i32 512` (= 0x200, bit 9 = +inf), threaded
; through to the intrinsic call without any rewriting; if a future
; change re-encodes class masks we want a loud failure here, not a
; silent one. The companion fixture `v_cmpx_ballot` covers the
; predicate-compare side of the same wave-mask plumbing.

; CHECK-LABEL: define amdgpu_kernel void @v_cmp_class_f32_kernel(

; The class call: per-lane i1 result, FP source bitcast to f32, mask
; threaded through as the literal `i32 512`. The `vclass` name is
; pinned by the handler (search for `"vclass"` in handle_valu_vcmp.cpp).
; CHECK: %vclass{{[0-9]*}} = call i1 @llvm.amdgcn.class.f32(float %{{[^,]+}}, i32 512)

; The wave-mask write-back, shared with the predicate-compare path:
; the i1 feeds amdgcn.ballot.i64 (target wave64), the i64 result is
; truncated back to the source's wave32 execTy (i32), and the
; truncated mask is then stored to the SGPR alloca. `vcmp_ballot` /
; `vcmp_ballot_trunc` are the names pinned by the V_CMP -> SGPR
; branch in handle_valu_vcmp.cpp (same identifiers asserted by the
; v_cmpx_ballot fixture).
; CHECK: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %vclass{{[0-9]*}})
; CHECK-NEXT: %vcmp_ballot_trunc = trunc i64 %vcmp_ballot to i32

; Negative assertions: the lift MUST NOT take the FCmp path (which
; would compare the float operand to the i32 mask reinterpreted as a
; float — silently miscompiling the entire kernel). It also MUST NOT
; sext the i1 directly into the SGPR — that's the pre-fix shape from
; the v_cmpx_ballot regression and would re-introduce the divergent
; SSA value into a wave-mask consumer.
; CHECK-NOT: fcmp {{.*}} float %{{[^,]+}}, {{.*}}i32
; CHECK-NOT: sext i1 %vclass{{[0-9]*}} to i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cmp_class_f32_kernel
	.p2align	8
	.type	v_cmp_class_f32_kernel,@function
v_cmp_class_f32_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	global_load_b32 v1, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_cmp_class_f32_e64 s4, v1, 0x200
	s_mov_b32 s2, s4
	
	;;#ASMEND
	v_mov_b32_e32 v1, s2
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cmp_class_f32_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 5
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
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_cmp_class_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         v_cmp_class_f32_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
