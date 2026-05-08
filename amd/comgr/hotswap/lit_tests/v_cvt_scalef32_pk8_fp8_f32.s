; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx1250 \
; RUN:     --emit-ir=v_cvt_scalef32_pk8_fp8_f32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=NATIVE
;
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_cvt_scalef32_pk8_fp8_f32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=CROSS
;
; Lift test for v_cvt_scalef32_pk8_fp8_f32 (gfx1250-only VOP3,
; opcode 0x2c3, profile VOP_V2I32_V8F32_F32 in
; VOP3Instructions.td:1883).  Takes 8 f32 inputs and a scalar f32
; scale multiplier, returns 2x i32 (8 packed FP8 bytes).
; Handler in transpiler/handle_valu.cpp under
; `if (sop == CanonicalOp::V_CVT_SCALEF32_PK8_FP8_F32)`.
;
; Two dispatch arms:
;   - hasTensorOps (gfx1250 same-target): emit
;     `int_amdgcn_cvt_scalef32_pk8_fp8_f32` directly.
;   - hasFP8ConversionInsts (e.g. gfx942/gfx950): software-emulate via
;     splat-multiply then four chained `int_amdgcn_cvt_pk_fp8_f32`
;     calls assembling the 2-dword result with appropriate
;     dword/word-select bits.

; ── Same-target arm: native gfx1250 intrinsic ───────────────────────

; NATIVE-LABEL: define amdgpu_kernel void @v_cvt_scalef32_pk8_fp8_f32_kernel(

; The handler emits the native intrinsic with name "cvt_scalef32_pk8_fp8".
; NATIVE: %cvt_scalef32_pk8_fp8{{[0-9]*}} = call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.f32(<8 x float> %{{.*}}, float %{{.*}})

; Negative: no software-emulation chain on the same-target path.
; NATIVE-NOT: call {{.*}}@llvm.amdgcn.cvt.pk.fp8.f32

; ── Cross-target gfx942/gfx950 arm: software emulation chain ────────

; CROSS-LABEL: define amdgpu_kernel void @v_cvt_scalef32_pk8_fp8_f32_kernel(

; The handler emits a vector splat of the scale (LLVM's
; CreateVectorSplat lowers to insertelement + shufflevector with
; `.splatinsert` / `.splat` suffixes), fmul against the source
; <8 x float>, then four chained pk_fp8 calls named pk_fp8_01 /
; pk_fp8_23 / pk_fp8_45 / pk_fp8_67 (verbatim from the handler's
; IRBuilder names).  The lo-half calls (pk_fp8_01 / pk_fp8_45)
; seed `i32 0` and `i1 false`; the hi-half calls (pk_fp8_23 /
; pk_fp8_67) chain through the previous lo-half result with `i1
; true`, assembling each output dword in two halves.  Match by
; instruction shape rather than by anonymous-temp number so any
; future renumbering doesn't cause spurious churn.
; CROSS-DAG: insertelement <8 x float> poison, float %{{.+}}, i64 0
; CROSS-DAG: shufflevector <8 x float> %{{.+}}, <8 x float> poison, <8 x i32> zeroinitializer
; CROSS-DAG: %scaled = fmul <8 x float>
; CROSS-DAG: %pk_fp8_01 = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %{{.+}}, float %{{.+}}, i32 0, i1 false)
; CROSS-DAG: %pk_fp8_23 = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %{{.+}}, float %{{.+}}, i32 %pk_fp8_01, i1 true)
; CROSS-DAG: %pk_fp8_45 = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %{{.+}}, float %{{.+}}, i32 0, i1 false)
; CROSS-DAG: %pk_fp8_67 = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %{{.+}}, float %{{.+}}, i32 %pk_fp8_45, i1 true)

; Negative: no native gfx1250 intrinsic on the cross-target path
; (would mean the dispatch silently mis-fired).
; CROSS-NOT: @llvm.amdgcn.cvt.scalef32.pk8.fp8.f32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cvt_scalef32_pk8_fp8_f32_kernel
	.p2align	8
	.type	v_cvt_scalef32_pk8_fp8_f32_kernel,@function
v_cvt_scalef32_pk8_fp8_f32_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp8_f32 v[0:1], v[2:9], v10
	;;#ASMEND
	global_store_b64 v11, v[0:1], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_scalef32_pk8_fp8_f32_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 4
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           v_cvt_scalef32_pk8_fp8_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_cvt_scalef32_pk8_fp8_f32_kernel.kd
    .vgpr_count:     12
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
