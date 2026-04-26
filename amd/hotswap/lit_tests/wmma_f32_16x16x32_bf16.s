; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native --emit-ir=wmma_f32_16x16x32_bf16_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for v_wmma_f32_16x16x32_bf16 (gfx1250 RDNA4 VOP3P opcode
; 0x062) lowered to gfx942 (CDNA3) via emitWMMAtoMFMA(..., BF16).
; See SemOp::V_WMMA_F32_16x16x32_BF16 in transpiler/semop.hpp; the
; matching dispatch in handle_valu_vop3p.cpp under
; `case SemOp::V_WMMA_F32_16x16x32_F16: case SemOp::V_WMMA_F32_16x16x32_BF16:`;
; the parameterised lowering body in transpiler/wmma_lowering.{hpp,cpp}.
;
; Bidirectional handler ↔ test back-reference: the F16 sibling case
; in the same dispatch block is covered transitively by the existing
; Gfx1250Gpu.Matmul128x128 gtest; this fixture covers the BF16 fork.
;
; INVARIANTS PINNED:
;
;   1. The lift dispatches the BF16 SemOp through the parameterised
;      WMMAtoMFMA helper (NOT a fallback to F16 dispatch). The
;      defining marker is the per-MFMA-call intrinsic name
;      `mfma.f32.16x16x16bf16.1k` — a regression that routed BF16
;      through the F16 path would emit `mfma.f32.16x16x16f16`
;      instead and produce wrong-element-type accumulators.
;
;   2. The K=32 input is decomposed into 2 MFMA(K=16) calls per
;      virtual Wave32 group, with the accumulator chained
;      (mfma1 → mfma2). Two virtual Wave32 groups × 2 MFMAs each
;      = 4 total MFMA calls in the lifted IR. Anything fewer
;      indicates a missing K-tile or a missing group-pass; anything
;      more indicates spurious redundant emits.
;
;   3. The CDNA3 bf16 MFMA intrinsic takes <4 x i16> arguments
;      (defined in IntrinsicsAMDGPU.td §3521 before bfloat became
;      a first-class LLVM type). The lowering emits a `bitcast
;      ... to <4 x i16>` for each per-MFMA fragment to match.
;
;   4. The accumulator is <4 x float>, identical to the F16 sibling.
;      A regression that switched to a different accumulator width
;      would surface as `<N x float>` for N != 4 here.
;
;   5. There is NO in-file `@llvm.amdgcn.strict.wwm*` marker around
;      the redistribute -> MFMA chain. The Wave64-collective
;      correctness guarantee (lanes 32-63 execute the MFMA pipeline
;      even on a partial-wave `blockDim == 32` launch) is provided
;      kernel-wide by a single `@llvm.amdgcn.init_whole_wave` call
;      at function entry, emitted by
;      `WaveNativeProjection::emitInitialExec`. That intrinsic
;      forces hardware EXEC = -1 for the remainder of the function
;      and captures the original per-lane active mask into the
;      transpiler's EXEC alloca, so `emitUnderExec` still gates
;      side effects against the logical wave32 active lanes while
;      the redistribute / MFMA / collect chain runs Wave64-wide.
;      See `hotswap/docs/wave-size-translation.md` §5.6.1 for the
;      register-allocator rationale behind this design.
;
; NEGATIVE PINS:
;
;   * NO `mfma.f32.16x16x16f16` calls in this kernel — that would
;     mean BF16 dispatch silently fell through to F16. The F16
;     intrinsic is correct for `V_WMMA_F32_16x16x32_F16` (separate
;     dispatch) but emitting it for the bf16 SemOp is a hard bug.
;
;   * NO `mfma.f32.16x16x16bf16` (without the `_1k` suffix) —
;     the non-_1k variant exists for older CDNA targets and uses
;     a different K dimension; the lowering chose `_1k` because
;     that's the K=16 variant matching our 2× decomposition.
;
;   * Exactly ONE `@llvm.amdgcn.init_whole_wave` call at function
;     entry — more than one would indicate a wave-projection bug;
;     zero would indicate the projection hook is not wired into
;     `AllocaRegFile::init`.
;
;   * Zero `@llvm.amdgcn.strict.wwm*` calls anywhere — a stray
;     WWM marker would revert the lowering to the old regalloc-
;     bottlenecked shape (see §5.6.1).

; CHECK-LABEL: define amdgpu_kernel void @wmma_f32_16x16x32_bf16_kernel(

; Kernel-entry EXEC virtualisation: exactly one init_whole_wave
; call, emitted by `WaveNativeProjection::emitInitialExec`.
; CHECK: call i1 @llvm.amdgcn.init.whole.wave()

; Per-MFMA bitcast to <4 x i16> (CDNA3 bf16 MFMA element type).
; CHECK: %{{.*}} = bitcast <2 x i32> %{{.*}} to <4 x i16>

; The bf16-specific CDNA3 intrinsic, called with <4 x i16> sources
; and <4 x float> accumulator. Pinning two separate MFMA chains
; (one per Wave32 virtual group).
;
; First group pass:
; CHECK: %mfma1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %{{[^,]+}}, <4 x i16> %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)
; CHECK: %mfma2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %{{[^,]+}}, <4 x i16> %{{[^,]+}}, <4 x float> %mfma1, i32 0, i32 0, i32 0)

; Second group pass (lane indices 32..63):
; CHECK: %mfma1{{[0-9]+}} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %{{[^,]+}}, <4 x i16> %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)
; CHECK: %mfma2{{[0-9]+}} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %{{[^,]+}}, <4 x i16> %{{[^,]+}}, <4 x float> %mfma1{{[0-9]+}}, i32 0, i32 0, i32 0)

; No in-file WWM markers anywhere; the kernel-entry init_whole_wave
; above is the sole hardware-EXEC-virtualisation signal. Only one
; init_whole_wave call per kernel.
; CHECK-NOT: call i32 @llvm.amdgcn.strict.wwm.i32(
; CHECK-NOT: call {{.*}} @llvm.amdgcn.strict.wwm
; CHECK-NOT: call {{.*}} @llvm.amdgcn.init.whole.wave

; Negative pin: the F16 intrinsic must NOT appear in this kernel
; (would indicate BF16 dispatch fell through to F16).
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16f16

; Negative pin: the non-_1k bf16 intrinsic is the wrong K variant.
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16bf16(

; Negative pin: native gfx12 WMMA intrinsic must NOT appear (target
; is gfx942 which does NOT have hasWMMA12).
; CHECK-NOT: @llvm.amdgcn.wmma.f32.16x16x32.bf16


	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_f32_16x16x32_bf16_kernel
	.p2align	8
	.type	wmma_f32_16x16x32_bf16_kernel,@function
wmma_f32_16x16x32_bf16_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b128 s[24:27], s[0:1], 0x0
	s_load_b64 s[28:29], s[0:1], 0x10
	v_mov_b32_e32 v24, 0
	s_wait_kmcnt 0x0
	s_load_b256 s[0:7], s[24:25], 0x0
	s_load_b256 s[8:15], s[26:27], 0x0
	s_load_b256 s[16:23], s[28:29], 0x0
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[8:9], s[8:9]
	v_mov_b64_e32 v[16:17], s[16:17]
	v_mov_b64_e32 v[2:3], s[2:3]
	v_mov_b64_e32 v[4:5], s[4:5]
	v_mov_b64_e32 v[6:7], s[6:7]
	v_mov_b64_e32 v[10:11], s[10:11]
	v_mov_b64_e32 v[12:13], s[12:13]
	v_mov_b64_e32 v[14:15], s[14:15]
	v_mov_b64_e32 v[18:19], s[18:19]
	v_mov_b64_e32 v[20:21], s[20:21]
	v_mov_b64_e32 v[22:23], s[22:23]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23]
	s_clause 0x1
	global_store_b128 v24, v[20:23], s[28:29] offset:16
	global_store_b128 v24, v[16:19], s[28:29]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_f32_16x16x32_bf16_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 25
		.amdhsa_next_free_sgpr 30
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
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           wmma_f32_16x16x32_bf16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .symbol:         wmma_f32_16x16x32_bf16_kernel.kd
    .vgpr_count:     25
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
