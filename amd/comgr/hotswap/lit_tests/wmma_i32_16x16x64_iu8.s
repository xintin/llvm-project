; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native --emit-ir=wmma_i32_16x16x64_iu8_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for v_wmma_i32_16x16x64_iu8 (gfx1250 RDNA4 VOP3P opcode
; 0x072) lowered to gfx942 (CDNA3) via emitWMMAtoMFMA(..., IU8).
; See SemOp::V_WMMA_I32_16x16x64_IU8 in transpiler/semop.hpp; the
; matching dispatch in handle_valu_vop3p.cpp under the unified WMMA
; case block; the parameterised lowering body in
; transpiler/wmma_lowering.{hpp,cpp}.
;
; INVARIANTS PINNED:
;
;   1. Integer-accumulator MFMA intrinsic dispatched. The defining
;      marker is the per-MFMA-call intrinsic name
;      `mfma.i32.16x16x32.i8` — every other WMMA variant uses an
;      `mfma.f32.*` intrinsic. A regression that routed IU8 through
;      a wrong (or no) accumulator-type fork would surface as either
;      a wrong intrinsic name or a CreateCall type mismatch crash.
;
;   2. Accumulator pack type is `<4 x i32>`, not `<4 x float>`. The
;      IU8 path is the ONLY non-`<4 x float>` accumulator in the
;      WMMA-to-MFMA helper today; the inputType switch carries the
;      accumulator pack type through alongside the intrinsic.
;
;   3. The K=64 input is decomposed into 2 MFMA(K=32) calls per
;      virtual Wave32 group, with the accumulator chained
;      (mfma1 → mfma2). Two virtual Wave32 groups × 2 MFMAs each =
;      4 total MFMA calls in the lifted IR. Identical decomposition
;      shape to the FP8/BF8 siblings, validating the shared
;      redistribution path.
;
;   4. The CDNA3 i8 MFMA intrinsic takes `i64` for A/B (defined in
;      IntrinsicsAMDGPU.td §3560 before AMDGCN had a packed-i8
;      vector type at all). The lowering emits a
;      `bitcast <2 x i32> ... to i64` for each per-MFMA fragment to
;      match — same pack type as the FP8/BF8 siblings, divergent
;      only in the dispatched intrinsic name.
;
;   5. There is NO in-file `@llvm.amdgcn.strict.wwm*` marker around
;      the redistribute -> MFMA chain. Partial-wave correctness is
;      supplied kernel-wide by a single `@llvm.amdgcn.init_whole_wave`
;      call at function entry (emitted by
;      `WaveNativeProjection::emitInitialExec`), which forces
;      hardware EXEC = -1 for the remainder of the function while
;      preserving the source-modeled EXEC in the transpiler's
;      alloca for `emitUnderExec` side-effect gating. See
;      `hotswap/docs/wave-size-translation.md` §5.6.1 for the
;      register-allocator rationale behind this design.
;
; NEGATIVE PINS:
;
;   * NO `mfma.f32.*` calls in this kernel — those would mean IU8
;     dispatch silently fell through to a f32-accumulator handler.
;
;   * NO native `wmma.i32.16x16x64.iu8` intrinsic — target is gfx942
;     which does NOT have hasWMMA12; surface here would mean the
;     target-capability gate is broken.
;
;   * NO i32-acc MFMAs of WRONG K (mfma.i32.16x16x16i8, K=16) — the
;     dispatch must select the K=32 i8 MFMA (16x16x32) exactly.
;
;   * Exactly ONE `@llvm.amdgcn.init_whole_wave` call at function
;     entry — more than one would indicate a wave-projection bug;
;     zero would indicate the projection hook is not wired in.
;
;   * Zero `@llvm.amdgcn.strict.wwm*` calls anywhere — a stray
;     WWM marker would revert the lowering to the old regalloc-
;     bottlenecked shape.

; CHECK-LABEL: define amdgpu_kernel void @wmma_i32_16x16x64_iu8_kernel(

; Kernel-entry EXEC virtualisation: exactly one init_whole_wave
; call, emitted by `WaveNativeProjection::emitInitialExec`.
; CHECK: call i1 @llvm.amdgcn.init.whole.wave()

; Per-MFMA bitcast to i64 (CDNA3 i8 MFMA element type).
; CHECK: %{{.*}} = bitcast <2 x i32> %{{.*}} to i64

; The IU8-specific CDNA3 intrinsic, called with i64 sources and
; <4 x i32> accumulator. Pinning two separate MFMA chains (one per
; Wave32 virtual group).
;
; First group pass:
; CHECK: %mfma1 = call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x i32> %{{[^,]+}}, i32 0, i32 0, i32 0)
; CHECK: %mfma2 = call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x i32> %mfma1, i32 0, i32 0, i32 0)

; Second group pass (lane indices 32..63):
; CHECK: %mfma1{{[0-9]+}} = call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x i32> %{{[^,]+}}, i32 0, i32 0, i32 0)
; CHECK: %mfma2{{[0-9]+}} = call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x i32> %mfma1{{[0-9]+}}, i32 0, i32 0, i32 0)

; No in-file WWM markers anywhere; the kernel-entry init_whole_wave
; above is the sole hardware-EXEC-virtualisation signal. Only one
; init_whole_wave call per kernel.
; CHECK-NOT: call i32 @llvm.amdgcn.strict.wwm.i32(
; CHECK-NOT: call {{.*}} @llvm.amdgcn.strict.wwm
; CHECK-NOT: call {{.*}} @llvm.amdgcn.init.whole.wave

; Negative pin: NO f32-accumulator MFMA in this kernel. A regression
; that routed IU8 through the f32-accumulator switch arm would
; surface as `mfma.f32.16x16x32.fp8.fp8` (or any other f32 MFMA).
; CHECK-NOT: @llvm.amdgcn.mfma.f32.

; Negative pin: native gfx12 WMMA intrinsic must NOT appear (target
; is gfx942 which does NOT have hasWMMA12).
; CHECK-NOT: @llvm.amdgcn.wmma.i32.16x16x64.iu8

; Negative pin: must dispatch to the K=32 i8 MFMA, not the older
; K=16 i8 MFMA (the latter would imply wrong K-decomposition).
; CHECK-NOT: @llvm.amdgcn.mfma.i32.16x16x16i8


	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_i32_16x16x64_iu8_kernel
	.p2align	8
	.type	wmma_i32_16x16x64_iu8_kernel,@function
wmma_i32_16x16x64_iu8_kernel:
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
	v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23]
	s_clause 0x1
	global_store_b128 v24, v[20:23], s[28:29] offset:16
	global_store_b128 v24, v[16:19], s[28:29]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_i32_16x16x64_iu8_kernel
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
    .name:           wmma_i32_16x16x64_iu8_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .symbol:         wmma_i32_16x16x64_iu8_kernel.kd
    .vgpr_count:     25
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
