; RUN: %raise_cli %wmma_f32_16x16x4_f32_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --disable-wave-native \
; RUN:     --emit-ir=wmma_f32_16x16x4_f32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; K=4 WMMA cross-target lift — MODULOREPLICATION regression pin.
; Runs the same `.co` as `wmma_f32_16x16x4_f32.ll` but with
; `--disable-wave-native` so `ModuloReplicationProjection`
; engages instead of `WaveNativeProjection`.
;
; Why this sibling exists
; =======================
;
; Pre-Session-8 (2026-04-23) `handle_valu_vop3p.cpp`'s K=4 arm
; refused the MODREP lift with `RaiseFailure::unsupportedShape`
; whenever `kernelHasPermlane16Swap` was true — a conservative
; sidecar refusal added during the matmul_fp16 multi-WMMA
; investigation.  The root cause turned out to be one layer up
; (the symmetric `v_permlane16_swap_b32` lift; see
; `handle_valu_cross_lane.cpp::emitPermLaneSwapEmulation` and
; matrix-translation.md §12.4.7), so the refusal was dropped and
; the K=4 MFMA redistribution runs in both projections.  No
; corpus kernel today exercises K=4 under MODREP — the gap
; between "K=4 MODREP drops through" and "K=4 MODREP is tested"
; is what this fixture closes.
;
; Invariants pinned (deliberately a subset of the WaveNative
; sibling's contract — the projection-specific fixture asserts
; the shared `emitWMMAtoMFMA_F32_16x16x4` structure, not the
; projection-specific EXEC-virtualisation story):
;
;   1. The lift SUCCEEDS (pre-Session-8 it refused loudly; a
;      regression to the refusal gate would fail the CHECK-LABEL
;      because the kernel wouldn't be emitted).
;
;   2. Exactly ONE `mfma.f32.16x16x4f32` call.  Under MODREP
;      `numSourceWavesPerTarget() == 1` (phantom-lane regime
;      single-source-wave projection), so
;      `emitWMMAtoMFMA_F32_16x16x4` emits ONE group pass — the
;      WaveNative sibling's two-pass layout is projection-
;      specific and must NOT surface under MODREP.
;
;   3. The MFMA signature is `(float, float, <4 x float>)` — NOT
;      the K=16 / K=32 packed shapes.  A dispatch misroute into
;      the K=32/K=64 WMMA arm would surface as a differently-
;      typed MFMA call.
;
;   4. MODREP-specific: NO `@llvm.amdgcn.init.whole.wave` call.
;      `WaveNativeProjection::emitInitialExec` emits exactly one
;      at kernel entry to make HW EXEC=-1 a kernel-wide ambient;
;      `ModuloReplicationProjection` does NOT (HW EXEC remains
;      the source-active mask).  Per-MFMA `strict.wwm` brackets
;      (emitted by `wrapAsWWMValue` under MODREP) are the
;      wave-scoped EXEC=-1 mechanism this projection uses.
;
;   5. At least one `@llvm.amdgcn.strict.wwm.*` call — the
;      MODREP-specific WWM bracket around the lane
;      redistribution / MFMA / collect chain that substitutes for
;      the WaveNative init_whole_wave ambient.  A regression that
;      loses both would ship a silently miscompiled kernel
;      (the MFMA would execute under the source-active EXEC
;      mask, leaving the phantom lanes' MFMA inputs undef).

; CHECK-LABEL: define amdgpu_kernel void @wmma_f32_16x16x4_f32_kernel(

; Exactly one MFMA call — MODREP is single-source-wave.
; CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{[^,]+}}, float %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)

; MODREP-specific strict.wwm wrap around the MFMA result.  The
; projection-specific WWM bracket (`wrapAsWWMValue`) substitutes
; for the WaveNative init_whole_wave kernel-entry ambient.  A
; regression that loses both would ship a silently miscompiled
; kernel (the MFMA would execute under the source-active EXEC
; mask, leaving the phantom lanes' MFMA inputs undef).
; CHECK: call {{.*}} @llvm.amdgcn.strict.wwm

; Exactly one MFMA call (anchored AFTER the positive check).
; CHECK-NOT: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(

; MODREP-specific: NO kernel-entry init_whole_wave (that's the
; WaveNative sibling's signature).
; CHECK-NOT: call {{.*}} @llvm.amdgcn.init.whole.wave

; Negative: no native gfx1250 WMMA intrinsic (gfx942 target).
; CHECK-NOT: @llvm.amdgcn.wmma.f32.16x16x4.f32

; Negative: no cross-K MFMA intrinsics.
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16f16
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16bf16
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x32_
