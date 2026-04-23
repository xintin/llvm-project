; RUN: %raise_cli %wmma_phantom_lane_refuse_co \
; RUN:     --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=wmma_phantom_lane_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; Phantom-lane regime `v_wmma_f32_16x16x4_f32` (K=4 f32) lowering.
; Fixture name kept from the pre-2026-04-23 era when this opcode in
; MODREP refused unconditionally; the gate is now surgical (refuses
; only the multi-WMMA-per-K-iter regime marked by
; `v_permlane16_swap_b32` — see `handle_valu_vop3p.cpp`'s K=4 f32
; arm and matrix-translation.md §12.4.4).  This kernel has a SINGLE
; WMMA and no permlane16_swap, so it takes the MODREP
; `emitWMMAtoMFMA_F32_16x16x4` path and lifts correctly.
;
; Regression fence: if the surgical gate regressed to an
; unconditional refusal, this test's RUN line would exit non-zero
; and fail.  If the gate regressed to permit multi-WMMA kernels,
; the sibling `matmul_fp16` compare_correctness recipe would
; surface silently-wrong numerics and the companion
; `wmma_phantom_lane_refuse_multiwmma` fixture (TODO: add if
; needed) would catch it.
;
; CHECK anchors pin the expected IR shape:
;   (1) the phantom-lane → MODREP fallback log (unchanged from the
;       pre-surgical era — still fires because `__launch_bounds__(32)`);
;   (2) the MFMA K=4 f32 intrinsic emission (proves the lowering
;       reached `emitWMMAtoMFMA_F32_16x16x4`);
;   (3) the `strict.wwm` wrap that `wmma_lowering.cpp` inserts on
;       the MFMA output under MODREP so `SIWholeQuadMode` emits the
;       EXEC=-1 save/restore around the MFMA collective.

; CHECK: phantom-lane regime
; CHECK-SAME: max_flat_workgroup_size=32
; CHECK-SAME: target wavefront width=64
; CHECK-SAME: falling back to ModuloReplicationProjection

; CHECK: define amdgpu_kernel void @wmma_phantom_lane_refuse_kernel
; CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float {{.*}}, float {{.*}}, <4 x float> {{.*}}, i32 0, i32 0, i32 0)
; CHECK: call <4 x float> @llvm.amdgcn.strict.wwm.v4f32(<4 x float>
