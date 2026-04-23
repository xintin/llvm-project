; RUN: %raise_cli %wmma_phantom_lane_f16_chain_co \
; RUN:     --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=wmma_f16_chain_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; 2-WMMA-chain K=32 f16 phantom-lane regime lowering fixture.
; Kernel: `__launch_bounds__(32)` with two back-to-back
; `__builtin_amdgcn_wmma_f32_16x16x32_f16` calls chaining through the
; accumulator — the SINGLE-range-per-operand pattern (no
; `v_permlane16_swap_b32`) that the MODREP path handles correctly.
;
; This fixture covers the SINGLE-WMMA / chain regime; the multi-WMMA-
; per-K-iter fragment-shuffle regime (which emits
; `v_permlane16_swap_b32`) is pinned separately by
; `wmma_phantom_lane_refuse` and is STILL refused — see the
; `handle_valu_vop3p.cpp` K=32/K=64 diagnostic for the surgical gate
; criterion and matrix-translation.md §12.4.4 for the root-cause
; characterisation.
;
; The lifting produces chained MFMA calls under `strict.wwm` for the
; WWM-scoped EXEC=-1 region that the MODREP projection needs around
; each WMMA→MFMA decomposition (K=32 → 2× K=16).  CHECK anchors pin
; that shape: the MFMA intrinsic name, at least one `strict.wwm`
; wrap, and the downstream collect path on `ds_bpermute`.
;
; CHECK: phantom-lane regime
; CHECK-SAME: max_flat_workgroup_size=32
; CHECK-SAME: target wavefront width=64
; CHECK: define amdgpu_kernel void @wmma_f16_chain_kernel
; CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> {{.*}}, <4 x half> {{.*}}, <4 x float> {{.*}}
; CHECK: call <4 x float> @llvm.amdgcn.strict.wwm.v4f32(<4 x float>
; CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 {{.*}}, i32 {{.*}}
; CHECK: call i32 @llvm.amdgcn.strict.wwm.i32(i32
