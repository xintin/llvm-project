; RUN: %raise_cli %writelane_sgpr_forced_use_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --enable-writelane-rewrite \
; RUN:     --emit-ir=writelane_sgpr_forced_use_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=THREADLOOP
;
; RUN: %raise_cli %writelane_sgpr_forced_use_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --disable-writelane-rewrite \
; RUN:     --emit-ir=writelane_sgpr_forced_use_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=UNCHANGED
;
; Pins the forward use-chain classifier's SGPR-forced route in
; `rewrite_cross_lane_divergent.cpp::classifyForwardUseChain`. The
; fixture kernel (writelane_sgpr_forced_use.hip) chains a divergent
; `v_writelane_b32` into a `v_readfirstlane_b32`; the classifier must
; detect the readfirstlane as an SGPR-forced sink and route the whole
; function through the analysis-triggered ThreadLoopProjection retry
; rather than emit a `ds_bpermute`-shaped rewrite that the backend
; would immediately re-collapse via readfirstlane.
;
; THREADLOOP path (--enable-writelane-rewrite on):
;   Asserts four stable anchors the raiser surfaces at the route
;   boundary:
;     (a) the post-raise fallback line names the retry as
;         analysis-triggered (no user opt-in);
;     (b) the classifier's detail names the offending consumer
;         (`readfirstlane`) so the next investigation knows where to
;         look — the exact string is the intrinsic name as rendered
;         by `Function::getName`, stable across LLVM versions;
;     (c) the projection selection line proves the retry actually
;         entered ThreadLoopProjection;
;     (d) the raised IR uses ThreadLoop's source-wave-scoped lane-op
;         lowering (`writelane_srcwave` + `readfirstlane_srcwave`) rather
;         than either the post-raise `cwd_*` rewrite or target-wave-native
;         AMDGPU lane intrinsics.  That is the principled point of this
;         route: do not allow-list an SGPR-forced sink; translate the
;         original scalar-source shape inside the conservative projection
;         boundary instead.
;
; UNCHANGED path (flag off):
;   The kernel has no WMMA, so the Phase 1.4.5
;   `WaveIdLiftScalarized` classifier does NOT refuse the lift (the
;   three-way co-occurrence requires WMMA). The raiser emits
;   `@llvm.amdgcn.writelane` + `@llvm.amdgcn.readfirstlane` verbatim
;   and exits successfully. This arm is the regression guard for the
;   "no behaviour change when the flag is off" contract — ensures
;   the classifier does not leak refusal into the flag-off path.

; THREADLOOP: transpiler: post-raise fallback: retrying kernel 'writelane_sgpr_forced_use_kernel' under ThreadLoopProjection
; THREADLOOP-SAME: analysis-triggered, no user opt-in

; THREADLOOP: SGPR-forced consumer
; THREADLOOP-SAME: readfirstlane

; THREADLOOP: selected ThreadLoopProjection
; THREADLOOP-LABEL: define amdgpu_kernel void @writelane_sgpr_forced_use_kernel(
; THREADLOOP: writelane_srcwave = select i1
; THREADLOOP: readfirstlane_srcwave = call i32 @llvm.amdgcn.ds.bpermute
; THREADLOOP-NOT: cwd_writelane_rewritten
; THREADLOOP-NOT: call i32 @llvm.amdgcn.writelane
; THREADLOOP-NOT: call i32 @llvm.amdgcn.readfirstlane

; UNCHANGED-LABEL: define amdgpu_kernel void @writelane_sgpr_forced_use_kernel(
; UNCHANGED: call i32 @llvm.amdgcn.writelane
; UNCHANGED: call i32 @llvm.amdgcn.readfirstlane
; UNCHANGED-NOT: cwd_lane_id_lo
; UNCHANGED-NOT: cwd_writelane_rewritten
