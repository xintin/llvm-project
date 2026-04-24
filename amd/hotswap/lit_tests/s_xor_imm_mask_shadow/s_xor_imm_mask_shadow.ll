; RUN: %raise_cli %s_xor_imm_mask_shadow_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --disable-wave-native \
; RUN:     --emit-ir=s_xor_imm_mask_shadow_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Pins immediate-source SOP2 shadow propagation for:
;   v_cmp_* -> s_xor_b32 sN, sN, -1 -> v_cndmask ... sN
;
; Pre-fix, `tryGetSrcWaveMaskI1` returned null for non-reg SOP2 operands,
; so `s_xor_b32 sN, sN, -1` could not re-record a derived per-lane i1 shadow.
; The downstream SGPR-conditioned cndmask then fell back to extracting lane bits
; from the narrow SGPR mask, reintroducing cross-widening loss.
;
; CHECK-LABEL: define amdgpu_kernel void @s_xor_imm_mask_shadow_kernel(
;
; Producer compare.
; CHECK: [[CMP:%[[:alnum:]_.]+]] = fcmp oge float %{{[^,]+}}, 5.000000e-01
;
; Immediate operand `-1` must still participate in i1-space derivation via
; source->exec-width extraction (no null/non-reg early-out):
; CHECK: [[IMM_MASK:%[[:alnum:]_.]+]] = icmp ne i64 %mask_lane_bit, 0
; CHECK: [[INV:%[[:alnum:]_.]+]] = xor i1 [[CMP]], [[IMM_MASK]]
;
; SGPR-conditioned cndmask must consume the derived i1 directly (shadow path),
; not a fallback extract chain from narrow SGPR bits.
; CHECK: %cndmask = select i1 [[INV]], i32 1065353216, i32 -1082130432
;
; CHECK-NOT: %mask_lane_i1

