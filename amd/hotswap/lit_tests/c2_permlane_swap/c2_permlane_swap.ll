; RUN: %raise_cli %c2_permlane_swap_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c2_permlane_swap_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; The P4 rewrite (v_permlane16_swap_b32 lift) has landed — see
; the permlane16_swap row of hotswap/docs/wave-size-translation.md
; §5.3. The classifier's LaneGroupShuffle site accepts
; V_PERMLANE16_SWAP_B32 as outcome (b) because
; `handle_valu_cross_lane.cpp` emulates the two-VGPR exchange
; through paired `llvm.amdgcn.ds.bpermute` calls (the same target-
; independent path the P2 permlane16/permlanex16 emulation uses,
; for the same reason: gfx942 lacks native isel for
; `llvm.amdgcn.permlane16.swap`, per upstream LLVM's
; `test/CodeGen/AMDGPU/llvm.amdgcn.permlane16.swap.ll` ERR-SDAG
; assertion).
;
; Source-ISA-specific emission shape.  This fixture runs against
; a gfx1250 (wave32) source, where the MI400 Shader Programming
; Guide § V_PERMLANE16_SWAP_B32 pragma pins an ASYMMETRIC per-
; lane semantic: only two of the four 16-lane rows move, and the
; other two retain their original value.  Consequently the lift
; emits TWO `ds_bpermute` calls (one per output VGPR) followed
; by TWO per-lane `select` instructions keyed on the half-bit
; `lane AND 16 == 0` (the "low row" of each partnered pair).
; The wave64 sibling fixture `lit_tests/v_permlane32_swap_b32/`
; pins the pre-Session-8 SYMMETRIC shape (no selects, raw
; bpermute outputs) that the gfx950-source arm emits — the
; `isWave32()` gate in `emitPermLaneSwapEmulation` selects
; between the two shapes.
;
; This test asserts:
;   1. The raise succeeds (the classifier marks
;      V_PERMLANE16_SWAP_B32 [implemented]).
;   2. The emitted IR contains TWO calls to `llvm.amdgcn.ds.bpermute`
;      — one per output VGPR (vdst and src0_out).  Handler-local
;      SSA names `pls16_bperm_src0` / `pls16_bperm_vdst` pin the
;      cross-wired shape: first bpermute reads `src0_in`, second
;      reads `vdst_in`.
;   3. The signature property is the partner-lane XOR with 0x10
;      (= 16): each lane's source-lane index is `lane_id XOR 16`.
;   4. The ASYMMETRIC per-lane select shape: half-bit AND, icmp
;      EQ against 0, and two selects whose outputs are the final
;      `new_vdst` / `new_src0_out` values written to the tied
;      VGPR pair.  A regression to the symmetric lift (bpermute
;      results written directly to the output VGPRs, no selects)
;      would fail these CHECKs — the exact shape miscompile that
;      shipped pre-Session-8 and corrupted matmul_fp16.
;   5. The intrinsic declaration is present.

; CHECK-LABEL: define amdgpu_kernel void @c2_permlane_swap_kernel(

; (1)+(3) Partner = lane_id XOR 16, byte-address shift by 2.
; CHECK: %pls16_partner{{[0-9]*}} = xor i32 %{{[^,]+}}, 16
; CHECK: %pls16_addr{{[0-9]*}} = shl i32 %pls16_partner{{[0-9]*}}, 2

; (2) Two ds_bpermute calls, cross-wired.  Pinning on the
; handler-local twines (`pls16_bperm_src0` reads src0_in,
; `pls16_bperm_vdst` reads vdst_in) catches both a single-
; bpermute emission AND a same-source / reversed cross-wiring
; that a generic "two bpermutes" CHECK would accept.
; CHECK: %pls16_bperm_src0{{[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(i32 %pls16_addr{{[0-9]*}}, i32 %{{[^)]+}})
; CHECK: %pls16_bperm_vdst{{[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(i32 %pls16_addr{{[0-9]*}}, i32 %{{[^)]+}})

; (4) Asymmetric per-lane select on the half-bit.  A regression
; to the pre-Session-8 symmetric lift would write the bpermute
; results straight to the output VGPRs without this half-bit
; select, failing both of the CHECKs below.
; CHECK: %pls16_half_bit{{[0-9]*}} = and i32 %{{[^,]+}}, 16
; CHECK: %pls16_is_lane_low{{[0-9]*}} = icmp eq i32 %pls16_half_bit{{[0-9]*}}, 0
; CHECK: %pls16_new_vdst{{[0-9]*}} = select i1 %pls16_is_lane_low{{[0-9]*}}, i32 %{{[^,]+}}, i32 %pls16_bperm_src0{{[0-9]*}}
; CHECK: %pls16_new_src0_out{{[0-9]*}} = select i1 %pls16_is_lane_low{{[0-9]*}}, i32 %pls16_bperm_vdst{{[0-9]*}}, i32 %{{[^,]+}}

; (5) The intrinsic declaration must be present.
; CHECK:      declare {{.*}}i32 @llvm.amdgcn.ds.bpermute(i32, i32)
