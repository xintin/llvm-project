; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_add_co_u32_sgpr_carry_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Regression fence for the carry-chain SGPR-operand bug class
; (hotswap/docs/modrep-predicate-chain.md §6.4, fixed
; 2026-04-22 in `handle_valu.cpp`'s `readCarryInI1` /
; `writeCarryOutI1` helpers).
;
; PRE-FIX SHAPE (the bug). The six carry-chain handlers
; (V_{ADD,SUB,SUBREV}_CO_(CI_)U32) hardcoded `loadVCC` / `storeVCC`
; for both carry-in (ci variants) and carry-out, silently ignoring
; the explicit scalar operand the e64 / VOP3B encoding names. For a
; pair like:
;
;   v_add_co_u32 vX, s0, vA, vB              (carry-OUT to s0)
;   v_add_co_ci_u32_e64 vY, s0, vC, vD, s0   (carry-IN from s0,
;                                             carry-OUT to s0)
;
; the pre-fix handler would:
;   * ignore the `s0` sdst on the first instruction, writing carry
;     to VCC instead;
;   * ignore the `s0` ssrc2 on the second instruction, reading
;     carry from VCC (which was never written by the intended
;     producer) — a stale carry-in.
;
; POST-FIX SHAPE (this fixture pins). `writeCarryOutI1` sees the
; sdst=`s0` ParsedReg and:
;   1. Ballots the per-lane carry i1 to source-wave-mask width via
;      `projection.ballotI1ToWidth` → `amdgcn.ballot.i64` (WaveNative
;      cross-widening source width is still i64 here, then trunc'd
;      to source width by `writeRegExecWidth`).
;   2. Stores the narrow mask to the s0 alloca via
;      `writeRegExecWidth`.
;   3. Records the fresh per-lane i1 shadow via
;      `ctx.recordSgprWaveMaskI1(0, carryI1, isPair=false)` so the
;      next consumer reads the i1 directly (bypassing the lossy
;      narrow-mask extract).
;
; Then `readCarryInI1` on the second instruction's ssrc2=`s0` hits
; the fresh-shadow cache and returns the SAME i1 SSA value the first
; add produced — no store-load round-trip through VCC, no lossy
; narrow-mask extract.
;
; The net IR shape for the pair is a pure dataflow chain:
;
;   %addCarryI1 = extractvalue {i32, i1} %add1_with_ov, 1
;   %carry_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %addCarryI1)
;   ...
;   %addCarryZext = zext i1 %addCarryI1 to i32              ; <-- SAME %addCarryI1
;   %ci = call {i32, i1} @llvm.uadd.with.overflow.i32(..., %addCarryZext)
;   ...
;
; A regression that re-introduces hardcoded-VCC would break the SSA
; chain — the second add's carry-in would `load i1, ptr %vcc`
; instead of zext-forwarding the first add's carry.

; CHECK-LABEL: define amdgpu_kernel void @v_add_co_u32_sgpr_carry_kernel(

; First carry-chain add (V_ADD_CO_U32 with sdst=s0) — classic
; uadd_with_overflow → ballot-and-store to s0 shape.
; CHECK: [[ADD1:%[[:alnum:]_.]+]] = call { i32, i1 } @llvm.uadd.with.overflow.i32
; CHECK: [[CARRY1:%[[:alnum:]_.]+]] = extractvalue { i32, i1 } [[ADD1]], 1
;
; The carry-out ballots to i64 (WaveNative target width) before
; truncation to source wave-mask width — this is the `carry_ballot`
; twine from `writeCarryOutI1`. A regression that skips the ballot
; (e.g. direct storeSGPR32 of a per-lane zext) fails this pin.
; CHECK: %carry_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 [[CARRY1]])

; Second carry-chain add (V_ADD_CO_CI_U32_e64 with ssrc2=s0 and
; sdst=s0) — the carry-IN must be the SAME i1 that the first add
; produced (fresh-shadow lookup), NOT a fresh `load i1, ptr %vcc`.
; The zext of the SAME `[[CARRY1]]` is what pins this.
; CHECK: zext i1 [[CARRY1]] to i32

; The second add's combined carry-out is also routed via
; writeCarryOutI1 → ballot-and-store to s0. Distinct `carry_ballot`
; SSA name (suffixed with a number) confirms a separate ballot
; call.
; CHECK: %carry_ballot{{[0-9]+}} = call i64 @llvm.amdgcn.ballot.i64

; NEGATIVE PIN: no `load i1, ptr %vcc` between the first and
; second adds. If the pre-fix handler were restored, the second
; add's carry-in would route through `loadVCC` which lowers to
; `load i1, ptr %vcc` — FileCheck-NOT across the relevant span
; catches the regression directly.
; CHECK-NOT: load i1, ptr %vcc

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_add_co_u32_sgpr_carry_kernel
	.p2align	8
	.type	v_add_co_u32_sgpr_carry_kernel,@function
v_add_co_u32_sgpr_carry_kernel:         ; @v_add_co_u32_sgpr_carry_kernel
; %bb.0:
	s_load_b128 s[4:7], s[0:1], 0x0
	v_dual_add_nc_u32 v1, 1, v0 :: v_dual_add_nc_u32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_nc_u32 v3, 3, v0 :: v_dual_bitop2_b32 v1, 31, v1 bitop3:0x40
	v_and_b32_e32 v2, 31, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v3, 31, v3
	s_wait_kmcnt 0x0
	s_clause 0x3
	global_load_b32 v4, v0, s[6:7] scale_offset
	global_load_b32 v5, v1, s[6:7] scale_offset
	global_load_b32 v6, v2, s[6:7] scale_offset
	global_load_b32 v7, v3, s[6:7] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_add_co_u32 v2, s0, v4, v5
	v_add_co_ci_u32_e64 v3, s0, v6, v7, s0
	
	;;#ASMEND
	global_store_b64 v0, v[2:3], s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_add_co_u32_sgpr_carry_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 8
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_add_co_u32_sgpr_carry_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_add_co_u32_sgpr_carry_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
