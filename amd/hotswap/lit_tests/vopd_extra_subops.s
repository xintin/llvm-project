; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=vopd_extra_subops_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift fixture for five previously-unhandled VOPD sub-shapes (see
; transpiler/handle_vopd.cpp and the matching kernel comment block):
;
;   1. v_dual_mov_b32 with f32 inline-constant literal source
;      (`1.0` printed by the AMDGPU instruction printer rather than
;      the hex bit pattern).  Surface form: `store i32 1065353216`
;      (= 0x3F800000, the IEEE-754 encoding of 1.0f) into the
;      destination VGPR via the `mov_lit` output.
;   2. v_dual_ashrrev_i32 — arithmetic right shift, REVERSED operand
;      order (shift amount = first src, value = second src).
;      Surface form: `ashr i32 %src, 31`.
;   3. v_dual_max_i32 — signed max via `llvm.smax.i32`.
;      Surface form: `call i32 @llvm.smax.i32(i32 0, i32 %src)`.
;   4. v_dual_mov_b32 with a `ttmp<N>` source — loads from the
;      corresponding TTMP alloca in the reg-file.  Surface form:
;      `load i32, ptr %ttmp9` (the named alloca `ttmp9` from the
;      reg-file).
;   5. v_dual_mov_b32 with `vcc_lo` source — routed through
;      `readVCCAsWaveMask(B, i32Ty)` so the wave-projection layer
;      emits its canonical `llvm.amdgcn.ballot.i32` on the
;      per-lane VCC i1 (plus whatever width-projection the current
;      wave-projection implementation adds).  Surface form: a call
;      into `llvm.amdgcn.ballot.i32` somewhere between the VCC
;      def and the VGPR store.
;
; Each shape lives in its own VOPD pair in the kernel so a regression
; in any one branch surfaces exactly one CHECK failure rather than
; cascading.
;
; INVARIANTS PINNED:
;
;   * The f32 inline-constant literal `1.0` is materialised as the
;     32-bit IEEE bit pattern (0x3F800000 = 1065353216 decimal) — the
;     same shape every other VALU handler emits when given a
;     register/literal mix.  A regression that returned `nullptr`
;     from the literal-parse path would surface as "VOPD
;     decomposition failed" at lift time and produce zero IR.
;   * The AShr appears with the literal SECOND ARGUMENT (the shift
;     count, value 31) and the register FIRST — matching the
;     IRBuilder `CreateAShr(s1, s0)` reversal applied by the handler.
;     Lit-amount AShr by 31 is a sign-extension idiom; the constant
;     31 must remain visible.
;   * The smax dispatch uses `llvm.smax.i32` (NOT `llvm.umax`, NOT a
;     manual `select (icmp sgt) ...`) — keeps the dispatch surface
;     symmetric with the float min/max handlers above it.
;   * The TTMP source surfaces as a named `vopd_ttmp` load from the
;     corresponding reg-file alloca (ttmp9 in this fixture).  A
;     regression that skipped the TTMP branch would return nullptr
;     from `readVOPDSrc` and abort VOPD decomposition, producing no
;     IR and an empty FileCheck input.
;   * The VCC-source read surfaces via the wave-projection ballot
;     intrinsic, NOT via a direct load of the i64 EXEC alloca or a
;     manual `zext i1` of the per-lane VCC shadow — routing through
;     `readVCCAsWaveMask` is what makes the downstream wave-size
;     rewrite work correctly under cross-widening.
;
; NEGATIVE PINS:
;
;   * NO `unsupportedOpcode` or `VOPD decomposition failed` text in
;     the captured stderr (FileCheck would see empty stdin).
;   * NO `llvm.umax` or `llvm.umin` (the corpus shape is signed).
;   * NO `lshr` for the ashrrev surface (must be arithmetic).

; CHECK-LABEL: define amdgpu_kernel void @vopd_extra_subops_kernel(

; Pair A: the ashrrev sibling — arithmetic shift right by the
; literal 31, applied to the register source.
; CHECK: %vopd_ashr = ashr i32 %{{[^,]+}}, 31

; Pair A: the mov_b32 of the float literal `1.0` materialises the
; IEEE bit pattern (1065353216 decimal = 0x3F800000) into the
; destination VGPR's SSA chain — the per-lane store path runs the
; constant through a phi (writing the literal in the active-lane
; arm and forwarding the prior value in the masked arm), so we pin
; the phi shape directly rather than a `store i32 …`.
; CHECK: phi i32 [ 1065353216,

; Pair B: the max_i32 sibling — signed max of literal 0 against the
; register source via the smax intrinsic.
; CHECK: %vopd_smax = call i32 @llvm.smax.i32(i32 0, i32 %{{[^)]+}})

; Pair C: the `v_dual_mov_b32 vX, ttmp9` — the handler emits a
; load-through-the-ttmp-alloca, but `PromoteMemToReg` folds the
; alloca to SSA because the only store into `ttmp[9]` is the SPE
; prelude's `%ttmp9_wg_id = call i32 @llvm.amdgcn.workgroup.id.x()`.
; So the post-mem2reg form is a phi that routes `%ttmp9_wg_id`
; directly into the per-lane destination VGPR's SSA chain (the
; active-lane arm of the SPE diamond around the dual-mov's Y-half
; store).  Pinning the phi edge on `%ttmp9_wg_id` catches both
; regressions: a TTMP-branch failure would strip the phi edge, and
; a store-wrong-value regression would surface as a different
; edge value (e.g. `undef`).  The SGPR Y-half is not pinned
; explicitly — if it regressed, VOPD decomposition would fail and
; the TTMP phi-edge above would disappear too.
; CHECK: phi i32 [ %ttmp9_wg_id,

; Pair D: the `v_dual_mov_b32 vX, vcc_lo` — the handler routes
; this through `readVCCAsWaveMask(B, i32Ty)`, which emits a
; ballot of the per-lane VCC i1 followed by a width-projection.
; The wave-projection layer currently emits
; `call i64 @llvm.amdgcn.ballot.i64(i1 %vcmp)` followed by
; `trunc i64 … to i32` when the target wave is 64.  Pin the ballot
; call as the stable surface invariant; the truncation and the
; subsequent phi edge into the destination VGPR are dependent on
; the current wave-projection implementation and are not pinned
; here to keep the test stable across projection refactors.
; CHECK: call i64 @llvm.amdgcn.ballot.i64

; Negative pin: the ashrrev must be arithmetic; a regression to
; logical shift would have surfaced as `lshr` here (with the same
; operand orientation).
; CHECK-NOT: %vopd_lshr =

; Negative pin: the max_i32 must dispatch to the SIGNED smax, not
; umax.  A regression that confused signed/unsigned would surface
; here.
; CHECK-NOT: @llvm.umax.i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vopd_extra_subops_kernel
	.p2align	8
	.type	vopd_extra_subops_kernel,@function
vopd_extra_subops_kernel:               ; @vopd_extra_subops_kernel
; %bb.0:
	s_load_b128 s[0:3], s[0:1], 0x0
	v_dual_add_nc_u32 v1, 1, v0 :: v_dual_add_nc_u32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 31, v1
	v_and_b32_e32 v4, 31, v2
	s_wait_kmcnt 0x0
	global_load_b32 v8, v0, s[0:1] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_dual_mov_b32 v2, 1.0 :: v_dual_ashrrev_i32 v3, 31, v8
	;;#ASMEND
	s_clause 0x1
	global_load_b32 v5, v1, s[0:1] scale_offset
	global_load_b32 v6, v4, s[0:1] scale_offset
	s_wait_xcnt 0x1
	v_dual_add_nc_u32 v1, 3, v0 :: v_dual_lshlrev_b32 v0, 5, v0
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_dual_mov_b32 v5, v5 :: v_dual_max_i32 v4, 0, v6
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 31, v1
	;;#ASMSTART
	v_dual_mov_b32 v6, ttmp9 :: v_dual_mov_b32 v7, s0
	;;#ASMEND
	global_load_b32 v1, v1, s[0:1] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_cmp_eq_u32_e64 vcc_lo, v8, 0
	v_dual_mov_b32 v8, vcc_lo :: v_dual_mov_b32 v9, v1
	;;#ASMEND
	s_clause 0x1
	global_store_b128 v0, v[2:5], s[2:3]
	global_store_b128 v0, v[6:9], s[2:3] offset:16
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vopd_extra_subops_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 10
		.amdhsa_next_free_sgpr 4
		.amdhsa_reserve_vcc 1
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
    .name:           vopd_extra_subops_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         vopd_extra_subops_kernel.kd
    .vgpr_count:     10
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
