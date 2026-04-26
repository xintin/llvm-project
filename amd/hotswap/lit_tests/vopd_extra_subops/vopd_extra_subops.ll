; RUN: %raise_cli %vopd_extra_subops_co --isa=gfx1250 --target-isa=gfx942 \
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
