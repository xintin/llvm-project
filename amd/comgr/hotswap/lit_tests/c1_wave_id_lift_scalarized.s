; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --disable-writelane-rewrite \
; RUN:     --emit-ir=c1_wave_id_lift_scalarized_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE
;
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --enable-writelane-rewrite \
; RUN:     --emit-ir=c1_wave_id_lift_scalarized_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=REWRITTEN
;
; hotswap/docs/wave-size-translation.md §6 Class 1 table — the two
; terminal outcomes for the "canonical wave_id BFE lift scalarised
; through a cross-lane primitive under WMMA" pattern. Companion to
; lit_tests/c1_ttmp_wave_id_lift (rescue row) — together with this
; fixture's REFUSE / REWRITTEN siblings they pin all three outcomes
; for every downstream consumer of the `s_bfe_u32 sDST, ttmp8,
; 0x50019` wave_id read under cross-widening (gfx1250 → gfx942 /
; gfx950):
;
;   (a) rescue  — no v_writelane / v_readlane under WMMA: pinned by
;                 the c1_ttmp_wave_id_lift fixture;
;   (b) refuse  — all three co-occur AND the --enable-writelane-
;                 rewrite flag is OFF: the REFUSE RUN line pins the
;                 pre-Phase-6.5 refusal contract verbatim so the
;                 flag-off path stays honest even after the rewrite
;                 pass lands;
;   (c) rewrite — all three co-occur AND the --enable-writelane-
;                 rewrite flag is ON: the REWRITTEN RUN line pins the
;                 Phase 6.5 rewrite-to-select IR shape that
;                 `rewriteCrossLaneDivergent` in
;                 `rewrite_cross_lane_divergent.cpp` emits once the
;                 classifier in
;                 `wave_size_obstruction.cpp::buildObstructionReport`
;                 tags the WaveIdLiftScalarized site as
;                 RewriteId::PostRaiseCrossLaneRewrite (an
;                 *implemented* rewrite) instead of refusing outright.
;
; Detection logic lives in wave_size_obstruction.cpp's
; buildObstructionReport, which tracks the three conditions
; (`canonicalWaveIdBfeSites`, `crossLaneScalarSites`, `haveWMMA`) in
; separate vectors and emits one ObstructionKind::WaveIdLiftScalarized
; site per v_writelane / v_readlane once all three are non-empty.
;
; REFUSE path (--disable-writelane-rewrite — default is now ON, so
; the REFUSE sibling explicitly opts out of the rewrite):
;   We assert the three stable anchors the classifier + raise_cli
;   surface at the refusal boundary:
;
;     (a) the pre-translation abort line carries the cross-wave-lane-
;         id leak diagnostic, names the offending mnemonic
;         (v_writelane_b32), and references the wave-size-
;         translation.md §7 unrewritable table;
;     (b) at least one per-site trace names the ObstructionKind in
;         human-readable form AND includes the Class 1 cross-
;         reference AND the WMMA co-occurrence rationale
;         ("v_writelane/v_readlane + WMMA"); and
;     (c) the raise_cli wrapper's kerneldex-style failure line pins
;         the kernel name and mnemonic so coverage tooling can
;         bucket on it.
;
;   Matching is substring-based throughout so the test stays
;   resilient to future wording tightenings in the detail text.

; REFUSE: transpiler: pre-translation abort:
; REFUSE-SAME: cross-wave-lane-id-leak
; REFUSE-SAME: v_writelane_b32
; REFUSE-SAME: wave-size-translation.md

; REFUSE: WaveIdLiftScalarized
; REFUSE-SAME: Class 1
; REFUSE-SAME: v_writelane/v_readlane
; REFUSE-SAME: WMMA

; The outcome tag in the post-site summary keys on the (c) refuse
; branch of §7's 3-outcome decision procedure, same contract as the
; sibling c1_lane_id_leak fixture asserts for its MbcntHiLaneIdLeak
; path.
; REFUSE: outcome: (c) refuse

; raise_cli's outer-tool failure line names the kernel and mnemonic
; in kerneldex-style so coverage tooling can bucket on it.
; REFUSE: raise_cli: kernel 'c1_wave_id_lift_scalarized_kernel' failed to raise:
; REFUSE-SAME: v_writelane_b32

; REWRITTEN path (--enable-writelane-rewrite ON):
;   The classifier in buildObstructionReport re-tags the site as
;   RewriteId::PostRaiseCrossLaneRewrite and `rewriteImplemented =
;   true`, which lets it pass the Phase 1.4.5 pre-translation refusal
;   gate. In Phase 6.5 (post-mem2reg) the rewrite pass then replaces
;   the scalar-source v_writelane with a per-lane `select` keyed on
;   `(lane_id & (W_s-1)) == lane_idx`, materialising the canonical
;   `mbcnt_lo(-1, 0); mbcnt_hi(-1, prev)` target-wave lane_id in the
;   function's entry block with the rewriter-specific `cwd_` prefix.
;
;   We assert the four stable signatures unique to the rewrite
;   (matching the writelane_divergent_rewrite fixture's IR-shape
;   contract exactly — same mbcnt pair, same predicate shape, same
;   select name — so any future refactor that generalises the three
;   fixtures shares one ground truth):
;     (a) raise succeeds (no `transpiler: pre-translation abort`
;         line reaches stdout — it would leak through to stdout only
;         on a pipeline-internal failure, so its absence together
;         with the kernel body below is sufficient);
;     (b) the `cwd_lane_id_lo` + `cwd_lane_id` two-step mbcnt pair;
;     (c) the `cwd_wl_mask` lane-equality predicate;
;     (d) the `cwd_writelane_rewritten` select replaces the original
;         `@llvm.amdgcn.writelane` call.

; REWRITTEN-LABEL: define amdgpu_kernel void @c1_wave_id_lift_scalarized_kernel(
; REWRITTEN: %cwd_lane_id_lo = call i32 @llvm.amdgcn.mbcnt.lo
; REWRITTEN: %cwd_lane_id = call i32 @llvm.amdgcn.mbcnt.hi
; REWRITTEN: %cwd_wl_mask = icmp eq
; REWRITTEN: %cwd_writelane_rewritten = select i1
; REWRITTEN-NOT: call i32 @llvm.amdgcn.writelane

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c1_wave_id_lift_scalarized_kernel
	.p2align	8
	.type	c1_wave_id_lift_scalarized_kernel,@function
c1_wave_id_lift_scalarized_kernel:      ; @c1_wave_id_lift_scalarized_kernel
; %bb.0:
	s_load_b256 s[4:11], s[0:1], 0x0
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_wait_xcnt 0x0
	s_load_b32 s0, s[0:1], 0x2c
	s_wait_xcnt 0x0
	;;#ASMSTART
	s_bfe_u32 s1, ttmp8, 0x50019
	
	;;#ASMEND
	;;#ASMSTART
	v_writelane_b32 v26, s1, 0
	
	;;#ASMEND
	v_dual_mov_b32 v1, 0 :: v_dual_bitop2_b32 v26, s1, v26 bitop3:0x14
	s_add_co_i32 s2, s2, 1
	s_and_b32 s3, ttmp6, 15
	s_mul_i32 s2, ttmp9, s2
	s_wait_kmcnt 0x0
	s_clause 0x5
	global_load_b128 v[2:5], v1, s[4:5]
	global_load_b128 v[10:13], v1, s[6:7]
	global_load_b128 v[14:17], v1, s[6:7] offset:16
	global_load_b128 v[22:25], v1, s[8:9] offset:16
	global_load_b128 v[6:9], v1, s[4:5] offset:16
	global_load_b128 v[18:21], v1, s[8:9]
	s_wait_xcnt 0x1
	s_getreg_b32 s4, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s3, s3, s2
	s_and_b32 s0, s0, 0xffff
	s_cmp_eq_u32 s4, 0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[2:9], v[10:17], v[18:25]
	s_cselect_b32 s2, ttmp9, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_u32 v0, s2, s0, v0
	s_clause 0x1
	global_store_b128 v1, v[18:21], s[8:9]
	global_store_b128 v1, v[22:25], s[8:9] offset:16
	global_store_b32 v0, v26, s[10:11] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c1_wave_id_lift_scalarized_kernel
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 27
		.amdhsa_next_free_sgpr 12
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
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         36
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         46
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         48
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         50
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         52
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         54
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         96
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .max_flat_workgroup_size: 1024
    .name:           c1_wave_id_lift_scalarized_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         c1_wave_id_lift_scalarized_kernel.kd
    .vgpr_count:     27
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
