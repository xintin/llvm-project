; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=c5_predicate_chain_phantom_lane_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR_WN
;
; Paired MODREP invocation: `--disable-wave-native` forces the
; MODREP refusal arm, which fires on ANY C5 site regardless of the
; WG size. This pins that the phantom-lane bit is a WaveNative-
; specific annotation — under MODREP the diagnostic does NOT carry
; the "phantom-lane regime" prefix because the refusal reason is
; the unconditional MODREP replica-1 EXEC-share trap, not the
; WaveNative phantom-lane collapse.
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --disable-wave-native \
; RUN:     --emit-ir=c5_predicate_chain_phantom_lane_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR_MR
;
; Regression fence for the phantom-lane sub-case of the Class-5
; classifier (`c5_predicate_chain_classifier.hpp` file-header
; docstring + the `phantomLaneGuaranteed` predicate in
; `classifyPredicateChain`).
;
; Under the post-graduation WaveNative default, the classifier
; suppresses its MODREP-specific refusal on the premise that
; `init_whole_wave` + per-source-lane-modeled EXEC give each target
; lane a 1:1 source-lane mapping — so `tid < K` evaluates the same
; way the source HW would have. That premise HOLDS for the common
; case `max_flat_workgroup_size >= targetWaveSize` (where every
; target wavefront lane corresponds to a source-kernel thread), but
; it COLLAPSES when the kernel's launch bounds force a WG smaller
; than the target wavefront width: hardware activates the spare
; lanes via `init_whole_wave`, their architectural `tid` is their
; hardware lane index (32..63 on gfx942 wave64), and the source
; kernel never modelled computation at those lane positions. The
; compiler's `@llvm.amdgcn.workitem.id.x()` lift returns those
; out-of-range `tid` values verbatim, the compile-time-K predicate
; partitions lanes into "modelled" vs "phantom", and any convergent
; cross-lane op (`ds_bpermute`, `ds_swizzle`, `permlane*`) reading
; from a phantom lane observes unmodelled state — the silent
; miscompile `canary_bitmatrix_composite` empirically produces
; (see the canary's docstring for the runtime evidence).
;
; The classifier's phantom-lane arm closes this gap by making the
; refusal fire under WaveNative iff the HSACO's
; `max_flat_workgroup_size` is statically below the target
; wavefront width. This fixture forces that state via
; `__launch_bounds__(32)` on a gfx1250 (wave32) source compiled
; against a gfx942 (wave64) target.
;
; Paired with `lit_tests/c5_predicate_chain_tid/` (same icmp shape,
; no launch_bounds → max_flat_workgroup_size = 1024 > 64, no
; phantom lanes, WaveNative still suppresses). The pair pins BOTH
; WaveNative arms end-to-end.

; STDERR_WN: transpiler: pre-translation abort:
; STDERR_WN-SAME: cross-wave-predicate-chain
; STDERR_WN-SAME: workitem.id.x-predicate-chain-classifier

; Phantom-lane prefix names the distinguishing evidence — the HSACO
; WG bound and the target wavefront width.
; STDERR_WN: phantom-lane regime:
; STDERR_WN-SAME: max_flat_workgroup_size
; STDERR_WN-SAME: 32
; STDERR_WN-SAME: target wavefront width
; STDERR_WN-SAME: 64

; The base C5-shape reason is still present (the phantom-lane
; detail is a prefix on the underlying refusal reason, not a
; replacement) so operators can follow the chain from the
; WaveNative-specific cause back to the C5 class.
; STDERR_WN: icmp ult
; STDERR_WN-SAME: compile-time constant 16
; STDERR_WN-SAME: W_s-1=31

; Outcome line carries the phantom-lane sub-case annotation so
; triage tools can bucket separately from the baseline MODREP
; refusal.
; STDERR_WN: outcome: (c) refuse
; STDERR_WN-SAME: WorkitemIdPredicateChain
; STDERR_WN-SAME: Class 5 phantom-lane sub-case

; Outer-tool failure line.
; STDERR_WN: raise_cli: kernel 'c5_predicate_chain_phantom_lane_kernel' failed to raise:
; STDERR_WN-SAME: cross-wave-predicate-chain

; MODREP assertions: same refusal CLASS fires, but WITHOUT the
; phantom-lane prefix — MODREP refuses every C5 site unconditionally
; (the `waveNative=false` arm doesn't consult `maxFlatWorkgroupSize`).
; Pins that the phantom-lane annotation is WaveNative-only.
; STDERR_MR: transpiler: pre-translation abort:
; STDERR_MR-SAME: cross-wave-predicate-chain
; STDERR_MR: icmp ult
; STDERR_MR-SAME: compile-time constant 16
; STDERR_MR-SAME: W_s-1=31
; STDERR_MR: outcome: (c) refuse
; STDERR_MR-SAME: WorkitemIdPredicateChain
; STDERR_MR-SAME: Class 5

; MODREP outcome line MUST NOT carry the phantom-lane sub-case
; annotation — if this CHECK-NOT fires it means the MODREP arm
; started consulting `maxFlatWorkgroupSize` (which would blur the
; two refusal reasons and break the WaveNative-specific
; attribution contract).
; STDERR_MR-NOT: phantom-lane sub-case
; STDERR_MR-NOT: phantom-lane regime

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c5_predicate_chain_phantom_lane_kernel
	.p2align	8
	.type	c5_predicate_chain_phantom_lane_kernel,@function
c5_predicate_chain_phantom_lane_kernel: ; @c5_predicate_chain_phantom_lane_kernel
; %bb.0:
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x14
	s_load_b64 s[2:3], s[0:1], 0x0
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s0, ttmp9, s1
	v_mad_u32 v1, s0, s4, v0
	;;#ASMSTART
	v_cmp_lt_u32_e64 s0, v0, 16
	
	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v0, -1, v0, s0
	
	;;#ASMEND
	global_store_b32 v1, v0, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c5_predicate_chain_phantom_lane_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
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
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 32
    .name:           c5_predicate_chain_phantom_lane_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c5_predicate_chain_phantom_lane_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
