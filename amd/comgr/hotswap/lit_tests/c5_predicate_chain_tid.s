; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --disable-wave-native \
; RUN:     --emit-ir=c5_predicate_chain_tid_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Paired non-refusal RUN under the WaveNative default (no
; `--disable-wave-native`). Same kernel binary; the projection
; gate inside `classifyPredicateChain` suppresses refusal under
; `enableWaveNative=true`, so raise_cli must succeed and the
; lifted IR must contain the kernel body. Pins the #11
; WaveNative-default contract: the env-var-free, flag-free
; invocation that reflects how the ROCR runtime calls the
; transpiler today MUST not refuse a C5-shape kernel. A
; regression that accidentally flips `waveNative=true` back to
; a short-circuit-to-empty-report would skip the walk entirely
; and silently pass here too — the gtest
; `WaveNativeProjectionGate` catches that by asserting
; `observedSites.size() == 1` under the same shape.
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=c5_predicate_chain_tid_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR_WN
;
; `--disable-wave-native` on the first RUN forces
; `ModuloReplicationProjection` — the narrow-O1 classifier's
; refusal rationale is MODREP-scoped (the replica-1-vs-source-
; wave-0 divergence it catches is a MODREP artefact), so the
; classifier's `waveNative` gate suppresses refusal under the
; post-graduation WaveNative default. The STDERR fixture pins
; the refusal on MODREP specifically; the IR_WN fixture pins
; non-refusal under the WaveNative default. See
; c5_predicate_chain_classifier.hpp's `waveNative` parameter
; docstring and hotswap/docs/modrep-predicate-chain.md §6
; "Picked: WaveNative as default" for the graduation rationale.
;
; Regression fence for the Class-5 "wave-size-sensitive predicate chain"
; narrow-O1 classifier
; (hotswap/docs/modrep-predicate-chain.md §5, Phase-2 narrowing per
;  §5 O1 "narrow-O1, as landed" of that doc;
;  transpiler/c5_predicate_chain_classifier.{hpp,cpp}).
;
; Under cross-widening (gfx1250 → gfx942 / gfx950), a wave32 source
; kernel whose `@llvm.amdgcn.workitem.id.x()` flows into an `icmp`
; against a compile-time constant `K` with `0 < K <= W_s - 1 = 31`,
; and whose chain to that icmp does NOT pass through an `and v, K'`
; with `K' <= W_s - 1`, partitions lanes by position within a single
; source wave. Modulo-replication's target-replica-1 lanes have
; architectural `tid >= W_s` so the predicate evaluates differently
; for them than for source wave 0's lane L — a miscompile the
; classifier has no safe rewrite for today (§5 O2 is deferred per
; §6.2). The only correct outcome is the (c) refusal branch of
; wave-size-translation.md §7's 3-outcome decision procedure.
;
; Paired with lit_tests/c5_predicate_chain_masked/ which pins the
; non-refusal path (chain AND-masked by 31 before reaching the icmp).
; The pair is the principled soundness-not-completeness regression
; fence for the classifier: REFUSE on unmasked + compile-time-K,
; OK on masked + same constant.
;
; Non-refusal contract for passing Triton baselines
; (`vecadd_f16`, `rope_fp32`, `canary_dpp_compound_add_fp32`) is
; enforced by `compare_correctness`'s end-to-end MATCH row rather
; than lit; see the §7 Test-surface table in
; `hotswap/docs/modrep-predicate-chain.md`. The narrowing rule ensures
; the classifier does not fire on those recipes — they use dynamic
; kernargs (`tid < %arg_N`), not compile-time constants, as the
; icmp's other operand.

; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-predicate-chain
; STDERR-SAME: workitem.id.x-predicate-chain-classifier

; Per-site trace names the ObstructionKind in human-readable form
; and includes the Class-5 cross-reference. The narrow-O1 classifier's
; refusal detail also carries the offending `icmp` predicate name and
; the failing constant so a reader of the diagnostic knows the
; triggering instruction's operand, not just the class.
; STDERR: icmp ult
; STDERR-SAME: compile-time constant 16
; STDERR-SAME: W_s-1=31
; STDERR: outcome: (c) refuse
; STDERR-SAME: WorkitemIdPredicateChain
; STDERR-SAME: Class 5

; raise_cli's outer-tool failure line names the kernel in
; kerneldex-style so coverage tooling can bucket on it.
; STDERR: raise_cli: kernel 'c5_predicate_chain_tid_kernel' failed to raise:
; STDERR-SAME: cross-wave-predicate-chain

; Under the WaveNative default the same kernel must raise
; cleanly. The IR contains the kernel body (IR-LABEL) and the
; `workitem.id.x()` lift that the MODREP-path refusal named —
; the source of the C5 shape is still there; only the projection
; has changed so the classifier's refusal is suppressed.
; IR_WN-LABEL: define amdgpu_kernel void @c5_predicate_chain_tid_kernel(
; IR_WN: call i32 @llvm.amdgcn.workitem.id.x()

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c5_predicate_chain_tid_kernel
	.p2align	8
	.type	c5_predicate_chain_tid_kernel,@function
c5_predicate_chain_tid_kernel:          ; @c5_predicate_chain_tid_kernel
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
	.amdhsa_kernel c5_predicate_chain_tid_kernel
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
    .max_flat_workgroup_size: 1024
    .name:           c5_predicate_chain_tid_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c5_predicate_chain_tid_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
