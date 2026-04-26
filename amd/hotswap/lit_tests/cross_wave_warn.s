; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=cross_wave_writer_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Cross-wave translation (wave32 source → wave64 target) of a kernel
; whose only EXEC writer is lane-position-INDEPENDENT — here, a
; `v_cmpx_lt_u32_e64 threadIdx.x, 64` bounds check. The constant 64
; is structurally ≥ W_s so the Class-5 predicate-chain classifier
; in `c5_predicate_chain_classifier.cpp` accepts it as a bounds
; check rather than a lane-position gate. The Phase 1.4.5 MC-level
; classifier in `wave_size_obstruction.cpp` similarly classifies it
; as outcome (a) oblivious: the v_cmpx is an EXEC writer, but with
; no v_mbcnt_* in the kernel the syntactic co-occurrence heuristic
; does not flag it, and no refusal fires.
;
; Pre-C5-classifier-landing this fixture used K=8, exercising a
; genuine lane-position-scoped predicate under cross-wave — the C5
; classifier now refuses that exact shape (lit_tests/c5_predicate_chain_tid
; is the principled positive fixture for the refusal). The fixture
; was updated to K=64 rather than xfail'd so the cross-wave warning
; path still has an IR-generating producer.
;
; This is the regression-fence counterpart to the c4_lane_dep_cmpx
; lit test, which uses exactly the same v_cmpx shape but adds an
; `v_mbcnt_lo_u32_b32` to the body so the co-occurrence heuristic
; catches it. The pair of fixtures pins both directions of the
; classifier's decision.
;
; Historical note. Before Phase 1.4.5 landed, the raiser's warn-
; only Phase 1.4 gate printed a stderr banner (`transpiler:
; WARNING: cross-wave translation of an EXEC-manipulating kernel
; relies on modulo-replication, which is not provably correct in
; general...`). That banner is now routed through LLVM_DEBUG under
; DEBUG_TYPE="wave-projection" — it surfaces only with
; `-debug-only=wave-projection`. The classifier's per-site trace,
; emitted via the same DEBUG_TYPE, subsumes the banner's content.
; See `wave_projection.cpp:emitCrossWaveWarning` for the shim and
; `wave_size_obstruction.cpp:renderObstructionTrace` for the new
; format.
;
; We assert structurally: the raise succeeds and the emitted IR
; contains the kernel body — not a particular SSA name, which is
; not stable across LLVM versions.

; CHECK-LABEL: define amdgpu_kernel void @cross_wave_writer_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cross_wave_writer_kernel
	.p2align	8
	.type	cross_wave_writer_kernel,@function
cross_wave_writer_kernel:               ; @cross_wave_writer_kernel
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_dual_mov_b32 v3, 0 :: v_dual_lshlrev_b32 v2, 2, v0
	v_mov_b32_e32 v1, 0xaa
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[2:3], s[0:1], v[2:3]
	;;#ASMSTART
	v_cmpx_lt_u32_e64 v0, 64
	global_store_b32 v[2:3], v1, off
	s_wait_storecnt 0
	s_mov_b32 exec_lo, -1
	
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cross_wave_writer_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           cross_wave_writer_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         cross_wave_writer_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
