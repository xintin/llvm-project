; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_cmp_cndmask_sgpr_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift fixture for the V_CMP_*_e64 -> SGPR -> V_CNDMASK_B32_e64
; per-lane-predication idiom, the canonical libdevice-math branch
; shape (asin / acos / atan / log / exp) and every Triton e64-
; across-BB epilogue.
;
; This test pins the SHADOW / FUSED path introduced by
; `RaiseContext::lastSgprWaveMaskI1`: when a V_CMP producer and a
; V_CNDMASK consumer are in the same basic block with no intervening
; scalar write to the mask SGPR, the consumer reads the per-lane
; compare `i1` directly instead of going through the lossy narrow-
; ballot / extractLaneBitFromWaveMask round-trip. See
; hotswap/docs/sgpr-wave-mask-translation.md section 3.1 for the
; full design rationale; the sibling fixture
; `v_cmp_cndmask_sgpr_scalar_clobber/` pins the fallback path that
; fires when an intervening scalar write invalidates the shadow.
;
; Three contract pieces covered:
;
;   1. The compare still produces a wave-mask store to the SGPR:
;      full-width target-hardware ballot (`ballot.i64`) truncated
;      to source width (`trunc i64 to i32`). The narrow store
;      survives so scalar consumers of the SGPR (`s_mov_b32 sM,
;      sN`, arithmetic) see a well-defined value, matching source
;      semantics for the 32-bit scalar role. Shared identifiers
;      `vcmp_ballot` + `vcmp_ballot_trunc` with the sibling
;      `v_cmpx_ballot` / `v_cmp_class_f32` fixtures.
;   2. The consumer bypasses the extract chain. The `select`
;      directly consumes the `%vcmpf` i1 (per-lane, full-fidelity)
;      — NO `mask_lane_idx` / `mask_at_lane` / `mask_lane_bit` /
;      `mask_lane_i1` chain feeds the cndmask.
;   3. The cndmask carries an `i1` condition, not a wave-uniform
;      wide-integer — the two-round-fix-ago pre-fix shape that
;      this fixture also forbids.
;
; CHECK-LABEL: define amdgpu_kernel void @v_cmp_cndmask_sgpr_kernel(

; (1) The per-lane fcmp i1 (the producer-side "truth" source of the
; whole dataflow), captured into a named SSA value so later CHECK
; lines can correlate the cndmask to this exact i1.
; CHECK: [[CMP:%vcmpf[0-9]*]] = fcmp oge float %{{[^,]+}}, 5.000000e-01

; (2) Writer preserves the wave-mask store for any scalar consumer
; of the SGPR: full-width ballot + trunc + `writeRegExecWidth` to the
; source-width (i32) SGPR alloca. The store itself is lowered by the
; reg file into a `store i32 ... sgpr...` that is NOT explicitly
; pinned here (the alloca names are implementation-private); what IS
; pinned is the ballot / trunc pair preceding it.
; CHECK: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 [[CMP]])
; CHECK-NEXT: %vcmp_ballot_trunc = trunc i64 %vcmp_ballot to i32

; (3) The consumer reads the cached per-lane `i1` directly. The
; cndmask's `select i1` condition is the SAME SSA value [[CMP]] that
; the V_CMP produced. No extract chain between the V_CMP and the
; cndmask — the shadow path short-circuits
; `extractLaneBitFromWaveMask`.
;
; `-1.0` is 0xbf800000 = -1082130432 as i32; `+1.0` is 0x3f800000 =
; 1065353216. The cndmask materialises them as int constants because
; its sources are the inline-asm literals emitted by the gfx1250
; assembler for `v_cndmask_b32_e64 v, -1.0, 1.0, s4` (see
; v_cmp_cndmask_sgpr.hip).
; CHECK: %cndmask = select i1 [[CMP]], i32 1065353216, i32 -1082130432

; NEGATIVE assertions, ordered.

; (a) The fallback extract chain — `mask_lane_idx` / `mask_at_lane`
; / `mask_lane_bit` / `mask_lane_i1` — must NOT be emitted between
; the V_CMP and the cndmask. The sibling `scalar_clobber` fixture
; confirms this chain IS emitted when a scalar write invalidates the
; shadow; here, with no invalidation, it must be absent.
; CHECK-NOT: %mask_lane_idx{{[0-9]*}} = zext i32 %{{[^ ]+}} to i64
; CHECK-NOT: %mask_at_lane{{[0-9]*}} = lshr i64 %{{[^,]+}},
; CHECK-NOT: %mask_lane_i1{{[0-9]*}} = icmp ne i64

; (b) The cndmask `select` MUST carry an `i1` condition, not a
; wide-integer — the earliest pre-fix shape from two rounds ago
; (`ICmpNE <sgpr-load>, 0` feeding a uniform `select` across the
; wave).
; CHECK-NOT: select i32 %{{.*}}, i32 {{.*}}, i32 {{.*}}

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cmp_cndmask_sgpr_kernel
	.p2align	8
	.type	v_cmp_cndmask_sgpr_kernel,@function
v_cmp_cndmask_sgpr_kernel:              ; @v_cmp_cndmask_sgpr_kernel
; %bb.0:
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	global_load_b32 v1, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_cmp_ge_f32_e64 s4, |v1|, 0.5
	v_cndmask_b32_e64 v1, -1.0, 1.0, s4
	
	;;#ASMEND
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cmp_cndmask_sgpr_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 5
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
    .name:           v_cmp_cndmask_sgpr_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         v_cmp_cndmask_sgpr_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
