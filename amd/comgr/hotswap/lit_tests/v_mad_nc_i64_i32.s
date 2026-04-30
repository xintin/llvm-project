; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_mad_nc_i64_i32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_mad_nc_i64_i32. Pins that the gfx1250 VOP3 no-carry
; signed 64-bit multiply-add lowers to the canonical
; `add(mul(sext s0, sext s1), s2_i64)` IR.  The handler lives in
; transpiler/handle_valu.cpp under
;   `if (sop == SemOp::V_MAD_NC_I64_I32) { ... }`
; with the matching SemOp in transpiler/semop.hpp under the
; `V_MAD_NC_*` block, and the decoder mapping in
; transpiler/opcode_map.cpp at
;   `E(V_MAD_NC_I64_I32_e64, V_MAD_NC_I64_I32)`.
;
; Per AMDGPU/VOP3Instructions.td:196 and :2129 the instruction has
; profile `VOP_I32_I32_I64_DPP` (`VOPProfile<[i64, i32, i32, i64]>`)
; and no SDPattern — it's purely a TableGen pseudo.  The AMDGPU
; backend's `SelectMad64_32` in AMDGPUISelDAGToDAG.cpp:1220 matches
; the idiomatic widening-MAD IR (two sign-extended i32 factors, i64
; accumulator) and re-emits the target-specific MAD on gfx1250 /
; gfx942.  The emitted IR must therefore contain:
;   * two `sext i32 ... to i64` for the factors (NOT zext — the `i` in
;     `v_mad_nc_i64_i32` names signed widening);
;   * a `mul {{.*}}i64` of those sign-extended values (or an isa-
;     legal commuted form the backend can still match);
;   * an `add {{.*}}i64` whose second operand is the i64 accumulator
;     (the canonical breadcrumb is the `vmad_nc_i64` value-name the
;     handler sets, mirroring `vmad_co64` from `V_MAD_CO_U64_U32`).

; CHECK-LABEL: define amdgpu_kernel void @v_mad_nc_i64_i32_kernel(

; Two signed widenings of the 32-bit factors.
; CHECK: sext i32 %{{[^ ]+}} to i64
; CHECK: sext i32 %{{[^ ]+}} to i64

; Widening multiply — 64-bit product of the sign-extended factors.
; Commutativity of `mul` means CHECK can't pin operand order.
; CHECK: mul {{.*}}i64 %{{[^,]+}}, %{{[^,]+}}

; Canonical accumulator add emitted with the handler's `vmad_nc_i64`
; value-name.  If the handler ever renames the value, update both
; semop.hpp's `V_MAD_NC_*` comment block (which names this breadcrumb)
; and the matching sibling row in handle_valu.cpp.
; CHECK: %vmad_nc_i64 = add {{.*}}i64

; Negative checks.
;   - No zext on the factor path — that would lower to v_mad_nc_u64_u32
;     (the unsigned sibling), not v_mad_nc_i64_i32.  Pre-gate the test
;     against an accidental handler swap.
; CHECK-NOT: %vmad_nc_u64 = add {{.*}}i64
;   - No `with.overflow` intrinsic CALL-site — the "nc" in the opcode
;     name means we deliberately do NOT model the carry-out.  The
;     `call` keyword pins this to actual use-sites rather than
;     module-level `declare` lines the LLVM textual writer emits
;     even for unused intrinsics pulled in transitively.  Emitting
;     an overflow intrinsic call here would bloat the IR and miss
;     the backend's `SelectMad64_32` pattern (which matches only
;     the plain add/mul shape).
; CHECK-NOT: call {{.*}}@llvm.smul.with.overflow
; CHECK-NOT: call {{.*}}@llvm.sadd.with.overflow
;   - No narrowing mul-i24 fallback — the 32-bit tid arithmetic
;     inside the kernel still surfaces as `add i32 / mul i32`, but
;     neither of those is the MAD we care about; use the
;     `vmad_nc_i64` breadcrumb above as the anchor and guard
;     against the specific i24 builtin that a mis-handler swap to
;     the narrow MAD family would introduce.
; CHECK-NOT: call {{.*}}@llvm.amdgcn.mul.i24
;   - No saturating-add intrinsic.  The HIP `asm volatile` in the
;     .hip file doesn't set the VOP3 clamp bit, so this fixture
;     exercises the handler's `clamp = 0` fast-path (plain
;     `add i64`).  If a future fixture encodes `clamp = 1` (or a
;     corpus producer surfaces and we graduate the handler to
;     emit saturation per the block comment in
;     `handle_valu.cpp`'s V_MAD_NC_* arm), a sibling fixture will
;     cover `llvm.sadd.sat.i64` emission.  Until then, any
;     appearance here means the handler silently promoted to
;     saturation and we need to investigate.
; CHECK-NOT: call {{.*}}@llvm.sadd.sat.i64

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_mad_nc_i64_i32_kernel
	.p2align	8
	.type	v_mad_nc_i64_i32_kernel,@function
v_mad_nc_i64_i32_kernel:                ; @v_mad_nc_i64_i32_kernel
; %bb.0:
	s_load_b32 s2, s[0:1], 0x2c
	s_bfe_u32 s3, ttmp6, 0x4000c
	s_load_b256 s[4:11], s[0:1], 0x0
	s_add_co_i32 s3, s3, 1
	s_and_b32 s12, ttmp6, 15
	s_wait_xcnt 0x0
	s_mul_i32 s0, ttmp9, s3
	s_getreg_b32 s1, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s12, s12, s0
	s_wait_kmcnt 0x0
	s_and_b32 s0, s2, 0xffff
	s_cmp_eq_u32 s1, 0
	s_cselect_b32 s1, ttmp9, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_u32 v2, s1, s0, v0
	s_clause 0x2
	global_load_b32 v3, v2, s[6:7] scale_offset
	global_load_b32 v4, v2, s[8:9] scale_offset
	global_load_b64 v[0:1], v2, s[10:11] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_mad_nc_i64_i32 v[0:1], v3, v4, v[0:1]
	
	;;#ASMEND
	global_store_b64 v2, v[0:1], s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_mad_nc_i64_i32_kernel
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 13
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
    .name:           v_mad_nc_i64_i32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     13
    .symbol:         v_mad_nc_i64_i32_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
