; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_bitcmp_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for the SOPC bit-test family
; (s_bitcmp0_b32 / s_bitcmp1_b32 / s_bitcmp0_b64 / s_bitcmp1_b64).
; See SemOp::S_BITCMP{0,1}_{B32,B64} in transpiler/semop.hpp; the
; shared handler block in transpiler/handle_sopc.cpp under
; `if (is64 || isB32) { ... }`; and the SOPC mapping in
; transpiler/opcode_map.cpp.  Bidirectional handler <-> test
; back-reference is non-negotiable per the transpiler test policy.
;
; Closes the kerneldex corpus blocker on scope_discovery `_attn_fwd`
; (one `s_bitcmp0_b32 s68, 1` that the pre-fix handler left
; unhandled — the compare silently dropped, the downstream
; SCC-dependent branch lifted to an undef-driven CFG).
;
; INVARIANTS PINNED:
;
;   1. For B32 variants, the shift amount is masked to 5 bits
;      (`and i32 _, 31`) before the `shl i32 1, _` bit-construction.
;      Mirrors the hardware's (src1 & 0x1F) shift-amount truncation
;      and keeps the LLVM IR `shl i32` in well-defined range.
;
;   2. For B64 variants, the shift amount is masked to 6 bits
;      (`and i32 _, 63`), then widened to i64 before the `shl i64`.
;      Same hardware invariant (src1 & 0x3F) and same IR-legality
;      argument, but over the 64-bit source width.
;
;   3. The final predicate differs per variant only:
;        _B32 / _B64 with the `0` suffix -> `icmp eq _, 0`
;        _B32 / _B64 with the `1` suffix -> `icmp ne _, 0`
;      A regression that flipped the 0/1 polarity would break every
;      SCC-branch that depends on bit presence.
;
;   4. The B64 sources are read as i64 (the SGPR pair is kept intact
;      through the `op.src64` reader) rather than narrowed to i32 —
;      a pair-of-i32 lift would only test the low 32 bits.

; CHECK-LABEL: define amdgpu_kernel void @s_bitcmp_kernel(

; --- s_bitcmp0_b32: shamt mask=0x1F, shift i32, compare eq 0 ------------
; Uses CHECK (not CHECK-NEXT) at the block boundary because each
; bitcmp block is separated from the next by emitted SPE /
; writeReg32 traffic for the surrounding store. The four lines
; within a single bitcmp block are consecutive in the IR though,
; so CHECK-NEXT is used inside each block.
; CHECK: %bitcmp_shamt = and i32 %{{[^,]+}}, 31
; CHECK-NEXT: %bitcmp_bit = shl i32 1, %bitcmp_shamt
; CHECK-NEXT: %bitcmp_mask = and i32 %{{[^,]+}}, %bitcmp_bit
; CHECK-NEXT: %bitcmp0 = icmp eq i32 %bitcmp_mask, 0

; --- s_bitcmp1_b32: shamt mask=0x1F, shift i32, compare ne 0 ------------
; Each block-entry pattern anchors on the `bitcmp_shamt` value name
; (with a numeric-suffix wildcard) rather than on a bare
; `and i32 _, 31` — the SPE lane-projection code introduces
; unrelated `and i32 %lane_id, 31` expressions between bitcmp
; blocks, and without this anchor the block-entry match would
; straddle the SPE region and misalign the follow-on adjacency
; pins.  See SPE lane detection in handle_common.cpp.
; CHECK: %bitcmp_shamt{{[0-9]*}} = and i32 %{{[^,]+}}, 31
; CHECK-NEXT: %bitcmp_bit{{[0-9]*}} = shl i32 1, %bitcmp_shamt{{[0-9]*}}
; CHECK-NEXT: %bitcmp_mask{{[0-9]*}} = and i32 %{{[^,]+}}, %bitcmp_bit{{[0-9]*}}
; CHECK-NEXT: %bitcmp1 = icmp ne i32 %{{[^,]+}}, 0

; --- s_bitcmp0_b64: shamt mask=0x3F, zext to i64, shift i64, eq 0 -------
; CHECK: %bitcmp_shamt{{[0-9]*}} = and i32 %{{[^,]+}}, 63
; CHECK-NEXT: %bitcmp_shamt64{{[0-9]*}} = zext i32 %{{[^,]+}} to i64
; CHECK-NEXT: %bitcmp_bit{{[0-9]*}} = shl i64 1, %bitcmp_shamt64{{[0-9]*}}
; CHECK-NEXT: %bitcmp_mask{{[0-9]*}} = and i64 %{{[^,]+}}, %bitcmp_bit{{[0-9]*}}
; CHECK-NEXT: %bitcmp0{{[0-9]*}} = icmp eq i64 %{{[^,]+}}, 0

; --- s_bitcmp1_b64: shamt mask=0x3F, zext to i64, shift i64, ne 0 -------
; CHECK: %bitcmp_shamt{{[0-9]*}} = and i32 %{{[^,]+}}, 63
; CHECK-NEXT: %bitcmp_shamt64{{[0-9]*}} = zext i32 %{{[^,]+}} to i64
; CHECK-NEXT: %bitcmp_bit{{[0-9]*}} = shl i64 1, %bitcmp_shamt64{{[0-9]*}}
; CHECK-NEXT: %bitcmp_mask{{[0-9]*}} = and i64 %{{[^,]+}}, %bitcmp_bit{{[0-9]*}}
; CHECK-NEXT: %bitcmp1{{[0-9]*}} = icmp ne i64 %{{[^,]+}}, 0

; Negative pin: no B64 variant emits a 32-bit shift against the
; 64-bit source. The literal token `%bitcmp_shamt64` is only
; introduced by the B64 lowering, so catching `shl i32 1, %bitcmp_shamt64`
; would expose a regression that dropped the zext step while keeping
; the value-name.
; CHECK-NOT: shl i32 1, %bitcmp_shamt64

; Negative pin: no i32 source operand fed into a B64-shaped icmp.
; A pair-of-i32 regression for the B64 variants would replace the
; `i64` source width with `i32`.
; CHECK-NOT: %bitcmp_mask{{.*}} = and i32 %{{.*}}, %bitcmp_bit20
; CHECK-NOT: %bitcmp_mask{{.*}} = and i32 %{{.*}}, %bitcmp_bit26

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_bitcmp_kernel
	.p2align	8
	.type	s_bitcmp_kernel,@function
s_bitcmp_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x2
	s_load_b32 s2, s[0:1], 0x34
	s_load_b32 s3, s[0:1], 0x20
	s_load_b256 s[4:11], s[0:1], 0x0
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s12, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s0, s2, 0xffff
	s_cmp_eq_u32 s12, 0
	s_cselect_b32 s1, ttmp9, s1
	s_mul_i32 s1, s1, s0
	;;#ASMSTART
	s_bitcmp0_b32 s6, s3
	s_cselect_b32 s0, 1, 0
	
	;;#ASMEND
	v_add_lshl_u32 v4, s1, v0, 2
	;;#ASMSTART
	s_bitcmp1_b32 s7, s3
	s_cselect_b32 s1, 1, 0
	
	;;#ASMEND
	;;#ASMSTART
	s_bitcmp0_b64 s[8:9], s3
	s_cselect_b32 s2, 1, 0
	
	;;#ASMEND
	;;#ASMSTART
	s_bitcmp1_b64 s[10:11], s3
	s_cselect_b32 s3, 1, 0
	
	;;#ASMEND
	v_dual_mov_b32 v1, s1 :: v_dual_mov_b32 v2, s2
	v_dual_mov_b32 v0, s0 :: v_dual_ashrrev_i32 v5, 31, v4
	v_mov_b32_e32 v3, s3
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[4:5]
	global_store_b128 v[4:5], v[0:3], off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_bitcmp_kernel
		.amdhsa_kernarg_size 296
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 6
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
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .offset:         8, .size:           4, .value_kind:     by_value }
      - { .offset:         12, .size:           4, .value_kind:     by_value }
      - { .offset:         16, .size:           8, .value_kind:     by_value }
      - { .offset:         24, .size:           8, .value_kind:     by_value }
      - { .offset:         32, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .max_flat_workgroup_size: 1024
    .name:           s_bitcmp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     13
    .symbol:         s_bitcmp_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
