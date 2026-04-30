; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_lshr_b64_imm_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for the gfx9+/gfx1250 SOP2 64-bit logical right shift
; (`s_lshr_b64`) with an immediate shift count. See SemOp::S_LSHR_B64
; in transpiler/semop.hpp; the matching handler block in
; transpiler/handle_sop2.cpp under `if (sop == SemOp::S_LSHR_B64)`;
; and the SOP2 mapping in transpiler/opcode_map.cpp.
;
; INVARIANTS PINNED:
;
;   1. The kernel signature carries the i64 source-operand argument
;      directly (`i64 %arg1`); no by_value-style decomposition. This
;      pins the typical `s_lshr_b64` corpus shape: shifting a 64-bit
;      kernarg-derived pointer/integer by an immediate.
;
;   2. The lift emits a single `lshr i64 ..., 16`. The handler
;      builds `CreateZExt(op.src(1), i64Ty)` followed by
;      `CreateLShr(...)`, which constant-folds the zext away when
;      src1 is the immediate `16` (the corpus shape). A regression
;      that emitted a 32-bit shift, masked the count with `urem 64`,
;      or routed the immediate through the wrong width would change
;      this exact instruction shape.
;
;   3. NO `zext i32 16 to i64` (the literal would be a sign that the
;      handler is leaving an unfolded zext on a constant — harmless
;      but indicates the immediate-folding path regressed).
;
;   4. NO 32-bit shift on the source value — the .td definition
;      `SOP2_64_32` makes the source 64-bit; a regression to a
;      pair-of-i32 lift would surface as `lshr i32`.

; CHECK-LABEL: define amdgpu_kernel void @s_lshr_b64_imm_kernel(
; CHECK-SAME: ptr addrspace(1) %arg0
; CHECK-SAME: i64 %arg1

; The s_lshr_b64 lift. The `lshr64` value-name is the canonical
; breadcrumb the handler emits (mirrors `shl64` for S_LSHL_B64 and
; `ashr64` for S_ASHR_I64). The shift count is the literal `16`
; from the inline asm.
; CHECK: %lshr64 = lshr i64 %{{[^,]+}}, 16

; Negative pin: no leftover zext-of-constant shape (the immediate
; `16` must constant-fold through the i32→i64 widen).
; CHECK-NOT: zext i32 16 to i64

; Negative pin: the shift must be 64-bit. A regression that lowered
; src0 to a pair of i32s would emit `lshr i32 ...` instead.
; CHECK-NOT: %lshr64 = lshr i32

; Negative pin: no defensive shift-count masking (the no-fallback
; rule rejects `urem` on the shift amount; if the source binary
; supplies an out-of-range count, the lifted IR's poison is the
; correct surfacing of a real source bug).
; CHECK-NOT: urem i64 {{.*}}, 64

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_lshr_b64_imm_kernel
	.p2align	8
	.type	s_lshr_b64_imm_kernel,@function
s_lshr_b64_imm_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x1c
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s3, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_cmp_eq_u32 s3, 0
	s_cselect_b32 s0, ttmp9, s1
	v_mad_u32 v2, s0, s2, v0
	;;#ASMSTART
	s_lshr_b64 s[0:1], s[6:7], 16
	
	;;#ASMEND
	v_mov_b64_e32 v[0:1], s[0:1]
	global_store_b64 v2, v[0:1], s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_lshr_b64_imm_kernel
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
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
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .offset:         8, .size:           8, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           s_lshr_b64_imm_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         s_lshr_b64_imm_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
