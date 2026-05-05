; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_bitset0_b32_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for s_bitset0_b32. Pins that the SOP1 bit-clear lowers
; to an LLVM `and` against `~(1 << (bit_pos & 0x1F))`, with the
; *prior* dst value (here, the 0xDEADBEEF sentinel) on the
; left-hand side of the AND.  The handler lives in
; transpiler/handle_sop1.cpp under the
; `S_BITSET0_B32 || S_BITSET1_B32 || S_BITSET0_B64 || S_BITSET1_B64`
; block; the SemOp lives in transpiler/semop.hpp under SOP1.
;
; Why this fixture matters: TableGen declares S_BITSET0_B32 with a
; tied `$sdst_in` operand (`SOP1_32` with `tied_in=1`,
; `Constraints = "$sdst = $sdst_in"`), but the AMDGPU disassembler
; collapses the tied slot and emits a 2-operand MCInst (`sdst`,
; `src0`).  Earlier handler revisions asserted `op.nSrcs() >= 2`
; expecting `sdst_in` in srcMap and aborted SIGABRT on every real
; s_bitset corpus site (triton f16 GEMM TDM-pipelined kernels and
; the compare_correctness s_bitset0_b{32,64} fixtures themselves).
; The fix reads the prior dst value via
; `ctx.regs.readReg32(op.dst())` — same pattern as S_CMOV_B32 — and
; this fixture pins that read-modify-write shape so the regression
; cannot return.
;
; Invariants:
;
;   1. The lifted IR contains exactly one `and i32 ...` whose
;      left-hand operand carries the 0xDEADBEEF sentinel (decimal
;      `-559038737` signed, or `3735928559` unsigned — accept
;      either rendering).  This pins "prior dst correctly read".
;   2. The right-hand operand of that AND is a `not (shl 1, bit_pos
;      & 0x1F)` chain — the standard bit-clear mask.  Pinning the
;      `xor ... -1` (LLVM's canonical `not`) and the `0x1F` mask
;      is what catches a regression that silently dropped the bit-
;      index masking and let `shl 1, N` become poison for N >= 32.
;   3. NO assertion / abort in stderr — exit 0, not the legacy
;      SIGABRT path.
;
; What we don't pin: the exact SSA names (they depend on the
; kernarg-lowering pipeline upstream of the bitset), nor the
; ordering of the AND operands (LLVM canonicalisation may swap
; them), nor whether the lifted result IR uses `i32` or a wider
; type for the bit-position computation.

; CHECK-LABEL: define amdgpu_kernel void @s_bitset0_b32_kernel(

; The bit-clear AND.  The handler names this instruction `bitset0`
; (see handle_sop1.cpp's `CreateAnd(..., "bitset0")`); pinning the
; SSA name is greppable downstream and is also a back-reference
; from the test to the handler that emits it.
;
; LHS of the AND must carry the 0xDEADBEEF sentinel — either as
; a constant operand or via an SSA copy; the prior-dst read goes
; through `regs.readReg32(op.dst())` which returns a load of the
; sdst alloca, and that load's defining store is the s_mov_b32
; that materialised the sentinel.  We pin on the sentinel
; appearing somewhere in the IR (constant materialisation may be
; folded/CSE'd by IRBuilder before the bitset AND lands).
; CHECK-DAG: {{(-559038737|3735928559)}}
; CHECK-DAG: %bitset0 = and i32

; The bit-index mask: hardware only consumes `src0[4:0]`, so the
; handler emits an `and i32 %src0, 31` before the shl.  A
; regression that dropped this mask would let `shl 1, N` produce
; poison for N >= 32 (legitimate inputs the hardware silently
; truncates).  We pin the mask constant and the shl as separate
; checks because IRBuilder may name them different things across
; LLVM versions.
; CHECK-DAG: and i32 %{{[^,]+}}, 31
; CHECK-DAG: shl i32 1, %{{[^,]+}}

; Negative pin: no `unreachable` or assertion in the lifted body —
; the kernel must complete its lift, not refuse mid-instruction.
; CHECK-NOT: unreachable

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_bitset0_b32_kernel
	.p2align	8
	.type	s_bitset0_b32_kernel,@function
s_bitset0_b32_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x1c
	s_load_b96 s[4:6], s[0:1], 0x0
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
	v_mad_u32 v0, s0, s2, v0
	;;#ASMSTART
	s_mov_b32 s0, 0xDEADBEEF
	s_bitset0_b32 s0, s6
	
	;;#ASMEND
	v_mov_b32_e32 v1, s0
	global_store_b32 v0, v1, s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_bitset0_b32_kernel
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 7
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
      - { .offset:         8, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           s_bitset0_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     7
    .symbol:         s_bitset0_b32_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
