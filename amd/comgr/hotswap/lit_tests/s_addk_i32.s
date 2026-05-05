; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=s_addk_i32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift fixture for `s_addk_i32` (renamed to `s_addk_co_i32` in
; gfx12+ assembly). Pins the correct operand-index contract for
; SOPK_32TIE-class opcodes with tied-def `$src0`: `op.src(1)` is
; the `$simm16` immediate, NOT `op.src(0)` (which aliases the
; tied SGPR).
;
; Regression this fixture guards. A prior revision of the handler
; in `handle_sopk.cpp` read `op.src(0)` expecting the immediate —
; so the add became `dst + dst` (doubling the prior value) instead
; of `dst + K`. For Triton-compiled `layer_norm` the compiler
; emits `s_addk_co_i32 sN, 0x400` as the loop-counter increment;
; the doubled-zero counter stalls the loop forever and the kernel
; hangs. The same bug was latent in `S_MULK_I32` (identical
; TableGen class `SOPK_32TIE`). After the fix, `%addk = add i32
; <prior-dst>, 1024` with the LITERAL 1024 as the second operand.
; The fixture also pins a high-bit simm16 (`0xd000`) as `-12288`;
; hardware sign-extends SOPK simm16 fields.
;
; The `.hip` sibling forces `s_addk_co_i32 %[x], 0x400` via inline
; asm with a `+s` constraint so hipcc cannot substitute a
; different increment (e.g. `s_add_i32 sN, sN, sM`).
;
; CHECK-LABEL: define amdgpu_kernel void @s_addk_i32_kernel(

; Positive: the lift materialises the immediate `1024` (= 0x400)
; as a plain i32 constant in the add, and the first operand is
; some SSA value (`%csel` in the captured IR; the name is
; projection-independent but kept generic here so renames in the
; source don't break the fixture).
;
; `add i32 %<prior-dst>, 1024` with the SECOND operand being the
; literal 1024 is the post-fix shape; the NEGATIVE assertions
; below forbid the pre-fix `add i32 %x, %x` shape.
; CHECK: %addk = add i32 %{{[^,]+}}, 1024

; The SCC overflow computation feeds the same (prior-dst, 1024)
; pair into `sadd_with_overflow`. This pins the SCC contract's
; operand symmetry — if a future rewrite gets the imm arg right
; on the add but wrong on the SCC overflow (or vice versa), this
; CHECK catches the drift.
;
; `s_addk_co_i32` sets SCC from signed overflow.
; CHECK: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %{{[^,]+}}, i32 1024)
; CHECK: %addk{{[0-9]*}} = add i32 %{{[^,]+}}, -12288
; CHECK: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %{{[^,]+}}, i32 -12288)

; NEGATIVE assertions.

; (a) The pre-fix shape was `add i32 %x, %x` with the same SSA
; value on both sides. Forbid that exact shape. FileCheck's
; `[[NAME]]` / same-variable-twice pattern would be ideal but lit
; regex doesn't support back-references natively; use a
; sufficient-guard by forbidding any `add i32 %<sym>, %<sym>`
; where the whole add's second operand is an SSA ref (not a
; constant). A post-fix kernel emits `add i32 %..., 1024`, so no
; `, %` appears as the second operand on any `add i32` that
; produces `%addk`.
; CHECK-NOT: %addk = add i32 %{{[^,]+}}, %{{[^,]+}}

; (b) The SCC overflow intrinsic must not receive two SSA
; operands either (it would happen in lockstep with the add bug).
; CHECK-NOT: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %{{[^,]+}}, i32 %{{[^,]+}})

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_addk_i32_kernel
	.p2align	8
	.type	s_addk_i32_kernel,@function
s_addk_i32_kernel:                      ; @s_addk_i32_kernel
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_and_b32 s3, ttmp6, 15
	s_add_co_i32 s2, s2, 1
	s_getreg_b32 s4, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s2, ttmp9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s3, s3, s2
	s_cmp_eq_u32 s4, 0
	s_cselect_b32 s2, ttmp9, s3
	;;#ASMSTART
	s_addk_co_i32 s2, 0x400
	s_addk_co_i32 s2, 0xd000
	
	;;#ASMEND
	v_add_nc_u32_e32 v1, s2, v0
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_addk_i32_kernel
		.amdhsa_kernarg_size 8
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           s_addk_i32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         s_addk_i32_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
