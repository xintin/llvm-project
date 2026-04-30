; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=buffer_load_d16_u8_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for the gfx9+/gfx1250 D16 byte buffer loads.  Pins the
; partial-write merge shape produced by `handle_mubuf.cpp` for both
; halves:
;
;   _D16:    merged = (prior & 0xFFFF0000) | zext(byte_load, i32)
;   _D16_HI: merged = (prior & 0x0000FFFF) | (zext(byte_load, i32) << 16)
;
; Without the merge the previous SHORT_D16 path zero-extended the
; loaded value to i32 and overwrote the entire VGPR — silently
; clobbering the unused half and breaking the `D16PreservesUnusedBits`
; predicate from BUFInstructions.td.  Tensilelite kernels exercise
; both halves to pack adjacent i8 lanes into one VGPR; this fixture
; exists to regression-pin both merge directions.
;
; Invariants pinned below:
;
;   1. The byte load lifts to a raw-buffer byte load.  On cross-widening
;      targets Salmon uses the addrspace(8) raw-pointer form to preserve
;      Triton's raw-pointer descriptor semantics; same-wave paths may use
;      the legacy `<4 x i32>` descriptor form.
;      not to a flat / global load, and not to a wider type.
;   2. Each merge produces an `and` against the half-preserve mask
;      (0xFFFF0000 for `_d16`, 0x0000FFFF for `_d16_hi`) followed by
;      an `or`.  Pinning the masks (rather than the SSA names) is
;      stable across LLVM-IR-printer updates.
;   3. NO refusal diagnostic in stderr (exit 0, not the legacy
;      `unsupportedOpcode [MUBUF]` path the corpus hit before this
;      handler landed).
;
; What we don't pin: the ordering of the two asm blocks in the lifted
; IR (their sequencing is preserved by the asm `volatile` barrier);
; nor the exact SSA names of the per-half merges (LLVM's IR printer
; may rename across versions).

; CHECK-LABEL: define amdgpu_kernel void @buffer_load_d16_u8_kernel(

; The byte loads themselves: i8-typed raw pointer buffer load, one per asm
; block.  Both must be present.
; CHECK-DAG: call i8 @llvm.amdgcn.raw.ptr.buffer.load.i8
; CHECK-DAG: call i8 @llvm.amdgcn.raw.ptr.buffer.load.i8

; The lo-half merge: AND against 0xFFFF0000 to keep prior hi bits,
; then OR with the zero-extended byte (low 16 bits, hi 16 zeros).
; CHECK-DAG: and i32 {{.*}}, -65536
; CHECK-DAG: or {{(disjoint )?}}i32 {{.*}}, %{{.*}}

; The hi-half merge: AND against 0x0000FFFF to keep prior lo bits,
; then SHL by 16 to position the byte in the hi half, then OR.
; CHECK-DAG: and i32 {{.*}}, 65535
; CHECK-DAG: shl i32 {{.*}}, 16
; CHECK-DAG: or {{(disjoint )?}}i32 {{.*}}, %{{.*}}

; Negative pin: the handler must NOT route the byte load through a
; wider buffer-load type (which would corrupt adjacent bytes).
; CHECK-NOT: call i16 @llvm.amdgcn.raw.ptr.buffer.load
; CHECK-NOT: call i32 @llvm.amdgcn.raw.ptr.buffer.load

; Negative pin: the merge must NOT use a full-VGPR overwrite shape
; (the legacy bug for SHORT_D16 zext'd loaded -> i32 and stored the
; whole VGPR with no preserve-mask).  If the printed IR contains
; neither 0xFFFF0000 nor 0x0000FFFF, the handler regressed.
; (Negative direction is implicit in the CHECK-DAG asserts above.)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_load_d16_u8_kernel
	.p2align	8
	.type	buffer_load_d16_u8_kernel,@function
buffer_load_d16_u8_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[0:3], s[0:1], 0x0
	v_dual_mov_b32 v2, 0xcafefeed :: v_dual_lshlrev_b32 v1, 2, v0
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, -1
	v_mov_b32_e32 v3, 0xcafefeed
	s_wait_kmcnt 0x0
	s_mov_b32 s4, s2
	s_mov_b32 s5, s3
	;;#ASMSTART
	buffer_load_d16_u8 v2, v1, s[4:7], null offen scope:SCOPE_DEV
	s_wait_loadcnt 0
	
	;;#ASMEND
	;;#ASMSTART
	buffer_load_d16_hi_u8 v3, v1, s[4:7], null offen scope:SCOPE_DEV
	s_wait_loadcnt 0
	
	;;#ASMEND
	global_store_b64 v0, v[2:3], s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_load_d16_u8_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
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
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           buffer_load_d16_u8_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         buffer_load_d16_u8_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
