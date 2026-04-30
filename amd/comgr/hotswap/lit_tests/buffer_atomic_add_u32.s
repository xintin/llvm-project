; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=buffer_atomic_add_u32_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for the gfx11+/gfx1250 VBUFFER buffer-atomic add
; (`buffer_atomic_add_u32`).  The kernel is dispatched via the
; raw-buffer atomic lowering branch in transpiler/handle_mubuf.cpp
; (`llvm.amdgcn.raw.buffer.atomic.add`); the new bit is the
; `VBUF4(BUFFER_ATOMIC_ADD, BUFFER_ATOMIC_ADD)` entry in
; transpiler/opcode_map.cpp that maps the gfx12 `_VBUFFER_*`
; addressing-mode opcodes to the same SemOp the legacy `_OFFSET /
; _OFFEN / _IDXEN / _BOTHEN` MUBUF forms already mapped to.
;
; This regression-pins the failure mode that motivated the change:
; scope_discovery___sum_bitmatrix_rows refused on
; `buffer_atomic_add_u32 v0, v1, s[4:7], null offen` with
; `unsupportedOpcode [MUBUF]` because no opcode_map entry existed
; for `BUFFER_ATOMIC_ADD_VBUFFER_OFFEN`, despite the SemOp +
; handler being present.
;
; Invariants:
;
;   1. The buffer atomic lifts to the raw-buffer atomic intrinsic,
;      preserving the SRSRC descriptor, per-lane vaddr, soffset, and
;      hardware OOB behavior. Lowering through a generic flat
;      `atomicrmw` loses that descriptor-relative address and can
;      fault on kernels such as GPT-OSS `_sum_bitmatrix_rows`.
;   2. NO refusal diagnostic in stderr — exit 0, not the legacy
;      `unsupportedOpcode [MUBUF]` path.
;
; What we don't pin: the exact pointer-construction shape upstream
; of the atomic (the SRD-to-pointer translation goes through the
; raw_buffer descriptor decoder which has its own dedicated
; coverage); nor the address space of the resulting pointer (the
; backend chooses based on the SRD).

; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_add_u32_kernel(

; The atomic itself: a raw-buffer atomic intrinsic. Use a loose match
; on the operands because their SSA names depend on kernarg lowering.
; CHECK: call i32 @llvm.amdgcn.raw.buffer.atomic.add

; Negative pin: do not lower through a flat pointer atomic.
; CHECK-NOT: atomicrmw add

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_atomic_add_u32_kernel
	.p2align	8
	.type	buffer_atomic_add_u32_kernel,@function
buffer_atomic_add_u32_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b96 s[0:2], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 2, v0
	s_mov_b32 s3, 0x27000
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v1, s2
	s_mov_b32 s2, -1
	;;#ASMSTART
	buffer_atomic_add_u32 v1, v0, s[0:3], null offen scope:SCOPE_DEV
	s_wait_loadcnt 0
	
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_atomic_add_u32_kernel
		.amdhsa_kernarg_size 12
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
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
    .kernarg_segment_size: 12
    .max_flat_workgroup_size: 1024
    .name:           buffer_atomic_add_u32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_atomic_add_u32_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
