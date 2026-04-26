; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=buffer_atomic_add_u32_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for the gfx11+/gfx1250 VBUFFER buffer-atomic add
; (`buffer_atomic_add_u32`).  The kernel is dispatched via the
; existing AtomicRMW lowering branch in transpiler/handle_mubuf.cpp
; (`if (sop >= BUFFER_ATOMIC_ADD && sop <= BUFFER_ATOMIC_PK_ADD_F16)
; { ... AtomicRMWInst::Add ... }`); the new bit is the
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
;   1. The buffer atomic lifts to an `atomicrmw add` IR instruction
;      (not to an `@llvm.amdgcn.raw.buffer.atomic.add` intrinsic
;      call) — the existing handler models all the commutative
;      buffer atomics as AtomicRMW so the gfx942 backend can
;      re-lower to whichever native form fits the target ISA.
;   2. Ordering is `monotonic` — the `scope:SCOPE_DEV` modifier
;      on gfx12 maps to monotonic in the existing handler (no
;      acquire/release/seq_cst implied by SCOPE_DEV alone).
;   3. NO refusal diagnostic in stderr — exit 0, not the legacy
;      `unsupportedOpcode [MUBUF]` path.
;
; What we don't pin: the exact pointer-construction shape upstream
; of the atomic (the SRD-to-pointer translation goes through the
; raw_buffer descriptor decoder which has its own dedicated
; coverage); nor the address space of the resulting pointer (the
; backend chooses based on the SRD).

; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_add_u32_kernel(

; The atomic itself: `atomicrmw add` with monotonic ordering.
; Using a loose match on the pointer + value operands because their
; SSA names depend on the kernarg lowering pipeline upstream.
; CHECK: atomicrmw add ptr {{.*}} monotonic

; Negative pin: no `raw.buffer.atomic` intrinsic should appear —
; the handler models all commutative buffer atomics as AtomicRMW
; rather than as a buffer-intrinsic call.  A regression that
; routed BUFFER_ATOMIC_ADD through the intrinsic path would be a
; semantic change worth catching here.
; CHECK-NOT: call {{.*}}@llvm.amdgcn.raw.buffer.atomic.add
; CHECK-NOT: call {{.*}}@llvm.amdgcn.struct.buffer.atomic.add

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
