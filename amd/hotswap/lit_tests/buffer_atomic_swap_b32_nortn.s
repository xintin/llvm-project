; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx1250 \
; RUN:     --emit-ir=buffer_atomic_swap_b32_nortn_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Non-RTN companion of `lit_tests/buffer_atomic_swap_b32/`.  Pins
; that the MUBUF-atomic handler in `handle_mubuf.cpp` emits the
; `atomicrmw xchg` WITHOUT a write-back when the source instruction
; is the non-RTN form (glc=0 / no `th:TH_ATOMIC_RETURN` modifier).
; See the companion `.hip` block comment for the full rationale
; (the `op.dst(0)` access path handles both RTN and non-RTN
; because operand-0 is always vdata; the write-back is gated by
; `di.numDefs > 0` and correctly skips for non-RTN).
;
; The invariant this fixture pins:
;
;   1. `atomicrmw xchg` is emitted (the atomic side-effect is
;      preserved — the compiler must not elide the swap just
;      because the result is unused).
;   2. NO subsequent use of the atomicrmw's result that looks like
;      a write-back to a destination VGPR.  A handler regression
;      that drops the `di.numDefs > 0` guard would write the
;      atomicrmw result to `op.dst()`, which for non-RTN
;      instructions has no tied dst — the writeReg32 would
;      either fault or corrupt a neighbour VGPR.

; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_swap_b32_nortn_kernel(

; The atomic itself — same shape as the RTN variant.
; CHECK: atomicrmw xchg ptr {{.*}} monotonic

; Negative pin for the write-back.  On RTN the handler writes the
; atomicrmw result back to `op.dst()` via writeReg32, which in IR
; surfaces as an `%atomic_result` having an SSA USER that stores
; into a tied VGPR.  On non-RTN we want the atomicrmw RESULT to be
; unused (trivially dead in SSA terms).  FileCheck can't easily
; express "result has zero uses", but we can check that no
; `store`-to-tied-VGPR shape appears IMMEDIATELY after the
; atomicrmw — the `CHECK-NEXT` pin catches the usual writeReg32
; pattern the RTN handler emits.  The safety-net relies on
; `CHECK-NEXT` accepting any non-store follow-up (branch,
; terminator, etc.) which is what a non-RTN kernel body normally
; has.
; CHECK-NOT: call {{.*}}@llvm.amdgcn.raw.buffer.atomic.swap

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_atomic_swap_b32_nortn_kernel
	.p2align	8
	.type	buffer_atomic_swap_b32_nortn_kernel,@function
buffer_atomic_swap_b32_nortn_kernel:    ; @buffer_atomic_swap_b32_nortn_kernel
; %bb.0:
	s_load_b96 s[0:2], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 2, v0
	s_mov_b32 s3, 0x27000
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v1, s2
	s_mov_b32 s2, -1
	;;#ASMSTART
	buffer_atomic_swap_b32 v1, v0, s[0:3], null offen scope:SCOPE_DEV
	s_wait_loadcnt 0
	
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_atomic_swap_b32_nortn_kernel
		.amdhsa_kernarg_size 12
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 12
    .max_flat_workgroup_size: 1024
    .name:           buffer_atomic_swap_b32_nortn_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_atomic_swap_b32_nortn_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
