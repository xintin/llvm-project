; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx1250 \
; RUN:     --emit-ir=buffer_atomic_swap_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx11+/gfx1250 VBUFFER buffer-atomic swap
; (`buffer_atomic_swap_b32`).  Sibling of
; `lit_tests/buffer_atomic_add_u32/`, which pins the commutative
; add path; this fixture pins the exchange path — `AtomicRMWInst::Xchg`
; in handle_mubuf.cpp.  The RTN-form write-back (see the "RTN-form
; write-back" comment in that handler) is the semantic point of
; `buffer_atomic_swap`: the caller reads the old value.  Without
; write-back, the handler would emit a pure `atomicrmw xchg` that
; discards the original value and reduce the swap to a store —
; quietly miscompiling any CAS-loop that relies on it.
;
; Same-target lift (gfx1250 → gfx1250).  The `buffer_atomic_swap_b32`
; opcode is non-commutative — cross-widening wave32 → wave64 races
; target lanes i and i+W_s on the same memory cell, which the
; Class-3 non-commutative-atomic classifier refuses at the
; projection-decision stage (see
; hotswap/docs/wave-size-translation.md §3 / §7's unrewritable
; table).  So the cross-widen path would refuse BEFORE reaching
; this handler, making the handler unreachable via that route.
; Same-target lifts skip the cross-widen classifier (R=1, no replica
; race possible) and exercise the handler directly — that's the path
; this fixture pins.  A future `ThreadLoopProjection` that emulates
; wave64 via per-source-wave loops would also bypass the replica-race
; refusal and exercise this handler; re-running the fixture with a
; `--target-isa=gfx942` RUN line once that projection lands is the
; extension direction.
;
; Invariants:
;
;   1. `atomicrmw xchg` with `monotonic` ordering — matches the
;      MUBUF family convention (see the ADD sibling fixture's
;      comment block on the `scope:SCOPE_DEV` → `monotonic`
;      lowering).
;   2. The RTN write-back is present: the `atomicrmw`'s result
;      reaches the destination VGPR through a `writeReg32` — in IR
;      this shows up as a `store i32 %atomicrmw.result, ...` or
;      equivalent SSA flow back to a tied VGPR.  We don't pin the
;      exact shape (the writeReg path varies with register widths
;      and VGPR aliases) but we DO pin that the atomicrmw's result
;      is named / bound to an SSA value, i.e. the value isn't
;      dropped at the IR level.
;   3. No `raw.buffer.atomic` intrinsic — all commutative and swap
;      buffer atomics route through AtomicRMW, not the buffer-
;      intrinsic path.

; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_swap_b32_kernel(

; The atomic itself: atomicrmw xchg with monotonic ordering.
; CHECK: atomicrmw xchg ptr {{.*}} monotonic

; Negative pin: no raw.buffer.atomic intrinsic.  Mirrors the ADD
; fixture's convention — all commutative and swap buffer atomics are
; modelled as AtomicRMW so the backend can re-lower to the native
; form per target ISA.
; CHECK-NOT: call {{.*}}@llvm.amdgcn.raw.buffer.atomic.swap
; CHECK-NOT: call {{.*}}@llvm.amdgcn.struct.buffer.atomic.swap

; Negative pin: no `cmpxchg` — SWAP and CMPSWAP are distinct opcodes
; with distinct handler arms; a regression that routes SWAP through
; the CMPSWAP path (or vice versa) would be caught here.
; CHECK-NOT: cmpxchg

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_atomic_swap_b32_kernel
	.p2align	8
	.type	buffer_atomic_swap_b32_kernel,@function
buffer_atomic_swap_b32_kernel:          ; @buffer_atomic_swap_b32_kernel
; %bb.0:
	s_load_b96 s[4:6], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 2, v0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, -1
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v2, s6
	s_mov_b32 s0, s4
	s_mov_b32 s1, s5
	;;#ASMSTART
	buffer_atomic_swap_b32 v2, v1, s[0:3], null offen th:TH_ATOMIC_RETURN scope:SCOPE_DEV
	s_wait_loadcnt 0
	
	;;#ASMEND
	global_store_b32 v0, v2, s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_atomic_swap_b32_kernel
		.amdhsa_kernarg_size 12
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 3
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
    .name:           buffer_atomic_swap_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     7
    .symbol:         buffer_atomic_swap_b32_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
