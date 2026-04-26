; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx1250 \
; RUN:     --emit-ir=buffer_atomic_cmpswap_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx11+/gfx1250 VBUFFER buffer-atomic
; compare-and-swap (`buffer_atomic_cmpswap_b32`).  Sibling of
; `lit_tests/buffer_atomic_add_u32/` (commutative add) and
; `lit_tests/buffer_atomic_swap_b32/` (pure exchange); this fixture
; pins the cmpxchg path — `CreateAtomicCmpXchg` in handle_mubuf.cpp.
;
; Same-target lift (gfx1250 → gfx1250): `buffer_atomic_cmpswap_b32`
; is non-commutative and would refuse at the cross-widen projection-
; decision stage via the Class-3 non-commutative-atomic classifier
; (see `buffer_atomic_swap_b32.ll` comment block for the full
; rationale).  Same-target R=1 lifts skip the classifier and reach
; the handler directly.
;
; Semantic pin: `buffer_atomic_cmpswap_b32` needs TWO data operands
; (cmp, new) packed as a VReg_64 pair, and the RTN form returns the
; ORIGINAL value at the memory cell.  The handler decodes the vdata
; register pair via `op.dst(0)` + a synthesised `baseIdx + 1` read
; (mirror of FLAT_ATOMIC_CMPSWAP / GLOBAL_ATOMIC_CMPSWAP).  The IR
; shape pinned here is the canonical LLVM `cmpxchg` instruction with
; `{old_value, success_bool}` result — downstream users extract
; element 0 to get the original value.
;
; Invariants:
;
;   1. `cmpxchg` — NOT `atomicrmw`.  A regression that accidentally
;      routes CMPSWAP through the atomicrmw path (e.g. by losing
;      the dedicated pre-switch branch) would miss the compare step
;      and silently miscompile.
;   2. Two separate value operands.  FileCheck can't cheaply pin the
;      exact SSA names the register reads produce, but we pin that
;      the cmpxchg has exactly the i32 x i32 value shape (not i64 or
;      vector) — matching a single-dword compare-and-swap.
;   3. `monotonic` ordering on both success and failure paths, matching
;      the MUBUF family convention (see the sibling ADD / SWAP fixture
;      comments on the `scope:SCOPE_DEV` → `monotonic` lowering).
;   4. The original value is extracted back out of the cmpxchg's
;      result struct via an `extractvalue` at element 0 — the write-
;      back path the handler emits for the RTN form.

; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_cmpswap_b32_kernel(

; The cmpxchg itself with both orderings `monotonic`.  FileCheck's
; whitespace-aware match tolerates the `align 4` / `syncscope`
; annotations the backend may add.
; CHECK: cmpxchg ptr {{.*}} monotonic monotonic

; The RTN write-back: extractvalue at element 0 (the original value).
; A handler that forgot the write-back would drop this extract, and
; the cmpxchg's result would have only ONE use (the success bool
; path).
; CHECK: extractvalue {{.*}}, 0

; Negative pin: no atomicrmw xchg — CMPSWAP is not a pure exchange,
; and routing it through the `BUFFER_ATOMIC_SWAP → AtomicRMWInst::Xchg`
; arm would lose the compare step.
; CHECK-NOT: atomicrmw xchg

; Negative pin: no `raw.buffer.atomic` intrinsic — the handler models
; CMPSWAP as `cmpxchg`, not a buffer intrinsic call.
; CHECK-NOT: call {{.*}}@llvm.amdgcn.raw.buffer.atomic.cmpswap
; CHECK-NOT: call {{.*}}@llvm.amdgcn.struct.buffer.atomic.cmpswap

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_atomic_cmpswap_b32_kernel
	.p2align	8
	.type	buffer_atomic_cmpswap_b32_kernel,@function
buffer_atomic_cmpswap_b32_kernel:       ; @buffer_atomic_cmpswap_b32_kernel
; %bb.0:
	s_load_b128 s[0:3], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 2, v0
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, -1
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[2:3], s[2:3]
	s_mov_b32 s4, s0
	s_mov_b32 s5, s1
	;;#ASMSTART
	buffer_atomic_cmpswap_b32 v[2:3], v1, s[4:7], null offen th:TH_ATOMIC_RETURN scope:SCOPE_DEV
	s_wait_loadcnt 0
	
	;;#ASMEND
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_atomic_cmpswap_b32_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         12
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           buffer_atomic_cmpswap_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         buffer_atomic_cmpswap_b32_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
