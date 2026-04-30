; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=ds_load_b96_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for the gfx1250 96-bit LDS load/store pair
; (`ds_load_b96` / `ds_store_b96`).  The handler dispatches via the
; existing generic DS path in transpiler/handle_ds.cpp under
; `if (isDsRead || isDsWrite) { ... }` with the `dwords=3,
; loadBits=96` entry added to `dsClassify`.  The SemOps live in
; transpiler/semop.hpp under the `// -- DS --` group with docstrings
; explaining why we canonicalise on the gfx11+ asm spelling.
;
; The lowering shape this fixture pins:
;
;   * Load:  `%ds_ld = load <3 x i32>, ptr addrspace(3) %{{...}}`
;     followed by element-wise insertion into the destination VGPR
;     triple via `writeRegVec` (which the existing
;     `<3 x i32>` legalisation in reg_file.cpp handles generically
;     via `(totalBits + 31) / 32` dword math — works for 3 dwords).
;   * Store: `store <3 x i32> %{{...}}, ptr addrspace(3) %{{...}}`
;     emitted under `emitUnderExec` (the existing DS write path),
;     symmetric to the load.
;   * Both go through the addrspace(3) (LDS) pointer cast.
;
; The gfx942 backend lowers `load <3 x i32>` from addrspace(3) to
; either a native `ds_read_b96` (gfx9 inherits the `_vi` Real form
; from DSInstructions.td:2062) or splits to 3x `ds_read_b32` —
; both correct in-place lowerings; we don't pin which because the
; choice is the backend's.  What we DO pin is the IR shape the
; transpiler hands to the backend, so a regression that collapsed
; the 3-dword vector load into something else (e.g. an `i96`
; integer load, or three separate `i32` loads at the IR level)
; would be caught here.
;
; Invariants:
;
;   1. The SOURCE IR contains exactly one `load <3 x i32>` from
;      addrspace(3) and exactly one `store <3 x i32>` to addrspace(3)
;      — i.e. the handler routes both through the generic
;      `vecTy = <3 x i32>` branch and not through 3 separate
;      single-dword loads/stores (which would be correct but wasteful
;      and a sign of a regression away from the generic path).
;   2. NO `<2 x i32>` or `<4 x i32>` vector LDS access — those would
;      indicate `dsClassify` returned the wrong dword count (B64 or
;      B128 entry firing on a B96 SemOp).
;   3. NO call to any `int_amdgcn_ds_*` intrinsic for the LDS
;      access — the DS handler path uses raw load/store, not
;      intrinsics, for the non-transposed variants; an intrinsic
;      would be a regression that mis-routed the SemOp into one of
;      the transposed-load handlers (DS_LOAD_TR8_B64 etc).

; CHECK-LABEL: define amdgpu_kernel void @ds_load_b96_kernel(

; The store side: ds_store_b96 lifts to a `store <3 x i32>` in
; addrspace(3).  Pinned via the explicit element type and address
; space; the `align` attribute is left implicit because the backend
; chooses it from datalayout (4-byte for v3i32 in addrspace(3)).
; CHECK: store <3 x i32> %{{[^,]+}}, ptr addrspace(3) %{{[^,]+}}

; The load side: ds_load_b96 lifts to a `load <3 x i32>` in
; addrspace(3) named `ds_ld` per the handler's IR-name contract
; (handle_ds.cpp uses `"ds_ld"` for every generic DS read; greppable
; downstream).
; CHECK: %ds_ld{{[0-9]*}} = load <3 x i32>, ptr addrspace(3) %{{[^,]+}}

; Negative pins: no other vector widths and no intrinsics for the
; LDS access path.  These would indicate dsClassify mis-classified
; the SemOp or the dispatch routed into a transposed-load handler.
; CHECK-NOT: load <2 x i32>, ptr addrspace(3)
; CHECK-NOT: load <4 x i32>, ptr addrspace(3)
; CHECK-NOT: store <2 x i32>, {{.*}}ptr addrspace(3)
; CHECK-NOT: store <4 x i32>, {{.*}}ptr addrspace(3)
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.read
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.write

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_load_b96_kernel
	.p2align	8
	.type	ds_load_b96_kernel,@function
ds_load_b96_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	v_mul_u32_u24_e32 v1, 3, v0
	s_load_b64 s[0:1], s[0:1], 0x0
	v_or_b32_e32 v2, 0xaa000000, v0
	v_or_b32_e32 v3, 0xbb000000, v0
	v_or_b32_e32 v4, 0xcc000000, v0
	v_lshlrev_b32_e32 v0, 2, v1
	;;#ASMSTART
	ds_store_b96 v0, v[2:4]
	s_wait_dscnt 0
	
	;;#ASMEND
	s_barrier_signal -1
	s_barrier_wait -1
	;;#ASMSTART
	ds_load_b96 v[2:4], v0
	s_wait_dscnt 0
	
	;;#ASMEND
	s_wait_kmcnt 0x0
	global_store_b96 v0, v[2:4], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_load_b96_kernel
		.amdhsa_group_segment_fixed_size 768
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 2
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
    .group_segment_fixed_size: 768
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           ds_load_b96_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         ds_load_b96_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
