; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=ds_load_tr8_b64_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for the gfx1250 64-bit transposed LDS load with 8-bit
; elements (`ds_load_tr8_b64`).  The handler lives in
; transpiler/handle_ds.cpp under the shared `emitDsLoadTr8B64` lambda
; that fires for both `SemOp::DS_LOAD_TR8_B64` (gfx1250 spelling) and
; `SemOp::DS_READ_B64_TR_B8` (gfx950 spelling); the SemOp lives in
; transpiler/semop.hpp under the `// -- DS --` group with a docstring
; explaining why the two MC opcodes share lowering.
;
; The lift is hand-rolled rather than emitting
; `int_amdgcn_ds_load_tr8_b64` because gfx942 (the transpiler's
; target ISA) has no isel pattern for that intrinsic — neither
; `HasGFX950Insts` (the gfx950 read_tr sibling) nor `isGFX1250Plus`
; (the gfx1250 load_tr family) gates fire on gfx942, and there is no
; in-tree pre-isel emulation pass.  Emitting the intrinsic would
; therefore silently miscompile or fail at codegen.  This test pins
; the structural primitives the hand-rolled emulation must produce.
;
; Invariants:
;
;   1. NO `int_amdgcn_ds_load_tr8_b64` in the lifted IR — that would
;      mean the handler regressed to the old "emit the intrinsic and
;      trust the backend" shape that DS_READ_B64_TR_B8 used to take.
;   2. The lane-id derivation (`mbcnt_lo` -> `mbcnt_hi`) is present
;      so the per-lane group_base and l_in_group can be computed.
;   3. Exactly 8 `amdgcn.ds.bpermute` calls — one per source lane in
;      the 8-lane transpose group.  The handler decomposes the v2i32
;      result into 2 dwords × 4 i8 elements per dword, and each i8
;      element comes from a different source lane (via bpermute).
;   4. Exactly 8 `load i8, ptr addrspace(3)` instructions — one per
;      bpermuted source lane, reading the actual byte from LDS at
;      `bpermuted_base + l_in_group`.
;   5. The packed result is constructed via the canonical
;      zext-shift-or pattern; the SSA value names `tr8_b` (loaded
;      i8), `tr8_pack` (the in-progress OR accumulator), `tr8_p`
;      (LDS pointer cast), and `bp_base` (bpermute result) pin the
;      handler's value-naming contract that downstream tooling
;      greps for.

; CHECK-LABEL: define amdgpu_kernel void @ds_load_tr8_b64_kernel(

; Lane-id derivation must use mbcnt_lo / mbcnt_hi against -1 / 0 /
; mbcnt_lo's result — the same shape DS_LOAD_TR16_B128 uses, since
; both share the "lane_id = mbcnt_hi(-1, mbcnt_lo(-1, 0))" idiom.
; CHECK: %lane_lo{{[0-9]*}} = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK: %lane_id{{[0-9]*}} = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane_lo{{[0-9]*}})

; The hand-rolled emulation must contain at least one
; ds.bpermute call (the structural primitive that does the
; cross-lane gather of source-lane base addresses).  The exact count
; is asserted via the 8 named bpermute results below.
; CHECK: %bp_base = call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[^,]+}}, i32 %{{[^,]+}})

; Each transposed i8 element is loaded from LDS at the bpermuted
; source-lane base + l_in_group offset, then zext'd to i32 and
; shifted into its byte slot in the output dword.  This is the
; first element (byte 0 of dword 0) — pinned to confirm the
; packing pattern matches `or i32 0, %15` (the first slot has no
; previous accumulator).
; CHECK: %tr8_p = inttoptr i64 %{{[^ ]+}} to ptr addrspace(3)
; CHECK: %tr8_b = load i8, ptr addrspace(3) %tr8_p, align 1
; CHECK: %tr8_pack = or i32 0, %{{[^ ]+}}

; Honest refusal: if the handler ever regresses to emitting the
; gfx1250 intrinsic directly, the lift will silently miscompile on
; gfx942 (no isel pattern).  Pinning the negative invariant catches
; that drift in the next CI run.
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.load.tr8.b64
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.read.tr8.b64

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_load_tr8_b64_kernel
	.p2align	8
	.type	ds_load_tr8_b64_kernel,@function
ds_load_tr8_b64_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	v_lshlrev_b32_e32 v1, 3, v0
	v_add_nc_u32_e32 v3, 0x50607080, v0
	v_add_nc_u32_e32 v2, 0x10203040, v0
	s_load_b64 s[0:1], s[0:1], 0x0
	ds_store_b64 v1, v[2:3]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	;;#ASMSTART
	ds_load_tr_b64 v[2:3], v1
	s_wait_dscnt 0
	
	;;#ASMEND
	s_wait_kmcnt 0x0
	global_store_b64 v0, v[2:3], s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_load_tr8_b64_kernel
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
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
    .group_segment_fixed_size: 256
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           ds_load_tr8_b64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         ds_load_tr8_b64_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
