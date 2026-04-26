; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=ds_load_2addr_b32_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for the gfx11+ two-address LDS load family
; (DS_READ2_B32 / DS_READ2ST64_B32).  The handler dispatches through
; the dedicated `ds2Classify` block in transpiler/handle_ds.cpp;
; the companion `.hip` source documents WHY this fixture exists — in
; short, to pin the correct per-offset-independent lowering so we
; never regress back to the silent-miscompile `load <2 x i32>` shape
; the generic single-offset path produced before the rewrite.
;
; Lowering shape this fixture pins:
;
;   * DS_READ2_B32 offset0:4 offset1:6 emits
;       %ds2_ld0 = load i32, ptr addrspace(3) %..., align 4   ; byte 16
;       %ds2_ld1 = load i32, ptr addrspace(3) %..., align 4   ; byte 24
;     (two independent dword loads at raw-field * 4 byte offsets —
;     NOT a contiguous `<2 x i32>` vector load).
;
;   * DS_READ2ST64_B32 offset0:2 offset1:3 emits
;       %ds2_ld0 = load i32, ptr addrspace(3) %..., align 4   ; byte 512
;       %ds2_ld1 = load i32, ptr addrspace(3) %..., align 4   ; byte 768
;     (×256 byte stride per raw-field unit — the ST64 opcode's
;     distinguishing feature).
;
;   * Both variants route the byte offset through the chain
;       %ds2_addr = zext i32 %{{...}} to i64
;       %ds2_off  = add i64 %ds2_addr, <imm>
;       %ds2_p{0,1} = inttoptr i64 %ds2_off to ptr addrspace(3)
;     where `<imm>` is the per-access byte offset.  Pinning the
;     immediate values {16, 24, 512, 768} catches any regression to
;     the unscaled-raw-field shape the prior handler used.

; CHECK-LABEL: define amdgpu_kernel void @ds_load_2addr_b32_kernel(

; All assertions are CHECK-DAG so the order of the two inline-asm
; sites (DS_READ2_B32 first, DS_READ2ST64_B32 second) is irrelevant —
; the handler emits each site as an adjacent (add + inttoptr + load)
; triple, and either order within the function body is
; semantically valid.
;
; Per-access byte offsets — four distinct immediate values pinning
; both the non-ST64 (x4 scaling) and ST64 (x256 scaling) byte-offset
; arithmetic.  A regression to unscaled raw-field offsets (4, 6, 2,
; 3) would fail to match; a regression to contiguous single-offset
; lifting (only offset0 consumed) would also fail by missing half
; the adds.
; CHECK-DAG: add i64 %{{.*}}, 16
; CHECK-DAG: add i64 %{{.*}}, 24
; CHECK-DAG: add i64 %{{.*}}, 512
; CHECK-DAG: add i64 %{{.*}}, 768

; Four independent dword loads in address space 3 (LDS), each
; carrying the explicit `align 4` the `CreateAlignedLoad` path
; stamps.  Each load is pinned through its DISTINCT pointer SSA
; name — the first inline-asm site's pair is emitted as `%ds2_p0`
; / `%ds2_p1`; the second site re-enters the handler and LLVM's
; value-symbol-table auto-renames the repeated base names to
; `%ds2_p0<N>` / `%ds2_p1<N>` (the `[0-9]+` regex is the
; stability-safe version of that).  Four separate -DAG lines are
; used (rather than a single counted variant) because counted
; directives are order-sensitive under -DAG neighbourhood rules.
; CHECK-DAG: load i32, ptr addrspace(3) %ds2_p0, align 4
; CHECK-DAG: load i32, ptr addrspace(3) %ds2_p1, align 4
; CHECK-DAG: load i32, ptr addrspace(3) %ds2_p0{{[0-9]+}}, align 4
; CHECK-DAG: load i32, ptr addrspace(3) %ds2_p1{{[0-9]+}}, align 4

; Negative pins.  These would indicate a regression back to the
; generic single-offset DS_READ path (contiguous `<2 x i32>` vector
; load) or to the B64 / B128 widths (wrong dsClassify entry firing).
; CHECK-NOT: load <2 x i32>, ptr addrspace(3)
; CHECK-NOT: load <4 x i32>, ptr addrspace(3)
; CHECK-NOT: load i64, ptr addrspace(3)
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.read

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_load_2addr_b32_kernel
	.p2align	8
	.type	ds_load_2addr_b32_kernel,@function
ds_load_2addr_b32_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v1, 0
	;;#ASMSTART
	ds_load_2addr_b32 v[2:3], v1 offset0:4 offset1:6
	s_wait_dscnt 0
	
	;;#ASMEND
	;;#ASMSTART
	ds_load_2addr_stride64_b32 v[4:5], v1 offset0:2 offset1:3
	s_wait_dscnt 0
	
	;;#ASMEND
	s_wait_kmcnt 0x0
	global_store_b128 v0, v[2:5], s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_load_2addr_b32_kernel
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 6
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
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           ds_load_2addr_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         ds_load_2addr_b32_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
