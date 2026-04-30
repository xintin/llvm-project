; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_dual_cndmask_b32_sgpr_cond_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Regression fence for the VOPD `v_dual_cndmask_b32` SGPR-condition
; bug (hotswap/docs/modrep-predicate-chain.md §6.4, fixed 2026-04-22
; in `handle_vopd.cpp`).
;
; PRE-FIX SHAPE (the bug). Both halves of a VOPD `v_dual_cndmask_b32
; ... :: v_dual_cndmask_b32 ...` pair produced a `select` where the
; condition was ALWAYS `ctx.regs.loadVCC(ctx.B)`, regardless of what
; scalar the encoding actually named. When the paired instruction
; wrote `vcc_lo` and the other half's real condition source was `s0`
; (Triton's Kogge-Stone scan at distance 8/16 emits exactly this),
; both halves got the wrong, identical predicate — silently
; miscompiling `canary_bpermute_scan_fp32` (scan output = 2× partial
; sum) and `corpus_layernorm_fp32` (~15% WRONG rows on small-N
; reductions).
;
; POST-FIX SHAPE (this fixture pins). The handler parses
; `operands[3]` and dispatches per-operand:
;
;   * SGPR source (`s<N>`) → prefer `lookupSgprWaveMaskI1(N)`'s
;     fresh `i1` (a V_CMP shadow from the current BB); fall back to
;     `projection.extractLaneBitFromWaveMask` on the raw SGPR alloca.
;     Either path produces a per-lane `i1` distinct from the VCC
;     alloca's `i1`.
;   * VCC source (`vcc_lo` / `vcc`) → `loadVCC` (the i1 stored to the
;     VCC alloca by the preceding V_CMP_*_E32 writer; under LLVM
;     store-load forwarding this collapses to the producer `icmp`
;     directly).
;
; The kernel sets:
;   - `s0 = (tid > 8)` via `v_cmp_lt_u32_e64 s0, 8, tid`
;   - `vcc_lo = (tid > 3)` via `v_cmp_lt_u32_e32 vcc_lo, 3, tid`
;
; then issues one VOPD pair whose halves consume DIFFERENT scalars:
;
;   v_dual_cndmask_b32 %[d0], %[a], %[b], s0           (first half)
;   v_dual_cndmask_b32 %[d1], %[c], %[d], vcc_lo       (second half)
;
; The two halves MUST lower to two distinct select chains whose
; conditions are DIFFERENT per-lane `i1` values — the SGPR path's
; `(tid > 8)` cmp for the first half, the VCC path's `(tid > 3)`
; cmp for the second. A regression that re-introduces the
; hardcoded `loadVCC` for both halves uses the same i1 for both
; selects; the CHECK-NOT pin below catches it.

; CHECK-LABEL: define amdgpu_kernel void @v_dual_cndmask_b32_sgpr_cond_kernel(

; Both V_CMP producers show up in the IR with distinct predicate
; constants (8 for the SGPR path, 3 for the VCC path). Capture each
; SSA name so the CHECKs below can pin the downstream selects.
;
; `v_cmp_lt_u32_e64 s0, 8, tid` lifts to `icmp ult 8, tid`
; (= tid > 8). The e64→SGPR path also populates the fresh-shadow
; cache, which the VOPD-cndmask handler consumes directly.
; CHECK: [[SGPR_CMP:%[[:alnum:]_.]+]] = icmp ult i32 8,
;
; `v_cmp_lt_u32_e32 vcc_lo, 3, tid` lifts to `icmp ult 3, tid`
; (= tid > 3) and stores into the VCC alloca. Under LLVM store-load
; forwarding within a single BB the subsequent VOPD-cndmask
; consumer reads this i1 directly.
; CHECK: [[VCC_CMP:%[[:alnum:]_.]+]] = icmp ult i32 3,

; The VOPD pair's FIRST half consumes the SGPR-path i1. Named
; `vopd_cndmask` by the handler's IRBuilder Twine.
; CHECK: %vopd_cndmask = select i1 [[SGPR_CMP]],

; The VOPD pair's SECOND half consumes the VCC-path i1. Named
; `vopd_cndmask{{[0-9]+}}` (distinct SSA version) — importantly,
; the condition is the DIFFERENT `i1` from the first half. The
; pre-fix bug would have re-used [[SGPR_CMP]] or [[VCC_CMP]] for
; BOTH halves (specifically whatever was most-recently in the VCC
; alloca); the post-fix handler correctly distinguishes them.
; CHECK: %vopd_cndmask{{[0-9]+}} = select i1 [[VCC_CMP]],

; NEGATIVE PIN: no `llvm.amdgcn.cndmask` intrinsic call — the
; handler lowers cndmask to a native LLVM `select`, not an
; AMDGPU-specific intrinsic.
; CHECK-NOT: call i32 @llvm.amdgcn.cndmask

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_dual_cndmask_b32_sgpr_cond_kernel
	.p2align	8
	.type	v_dual_cndmask_b32_sgpr_cond_kernel,@function
v_dual_cndmask_b32_sgpr_cond_kernel:    ; @v_dual_cndmask_b32_sgpr_cond_kernel
; %bb.0:
	s_load_b128 s[4:7], s[0:1], 0x0
	v_dual_add_nc_u32 v1, 1, v0 :: v_dual_add_nc_u32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_nc_u32 v3, 3, v0 :: v_dual_bitop2_b32 v1, 31, v1 bitop3:0x40
	v_and_b32_e32 v2, 31, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v3, 31, v3
	s_wait_kmcnt 0x0
	s_clause 0x3
	global_load_b32 v4, v0, s[6:7] scale_offset
	global_load_b32 v5, v1, s[6:7] scale_offset
	global_load_b32 v6, v2, s[6:7] scale_offset
	global_load_b32 v7, v3, s[6:7] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_cmp_lt_u32_e64 s0, 8, v0
	v_cmp_lt_u32_e32 vcc_lo, 3, v0
	v_dual_cndmask_b32 v2, v4, v5, s0 :: v_dual_cndmask_b32 v3, v6, v7, vcc_lo
	
	;;#ASMEND
	global_store_b64 v0, v[2:3], s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_dual_cndmask_b32_sgpr_cond_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
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
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_dual_cndmask_b32_sgpr_cond_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         v_dual_cndmask_b32_sgpr_cond_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
