; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=smem_kernarg_const_delta_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Regression pin for issue #21 ("SMEM handler ignores constant
; offsets applied to the kernarg-segment-ptr SGPR, silently
; miscompiling Tensile UniversalArgs kernels"). See the matching
; .hip fixture for the full narrative; the short version is:
;
;   Before the fix, `s_add_u32 s0, s0, 0x10 ; s_addc_u32 s1, s1, 0
;   ; s_load_b64 s[2:3], s[0:1], 0` raised to an
;   `extractKernargDword(0)` pair — i.e. the raiser pulled kernarg
;   bytes [0,8) (arg0, arg1) into s[2:3] instead of kernarg bytes
;   [16,24) (the low/high dwords of arg4, the output pointer). Every
;   Tensile UniversalArgs kernel (gfx1250 BF16/BF8/FP8/I32/MFMA) was
;   silently miscompiling through this path.
;
;   After the fix, the `raise_context.hpp::KernargPtrDelta` tracker
;   records the 64-bit const delta `+16` advanced by the s_add/
;   s_addc pair, and `handle_smem.cpp` threads it through so the
;   `s_load_b64 s[2:3], s[0:1], 0` correctly routes to
;   `extractKernargDword(16)` and `extractKernargDword(20)` — i.e.
;   s2 = ka_lo(arg4), s3 = ka_hi(arg4).
;
; INVARIANTS PINNED:
;
;   1. The kernel signature decomposes to four i32 scalars followed
;      by a `ptr addrspace(1)` — the canonical AMDGPU kernarg ABI
;      for `(unsigned int, unsigned int, unsigned int, unsigned int,
;      unsigned int *)`. Any regression that collapses the layout
;      to a single by_value aggregate or reorders the slots would
;      also move the byte offset of `arg4` and invalidate the
;      downstream pin.
;
;   2. The shifted `s_load_b64 s[2:3], s[0:1], 0` correctly reaches
;      kernarg bytes [16, 24), materialising %arg4's low/high
;      dwords into s2/s3. The raiser emits:
;
;        %ka_p2iN = ptrtoint ptr addrspace(1) %arg4 to i64
;        %ka_loN  = trunc i64 %ka_p2iN to i32
;        %ka_p2iN = ptrtoint ptr addrspace(1) %arg4 to i64
;        %ka_hi_shrN = lshr i64 %ka_p2iN, 32
;        %ka_hiN     = trunc i64 %ka_hi_shrN to i32
;
;      and stores `ka_loN` to s2's alloca and `ka_hiN` to s3's.
;      After mem2reg the SGPR alloca file collapses; the loaded
;      values flow into the downstream `s_mov_b32 %[lo], s2 ;
;      s_mov_b32 %[hi], s3` which read s2 and s3 — observable in
;      the raiser's per-VGPR phi-under-EXEC shape (the `out[tid*2+0]
;      = lo ; out[tid*2+1] = hi` stores each take one of these).
;
;      The `%arg4` references in these phi arms are the load-
;      bearing pin: a regression that mis-routed the shifted load
;      (e.g. picking kernarg offset 0 rather than 16, or picking
;      wrong-direction dwords) would change the operand of one of
;      these phis from `%ka_lo*`/`%ka_hi*` (both derived from
;      `%arg4`) to `%arg0`/`%arg1` (kernarg offsets 0/4) and fail
;      this pin.
;
;   3. No refusal diagnostic on stderr: the raiser must NOT hit the
;      new "kernarg-pair SGPR is mid-64-bit-add" / "been modified
;      in a way the raiser cannot fold to a 64-bit constant delta"
;      refusals on this canonical shape. With stderr redirected to
;      /dev/null the test would fail via the CHECK pins below (no
;      kernel body emitted) if a refusal fired.
;
; WHAT WE DO NOT PIN:
;
;   * The exact basic-block names (`%spe_do{N}`) on the phi RHS —
;     the raiser's emitUnderExec helper renumbers these as new SPE
;     scopes are added; we use `{{[a-zA-Z_0-9]+}}` regex placeholders.
;   * The exact SSA names of intermediate `ka_lo*` / `ka_hi*` /
;     `ka_p2i*` / `ka_hi_shr*` values — those are LLVM-IR-printer-
;     renumbered across raiser revisions. We pin the `ka_lo` /
;     `ka_hi` prefix only.
;   * The downstream global stores — the compiler's choice of how
;     to pack the `out[0..2]` writes is not part of the SMEM-delta
;     handler contract.

; CHECK-LABEL: define amdgpu_kernel void @smem_kernarg_const_delta_kernel(
; CHECK-SAME: i32 %arg0
; CHECK-SAME: i32 %arg1
; CHECK-SAME: i32 %arg2
; CHECK-SAME: i32 %arg3
; CHECK-SAME: ptr addrspace(1) %arg4

; The shifted `s_load_b64 s[2:3], s[0:1], 0` must resolve to
; kernarg byte 16 = the low/high dwords of %arg4. Both %ka_lo* and
; %ka_hi* (derived from ptrtoint(%arg4)) must appear on the active
; (SPE do-block) arm of the VGPR phis that flow from s2/s3 into the
; subsequent global stores. CHECK-DAG accepts either s2/s3 ordering
; (stable across mem2reg passes isn't guaranteed).
; CHECK-DAG: phi i32 [ %ka_lo{{[0-9]*}}, %{{[a-zA-Z_0-9]+}}
; CHECK-DAG: phi i32 [ %ka_hi{{[0-9]*}}, %{{[a-zA-Z_0-9]+}}

; Negative pin: the pre-fix failure mode routed s2/s3 from
; kernarg byte 0 (= %arg0) and kernarg byte 4 (= %arg1) respectively.
; Any phi arm that still references %arg0 / %arg1 on the active
; (SPE do-block) path of the load-result VGPRs means the const-
; delta tracker did not fire.
; CHECK-NOT: phi i32 [ %arg0, %{{[a-zA-Z_0-9]+}}
; CHECK-NOT: phi i32 [ %arg1, %{{[a-zA-Z_0-9]+}}

; Negative pin: a mis-resolution that emitted an `undef` or `0` on
; the active arm of the load-result phis would also indicate the
; kernarg fast path mis-routed the dwords (e.g. out-of-range
; extractKernargDword returning nullptr and the handler silently
; substituting a literal instead of refusing).
; CHECK-NOT: phi i32 [ i32 0, %{{[a-zA-Z_0-9]+}}
; CHECK-NOT: phi i32 [ i32 undef, %{{[a-zA-Z_0-9]+}}

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	smem_kernarg_const_delta_kernel
	.p2align	8
	.type	smem_kernarg_const_delta_kernel,@function
smem_kernarg_const_delta_kernel:        ; @smem_kernarg_const_delta_kernel
; %bb.0:
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[8:9], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_add_co_i32 s4, s5, s4
	;;#ASMSTART
	s_add_u32  s0, s0, 0x10
	s_addc_u32 s1, s1, 0
	s_load_b64 s[2:3], s[0:1], 0
	s_wait_kmcnt 0
	s_mov_b32 s5, s2
	s_mov_b32 s10, s3
	
	;;#ASMEND
	s_add_co_i32 s0, s4, s6
	v_dual_mov_b32 v0, s5 :: v_dual_lshlrev_b32 v3, 3, v0
	s_add_co_i32 s0, s0, s7
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v1, s10 :: v_dual_mov_b32 v2, s0
	global_store_b96 v3, v[0:2], s[8:9]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel smem_kernarg_const_delta_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 11
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
      - .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         4
        .size:           4
        .value_kind:     by_value
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         12
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           smem_kernarg_const_delta_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .symbol:         smem_kernarg_const_delta_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
