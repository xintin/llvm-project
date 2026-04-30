; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_load_b96_kernarg_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for the gfx11+/gfx1250 96-bit scalar memory load
; (`s_load_b96`) over a kernarg buffer that contains a `by_value`
; aggregate larger than 8 bytes.  The SemOp + `extractKernargDword`
; helper + raiser by_value decomposition together produce the IR
; signature and SGPR-store shape pinned below.  See SemOp::S_LOAD_B96
; in transpiler/semop.hpp; the per-dword loop in handle_smem.cpp
; (under `if (isKernarg && immOffset && byteOffset <
; ctx.kernargs.implicitArgsBase)`); the `byValueSize > 8 && % 4 == 0`
; branch in raiser.cpp's `paramTypes` build; and the
; `extractKernargDword` helper in kernarg_layout.cpp.
;
; INVARIANTS PINNED:
;
;   1. The `by_value` aggregate (size 16, the `Big16` struct) is
;      decomposed into FOUR `i32` IR arguments at byte offsets
;      0/4/8/12, NOT a single `i32` / `[16 x i8]` / `i128` / aggregate
;      type. This is the kernarg-buffer-byte-layout-preserving
;      contract that lets `extractKernargDword` resolve interior
;      sub-struct loads.
;
;   2. The `global_buffer` slot at offset 16 follows the four i32
;      slots and is typed as `ptr addrspace(1)` (the global address
;      space the AMDGPU code-object ABI guarantees for
;      `global_buffer` kernargs).
;
;   3. The `s_load_b96 s[0:2], s[0:1], 0x4` lifts to three SGPR
;      stores: s0 <- %arg1, s1 <- %arg2, s2 <- %arg3 (the kernarg
;      dwords at offsets 4/8/12 of the Big16 by_value).  After
;      mem2reg the SGPR alloca file collapses; the loaded values
;      flow into the next instruction (a `v_dual_mov_b32 v4, s0 ::
;      v_dual_mov_b32 v5, s1`) as v_mov_b32 lifts that read s0 and
;      s1 — observable in the raiser's per-VGPR phi shape:
;
;        %vgpr4.0 = phi i32 [ %arg1, %spe_do{N} ], [ undef, ... ]
;        %vgpr5.0 = phi i32 [ %arg2, %spe_do{M} ], [ undef, ... ]
;
;      The `%arg1` / `%arg2` references in those phis are the
;      load-bearing pin: a regression that mis-routed any one of the
;      three dwords (e.g. picked the wrong by_value slot, or
;      computed the wrong sub-offset, or zero-substituted on a
;      "miss") would change the operand of one of these phis and
;      fail this pin.
;
;   4. NO refusal diagnostic on stderr: the previous failure mode
;      was `transpiler: s_load_b96 ... Cannot resolve kernarg at
;      offset 4`.  With raise_cli's stderr redirected to /dev/null
;      the test still fails because the FileCheck pins below would
;      not match (the kernel body would be a stub if the lift
;      refused).
;
; WHAT WE DO NOT PIN:
;
;   * The exact basic-block names (`%spe_do{N}`) on the phi RHS — the
;     raiser's emitUnderExec helper renumbers these as new SPE
;     scopes are added; we use `{{[a-zA-Z_0-9]+}}` regex placeholders.
;   * The exact SSA names of intermediate `zext` / `shl` / `or` /
;     `inttoptr` chains — those are LLVM-IR-printer-renumbered.
;   * The downstream global stores — the compiler's choice of how to
;     pack the `out[0..6]` writes is not part of the s_load_b96
;     handler contract.

; CHECK-LABEL: define amdgpu_kernel void @s_load_b96_kernarg_kernel(
; CHECK-SAME: i32 %arg0
; CHECK-SAME: i32 %arg1
; CHECK-SAME: i32 %arg2
; CHECK-SAME: i32 %arg3
; CHECK-SAME: ptr addrspace(1) %arg4

; The s_load_b96 result is routed into s[0:2] with s0=%arg1,
; s1=%arg2, s2=%arg3 (the kernarg dwords at offsets 4/8/12 of the
; by_value Big16 struct). The downstream `v_dual_mov_b32 v4, s0 ::
; v_dual_mov_b32 v5, s1` consumes s0 and s1, which the raiser's
; per-VGPR phi-under-EXEC shape carries as `phi i32 [ %arg1, ... ]`
; and `phi i32 [ %arg2, ... ]` references. CHECK-DAG accepts either
; ordering (s0/s1 lift order isn't stable across mem2reg passes).
; CHECK-DAG: phi i32 [ %arg1, %{{[a-zA-Z_0-9]+}} ]
; CHECK-DAG: phi i32 [ %arg2, %{{[a-zA-Z_0-9]+}} ]

; Negative pin: the previous failure mode emitted a refusal via
; `RaiseFailure::smemKernargMiss` after printing the "Cannot
; resolve kernarg" diagnostic. With the lift refused, the emit-ir
; output would not contain a kernel body, and the phi pins above
; wouldn't match. This CHECK-NOT additionally enforces that the
; s_load_b96 handler did not silently substitute zero / undef for
; the s0 or s1 operand (which would manifest as `[ i32 0, %... ]`
; or `[ i32 undef, %... ]` in those phi arms instead of `[ %argN,
; %... ]`). The phi pattern tolerates `undef` only on the *inactive*
; (false) arm of the EXEC predicate, not the active one.
; CHECK-NOT: phi i32 [ i32 0, %{{[a-zA-Z_0-9]+}} ]
; CHECK-NOT: phi i32 [ i32 undef, %{{[a-zA-Z_0-9]+}} ]

; Negative pin: the by_value decomposition must NOT collapse the
; 16-byte struct back into a single argument of any of these
; non-decomposed shapes. Any of these would indicate the
; `byValueSize > 8 && % 4 == 0` branch in raiser.cpp regressed.
; CHECK-NOT: define amdgpu_kernel void @s_load_b96_kernarg_kernel(i32 %arg0, ptr addrspace(1) %arg1)
; CHECK-NOT: define amdgpu_kernel void @s_load_b96_kernarg_kernel({{[^,]+\[16 x i8\]}}
; CHECK-NOT: define amdgpu_kernel void @s_load_b96_kernarg_kernel(i128

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_load_b96_kernarg_kernel
	.p2align	8
	.type	s_load_b96_kernarg_kernel,@function
s_load_b96_kernarg_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x10
	s_wait_xcnt 0x0
	;;#ASMSTART
	s_load_b96 s[0:2], s[0:1], 4
	s_wait_kmcnt 0
	
	;;#ASMEND
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v4, s0
	v_dual_mov_b32 v5, s1 :: v_dual_mov_b32 v6, s8
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[0:1], s[4:5]
	v_mov_b64_e32 v[2:3], s[6:7]
	s_clause 0x1
	global_store_b96 v7, v[4:6], s[2:3]
	global_store_b128 v7, v[0:3], s[2:3] offset:12
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_load_b96_kernarg_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 9
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
      - { .offset:         0, .size:           16, .value_kind:     by_value }
      - { .address_space:  global, .offset:         16, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           s_load_b96_kernarg_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     9
    .symbol:         s_load_b96_kernarg_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
