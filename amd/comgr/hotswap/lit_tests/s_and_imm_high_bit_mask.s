; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --disable-wave-native \
; RUN:     --emit-ir=s_and_imm_high_bit_mask_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Pins `readOpExecWidth`'s immediate-operand path for wave32 sources
; on literals with bit 31 set — the shape this kernel emits is
;   v_cmp_ge_f32_e64 s2, |val|, 0.5
;   s_and_b32 s2, s2, 0xFFFF0000
;   v_cndmask_b32_e64 result, 0, 1, s2
; After the topk commit (`transpiler: preserve SOP2 immediate-mask
; shadow propagation`, 96cbb53e20) routed non-register SOP2 sources
; through `tryGetSrcWaveMaskI1` -> `srcExecWidth` ->
; `readOpExecWidth`, the immediate path became live on wave32
; sources. An earlier `ConstantInt::getSigned(i32Ty, imm)` then
; tripped APInt's signed-range assertion on `imm >= 0x80000000`
; because the raw `int64_t` container decodes `0xFFFF0000` as
; `+4294901760`, outside the signed i32 range `[-2^31, 2^31 - 1]`.
;
; Distinct from the topk fixture `s_xor_imm_mask_shadow`: that uses
; literal `-1` which MC decodes as `int64_t(-1)` (inside the signed
; i32 range and therefore valid for `getSigned`). This test pins
; the cross into the bit-pattern region that the topk fixture does
; not cover.
;
; The fix treats the MC literal as an unsigned bit pattern and
; materialises it via `ConstantInt::get(..., IsSigned=false)`,
; matching `readOp32`'s existing contract. The IR below witnesses
; that the immediate flowed through `readOpExecWidth`'s extraction
; helpers: `mask_at_lane` is an `lshr` of a large NEGATIVE i64
; constant (LLVM's signed print of the replicated `0xFFFF0000` /
; `0xFFFF0000_FFFF0000` bit pattern) against the per-lane index —
; the exact shape the ModRep widen-to-exec path produces when given
; an i32 `0xFFFF0000` narrow.

; CHECK-LABEL: define amdgpu_kernel void @s_and_imm_high_bit_mask_kernel(

; Producer v_cmp_ge on the absolute-value source.
; CHECK: %vcmpf = fcmp oge float %{{[^,]+}}, 5.000000e-01

; Immediate-source shadow propagation: `readOpExecWidth` on the
; `0xFFFF0000` literal must extract a per-lane i1 via
; `extractLaneBitFromWaveMask`. LLVM's signed print of
; `0xFFFF0000_FFFF0000` (the ModRep-widened wave-mask) is
; `-281470681808896`; the `lshr` walks the per-lane bit out of
; that constant. The negative sign in the signed print is the
; witness that the fix's unsigned-bit-pattern materialisation
; landed correctly (a regression to `ConstantInt::getSigned`
; would crash here before ever emitting this IR).
; CHECK: %mask_at_lane = lshr i64 -281470681808896, %mask_lane_idx
; CHECK: %mask_lane_bit = and i64 %mask_at_lane, 1
; CHECK: %mask_lane_i1 = icmp ne i64 %mask_lane_bit, 0

; The narrow SGPR `and` preserves the bit pattern.  LLVM prints
; `0xFFFF0000` as `-65536` (signed i32), which is the canonical
; form of the 32-bit literal as a ConstantInt after the
; bit-pattern fix.
; CHECK: %and = and i32 %{{[^,]+}}, -65536

; Derived per-lane i1 is `%vcmpf AND %mask_lane_i1` — the
; shadow-propagation product that the downstream v_cndmask
; consumes directly (no fallback narrow-SGPR extraction).
; CHECK: %wave_mask_and = and i1 %vcmpf, %mask_lane_i1
; CHECK: %cndmask = select i1 %wave_mask_and, i32 1, i32 0

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_and_imm_high_bit_mask_kernel
	.p2align	8
	.type	s_and_imm_high_bit_mask_kernel,@function
s_and_imm_high_bit_mask_kernel:         ; @s_and_imm_high_bit_mask_kernel
; %bb.0:
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	global_load_b32 v1, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_cmp_ge_f32_e64 s2, |v1|, 0.5
	s_and_b32 s2, s2, 0xFFFF0000
	v_cndmask_b32_e64 v1, 0, 1, s2
	
	;;#ASMEND
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_and_imm_high_bit_mask_kernel
		.amdhsa_kernarg_size 16
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
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           s_and_imm_high_bit_mask_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         s_and_imm_high_bit_mask_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
