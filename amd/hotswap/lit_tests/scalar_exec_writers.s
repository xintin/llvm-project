; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir=divergent_exec_kernel 2>/dev/null | %FileCheck %s
;
; Audit: V_CMPX, S_AND_B64→EXEC, and S_MOV_B64→EXEC all route their
; EXEC writes through `storeExec` (handle_valu.cpp / handle_sop2.cpp /
; handle_sop1.cpp respectively). Post-mem2reg, the observable is that
; the `lshr i64 <exec>, %spe_lane_mod` in the SPE active-bit
; computation following each writer consumes the NEW EXEC SSA value
; (not the previous one).
;
; Concretely the fixture produces, in order:
;
;   global_store_dword        (initial under EXEC = -1)
;   v_cmpx_lt_u32_e64 ...     <-- writes EXEC to `%cmpx_exec`
;   global_store_dword        (under %cmpx_exec)
;   s_mov_b64 exec, -1        <-- writes EXEC back to -1
;   v_cmpx_ge_u32_e64 ...     <-- writes EXEC to `%cmpx_exec64`
;   v_cmp_lt_u32_e64 s[4:5]   (SGPR mask — NOT EXEC)
;   s_and_b64 exec, exec, s   <-- writes EXEC to `%and64`
;   global_store_dword        (under %and64)
;   s_mov_b64 exec, -1        <-- writes EXEC back to -1
;
; We assert the lshr following each side-effectful store uses the
; right EXEC SSA value.

; CHECK-LABEL: define amdgpu_kernel void @divergent_exec_kernel(

; Initial store is under full EXEC. We match any `store i32 <val>`
; irrespective of whether <val> is a constant or an SSA name.
; CHECK:       lshr i64 -1, %{{[^ ]+}}
; CHECK:       store i32 {{.*}}, ptr addrspace(1) %{{[^ ]+}}, align 4

; After the first v_cmpx, the EXEC SSA value becomes %cmpx_exec.
; This is the most important signal: the V_CMPX handler routed its
; write through storeExec, mem2reg promoted the alloca to SSA, and
; the next SPE diamond correctly consumes the narrowed mask.
; CHECK:       %cmpx_exec = and i64 -1, %{{[^ ]+}}
; CHECK:       lshr i64 %cmpx_exec, %{{[^ ]+}}

; Store under the narrowed EXEC (valA = 0xAA = 170).
; CHECK:       store i32 {{.*}}, ptr addrspace(1) %{{[^ ]+}}, align 4

; Between region 1 and region 2 the fixture issues
; `s_mov_b64 exec, -1` to restore EXEC. We audit the S_MOV_B64 →
; EXEC handler path by observing the SECOND v_cmpx's `and i64 -1,
; %...` consumes the constant `-1` — meaning the SSA graph reset
; cleanly after the mov. If S_MOV_B64's handler did NOT route its
; write through `storeExec`, this would still read `%cmpx_exec`
; (the mask from region 1) and we'd see
; `%cmpx_exec64 = and i64 %cmpx_exec, %...` instead.
; CHECK:       %cmpx_exec{{[0-9]+}} = and i64 -1, %{{[^ ]+}}

; s_and_b64 exec, exec, s[4:5] writes EXEC to %and64. This proves
; the S_AND_B{32,64} handler on an EXEC destination routes through
; storeExec — same SSA-level contract as the V_CMPX path.
; CHECK:       %and64 = and i64 %{{[^ ]+}}, %{{[^ ]+}}
; CHECK:       lshr i64 %and64, %{{[^ ]+}}

; Final store under the %and64 mask (valB = 0xBB = 187).
; CHECK:       store i32 {{.*}}, ptr addrspace(1) %{{[^ ]+}}, align 4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	divergent_exec_kernel
	.p2align	8
	.type	divergent_exec_kernel,@function
divergent_exec_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v3, 0
	v_lshlrev_b32_e32 v2, 2, v0
	v_mov_b32_e32 v1, 0xaa
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[4:5], s[0:1], 0, v[2:3]
	v_mov_b32_e32 v2, 0xbb
	;;#ASMSTART
	global_store_dword v[4:5], v3, off
	s_waitcnt vmcnt(0)
	v_cmpx_lt_u32_e64 exec, v0, 16
	global_store_dword v[4:5], v1, off
	s_waitcnt vmcnt(0)
	s_mov_b64 exec, -1
	v_cmpx_ge_u32_e64 exec, v0, 16
	v_cmp_lt_u32_e64 s[4:5], v0, 32
	s_and_b64 exec, exec, s[4:5]
	global_store_dword v[4:5], v2, off
	s_waitcnt vmcnt(0)
	s_mov_b64 exec, -1
	
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel divergent_exec_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 6
		.amdhsa_accum_offset 8
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           divergent_exec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         divergent_exec_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
