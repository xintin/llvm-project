; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=scratch_refusal_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for a structurally invalid FLAT scratch source kernel:
; it executes `scratch_*` instructions but its KD does not request any private
; segment allocation. Salmon must not invent scratch backing in that case.
;
; === Why this still refuses ===
;
; Valid scratch lifting requires source KD private backing, surfaced as
; non-zero `.private_segment_fixed_size` plus the corresponding
; `compute_pgm_rsrc2.ENABLE_PRIVATE_SEGMENT` launch state. This fixture
; intentionally omits that state while using scratch opcodes, so the raiser
; refuses with the raw source scratch KD fields in the diagnostic.
;
; === What this fixture asserts ===
;
; Two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code — the
;      test passes only when raise_cli actually failed).
;   2. The stderr diagnostic names the offending mnemonic
;      (`scratch_store_b32`), the encoding format (`FLAT`), and the
;      architecturally-specific rationale — crucially the back-pointer
;      to the `buffer_store_no_scratch_alloca` regression guard, so a
;      future reader tracing the failure has a single anchor for the
;      ABI-level design constraint. Also pins the
;      `flat_scratch_init` mention and the mnemonic-family token
;      (`scratch_*`) so a future diagnostic rewrite that drifted to a
;      generic wording would fail this fixture.
;
; raise_cli's failure-line format is fixed (raise_cli.cpp:213):
;   `kernel '<name>' failed to raise: <mnemonic> [<format>]
;    @offset=0x<offset> :: <detail>`
;
; The handler also emits an explicit `transpiler: FLAT scratch
; refused: ...` line; pinning that line keeps the diagnostic text
; from drifting into something less actionable for users who read
; raise_cli's stderr directly.

; The handler-side refusal line names the family and dumps source KD scratch
; fields.
; STDERR: transpiler: FLAT scratch refused: scratch_store_b32
; STDERR-SAME: private_segment_fixed_size=0
; STDERR-SAME: enable_private_segment=0

; The raise_cli failure line names the mnemonic, the encoding format,
; and the detailed rationale (same source KD fields, independently cited — a
; future split of stderr vs. the returned RaiseFailure must keep both
; sides audit-clean).
; STDERR: raise_cli: kernel 'scratch_refusal_kernel' failed to raise:
; STDERR-SAME: scratch_store_b32
; STDERR-SAME: [FLAT]
; STDERR-SAME: zero private_segment_fixed_size
; STDERR-SAME: source_scratch_kd

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	scratch_refusal_kernel
	.p2align	8
	.type	scratch_refusal_kernel,@function
scratch_refusal_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mul_u32_u24_e32 v1, 3, v0
	;;#ASMSTART
	scratch_store_b32 off, v1, off offset:0
	scratch_load_b32  v1, off, off offset:0
	
	;;#ASMEND
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel scratch_refusal_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           scratch_refusal_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         scratch_refusal_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
