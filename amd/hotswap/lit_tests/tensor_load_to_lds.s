; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=tensor_load_to_lds_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for VIMAGE TENSOR `tensor_load_to_lds_d2`. Pins
; the contractual cross-target loud-failure behaviour of
; transpiler/handle_vimage.cpp under SemOp::TENSOR_LOAD_TO_LDS.
;
; The gfx1250 TENSOR cnt unit (`MIMGInstructions.td:2049-2113`,
; `let SubtargetPredicate = isGFX125xOnly`) has no equivalent on
; gfx942. The handler refuses with `RaiseFailure::unsupportedShape`
; carrying the `VIMAGE` format bucket; the user-rules forbid silent
; fallbacks so a "synth a global_load chain" stub would be a
; regression. The matching LLVM intrinsic
; (`int_amdgcn_tensor_load_to_lds`) is itself gated isGFX125xOnly
; in IntrinsicsAMDGPU.td:4213, so even an intrinsic-emit on a
; non-gfx1250 target would fail at codegen — the principled lift
; is the loud refusal pinned here.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code — the
;      test passes only when raise_cli actually failed).
;   2. The stderr diagnostic from raise_cli names the offending
;      mnemonic (`tensor_load_to_lds`) and the encoding format
;      (`VIMAGE`). raise_cli's failure-line format is fixed
;      (raise_cli.cpp:213): `kernel '<name>' failed to raise:
;      <mnemonic> [<format>]`.
;
; The handler also emits an explicit `transpiler: VIMAGE: ...`
; line that names the architectural mismatch and the
; same-target intrinsic; pinning that line keeps the diagnostic
; text from drifting into something less actionable for users
; who read raise_cli's stderr directly.

; STDERR: transpiler: VIMAGE: tensor_load_to_lds
; STDERR-SAME: gfx1250 TENSORcnt unit
; STDERR-SAME: amdgcn.tensor.load.to.lds

; STDERR: raise_cli: kernel 'tensor_load_to_lds_kernel' failed to raise:
; STDERR-SAME: tensor_load_to_lds
; STDERR-SAME: [VIMAGE]

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=tensor_load_to_lds_kernel 2>&1 | %FileCheck %s --check-prefix=IR
;
; Lift fixture for VIMAGE TENSOR `tensor_load_to_lds_d2` — same-target
; (gfx1250 -> gfx1250) intrinsic-emit path. Pins the principled
; lift in transpiler/handle_vimage.cpp under
; SemOp::TENSOR_LOAD_TO_LDS when `ctx.targetIsa.hasTensorOps` is
; true. Companion fixture to `tensor_load_to_lds.ll`, which pins
; the cross-target (gfx942) loud refusal.
;
; The MIMGInstructions.td:2049-2113 `VIMAGE_TENSOR_Pseudo` operand
; layout for the `_d2` form is `vaddr0:SReg_128, vaddr1:SReg_256,
; r128:imm, cpol:imm`. The matching LLVM intrinsic
; (IntrinsicsAMDGPU.td:4197-4214) is
;
;   void llvm.amdgcn.tensor.load.to.lds(<4 x i32> grp0,
;                                        <8 x i32> grp1,
;                                        <4 x i32> grp2,
;                                        <4 x i32> grp3,
;                                        <8 x i32> grp4_reserved,
;                                        i32 immarg cachepolicy)
;
; The handler marshals each SGPR range into the matching
; `<n x i32>` via `loadSGPR32` + `insertelement`, zero-fills the
; unused `_d2` groups (2 and 3) and the always-reserved group 4,
; and threads the `cpol` immediate through as `i32 0` for the
; corpus encoding (the hand-encoded `D0710001 7C000000 7C7C0428`
; payload sets cpol=0).
;
; We pin two things:
;   1. The eight insertelement chain that materialises group 0 from
;      s40..s43 and group 1 from s4..s11. The fixture's inline-asm
;      clobber list never assigns those SGPRs, so the loads fold to
;      `undef` — the structural shape of the chain is what matters.
;   2. The intrinsic call's argument vector: groups 0/1 are the
;      marshalled SGPR vectors; groups 2/3 are <4 x i32>
;      zeroinitializer; group 4 is <8 x i32> zeroinitializer; cpol
;      is the constant `i32 0`.
;
; Drift indicators:
;   * If a future LLVM rename swaps the intrinsic name (e.g. drops
;     `tensor.` prefix) the IR check fails immediately and pinpoints
;     the rename rather than letting a silently mis-named intrinsic
;     reach the backend.
;   * If the operand-marshalling order changes (group ordering,
;     vector widths, or the reserved-group convention) the
;     insertelement / call-argument shape diverges and FileCheck
;     reports the exact line.

; Group 0: <4 x i32> built from four sequential SGPR reads (s40..s43).
; The first lane seeds the chain off `poison`; the last lane
; (`i64 3`) closes it. LLVM's instnamer suffixes the SSA values
; (`%td_grp0`, `%td_grp02`, ...) so we use a regex on the trailing
; numeric.
; IR: %td_grp0{{[0-9]*}} = insertelement <4 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp0{{[0-9]*}} = insertelement <4 x i32> %td_grp0{{[0-9]*}}, i32 {{.*}}, i64 3

; Group 1: <8 x i32> built from eight sequential SGPR reads (s4..s11).
; IR: %td_grp1{{[0-9]*}} = insertelement <8 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp1{{[0-9]*}} = insertelement <8 x i32> %td_grp1{{[0-9]*}}, i32 {{.*}}, i64 7

; The intrinsic call: groups 2/3 are <4 x i32> zeroinitializer
; (unused for the `_d2` form), group 4 is <8 x i32> zeroinitializer
; (always reserved), and the cpol immediate is `i32 0` for the
; corpus payload.
; IR: call void @llvm.amdgcn.tensor.load.to.lds(
; IR-SAME: <4 x i32> %td_grp0
; IR-SAME: <8 x i32> %td_grp1
; IR-SAME: <4 x i32> zeroinitializer
; IR-SAME: <4 x i32> zeroinitializer
; IR-SAME: <8 x i32> zeroinitializer
; IR-SAME: i32 0

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	tensor_load_to_lds_kernel
	.p2align	8
	.type	tensor_load_to_lds_kernel,@function
tensor_load_to_lds_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	;;#ASMSTART
	.long 0xd0710001
	.long 0x7c000000
	.long 0x7c7c0428
	
	;;#ASMEND
	s_wait_kmcnt 0x0
	global_store_b32 v0, v0, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel tensor_load_to_lds_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 44
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
    .name:           tensor_load_to_lds_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .symbol:         tensor_load_to_lds_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
