; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_prefetch_b8_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for FLAT `global_prefetch_b8`.  Pins the
; contractual cross-target loud-failure behaviour of
; transpiler/handle_flat.cpp under `SemOp::GLOBAL_PREFETCH_B8`.
;
; The gfx1250 VMEM-prefetch unit (FLATInstructions.td: VFLAT 0x05D
; real `global_prefetch_b8`; `FLAT_Prefetch_Pseudo` with
; `has_vdst = 0`) has no equivalent on gfx942.  The matching LLVM
; intrinsic `int_amdgcn_global_prefetch` (IntrinsicsAMDGPU.td:3211)
; is gated by `HasVmemPrefInsts`, which is only set on gfx1250+
; (LLVM PR #150466 added both the intrinsic + the subtarget feature
; bit; the AMDGPU.td feature line on gfx1250 is `+vmem-pref-insts`,
; absent on every gfx9xx and every other gfx10/11/12 line).  No
; codegen path lowers the intrinsic on a non-gfx1250 backend.
;
; The closest sibling `int_amdgcn_s_prefetch_data` is gated on
; `HasSafeSmemPrefetch`, which is also strictly gfx12-onwards, but
; even on a hypothetical gfx12-as-target compilation we could NOT
; substitute it for `global.prefetch` here: the SMEM prefetch
; requires a UNIFORM SGPR base pointer (the lifter sees a divergent
; per-lane VGPR address derived from `vdiv = saddr + voff_zext`),
; and proving uniformity at lift time would require a divergence
; analysis pass we do not run.  The user-rules forbid silent
; fallbacks; silently dropping the hint would mask both the
; cross-target capability gap AND the resulting pipeline-stall
; regression in any software-pipelined kernel that relied on the
; prefetch overlapping a prior compute chain (the dominant Triton
; corpus shape that introduces this op).  The principled lift on a
; non-gfx1250 target IS the loud refusal.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code — the
;      test passes only when raise_cli actually failed).
;   2. The stderr diagnostic from raise_cli names the offending
;      mnemonic (`global_prefetch_b8`) and the encoding format
;      (`FLAT`).  raise_cli's failure-line format is fixed
;      (raise_cli.cpp:213): `kernel '<name>' failed to raise:
;      <mnemonic> [<format>] @offset=0x<offset> :: <detail>`.
;
; The handler also emits an explicit `transpiler: FLAT: ...` line
; that names the architectural mismatch and the same-target
; intrinsic; pinning that line keeps the diagnostic text from
; drifting into something less actionable for users who read
; raise_cli's stderr directly.

; STDERR: transpiler: FLAT: global_prefetch_b8
; STDERR-SAME: gfx1250 VMEM-prefetch unit
; STDERR-SAME: amdgcn.global.prefetch
; STDERR-SAME: HasVmemPrefInsts

; STDERR: raise_cli: kernel 'global_prefetch_b8_kernel' failed to raise:
; STDERR-SAME: global_prefetch_b8
; STDERR-SAME: [FLAT]

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=global_prefetch_b8_kernel 2>&1 | %FileCheck %s --check-prefix=IR
;
; Lift fixture for FLAT `global_prefetch_b8` — the same-target
; (gfx1250 → gfx1250) intrinsic-emit path.  Pins the principled
; lift in transpiler/handle_flat.cpp under
; `SemOp::GLOBAL_PREFETCH_B8` when `ctx.targetIsa.hasTensorOps` is
; true (the proxy for `FeatureVmemPrefInsts` / `+vmem-pref-insts`,
; both of which are co-resident with the gfx1250-only
; `+gfx1250-insts` feature on the only sub-target that owns the
; encoding).  Companion fixture to `global_prefetch_b8.ll`, which
; pins the cross-target (gfx942) loud refusal.
;
; The FLATInstructions.td `FLAT_Prefetch_Pseudo` multiclass yields
; two operand-shape variants — plain (vaddr:VGPR_64, off:imm,
; cpol:imm) and SADDR (saddr:SGPR_64, vaddr:VGPR_32, off:imm,
; cpol:imm), both with `has_vdst = 0` (the prefetch returns no
; data, only injects a cache-line warm-up hint).  The fixture's
; HIP source uses the clang builtin
; `__builtin_amdgcn_global_prefetch` which lowers to the VFLAT
; 0x05D real (`global_prefetch_b8`); for kernarg-derived pointers
; the AMDGPU disassembler prints the SADDR variant
; `global_prefetch_b8 v0, s[0:1] [offset:N]`).
;
; The matching LLVM intrinsic (IntrinsicsAMDGPU.td:3211, added in
; PR #150466) is
;
;   void llvm.amdgcn.global.prefetch(
;       ptr addrspace(1)        %gaddr,    // captures(none)
;       i32 immarg              %cpol)
;
; The handler:
;   * decodes the global address — for SADDR it issues
;     `add i64 saddr, zext i32 vaddr → i64 (named `saddr_vaddr`)`,
;     for plain it reuses the FLAT decode helper that produces a
;     `i64` from `VGPR_64`;
;   * casts the resulting `i64` to `ptr addrspace(1)` via
;     `inttoptr i64 ... to ptr addrspace(1)`;
;   * folds the FLAT `flat_offset` immediate (when non-zero) onto
;     the pointer via a non-inbounds `getelementptr i8, ptr
;     addrspace(1) %p, i64 <off>` (named `prefetch_addr` in the
;     emitted IR).  This matches the AMDGPU backend's expectation
;     that `flat_offset` re-folds back into the VFLAT real's
;     `offset:` field rather than burning a separate ALU
;     instruction;
;   * threads the FLAT `cpol` immediate through as the trailing
;     `i32 immarg`;
;   * does NOT wrap the call in `ctx.emitUnderExec`: the intrinsic
;     carries the EXEC mask implicitly through
;     `IntrInaccessibleMemOrArgMemOnly` — it is semantically a
;     hint with no observable side effect on inactive lanes, so an
;     extra `if-spe-active` guard would gratuitously inflate IR
;     for what hardware executes as a single broadcast hint.
;
; We pin two concrete sub-shapes from the fixture's pair of
; prefetch calls:
;
;   1. The plain (no-offset) call:
;      * `add i64 saddr, zext_voff   → i64 %saddr_vaddr*`
;      * `inttoptr i64 ...           → ptr addrspace(1)`
;      * `call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) ..., i32 8)`
;
;   2. The byte-offset (flat_offset=256) call:
;      * the same SADDR sum
;      * `getelementptr i8, ptr addrspace(1) ..., i64 256` named
;        `prefetch_addr`
;      * `call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %prefetch_addr, i32 8)`
;
; Drift indicators:
;   * If a future LLVM rename swaps the intrinsic name (e.g.
;     `amdgcn.global.prefetch.b8` to expose the width), the IR
;     check fails immediately and pinpoints the rename rather than
;     letting a silently mis-named intrinsic reach the backend.
;   * If the FLAT-offset folding regresses (e.g. handler emits
;     `add i64` instead of `getelementptr i8`), the AMDGPU backend
;     loses the ability to re-fold `offset:` and the regression
;     surfaces as a stray ALU instruction in the disasm.

; First call: SADDR sum with flat_offset == 0 — no GEP, just the
; intrinsic on the inttoptr'd sum.
; IR: %saddr_vaddr{{[0-9]*}} = add i64
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %{{[0-9]+}}, i32 8)

; Second call: SADDR sum with flat_offset == 256 — non-inbounds
; `getelementptr i8` named `prefetch_addr`.
; IR: %saddr_vaddr{{[0-9]*}} = add i64
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %prefetch_addr{{[0-9]*}} = getelementptr i8, ptr addrspace(1) %{{[0-9]+}}, i64 256
; IR: call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %prefetch_addr{{[0-9]*}}, i32 8)

; The intrinsic declaration must match the upstream signature: the
; `captures(none)` attribute on the pointer argument is what
; permits the AMDGPU backend to safely fold the prefetch through
; passes that move pointers.  Drift here would silently weaken the
; aliasing guarantee.
; IR: declare void @llvm.amdgcn.global.prefetch(ptr addrspace(1) captures(none), i32 immarg)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_prefetch_b8_kernel
	.p2align	8
	.type	global_prefetch_b8_kernel,@function
global_prefetch_b8_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 2, v0
	s_wait_kmcnt 0x0
	global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
	global_prefetch_b8 v0, s[0:1] offset:256 scope:SCOPE_SE
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_prefetch_b8_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
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
    .name:           global_prefetch_b8_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_prefetch_b8_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
