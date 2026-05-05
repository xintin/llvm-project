; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=setpc_pattern_b_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for s_set_pc_i64 — Pattern B (subroutine return via an
; SGPR pair stashed at the call site). Pins that the SOP1
; indirect set-PC lowers to a `cmp eq + br` cascade enumerating
; the resolved return targets when the static analysis in
; transpiler/setpc_analysis.cpp can match the source SGPR pair to
; a chainTerminator recorded at a call site (the canonical
; getpc + add chain whose value is the return address). The
; raiser also rewrites the chain-terminator s_add_co_ci_u32 to
; store the plain i64 marker `resolvedReturnAddr` (the source-MC
; byte offset of the intended return BB) into the ret-pair — the
; binary PC the SOP2 chain would otherwise yield is unusable, and
; a per-predecessor integer marker lets the downstream cascade
; compare fold across the phi join to a constant branch.
;
; The handler lives in transpiler/handle_sop1.cpp under
; `if (sop == SemOp::S_SET_PC_I64) { ... case IndirectB: }` and
; routes through the file-local `emitEnumeratedDispatch` helper.
; The SemOp doc in transpiler/semop.hpp pins the lowering
; contract; the call-site rewrite hook lives in raiser.cpp's
; main loop on `S_ADDC_U32` (see the post-handler block keyed on
; `setpcAnalysis.chainTerminators`).
;
; Why a cascade instead of `indirectbr` (the very first revision
; of this fixture pinned): LLVM's `FixIrreducible` pass (relied on
; by AMDGPU's structurizer) only handles `UncondBrInst` /
; `CondBrInst` / `CallBrInst` predecessors of an irreducible
; cycle header — `indirectbr` (and `switch`) crash llc with
; "unsupported block terminator" when the dispatch block lands
; inside an irreducible cycle (the dominant tensilelite
; call/return CFG shape). The cascade is FixIrreducible-
; compatible and mem2reg+SCCP+InstCombine-foldable to the same
; final codegen as a fully-folded `indirectbr` whenever the
; chain-rewriter's per-pred marker makes each cmp resolve to a
; constant `i1 true`. See the rationale block on
; `emitEnumeratedDispatch` in handle_sop1.cpp.
;
; Why integer markers instead of `ptrtoint(blockaddress)` (the
; intermediate revision of this fixture pinned): AMDGPU's
; instruction selector has no pattern to materialise a
; `BlockAddress` constant as an i64 register value (there is no
; relocation for "address of arbitrary BB inside a kernel"), so
; any `BlockAddress` SDNode that survives the middle-end pipeline
; aborts llc with
;   `Cannot select: t1: i64 = BlockAddress<@kernel, %bb_N>`.
; The storeSGPR64 hi/lo split defeats SCCP's ability to fold
; `icmp eq ptr (inttoptr (ptrtoint BA to i64)), BA` across phi
; joins in complex CFGs. Using a plain integer marker sidesteps
; ISel entirely: `BlockAddress` only appears as the `label`
; operand of the `br` (which DOES have a codegen pattern), and
; the `icmp eq i64 %marker, <offset>` folds cleanly under
; mem2reg + SCCP + InstCombine on the hot path.
;
; Why these CHECKs:
;   * The first s_set_pc_i64 in the .hip fixture is Pattern A and
;     lowers to `br label %bb_0x30` (the .Lcallee target). That
;     half of the fixture is also pinned here so a Pattern B
;     regression that accidentally mis-classified the Pattern A
;     site as IndirectB would surface as a `dispatch_*_cmp_0`
;     in bb_0x0 instead of a `br`.
;   * The second s_set_pc_i64 is Pattern B and must lower to the
;     enumerated cascade. The destination must be `bb_0x28` —
;     the .Lret label — derived in the .hip header from the
;     chain math:
;        s_get_pc_i64 records s[8:9] = (di.offset + di.size)
;                                    = 0x08 + 4 = 0x0C
;        s_add_co_u32 with imm=28 ⇒ s[8:9] = 0x0C + 28 = 0x28 = 40
;     The analysis records chainTerminators[0x10] = (0x28, 8) and
;     pendingB at 0x38 with retPair=8; Pass 2 builds
;     targetsByPair[8] = [0x28], so the IndirectB site emits
;     `icmp eq i64 %ret_pc_marker, 40` and branches to %bb_0x28
;     on match, otherwise to the trap BB.
;   * `bb_0x28:` confirms the dispatch target block actually
;     exists in the emitted IR.
;   * The dispatch BB names are deterministic — the handler
;     embeds the source-MC byte offset of the dispatching
;     instruction (here 0x38, the offset of the second
;     `s_set_pc_i64`) into both the cmp and the trap BB names so
;     multiple dispatch sites in the same kernel never collide.
;   * The unreachable BB and its `unreachable` terminator are
;     pinned because they're the trap path: if the runtime
;     marker ever fails to match any enumerated target (an
;     analysis invariant violation), control reaches the
;     unreachable rather than silently jumping somewhere bogus.
;     A regression that emitted a fall-through `br label %...`
;     instead of `unreachable` would silently miscompile.
;   * Negative `CHECK-NOT: indirectbr` regression-pins that no
;     `indirectbr` ever leaks back into the lifted IR — the
;     contract is that ALL enumerated-dispatch sites lower to
;     the cascade.
;   * Negative `CHECK-NOT: blockaddress(` regression-pins the
;     ISel-safety fix: the enumerated-dispatch lowering must
;     NEVER emit a `blockaddress` constant into the lifted IR
;     (neither as an ssa value, an `icmp` operand, nor a
;     `ptrtoint` operand) — `BlockAddress` may only appear as a
;     `br`'s label operand (which is rendered as `label %bb_...`
;     in textual IR, not as `blockaddress(...)`).

; CHECK-LABEL: define amdgpu_kernel void @setpc_pattern_b_kernel(

; Pattern A site (first s_set_pc_i64) lowers to a direct br to
; the chain-resolved callee. The exact hex offset of the target
; block is a product of the fixture's instruction layout (pinned
; by hipcc for this fixture) and is regenerated alongside the
; `.s` source — we assert the lowering shape (direct `br label
; %bb_...`) plus the absence of `indirectbr` / `blockaddress`
; below.
; CHECK: bb_0x0:
; CHECK: br label %bb_0x38

; The dispatch's destination block must exist in the emitted IR
; — pin it before the cascade line because LLVM emits BB label
; definitions in CFG order, and the .Lret target appears in the
; function body before the cascade is materialised.
; CHECK: bb_0x30:

; Pattern B site (second s_set_pc_i64) lowers to the enumerated
; cmp+br cascade. The ret-pair is read as i64 (an or of a
; zext-from-lo with a shifted zext-from-hi — the storeSGPR64 hi/lo
; split, seen through mem2reg) and compared against the resolved
; return target's source-MC byte offset as a plain i64 constant.
; With one call site recorded by the analysis, the cascade has
; exactly one cmp step and falls through to
; `dispatch_0x38_unreachable` on miss. The handler names the
; ret-pair load's `or` result `ret_pc_marker` and the cascade BB /
; cmp by the dispatching-instruction offset (0x38); this pins
; both the lowering shape and the marker-comparison direction.
; CHECK: %ret_pc_marker = or i64 %{{[^ ]+}}, %{{[^ ]+}}
; CHECK-NEXT: %dispatch_0x40_cmp_0 = icmp eq i64 %ret_pc_marker, 48
; CHECK-NEXT: br i1 %dispatch_0x40_cmp_0, label %bb_0x30, label %dispatch_0x40_unreachable

; Trap BB at the end of the cascade — control reaches here only
; if the runtime marker fails to match the enumerated target,
; which is impossible by analysis construction.
; CHECK: dispatch_0x40_unreachable:
; CHECK-NEXT: unreachable

; No `indirectbr` may leak back into the lifted IR.
; CHECK-NOT: indirectbr
; No `blockaddress` constant may leak back into the lifted IR —
; every dispatch target must be reached via a plain label-operand
; branch, not via a materialised `BlockAddress` value.
; CHECK-NOT: blockaddress(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	setpc_pattern_b_kernel
	.p2align	8
	.type	setpc_pattern_b_kernel,@function
setpc_pattern_b_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	;;#ASMSTART
	s_get_pc_i64 s[8:9]
	s_add_co_u32 s8, s8, 28
	s_add_co_ci_u32 s9, s9, 0
	s_get_pc_i64 s[10:11]
	s_add_co_u32 s10, s10, 24
	s_add_co_ci_u32 s11, s11, 0
	s_set_pc_i64 s[10:11]
	s_branch 2
	v_mov_b32 v1, 0xCAFE0001
	v_mov_b32 v1, 0xDEAD0001
	s_set_pc_i64 s[8:9]
	
	;;#ASMEND
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel setpc_pattern_b_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 12
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
    .name:           setpc_pattern_b_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         setpc_pattern_b_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
