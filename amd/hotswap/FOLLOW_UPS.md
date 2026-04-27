# Follow-ups

## FP MODE and Constrained FP Semantics

Salmon currently preserves many `MODE` register writes at the HWREG level
instead of silently dropping them, but the raised arithmetic IR still uses
ordinary LLVM floating-point operations and intrinsics such as `fadd`, `fmul`,
`fsub`, and `llvm.fma.*`.

That is sufficient for the current compiler-generated GPT-OSS/SGLang blocker:
`s_fmac_f32` is kept fused by lowering to `llvm.fma.f32`, preserving the
architectural `fma(src0, src1, old dst)` operation rather than approximating it
as multiply plus add.

The remaining gap is dynamic FP environment modelling. LLVM's AMDGPU
TableGen marks scalar FP operations as `Uses = [MODE]` and
`mayRaiseFPException`, and the programming manual defines MODE fields for
rounding and denormal handling. A fully general lift of arbitrary binaries
that mutate these fields should either:

- track MODE values and lower affected FP operations with constrained FP
  semantics where LLVM can represent the requested rounding and exception
  behavior, or
- refuse loudly when the kernel changes FP state in a way Salmon cannot model
  faithfully.

Until that exists, adding new FP opcode support should keep operations in the
closest non-approximating LLVM representation available, avoid fast-math flags,
and document any MODE-sensitive assumptions in the opcode fixture or handler.
