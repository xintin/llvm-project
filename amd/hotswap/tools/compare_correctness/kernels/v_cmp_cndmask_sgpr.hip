// Single-instruction runtime probe for the V_CMP_*_e64 -> SGPR ->
// V_CNDMASK_B32_e64 per-lane-predication idiom. Companion to the
// lit fixture at `lit_tests/v_cmp_cndmask_sgpr/` — the lit file
// pins the raised-IR shape (per-lane extract via the
// WaveProjection's `extractLaneBitFromWaveMask` instead of the
// pre-fix wave-uniform `icmp ne <sgpr>, 0`); this kernel pins the
// runtime behaviour of that lift on gfx942 hardware.
//
// Kernel: for each thread, write `|x| >= 0.5 ? +1.0f : -1.0f` into
// `out[tid]`. The gfx1250 path forces the idiom via inline asm so
// hipcc cannot substitute an e32 -> VCC form; the gfx942 native
// path expresses the same semantics in plain C so native remains
// a CPU-reference-backed self-check.
//
// Pre-fix salmon collapsed the SGPR mask to one wave-uniform `i1`
// (`icmp ne <sgpr-as-integer>, 0`), making every lane of a wave
// pick the same side of the cndmask. On data-dependent inputs
// (random floats in [-1, 1]) roughly half the lanes mis-selected,
// surfacing as ~50% wrong elements here. Post-fix the same-wave
// and modulo-replication paths recover bit-exactly; the wave-
// native cross-widening direction has a separate residual (the
// V_CMP->SGPR narrow-write-and-replicate lossy path documented in
// `handle_valu_vcmp.cpp`) that the obstruction-classifier
// follow-up is meant to refuse rather than miscompile.

#include <hip/hip_runtime.h>

// Clamp absolute value onto [0, 1] before the compare. With the CPU
// reference defined on the same clamped magnitude, both the native
// HIP run and the CPU gold apply the same deterministic rule on the
// boundary (|x| == 0.5), avoiding any wobble from different
// |x|-vs-0.5 tie-break choices the compare might make.
extern "C" __global__ void v_cmp_cndmask_sgpr(const float *x, float *out,
                                              int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  float val = x[tid];
  float result;
#if defined(__gfx1250__)
  asm volatile(
      // Wave32 source: v_cmp_ge_f32_e64 sdst is a single SGPR (s4),
      // not an SGPR pair. The abs modifier on src0 forces the e64
      // encoding so hipcc cannot substitute an e32 -> VCC form.
      "v_cmp_ge_f32_e64 s4, |%[val]|, 0.5\n\t"
      // Per-lane predicated select driven by s4. The e64 form is
      // what lets the mask source be an arbitrary SGPR rather than
      // the implicit VCC an e32 cndmask would read.
      "v_cndmask_b32_e64 %[r], -1.0, 1.0, s4\n\t"
      : [r] "=v"(result)
      : [val] "v"(val)
      : "s4");
#else
  // gfx942 native: produce the same observable output as the
  // gfx1250 asm path. Absolute value is __builtin_fabsf so the
  // native compiler cannot strength-reduce into a sign-bit trick
  // that would compare differently at the boundary.
  result = (__builtin_fabsf(val) >= 0.5f) ? 1.0f : -1.0f;
#endif
  out[tid] = result;
}
