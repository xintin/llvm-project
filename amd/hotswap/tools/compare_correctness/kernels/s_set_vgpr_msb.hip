// Minimal reproducer: force the gfx1250 s_set_vgpr_msb path by
// pressuring the register allocator into using VGPRs 256..511.
//
// Rationale (2026-04-20, salmon kerneldex triage):
//
//   Gfx1250Gpu.Matmul128x128_1tile fails numerically (16 238 errors,
//   maxErr=4.5e+1). The 64x64 version of the same kernel passes. An
//   instruction-level diff of the two HSACOs (llvm-objdump -d) shows
//   that s_set_vgpr_msb appears exactly 3 times in the 128x128 kernel
//   and zero times in the 64x64 kernel. s_set_vgpr_msb is the gfx1250
//   mechanism for naming VGPRs >= 256 in the subsequent ALU
//   instruction (it stores a 2-bit MSB per operand slot; each
//   slot's effective VGPR index is (raw_index + 256 * msb)).
//
//   The transpiler has a computeVGPRAdjust() helper that applies
//   these MSBs when decoding the NEXT instruction after an
//   s_set_vgpr_msb site. The large matmul needs the mechanism to
//   work under WMMA K=4 lowering, where the accumulator tile spans
//   enough VGPRs that the LLVM backend demotes names > 255.
//
//   The 128x128 kernel also uniquely uses v_bitop3_b32, which IS
//   handled (handle_valu.cpp). So the residual axis is almost
//   certainly the MSB path: either (a) the one-shot reset of
//   ctx.vgprMSBs fires too early for a VOPD pair / DPP-modified /
//   multi-operand instruction, (b) the adjustment isn't applied to
//   VMEM address operands, or (c) MSBs are dropped during a
//   classifier refusal that didn't actually refuse.
//
//   This kernel is the minimal direct probe: a pointwise kernel with
//   enough live state that the register allocator is likely to use
//   VGPRs >= 256 on gfx1250, without any WMMA or cross-lane ops in
//   the way.
//
// Design:
//
//   The kernel carries ~80 live f32 values through a long dependency
//   chain. On gfx1250 the compiler emits s_set_vgpr_msb between
//   blocks of the chain as the allocator spills into the upper bank.
//   On gfx942 the compiler bottoms out at 256 VGPRs and spills to
//   scratch, but the semantics are the same pointwise map.
//
//   The CPU reference is an identical scalar replica of the same
//   dependency chain; the harness compares element-wise.
//
//   Wired into the harness as makeSSetVgprMsbRecipe() in
//   compare_correctness.cpp and registered in KERNELS in the
//   Makefile. The CPU reference in that recipe is kept in lock-step
//   with the chain below; update both when changing either.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_set_vgpr_msb_probe(const float *__restrict__ in,
                                                float *__restrict__ out,
                                                int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  float x = in[tid];

  // 64 independent live values that must all be kept alive across the
  // combining phase below. The compiler cannot collapse these at
  // codegen because each uses a distinct non-linear operation (the
  // sign bit flip, the fma, and the multiplicative step all bind
  // against the previous live's value).
  float v[64];
  v[0] = x;
  for (int i = 1; i < 64; ++i) {
    float prev = v[i - 1];
    float step = static_cast<float>(i);
    v[i] = ((i & 1) ? -prev : prev) * 1.001f + step * 1e-3f;
  }

  // Combining phase: each combined value depends on a disjoint pair
  // of the 64 lives, so all 64 must be resident at the same time.
  // On gfx1250 with BLOCK_SIZE=128 this is enough live state to spill
  // past VGPR 255 and force s_set_vgpr_msb sequences.
  float sum = 0.0f;
  for (int i = 0; i < 32; ++i) {
    sum += v[i] * v[63 - i];
  }
  out[tid] = sum;
}
