// 16x16x4 f32xf32 -> f32 GEMM probe for v_wmma_f32_16x16x4_f32 (gfx1250
// RDNA4 WMMA, VOP3P opcode 0x05D). The kernel computes, per block,
// one independent 16x16 output tile:
//
//   D[i][j] = sum_{k=0..3} A[i][k] * B[k][j]  +  C[i][j]
//
// The grid is `N` blocks of 32 threads (one Wave32) — each block
// computes one tile from `A[tile][16][4]`, `B[tile][4][16]`,
// `C[tile][16][16]` contiguous in global memory.
//
// Why this recipe: it probes the Salmon transpiler's
// `emitWMMAtoMFMA_F32_16x16x4` cross-target lowering
// (transpiler/wmma_lowering.cpp) against a CPU-computed gold reference.
// On gfx1250 the kernel emits exactly one `v_wmma_f32_16x16x4_f32` per
// tile — the same MC opcode the Tensilelite f32 GEMMs in the kerneldex
// corpus emit (e.g. `TensileLibrary_SS_SS_HA_Bias_SAV_UA_*_gfx1250.co`).
// Salmon translates that WMMA to a ds_bpermute / MFMA cascade on
// gfx942; this harness proves that the cascade produces the correct
// numerical output instead of just "a code object that loads and
// launches".
//
// Register-layout contract (matches the Matrix Instruction Calculator
// equations reproduced verbatim in transpiler/wmma_lowering.cpp):
//
//   A — <2 x float> per Wave32 lane (2 VGPRs):
//       i = lane % 16,  k = 2 * floor(lane / 16) + GPR
//         lanes 0..15   GPR 0 -> k=0,  GPR 1 -> k=1
//         lanes 16..31  GPR 0 -> k=2,  GPR 1 -> k=3
//
//   B — <2 x float> per lane, same layout equation as A but indexed
//       as B[k][j] (lane % 16 is the N dimension column).
//
//   C/D — <8 x float> per lane (8 VGPRs):
//       i = 8 * floor(lane / 16) + GPR,  j = lane % 16
//         lanes 0..15   -> rows 0..7
//         lanes 16..31  -> rows 8..15
//
// The gfx942 fallback: this kernel must ALSO build for the native
// gfx942 .co in compare_correctness's three-way sweep (native /
// legacy / salmon). gfx942 has no WMMA; it has MFMA — but the MFMA
// K=4 f32 variant needs Wave64 data and we only launch 32 threads
// per tile. Rather than author a second cross-lane MFMA decomposition
// here (the transpiler IS that decomposition, and that's what we're
// testing), we fall back to a scalar dot product via shared memory:
// each active lane publishes its A/B fragment into `sA[16][4]` /
// `sB[4][16]`, everyone syncs, then each lane scalar-computes its
// 8 D rows. This produces the same numerics as the CPU gold (both
// are plain f32 scalar matmul with a running accumulator) and is
// therefore a trustworthy native baseline.
//
// Launch contract:
//   gridDim.x  = N        (tile count passed by the harness as its
//                          `N` parameter)
//   blockDim.x = 32       (enforced by the recipe's `validate` hook —
//                          smaller violates the WMMA wave-32 contract,
//                          larger breaks the tile<->block mapping)

#include <hip/hip_runtime.h>

typedef float v2f __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));

extern "C" __global__ void
wmma_f32_16x16x4_gemm(const float *A, const float *B, const float *C,
                      float *D) {
  const unsigned tile = blockIdx.x;
  const unsigned lane = threadIdx.x & 31u;

  const float *tA = A + tile * 16u * 4u;
  const float *tB = B + tile * 4u * 16u;
  const float *tC = C + tile * 16u * 16u;
  float *tD = D + tile * 16u * 16u;

  const unsigned i_ab = lane & 15u;
  const unsigned k_base = 2u * (lane >> 4);
  const unsigned j_ab = lane & 15u;
  const unsigned i_cd_base = 8u * (lane >> 4);
  const unsigned j_cd = lane & 15u;

  v2f a, b;
  a.x = tA[i_ab * 4u + (k_base + 0u)];
  a.y = tA[i_ab * 4u + (k_base + 1u)];
  b.x = tB[(k_base + 0u) * 16u + j_ab];
  b.y = tB[(k_base + 1u) * 16u + j_ab];

  v8f c;
  for (int g = 0; g < 8; ++g)
    c[g] = tC[(i_cd_base + unsigned(g)) * 16u + j_cd];

  v8f d;

#if defined(__gfx1250__)
  d = __builtin_amdgcn_wmma_f32_16x16x4_f32(
      /*a_neg=*/false, a,
      /*b_neg=*/false, b,
      /*c_mod=*/(short)0, c,
      /*a_reuse=*/false,
      /*b_reuse=*/false);
#else
  // Scalar gather-and-multiply fallback; see file-header comment.
  __shared__ float sA[16][4];
  __shared__ float sB[4][16];
  sA[i_ab][k_base + 0u] = a.x;
  sA[i_ab][k_base + 1u] = a.y;
  sB[k_base + 0u][j_ab] = b.x;
  sB[k_base + 1u][j_ab] = b.y;
  __syncthreads();

  for (int g = 0; g < 8; ++g) {
    unsigned i = i_cd_base + unsigned(g);
    float acc = c[g];
    for (int k = 0; k < 4; ++k)
      acc += sA[i][k] * sB[k][j_cd];
    d[g] = acc;
  }
#endif

  for (int g = 0; g < 8; ++g)
    tD[(i_cd_base + unsigned(g)) * 16u + j_cd] = d[g];
}
