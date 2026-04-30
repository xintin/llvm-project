"""`tl.sort` probe on fp32 at (BLOCK_M=32, BLOCK_N=16) with
DETERMINISTIC lane-identifiable input.

Sibling to `canary_tl_sort_fp32_deterministic` (which pinned
the BLOCK_N=32 cross-16 bitonic merge bug) and to
`canary_tl_sort_fp32_n16` (which uses random input).  The N=32
recipes graduated under the `rewrite_permlane16_swap_selfpreserve`
pass (2026-04-23), but N=16 continues to show WRONG 1056/8192
(~12.9%) with random input and the pre-existing WRONG verdict
is stable across the cross-16 fix.  N=16 does NOT use
`v_permlane16_swap_b32` at all (full 16-element sort fits in
one 16-lane half of a wave32, so only `v_mov_b32_dpp` with
quad_perm:[1,0,3,2] / quad_perm:[2,3,0,1] / row_shr:4 /
row_shl:4 / row_shr:8 / row_shl:8 are emitted).  So the N=16
residual is a separate bug class in the DPP cross-[1, 2, 4, 8]
compositions.

With deterministic input `X[r, c] = c.0` the sort output reads
out which LANE each output slot pulled from — any mis-routing
in the DPP network is immediately visible in the output values.

Expected under correct descending sort of [0, 1, ..., 15]:
  output = [15, 14, 13, ..., 1, 0]

Any deviation signature pinpoints the specific DPP stage that's
broken:
  * "two sorted 8-halves": cross-8 (row_shr/row_shl:8) broken
  * "four sorted 4-groups": cross-4 (quad_perm-spanning) broken
  * "pairs swapped wrong": cross-2 / cross-1 broken
"""
import triton
import triton.language as tl


@triton.jit
def _canary_tl_sort_fp32_n16_deterministic(
    Y, stride_ym,
    n_rows,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid * BLOCK_M >= n_rows:
        return
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m[:, None] < n_rows

    # Deterministic input: X[r, c] = c.
    x = tl.broadcast_to(offs_n[None, :].to(tl.float32), [BLOCK_M, BLOCK_N])
    y = tl.sort(x, dim=1, descending=True)

    Y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(Y_ptrs, y, mask=mask_m)


RECIPES = [{
    "name": "canary_tl_sort_fp32_n16_deterministic",
    "kernel_fn":     _canary_tl_sort_fp32_n16_deterministic,
    "kernel_symbol": "_canary_tl_sort_fp32_n16_deterministic",
    "signature": {
        "Y":           "*fp32",
        "stride_ym":   "i32",
        "n_rows":      "i32",
    },
    "constexprs": {"BLOCK_M": 32, "BLOCK_N": 16},
    "harness_constants": {"N_COLS": 16},
    "scalar_args": {
        "stride_ym": "N_COLS",
        "n_rows":    "N_ROWS",
    },
    "num_warps": 4,
    "shape_dim": "N_ROWS",
    "default_shapes": [512],
    "grid": {
        "x": "ceil_div(N_ROWS, BLOCK_M)",
        "y": "1",
        "z": "1",
    },
    "inputs": [],
    "outputs": [
        {"name": "Y", "dtype": "fp32", "elems": "N_ROWS * N_COLS"},
    ],
    "comparator": {"kind": "abs", "tol": 0.0},
}]
