"""`tl.sort` probe at BLOCK_N=16 with XOR-1 permuted deterministic input.

Companion to `canary_tl_sort_fp32_n16_deterministic`.  With
`X[r, c] = c ^ 1`, every adjacent pair is pre-swapped (row is
`[1, 0, 3, 2, 5, 4, ..., 15, 14]`).  A correct descending sort
must reorder EVERY pair to produce `[15, 14, ..., 1, 0]`.

If the cross-[1,2,4,8] DPP network has a specific stage broken
in a way that depends on the INPUT pair ordering, this pattern
will expose it where the monotonic deterministic probe doesn't.
The monotonic probe's input is already sorted descending if you
read lane 15 first, so the bitonic network's compare-and-swap
choices all fall one way; XOR-1 forces the opposite branch on
every distance-1 compare.
"""
import triton
import triton.language as tl


@triton.jit
def _canary_tl_sort_fp32_n16_xor1(
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

    # Deterministic XOR-1 permuted input: X[r, c] = c ^ 1.
    x = tl.broadcast_to((offs_n[None, :] ^ 1).to(tl.float32), [BLOCK_M, BLOCK_N])
    y = tl.sort(x, dim=1, descending=True)

    Y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(Y_ptrs, y, mask=mask_m)


RECIPES = [{
    "name": "canary_tl_sort_fp32_n16_xor1",
    "kernel_fn":     _canary_tl_sort_fp32_n16_xor1,
    "kernel_symbol": "_canary_tl_sort_fp32_n16_xor1",
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
