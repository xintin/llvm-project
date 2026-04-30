"""`tl.sort` probe on fp32 at (BLOCK_M=32, BLOCK_N=4).

Third in the tl.sort size ladder after canary_tl_sort_fp32
(N=32, graduates under the cross-16 selfpreserve rewrite) and
canary_tl_sort_fp32_n16 (N=16, residual 12.9% WRONG — open).

Discriminator for the `topk_forward_bisect_m2_strict` residual,
which calls `tl.sort(acc, dim=1, descending=True)` at the end of
`streaming_topk` where `acc.shape[1] = N_EXPTS_ACT = 4`.  If the
FINAL `tl.sort` at BLOCK_N=4 has the same class of bug as
BLOCK_N=16 (pair-swap inversion on random inputs at specific
rows), both residuals are unified — `streaming_topk`'s final
sort inherits the tl.sort BLOCK_N<32 bug.  If this probe MATCHES,
m2_strict's residual comes from streaming_topk's own merge /
exclude logic (independent bug class).

4-element sort is bitonic distance-1 → distance-2, i.e., two
stages of compare-and-swap.  Same value range as the N=16 /
N=32 probes (uniform [-4, 4]) under `abs tol=0.0`.
"""
import triton
import triton.language as tl


@triton.jit
def _canary_tl_sort_fp32_n4(
    X, stride_xm,
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

    x = tl.load(X + offs_m[:, None] * stride_xm + offs_n[None, :],
                mask=mask_m, other=float("-inf"))
    y = tl.sort(x, dim=1, descending=True)

    Y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(Y_ptrs, y, mask=mask_m)


RECIPES = [{
    "name": "canary_tl_sort_fp32_n4",
    "kernel_fn":     _canary_tl_sort_fp32_n4,
    "kernel_symbol": "_canary_tl_sort_fp32_n4",
    "signature": {
        "X":           "*fp32",
        "stride_xm":   "i32",
        "Y":           "*fp32",
        "stride_ym":   "i32",
        "n_rows":      "i32",
    },
    "constexprs": {"BLOCK_M": 32, "BLOCK_N": 4},
    "harness_constants": {"N_COLS": 4},
    "scalar_args": {
        "stride_xm": "N_COLS",
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
    "inputs": [
        {"name": "X", "dtype": "fp32", "elems": "N_ROWS * N_COLS",
         "range_lo": -4.0, "range_hi": 4.0},
    ],
    "outputs": [
        {"name": "Y", "dtype": "fp32", "elems": "N_ROWS * N_COLS"},
    ],
    "comparator": {"kind": "abs", "tol": 0.0},
}]
