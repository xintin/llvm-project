"""`tl.sort` probe at BLOCK_N=16 with alternating row sort-direction.

Even rows are ascending [0, 1, ..., 15]; odd rows are descending
[15, 14, ..., 0].  Descending sort output should be the same for
every row: [15, 14, ..., 0].

Discriminator for the n16 residual: if the miscompile is
triggered by CROSS-ROW compare-result divergence (lanes in
different rows producing different compare results at the same
stage, with a subsequent SGPR / VCC round-trip picking the wrong
bit-half), this probe's alternating input forces lane-pairs at
every cross-row boundary to produce opposite compare results at
stage 1.
"""
import triton
import triton.language as tl


@triton.jit
def _canary_tl_sort_fp32_n16_altrow(
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

    # Alternating row orientation: even rows ascending, odd rows descending.
    parity = offs_m[:, None] & 1
    ascending = offs_n[None, :].to(tl.float32)
    descending = (15 - offs_n[None, :]).to(tl.float32)
    x = tl.where(parity == 0, ascending, descending)
    y = tl.sort(x, dim=1, descending=True)

    Y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(Y_ptrs, y, mask=mask_m)


RECIPES = [{
    "name": "canary_tl_sort_fp32_n16_altrow",
    "kernel_fn":     _canary_tl_sort_fp32_n16_altrow,
    "kernel_symbol": "_canary_tl_sort_fp32_n16_altrow",
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
