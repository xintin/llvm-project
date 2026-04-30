"""`tl.sort` probe at BLOCK_N=16 with per-row distinct monotonic input.

Companion to `canary_tl_sort_fp32_n16_deterministic` (which is
per-row identical and passes) and `canary_tl_sort_fp32_n16`
(random and fails ~12.9%).  Tests `X[r, c] = c + r * 100.0` —
per-row distinct values but each row is monotonic in the
descending-sort direction (already sorted, just needs identity).

Expected output: `Y[r, c] = (15 - c) + r * 100.0` per row (the
descending sort of each row).

Discriminator between three hypotheses for the n16 random
residual:
  * If this MATCHES:  bug is specific to per-row-random INPUT
    ORDERING combinations (points at a compare primitive that
    reads different values per lane and feeds a shared-mask
    cndmask with a cross-lane leak).
  * If this FAILS:    bug affects any per-row-distinct input,
    which would rule out the "cross-row identical-value"
    hypothesis.
"""
import triton
import triton.language as tl


@triton.jit
def _canary_tl_sort_fp32_n16_row_offset(
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

    # Per-row distinct, monotonic ascending within row.
    x = (offs_m[:, None].to(tl.float32) * 100.0
         + offs_n[None, :].to(tl.float32))
    y = tl.sort(x, dim=1, descending=True)

    Y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(Y_ptrs, y, mask=mask_m)


RECIPES = [{
    "name": "canary_tl_sort_fp32_n16_row_offset",
    "kernel_fn":     _canary_tl_sort_fp32_n16_row_offset,
    "kernel_symbol": "_canary_tl_sort_fp32_n16_row_offset",
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
