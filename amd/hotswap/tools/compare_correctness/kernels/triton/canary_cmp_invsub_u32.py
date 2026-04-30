"""Minimal cmp->u32->inv_sub probe."""

import triton
import triton.language as tl


@triton.jit
def _canary_cmp_invsub_u32(
    X, stride_xm,
    CMP, INV, stride_ym,
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
    xptr = X + offs_m[:, None] * stride_xm + offs_n[None, :]
    x = tl.load(xptr, mask=mask_m, other=0.0)
    cmp_u32 = tl.where(x > 0.0, 1, 0).to(tl.uint32)
    inv_sub = (1 - cmp_u32).to(tl.uint32)
    yptr = offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(CMP + yptr, cmp_u32, mask=mask_m)
    tl.store(INV + yptr, inv_sub, mask=mask_m)


RECIPES = [{
    "name": "canary_cmp_invsub_u32",
    "kernel_fn": _canary_cmp_invsub_u32,
    "kernel_symbol": "_canary_cmp_invsub_u32",
    "signature": {
        "X": "*fp32",
        "stride_xm": "i32",
        "CMP": "*u32",
        "INV": "*u32",
        "stride_ym": "i32",
        "n_rows": "i32",
    },
    "constexprs": {"BLOCK_M": 32, "BLOCK_N": 16},
    "harness_constants": {"N_COLS": 16},
    "scalar_args": {
        "stride_xm": "N_COLS",
        "stride_ym": "N_COLS",
        "n_rows": "N_ROWS",
    },
    "num_warps": 2,
    "shape_dim": "N_ROWS",
    "default_shapes": [32, 64, 256],
    "grid": {"x": "ceil_div(N_ROWS, BLOCK_M)", "y": "1", "z": "1"},
    "inputs": [{
        "name": "X", "dtype": "fp32",
        "elems": "N_ROWS * N_COLS",
        "range_lo": -4.0, "range_hi": 4.0,
    }],
    "outputs": [
        {"name": "CMP", "dtype": "u32", "elems": "N_ROWS * N_COLS"},
        {"name": "INV", "dtype": "u32", "elems": "N_ROWS * N_COLS"},
    ],
    "comparator": {"kind": "abs", "tol": 0.0},
}]

