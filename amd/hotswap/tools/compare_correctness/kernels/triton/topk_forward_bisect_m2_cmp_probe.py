"""Probe only the iteration-2 compare materialization in topk MODE=2.

This isolates:
  cmp_u32 = tl.where(m2 > t1, 1, 0)
from the rest of the bitop3/inversion logic used in
topk_forward_bisect_m2_bitop3_probe.
"""

import triton
import triton.language as tl
from triton_kernels.topk_details._topk_forward import fpval_to_key, indx_to_key


@triton.jit
def _topk_forward_bisect_m2_cmp_probe(
    X, stride_xm,
    CMP, stride_ym,
    n_rows, n_expts_tot,
    BLOCK_M: tl.constexpr,
    N_EXPTS_PAD: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(N_EXPTS_PAD == 128, "probe calibrated for 128 experts")
    tl.static_assert(BLOCK_N == 32, "probe calibrated for BLOCK_N=32")

    pid = tl.program_id(0)
    if pid * BLOCK_M >= n_rows:
        return

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    mask_m = offs_m[:, None] < n_rows

    x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
    x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
    x_ultype: tl.constexpr = tl.uint32 if x_nbits < 16 else tl.dtype(f"uint{x_nbits * 2}")

    offs_x_n = 96 + tl.arange(0, BLOCK_N)
    mask_n = offs_x_n[None, :] < n_expts_tot
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :]
    x = tl.load(X_ptrs, mask=(mask_m & mask_n), other=float("-inf"))
    x = fpval_to_key(x.to(x_utype, bitcast=True))
    x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
    acc = tl.topk(x, N_EXPTS_ACT, dim=1)

    acc = tl.bitonic_merge(acc)
    X_ptrs -= BLOCK_N
    offs_x_n -= BLOCK_N
    x = tl.load(X_ptrs, mask=mask_m, other=float("-inf"))
    x = fpval_to_key(x.to(x_utype, bitcast=True))
    x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
    acc1 = tl.maximum(acc, tl.topk(x, N_EXPTS_ACT, dim=1))

    m2 = tl.bitonic_merge(acc1)
    X_ptrs -= BLOCK_N
    offs_x_n -= BLOCK_N
    x = tl.load(X_ptrs, mask=mask_m, other=float("-inf"))
    x = fpval_to_key(x.to(x_utype, bitcast=True))
    x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
    t1 = tl.topk(x, N_EXPTS_ACT, dim=1)

    cmp_u32 = tl.where(m2 > t1, 1, 0).to(tl.uint32)
    out_ptrs = offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(CMP + out_ptrs, cmp_u32, mask=mask_m)


_N_ROWS = 512
_N_COLS = 128
_N_EXPTS_ACT = 4


RECIPES = [{
    "name": "topk_forward_bisect_m2_cmp_probe",
    "kernel_fn": _topk_forward_bisect_m2_cmp_probe,
    "kernel_symbol": "_topk_forward_bisect_m2_cmp_probe",
    "signature": {
        "X": "*bf16",
        "stride_xm": "i32",
        "CMP": "*u32",
        "stride_ym": "i32",
        "n_rows": "i32",
        "n_expts_tot": "i32",
    },
    "constexprs": {
        "BLOCK_M": 32,
        "N_EXPTS_PAD": 128,
        "N_EXPTS_ACT": _N_EXPTS_ACT,
        "BLOCK_N": 32,
    },
    "harness_constants": {
        "N_COLS": _N_COLS,
        "N_EXPTS_ACT_HC": _N_EXPTS_ACT,
    },
    "scalar_args": {
        "stride_xm": "N_COLS",
        "stride_ym": "N_EXPTS_ACT_HC",
        "n_rows": "N_ROWS",
        "n_expts_tot": "N_COLS",
    },
    "num_warps": 4,
    "shape_dim": "N_ROWS",
    "default_shapes": [_N_ROWS],
    "grid": {"x": "ceil_div(N_ROWS, BLOCK_M)", "y": "1", "z": "1"},
    "inputs": [{
        "name": "X", "dtype": "bf16",
        "elems": "N_ROWS * N_COLS",
        "range_lo": -4.0, "range_hi": 4.0,
    }],
    "outputs": [
        {"name": "CMP", "dtype": "u32", "elems": "N_ROWS * N_EXPTS_ACT_HC"},
    ],
    "comparator": {"kind": "abs", "tol": 0.0},
}]

