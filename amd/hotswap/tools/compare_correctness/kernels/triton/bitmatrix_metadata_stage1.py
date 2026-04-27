"""Regression recipe for GPT-OSS `_bitmatrix_metadata_compute_stage1`.

This mirrors the captured full-v4 fixture
`triton_kernels.tensor_details.bitmatrix.bitmatrix_metadata_compute_stage1.1974142.000.pt`.
That fixture used to report a Salmon `wrong-result` on `arg5`, the
`ColOffs` output:

* grid: 37 programs
* `n_cols = 32`
* `n_combined_indx = 4096`
* `shape_pm = 16`
* `stride_pm = 1`
* `stride_pn = 16`
* `BLOCK = 1024`, `BLOCK_M = BLOCK_N = 512`

The recipe keeps those dimensions fixed and compares the two deterministic
outputs that are not read as inputs:

* `CombinedIndx`: memset to the sentinel `-1`
* `ColOffs`: prefix sums over `ColSum`, matching the fixture's `arg5`

`PartialColSum` is an input because the kernel reads it before writing the
per-column partial prefix sums in place; the current Triton AOT harness
models each pointer as either input or output, not both.  That is sufficient
for this regression because the historical mismatch was isolated to `ColOffs`.
"""

from triton_kernels.tensor_details.bitmatrix import (
    _bitmatrix_metadata_compute_stage1,
)


_N_COLS = 32
_SHAPE_PM = 16
_STRIDE_PM = 1
_STRIDE_PN = 16
_BLOCK = 1024
_BLOCK_M = 512
_BLOCK_N = 512


RECIPES = [
    {
        "name": "bitmatrix_metadata_stage1",
        "kernel_fn": _bitmatrix_metadata_compute_stage1,
        "kernel_symbol": "_bitmatrix_metadata_compute_stage1",
        "signature": {
            "CombinedIndx": "*i32",
            "n_combined_indx": "i32",
            "sentinel": "i32",
            # BLOCK: tl.constexpr -> see constexprs.
            "ColSum": "*i32",
            "ColOffs": "*i32",
            "n_cols": "i32",
            "PartialColSum": "*i32",
            "shape_pm": "i32",
            "stride_pm": "i32",
            "stride_pn": "i32",
            # BLOCK_M, BLOCK_N: tl.constexpr -> see constexprs.
        },
        "constexprs": {
            "BLOCK": _BLOCK,
            "BLOCK_M": _BLOCK_M,
            "BLOCK_N": _BLOCK_N,
        },
        "harness_constants": {
            "N_COLS": _N_COLS,
            "SHAPE_PM": _SHAPE_PM,
            "STRIDE_PM": _STRIDE_PM,
            "STRIDE_PN": _STRIDE_PN,
        },
        "scalar_args": {
            "n_combined_indx": "N_COMBINED_INDX",
            "sentinel": "-1",
            "n_cols": "N_COLS",
            "shape_pm": "SHAPE_PM",
            "stride_pm": "STRIDE_PM",
            "stride_pn": "STRIDE_PN",
        },
        "num_warps": 8,
        "shape_dim": "N_COMBINED_INDX",
        "default_shapes": [4096],
        "grid": {
            "x": "N_COLS + 1 + ceil_div(N_COMBINED_INDX, BLOCK)",
            "y": "1",
            "z": "1",
        },
        "inputs": [
            {"name": "ColSum", "dtype": "i32", "elems": "N_COLS"},
            {
                "name": "PartialColSum",
                "dtype": "i32",
                "elems": "SHAPE_PM * N_COLS",
            },
        ],
        "outputs": [
            {
                "name": "CombinedIndx",
                "dtype": "i32",
                "elems": "N_COMBINED_INDX",
            },
            {"name": "ColOffs", "dtype": "i32", "elems": "N_COLS"},
        ],
        "comparator": {"kind": "abs", "tol": 0.0},
    }
]
