"""compare_correctness shim for triton-tutorial-07-extern-functions (asin).

The actual kernel lives in `_corpus/extracted/asin_kernel.py`, pulled
by `_corpus/pull.py` from a pinned triton-lang/triton commit.

This is the salmon-coverage canary for libdevice intrinsics: the kernel
calls `libdevice.asin(x)`, which lowers to an extern call into the AMD
device library (ocml/ockl) at link time.  If salmon's IR raiser can't
preserve those external linkages across the gfx1250→gfx942 path, this
recipe is where it shows up first.
"""
from _corpus import load_corpus_jit

asin_kernel = load_corpus_jit("asin_kernel")


RECIPES = [
    {
        "name": "corpus_asin_fp32",
        "kernel_fn": asin_kernel,
        "kernel_symbol": "asin_kernel",
        "signature": {
            "x_ptr":      "*fp32",
            "y_ptr":      "*fp32",
            "n_elements": "i32",
        },
        "constexprs": {"BLOCK_SIZE": 1024},
        "num_warps": 4,
        "shape_dim": "n_elements",
        "default_shapes": [1024, 4096, 65536, 98432],
        "grid": {
            "x": "ceil_div(n_elements, BLOCK_SIZE)",
            "y": "1",
            "z": "1",
        },
        # asin's domain is [-1, 1].  Outside that range it returns NaN;
        # both paths agreeing on NaN is treated as a match by the
        # comparator, but it would mask any genuine arithmetic
        # disagreement, so we keep the inputs strictly inside the
        # domain.  range_hi=0.999 leaves a thin guard band so floating-
        # point fuzz at the boundary doesn't sneak NaNs in.
        "inputs": [
            {"name": "x_ptr", "dtype": "fp32",
             "elems": "n_elements",
             "range_lo": -0.999, "range_hi": 0.999},
        ],
        "outputs": [
            {"name": "y_ptr", "dtype": "fp32", "elems": "n_elements"},
        ],
        # asin is a bandwidth-bound elementwise transform; rounding is
        # consistent across wave sizes so absolute tolerance is fine.
        # 1e-5 is the manufacturer's documented ULP bound for the
        # libdevice asin polynomial in fp32.
        "comparator": {"kind": "abs", "tol": 1e-5},
    }
]
