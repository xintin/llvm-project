"""compare_correctness shim for triton-tutorial-01-vector-add (fp32).

The actual kernel lives in `_corpus/extracted/add_kernel.py`, pulled by
`_corpus/pull.py` from a pinned triton-lang/triton commit + sha256.
This file just wraps it in our recipe schema so `aot_compile.py` can
build it for both gfx942 and gfx1250 and the harness can sweep shapes
through it.

Why a separate fp32 wrapper alongside `vecadd_f16.py`?
    `vecadd_f16` is a hand-written kernel; this is the upstream tutorial.
    Keeping both means a salmon regression on the upstream version is
    a finding in its own right, separate from anything specific to our
    handwritten shape.
"""
from _corpus import load_corpus_jit

add_kernel = load_corpus_jit("add_kernel")


RECIPES = [
    {
        "name": "corpus_add_fp32",
        "kernel_fn": add_kernel,
        "kernel_symbol": "add_kernel",
        "signature": {
            "x_ptr":      "*fp32",
            "y_ptr":      "*fp32",
            "output_ptr": "*fp32",
            "n_elements": "i32",
        },
        "constexprs": {"BLOCK_SIZE": 1024},
        "num_warps": 4,
        "shape_dim": "n_elements",
        # Mix of clean BLOCK_SIZE multiples and "ragged" sizes that
        # exercise the trailing-mask path the kernel uses.
        "default_shapes": [1024, 4096, 65536, 98432],
        "grid": {
            "x": "ceil_div(n_elements, BLOCK_SIZE)",
            "y": "1",
            "z": "1",
        },
        "inputs": [
            {"name": "x_ptr", "dtype": "fp32", "elems": "n_elements"},
            {"name": "y_ptr", "dtype": "fp32", "elems": "n_elements"},
        ],
        "outputs": [
            {"name": "output_ptr", "dtype": "fp32", "elems": "n_elements"},
        ],
        # fp32 add of two values in [-1, 1] is bit-exact across wave
        # sizes; the looser `1e-6` here is paranoia margin for any
        # rounding the transpiler paths might introduce.
        "comparator": {"kind": "abs", "tol": 1e-6},
    }
]
