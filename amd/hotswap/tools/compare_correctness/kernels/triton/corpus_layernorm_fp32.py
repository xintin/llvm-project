"""compare_correctness shim for triton-tutorial-05-layer-norm (fp32).

The kernel itself lives in ``_corpus/extracted/_layer_norm_fwd_fused.py``,
pulled by ``_corpus/pull.py`` from a pinned triton-lang/triton commit
+ sha256.  This shim wires the forward pass into our recipe schema.

Schema notes specific to this recipe:

  * The kernel's signature uses single-letter pointer names (``X``,
    ``Y``, ``W``, ``B``, ``Mean``, ``Rstd``).  Our schema treats the
    sig arg name as the join key for inputs / outputs / scalar_args,
    so the input and output buffer names below must match exactly.
  * ``shape_dim`` sweeps ``N`` (the column count, i.e. the reduction
    width).  Number of rows ``M`` is fixed via ``harness_constants``
    because we only have a single named shape dim today.  ``M`` is
    not a kernel parameter at all (the kernel reads its row id from
    ``tl.program_id(0)``) — it lives purely in the harness, used to
    size the launch grid (``grid.x = M``) and the input / output
    buffers (``elems = "M * N"``).  Putting it in ``constexprs`` would
    fail Triton's ASTSource which requires every constexpr name to be
    in the kernel's argument list.
  * ``stride`` is a runtime i32 sig arg derived from ``N`` (the kernel
    treats X / Y as M-by-N row-major), so it goes into ``scalar_args``
    as the expression ``"N"``.
  * ``eps`` is the first **floating-point scalar arg** the harness
    handles.  It's a literal in ``scalar_args`` (a JSON number rather
    than a string expression) and the C++ dispatch packs it into the
    by_value kernarg slot at the right width.  1e-5 matches PyTorch's
    default LayerNorm epsilon.
  * Three outputs of mixed sizes (Y is M*N, Mean and Rstd are M each).
    Per-output comparators let Y use ``rel-rms`` (where wave-size
    reduction-order drift dominates) while the small Mean / Rstd
    buffers can use a tighter elementwise ``rel`` tolerance.
"""
from _corpus import load_corpus_jit

_layer_norm_fwd_fused = load_corpus_jit("_layer_norm_fwd_fused")


RECIPES = [
    {
        "name": "corpus_layernorm_fp32",
        "kernel_fn": _layer_norm_fwd_fused,
        "kernel_symbol": "_layer_norm_fwd_fused",
        "signature": {
            "X":      "*fp32",
            "Y":      "*fp32",
            "W":      "*fp32",
            "B":      "*fp32",
            "Mean":   "*fp32",
            "Rstd":   "*fp32",
            "stride": "i32",
            "N":      "i32",
            "eps":    "fp32",
        },
        "constexprs": {
            "BLOCK_SIZE": 1024,
        },
        "harness_constants": {
            "M": 16,
        },
        "num_warps": 4,
        "shape_dim": "N",
        "default_shapes": [128, 256, 512, 1024],
        "grid": {
            "x": "M",
            "y": "1",
            "z": "1",
        },
        "scalar_args": {
            "stride": "N",
            "eps":    1e-5,
        },
        "inputs": [
            {"name": "X", "dtype": "fp32", "elems": "M * N",
             "range_lo": -2.0, "range_hi": 2.0},
            {"name": "W", "dtype": "fp32", "elems": "N",
             "range_lo": -1.0, "range_hi": 1.0},
            {"name": "B", "dtype": "fp32", "elems": "N",
             "range_lo": -1.0, "range_hi": 1.0},
        ],
        "outputs": [
            {"name": "Y",    "dtype": "fp32", "elems": "M * N",
             "comparator": {"kind": "rel-rms", "tol": 1e-3}},
            {"name": "Mean", "dtype": "fp32", "elems": "M",
             "comparator": {"kind": "rel", "tol": 1e-3}},
            {"name": "Rstd", "dtype": "fp32", "elems": "M",
             "comparator": {"kind": "rel", "tol": 1e-3}},
        ],
        "comparator": {"kind": "rel-rms", "tol": 1e-3},
    }
]
