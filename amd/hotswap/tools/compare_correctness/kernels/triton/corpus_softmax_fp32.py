"""compare_correctness shim for triton-tutorial-02-fused-softmax (fp32).

The kernel itself lives in `_corpus/extracted/softmax_kernel.py`, pulled
by `_corpus/pull.py` from a pinned triton-lang/triton commit + sha256.
This shim wires it into our recipe schema.

Schema notes specific to this recipe:

  * ``shape_dim`` sweeps ``n_cols``.  The number of rows is fixed via
    a ``harness_constants`` entry — the harness only sweeps one named
    dim today, and the interesting axis for salmon is the reduction
    width (n_cols) because that's what controls how many wavefronts
    participate in the row-wise max / sum.
  * ``n_rows`` is declared in ``harness_constants`` (not ``constexprs``)
    on purpose: it IS a runtime kernarg slot in the kernel signature,
    and putting it in ``constexprs`` would let Triton bake it into the
    code and silently drop the kernarg slot — which would then break
    our sig-vs-metadata lockstep at dispatch.  In ``harness_constants``
    it stays a runtime arg with a value the auto-resolve path picks up.
  * ``input_row_stride`` and ``output_row_stride`` are *runtime* i32
    sig args, but their values are derived from ``n_cols`` — so they
    show up in ``scalar_args`` as expressions.  The C++ dispatch
    evaluates them per launch in the same scope as ``elems`` / ``grid``.
  * ``BLOCK_SIZE`` must be a power of two ≥ ``n_cols``.  The default
    sweep stays ≤ 1024 so a single ``BLOCK_SIZE = 1024`` covers every
    shape.  Adding a 2048 sweep shape would mean either splitting into
    multiple recipes or extending the schema with shape-keyed
    constexprs — out of scope here.
  * ``num_stages`` is a Triton tuning constexpr; on this kernel it
    only affects pipelining, not the result, so any small int works.
  * Comparator is ``rel-rms`` because the reduction order shuffles
    differently between wave32 (gfx1250) and wave64 (gfx942 gold).
    Pointwise relative error blows up on rows where ``denominator`` is
    near the floating-point cancellation boundary; the bulk-RMS norm
    stays small.
"""
from _corpus import load_corpus_jit

softmax_kernel = load_corpus_jit("softmax_kernel")


RECIPES = [
    {
        "name": "corpus_softmax_fp32",
        "kernel_fn": softmax_kernel,
        "kernel_symbol": "softmax_kernel",
        "signature": {
            "output_ptr":        "*fp32",
            "input_ptr":         "*fp32",
            "input_row_stride":  "i32",
            "output_row_stride": "i32",
            "n_rows":            "i32",
            "n_cols":            "i32",
        },
        "constexprs": {
            "BLOCK_SIZE": 1024,
            "num_stages": 4,
        },
        "harness_constants": {
            "n_rows": 16,
        },
        "num_warps": 4,
        "shape_dim": "n_cols",
        "default_shapes": [128, 256, 512, 1024],
        "grid": {
            "x": "n_rows",
            "y": "1",
            "z": "1",
        },
        "scalar_args": {
            "input_row_stride":  "n_cols",
            "output_row_stride": "n_cols",
        },
        "inputs": [
            {"name": "input_ptr", "dtype": "fp32",
             "elems": "n_rows * n_cols",
             "range_lo": -2.0, "range_hi": 2.0},
        ],
        "outputs": [
            {"name": "output_ptr", "dtype": "fp32",
             "elems": "n_rows * n_cols"},
        ],
        "comparator": {"kind": "rel-rms", "tol": 1e-3},
    }
]
