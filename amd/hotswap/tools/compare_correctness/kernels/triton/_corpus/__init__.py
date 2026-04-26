"""Helpers for kernels/triton/corpus_*.py shim recipes.

The puller (`pull.py`) writes self-contained extracted .py files to
`_corpus/extracted/<func_name>.py`.  Each shim recipe loads its kernel
function via `load_corpus_jit("func_name")` from this module, packages
it into a recipe spec, and `aot_compile.py` does the rest.

We deliberately load the *extracted* file rather than the original
upstream tutorial, because importing the tutorial wholesale runs all of
its host-side code (torch tensor creation, kernel launches, benchmark
plotting).  The extracted file contains just the @triton.jit definition
and the imports Triton needs to compile it — no side effects.
"""

from __future__ import annotations

import importlib.util
import os
import sys

CORPUS_DIR    = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DIR = os.path.join(CORPUS_DIR, "extracted")


def load_corpus_jit(fn_name: str):
    """Import the extracted file `_corpus/extracted/<fn_name>.py` and
    return its `<fn_name>` attribute (a `@triton.jit` callable).

    Raises a helpful error if the file is missing — almost always means
    the user hasn't run pull.py yet."""
    path = os.path.join(EXTRACTED_DIR, f"{fn_name}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"  The corpus has not been pulled yet.  From the tool root:\n"
            f"    python kernels/triton/_corpus/pull.py\n"
            f"  See kernels/triton/_corpus/pull.py for the manifest."
        )
    mod_name = f"compare_correctness_corpus_{fn_name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, fn_name):
        raise AttributeError(
            f"{path} loaded but exposes no {fn_name!r} attribute.  "
            f"Re-run pull.py?"
        )
    return getattr(module, fn_name)
