#!/usr/bin/env python3
"""Pull a corpus of upstream Triton kernels into the local tree so the
compare_correctness harness can stress-test salmon against them.

Workflow

    cd kernels/triton/_corpus
    python pull.py            # downloads everything in MANIFEST below
    python pull.py --status   # show what's installed and at which pin
    python pull.py --refresh  # re-download even if already present
                              # (use after editing the manifest)

What this writes

    kernels/triton/_corpus/upstream/<id>.py
        Verbatim upstream source.  Pinned to the manifest's commit SHA and
        verified against the manifest's sha256.  Both .gitignored.

    kernels/triton/_corpus/extracted/<func_name>.py
        Just the @triton.jit function and the imports that Triton's
        compile flow needs (`triton`, `triton.language as tl`, plus any
        per-entry `extra_imports`).  Importing these has no side effects;
        importing the original upstream tutorials does (they spin up a
        torch device, run the kernel, render benchmark plots, etc.).

        Wrapper recipes in kernels/triton/<name>.py import these
        extracted files via the `_corpus.load_corpus_jit("name")` helper.

What this does NOT do

    *Wrapper recipes are committed to the tree manually*, not generated.
    Each wrapper sets a recipe schema (signature / shape sweep / inputs /
    outputs / comparator) for one upstream kernel.  See
    kernels/triton/corpus_*.py.  Run `python status.py` to see which
    upstream entries currently have a wrapper recipe and which don't.

Why pin to a commit + sha256

    Triton's tutorials ship as documentation that drifts.  Without a pin,
    a passing run today turns into a "what changed?" hunt next week.  The
    sha256 is a defence-in-depth check on top of the commit pin (GitHub
    serves consistent blobs from a given ref, but pinning the bytes lets
    a future machine catch any silent rewrite).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional


# -- Manifest ---------------------------------------------------------------
#
# Each entry pins one upstream Triton kernel source file.  Fields:
#
#   id            short slug, used as the upstream-file basename
#   url           where to download from (raw.githubusercontent.com, etc.)
#   sha256        expected hash of the downloaded bytes
#   functions     list of @triton.jit function names to extract
#   extra_imports import lines to prepend to each extracted file (optional)
#   notes         free-form note shown by --status
#
# The url should embed a commit SHA, never a moving branch.  The sha256
# pin makes drift visible: if the upstream file is rewritten under us,
# the fetch fails loud rather than silently changing what we test.

@dataclass
class CorpusEntry:
    id: str
    url: str
    sha256: str
    functions: List[str]
    extra_imports: List[str] = field(default_factory=list)
    notes: str = ""


# Pinned to triton-lang/triton @ 6ee5472e7defefefb15248e321f35abb5c243b38
# (HEAD of `python/tutorials/` on 2026-04-16).  Refresh with
#   curl -fsSL https://raw.githubusercontent.com/triton-lang/triton/<NEW>/python/tutorials/<file> | sha256sum
# and then rerun `pull.py --refresh`.
_TRITON_PIN = "6ee5472e7defefefb15248e321f35abb5c243b38"
_TRITON_RAW = (
    f"https://raw.githubusercontent.com/triton-lang/triton/{_TRITON_PIN}/"
    "python/tutorials/"
)

MANIFEST: List[CorpusEntry] = [
    CorpusEntry(
        id="triton-tutorial-01-vector-add",
        url=_TRITON_RAW + "01-vector-add.py",
        sha256="842430949e0ccde4fbce07606cce3ac4bac36bf21b2b12619a31b795ca4029b3",
        functions=["add_kernel"],
        notes="elementwise add; same shape as our vecadd_f16, fp32",
    ),
    CorpusEntry(
        id="triton-tutorial-02-fused-softmax",
        url=_TRITON_RAW + "02-fused-softmax.py",
        sha256="8a65b696d5b02a3900380395ecd883671afeb66328db108f55622ed8853a05da",
        functions=["softmax_kernel"],
        notes=(
            "row-wise softmax with masking; reduction across cols.  "
            "Wrapper recipe needs runtime scalar support for the strides "
            "(not yet wired — see status.py)."
        ),
    ),
    CorpusEntry(
        id="triton-tutorial-05-layer-norm",
        url=_TRITON_RAW + "05-layer-norm.py",
        sha256="31cda7bf7bb9b637ea662b789188d37ff5981c3973626d4e2a293db633f3bb87",
        functions=["_layer_norm_fwd_fused"],
        notes=(
            "fused layer-norm forward.  Three outputs of mixed sizes "
            "(Y is M*N, Mean and Rstd are M).  Needs float scalar (eps) "
            "and runtime stride support before a wrapper can run."
        ),
    ),
    CorpusEntry(
        id="triton-tutorial-07-extern-functions",
        url=_TRITON_RAW + "07-extern-functions.py",
        sha256="c9412656298e9bd737d944ac94c5dd6ec427cd30b7657c09d36fa5495ce0d4ba",
        functions=["asin_kernel"],
        extra_imports=["from triton.language.extra import libdevice"],
        notes=(
            "elementwise asin via libdevice; verifies the salmon path "
            "handles intrinsic calls correctly.  Inputs must stay in "
            "[-1, 1] for asin to be defined — wrapper sets that range."
        ),
    ),
    # -- Below this line: tutorials added for breadth-only crash sweeping --
    # via triton_corpus_runner.  No compare_correctness wrapper recipes
    # exist for them yet; status.py will list them as "missing wrapper".
    # These are intentionally the AMD-tractable subset of tutorials 03-08:
    #
    #   03 matmul        : tl.dot, autotuned, fp16 + fp8.  Most common
    #                      transformer building block; great stress test
    #                      for the MFMA selection path.
    #   04 dropout       : tl.rand / philox seeded RNG.  Exercises the
    #                      RNG intrinsic translation.
    #   06 fused-attn    : multi-kernel flash-attention 2 (fwd + bwd).
    #                      Uses tl.dot in inner loops, shared memory,
    #                      autotuning.  Heavy.
    #   08 grouped-gemm  : autotuned variable-shape GEMM (TMA path is
    #                      compiled-out on AMD; the non-TMA kernel runs).
    #
    # Tutorials 09 (persistent matmul), 10 (block-scaled mxfp/nvfp) and
    # 11 (programmatic-dependent-launch) are NVIDIA-Hopper/Blackwell
    # specific and Triton-rocm refuses to compile them on AMD, so they'd
    # only contribute baseline-fail noise — skip.
    CorpusEntry(
        id="triton-tutorial-03-matrix-multiplication",
        url=_TRITON_RAW + "03-matrix-multiplication.py",
        sha256="f88a9506694c58d483ad3cdb4aa672c23df0300803bcfaca13b9396184825008",
        functions=["matmul_kernel", "leaky_relu"],
        notes=(
            "autotuned tile-based GEMM (fp16 + fp8).  tl.dot path; "
            "exercises the MFMA-instruction selection / pipeline."
        ),
    ),
    CorpusEntry(
        id="triton-tutorial-04-low-memory-dropout",
        url=_TRITON_RAW + "04-low-memory-dropout.py",
        sha256="43ea69769349d20654fe7bb983e60ad9c38591d195891caf3a56053d9b0ab788",
        functions=["_dropout", "_seeded_dropout"],
        notes=(
            "naïve dropout + Philox-seeded dropout.  Exercises tl.rand "
            "intrinsic translation (the seeded one) and a simple "
            "pointer-arith elementwise path (the naïve one)."
        ),
    ),
    CorpusEntry(
        id="triton-tutorial-06-fused-attention",
        url=_TRITON_RAW + "06-fused-attention.py",
        sha256="5b16be48b7b781ef018931c789b191f3390886ac479bb5df8a3418e5612ab317",
        functions=[
            "_attn_fwd_inner",
            "_maybe_make_tensor_desc",
            "_attn_fwd",
            "_attn_bwd_preprocess",
            "_attn_bwd_dkdv",
            "_attn_bwd_dq",
            "_attn_bwd",
        ],
        notes=(
            "flash-attention-2 fwd + bwd, multi-kernel.  Heaviest "
            "tutorial in the set.  Uses tl.dot, shared memory, "
            "autotuning, and a TMA-vs-non-TMA branch (TMA is "
            "compiled-out on AMD)."
        ),
    ),
    CorpusEntry(
        id="triton-tutorial-08-grouped-gemm",
        url=_TRITON_RAW + "08-grouped-gemm.py",
        sha256="0b474fc69d34dfde591454ca735d10abdc72bcc03d6b42530390aceaa2844c3b",
        functions=["grouped_matmul_kernel", "grouped_matmul_tma_kernel"],
        notes=(
            "variable-shape grouped GEMM.  The TMA kernel will fail "
            "to compile on AMD (no TMA descriptor support); the "
            "non-TMA kernel exercises a many-small-tiles autotuning "
            "and indirect dispatch pattern."
        ),
    ),
]


# -- Paths ------------------------------------------------------------------

CORPUS_DIR    = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR  = os.path.join(CORPUS_DIR, "upstream")
EXTRACTED_DIR = os.path.join(CORPUS_DIR, "extracted")


# -- Download + verify ------------------------------------------------------

def _read_url(url: str) -> bytes:
    """Fetch URL bytes.  Loud on any non-200; we never silently succeed
    on a redirected error page."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "compare_correctness-corpus-puller/1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"GET {url}: HTTP {resp.status} {resp.reason}"
                )
            return resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GET {url}: HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"GET {url}: {e.reason}") from e


def _verify_sha(data: bytes, expected: str, src: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"{src}: sha256 mismatch.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"  If upstream changed intentionally, update MANIFEST in pull.py."
        )


# -- @triton.jit extraction -------------------------------------------------

def _extract_jit(source: str, fn_name: str, src_label: str) -> str:
    """Return the source text of the @triton.jit-decorated function
    `fn_name` in `source`.

    We use ast to locate the FunctionDef node (so triton's own jit
    decoration / argument annotation tricks don't trip a regex), then
    splice the original text from the leading decorator line through the
    end of the function body.  Returning *source text* (not a Python
    object) is deliberate: Triton's jit machinery calls
    inspect.getsource() on its function, which only works when the
    function is defined in a real .py file."""
    try:
        tree = ast.parse(source, filename=src_label)
    except SyntaxError as e:
        raise RuntimeError(
            f"{src_label}: failed to parse upstream source ({e})"
        ) from e

    target: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            # Confirm at least one decorator is `triton.jit` (or `jit`).
            # Loose match — upstream sometimes uses `@triton.jit`,
            # sometimes `@jit` with `from triton import jit`.
            has_jit = any(
                _decorator_name_ends_with(d, "jit") for d in node.decorator_list
            )
            if not has_jit:
                raise RuntimeError(
                    f"{src_label}: function {fn_name!r} is not @triton.jit "
                    f"decorated; refusing to extract"
                )
            target = node
            break
    if target is None:
        raise RuntimeError(
            f"{src_label}: no @triton.jit function named {fn_name!r}"
        )

    # AST line numbers are 1-based; cover decorators too.
    start_line = (
        min(d.lineno for d in target.decorator_list)
        if target.decorator_list else target.lineno
    )
    end_line = target.end_lineno  # inclusive
    if end_line is None:
        raise RuntimeError(
            f"{src_label}: ast lacks end_lineno for {fn_name!r}; "
            f"is this Python 3.8+?"
        )

    lines = source.splitlines(keepends=True)
    return "".join(lines[start_line - 1 : end_line])


def _decorator_name_ends_with(d: ast.expr, suffix: str) -> bool:
    """True if a decorator AST node's textual form ends with `.suffix`
    or equals `suffix`.  Handles `@jit`, `@triton.jit`, `@x.y.jit`."""
    if isinstance(d, ast.Name):
        return d.id == suffix
    if isinstance(d, ast.Attribute):
        return d.attr == suffix
    if isinstance(d, ast.Call):
        return _decorator_name_ends_with(d.func, suffix)
    return False


def _write_extracted(entry: CorpusEntry, fn_name: str, body: str) -> str:
    """Write a self-contained extracted .py file for a single @jit
    function.  Returns the absolute path."""
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    out_path = os.path.join(EXTRACTED_DIR, f"{fn_name}.py")
    header = (
        f'"""Auto-extracted from {entry.id} by '
        f"kernels/triton/_corpus/pull.py.  Do not edit by hand."
        '"""\n'
        "import triton\n"
        "import triton.language as tl\n"
    )
    for line in entry.extra_imports:
        header += line.rstrip() + "\n"
    header += "\n"
    with open(out_path, "w") as f:
        f.write(header)
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")
    return out_path


# -- Top-level orchestration ------------------------------------------------

def _fetch_one(entry: CorpusEntry, force: bool) -> dict:
    """Download (if needed) + verify + extract one entry.  Returns a
    summary dict for status reporting."""
    os.makedirs(UPSTREAM_DIR, exist_ok=True)
    upstream_path = os.path.join(UPSTREAM_DIR, f"{entry.id}.py")

    cached_ok = False
    data: Optional[bytes] = None
    if os.path.exists(upstream_path) and not force:
        with open(upstream_path, "rb") as f:
            data = f.read()
        try:
            _verify_sha(data, entry.sha256, upstream_path)
            cached_ok = True
        except RuntimeError:
            data = None  # fall through and re-download

    if data is None:
        print(f"  fetch {entry.id}")
        data = _read_url(entry.url)
        _verify_sha(data, entry.sha256, entry.url)
        with open(upstream_path, "wb") as f:
            f.write(data)
    else:
        print(f"  cached {entry.id}")

    source = data.decode("utf-8")
    extracted_paths = []
    for fn in entry.functions:
        body = _extract_jit(source, fn, entry.id)
        path = _write_extracted(entry, fn, body)
        extracted_paths.append(path)

    return {
        "id": entry.id,
        "upstream": upstream_path,
        "extracted": extracted_paths,
        "cached": cached_ok,
        "notes": entry.notes,
    }


def cmd_pull(force: bool) -> int:
    print(f"Pulling {len(MANIFEST)} corpus entries into {CORPUS_DIR}")
    summary = []
    for entry in MANIFEST:
        try:
            summary.append(_fetch_one(entry, force))
        except RuntimeError as e:
            print(f"  FAIL {entry.id}: {e}", file=sys.stderr)
            return 1

    # Persist a machine-readable index so status.py and humans can ask
    # "what does my tree contain?" without re-running the puller.
    index_path = os.path.join(CORPUS_DIR, "INDEX.json")
    with open(index_path, "w") as f:
        json.dump(
            {
                "manifest_pin": _TRITON_PIN,
                "entries": summary,
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"\nWrote {index_path}")
    print("Next: build the shim recipes that import these extracted files")
    print("      (see kernels/triton/corpus_*.py for examples)")
    return 0


def cmd_status() -> int:
    print(f"Corpus pin: {_TRITON_PIN}")
    print(f"Upstream dir:  {UPSTREAM_DIR}")
    print(f"Extracted dir: {EXTRACTED_DIR}\n")
    for entry in MANIFEST:
        upstream_path = os.path.join(UPSTREAM_DIR, f"{entry.id}.py")
        if os.path.exists(upstream_path):
            with open(upstream_path, "rb") as f:
                actual = hashlib.sha256(f.read()).hexdigest()
            ok = actual == entry.sha256
            present = "OK " if ok else "BAD"
        else:
            present = "MISSING"
        extracted_status = []
        for fn in entry.functions:
            p = os.path.join(EXTRACTED_DIR, f"{fn}.py")
            extracted_status.append(f"{fn}={'present' if os.path.exists(p) else 'missing'}")
        print(f"  [{present}] {entry.id}")
        print(f"          {' '.join(extracted_status)}")
        if entry.notes:
            print(f"          note: {entry.notes}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="re-download even if cached")
    ap.add_argument("--status", action="store_true",
                    help="show installed entries instead of fetching")
    args = ap.parse_args()
    if args.status:
        return cmd_status()
    return cmd_pull(force=args.refresh)


if __name__ == "__main__":
    sys.exit(main())
