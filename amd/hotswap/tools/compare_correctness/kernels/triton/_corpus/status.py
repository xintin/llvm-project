#!/usr/bin/env python3
"""Report which corpus entries have a working compare_correctness recipe.

Usage

    cd kernels/triton/_corpus
    python status.py            # static report: pull state + wrapper presence
    python status.py --run      # also invoke compare_correctness per recipe
                                # and tabulate pass / fail per shape

Output is a Markdown table (so it pastes cleanly into the corpus README
or a PR description).  Columns:

    pulled       MANIFEST entry has been downloaded + verified locally
    wrapper      kernels/triton/<name>.py shim exists and references the
                 extracted function
    built        kernels/build/<name>.sidecar.json exists (i.e. the AOT
                 step has run; required before --run can do anything)
    salmon       result of the salmon column under compare_correctness
                 (ALL_MATCH | MISMATCH | CRASH | NOT_BUILT | SKIPPED)

The salmon column is the one that matters for a "stress-test salmon"
report: legacy is widely known to crash on these tests today, native is
the gold by construction, so what's left to measure is salmon's
arithmetic agreement vs. the gold.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

CORPUS_DIR    = os.path.dirname(os.path.abspath(__file__))
TRITON_DIR    = os.path.dirname(CORPUS_DIR)                         # kernels/triton
KERNELS_DIR   = os.path.dirname(TRITON_DIR)                         # kernels
TOOL_ROOT     = os.path.dirname(KERNELS_DIR)                        # compare_correctness/
BUILD_DIR     = os.path.join(TOOL_ROOT, "kernels", "build")
UPSTREAM_DIR  = os.path.join(CORPUS_DIR, "upstream")
EXTRACTED_DIR = os.path.join(CORPUS_DIR, "extracted")

sys.path.insert(0, CORPUS_DIR)
import pull  # type: ignore  # local sibling module


# -- Discovery --------------------------------------------------------------

def _is_pulled(entry: pull.CorpusEntry) -> bool:
    """A corpus entry is 'pulled' iff the upstream file exists, its sha
    matches the manifest pin, and every declared @jit function has been
    extracted."""
    upstream = os.path.join(UPSTREAM_DIR, f"{entry.id}.py")
    if not os.path.exists(upstream):
        return False
    with open(upstream, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != entry.sha256:
            return False
    for fn in entry.functions:
        if not os.path.exists(os.path.join(EXTRACTED_DIR, f"{fn}.py")):
            return False
    return True


_LOAD_CALL_RE = re.compile(r'load_corpus_jit\(\s*[\'"]([^\'"]+)[\'"]\s*\)')


def _wrappers_for(fn_name: str) -> List[str]:
    """Return the list of `kernels/triton/corpus_*.py` shim recipe stems
    that import `fn_name` via load_corpus_jit.  Static text scan — we
    don't import the shim files because that would force a Triton venv
    on the host running this script."""
    out = []
    for path in sorted(glob.glob(os.path.join(TRITON_DIR, "corpus_*.py"))):
        with open(path) as f:
            text = f.read()
        for m in _LOAD_CALL_RE.finditer(text):
            if m.group(1) == fn_name:
                out.append(os.path.splitext(os.path.basename(path))[0])
                break
    return out


def _is_built(recipe_name: str) -> bool:
    return os.path.exists(
        os.path.join(BUILD_DIR, f"{recipe_name}.sidecar.json")
    )


# -- Optional --run -----------------------------------------------------------

def _run_compare(recipe_name: str) -> Tuple[str, Optional[Dict[str, int]]]:
    """Invoke compare_correctness for one recipe.  Parse the summary
    block at the end and report a salmon verdict.  Returns (status,
    counts) where counts is the parsed summary dict (or None if the run
    failed before producing a summary)."""
    binary = os.path.join(TOOL_ROOT, "compare_correctness")
    if not os.path.exists(binary):
        return "NO_BINARY", None
    intercept = os.path.join(TOOL_ROOT, "libsalmon_intercept.so")
    env = dict(os.environ)
    if os.path.exists(intercept):
        env["LD_PRELOAD"] = intercept
    try:
        result = subprocess.run(
            [binary, f"--recipe={recipe_name}"],
            cwd=TOOL_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", None
    counts = _parse_summary(result.stdout)
    if counts is None:
        return "NO_SUMMARY", None
    salmon_match    = counts.get("salmon_match", 0)
    salmon_mismatch = counts.get("salmon_mismatch", 0)
    salmon_crash    = counts.get("salmon_crash", 0)
    if salmon_crash > 0:
        return "CRASH", counts
    if salmon_mismatch > 0:
        return "MISMATCH", counts
    if salmon_match > 0:
        return "ALL_MATCH", counts
    return "EMPTY", counts


# Match a row from the harness's summary table:
#     match                  4         0         4
_SUM_ROW_RE = re.compile(
    r"^\s*(match|mismatch|crash/no-exit|gold|gold-missing)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s*$",
    re.MULTILINE,
)


def _parse_summary(stdout: str) -> Optional[Dict[str, int]]:
    """Pull the final summary block out of compare_correctness output.
    Schema: {native_match, native_mismatch, ..., salmon_crash}."""
    matches = list(_SUM_ROW_RE.finditer(stdout))
    if not matches:
        return None
    row_keys = {
        "match":         "match",
        "mismatch":      "mismatch",
        "crash/no-exit": "crash",
        "gold":          "gold",
        "gold-missing":  "gold_missing",
    }
    out: Dict[str, int] = {}
    for m in matches:
        key = row_keys.get(m.group(1))
        if key is None:
            continue
        for col_idx, col in enumerate(("native", "legacy", "salmon"), start=2):
            out[f"{col}_{key}"] = int(m.group(col_idx))
    return out


# -- Markdown rendering ------------------------------------------------------

def _render(rows: List[Dict[str, str]]) -> str:
    cols = ["entry", "function", "wrapper", "pulled", "built", "salmon", "notes"]
    widths = {c: max(len(c), max((len(r.get(c, "")) for r in rows), default=0))
              for c in cols}
    def fmt(values: List[str]) -> str:
        return "| " + " | ".join(
            v.ljust(widths[c]) for c, v in zip(cols, values)
        ) + " |"
    lines = [fmt(cols)]
    lines.append(
        "|" + "|".join("-" * (widths[c] + 2) for c in cols) + "|"
    )
    for r in rows:
        lines.append(fmt([r.get(c, "") for c in cols]))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true",
                    help="invoke compare_correctness per recipe and report "
                         "the salmon verdict")
    args = ap.parse_args()

    rows = []
    for entry in pull.MANIFEST:
        pulled = _is_pulled(entry)
        for fn in entry.functions:
            wrappers = _wrappers_for(fn) if pulled else []
            if not wrappers:
                rows.append({
                    "entry":    entry.id,
                    "function": fn,
                    "wrapper":  "—",
                    "pulled":   "yes" if pulled else "no",
                    "built":    "—",
                    "salmon":   "NOT_WIRED",
                    "notes":    entry.notes,
                })
                continue
            for w in wrappers:
                built = _is_built(w)
                if args.run and built:
                    salmon, _ = _run_compare(w)
                else:
                    salmon = "NOT_BUILT" if not built else "SKIPPED"
                rows.append({
                    "entry":    entry.id,
                    "function": fn,
                    "wrapper":  w,
                    "pulled":   "yes" if pulled else "no",
                    "built":    "yes" if built else "no",
                    "salmon":   salmon,
                    "notes":    entry.notes,
                })

    print(_render(rows))
    if not args.run:
        print("\n(use --run to also invoke compare_correctness per recipe; "
              "requires a built binary and libsalmon_intercept.so)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
