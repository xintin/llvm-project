#!/usr/bin/env python3
"""Run GPT-OSS scope-discovery stages and emit a per-stage coverage report.

This is a tracked wrapper around the local scope-discovery capture driver:

  ../../scope_discovery/capture/run_gpt_oss.py

It is meant to be launched directly or through ``triton_corpus_runner``.
When launched through the runner, the child process inherits the selected
mode's environment: native, legacy, or Salmon with the forced gfx1250 Triton
target.

The wrapper intentionally stores reports only where requested via ``--json``,
``GPT_OSS_SCOPE_JSON``, or ``SCOPE_DISCOVERY_REPORT_DIR``. It never writes
reports or dumps under ``/data/gpt-oss``; that tree is reserved for GPT-OSS
source and large model data.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from typing import Optional


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCOPE_DRIVER = os.path.normpath(
    os.path.join(HERE, "..", "..", "scope_discovery", "capture", "run_gpt_oss.py")
)

STAGES = [
    "matmul_ogs_dense",
    "matmul_ogs_batched_dense",
    "matmul_ogs_fp32acc_large",
    "mxfp_downcast",
    "mxfp_upcast",
    "matmul_ogs_mxfp",
    "matmul_ogs_mxfp_large",
    "topk",
    "compaction",
    "swiglu",
    "reduce",
    "attention",
    "attention_120b",
]


def _mode_label() -> str:
    if os.environ.get("HSA_HOTSWAP_IR_RAISER") == "1":
        return "salmon"
    if os.environ.get("HSA_HOTSWAP_ISA_OVERRIDE"):
        return "legacy"
    return "native"


def _default_json_path(cli_path: str) -> str:
    if cli_path:
        return cli_path
    env_path = os.environ.get("GPT_OSS_SCOPE_JSON", "").strip()
    if env_path:
        return env_path
    report_dir = os.environ.get("SCOPE_DISCOVERY_REPORT_DIR", "").strip()
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        return os.path.join(report_dir, f"gpt_oss_scope_{_mode_label()}.json")
    return ""


def _tail(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return "...[truncated]\n" + text[-limit:]


def _classify(stage: str, returncode: int, stderr: str) -> tuple[str, str]:
    ok_marker = f"[ok]   {stage}"
    fail_marker = f"[fail] {stage}:"
    if fail_marker in stderr:
        return "fail", "stage reported failure"
    if returncode != 0:
        return "fail", f"driver exit {returncode}"
    if ok_marker in stderr:
        return "ok", "ok"
    return "unknown", "driver exited without an ok/fail marker"


def _run_stage(driver: str, stage: str, timeout_s: float) -> dict:
    cmd = [sys.executable, driver, "--stage", stage]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
        )
        elapsed = time.monotonic() - t0
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - t0
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "stage": stage,
            "status": "timeout",
            "detail": f"exceeded {timeout_s:.0f}s",
            "elapsed_s": elapsed,
            "stderr_tail": _tail(stderr),
        }

    status, detail = _classify(stage, proc.returncode, proc.stderr)
    return {
        "stage": stage,
        "status": status,
        "detail": detail,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "stderr_tail": _tail(proc.stderr),
    }


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _print_summary(results: list[dict]) -> None:
    print("[gpt-oss-scope] stage summary")
    for r in results:
        print(
            f"  [{r['status']}] {r['stage']} "
            f"({r.get('elapsed_s', 0.0):.1f}s) {r.get('detail', '')}".rstrip()
        )


def _stages_from_env() -> list[str]:
    raw = os.environ.get("GPT_OSS_SCOPE_STAGES", "").strip()
    if not raw:
        return []
    stages = [s.strip() for s in raw.split(",") if s.strip()]
    invalid = [s for s in stages if s not in STAGES]
    if invalid:
        raise ValueError(f"invalid GPT_OSS_SCOPE_STAGES entries: {invalid}")
    return stages


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--driver", default=DEFAULT_SCOPE_DRIVER,
                    help=f"path to run_gpt_oss.py (default {DEFAULT_SCOPE_DRIVER})")
    ap.add_argument("--stage", action="append", choices=STAGES,
                    help="stage to run; repeatable. Default: all stages")
    ap.add_argument("--list-stages", action="store_true",
                    help="print valid stage names and exit")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-stage timeout in seconds (default 300)")
    ap.add_argument("--json", default="",
                    help="write machine-readable stage report")
    args = ap.parse_args(argv)

    if args.list_stages:
        for stage in STAGES:
            print(stage)
        return 0

    if not os.path.isfile(args.driver):
        raise FileNotFoundError(f"scope-discovery driver not found: {args.driver}")

    stages = args.stage or _stages_from_env() or STAGES
    print(
        f"[gpt-oss-scope] mode={_mode_label()} stages={len(stages)} "
        f"driver={args.driver}",
        flush=True,
    )

    results = [_run_stage(args.driver, stage, args.timeout) for stage in stages]
    _print_summary(results)

    json_path = _default_json_path(args.json)
    if json_path:
        _write_json(json_path, {
            "generated": datetime.datetime.now().isoformat(),
            "mode": _mode_label(),
            "driver": args.driver,
            "results": results,
        })
        print(f"[gpt-oss-scope] wrote {json_path}", flush=True)

    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
