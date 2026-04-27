#!/usr/bin/env python3
"""Build a per-run index for GPT-OSS exact kernel fixtures.

The index is deliberately lightweight: it ties together capture manifest
entries, replay verdicts, and optional launch/proof logs so agents can choose
one kernel fixture and know whether they are looking at a true correctness
failure, a Salmon translation gap, or replay infrastructure that still needs
work.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _fixture_key(path: str) -> str:
    return str(Path(path))


def _launch_symbols(launch_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    symbols: dict[str, set[str]] = defaultdict(set)
    for row in launch_rows:
        if row.get("event") != "hip_module_launch" or row.get("phase") != "before":
            continue
        name = row.get("kernel_name")
        if not name:
            continue
        symbols[name].add(name)
        if "_matmul_ogs" in name:
            symbols["_matmul_ogs"].add(name)
        if name == "_reduce":
            symbols["_reduce"].add(name)
    return {k: sorted(v) for k, v in symbols.items()}


def _next_action(status: str) -> str:
    if status == "fail":
        status = "wrong-result"
    if status == "native-crash":
        status = "native-runtime-fail"
    if status == "salmon-crash":
        status = "salmon-runtime-fail"
    if status == "pass":
        return "No immediate action; keep as regression coverage."
    if status == "wrong-result":
        return "Assign as Salmon correctness bug; inspect mismatching tensor path and reduce fixture."
    if status == "salmon-translate-fail":
        return "Assign as Salmon opcode/translation coverage bug."
    if status == "salmon-runtime-fail":
        return "Assign as Salmon runtime/ABI/codegen bug; rerun with serialized HIP if needed."
    if status == "native-runtime-fail":
        return "Fix replay context first; native must pass before comparing Salmon."
    if status == "replay-unsupported":
        return "Fix fixture replay reconstruction; not a kernel verdict yet."
    return "Investigate unclassified status."


def _detail(result: dict[str, Any] | None) -> str:
    if not result:
        return "not replayed"
    status = result.get("status", "")
    if status == "pass":
        return "mismatches=0"
    if status in {"wrong-result", "fail"}:
        return (
            f"mismatches={result.get('mismatches')} "
            f"max_abs={result.get('max_abs')} tensor={result.get('tensor_path')}"
        )
    log = result.get("log", "")
    for line in log.splitlines():
        if (
            "ReplayUnsupported" in line
            or "unsupported instruction" in line
            or "ModuleNotFoundError" in line
            or "illegal memory access" in line
            or "failed to specialize" in line
        ):
            return line.strip()
    return status


def build_index(
    fixture_dir: Path,
    replay_path: Path | None,
    launch_path: Path | None,
    proof_path: Path | None,
) -> list[dict[str, Any]]:
    manifest_rows = _read_jsonl(fixture_dir / "manifest.jsonl")
    replay_rows = _read_jsonl(replay_path) if replay_path else []
    launch_rows = _read_jsonl(launch_path) if launch_path else []
    proof_rows = _read_jsonl(proof_path) if proof_path else []

    replay_by_fixture = {_fixture_key(r["fixture"]): r for r in replay_rows if "fixture" in r}
    launch_by_kernel = _launch_symbols(launch_rows)
    salmon_ok = sum(1 for r in proof_rows if r.get("event") == "salmon_result" and r.get("success"))
    salmon_failed = sum(1 for r in proof_rows if r.get("event") == "salmon_result" and r.get("success") is False)

    entries: list[dict[str, Any]] = []
    for row in manifest_rows:
        fixture = _fixture_key(row["path"])
        result = replay_by_fixture.get(fixture)
        kernel = row["kernel_name"]
        compiled_symbols = launch_by_kernel.get(kernel, [])
        entries.append(
            {
                "fixture": fixture,
                "python_module": row.get("module_name", ""),
                "python_kernel": kernel,
                "compiled_symbols": compiled_symbols,
                "replay_status": result.get("status", "not-replayed") if result else "not-replayed",
                "detail": _detail(result),
                "next_action": _next_action(result.get("status", "not-replayed") if result else "not-replayed"),
            }
        )

    return [
        {
            "summary": {
                "fixture_dir": str(fixture_dir),
                "fixture_count": len(entries),
                "salmon_ok_count": salmon_ok,
                "salmon_failed_count": salmon_failed,
            },
            "entries": entries,
        }
    ]


def write_outputs(index: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = index[0]
    (out_dir / "index.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    lines = [
        "# GPT-OSS Fixture Index",
        "",
        f"- Fixture dir: `{data['summary']['fixture_dir']}`",
        f"- Fixtures: **{data['summary']['fixture_count']}**",
        f"- Salmon OK count from proof: **{data['summary']['salmon_ok_count']}**",
        f"- Salmon FAILED count from proof: **{data['summary']['salmon_failed_count']}**",
        "",
        "| Status | Python kernel | Compiled symbol(s) | Detail | Next action |",
        "|---|---|---|---|---|",
    ]
    for e in data["entries"]:
        symbols = ", ".join(f"`{s}`" for s in e["compiled_symbols"]) or "n/a"
        lines.append(
            "| {status} | `{mod}.{kern}` | {symbols} | {detail} | {action} |".format(
                status=e["replay_status"],
                mod=e["python_module"],
                kern=e["python_kernel"],
                symbols=symbols,
                detail=str(e["detail"]).replace("|", "\\|"),
                action=e["next_action"].replace("|", "\\|"),
            )
        )
    (out_dir / "index.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture_dir", type=Path)
    ap.add_argument("--replay", type=Path, default=None)
    ap.add_argument("--launch", type=Path, default=None)
    ap.add_argument("--proof", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    fixture_dir = args.fixture_dir.resolve()
    out_dir = args.out_dir or fixture_dir
    index = build_index(fixture_dir, args.replay, args.launch, args.proof)
    write_outputs(index, out_dir)
    print(out_dir / "index.md")
    print(out_dir / "index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
