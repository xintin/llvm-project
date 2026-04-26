#!/usr/bin/env python3
"""Compare native gfx942 GPT-OSS against gfx1250->gfx942 Salmon GPT-OSS.

This harness runs the same deterministic prompt twice:

* native: SGLang on gfx942, no Salmon preload, using the MI300-friendly BF16
  GPT-OSS checkpoint.
* salmon: SGLang forced to compile gfx1250 wave32 kernels and load them through
  Salmon, using the canonical OpenAI MXFP4 checkpoint.

The result is a detailed JSON report and a Markdown summary with output text,
token/logprob diffs, timing, and a structured Salmon proof gate. Model data
lives under /data/gpt-oss; reports, logs, proof JSONL, and dumps go under /tmp
by default.
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import math
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[7]
SMOKE = HERE / "sglang_gpt_oss_smoke.py"
DEFAULT_VENV = HERE / ".venv-sglang-rocm720"
DEFAULT_LIBHSA = (
    REPO / "projects/rocr-runtime/build/rocr/lib/libhsa-runtime64.so.1"
)
DEFAULT_LIBSALMON = (
    HERE.parent / "compare_correctness/libsalmon_intercept.so"
)
DEFAULT_RULES = HERE / "_empty_rules.json"
DEFAULT_NATIVE_MODEL_ROOT = "/data/gpt-oss/hf-cache/models--lmsys--gpt-oss-20b-bf16"
DEFAULT_SALMON_MODEL_ROOT = "/data/gpt-oss/hf-cache/models--openai--gpt-oss-20b"


@dataclass
class RunSpec:
    name: str
    model_path: str
    result_json: Path
    log_path: Path
    proof_jsonl: Optional[Path]
    dump_dir: Optional[Path]
    salmon: bool
    extra_env: dict[str, str]


def _latest_snapshot(model_root: str) -> str:
    root = Path(model_root)
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(f"snapshot directory not found: {snapshots}")
    candidates = [p for p in snapshots.iterdir() if (p / "config.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"no snapshots with config.json under {snapshots}")
    # Prefer the most recently materialized snapshot if there are multiple.
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def _base_env(venv: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{HERE}:{env.get('PYTHONPATH', '')}"
    env["HIP_VISIBLE_DEVICES"] = env.get("HIP_VISIBLE_DEVICES", "0")
    env["TRITON_CORPUS_SCRIPT"] = str(SMOKE)
    env["GPT_OSS_SGLANG_MAX_NEW_TOKENS"] = env.get(
        "GPT_OSS_SGLANG_MAX_NEW_TOKENS", "16"
    )
    env["GPT_OSS_SGLANG_RETURN_HIDDEN_STATES"] = env.get(
        "GPT_OSS_SGLANG_RETURN_HIDDEN_STATES", "1"
    )
    env.pop("GPT_OSS_SGLANG_PROMPT", None)
    return env


def _run(venv: Path, spec: RunSpec, timeout_s: float) -> dict:
    python = venv / "bin/python"
    if not python.exists():
        raise FileNotFoundError(f"venv python not found: {python}")

    env = _base_env(venv)
    env.update(spec.extra_env)
    env["GPT_OSS_MODEL_PATH"] = spec.model_path
    env["GPT_OSS_SGLANG_RESULT_JSON"] = str(spec.result_json)
    env["SGLANG_LOG_LEVEL"] = env.get("SGLANG_LOG_LEVEL", "warning")
    if spec.proof_jsonl is not None:
        env["HSA_SALMON_PROOF_LOG"] = str(spec.proof_jsonl)
    else:
        env.pop("HSA_SALMON_PROOF_LOG", None)

    if spec.salmon:
        env["TRITON_CORPUS_FORCE_TARGET"] = "gfx1250:32"
        env["HSA_HOTSWAP_ISA_OVERRIDE"] = "gfx942"
        env["HSA_HOTSWAP_RULES"] = str(DEFAULT_RULES)
        env["HSA_HOTSWAP_IR_RAISER"] = "1"
        env["HSA_SALMON_STRICT"] = "1"
        env["LD_PRELOAD"] = f"{DEFAULT_LIBHSA}:{DEFAULT_LIBSALMON}"
        env["LD_LIBRARY_PATH"] = (
            f"{DEFAULT_LIBHSA.parent}:{env.get('LD_LIBRARY_PATH', '')}"
        )
        if spec.dump_dir is not None:
            env["HSA_SALMON_DUMP_DIR"] = str(spec.dump_dir)
    else:
        for key in (
            "TRITON_CORPUS_FORCE_TARGET",
            "HSA_HOTSWAP_ISA_OVERRIDE",
            "HSA_HOTSWAP_RULES",
            "HSA_HOTSWAP_IR_RAISER",
            "HSA_SALMON_STRICT",
            "HSA_SALMON_DUMP_DIR",
            "LD_PRELOAD",
        ):
            env.pop(key, None)

    spec.result_json.unlink(missing_ok=True)
    if spec.proof_jsonl is not None:
        spec.proof_jsonl.unlink(missing_ok=True)
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.dump_dir is not None:
        subprocess.run(["rm", "-rf", str(spec.dump_dir)], check=False)

    with open(spec.log_path, "w") as log:
        proc = subprocess.run(
            [str(python), "-m", "_bootstrap"],
            cwd=HERE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )

    result = {
        "name": spec.name,
        "returncode": proc.returncode,
        "log_path": str(spec.log_path),
        "result_json": str(spec.result_json),
        "proof_jsonl": str(spec.proof_jsonl) if spec.proof_jsonl else None,
    }
    if spec.result_json.is_file():
        with open(spec.result_json) as f:
            result["result"] = json.load(f)
    else:
        result["result"] = None
    return result


def _read_jsonl(path: Path) -> list[dict]:
    events = []
    if not path.is_file():
        return events
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                events.append(
                    {
                        "event": "malformed_jsonl",
                        "line": line_no,
                        "error": str(exc),
                        "raw": line,
                    }
                )
    return events


def _salmon_proof(proof_path: Path) -> dict:
    events = _read_jsonl(proof_path)
    if not events:
        return {
            "valid": False,
            "reason": f"proof log missing or empty: {proof_path}",
            "events": [],
        }

    forced = [e for e in events if e.get("event") == "triton_forced_target"]
    decisions = [e for e in events if e.get("event") == "transpile_decision"]
    salmon_results = [e for e in events if e.get("event") == "salmon_result"]
    ok_results = [e for e in salmon_results if e.get("success") is True]
    failed_results = [e for e in salmon_results if e.get("success") is False]
    proof = {
        "proof_log": str(proof_path),
        "forced_child_target": any(
            e.get("arch") == "gfx1250" and int(e.get("warp_size", 0)) == 32
            for e in forced
        ),
        "forced_target_event_count": len(forced),
        "orig_mach_values": sorted(
            {str(e.get("orig_mach")) for e in decisions if e.get("orig_mach")}
        ),
        "saw_gfx1250_code_object": any(
            e.get("source_gfx") == "gfx1250" for e in decisions
        ),
        "saw_gfx942_code_object": any(
            e.get("source_gfx") == "gfx942" for e in decisions
        ),
        "saw_loader_transpile_decision": any(
            e.get("source_gfx") == "gfx1250" and e.get("target_gfx") == "gfx942"
            for e in decisions
        ),
        "salmon_ok_count": len(ok_results),
        "salmon_failed_count": len(failed_results),
        "first_salmon_ok": ok_results[0] if ok_results else None,
        "first_salmon_failed": failed_results[0] if failed_results else None,
        "first_unsupported_instruction": (
            failed_results[0].get("fail_mnemonic") if failed_results else None
        ),
        "events": events,
    }
    proof["valid"] = (
        proof["forced_child_target"]
        and proof["saw_gfx1250_code_object"]
        and proof["saw_loader_transpile_decision"]
        and proof["salmon_ok_count"] > 0
        and proof["salmon_failed_count"] == 0
    )
    missing = []
    if not proof["forced_child_target"]:
        missing.append("child-process gfx1250 target forcing")
    if not proof["saw_gfx1250_code_object"]:
        missing.append("gfx1250 code object load")
    if not proof["saw_loader_transpile_decision"]:
        missing.append("loader-side Salmon transpile decision")
    if proof["salmon_ok_count"] == 0:
        missing.append("successful Salmon translation")
    if proof["salmon_failed_count"] > 0:
        missing.append("no Salmon translation failures")
    proof["reason"] = "ok" if proof["valid"] else "; missing/failed: " + ", ".join(missing)
    return proof


def _lp_value(item) -> float | None:  # noqa: ANN001
    if isinstance(item, (list, tuple)) and item:
        return item[0]
    return None


def _lp_token(item) -> int | None:  # noqa: ANN001
    if isinstance(item, (list, tuple)) and len(item) > 1:
        return item[1]
    return None


def _logprob_metrics(native_case: dict, salmon_case: dict) -> dict:
    n_meta = native_case.get("meta_info") or {}
    s_meta = salmon_case.get("meta_info") or {}
    n_lps = n_meta.get("output_token_logprobs") or []
    s_lps = s_meta.get("output_token_logprobs") or []
    n = min(len(n_lps), len(s_lps))
    diffs = []
    token_mismatches = 0
    first_mismatch = None
    for i in range(n):
        ntok, stok = _lp_token(n_lps[i]), _lp_token(s_lps[i])
        if ntok != stok:
            token_mismatches += 1
            if first_mismatch is None:
                first_mismatch = {
                    "index": i,
                    "native_token": ntok,
                    "salmon_token": stok,
                }
        nlp, slp = _lp_value(n_lps[i]), _lp_value(s_lps[i])
        if nlp is not None and slp is not None:
            diffs.append(float(slp) - float(nlp))

    abs_diffs = [abs(x) for x in diffs]
    top_metrics = []
    n_top = n_meta.get("output_top_logprobs") or []
    s_top = s_meta.get("output_top_logprobs") or []
    for i in range(min(len(n_top), len(s_top))):
        n_map = {_lp_token(x): _lp_value(x) for x in n_top[i] if _lp_token(x) is not None}
        s_map = {_lp_token(x): _lp_value(x) for x in s_top[i] if _lp_token(x) is not None}
        overlap = set(n_map) & set(s_map)
        top_diffs = [
            float(s_map[tok]) - float(n_map[tok])
            for tok in overlap
            if n_map[tok] is not None and s_map[tok] is not None
        ]
        top_metrics.append(
            {
                "index": i,
                "native_top_tokens": list(n_map.keys()),
                "salmon_top_tokens": list(s_map.keys()),
                "overlap": len(overlap),
                "mean_abs_logprob_delta": (
                    sum(abs(x) for x in top_diffs) / len(top_diffs)
                    if top_diffs
                    else None
                ),
                "max_abs_logprob_delta": (
                    max(abs(x) for x in top_diffs) if top_diffs else None
                ),
            }
        )

    return {
        "native_output_ids": native_case.get("output_ids", []),
        "salmon_output_ids": salmon_case.get("output_ids", []),
        "output_token_count_native": len(n_lps),
        "output_token_count_salmon": len(s_lps),
        "compared_output_token_count": n,
        "output_token_id_mismatches": token_mismatches,
        "first_output_token_id_mismatch": first_mismatch,
        "selected_logprob_delta_mean": sum(diffs) / len(diffs) if diffs else None,
        "selected_logprob_delta_mean_abs": (
            sum(abs_diffs) / len(abs_diffs) if abs_diffs else None
        ),
        "selected_logprob_delta_max_abs": max(abs_diffs) if abs_diffs else None,
        "selected_logprob_delta_l2": (
            math.sqrt(sum(x * x for x in diffs)) if diffs else None
        ),
        "top_logprob_metrics": top_metrics,
    }


def _tensor_data(meta_info: dict, key: str) -> list[float] | None:
    value = meta_info.get(key)
    if isinstance(value, dict) and value.get("__tensor__") and isinstance(
        value.get("data"), list
    ):
        return [float(x) for x in value["data"]]
    return None


def _tensor_metrics(native_case: dict, salmon_case: dict) -> dict | None:
    n_meta = native_case.get("meta_info") or {}
    s_meta = salmon_case.get("meta_info") or {}
    n_tensor = _tensor_data(n_meta, "hidden_states")
    s_tensor = _tensor_data(s_meta, "hidden_states")
    if n_tensor is None or s_tensor is None:
        return None
    n = min(len(n_tensor), len(s_tensor))
    if n == 0:
        return None
    diffs = [s_tensor[i] - n_tensor[i] for i in range(n)]
    abs_diffs = [abs(x) for x in diffs]
    n_l2 = math.sqrt(sum(x * x for x in n_tensor[:n]))
    diff_l2 = math.sqrt(sum(x * x for x in diffs))
    return {
        "native_shape": n_meta.get("hidden_states", {}).get("shape"),
        "salmon_shape": s_meta.get("hidden_states", {}).get("shape"),
        "compared_elements": n,
        "mean_abs": sum(abs_diffs) / n,
        "max_abs": max(abs_diffs),
        "l2": diff_l2,
        "relative_l2": diff_l2 / n_l2 if n_l2 else None,
        "mean_signed": sum(diffs) / n,
    }


def _fmt(value, digits: int = 4) -> str:  # noqa: ANN001
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _completion_tokens(result: dict) -> int:
    return sum(
        (case.get("meta_info", {}).get("completion_tokens") or 0)
        for case in result.get("cases", [])
    )


def _aggregate_case_stats(cases: list[dict]) -> dict:
    comparable = [c for c in cases if c.get("logprob_metrics")]
    mismatched = [
        c for c in comparable if c["logprob_metrics"]["output_token_id_mismatches"]
    ]
    selected_mean_abs = [
        c["logprob_metrics"]["selected_logprob_delta_mean_abs"]
        for c in comparable
        if c["logprob_metrics"]["selected_logprob_delta_mean_abs"] is not None
    ]
    selected_max_abs = [
        c["logprob_metrics"]["selected_logprob_delta_max_abs"]
        for c in comparable
        if c["logprob_metrics"]["selected_logprob_delta_max_abs"] is not None
    ]
    numeric = [
        c
        for c in cases
        if c.get("expected_number") is not None
        and c.get("native_numeric_error") is not None
        and c.get("salmon_numeric_error") is not None
    ]
    return {
        "cases": len(cases),
        "token_mismatch_cases": len(mismatched),
        "avg_selected_logprob_mean_abs_delta": (
            sum(selected_mean_abs) / len(selected_mean_abs)
            if selected_mean_abs
            else None
        ),
        "max_selected_logprob_abs_delta": (
            max(selected_max_abs) if selected_max_abs else None
        ),
        "numeric_cases": len(numeric),
        "numeric_exact_cases": sum(
            1
            for c in numeric
            if c["native_numeric_error"] == 0
            and c["salmon_numeric_error"] == 0
            and c["salmon_minus_native_numeric"] == 0
        ),
    }


def _write_markdown_report(report: dict, report_path: Path) -> None:
    native = report["runs"][0]["result"] or {}
    salmon = report["runs"][1]["result"] or {}
    comparison = report["comparison"]
    proof = comparison.get("salmon_proof", {})
    cases = comparison["cases"]
    stats = _aggregate_case_stats(cases)

    native_tokens = _completion_tokens(native)
    salmon_tokens = _completion_tokens(salmon)
    native_gen_s = native.get("generate_s")
    salmon_gen_s = salmon.get("generate_s")
    native_tps = native_tokens / native_gen_s if native_gen_s else None
    salmon_tps = salmon_tokens / salmon_gen_s if salmon_gen_s else None

    lines = [
        "# GPT-OSS SGLang Native vs Salmon Summary",
        "",
        f"Generated: `{report['generated']}`",
        "",
        "## Verdict",
        "",
        f"- Both runs passed: **{_fmt(comparison['both_passed'])}**",
        f"- Salmon proof valid: **{_fmt(proof.get('valid'))}**",
        f"- Salmon proof reason: `{proof.get('reason', 'n/a')}`",
        f"- Cases run: **{stats['cases']}**",
        f"- Cases with output-token mismatches: **{stats['token_mismatch_cases']}**",
        f"- Numeric exact cases: **{stats['numeric_exact_cases']} / {stats['numeric_cases']}**",
        f"- Average selected-token logprob |delta|: **{_fmt(stats['avg_selected_logprob_mean_abs_delta'])}**",
        f"- Max selected-token logprob |delta|: **{_fmt(stats['max_selected_logprob_abs_delta'])}**",
        "",
        "## Salmon Translation Proof",
        "",
        "| Check | Value |",
        "|---|---:|",
        f"| Child process forced gfx1250 target | {_fmt(proof.get('forced_child_target'))} |",
        f"| Parent process forced gfx1250 target | {_fmt(proof.get('forced_parent_target'))} |",
        f"| Saw gfx1250 code object (`orig_mach=0x49`) | {_fmt(proof.get('saw_gfx1250_code_object'))} |",
        f"| Saw loader transpile decision | {_fmt(proof.get('saw_loader_transpile_decision'))} |",
        f"| Salmon OK count | {_fmt(proof.get('salmon_ok_count'))} |",
        f"| Salmon FAILED count | {_fmt(proof.get('salmon_failed_count'))} |",
        f"| First unsupported instruction | `{proof.get('first_unsupported_instruction') or 'n/a'}` |",
        "",
        "## Models",
        "",
        f"- Native: `{report['native_model']}`",
        f"- Salmon: `{report['salmon_model']}`",
        "",
        "## Timing",
        "",
        "| Run | Engine init (s) | Generate (s) | Total (s) | Output tokens | Tokens/s |",
        "|---|---:|---:|---:|---:|---:|",
        "| Native | "
        f"{_fmt(native.get('engine_init_s'))} | {_fmt(native_gen_s)} | "
        f"{_fmt(native.get('total_s'))} | {native_tokens} | {_fmt(native_tps)} |",
        "| Salmon | "
        f"{_fmt(salmon.get('engine_init_s'))} | {_fmt(salmon_gen_s)} | "
        f"{_fmt(salmon.get('total_s'))} | {salmon_tokens} | {_fmt(salmon_tps)} |",
        "",
        "## Per-Case Diffs",
        "",
        "| Case | Text sim | Token mismatches | Mean |delta logprob| | Max |delta logprob| | Native tok/s | Salmon tok/s | Numeric delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        metrics = case.get("logprob_metrics") or {}
        lines.append(
            "| "
            f"{case['name']} | "
            f"{_fmt(case.get('sequence_similarity'))} | "
            f"{metrics.get('output_token_id_mismatches', 'n/a')} / {metrics.get('compared_output_token_count', 'n/a')} | "
            f"{_fmt(metrics.get('selected_logprob_delta_mean_abs'))} | "
            f"{_fmt(metrics.get('selected_logprob_delta_max_abs'))} | "
            f"{_fmt(case.get('native_tokens_per_s'))} | "
            f"{_fmt(case.get('salmon_tokens_per_s'))} | "
            f"{_fmt(case.get('salmon_minus_native_numeric'))} |"
        )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "cd projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/tools/triton_corpus_runner",
            "GPT_OSS_SGLANG_RETURN_HIDDEN_STATES=0 \\",
            "  python3 compare_gpt_oss_sglang.py \\",
            "    --native-mxfp4 \\",
            "    --out-dir /tmp/gpt_oss_sglang_compare_mxfp4_native \\",
            "    --timeout 900",
            "```",
            "",
            f"Detailed JSON: `{report_path}`",
        ]
    )
    md_path = report_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--venv", default=str(DEFAULT_VENV))
    ap.add_argument("--native-model", default="")
    ap.add_argument("--salmon-model", default="")
    ap.add_argument(
        "--native-mxfp4",
        action="store_true",
        help="use the canonical MXFP4 checkpoint for native too, forcing "
        "SGLang's MXFP4 gate without Salmon",
    )
    ap.add_argument("--out-dir", default="/tmp/gpt_oss_sglang_compare")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    salmon_model = args.salmon_model or _latest_snapshot(DEFAULT_SALMON_MODEL_ROOT)
    if args.native_mxfp4:
        native_model = args.native_model or salmon_model
    else:
        native_model = args.native_model or _latest_snapshot(DEFAULT_NATIVE_MODEL_ROOT)

    native = RunSpec(
        name="native_gfx942_bf16",
        model_path=native_model,
        result_json=out_dir / "native_result.json",
        log_path=out_dir / "native.log",
        proof_jsonl=None,
        dump_dir=None,
        salmon=False,
        extra_env={
            "SGLANG_USE_AITER": "0",
            "GPT_OSS_SGLANG_MOE_RUNNER_BACKEND": "triton_kernel",
            "AITER_ROOT": "/home/mluecke/aiter-sglang-v0.1.12.post1",
            "AITER_CORPUS_CONFIG_DIR": "/tmp/sglang_aiter_configs_1060",
            **({"GPT_OSS_SGLANG_FORCE_MXFP4": "1"} if args.native_mxfp4 else {}),
        },
    )
    salmon = RunSpec(
        name="salmon_gfx1250_to_gfx942_mxfp4",
        model_path=salmon_model,
        result_json=out_dir / "salmon_result.json",
        log_path=out_dir / "salmon.log",
        proof_jsonl=out_dir / "salmon_proof.jsonl",
        dump_dir=out_dir / "salmon_dump",
        salmon=True,
        extra_env={
            "SGLANG_USE_AITER": "0",
            "GPT_OSS_SGLANG_MOE_RUNNER_BACKEND": "triton_kernel",
            "AITER_ROOT": "/home/mluecke/aiter-sglang-v0.1.12.post1",
            "AITER_CORPUS_CONFIG_DIR": "/tmp/sglang_aiter_configs_1060",
        },
    )

    runs = [_run(Path(args.venv), native, args.timeout), _run(Path(args.venv), salmon, args.timeout)]

    native_result = runs[0].get("result") or {}
    salmon_result = runs[1].get("result") or {}
    native_text = native_result.get("output", "")
    salmon_text = salmon_result.get("output", "")
    similarity = (
        difflib.SequenceMatcher(None, native_text, salmon_text).ratio()
        if native_text and salmon_text
        else 0.0
    )

    native_cases = {c["name"]: c for c in native_result.get("cases", [])}
    salmon_cases = {c["name"]: c for c in salmon_result.get("cases", [])}
    case_comparisons = []
    for name in sorted(set(native_cases) | set(salmon_cases)):
        n = native_cases.get(name, {})
        s = salmon_cases.get(name, {})
        n_text = n.get("output", "")
        s_text = s.get("output", "")
        n_num = n.get("observed_number")
        s_num = s.get("observed_number")
        numeric_delta = None
        if n_num is not None and s_num is not None:
            numeric_delta = float(s_num) - float(n_num)
        case_comparisons.append(
            {
                "name": name,
                "native_output": n_text,
                "salmon_output": s_text,
                "sequence_similarity": (
                    difflib.SequenceMatcher(None, n_text, s_text).ratio()
                    if n_text and s_text
                    else 0.0
                ),
                "expected_number": n.get("expected_number", s.get("expected_number")),
                "native_observed_number": n_num,
                "salmon_observed_number": s_num,
                "native_numeric_error": n.get("numeric_error"),
                "salmon_numeric_error": s.get("numeric_error"),
                "salmon_minus_native_numeric": numeric_delta,
                "native_generate_s": n.get("generate_s"),
                "salmon_generate_s": s.get("generate_s"),
                "native_tokens_per_s": (
                    (n.get("meta_info", {}).get("completion_tokens") or 0)
                    / n["generate_s"]
                    if n.get("generate_s")
                    else None
                ),
                "salmon_tokens_per_s": (
                    (s.get("meta_info", {}).get("completion_tokens") or 0)
                    / s["generate_s"]
                    if s.get("generate_s")
                    else None
                ),
                "logprob_metrics": _logprob_metrics(n, s),
                "hidden_state_metrics": _tensor_metrics(n, s),
            }
        )

    report = {
        "generated": datetime.datetime.now().isoformat(),
        "native_model": native_model,
        "salmon_model": salmon_model,
        "runs": runs,
        "comparison": {
            "native_output": native_text,
            "salmon_output": salmon_text,
            "sequence_similarity": similarity,
            "cases": case_comparisons,
            "salmon_proof": _salmon_proof(Path(runs[1]["proof_jsonl"])),
        },
    }
    report["comparison"]["both_passed"] = (
        all(r["returncode"] == 0 for r in runs)
        and report["comparison"]["salmon_proof"]["valid"]
    )
    report_path = out_dir / "compare_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    _write_markdown_report(report, report_path)

    print(json.dumps(report["comparison"], indent=2))
    print(f"report: {report_path}")
    print(f"summary: {report_path.with_suffix('.md')}")
    return 0 if report["comparison"]["both_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
