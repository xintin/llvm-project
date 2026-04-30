#!/usr/bin/env python3
"""Run a deterministic GPT-OSS request through SGLang's offline engine.

This script is intended to be launched by ``triton_corpus_runner``:

  GPT_OSS_MODEL_PATH=/path/to/gpt-oss \
    python3 runner.py --script ./sglang_gpt_oss_smoke.py --modes native,salmon

The runner supplies the Salmon preload, forced gfx1250 Triton target, strict
mode, and artifact-capture environment. This script only owns the SGLang-level
request and its success criteria.

No fallback behavior is used: if SGLang is unavailable, the model path is not
provided, or generation produces no text, the script exits non-zero with a
clear error.
"""

from __future__ import annotations

import os
import json
import re
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AITER_ROOT = os.path.normpath(
    os.path.join(HERE, "..", "aiter_corpus_runner", "_aiter")
)


def _prepend_aiter_path() -> None:
    aiter_root = os.environ.get("AITER_ROOT", DEFAULT_AITER_ROOT).strip()
    if os.path.isdir(aiter_root):
        if aiter_root not in sys.path:
            sys.path.insert(0, aiter_root)
        print(f"[sglang-gpt-oss] AITER_ROOT={aiter_root}", flush=True)


def _patch_aiter_config_dir() -> None:
    """Redirect AITER's hardcoded /tmp/aiter_configs on shared machines."""
    try:
        import aiter.jit.core as core
    except ImportError:
        return

    user_cfg_dir = os.environ.get("AITER_CORPUS_CONFIG_DIR", "").strip()
    if not user_cfg_dir:
        user_cfg_dir = os.path.join("/tmp", f"sglang_aiter_configs_{os.getuid()}")
    os.environ.setdefault("AITER_CORPUS_CONFIG_DIR", user_cfg_dir)
    os.makedirs(user_cfg_dir, exist_ok=True)

    target_literal = "/tmp/aiter_configs/"

    def _wrapped(self, tuned_files, merge_name):  # noqa: ANN001
        import pandas as pd

        source_paths = [p for p in str(tuned_files).split(":") if p]
        if not source_paths:
            raise RuntimeError(f"AITER config merge {merge_name!r} has no sources")
        missing = [p for p in source_paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"AITER config merge {merge_name!r} missing files: {missing}"
            )

        frames = [pd.read_csv(p) for p in source_paths]
        merge_df = pd.concat(frames, ignore_index=True)
        merge_df = merge_df.drop_duplicates(keep="first").reset_index(drop=True)
        out_path = os.path.join(user_cfg_dir, f"{merge_name}.csv")
        tmp_path = f"{out_path}.tmp"
        merge_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, out_path)
        return out_path

    core.AITER_CONFIG.update_config_files = _wrapped
    print(
        f"[sglang-gpt-oss] AITER config dir={user_cfg_dir} "
        f"(redirected from {target_literal})",
        flush=True,
    )


def _patch_sglang_mxfp_support_for_forced_target() -> None:
    """Let SGLang accept GPT-OSS MXFP4 when Triton is forced to gfx1250.

    SGLang normally checks the *physical* HIP device before registering the
    ``mxfp4`` quantization method. In the Salmon path, the physical device is
    gfx942, but Triton compilation is deliberately forced to gfx1250 and the
    resulting HSACO is translated to gfx942 at load time. Without this patch,
    SGLang rejects the canonical ``openai/gpt-oss-*`` configs before any
    gfx1250 kernel can be compiled.
    """
    force_target = os.environ.get("TRITON_CORPUS_FORCE_TARGET", "")
    force_mxfp4 = _env_bool("GPT_OSS_SGLANG_FORCE_MXFP4", False)
    if not (force_mxfp4 or force_target.startswith("gfx125")):
        return

    import sglang.srt.utils as utils
    import sglang.srt.utils.common as common

    def _yes() -> bool:
        return True

    common.mxfp_supported = _yes
    utils.mxfp_supported = _yes
    print(
        "[sglang-gpt-oss] forced SGLang MXFP4 support "
        f"(TRITON_CORPUS_FORCE_TARGET={force_target!r})",
        flush=True,
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be boolean-like, got {raw!r}")


def _extract_text(output) -> str:  # noqa: ANN001
    if isinstance(output, dict):
        for key in ("text", "output", "generated_text"):
            value = output.get(key)
            if isinstance(value, str):
                return value
    text = getattr(output, "text", None)
    if isinstance(text, str):
        return text
    raise TypeError(f"cannot extract generated text from SGLang output: {output!r}")


DEFAULT_CASES = [
    {
        "name": "identity_salmon",
        "prompt": "Answer with exactly one word: salmon",
        "expected_text": "salmon",
    },
    {
        "name": "add_two_numbers",
        "prompt": "Answer with exactly one integer and no explanation: 17 + 25",
        "expected_number": 42,
    },
    {
        "name": "multiply_two_numbers",
        "prompt": "Answer with exactly one integer and no explanation: 13 * 7",
        "expected_number": 91,
    },
    {
        "name": "division_exact",
        "prompt": "Answer with exactly one integer and no explanation: 144 / 12",
        "expected_number": 12,
    },
    {
        "name": "subtraction",
        "prompt": "Answer with exactly one integer and no explanation: 1000 - 365",
        "expected_number": 635,
    },
    {
        "name": "square",
        "prompt": "Answer with exactly one integer and no explanation: 19 squared",
        "expected_number": 361,
    },
]


def _load_cases() -> list[dict]:
    raw = os.environ.get("GPT_OSS_SGLANG_PROMPTS_JSON", "").strip()
    if raw:
        cases = json.loads(raw)
    else:
        legacy_prompt = os.environ.get("GPT_OSS_SGLANG_PROMPT", "").strip()
        if legacy_prompt:
            cases = [{"name": "prompt", "prompt": legacy_prompt}]
        else:
            cases = DEFAULT_CASES
    if not isinstance(cases, list) or not cases:
        raise ValueError("prompt suite must be a non-empty JSON list")
    for i, case in enumerate(cases):
        if not isinstance(case, dict) or not case.get("prompt"):
            raise ValueError(f"prompt suite entry {i} must contain a prompt")
        case.setdefault("name", f"case_{i}")
    return cases


def _extract_first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group(0)) if match else None


def _jsonable(value):  # noqa: ANN001
    try:
        import torch

        if isinstance(value, torch.Tensor):
            cpu = value.detach().to("cpu")
            return {
                "__tensor__": True,
                "shape": list(cpu.shape),
                "dtype": str(cpu.dtype),
                "data": cpu.float().reshape(-1).tolist(),
            }
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_result_json(payload: dict) -> None:
    path = os.environ.get("GPT_OSS_SGLANG_RESULT_JSON", "").strip()
    if not path:
        return
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> int:
    total_t0 = time.perf_counter()
    _prepend_aiter_path()
    _patch_aiter_config_dir()
    _patch_sglang_mxfp_support_for_forced_target()

    model_path = (
        os.environ.get("GPT_OSS_MODEL_PATH")
        or os.environ.get("SGLANG_MODEL_PATH")
        or ""
    ).strip()
    if not model_path:
        raise RuntimeError(
            "GPT_OSS_MODEL_PATH or SGLANG_MODEL_PATH must point at the GPT-OSS "
            "model to test."
        )

    cases = _load_cases()
    max_new_tokens = _env_int("GPT_OSS_SGLANG_MAX_NEW_TOKENS", 8)
    top_logprobs_num = _env_int("GPT_OSS_SGLANG_TOP_LOGPROBS", 5)
    return_hidden_states = _env_bool("GPT_OSS_SGLANG_RETURN_HIDDEN_STATES", False)
    tp_size = _env_int("GPT_OSS_SGLANG_TP_SIZE", 1)
    dtype = os.environ.get("GPT_OSS_SGLANG_DTYPE", "bfloat16")
    attention_backend = os.environ.get("GPT_OSS_SGLANG_ATTENTION_BACKEND", "triton")
    sampling_backend = os.environ.get("GPT_OSS_SGLANG_SAMPLING_BACKEND", "pytorch")
    moe_runner_backend = os.environ.get("GPT_OSS_SGLANG_MOE_RUNNER_BACKEND", "").strip()
    trust_remote_code = _env_bool("GPT_OSS_SGLANG_TRUST_REMOTE_CODE", True)
    disable_cuda_graph = _env_bool("GPT_OSS_SGLANG_DISABLE_CUDA_GRAPH", False)

    try:
        from sglang import Engine
    except ImportError as exc:
        raise RuntimeError(
            "SGLang is not importable in this environment. Install sglang in "
            "the runner venv or point --triton-venv at an environment that "
            "contains it."
        ) from exc

    print(f"[sglang-gpt-oss] model_path={model_path}", flush=True)
    print(
        "[sglang-gpt-oss] "
        f"tp_size={tp_size} dtype={dtype} max_new_tokens={max_new_tokens}",
        flush=True,
    )

    engine_kwargs = {
        "model_path": model_path,
        "tp_size": tp_size,
        "dtype": dtype,
        "attention_backend": attention_backend,
        "sampling_backend": sampling_backend,
        "trust_remote_code": trust_remote_code,
    }
    if return_hidden_states:
        engine_kwargs["enable_return_hidden_states"] = True
    if disable_cuda_graph:
        engine_kwargs["disable_cuda_graph"] = True
    if moe_runner_backend:
        engine_kwargs["moe_runner_backend"] = moe_runner_backend
    engine_t0 = time.perf_counter()
    engine = Engine(**engine_kwargs)
    engine_init_s = time.perf_counter() - engine_t0
    try:
        result_cases = []
        total_generate_s = 0.0
        for case in cases:
            gen_t0 = time.perf_counter()
            outputs = engine.generate(
                [case["prompt"]],
                {"temperature": 0.0, "max_new_tokens": max_new_tokens},
                return_logprob=True,
                logprob_start_len=0,
                top_logprobs_num=top_logprobs_num,
                return_hidden_states=return_hidden_states,
            )
            generate_s = time.perf_counter() - gen_t0
            total_generate_s += generate_s
            if not isinstance(outputs, list) or len(outputs) != 1:
                raise RuntimeError(f"expected one SGLang output, got {outputs!r}")
            text = _extract_text(outputs[0]).strip()
            print(f"[sglang-gpt-oss] {case['name']} output={text!r}", flush=True)
            if not text:
                raise RuntimeError(f"{case['name']}: SGLang generated empty response")
            observed_number = _extract_first_number(text)
            numeric_error = None
            if "expected_number" in case and observed_number is not None:
                numeric_error = observed_number - float(case["expected_number"])
            result_cases.append(
                {
                    "name": case["name"],
                    "prompt": case["prompt"],
                    "output": text,
                    "generate_s": generate_s,
                    "expected_text": case.get("expected_text"),
                    "expected_number": case.get("expected_number"),
                    "observed_number": observed_number,
                    "numeric_error": numeric_error,
                    "output_ids": _jsonable(outputs[0].get("output_ids", [])),
                    "meta_info": _jsonable(outputs[0].get("meta_info", {})),
                }
            )
        _write_result_json(
            {
                "model_path": model_path,
                "cases": result_cases,
                "output": result_cases[0]["output"],
                "max_new_tokens": max_new_tokens,
                "top_logprobs_num": top_logprobs_num,
                "return_hidden_states": return_hidden_states,
                "tp_size": tp_size,
                "dtype": dtype,
                "attention_backend": attention_backend,
                "sampling_backend": sampling_backend,
                "moe_runner_backend": moe_runner_backend or None,
                "engine_init_s": engine_init_s,
                "generate_s": total_generate_s,
                "total_s": time.perf_counter() - total_t0,
            }
        )
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()

    print("[sglang-gpt-oss] PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
