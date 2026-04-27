"""Opt-in capture of exact Triton JIT launches from GPT-OSS/SGLang.

Set ``GPT_OSS_KERNEL_FIXTURE_DIR`` to a writable directory before running the
SGLang smoke/comparison harness.  This module patches Triton's JIT launch path
and writes one ``.pt`` fixture per matching kernel.  Fixtures contain the
module/function name, launch grid, constexpr kwargs, and CPU copies of tensor
arguments before and after the launch.  They are intended to be replayed by
``tools/compare_correctness/gpt_oss_fixture_replay.py``.

The hook is deliberately disabled by default.  When enabled, capture failures
raise immediately: a partially written fixture is worse than no fixture because
it can create false confidence in a correctness run.
"""

from __future__ import annotations

import fnmatch
import inspect
import json
import os
import re
import threading
from pathlib import Path
from typing import Any


_PATCHED = False
_LOCK = threading.Lock()
_COUNTS: dict[str, int] = {}
SCHEMA_VERSION = 2


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _patterns() -> list[str]:
    raw = os.environ.get("GPT_OSS_KERNEL_FIXTURE_FILTER", "").strip()
    if not raw:
        return ["*"]
    return [p.strip() for p in raw.split(",") if p.strip()]


def _max_per_kernel() -> int:
    raw = os.environ.get("GPT_OSS_KERNEL_FIXTURE_MAX_PER_KERNEL", "").strip()
    return int(raw) if raw else 1


def _max_tensor_bytes() -> int:
    raw = os.environ.get("GPT_OSS_KERNEL_FIXTURE_MAX_TENSOR_BYTES", "").strip()
    # Large enough for the first GPT-OSS fixtures, but finite so accidental
    # full-model tensors fail loudly instead of filling /tmp.
    return int(raw) if raw else 512 * 1024 * 1024


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "kernel"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return {"repr": repr(value), "type": type(value).__name__}


def _object_identity(value: Any) -> dict[str, Any]:
    return {
        "module": type(value).__module__,
        "type": type(value).__name__,
        "repr": repr(value),
    }


def _tensor_payload(tensor: Any, *, phase: str, kernel_name: str, arg_index: int) -> dict[str, Any]:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("_tensor_payload expects a torch.Tensor")
    nbytes = tensor.numel() * tensor.element_size()
    max_bytes = _max_tensor_bytes()
    if nbytes > max_bytes:
        raise RuntimeError(
            f"refusing to capture {phase} tensor arg {arg_index} for {kernel_name}: "
            f"{nbytes} bytes exceeds GPT_OSS_KERNEL_FIXTURE_MAX_TENSOR_BYTES={max_bytes}"
        )
    return {
        "kind": "tensor",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "device": str(tensor.device),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
        "data": tensor.detach().cpu().contiguous(),
    }


def _dtype_payload(dtype: Any) -> dict[str, Any]:
    import torch

    if isinstance(dtype, torch.dtype):
        return {"kind": "torch_dtype", "name": str(dtype)}
    return {
        "kind": "python_object_repr",
        **_object_identity(dtype),
    }


def _arg_payload(value: Any, *, phase: str, kernel_name: str, arg_index: int) -> dict[str, Any]:
    import torch

    if isinstance(value, torch.Tensor):
        return _tensor_payload(value, phase=phase, kernel_name=kernel_name, arg_index=arg_index)
    if type(value).__module__ == "triton.tools.tensor_descriptor" and type(value).__name__ == "TensorDescriptor":
        return {
            "kind": "tensor_descriptor",
            "base": _tensor_payload(value.base, phase=phase, kernel_name=kernel_name, arg_index=arg_index),
            "shape": [int(x) for x in value.shape],
            "strides": [int(x) for x in value.strides],
            "block_shape": [int(x) for x in value.block_shape],
            "padding": value.padding,
        }
    if (
        type(value).__module__.startswith("triton.")
        and type(value).__name__ == "TensorDescriptor"
        and hasattr(value, "base")
        and hasattr(value, "shape")
        and hasattr(value, "strides")
        and hasattr(value, "block_shape")
    ):
        return {
            "kind": "tensor_descriptor",
            "descriptor_type": _object_identity(value),
            "base": _tensor_payload(value.base, phase=phase, kernel_name=kernel_name, arg_index=arg_index),
            "shape": [int(x) for x in value.shape],
            "strides": [int(x) for x in value.strides],
            "block_shape": [int(x) for x in value.block_shape],
            "padding": getattr(value, "padding", "zero"),
        }
    if type(value).__module__ == "triton_kernels.tensor" and type(value).__name__ == "Tensor":
        storage = value.storage
        data = storage.data if hasattr(storage, "data") else storage
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"unsupported triton_kernels Tensor storage type: {type(data)}")
        payload = _tensor_payload(data, phase=phase, kernel_name=kernel_name, arg_index=arg_index)
        return {
            "kind": "triton_tensor",
            "tensor": payload,
            "dtype": _dtype_payload(value.dtype),
            "shape": [int(x) for x in value.shape],
            "shape_max": [int(x) for x in value.shape_max],
            "layout": {
                "module": type(value.storage.layout).__module__,
                "type": type(value.storage.layout).__name__,
                "shape": [
                    int(x)
                    for x in getattr(
                        value.storage.layout,
                        "initial_shape",
                        value.storage.data.shape,
                    )
                ],
            },
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _arg_payload(v, phase=phase, kernel_name=kernel_name, arg_index=arg_index)
                for v in value
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _arg_payload(v, phase=phase, kernel_name=kernel_name, arg_index=arg_index)
                for v in value
            ],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": {
                str(k): _arg_payload(v, phase=phase, kernel_name=kernel_name, arg_index=arg_index)
                for k, v in value.items()
            },
        }
    return {
        "kind": "value",
        "type": type(value).__name__,
        "value": _jsonable(value),
    }


def _same_payload(before: dict[str, Any], after: dict[str, Any]) -> bool:
    import torch

    if before.get("kind") != after.get("kind"):
        return False
    kind = before.get("kind")
    if kind == "tensor":
        return (
            before.get("dtype") == after.get("dtype")
            and before.get("shape") == after.get("shape")
            and torch.equal(before["data"], after["data"])
        )
    if kind == "triton_tensor":
        return (
            before.get("dtype") == after.get("dtype")
            and before.get("shape") == after.get("shape")
            and before.get("shape_max") == after.get("shape_max")
            and _same_payload(before["tensor"], after["tensor"])
        )
    if kind == "tensor_descriptor":
        return (
            before.get("shape") == after.get("shape")
            and before.get("strides") == after.get("strides")
            and before.get("block_shape") == after.get("block_shape")
            and before.get("padding") == after.get("padding")
            and _same_payload(before["base"], after["base"])
        )
    if kind in {"tuple", "list"}:
        return len(before.get("items", [])) == len(after.get("items", [])) and all(
            _same_payload(a, b) for a, b in zip(before["items"], after["items"])
        )
    if kind == "dict":
        return before.get("items", {}).keys() == after.get("items", {}).keys() and all(
            _same_payload(before["items"][k], after["items"][k]) for k in before["items"]
        )
    return before == after


def _elide_data(payload: dict[str, Any]) -> dict[str, Any]:
    kind = payload.get("kind")
    if kind == "tensor":
        out = dict(payload)
        out.pop("data", None)
        out["same_as_before"] = True
        return out
    if kind == "triton_tensor":
        out = dict(payload)
        out["tensor"] = _elide_data(payload["tensor"])
        out["same_as_before"] = True
        return out
    if kind == "tensor_descriptor":
        out = dict(payload)
        out["base"] = _elide_data(payload["base"])
        out["same_as_before"] = True
        return out
    if kind in {"tuple", "list"}:
        return {**payload, "items": [_elide_data(v) for v in payload["items"]], "same_as_before": True}
    if kind == "dict":
        return {**payload, "items": {k: _elide_data(v) for k, v in payload["items"].items()}, "same_as_before": True}
    return {**payload, "same_as_before": True}


def _after_arg_payload(value: Any, before_record: dict[str, Any], *, kernel_name: str, arg_index: int) -> dict[str, Any]:
    after = _arg_payload(value, phase="after", kernel_name=kernel_name, arg_index=arg_index)
    if _same_payload(before_record, after):
        return _elide_data(after)
    return after


def _matches(kernel_name: str, module_name: str) -> bool:
    text = f"{module_name}.{kernel_name}" if module_name else kernel_name
    return any(fnmatch.fnmatch(kernel_name, p) or fnmatch.fnmatch(text, p) for p in _patterns())


def _jit_source(self: Any) -> str | None:
    # Dynamic kernels from triton_kernels.specialize rewrite JITFunction._src
    # to the generated specialized signature while raw_src still points at the
    # unspecialized template.  Replay needs the launch-time signature, so check
    # _src/src before falling back to raw_src.
    for attr in ("src", "_src"):
        src = getattr(self, attr, None)
        if isinstance(src, str) and "def " in src:
            return src
    raw_src = getattr(self, "raw_src", None)
    if isinstance(raw_src, list) and raw_src:
        joined = "".join(str(x) for x in raw_src)
        if "def " in joined:
            return joined
    fn = getattr(self, "fn", None)
    if fn is None:
        return None
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return None


def _jit_provenance(self: Any, fn: Any) -> dict[str, Any]:
    return {
        "jit_type": _object_identity(self),
        "fn_module": getattr(fn, "__module__", ""),
        "fn_name": getattr(fn, "__name__", ""),
        "fn_qualname": getattr(fn, "__qualname__", ""),
        "repr": repr(self),
        "starting_line_number": getattr(self, "starting_line_number", None),
        "do_not_specialize": list(getattr(self, "do_not_specialize", []) or []),
        "do_not_specialize_on_alignment": list(
            getattr(self, "do_not_specialize_on_alignment", []) or []
        ),
        "has_raw_src": bool(getattr(self, "raw_src", None)),
    }


def install() -> None:
    """Install the Triton JIT capture hook when requested by env."""

    global _PATCHED
    fixture_dir = os.environ.get("GPT_OSS_KERNEL_FIXTURE_DIR", "").strip()
    if not fixture_dir:
        return
    if _PATCHED:
        return

    import torch
    from triton.runtime.jit import JITFunction

    out_dir = Path(fixture_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_run = JITFunction.run

    def _patched_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(self, "fn", None)
        kernel_name = getattr(fn, "__name__", getattr(self, "__name__", type(self).__name__))
        module_name = getattr(fn, "__module__", "")
        if not _matches(kernel_name, module_name):
            return original_run(self, *args, **kwargs)

        with _LOCK:
            seen = _COUNTS.get(kernel_name, 0)
            if seen >= _max_per_kernel():
                return original_run(self, *args, **kwargs)
            _COUNTS[kernel_name] = seen + 1
            ordinal = seen

        before = [
            _arg_payload(arg, phase="before", kernel_name=kernel_name, arg_index=i)
            for i, arg in enumerate(args)
        ]
        result = original_run(self, *args, **kwargs)
        torch.cuda.synchronize()
        after = [
            _after_arg_payload(arg, before[i], kernel_name=kernel_name, arg_index=i)
            for i, arg in enumerate(args)
        ]

        payload = {
            "schema_version": SCHEMA_VERSION,
            "pid": os.getpid(),
            "kernel_name": kernel_name,
            "module_name": module_name,
            "qualname": getattr(fn, "__qualname__", kernel_name),
            "jit_source": _jit_source(self),
            "jit_provenance": _jit_provenance(self, fn),
            "capture_env": {
                "TRITON_CORPUS_FORCE_TARGET": os.environ.get("TRITON_CORPUS_FORCE_TARGET", ""),
                "HSA_HOTSWAP_IR_RAISER": os.environ.get("HSA_HOTSWAP_IR_RAISER", ""),
                "HSA_SALMON_STRICT": os.environ.get("HSA_SALMON_STRICT", ""),
            },
            "grid": _jsonable(kwargs.get("grid")),
            "kwargs": {k: _jsonable(v) for k, v in kwargs.items() if k not in {"grid", "warmup"}},
            "warmup": _jsonable(kwargs.get("warmup")),
            "args_before": before,
            "args_after": after,
        }
        base = f"{_sanitize(module_name)}.{_sanitize(kernel_name)}.{os.getpid()}.{ordinal:03d}"
        tmp = out_dir / f".{base}.tmp.pt"
        final = out_dir / f"{base}.pt"
        torch.save(payload, tmp)
        os.replace(tmp, final)

        manifest = out_dir / "manifest.jsonl"
        event = {
            "event": "gpt_oss_kernel_fixture",
            "kernel_name": kernel_name,
            "module_name": module_name,
            "path": str(final),
            "pid": os.getpid(),
            "ordinal": ordinal,
        }
        with manifest.open("a") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
        return result

    JITFunction.run = _patched_run
    _PATCHED = True
