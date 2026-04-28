#!/usr/bin/env python3
"""Replay exact GPT-OSS Triton JIT fixtures under native and Salmon.

Fixtures are produced by setting ``GPT_OSS_KERNEL_FIXTURE_DIR`` while running
the SGLang GPT-OSS harness.  Each fixture records the real JIT function, grid,
constexpr kwargs, and tensor argument bytes from the production run.  This tool
replays each fixture twice:

* native: physical gfx942 target, no Salmon preload;
* salmon: forced gfx1250 target with the Salmon hotswap preload.

The final state of every tensor argument is compared against native.  That
keeps the replay generic: kernels that write in-place outputs, caches, metadata
buffers, or multiple result tensors all get checked without a handwritten
per-kernel output list.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[7]
TRITON_CORPUS_RUNNER = HERE.parent / "triton_corpus_runner"
DEFAULT_LIBHSA = REPO / "projects/rocr-runtime/build/rocr/lib/libhsa-runtime64.so.1"
DEFAULT_LIBSALMON = HERE / "libsalmon_intercept.so"
DEFAULT_RULES = TRITON_CORPUS_RUNNER / "_empty_rules.json"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return {"type": type(value).__name__, "repr": repr(value)}


def _load_fixture(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


class ReplayUnsupported(RuntimeError):
    """Fixture cannot currently be reconstructed by the replay tool."""


def _restore_tensor(record: dict[str, Any]) -> Any:
    import torch

    logical = record["data"].to("cuda")
    shape = list(record.get("shape", list(logical.shape)))
    stride = list(record.get("stride", list(logical.stride())))
    if shape == list(logical.shape) and stride == list(logical.stride()):
        return logical

    if not shape:
        return logical.reshape([])
    max_offset = 0
    for dim, st in zip(shape, stride):
        if dim > 0:
            max_offset += (int(dim) - 1) * int(st)
    storage_elems = max_offset + 1
    backing = torch.empty(storage_elems, dtype=logical.dtype, device="cuda")
    view = torch.as_strided(backing, size=shape, stride=stride)
    try:
        view.copy_(logical)
    except RuntimeError as err:
        if "more than one element of the written-to tensor refers to a single memory location" not in str(err):
            raise
        # Broadcast/overlapping views (commonly stride 0) cannot be filled via
        # copy_.  For ordinary pointer arguments the kernel receives explicit
        # stride scalars separately, so a contiguous logical tensor preserves
        # the values while avoiding an invalid overlapping write during replay.
        return logical.contiguous()
    return view


def _restore_value(record: dict[str, Any]) -> Any:
    import torch

    if record["kind"] == "tensor":
        return _restore_tensor(record)
    if record["kind"] == "tensor_descriptor":
        from triton.tools.tensor_descriptor import TensorDescriptor

        descriptor_type = record.get("descriptor_type") or {}
        if descriptor_type and descriptor_type.get("module") != "triton.tools.tensor_descriptor":
            raise ReplayUnsupported(
                "unsupported tensor descriptor type "
                f"{descriptor_type.get('module')}.{descriptor_type.get('type')}"
            )
        return TensorDescriptor(
            _restore_tensor(record["base"]),
            list(record["shape"]),
            list(record["strides"]),
            list(record["block_shape"]),
            record.get("padding", "zero"),
        )
    if record["kind"] == "triton_tensor":
        from triton_kernels.tensor import Tensor
        from triton_kernels.tensor_details.layout import StridedLayout

        dtype_record = record["dtype"]
        if dtype_record.get("kind") != "torch_dtype":
            raise ReplayUnsupported(
                "replay currently supports triton_kernels Tensor wrappers "
                f"with torch.dtype only, got {dtype_record}"
            )
        dtype_name = dtype_record["name"].removeprefix("torch.")
        dtype = getattr(torch, dtype_name)
        tensor = _restore_tensor(record["tensor"])
        layout_record = record.get("layout") or {}
        if layout_record.get("type") != "StridedLayout":
            raise ReplayUnsupported(f"unsupported triton_kernels Tensor layout: {layout_record}")
        # Tensor(torch.Tensor, ...) constructs a Storage with StridedLayout.
        # Replacing the layout explicitly preserves the recorded max shape.
        wrapped = Tensor(
            tensor,
            dtype=dtype,
            shape=list(record["shape"]),
            shape_max=list(record["shape_max"]),
        )
        wrapped.storage.layout = StridedLayout(layout_record.get("shape", list(tensor.shape)))
        return wrapped
    if record["kind"] == "tuple":
        return tuple(_restore_value(v) for v in record["items"])
    if record["kind"] == "list":
        return [_restore_value(v) for v in record["items"]]
    if record["kind"] == "dict":
        return {k: _restore_value(v) for k, v in record["items"].items()}
    if record["kind"] == "value":
        return record["value"]
    raise ReplayUnsupported(f"unknown fixture arg kind {record.get('kind')!r}")


def _load_kernel(module_name: str, qualname: str, kernel_name: str) -> Any:
    module = importlib.import_module(module_name)
    obj: Any = module
    # Prefer qualname when possible; fall back to kernel_name for plain module
    # globals.  Nested local functions are not expected in Triton JIT fixtures.
    path = qualname.split(".") if qualname and "<locals>" not in qualname else [kernel_name]
    for part in path:
        if not part:
            continue
        obj = getattr(obj, part)
    return obj


def _dynamic_kernel_globals(kernel_name: str) -> dict[str, Any]:
    import triton
    import triton.language as tl

    globals_dict: dict[str, Any] = {"triton": triton, "tl": tl}
    if kernel_name == "_matmul_ogs":
        mod = importlib.import_module("triton_kernels.matmul_ogs_details._matmul_ogs")
        globals_dict.update(mod.__dict__)
        try:
            swiglu = importlib.import_module("triton_kernels.swiglu_details._swiglu")
            globals_dict.update({"_swiglu_fn": swiglu._swiglu_fn})
        except Exception:
            pass
    elif kernel_name == "_reduce":
        mod = importlib.import_module("triton_kernels.reduce")
        globals_dict.update(mod.__dict__)
    else:
        # Broad but explicit fallback: the source can still refer only to
        # globals imported here or defined inside the saved source.
        globals_dict.update(importlib.import_module("triton").__dict__)
    return globals_dict


def _load_kernel_from_fixture(data: dict[str, Any]) -> Any:
    module_name = data["module_name"]
    kernel_name = data["kernel_name"]
    try:
        return _load_kernel(module_name, data.get("qualname", ""), kernel_name)
    except ModuleNotFoundError:
        src = data.get("jit_source")
        if not src:
            raise ReplayUnsupported(
                f"cannot import dynamic module {module_name!r} and fixture has no jit_source"
            )
        import triton
        from triton_kernels.specialize import define_kernel

        g = _dynamic_kernel_globals(kernel_name)
        provenance = data.get("jit_provenance") or {}
        jit_kwargs: dict[str, Any] = {}
        do_not_specialize = tuple(provenance.get("do_not_specialize") or ())
        if do_not_specialize:
            jit_kwargs["do_not_specialize"] = do_not_specialize
        do_not_specialize_on_alignment = tuple(
            provenance.get("do_not_specialize_on_alignment") or ()
        )
        if do_not_specialize_on_alignment:
            jit_kwargs["do_not_specialize_on_alignment"] = do_not_specialize_on_alignment
        module = types.SimpleNamespace(__name__=module_name)
        try:
            return define_kernel(src, module, attrs=jit_kwargs, **g)
        except Exception as err:
            # Fall back to direct exec for source that already creates a
            # JITFunction object in an importable-style environment.
            direct_err = err
        stored: list[Any] = []
        g["__stored_fixture_functions"] = stored
        src = textwrap.dedent(src)
        if src.lstrip().startswith("@"):
            exec(src + f"\n__stored_fixture_functions.append({kernel_name})\n", g)
            fn = stored[-1]
            if isinstance(fn, triton.runtime.jit.JITFunction):
                return fn
            return triton.jit(fn, **jit_kwargs)
        def_idx = src.find("def ")
        if def_idx < 0:
            raise ReplayUnsupported(f"fixture source for {module_name}.{kernel_name} has no def")
        src = src[def_idx:]
        exec(src + f"\n__stored_fixture_functions.append({kernel_name})\n", g)
        try:
            return triton.jit(stored[-1], **jit_kwargs)
        except Exception as err:
            raise ReplayUnsupported(
                f"failed to reconstruct dynamic kernel {module_name}.{kernel_name}: "
                f"define_kernel={type(direct_err).__name__}: {direct_err}; "
                f"direct={type(err).__name__}: {err}"
            ) from err


def _as_grid(value: Any) -> tuple[int, int, int] | tuple[int] | Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _run_child(fixture: Path, out_path: Path, num_stages: int | None = None) -> int:
    import torch

    data = _load_fixture(fixture)
    schema = int(data.get("schema_version", 1))
    if schema > 2:
        raise ReplayUnsupported(
            f"fixture schema_version={schema} is newer than this replay tool supports"
        )
    fn = _load_kernel_from_fixture(data)
    args = [_restore_value(r) for r in data["args_before"]]
    kwargs = dict(data.get("kwargs") or {})
    if num_stages is not None:
        kwargs["num_stages"] = num_stages
    grid = _as_grid(data["grid"])
    if grid is None:
        raise RuntimeError(f"{fixture}: fixture missing launch grid")

    fn[grid](*args, **kwargs)
    torch.cuda.synchronize()

    def collect(obj: Any, path: str, out: list[dict[str, Any]]) -> None:
        if isinstance(obj, torch.Tensor):
            out.append(
                {
                    "path": path,
                    "dtype": str(obj.dtype),
                    "shape": list(obj.shape),
                    "data": obj.detach().cpu().contiguous(),
                }
            )
            return
        if type(obj).__module__ == "triton_kernels.tensor" and type(obj).__name__ == "Tensor":
            collect(obj.storage.data, path + ".storage.data", out)
            return
        if (
            type(obj).__module__.startswith("triton.")
            and type(obj).__name__ == "TensorDescriptor"
            and hasattr(obj, "base")
        ):
            collect(obj.base, path + ".base", out)
            return
        if isinstance(obj, (tuple, list)):
            for i, item in enumerate(obj):
                collect(item, f"{path}[{i}]", out)
            return
        if isinstance(obj, dict):
            for k, item in obj.items():
                collect(item, f"{path}[{k!r}]", out)

    final = []
    for i, arg in enumerate(args):
        collect(arg, f"arg{i}", final)
    torch.save({"fixture": str(fixture), "final_tensors": final}, out_path)
    return 0


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    pp = [
        str(TRITON_CORPUS_RUNNER),
        str(HERE / "kernels/triton/_corpus/extracted"),
        "/home/mluecke/sglang-gpt-oss/python",
        "/home/nithin/triton/python/triton_kernels",
        env.get("GPT_OSS_FIXTURE_EXTRA_PYTHONPATH", ""),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join(p for p in pp if p)
    env["HIP_VISIBLE_DEVICES"] = env.get("HIP_VISIBLE_DEVICES", "0")
    env.pop("GPT_OSS_KERNEL_FIXTURE_DIR", None)
    return env


def _mode_env(mode: str) -> dict[str, str]:
    env = _base_env()
    if mode == "native":
        for key in (
            "TRITON_CORPUS_FORCE_TARGET",
            "HSA_HOTSWAP_ISA_OVERRIDE",
            "HSA_HOTSWAP_RULES",
            "HSA_HOTSWAP_IR_RAISER",
            "HSA_SALMON_STRICT",
            "LD_PRELOAD",
        ):
            env.pop(key, None)
        return env
    if mode != "salmon":
        raise RuntimeError(f"unknown replay mode {mode!r}")
    env["TRITON_CORPUS_FORCE_TARGET"] = "gfx1250:32"
    env["HSA_HOTSWAP_ISA_OVERRIDE"] = "gfx942"
    env["HSA_HOTSWAP_RULES"] = str(DEFAULT_RULES)
    env["HSA_HOTSWAP_IR_RAISER"] = "1"
    env["HSA_SALMON_STRICT"] = "1"
    env["LD_PRELOAD"] = f"{DEFAULT_LIBHSA}:{DEFAULT_LIBSALMON}"
    env["LD_LIBRARY_PATH"] = f"{DEFAULT_LIBHSA.parent}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def _spawn(
    mode: str,
    fixture: Path,
    out_path: Path,
    timeout_s: int,
    *,
    num_stages: int | None = None,
) -> tuple[int, str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--fixture",
        str(fixture),
        "--out",
        str(out_path),
    ]
    if num_stages is not None:
        cmd += ["--num-stages", str(num_stages)]
    proc = subprocess.run(
        cmd,
        env=_mode_env(mode),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout


_OUT_OF_RESOURCES_RE = re.compile(
    r"OutOfResources: out of resource: (?P<resource>[^,]+), "
    r"Required: (?P<required>\d+), Hardware limit: (?P<limit>\d+)"
)
_FORCED_TARGET_RE = re.compile(
    r"forced Triton target = GPUTarget\(backend='(?P<backend>[^']+)', "
    r"arch='(?P<arch>[^']+)', warp_size=(?P<warp_size>\d+)\)"
)
_SALMON_TARGET_RE = re.compile(
    r"salmon_intercept: active, target=(?P<arch>\S+) "
    r"\(wave_size=(?P<wave_size>\d+),"
)


def _resource_limit_failure(mode: str, log: str) -> dict[str, Any] | None:
    match = _OUT_OF_RESOURCES_RE.search(log)
    if not match:
        return None
    result: dict[str, Any] = {
        "status": "target-resource-limit",
        "mode": mode,
        "resource": match.group("resource"),
        "required_bytes": int(match.group("required")),
        "hardware_limit_bytes": int(match.group("limit")),
    }
    forced = _FORCED_TARGET_RE.search(log)
    if forced:
        result["generated_target"] = {
            "backend": forced.group("backend"),
            "arch": forced.group("arch"),
            "warp_size": int(forced.group("warp_size")),
        }
    salmon_target = _SALMON_TARGET_RE.search(log)
    if salmon_target:
        result["execution_target"] = {
            "arch": salmon_target.group("arch"),
            "wave_size": int(salmon_target.group("wave_size")),
        }
    return result


def _classify_child_failure(mode: str, log: str) -> str:
    if _resource_limit_failure(mode, log):
        return "target-resource-limit"
    if "ReplayUnsupported" in log:
        return "replay-unsupported"
    if "ModuleNotFoundError" in log or "failed to specialize argument of type" in log:
        return "replay-unsupported"
    if "salmon: unsupported instruction:" in log or "Unsupported instruction:" in log:
        return "salmon-translate-fail" if mode == "salmon" else "native-runtime-fail"
    if "PassManager::run failed" in log or "CompilationError:" in log:
        return "salmon-translate-fail" if mode == "salmon" else "native-runtime-fail"
    if "Triton Error [HIP]" in log and "no kernel image is available" in log:
        return "salmon-translate-fail" if mode == "salmon" else "native-runtime-fail"
    if "HIP error" in log or "AcceleratorError" in log or "illegal memory access" in log:
        return "salmon-runtime-fail" if mode == "salmon" else "native-runtime-fail"
    return "salmon-runtime-fail" if mode == "salmon" else "native-runtime-fail"


def _compare_tensors(native_path: Path, salmon_path: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    import torch

    native = torch.load(native_path, map_location="cpu", weights_only=False)["final_tensors"]
    salmon = torch.load(salmon_path, map_location="cpu", weights_only=False)["final_tensors"]
    if len(native) != len(salmon):
        return {"status": "wrong-result", "reason": f"tensor-count mismatch {len(native)} != {len(salmon)}"}

    worst = {"max_abs": 0.0, "mean_abs": 0.0, "mismatches": 0, "tensor_path": None}
    for n, s in zip(native, salmon):
        if n["path"] != s["path"] or n["dtype"] != s["dtype"] or n["shape"] != s["shape"]:
            return {
                "status": "wrong-result",
                "reason": "tensor metadata mismatch",
                "native": {k: n[k] for k in ("path", "dtype", "shape")},
                "salmon": {k: s[k] for k in ("path", "dtype", "shape")},
            }
        a = n["data"]
        b = s["data"]
        if a.dtype.is_floating_point:
            af = a.to(torch.float32)
            bf = b.to(torch.float32)
            both_nan = torch.isnan(af) & torch.isnan(bf)
            same_inf = torch.isinf(af) & torch.isinf(bf) & (torch.signbit(af) == torch.signbit(bf))
            comparable = ~(both_nan | same_inf)
            diff = torch.zeros_like(af)
            diff[comparable] = (af[comparable] - bf[comparable]).abs()
            scale = torch.maximum(torch.ones_like(af), af.abs())
            bad_class = (torch.isnan(af) != torch.isnan(bf)) | (torch.isinf(af) != torch.isinf(bf))
            bad_value = comparable & (diff > (atol + rtol * scale))
            mismatches = int((bad_class | bad_value).sum().item())
            finite_diff = diff[torch.isfinite(diff)]
            max_abs = float(finite_diff.max().item()) if finite_diff.numel() else 0.0
            mean_abs = float(finite_diff.mean().item()) if finite_diff.numel() else 0.0
        else:
            ne = a != b
            mismatches = int(ne.sum().item())
            max_abs = 1.0 if mismatches else 0.0
            mean_abs = float(mismatches) / float(a.numel() or 1)
        if mismatches or max_abs > worst["max_abs"]:
            worst = {
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "mismatches": mismatches,
                "tensor_path": n["path"],
            }
        if mismatches:
            return {"status": "wrong-result", **worst}
    return {"status": "pass", **worst}


def _iter_fixtures(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        fixtures = [path]
    else:
        fixtures = sorted(path.glob("*.pt"))
    if pattern:
        fixtures = [p for p in fixtures if fnmatch.fnmatch(p.name, pattern)]
    return fixtures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixtures", nargs="?", default="", help="fixture file or directory")
    ap.add_argument("--pattern", default="", help="fnmatch filter on fixture filename")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--json", default="", help="optional JSONL result path")
    ap.add_argument(
        "--native-num-stages",
        type=int,
        help="override Triton num_stages for native replay",
    )
    ap.add_argument(
        "--salmon-num-stages",
        type=int,
        help="override Triton num_stages for forced-target Salmon replay",
    )
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--fixture", default="", help=argparse.SUPPRESS)
    ap.add_argument("--out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--num-stages", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        if not args.fixture or not args.out:
            raise SystemExit("--child requires --fixture and --out")
        return _run_child(Path(args.fixture), Path(args.out), args.num_stages)

    if not args.fixtures:
        raise SystemExit("fixture file or directory is required")
    fixtures = _iter_fixtures(Path(args.fixtures), args.pattern)
    if not fixtures:
        raise SystemExit(f"no fixtures found under {args.fixtures!r}")

    out_json = Path(args.json) if args.json else None
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text("")

    failures = 0
    with tempfile.TemporaryDirectory(prefix="gpt_oss_fixture_replay_") as td:
        tmpdir = Path(td)
        for fixture in fixtures:
            data = _load_fixture(fixture)
            kernel = f"{data.get('module_name', '')}.{data.get('kernel_name', fixture.stem)}"
            native_out = tmpdir / f"{fixture.stem}.native.pt"
            salmon_out = tmpdir / f"{fixture.stem}.salmon.pt"
            n_rc, n_log = _spawn(
                "native",
                fixture,
                native_out,
                args.timeout,
                num_stages=args.native_num_stages,
            )
            if n_rc != 0:
                result = {
                    "fixture": str(fixture),
                    "kernel": kernel,
                    "native_num_stages": args.native_num_stages,
                    "salmon_num_stages": args.salmon_num_stages,
                    "log": n_log[-4000:],
                }
                result.update(
                    _resource_limit_failure("native", n_log)
                    or {"status": _classify_child_failure("native", n_log)}
                )
                failures += 1
                print(json.dumps(_jsonable(result), sort_keys=True))
                if out_json:
                    with out_json.open("a") as f:
                        f.write(json.dumps(_jsonable(result), sort_keys=True) + "\n")
                continue

            s_rc, s_log = _spawn(
                "salmon",
                fixture,
                salmon_out,
                args.timeout,
                num_stages=args.salmon_num_stages,
            )
            if s_rc != 0:
                result = {
                    "fixture": str(fixture),
                    "kernel": kernel,
                    "native_num_stages": args.native_num_stages,
                    "salmon_num_stages": args.salmon_num_stages,
                    "log": s_log[-4000:],
                }
                result.update(
                    _resource_limit_failure("salmon", s_log)
                    or {"status": _classify_child_failure("salmon", s_log)}
                )
                failures += 1
            else:
                cmp_result = _compare_tensors(native_out, salmon_out, atol=args.atol, rtol=args.rtol)
                result = {
                    "fixture": str(fixture),
                    "kernel": kernel,
                    "native_num_stages": args.native_num_stages,
                    "salmon_num_stages": args.salmon_num_stages,
                    **cmp_result,
                }
                failures += int(cmp_result["status"] != "pass")
            print(json.dumps(_jsonable(result), sort_keys=True))
            if out_json:
                with out_json.open("a") as f:
                    f.write(json.dumps(_jsonable(result), sort_keys=True) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
