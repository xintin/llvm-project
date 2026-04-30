"""Replay a single gfx1250 Tensile kernel with the checked-in gold inputs
and compare its outputs to the gold reference.

Spawned by `verify_transpile.py` once per `(code object, kernel)` pair.
The parent has already arranged for the process to:

  - have `LD_PRELOAD=libsalmon_intercept.so` set (so the .co's ELF
    e_flags are rewritten to the target ISA on load),
  - have a Salmon-enabled `libhsa-runtime64.so.1` reachable via
    `LD_LIBRARY_PATH` (so the ROCR hotswap hook fires and rewrites the
    code object in IR or, in `native` mode, no transpile at all),
  - have `hsaco_runner` importable via `PYTHONPATH=<corpus-root>`,
  - have the `HSA_HOTSWAP_*` env knobs set for the requested mode.

This worker reads `<kernel>.inputs.npz` + `<kernel>.outputs.npz`, loads
the `.co` through HIP, reuses the gold kernarg layout but with freshly
allocated device pointers, uploads the captured input buffers, launches,
copies outputs back, and diffs against the captured gold outputs.

Emits exactly one JSON line on stdout on success *or* failure; the
parent parses it. Non-zero exit status means either a launch error or
at least one output that fell outside the configured tolerance.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import struct
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from hsaco_runner import hip_ctypes as hip
from hsaco_runner.amdhsa_meta import ValueKind, read_kernel_metadata


def _load_module_via_intercept(data: bytes) -> ctypes.c_void_p:
    """Load a HIP module via the LD_PRELOAD'd Salmon intercept shim.

    `hsaco_runner.hip.load_module` goes through
    `ctypes.CDLL("libamdhip64.so").hipModuleLoadData`, which resolves the
    symbol with a handle-scoped `dlsym` on libamdhip64 itself. That lookup
    does **not** consult LD_PRELOAD'd libraries, so the Salmon intercept
    shim's `hipModuleLoadData` wrapper is silently bypassed and HIP sees a
    raw gfx1250 code object on a gfx942 host (→ HIP error 209).

    Instead, resolve `hipModuleLoadData` via `CDLL(None)` (RTLD_DEFAULT),
    which searches the main process's global symbol namespace first —
    including LD_PRELOAD'd symbols. With `libsalmon_intercept.so` in
    LD_PRELOAD, this returns the shim's wrapper, which patches the ELF
    e_flags before forwarding to the real runtime.
    """
    main = ctypes.CDLL(None)
    fn = main.hipModuleLoadData
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    fn.restype = ctypes.c_int

    buf = (ctypes.c_char * len(data)).from_buffer_copy(data)
    mod = ctypes.c_void_p()
    rc = fn(ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p))
    if rc != 0:
        raise hip.HipError(rc, "hipModuleLoadData(via RTLD_DEFAULT)")
    # Keep the source buffer alive for the lifetime of the module
    # (HIP does not own it after load).
    mod._keepalive = buf  # type: ignore[attr-defined]
    return mod


# Default per-value-type tolerance table. Each entry is (atol, rtol).
# These are starting points, not load-bearing guarantees — they are
# deliberately generous compared to machine-epsilon because the "gold"
# side is an FFM simulator running gfx1250 code and the "test" side is
# real gfx942 hardware running the transpiled code, so small numeric
# differences from reordered reductions or differing MFMA dataflow are
# expected.
_DEFAULT_TOLERANCES: dict[str, list[float]] = {
    "default": [0.0, 0.0],
    "f64":     [1e-8, 1e-6],
    "f32":     [1e-4, 1e-3],
    "f16":     [5e-3, 1e-2],
    "bf16":    [1e-3, 1e-2],
}


def _as_float_view(arr: np.ndarray, value_type: str) -> np.ndarray | None:
    """Return `arr` reinterpreted as a float array for tolerance compare,
    or None if the value_type is not a floating-point type this tool knows
    how to compare with a relative tolerance."""
    if value_type == "bf16":
        # Stored as uint16 (2-byte container). Widen to float32 by
        # shifting the bf16 bit pattern into the high half of a float32.
        u32 = arr.astype(np.uint32, copy=False)
        return (u32 << 16).view(np.float32).astype(np.float64, copy=False)
    if value_type in ("f16", "f32", "f64"):
        return arr.astype(np.float64, copy=False)
    return None


def _compare(got: np.ndarray, want: np.ndarray, value_type: str,
             atol: float, rtol: float) -> dict[str, Any]:
    """Diff one output buffer against its gold. Returns a dict with `ok`,
    counts, and magnitudes of the mismatch."""
    if got.shape != want.shape:
        return {
            "ok": False,
            "error": "shape mismatch",
            "got_shape": list(got.shape),
            "want_shape": list(want.shape),
        }
    if got.dtype != want.dtype:
        return {
            "ok": False,
            "error": "dtype mismatch",
            "got_dtype": str(got.dtype),
            "want_dtype": str(want.dtype),
        }

    n = int(got.size)
    gf = _as_float_view(got, value_type)
    wf = _as_float_view(want, value_type)

    if gf is not None and wf is not None:
        # Tolerance compare in float64, matched NaN/Inf tolerated position-wise.
        got_nan = np.isnan(gf)
        want_nan = np.isnan(wf)
        got_inf = np.isinf(gf)
        want_inf = np.isinf(wf)

        both_nan = got_nan & want_nan
        both_inf = got_inf & want_inf & (np.sign(gf) == np.sign(wf))

        diff = np.abs(gf - wf)
        tol = atol + rtol * np.abs(wf)
        pointwise_ok = (diff <= tol) | both_nan | both_inf

        n_mismatch = int((~pointwise_ok).sum())

        # For max_abs / max_rel exclude the both-NaN positions (diff is NaN there).
        finite_both = ~(got_nan | want_nan | got_inf | want_inf)
        if finite_both.any():
            max_abs = float(diff[finite_both].max())
            denom = np.maximum(np.abs(wf[finite_both]), 1e-300)
            max_rel = float((diff[finite_both] / denom).max())
        else:
            max_abs = 0.0
            max_rel = 0.0

        # Find the first offending linear index, if any.
        first_diff_idx = -1
        first_got = 0.0
        first_want = 0.0
        if n_mismatch:
            first_diff_idx = int(np.flatnonzero(~pointwise_ok.ravel())[0])
            first_got = float(gf.ravel()[first_diff_idx])
            first_want = float(wf.ravel()[first_diff_idx])

        return {
            "ok": n_mismatch == 0,
            "n": n,
            "mismatches": n_mismatch,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "got_nan": int(got_nan.sum()),
            "got_inf": int(got_inf.sum()),
            "want_nan": int(want_nan.sum()),
            "want_inf": int(want_inf.sum()),
            "first_diff_idx": first_diff_idx,
            "first_got": first_got,
            "first_want": first_want,
            "atol": atol,
            "rtol": rtol,
        }

    # Integer / raw-byte container (fp8/fp4/fp6/...). Bit-exact compare.
    mismatches_mask = (got != want)
    n_mismatch = int(mismatches_mask.sum())
    first_diff_idx = -1
    first_got = 0
    first_want = 0
    if n_mismatch:
        first_diff_idx = int(np.flatnonzero(mismatches_mask.ravel())[0])
        first_got = int(got.ravel()[first_diff_idx])
        first_want = int(want.ravel()[first_diff_idx])
    return {
        "ok": n_mismatch == 0,
        "n": n,
        "mismatches": n_mismatch,
        "exact": True,
        "first_diff_idx": first_diff_idx,
        "first_got": first_got,
        "first_want": first_want,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--co", required=True, type=Path)
    p.add_argument("--inputs-npz", required=True, type=Path)
    p.add_argument("--outputs-npz", required=True, type=Path)
    p.add_argument(
        "--tolerances",
        default=json.dumps(_DEFAULT_TOLERANCES),
        help=("JSON mapping AMDHSA value_type → [atol, rtol]; the special "
              "key `default` is used for any type not listed. Defaults "
              f"are {_DEFAULT_TOLERANCES}."),
    )
    args = p.parse_args()

    tolerances_raw = json.loads(args.tolerances)
    if "default" not in tolerances_raw:
        raise SystemExit("tolerances JSON must contain a 'default' key")

    result: dict[str, Any] = {
        "co": args.co.name,
        "inputs_npz": args.inputs_npz.name,
        "kernel": None,
        "module_loaded": False,
        "launched": False,
        "all_outputs_ok": False,
        "error": None,
        "outputs": {},
    }

    try:
        inputs = np.load(args.inputs_npz, allow_pickle=True)
        outputs = np.load(args.outputs_npz, allow_pickle=True)
        input_meta = json.loads(str(inputs["_metadata_json"]))
        output_meta = json.loads(str(outputs["_metadata_json"]))
        kernarg_gold = bytes(inputs["_kernarg_bytes"].tobytes())

        kernel_name = input_meta["name"]
        result["kernel"] = kernel_name
        result["gold_smoke_ok"] = bool(output_meta.get("smoke_ok", False))
        result["gold_smoke_error"] = str(output_meta.get("smoke_error", ""))

        co_bytes = args.co.read_bytes()

        # The .co's own ELF metadata is the authority for kernarg_segment_size;
        # assert the checked-in gold kernarg buffer still matches.
        co_metas = read_kernel_metadata(co_bytes)
        meta = next((m for m in co_metas if m.name == kernel_name), None)
        if meta is None:
            raise RuntimeError(f"kernel {kernel_name!r} not found in .co "
                               f"(kernels present: "
                               f"{[m.name for m in co_metas]})")
        if len(kernarg_gold) != meta.kernarg_segment_size:
            raise RuntimeError(
                f"gold _kernarg_bytes size ({len(kernarg_gold)}) != "
                f"kernel .kernarg_segment_size "
                f"({meta.kernarg_segment_size})"
            )

        grid = tuple(int(x) for x in input_meta["grid"])
        block = tuple(int(x) for x in input_meta["block"])
        shmem = int(input_meta["shared_mem_bytes"])

        with hip.HipContext(device=0) as ctx:
            # Route the module load through the LD_PRELOAD'd Salmon
            # intercept shim (see `_load_module_via_intercept` for why
            # this cannot just be `ctx.load_module`). Still append to
            # ctx's module list so cleanup on exit unloads it.
            mod = _load_module_via_intercept(co_bytes)
            ctx._modules.append(mod)
            result["module_loaded"] = True
            fn = hip.get_function(mod, kernel_name)

            kernarg = bytearray(kernarg_gold)

            # Every global_buffer arg needs a fresh device allocation
            # whose pointer we splice into the kernarg at the recorded
            # offset (overwriting whatever stale gold-capture pointer was
            # packed there). Kernarg scalars (sizes/strides/alpha/beta/
            # Gemm info/numWG/...) stay bit-identical to the gold.
            live: list[tuple[dict, hip.HipArray, bool]] = []
            for arg_info in sorted(input_meta["args"], key=lambda a: a["offset"]):
                if arg_info["value_kind"] != "global_buffer":
                    continue

                name = arg_info["name"]
                offset = int(arg_info["offset"])
                arg_size = int(arg_info["size"])
                direction = arg_info["direction"]  # "in" / "out" / "inout"
                npz_key = f"arg__{name}"
                has_input = npz_key in inputs.files
                has_output = npz_key in outputs.files

                if has_input:
                    host_buf = np.array(inputs[npz_key], copy=True)
                elif has_output:
                    # OUT-only buffer. Match the gold output's size + dtype
                    # so the D2H readback tensor has a known shape; seed
                    # with zeros so a kernel that only partially writes
                    # the buffer still has a deterministic starting state.
                    host_buf = np.zeros_like(outputs[npz_key])
                else:
                    raise RuntimeError(
                        f"arg {name!r} (direction={direction}) is not "
                        f"present in either inputs or outputs .npz — "
                        f"gold corpus is inconsistent"
                    )

                hip_arr = ctx.array(host_buf)
                live.append((arg_info, hip_arr, has_output))

                # Pointer slot in the kernarg buffer. Tensile kernels
                # always use 8-byte (global_buffer) pointer slots but
                # guard against an oddly-sized arg anyway.
                if arg_size == 8:
                    struct.pack_into("<Q", kernarg, offset, int(hip_arr.ptr))
                elif arg_size == 4:
                    struct.pack_into(
                        "<I", kernarg, offset,
                        int(hip_arr.ptr) & 0xFFFFFFFF,
                    )
                else:
                    raise RuntimeError(
                        f"unexpected global_buffer arg size {arg_size} "
                        f"for {name!r}"
                    )

            hip.launch(fn, grid, block, shmem, bytes(kernarg))
            hip.stream_synchronize(0)
            result["launched"] = True

            # D2H + compare for every kernarg that the gold side wrote.
            any_mismatch = False
            for arg_info, hip_arr, has_output in live:
                if not has_output:
                    continue
                name = arg_info["name"]
                value_type = arg_info["value_type"]
                hip_arr.copy_device_to_host()
                got = hip_arr.host_array.copy()
                want = outputs[f"arg__{name}"]

                atol, rtol = tolerances_raw.get(
                    value_type, tolerances_raw["default"]
                )
                cmp = _compare(got, want, value_type, float(atol), float(rtol))
                cmp["value_type"] = value_type
                cmp["direction"] = arg_info["direction"]
                result["outputs"][name] = cmp
                if not cmp.get("ok", False):
                    any_mismatch = True

            result["all_outputs_ok"] = not any_mismatch

    except hip.HipError as exc:
        result["error"] = f"HipError: {exc}"
        traceback.print_exc(file=sys.stderr)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)

    print(json.dumps(result), flush=True)
    return 0 if result["all_outputs_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
