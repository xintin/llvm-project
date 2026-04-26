# `tensile_gold_verify`

End-to-end **numerical** verification of the gfx1250 → gfx942 transpile
path against the per-kernel input/output dumps captured on the FFM
hardware simulator for every kernel in the Tensile gfx1250 corpus.

## What this answers that other tools do not

`smoke_test_compare_transpilers` answers "does the code object load?"
`compare_correctness/` answers "does a hand-authored or Triton kernel
produce the right result?" — its gold comes from a CPU reference or
from a same-source gfx942 build.

This tool answers the hardest question in the stack: **when the
transpiler takes a real, in-the-wild gfx1250 Tensile kernel and
translates it to gfx942, does the translated kernel compute the same
answer as the gfx1250 simulator did on identical inputs?**

The gold inputs / outputs come from the `users/tgymnich/hsaco-runner`
branch of `rocm-hotswap-testing` (146 TensileLite `.co` files, one
kernel each, captured with the FFM simulator via the `hsaco_runner`
package). The path under test is real gfx942 hardware, with the Salmon
LD_PRELOAD shim + Salmon-enabled ROCR rewriting each code object on
`hipModuleLoadData`.

## Prerequisites

1. gfx942 hardware.
2. A Salmon-enabled ROCR build tree (`ROCR_ENABLE_IR_RAISER=ON`),
   typically at `$HOME/rocm-systems/projects/rocr-runtime/build`.
   Verify with:
   ```bash
   nm -D --defined-only $ROCR_BUILD/rocr/lib/libhsa-runtime64.so.1 \
     | grep rocr_salmon_patch_elf
   ```
3. `libsalmon_intercept.so` — built as a side effect of building the
   sibling `compare_correctness/` tool:
   ```bash
   cd ../compare_correctness
   make ROCR_BUILD=$ROCR_BUILD libsalmon_intercept.so
   ```
4. A checkout of the gold corpus. This tree does not ship it; clone it
   alongside your rocm-systems checkout, for example:
   ```bash
   git clone --depth 1 --branch users/tgymnich/hsaco-runner \
     https://github.com/harsh-amd/rocm-hotswap-testing.git \
     $HOME/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/hotswap/test_data/hsaco-runner-gold
   ```
5. Python 3.8+ with `numpy` and `msgpack` installed. The corpus's
   `hsaco_runner/requirements.txt` lists them.

## Run

```bash
python3 verify_transpile.py \
  --corpus $HOME/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/hotswap/test_data/hsaco-runner-gold \
  --rocr-build $HOME/rocm-systems/projects/rocr-runtime/build
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--mode {salmon,legacy,native}` | pick the transpile engine (default `salmon`) |
| `--filter REGEX` | only run kernels whose `.name` matches |
| `--limit N` | stop after `N` kernels |
| `--timeout S` | per-kernel wall-clock timeout (default 60s) |
| `--include-failed` | also run kernels the gfx1250 sim smoke-failed (gold is `NaN`/all-zero there) |
| `--tolerances-json '{"default":[0,0],"f32":[1e-4,1e-3],...}'` | override per-dtype (atol, rtol) |
| `--report-json PATH` | dump the full machine-readable report |
| `--verbose` | print each kernel's verdict as it lands |

## How a child invocation works

For every runnable `(.co, kernel)` pair the parent spawns
`_child_worker.py` in its own process group (so a hung launch can be
`SIGKILL`-ed cleanly). The child:

1. `np.load`s the matching `.inputs.npz` and `.outputs.npz`.
2. `hipModuleLoadData` on the `.co`. The `libsalmon_intercept.so`
   preload patches the ELF `e_flags` so HIP accepts it, the
   Salmon-enabled ROCR then transpiles through the IR raiser (salmon)
   or byte-level rewriter (legacy).
3. Starts from the gold `_kernarg_bytes` buffer (every packed
   scalar / stride / size / numWG / alpha / beta is bit-identical to
   the gfx1250 sim run) and splices in freshly `hipMalloc`-ed
   pointers for every `global_buffer` arg at the recorded offset.
4. For inputs and inouts, H2D-uploads the stored `arg__<name>`
   numpy array exactly as it was uploaded on the sim side.
5. `hipModuleLaunchKernel`, `hipStreamSynchronize`.
6. D2H each output, diff against `arg__<name>` in the gold
   `.outputs.npz` with the configured (atol, rtol) for its dtype.

`bf16` is compared by widening both sides to `float32` via
`(uint16 << 16).view(float32)`; `fp8`/`fp4`/`fp6` container bytes are
compared bit-exact (numpy has no native float type to reinterpret them
to). Integer types are compared exactly.

## Status classes in the report

- **match** — every output buffer within tolerance of the gfx1250 sim gold.
- **mismatch** — at least one output differs beyond tolerance.
- **launch-error** — HIP or driver failure during dispatch.
- **crash** — child died by signal (SIGSEGV, etc.) before emitting a verdict.
- **timeout** — child exceeded `--timeout`; parent sent SIGKILL.
- **skipped** — gold itself smoke-failed on the sim (NaN or all-zero
  output); nothing to verify against. Opt-in via `--include-failed`
  if you want to see what the transpiled path does anyway.

## Exit code

- `0` if every runnable kernel's verdict is `match`.
- `1` if any runnable kernel is `mismatch` / `launch-error` / `crash` /
  `timeout`. Skipped kernels are not failures.
