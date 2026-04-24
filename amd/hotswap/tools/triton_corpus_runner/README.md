# `triton_corpus_runner` — breadth-only Salmon coverage sweep

A prototype of the **"find crashes only"** path discussed as a cheaper
counterpart to `compare_correctness`.  It runs each script in a
corpus three times — once natively, once through the legacy
hotswap byte translator, once through Salmon — and records what
happened to each run (`PASS` / `FAIL` / `CRASH` / `HANG`).  No
per-launch numerical verdict, no per-kernel shim, no AOT compile:
the entry cost for a new script is "drop a `.py` into a directory."

The intended workflow is the dual of `compare_correctness`:

| tool                 | depth                     | breadth                                | new entry cost                   |
|----------------------|---------------------------|----------------------------------------|----------------------------------|
| `compare_correctness`| per-tensor numerical match| ~5 hand-curated kernels                | author a Python recipe + sidecar |
| `triton_corpus_runner`| exit-status + assertions  | every `.py` in a directory             | drop a file in                   |

You'd use this to mass-screen the upstream Triton tutorials,
PyTorch-inductor `torch.compile` repros, and other Triton-using
scripts and get a quick "salmon broke N of M" number; then promote
the most interesting failures to per-kernel `compare_correctness`
recipes for a real numerical post-mortem.

## What it does

`runner.py` discovers `.py` scripts (defaulting to the kernels
already pulled by `compare_correctness/kernels/triton/_corpus/pull.py`)
and, for each `(script, mode)` pair, spawns a child Python process
that:

  1. Imports `_bootstrap` (this module), which monkey-patches Triton
     to force the requested target ISA and stubs out
     `triton.testing.do_bench` / `Mark.run` so tutorial benchmark
     loops don't dominate the wall clock — we only care about
     whether the kernel launched and whether the script's own
     `torch.testing.assert_close` (or similar) passed.
  2. Runs the script via `runpy.run_path(..., run_name='__main__')`
     so its `if __name__ == "__main__":` block fires.

The parent waits up to `--timeout` seconds, SIGKILLs hung children,
classifies the exit status, and prints:

  - a per-script grid (one row per script, one column per mode);
  - a Failures section with the last ~12 lines of stderr per
    non-passing run;
  - a Summary section that highlights "scripts that PASS native but
    not legacy/salmon" — the actual transpilation-coverage signal.

## Modes

| mode    | what it loads                                                                                                                                                       |
|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| native  | nothing special — torch/Triton compile for the actual gfx942 device.  Baseline; tells us whether the script even runs on this hardware.                            |
| legacy  | `LD_PRELOAD` chain pulls in system HIP + Salmon-enabled libhsa + the intercept shim; Triton is monkey-patched to compile for gfx1250 wave32; ROCR's hotswap byte translator turns the gfx1250 .co into gfx942 at load time. |
| salmon  | same as `legacy` plus `HSA_HOTSWAP_IR_RAISER=1`, so the hotswap hook routes through the Salmon IR raiser instead of the byte translator.                            |

## Verdicts

| verdict      | meaning                                                                 |
|--------------|-------------------------------------------------------------------------|
| `PASS`       | exit 0                                                                  |
| `FAIL`       | exit non-zero, normal termination (e.g. `assert_close` failure)         |
| `CRASH`      | terminated by signal (SIGSEGV / SIGABRT / SIGILL / ...)                 |
| `HANG`       | exceeded `--timeout`; child SIGKILLed                                   |
| `SPAWN_FAIL` | couldn't even start the child (missing python, missing libsalmon, ...) |

A `PASS` under native + non-`PASS` under legacy/salmon is a real
transpilation gap.  A non-`PASS` under native is a problem with the
script itself (or the local stack) and the summary calls those out
separately so they don't pollute the salmon-coverage signal.

## Usage

```
python3 runner.py                             # default: pulled tutorials
python3 runner.py --modes native              # quick baseline
python3 runner.py --script foo.py --modes salmon
python3 runner.py --scripts-dir ~/triton/python/tutorials --timeout 120
python3 runner.py --json out.json             # machine-readable record
```

Knobs of interest:

| flag               | default                                                                                | what                                                                       |
|--------------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| `--modes`          | `all`                                                                                  | comma-separated subset of `{native,legacy,salmon}`                         |
| `--timeout`        | `180` s                                                                                | per-child wall-clock; SIGKILL on overshoot                                 |
| `--triton-venv`    | `<HERE>/.venv-rocm7`                                                                   | self-contained rocm-7 venv built from the nightly torch wheel              |
| `--triton-pythonpath` | (empty)                                                                              | optional in-tree Triton tree to shadow the venv's bundled `triton-rocm`    |
| `--libsalmon`      | `../compare_correctness/libsalmon_intercept.so`                                        | the intercept shim built by `compare_correctness`                          |
| `--libhsa`         | `~/rocm-systems/projects/rocr-runtime/build/rocr/lib/libhsa-runtime64.so.1`            | Salmon-enabled libhsa                                                      |
| `--libamdhip`      | (empty)                                                                                | only set when the venv's torch ships HIP older than the Salmon ROCR        |
| `--no-stub-bench`  | off                                                                                    | run full `triton.testing.Mark.run` benchmark loops (slow)                  |
| `--gpu`            | `0`                                                                                    | `HIP_VISIBLE_DEVICES` for every child.  Pinning a single GPU localises any queue-reset blast radius on a shared box.  Pass `''` to clear it (give Triton/torch all visible devices) — only do this on a machine you own. |

## Setup — one-time, isolated to this directory

The default config expects a sibling `.venv-rocm7` next to `runner.py`,
populated from PyTorch's ROCm-7 nightly index.  Nothing about this
touches the system or the in-tree `~/rocm-systems/triton/.venv`:

```bash
cd .../triton_corpus_runner
python3.12 -m venv .venv-rocm7
.venv-rocm7/bin/pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.0
.venv-rocm7/bin/pip install numpy tabulate pytest    # tutorials need them
```

That gives you `torch 2.11.0.dev*+rocm7.0` with HIP `7.0.x` and the
matching `triton-rocm 3.6.0` wheel.  The HIP runtime in this wheel
is ABI-compatible with the Salmon ROCR's multi-ISA agent
enumeration, so no `--libamdhip` shadowing is needed.

`tabulate` is for tutorial 04's printing, `pytest` is for tutorial
06's `test_op` decorator (the script `import`s `pytest` even when
not actually invoking it under `pytest`).

You'll also want a freshly built `libsalmon_intercept.so`.  The
runner's default points at the one inside `compare_correctness/`,
which `make libsalmon_intercept.so` produces.

## Safety on a shared box

This tool **does not modify any system state** — no `/etc/ld.so.preload`,
no `/opt/rocm/*` writes, no system-Python installs.  Everything we
add lives under `$HOME` (the venv, the rebuilt ROCR, the intercept
shim).  `LD_PRELOAD` is per-process and only affects children we
spawn.  Other users' shells and processes never see any of it.

The one shared resource on a multi-tenant box is the GPU itself.
A buggy transpiled kernel that wedges an SQ/CU would force the
kernel-mode `amdgpu` driver to do a queue reset on the device it
ran on — affecting only users actively using that *same* device
index at that instant.  This risk class is the same as for any
GPU compute workload, not specific to the intercept.

We default `--gpu 0` so any blast radius is contained to one of
the box's GPUs.  Pass `--gpu N` to use a different device, or
`--gpu ''` to expose all of them (only do that on a machine you
own outright).

## Status on this machine (gfx942 + rocm-7.2.1 + the rocm-7 venv above)

The full sweep across the eight pulled upstream Triton tutorials
takes ~75 s and produces real triage:

| script                                  | native          | legacy            | salmon                      |
|-----------------------------------------|-----------------|-------------------|-----------------------------|
| `01-vector-add.py`                      | PASS            | CRASH (SIGABRT)   | CRASH (SIGSEGV post-launch) |
| `02-fused-softmax.py`                   | PASS            | CRASH (SIGABRT)   | FAIL (HIP 209 on load)      |
| `03-matrix-multiplication.py`           | PASS            | CRASH (SIGABRT)   | CRASH (SIGSEGV post-launch) |
| `04-low-memory-dropout.py`              | PASS            | CRASH (SIGABRT)   | FAIL (HIP 209 on load)      |
| `05-layer-norm.py`                      | PASS            | CRASH (SIGABRT)   | FAIL (HIP 209 on load)      |
| `06-fused-attention.py`                 | PASS\*          | PASS\*            | PASS\*                      |
| `07-extern-functions.py`                | PASS            | CRASH (SIGABRT)   | FAIL (HIP 209 on load)      |
| `08-grouped-gemm.py`                    | PASS            | CRASH (SIGABRT)   | FAIL (HIP 209 on load)      |

\* `06-fused-attention.py`'s `__main__` block only calls
`bench_flash_attention.run(...)`, which is `triton.testing.Mark.run`
under the hood and gets stubbed to a no-op by `_bootstrap.py` — the
script imports torch + triton + pytest and exits 0 without ever
launching a kernel.  Re-run with `--no-stub-bench` to actually
exercise it (or write a thin driver that calls `attention(...)`
directly).  This is a real edge case of the breadth-only contract:
"PASS" only proves the script ran end-to-end, not that any
transpilation actually happened.

> **Update:** A targeted smoke test (`smoke_tests/`, write-up in
> `INTEGRATION_GAP.md`) traced every salmon failure in the table
> below to a single cause: Triton's runtime JIT lowers the kernel
> to `buffer_load_b128` with hardware buffer descriptors, and
> salmon does not retarget the descriptor magic-word setup from
> the gfx1250 layout to the gfx942 layout.  The `compare_correctness`
> AOT path doesn't trigger this lowering and passes 4/4 numerically
> for the same kernel source.  Read `INTEGRATION_GAP.md` before
> drawing conclusions from the table or promoting any of these
> scripts into a `compare_correctness` recipe.

What the table is showing across the seven non-trivial scripts:

  * **legacy mode**: 7/7 consistently abort with `Either SourceMgr
    should be available — UNREACHABLE` from `MCContext.cpp:1091`,
    after the byte translator finishes disassembling and renaming
    (the `salmon: OK (… 944/944 instructions)`-style log line shows
    on every run for salmon-mode; the legacy SIGABRT is downstream
    of a successful translate).  Same signature on every script,
    consistent with a real bug in the byte-translator MC emission
    path that any non-trivial Triton kernel will reproduce.
  * **salmon mode** has two distinct failure clusters that the
    breadth runner cleanly separates:
    - **CRASH (SIGSEGV post-launch)** — vector-add and matmul.  The
      load + raise + transpile chain completes (`salmon: OK (gfx1250
      -> gfx942, N/N instructions)`), the kernel launches, and the
      host crashes later in torch's tensor `__repr__` — consistent
      with the launched kernel writing out-of-bounds memory and
      corrupting the host heap.
    - **FAIL (HIP 209 on load)** — softmax, dropout, layer-norm,
      extern-functions, grouped-gemm.  HIP rejects the patched code
      object at `hipModuleLoadDataEx` time with
      "no kernel image is available for execution on the device".
      The intercept, e_flags patch, and ISA-mismatch transpile path
      all engage (`salmon_intercept: redirected hipGetProcAddress`,
      `salmon_intercept: patched e_flags`, `hotswap: .note ISA
      mismatch: gfx1250 != gfx942 — transpile`), but the resulting
      ELF is malformed in a way that `hipModuleLoadDataEx`'s kernel
      validation rejects.  What differs between the two SIGSEGV-path
      kernels (which load fine) and the five HIP-209-path kernels
      (which don't) is the next deep-dive question.

Take-aways for triage:

  * **One signature for all 7 legacy crashes.**  Fix the
    `MCContext.cpp:1091` UNREACHABLE in the byte-translator MC path
    and we expect to recover all seven in one shot.
  * **Every salmon failure is currently downstream of one root
    cause** (see `INTEGRATION_GAP.md`).  Triton's JIT emits the
    buffer-descriptor lowering whenever it can attach
    `tt.divisibility = 16` + `tt.pointer_range = 32` to the kernel
    arguments — which is essentially always for these tutorials.
    Two cheap and complementary fixes:
       1. Make salmon **explicitly refuse** buffer-descriptor setup
          patterns it can't translate — converts silent corruption
          into an honest "unsupported" verdict, in line with the
          project's "always report errors" rule.
       2. Force Triton's JIT to skip that lowering for the runner's
          test corpus (e.g. by ensuring the kernarg attribute
          inference can't satisfy `pointer_range = 32`), so we
          unmask whatever *other* salmon gaps exist behind it.
  * **The two SIGSEGV cases are not closer to working than the
    HIP-209 ones.**  They share the same root cause; only the
    failure mode differs.  Promoting any of them into a
    `compare_correctness` recipe before the buffer-descriptor
    handling is decided will just produce a recipe that fails
    for the same reason all of them do.

## GPT-OSS exact attention check

For GPT-OSS's real operator (not Triton tutorial `06-fused-attention.py`),
run:

```bash
python3 runner.py --script ./gpt_oss_attention_operator.py --modes native,salmon
```

`gpt_oss_attention_operator.py` imports
`gpt_oss.triton.attention.attention` directly (from `GPT_OSS_SRC`,
default `/data/gpt-oss/src`) and compares it against `attention_ref` in
the same module via `torch.testing.assert_close`.

## How the salmon hook actually engages

For Triton-issued module loads to flow through the Salmon ELF
patcher, the LD_PRELOAD shim has to intercept three different
symbol-resolution paths.  This is documented mostly to save the
next person from rediscovering it the hard way.

| caller path                                                    | resolves through            | shim hook                         |
|----------------------------------------------------------------|-----------------------------|-----------------------------------|
| compare_correctness binary; static linkers                     | PLT, with global scope      | `hipModuleLoadData`               |
| Anything that links HIP normally and calls the `Ex` variant    | PLT, with global scope      | `hipModuleLoadDataEx`             |
| Triton's AMD backend driver                                    | `hipGetProcAddress` table   | `hipGetProcAddress`               |
| Triton's `dlopen(libamdhip64, RTLD_LOCAL)` for that table itself | `dlsym(handle, "hipGetProcAddress")` against an `RTLD_LOCAL` handle | `dlsym`                          |

Without the `dlsym` hook, the Triton path would silently bypass
LD_PRELOAD entirely (the symbol it asks for is scoped to the
locally-opened libamdhip64 handle, so the dynamic linker never
even consults preloaded modules).  With the hook in place we
return our `hipGetProcAddress` wrapper, which then redirects every
load symbol Triton asks for onto our patcher.

The `_bootstrap.py` module additionally:

  * monkey-patches `HIPDriver.get_current_target` to force a
    `gfx1250 wave32` `GPUTarget` (Triton's own `TRITON_OVERRIDE_ARCH`
    env var leaves `warp_size` pinned to the device, yielding a
    wave64-IR gfx1250 binary that the transpiler can't ingest);
  * stubs `triton.testing.do_bench` / `Mark.run` so tutorial
    benchmark loops don't dominate wall clock;
  * drains HIP's sticky `hipErrorInvalidValue` after Triton's
    `HIPUtils` probing, which would otherwise get re-raised by the
    next torch op.

## Files

  * `runner.py` — main entry, discovery, mode dispatch, reporting.
  * `_bootstrap.py` — child-side: forces Triton's target, drains
    HIP sticky error state, stubs benchmarks, runpy's the user
    script.  Documented in-line.
  * `_empty_rules.json` — minimal rules file for `HSA_HOTSWAP_RULES`
    (`{"version":1,"rules":[]}`); silences the JSON-parse warning
    the hook emits when fed `/dev/null`.
  * `.venv-rocm7/` — local-only ROCm-7 nightly torch venv, see
    Setup above.  Not committed.
  * `README.md` — this file.

## Relationship to `compare_correctness`

This tool **deliberately does not duplicate**
`compare_correctness`'s plumbing.  We share:

  * the same `libsalmon_intercept.so` (consumed via
    `--libsalmon`, defaults to the path inside
    `tools/compare_correctness/`);
  * the same default corpus (the `_corpus/upstream/*.py` files
    pulled by `compare_correctness/kernels/triton/_corpus/pull.py`).

We **do not** share the AOT compile path, the per-recipe Python
shims, the per-output comparator infrastructure, or the Triton venv
— those are exactly the surface area this tool trades away in
exchange for zero per-script setup.
