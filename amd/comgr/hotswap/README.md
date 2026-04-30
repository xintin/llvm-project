# AMD Hotswap Transpiler

> **Migration note:** This project has just moved from the ROCR
> `hotswap/transpiler` tree into LLVM under `amd/hotswap`. The source code and
> focused COMGR validation are current, but some design notes, comments, and
> docs may still describe the old ROCR-local layout or lag the new
> LLVM/COMGR-owned integration shape.

`amd/hotswap` contains the compiler-side AMDGPU binary translation library used
by the COMGR hotswap-transpile prototype. It lifts AMDGPU code objects to LLVM
IR, lowers them through the in-tree AMDGPU backend for a target ISA, and returns
a translated HSACO.

This directory is intentionally compiler-owned. Runtime policy, HIP preload
shims, GPT-OSS/SGLang runners, and ROCR loader integration live outside this
LLVM tree.

## Build Modes

### LLVM External Project

The intended in-tree development configuration is to build hotswap as an LLVM
external project together with COMGR and device-libs:

```bash
cmake -S llvm -B <build-dir> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
  -DLLVM_EXTERNAL_PROJECTS="device-libs;hotswap;comgr" \
  -DLLVM_EXTERNAL_DEVICE_LIBS_SOURCE_DIR=<llvm-project>/amd/device-libs \
  -DLLVM_EXTERNAL_HOTSWAP_SOURCE_DIR=<llvm-project>/amd/hotswap \
  -DLLVM_EXTERNAL_COMGR_SOURCE_DIR=<llvm-project>/amd/comgr \
  -DCOMGR_ENABLE_HOTSWAP_TRANSPILE=ON \
  -DLLVM_USE_LINKER=lld
```

Build the focused targets used by the COMGR smoke test:

```bash
cmake --build <build-dir> \
  --target hotswap-transpiler amd_comgr hotswap-transpile \
           FileCheck llc llvm-mc lld llvm-readelf llvm-objdump \
  --parallel 16
```

### Standalone Hotswap Library

For local library-only iteration, point `LLVM_DIR` at an LLVM build tree. An
LLVM install tree is not sufficient because hotswap uses AMDGPU target-private
headers and generated TableGen include files.

```bash
cmake -S amd/hotswap -B amd/hotswap/build -G Ninja \
  -DLLVM_DIR=<build-dir>/lib/cmake/llvm \
  -DCMAKE_CXX_COMPILER=clang++
cmake --build amd/hotswap/build --target hotswap-transpiler
```

## COMGR Integration

COMGR enables the prototype entry point with:

```text
COMGR_ENABLE_HOTSWAP_TRANSPILE=ON
```

When enabled, COMGR links the `hotswap::transpiler` target and exports:

```c
amd_comgr_status_t amd_comgr_hotswap_transpile(
    amd_comgr_data_t input,
    const char *source_isa_name,
    const char *target_isa_name,
    amd_comgr_data_t *output);
```

The API currently accepts an executable code object and returns a new executable
code object. It is prototype-level: options, diagnostics, cache policy, and API
stability still need a reviewed COMGR contract.

## Validation

The focused COMGR smoke test verifies argument validation and a real
`gfx950 -> gfx942` translation of `tests/vecadd_gfx950.co`:

```bash
<build-dir>/bin/llvm-lit -v \
  <build-dir>/tools/comgr/test-lit/hotswap-transpile.c
```

Expected result:

```text
PASS: Comgr :: hotswap-transpile.c
```

A direct manual invocation is also possible:

```bash
<build-dir>/tools/comgr/test-lit/hotswap-transpile \
  amd/hotswap/tests/vecadd_gfx950.co \
  amdgcn-amd-amdhsa--gfx950 \
  amdgcn-amd-amdhsa--gfx942 \
  --output=/tmp/vecadd_gfx942.co
```

## Tests Kept In Tree

The LLVM-facing tree keeps:

- assembly-based lit fixtures under `lit_tests/`;
- focused unit tests under `tests/`;
- the small `tests/vecadd_gfx950.co` fixture used by COMGR lit coverage.

Local validation harnesses that depend on a ROCR adapter, HIP preload shims,
GPT-OSS/SGLang, AITER, or large external corpora are intentionally not part of
this LLVM-facing directory. They are tracked separately in the migration scratch
area and can become a ROCR-side validation stack later.

## Current Limitations

- The pipeline still shells out to `llc`, `llvm-mc`, and `ld.lld`; an
  upstreamable runtime path should move to in-process LLVM/LLD or COMGR-owned
  entry points.
- The COMGR public entry point is provisional.
- Translation cache policy exists in the hotswap library but is not yet wired
  through the COMGR API.
- ROCR integration is a separate adapter patch stack; this directory only owns
  compiler-side translation.
