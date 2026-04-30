# AMD Hotswap Transpiler

`amd/comgr/hotswap` contains the compiler-side AMDGPU binary translation library used
by the COMGR hotswap-transpile prototype. It lifts AMDGPU code objects to LLVM
IR, lowers them through the in-tree AMDGPU backend for a target ISA, and returns
a translated HSACO.

This directory is compiler-owned and is consumed by COMGR. Runtime loaders
should call COMGR instead of including hotswap internals directly.

## Build Modes

### LLVM External Project

The intended in-tree development configuration is to build COMGR as an LLVM external project with hotswap enabled:

```bash
cmake -S llvm -B <build-dir> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
  -DLLVM_EXTERNAL_PROJECTS="device-libs;comgr" \
  -DLLVM_EXTERNAL_DEVICE_LIBS_SOURCE_DIR=<llvm-project>/amd/device-libs \
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
cmake -S amd/comgr/hotswap -B amd/comgr/hotswap/build -G Ninja \
  -DLLVM_DIR=<build-dir>/lib/cmake/llvm \
  -DCMAKE_CXX_COMPILER=clang++
cmake --build amd/comgr/hotswap/build --target hotswap-transpiler
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
  amd/comgr/hotswap/tests/vecadd_gfx950.co \
  amdgcn-amd-amdhsa--gfx950 \
  amdgcn-amd-amdhsa--gfx942 \
  --output=/tmp/vecadd_gfx942.co
```

## Tests Kept In Tree

The LLVM-facing tree keeps:

- assembly-based lit fixtures under `lit_tests/`;
- focused unit tests under `tests/`;
- the small `tests/vecadd_gfx950.co` fixture used by COMGR lit coverage.

Runtime and workload-level validation should live in the runtime integration
that calls COMGR. The hotswap tree keeps compiler-unit tests, lit fixtures, and
small code-object smoke coverage.

## Current Limitations

- The pipeline still shells out to `llc`, `llvm-mc`, and `ld.lld`; an
  upstreamable runtime path should move to in-process LLVM/LLD or COMGR-owned
  entry points.
- The COMGR public entry point is provisional.
- Translation cache policy exists in the hotswap library but is not yet wired
  through the COMGR API.
- Runtime integration is expected to use COMGR as the boundary; this directory
  owns compiler-side translation.
