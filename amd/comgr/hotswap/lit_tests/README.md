# Hotswap transpiler — lit tests

This directory holds FileCheck-style lit tests for the AMDGPU binary
transpiler (Hotswap). Unlike the gtest suite under `../tests/`, these
tests exercise the raiser at the IR level: they assemble a small
AMDGPU fixture with `llvm-mc` + `ld.lld`, feed the resulting hsaco to
`raise_cli --emit-ir`, and FileCheck either the emitted LLVM IR on
stdout or the raiser's diagnostic output on stderr.

Lit is well-suited for the kinds of properties we want to assert here:

*   Structural invariants of the raised IR — e.g. "every
    side-effectful instruction under divergent EXEC is wrapped in an
    SPE `spe_do` / `spe_skip` diamond keyed on `%spe_lane_active`".
*   Negative / diagnostic assertions — e.g. "the pre-translation abort
    gate fires when the code object contains an EXEC-writer whose
    SemOp is not on the SPE allow-list", or "cross-wave translation of
    an EXEC-manipulating kernel emits a modulo-replication warning".
*   Per-SemOp audits — asserting that every allow-listed EXEC-writer
    actually routes its write through `storeExec`, visible as the new
    EXEC SSA value being used by the next `lshr i64 %exec, %spe_lane_mod`.

The existing gtest suite still owns the *runtime* checks (the raised
kernel is linked and dispatched on a real GCN device, and the output
buffer is compared against hardware ground truth). Lit tests
complement that by giving us principled, fast, non-GPU IR assertions
that catch regressions in the raiser *before* they manifest as
hardware mismatches.

## Layout

Every test is a single self-contained `.s` file. The top of the file
carries the lit `RUN:` / `CHECK:` directives as AMDGPU-assembly
comments (`;`); the rest is the kernel body plus its `.amdhsa_kernel`
and `.amdgpu_metadata` blocks — exactly what `llvm-mc` needs to
assemble an amdhsa code object. No sidecar `.hip` or `.ll` files, no
build-time tool invocations, no fixture catalogues wired through
CMake: `lit.cfg.py` globs `*.s` under this directory and each test
drives its own assembly.

```
lit_tests/
├── <name>.s                — fixture + RUN + CHECK, one file per test
├── CMakeLists.txt          — harness (discovers FileCheck / llvm-mc / ld.lld)
├── lit.cfg.py              — lit configuration (substitutions, suite name)
├── lit.site.cfg.py.in      — CMake-generated site configuration
└── README.md               — this file
```

## RUN-line template

The canonical RUN block assembles, raises, and FileChecks in one
pipeline. Replace the arch / kernel name as needed; everything else
copies verbatim:

```
; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir=<kernel-name> 2>/dev/null | %FileCheck %s
```

*   `%llvm_mc` assembles the fixture against the declared ISA; the
    `.amdgcn_target` directive inside the source must match `-mcpu=`
    (llvm-mc will diagnose a mismatch).
*   `%ld_lld -shared` links the relocatable ELF into the DYN ELF that
    the amdhsa loader — and `raise_cli` — consume.
*   `%raise_cli ... --emit-ir[=<kernel>] 2>/dev/null` dumps the raised
    IR for a single kernel on stdout. Drop the `=<kernel>` when the
    fixture has one kernel (raise_cli picks it automatically). Redirect
    stderr to `/dev/null` for success-path tests; use `2>&1` plus `%not`
    for diagnostic/negative-path tests.

Cross-ISA tests add `--target-isa=<arch>` to the raise_cli invocation
so the raiser lowers for a different target than the source ISA (see
`wmma_scale_f32_16x16x128_f8f6f4.s` for the canonical two-RUN pattern:
one cross-arch refusal and one same-arch lift).

## Substitutions provided by the harness

Tests should only reference the substitutions below. Do not hard-code
absolute paths or tool names.

| Substitution | Meaning |
|---|---|
| `%raise_cli` | Path to the built `raise_cli` binary. |
| `%llvm_mc`   | `llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj` (tool + fixed flags). |
| `%ld_lld`    | Path to `ld.lld` (pass `-shared` explicitly so the hsaco shape stays visible). |
| `%FileCheck` | Path to `FileCheck` (LLVM). |
| `%not`       | Path to `not` (LLVM; runs its command and inverts the exit code). |
| `%s`         | Path to the current fixture. |
| `%t`         | Per-test scratch path prefix (e.g. `%t.o`, `%t.hsaco`). |

`llvm-mc`, `ld.lld`, `FileCheck`, `not`, and `llvm-lit` are all
resolved at CMake configure time from the LLVM build tree the parent
project links against; there is no runtime tool discovery.

## Writing style

*   Keep fixtures small. One kernel per file. Inline-assembly blocks
    (`;;#ASMSTART` / `;;#ASMEND` as emitted by clang for `asm volatile(...)`)
    are the recommended way to force the raiser to see a specific
    instruction sequence the rest of the surrounding kernel body does
    not already produce.
*   Keep the `.amdhsa_kernel` / `.amdgpu_metadata` blocks minimal —
    only values that differ from the LLVM defaults need to be listed
    (see `MCKernelDescriptor::getDefaultAmdhsaKernelDescriptor` for the
    per-subtarget defaults, and `AMDGPUMetadataVerifier.cpp` for which
    metadata fields are required vs. optional).
*   CHECK lines should match *SSA patterns*, not exact SSA names —
    the SSA numbering is not stable across LLVM versions. Use
    `CHECK:`, `CHECK-NEXT:`, `CHECK-SAME:`, and variable bindings
    `[[VAR:%[^ ,]+]]` liberally.
*   When a test exercises a diagnostic, FileCheck against `stderr`
    with `--check-prefix=STDERR` and match the *stable* substring of
    the message — "`pre-translation abort`", "`modulo-replication`",
    etc. — rather than pinning to the exact sentence, so future
    rewordings don't break the test.
*   Prefer `%not %raise_cli ... 2>&1 | %FileCheck` over
    `%raise_cli ... 2>&1 | %FileCheck` when asserting an abort; this
    catches regressions where the abort silently degrades into a
    warning.

## Adding a new test

1.  Create `lit_tests/<name>.s`.
2.  Write the RUN block at the top using the template above (adjust
    the `-mcpu=` arch and the `--emit-ir=` kernel name).
3.  Add `CHECK:` / `CHECK-LABEL:` / `CHECK-NOT:` / `CHECK-DAG:` lines
    pinning the invariant the test is guarding.
4.  Paste the kernel body — `.amdgcn_target`, `.text`, `.globl`,
    kernel code, `.rodata` with `.amdhsa_kernel`, and
    `.amdgpu_metadata` — after the CHECK block.
5.  Run `ctest -R transpiler-lit --output-on-failure` (or
    `llvm-lit -sv lit_tests/<name>.s` for a single test) and confirm
    it passes.

Do not commit built `.o`, `.hsaco`, or `Output/` artefacts. They are
produced per-test by lit and live in the build tree.
