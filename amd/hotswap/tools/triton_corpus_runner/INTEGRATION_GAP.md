# Why Triton scripts fail in salmon mode (corrected diagnosis)

This document captures the diagnosis from a smoke test prompted by the
question "every Triton script fails under salmon — is the integration
broken?".

**Short answer**: the runner integration is correct.  Salmon has at least
two distinct silent-miscompile sites that produce SIGSEGV at launch
time.  Neither is the SQ_BUF_RSRC descriptor mismatch the first
iteration of this document blamed: side-by-side IR + lowered-asm dumps
of a working kernel and two crashing kernels show the buffer-load
shapes are identical across them.

A previous version of this document attributed every salmon SIGSEGV to
"SQ_BUF_RSRC descriptor word-3 layout differs between gfx1250 and
gfx942".  That attribution was wrong (see "Why the original diagnosis
was wrong" below) and has been removed.

## The reproduction (what passes vs. what crashes)

Three kernels were dumped through salmon with `HSA_SALMON_DUMP_DIR`
pointing at a per-run directory; the dumped artefacts are the lifted
LLVM IR (`*.ll`), the gfx942 lowered text (`*.s`), the linker object
(`k0.o`), and the final `merged.hsaco`/`kernel.hsaco`.

  Kernel A — historical `Gfx1250Gpu.Softmax` classifier smoke
    Source: `transpiler/test_data/gfx1250/softmax_gfx1250.hsaco`
    Driver: `transpiler/tests/gfx1250_gpu_test.cpp` Softmax test
    Launch: grid=(1,1,1) wg=(128,1,1) over `n_rows=1, n_cols=512`
    Allocations: 4 MiB each (`1024 * 1024` f32) for in + out
    Historical result in this smoke: 0 errors, maxErr=0.000000e+00.
    Current gtest status: XFAIL; after the pipeline-launch fix the
    fixture reaches the GPU and produces invalid `+inf` output, so do
    not use this row as the current Softmax correctness verdict.

  Kernel B — `compare_correctness corpus_softmax_fp32` (SIGSEGV)
    Source: `_corpus/extracted/softmax_kernel.py` (Triton tutorial 02)
    Build: `aot_compile.py` for gfx1250 wave32, `BLOCK_SIZE=1024`,
           `num_warps=4`
    Launch: grid=(`n_rows`=16, 1, 1) wg=(128,1,1)
    Allocations: tightly sized to `n_rows * n_cols` f32
    Result: SIGSEGV in salmon mode for every `n_cols` in [128, 256,
            512, 1024]

  Kernel C — `_smoke_triton_jit_addk.py` (SIGSEGV)
    Source: same `add_kernel` from `_corpus/extracted/add_kernel.py`,
            but Triton-JIT-compiled in-process at first launch
    Launch: grid=(1,1,1) wg=(128,1,1) over `n_elements=1024`
    Allocations: `torch.empty_like(x)` for output (uninitialised)
    Result: SIGSEGV after launch

Note that `compare_correctness corpus_add_fp32` (the AOT counterpart to
Kernel C, same kernel source) PASSes 4/4 in salmon mode.  Same Triton
version, same `GPUTarget("hip", "gfx1250", warp_size=32)`, same
`num_warps=4` — but the JIT path attaches `tt.divisibility = 16` and
`tt.pointer_range = 32` based on runtime evidence and ends up emitting
a different kernel binary.

## Side-by-side IR + lowered-asm dumps

All three kernels lift to **the same** buffer-load intrinsic family
(`llvm.amdgcn.raw.buffer.load.{i32,v4i32}`) with a `<4 x i32>` SRD
operand:

```
; Kernel A (working softmax)
%buf_ld = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> %45, i32 %46, i32 0, i32 0)

; Kernel B (crashing corpus_softmax)
%buf_ld = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> %85, i32 %86, i32 0, i32 0)

; Kernel C (crashing JIT add_kernel)
%buf_ld = call <4 x i32> @llvm.amdgcn.raw.buffer.load.v4i32(<4 x i32> %31, i32 %32, i32 0, i32 0)
```

All three lower (via the gfx942 LLVM backend) to **the same** gfx9
buffer-load shape with **the same** SRD-fixup pattern:

```
; Kernel A
buffer_load_dword v1, v14, s[24:27], 0 offen
... (s_and_b32 sN, sM, 0xffff  +  s_mov_b32 sN+1, 0xffffff for the SRD setup)

; Kernel B
buffer_load_dword v1, v12, s[76:79], 0 offen
... (same s_and_b32 / s_mov_b32 SRD-fixup pattern at multiple sites)

; Kernel C
buffer_load_dwordx4 v[2:5], v1, s[8:11], 0 offen
s_mov_b32 s9,  0xffff      ; SRD dw1 placeholder
s_mov_b32 s10, 0xffffff    ; SRD dw2 (num_records / format word)
s_mov_b32 s11, 0           ; SRD dw3
... s_and_b32 s9, s2, 0xffff (restore base_addr_hi for second load)
```

The SRD bytes the gfx942 hardware actually sees in all three kernels
are gfx9-correct (LLVM's gfx942 backend constructs them).  The
buffer-load instruction shape is also identical.  **There is no
descriptor-encoding mismatch and no instruction-shape mismatch between
the working and crashing kernels.**

## What actually distinguishes the kernels

Distinct intrinsic sets in each lifted IR:

  Kernel A (working softmax):
    amdgcn.raw.buffer.load.i32, amdgcn.mbcnt.{lo,hi}, amdgcn.readlane,
    amdgcn.readfirstlane, amdgcn.s.barrier, amdgcn.update.dpp.i32,
    amdgcn.ds.bpermute, amdgcn.ballot, amdgcn.div.{fixup,fmas,scale},
    amdgcn.workgroup.id.{x,y}, amdgcn.workitem.id.x

  Kernel B (crashing corpus_softmax):
    same as A, MINUS amdgcn.update.dpp / amdgcn.ds.bpermute,
    PLUS amdgcn.implicitarg.ptr            <— unique discriminator

  Kernel C (crashing JIT add_kernel):
    amdgcn.raw.buffer.load.v4i32, amdgcn.mbcnt.{lo,hi},
    amdgcn.readfirstlane, amdgcn.workgroup.id.{x,y}, amdgcn.workitem.id.x
    (no implicitarg.ptr, no ds.bpermute, no update.dpp)

So the salmon-level discriminators are:

1. **`llvm.amdgcn.implicitarg.ptr`** — used by Kernel B, not by A or C.
   Salmon synthesises it in
   [handle_smem.cpp:81-94](../../handle_smem.cpp), via the path:

   ```cpp
   // base s[0:1] + immediate offset that lands past the user-kernarg
   // block: salmon emits implicitarg_ptr + (sourceOffset - sourceImplicitArgsBase)
   int implOffset = byteOffset - ctx.kernargs.implicitArgsBase;
   Value *implPtr = ctx.B.CreateCall(fnImplicitArgPtr, {}, "implicitarg_ptr");
   Value *gep = ctx.B.CreateInBoundsGEP(ctx.i8Ty, implPtr,
                                        ctx.B.getInt64(implOffset), "impl_gep");
   ctx.regs.storeSGPR32(ctx.B, dest.baseIdx,
                        ctx.B.CreateLoad(ctx.i32Ty, gep, "impl_load"));
   ```

   `implOffset` is computed against the **gfx1250** kernel's
   `implicitArgsBase`.  At runtime `implicitarg_ptr` returns a pointer
   to the **gfx942** runtime's hidden-arg block.  If either the base
   or the field layout differs between the two architectures /
   loaders, the kernel reads wrong values for `hidden_block_count_*`,
   `hidden_grid_dims`, `hidden_global_offset_*` etc.  Wrong grid dims
   feeding into address arithmetic produces out-of-bounds GPU loads
   that SIGSEGV on the host (page fault delivered through the HSA
   queue).  Today salmon emits this load silently — no warning, no
   refusal — so we cannot tell from the salmon log that the kernel
   reads cross-arch hidden args.

2. **`s_setreg_imm32_b32 mode, imm`** — emitted by Kernels A and B
   (Kernel C does not emit it).  Salmon currently treats this in
   [handle_sopk.cpp:83-95](../../handle_sopk.cpp) as
   `HwregWrite::WarnDrop`: it logs a warning, drops the write, and
   continues.  The comment on that case literally says:

   > "this is observationally correct only when downstream compute is
   > insensitive to the written bits.  Tighten to Abort once the corpus
   > allows it."

   The MODE register controls FP rounding mode, denormal handling,
   FTZ, IEEE mode and graphics-context bits.  If a kernel sets it
   intending downstream FP ops to behave a certain way and salmon
   drops the write, the kernel will silently produce wrong results
   (not necessarily SIGSEGV).  Kernel A passes despite this drop —
   the dropped bits genuinely don't affect its compute.  Kernel B
   may or may not depend on them; the SIGSEGV itself is more
   plausibly caused by the implicitarg.ptr issue, but until we
   refuse this path we can't be sure.

3. **Kernel C is its own bug.**  Its lowered `.s` ends with two
   `flat_store_dwordx4` writes — both to addresses near the private
   (scratch) aperture base, not to the real output kernarg pointer:

   ```
   v_mov_b32_e32 v10, 4
   v_mov_b32_e32 v11, s5            ; s5 = src_private_base hi32
   ...
   flat_store_dwordx4 v[10:11], v[2:5]   ; addr = (src_private_base.hi << 32) | 4
   v_mov_b32_e32 v10, 0
   flat_store_dwordx4 v[10:11], v[2:5]   ; addr = (src_private_base.hi << 32) | 0
   s_endpgm
   ```

   The real output buffer pointer never appears in any store.  This
   matches the salmon MUBUF-store handler's `select(oob, sinkFlat,
   realPtr)` pattern, where the OOB sink is a per-wave private alloca:
   if LLVM constant-folds `oob` to `true` (because the OOB predicate
   is expressed in terms that simplify to a tautology under the
   `<4 x i32>` SRD shape salmon emits), the realPtr branch is dropped
   and the store always sinks to scratch.  Diagnosing and fixing
   that select is a separate piece of work; this document only flags
   it.

## Why the original diagnosis was wrong

The first iteration of this document inferred from disassembly that
salmon left `0xfc000000` and `0xffffff` "magic constants" loaded into
the SRD SGPRs in their gfx1250 form.  The dumps prove the opposite:
the gfx942 LLVM backend re-encodes the SRD to gfx9-correct values
before the `buffer_load`, and it does so identically for the working
kernel and both crashing kernels.  The descriptor encoding the
hardware sees is gfx9-correct in every case.

Reproducer that demonstrates this:

```bash
mkdir -p /tmp/dump_softmax_works /tmp/dump_softmax_crashes /tmp/dump_jit_addk

# Working softmax via the gtest harness.
cd projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/build
HSA_SALMON_DUMP_DIR=/tmp/dump_softmax_works \
  ./transpiler_tests --gtest_filter='Gfx1250Gpu.Softmax'

# Crashing softmax via compare_correctness.
cd ../tools/compare_correctness
HSA_SALMON_DUMP_DIR=/tmp/dump_softmax_crashes \
HSA_HOTSWAP_ISA_OVERRIDE=gfx942 HSA_HOTSWAP_IR_RAISER=1 \
HSA_HOTSWAP_RULES=../triton_corpus_runner/_empty_rules.json \
LD_PRELOAD=./libsalmon_intercept.so \
  ./compare_correctness --recipe=corpus_softmax_fp32

# Crashing JIT add_kernel via the runner.
cd ../triton_corpus_runner
HSA_SALMON_DUMP_DIR=/tmp/dump_jit_addk \
  python3 runner.py --modes=salmon --script=smoke_tests/_smoke_triton_jit_addk.py

# Confirm identical buffer-load shapes:
grep -E '\bbuffer_(load|store)' /tmp/dump_softmax_works/*/softmax_kernel.s | head
grep -E '\bbuffer_(load|store)' /tmp/dump_softmax_crashes/*/softmax_kernel.s | head
grep -E '\bbuffer_(load|store)' /tmp/dump_jit_addk/*/add_kernel.s | head

# Confirm the unique implicitarg.ptr usage:
grep -l 'amdgcn.implicitarg.ptr' /tmp/dump_*/*/*.ll
```

## Where to fix this

Updated set of fixes, ordered by ratio of (signal recovered) /
(implementation effort):

A. **Promote the two known silent-drop sites to refusals under an
   opt-in env var.**  Add `HSA_SALMON_STRICT=1`; when set, the
   `handle_sopk.cpp` MODE-write path returns `Abort` instead of
   `WarnDrop`, and the `handle_smem.cpp` `implicitarg.ptr` lift path
   returns `RaiseFailure(...)` instead of emitting the cross-arch
   load.  The runner sets the env var; `compare_correctness` and the
   gtest binary do not.  Result: the runner gets honest `UNSUPPORTED`
   verdicts on the kernels that hit either site, while every
   currently-passing test stays passing.  This is the *immediate*
   action and is what the corresponding plan implements.

B. **Make the runner stop tripping on this in the first place.**
   Two sub-options:

     B.1  Force Triton's JIT to skip the buffer-descriptor lowering
          by arranging for `tt.pointer_range` to be 64 (e.g. by
          passing buffers above 4 GiB of base address, or by patching
          the AMD backend's specialization-attrs heuristic before
          calling the kernel).  Verify by re-running 01-vector-add
          and confirming the asm uses `global_load` again.  Note:
          this is now known to be at-best-orthogonal to the actual
          SIGSEGV cause for these kernels (since the buffer-load
          shape isn't itself broken), but it does avoid the JIT
          add_kernel flat-store-sink bug if that bug only triggers
          when the OOB predicate simplifies to true under the
          buffer-descriptor pattern.

     B.2  Build each tutorial's kernel via `aot_compile.py` ahead of
          time and substitute the AOT .hsaco in for the JIT cache
          entry before running the tutorial.  Most decoupled option,
          at the cost of per-kernel manifest maintenance and losing
          the "tests run the same binary the user would run" property.

C. **Bridge the `implicitarg.ptr` cross-arch layout in salmon.**  The
   real fix for Kernel B: have the salmon raiser either translate
   each gfx1250 hidden-arg byte offset to its gfx942 equivalent at
   lift time, or rebuild the implicit-arg block at module-load time
   so the kernel's lifted `implicitarg_ptr + offset` reads the value
   the gfx1250 kernel intended.  Needs a clear schema: which COv5
   hidden-arg fields salmon promises to bridge, what happens for
   non-COv5 inputs, and whether to bridge the *base* (the SGPR pair
   that points to the implicit-arg block) or just the *offsets* into
   it.

D. **Diagnose and fix the JIT add_kernel flat-store-sink bug
   (Kernel C).**  Salmon's MUBUF-store handler emits `store_ptr =
   select(oob, sinkFlat, realPtr)`.  In Kernel C this select is
   constant-folded by LLVM to "always sinkFlat", which means the
   kernel never writes to the real output buffer.  Diagnose how
   `oob` is being computed, and either tighten the predicate so it
   doesn't simplify to true, or fall back to a path that doesn't
   rely on a select-and-let-LLVM-DCE-it pattern.  Until this is
   fixed, fix A's strict mode will reclassify Kernel C as `CRASH`
   rather than `UNSUPPORTED` (the kernel doesn't trip either of the
   two strict-mode refusal sites).

A and C together are the eventual stable state for Kernel B.  A and D
together are the eventual stable state for Kernel C.  This document's
companion plan implements A as the immediate next step.

## So is the integration broken?

No.  Same three independent confirmations as before still hold:

  1. `compare_correctness corpus_add_fp32` passes 4/4 in salmon mode.
  2. `compare_correctness corpus_add_fp32` *invoked by the runner*
     still passes 4/4 in salmon mode under all three modes (native,
     legacy, salmon).
  3. The on-disk known-good `corpus_add_fp32.gfx1250.co` is
     byte-identical to what `aot_compile.py`-style direct
     `triton_compile()` calls produce inside the runner's environment.

The shim, the LD_PRELOAD chain, the libhsa selection, the env vars and
the GPU pin are wired correctly.  What is broken is the salmon
transpiler in two specific, now-localised ways that the immediate
plan addresses with explicit refusals.

## How to reproduce

```bash
# Confirm AOT path works through the runner's environment.
$ ./runner.py --modes all --gpu 0 --timeout 60 \
    --script smoke_tests/_smoke_compare_correctness.py
# all 3 modes PASS.

# Confirm JIT path of the same kernel crashes.
$ ./runner.py --modes all --gpu 0 --timeout 60 \
    --script smoke_tests/_smoke_triton_jit_addk.py
# native PASS, legacy SIGABRT (UNREACHABLE in MCContext, separate
# legacy-path bug), salmon SIGSEGV.

# Dump artefacts for the kernels discussed above.
$ HSA_SALMON_DUMP_DIR=/tmp/dump_softmax_works \
    ../../build/transpiler_tests --gtest_filter='Gfx1250Gpu.Softmax'
$ cd ../compare_correctness && \
    HSA_SALMON_DUMP_DIR=/tmp/dump_softmax_crashes \
    HSA_HOTSWAP_ISA_OVERRIDE=gfx942 HSA_HOTSWAP_IR_RAISER=1 \
    HSA_HOTSWAP_RULES=../triton_corpus_runner/_empty_rules.json \
    LD_PRELOAD=./libsalmon_intercept.so \
    ./compare_correctness --recipe=corpus_softmax_fp32
$ cd ../triton_corpus_runner && \
    HSA_SALMON_DUMP_DIR=/tmp/dump_jit_addk \
    python3 runner.py --modes=salmon \
      --script=smoke_tests/_smoke_triton_jit_addk.py

# Compare.
$ ls /tmp/dump_softmax_works/*/  # softmax_kernel.{ll,s}, kernel.hsaco
$ ls /tmp/dump_softmax_crashes/  # one salmon-XXXXXX subdir per child run
$ ls /tmp/dump_jit_addk/         # one salmon-XXXXXX subdir
```
