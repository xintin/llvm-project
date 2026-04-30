// COM: Test HotSwap DS2 s_wait_dscnt noop: a kernel with no DS2
// COM: instructions must pass through the wait-bump post-pass without
// COM: touching any s_wait_dscnt. The pass should also be cheap on this
// COM: shape: Pass 1 finds nothing to record and Pass 2 never runs.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Single-address DS load + non-zero wait: the wait imm must remain
// COM: 0x3 because no DS2 mnemonic precedes it.
// DISASM-LABEL: <test_ds2_wait_bump_noop_kernel>:
// DISASM: ds_load_b32
// DISASM: s_wait_dscnt 0x3
// DISASM: s_endpgm

// COM: Idempotency: the noop kernel must rewrite to the same bytes both
// COM: times (no DS2 means no bump on either pass).
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds2_wait_bump_noop_kernel
.p2align 8
.type test_ds2_wait_bump_noop_kernel,@function
test_ds2_wait_bump_noop_kernel:
  ds_load_b32 v0, v4
  s_wait_dscnt 0x3
  s_endpgm
.Ltest_ds2_wait_bump_noop_kernel_end:
.size test_ds2_wait_bump_noop_kernel, .Ltest_ds2_wait_bump_noop_kernel_end-test_ds2_wait_bump_noop_kernel

.rodata
.p2align 8
.amdhsa_kernel test_ds2_wait_bump_noop_kernel
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 2
  .amdhsa_group_segment_fixed_size 256
.end_amdhsa_kernel
