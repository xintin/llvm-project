// COM: Test HotSwap DS2 s_wait_dscnt drain semantic: a kernel with two
// COM: DS2 sites followed by `s_wait_dscnt 0` (drain-all) must NOT have
// COM: the wait imm bumped. The drain semantic is independent of how
// COM: many DS ops were issued, so it remains correct after the splits
// COM: PR #2281 / #2212 emit. Bumping the drain to s_wait_dscnt N would
// COM: relax it, leaving N split halves potentially in flight when the
// COM: wave returns -- the opposite of the compiler's intent.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Drain wait must remain s_wait_dscnt 0x0 even with two preceding
// COM: DS2 sites in the same basic block.
// DISASM-LABEL: <test_ds2_wait_bump_drain_kernel>:
// DISASM: ds_load_2addr_b32
// DISASM: ds_load_2addr_b32
// DISASM: s_wait_dscnt 0x0
// DISASM: s_endpgm

// COM: Idempotency: rewriting the patched output again should produce
// COM: identical bytes (the drain skip means our pass is a no-op on
// COM: this kernel both passes).
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds2_wait_bump_drain_kernel
.p2align 8
.type test_ds2_wait_bump_drain_kernel,@function
test_ds2_wait_bump_drain_kernel:
  ds_load_2addr_b32 v[0:1], v4 offset0:0 offset1:1
  ds_load_2addr_b32 v[2:3], v5 offset0:0 offset1:1
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_ds2_wait_bump_drain_kernel_end:
.size test_ds2_wait_bump_drain_kernel, .Ltest_ds2_wait_bump_drain_kernel_end-test_ds2_wait_bump_drain_kernel

.rodata
.p2align 8
.amdhsa_kernel test_ds2_wait_bump_drain_kernel
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 2
  .amdhsa_group_segment_fixed_size 256
.end_amdhsa_kernel
