// COM: Test HotSwap DS2 s_wait_dscnt bump: a kernel with two DS2 sites
// COM: followed by a non-zero s_wait_dscnt should have the wait imm
// COM: bumped by 2 to compensate for the splits the paired patches
// COM: (PR #2281 / #2212) emit at the DS2 sites.
// COM:
// COM: This test runs against the rewriter without a DS2 producer wired
// COM: in (the strong-symbol applyTrampolinePatches override is not on
// COM: amd-staging yet), so the wait gets bumped but the DS2 mnemonics
// COM: themselves are passed through unchanged. The bump is over-eager
// COM: in that configuration; on a build that includes #2281 / #2212
// COM: the DS2 sites would be rewritten to s_branch trampolines and the
// COM: bump count would be exact. Either way, the wait-bump pass is the
// COM: only correctness-relevant change for this kernel.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Two ds_load_2addr_b32 sites; original s_wait_dscnt 0x5 should
// COM: become s_wait_dscnt 0x7 (5 + 2 splits).
// DISASM-LABEL: <test_ds2_wait_bump_kernel>:
// DISASM: ds_load_2addr_b32
// DISASM: ds_load_2addr_b32
// DISASM: s_wait_dscnt 0x7
// DISASM: s_endpgm

// COM: Idempotency is intentionally NOT asserted: without a producer
// COM: that converts the DS2 mnemonics into trampoline branches, a
// COM: second rewrite would re-bump the (now 0x7) wait to 0x9. With a
// COM: producer wired in, the second rewrite would see s_branch in
// COM: place of the DS2 mnemonics and bump nothing -- that idempotency
// COM: comes from the producer, not from this pass alone. See file
// COM: header in comgr-hotswap-patch-ds2-wait-bump.cpp.

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds2_wait_bump_kernel
.p2align 8
.type test_ds2_wait_bump_kernel,@function
test_ds2_wait_bump_kernel:
  ds_load_2addr_b32 v[0:1], v4 offset0:0 offset1:1
  ds_load_2addr_b32 v[2:3], v5 offset0:0 offset1:1
  s_wait_dscnt 0x5
  s_endpgm
.Ltest_ds2_wait_bump_kernel_end:
.size test_ds2_wait_bump_kernel, .Ltest_ds2_wait_bump_kernel_end-test_ds2_wait_bump_kernel

.rodata
.p2align 8
.amdhsa_kernel test_ds2_wait_bump_kernel
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 2
  .amdhsa_group_segment_fixed_size 256
.end_amdhsa_kernel
