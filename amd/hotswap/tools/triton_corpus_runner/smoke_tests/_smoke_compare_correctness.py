"""Smoke test #1: drive the known-good compare_correctness binary through
the triton_corpus_runner's environment.

If ``compare_correctness corpus_add_fp32`` passes 4/4 in salmon mode when
launched standalone but fails when invoked through this script (which
inherits the runner's LD_PRELOAD chain, env vars, GPU pin, etc.), then
the runner's environment is what's breaking the integration — not the
salmon transpiler, not the kernel.

If it still passes 4/4 here, the runner's environment is fine, and the
gap is specifically in Triton's JIT path producing a binary that differs
from what aot_compile.py produces for the same kernel source.
"""
from __future__ import annotations

import os
import subprocess
import sys


CC_BIN = os.path.expanduser(
    "~/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/hotswap/"
    "transpiler/tools/compare_correctness/compare_correctness"
)
CC_DIR = os.path.dirname(CC_BIN)


def main() -> int:
    if not os.path.exists(CC_BIN):
        print(f"compare_correctness binary not found at {CC_BIN}", file=sys.stderr)
        return 2

    cmd = [CC_BIN, "--recipe=corpus_add_fp32"]
    print(f"[smoke] cwd={CC_DIR}", flush=True)
    print(f"[smoke] env LD_PRELOAD={os.environ.get('LD_PRELOAD', '<unset>')}",
          flush=True)
    print(f"[smoke] env HSA_HOTSWAP_ISA_OVERRIDE="
          f"{os.environ.get('HSA_HOTSWAP_ISA_OVERRIDE', '<unset>')}",
          flush=True)
    print(f"[smoke] env HSA_HOTSWAP_IR_RAISER="
          f"{os.environ.get('HSA_HOTSWAP_IR_RAISER', '<unset>')}",
          flush=True)
    print(f"[smoke] env HIP_VISIBLE_DEVICES="
          f"{os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}",
          flush=True)
    print(f"[smoke] cmd: {' '.join(cmd)}", flush=True)

    proc = subprocess.run(cmd, cwd=CC_DIR)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
