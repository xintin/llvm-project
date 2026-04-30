"""Like _dump_aotpath_co.py but also save the .amdgcn assembly so we can
diff against the JIT version and confirm whether AOT uses scalar
global_load (no buffer descriptors) vs JIT's buffer_load_b128 path."""
from __future__ import annotations

import os
import sys

import triton
from triton.compiler.compiler import (
    ASTSource,
    GPUTarget,
    compile as triton_compile,
)

EXTRACTED_DIR = os.path.expanduser(
    "~/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/hotswap/"
    "transpiler/tools/compare_correctness/kernels/triton/_corpus/extracted"
)
sys.path.insert(0, EXTRACTED_DIR)
from add_kernel import add_kernel  # noqa: E402


def main() -> int:
    print(f"[dump-aot] triton {triton.__version__}", flush=True)

    target = GPUTarget("hip", "gfx1250", 32)
    src = ASTSource(
        fn=add_kernel,
        signature={
            "x_ptr":      "*fp32",
            "y_ptr":      "*fp32",
            "output_ptr": "*fp32",
            "n_elements": "i32",
        },
        constexprs={"BLOCK_SIZE": 1024},
        attrs={"num_warps": 4},
    )
    compiled = triton_compile(src, target=target)

    for asm_kind, ext in (("ttir", "ttir"), ("amdgcn", "amdgcn"),
                         ("hsaco", "hsaco")):
        blob = compiled.asm.get(asm_kind)
        if blob is None:
            continue
        out = f"/tmp/aotpath_addk_gfx1250.{ext}"
        if isinstance(blob, str) and os.path.exists(blob):
            data = open(blob, "rb").read()
        elif isinstance(blob, (bytes, bytearray)):
            data = bytes(blob)
        else:
            data = (blob if isinstance(blob, str) else str(blob)).encode("utf-8")
        with open(out, "wb") as f:
            f.write(data)
        print(f"[dump-aot] wrote {out} ({len(data)} bytes)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
