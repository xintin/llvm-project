#!/usr/bin/env python3
"""Run GPT-OSS's exact attention operator and verify correctness.

This script is meant for `triton_corpus_runner`:

  python runner.py --script ./gpt_oss_attention_operator.py

It imports the real GPT-OSS Triton attention entrypoint
(`gpt_oss.triton.attention.attention`) and compares the output against
the reference implementation in the same module (`attention_ref`).

No fallback behavior is used: if GPT-OSS sources are unavailable or the
operator diverges from reference, the script exits non-zero with a clear
error.
"""

from __future__ import annotations

import os
import sys

import torch


def _load_gpt_oss_attention():
    gpt_oss_src = os.environ.get("GPT_OSS_SRC", "/data/gpt-oss/src")
    if not os.path.isdir(gpt_oss_src):
        raise FileNotFoundError(
            f"GPT_OSS_SRC does not exist: {gpt_oss_src}\n"
            "Set GPT_OSS_SRC to the GPT-OSS source tree containing "
            "gpt_oss/triton/attention.py."
        )
    if gpt_oss_src not in sys.path:
        sys.path.insert(0, gpt_oss_src)

    from gpt_oss.triton.attention import attention, attention_ref

    return attention, attention_ref


def _run_case(
    *,
    batch: int,
    seq: int,
    n_kv_heads: int,
    gqa: int,
    head_dim: int,
    sliding_window: int,
) -> None:
    attention, attention_ref = _load_gpt_oss_attention()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    q = torch.randn(batch, seq, n_kv_heads, gqa, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, n_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq, n_kv_heads, head_dim, device=device, dtype=dtype)
    sinks = torch.randn(n_kv_heads * gqa, device=device, dtype=dtype)
    sm_scale = 1.0 / (head_dim ** 0.5)
    start_q = torch.zeros((1,), dtype=torch.long, device=device)

    out = attention(q, k, v, sinks, sm_scale, sliding_window, start_q)
    ref = attention_ref(
        q,
        k,
        v,
        sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        start_q=start_q,
    )

    # Keep the correctness bar strict: this is a direct implementation-vs-ref
    # check from GPT-OSS's own attention module.
    torch.testing.assert_close(out, ref)
    torch.cuda.synchronize()


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA/HIP device visible to torch.")

    # GPT-OSS 20B-style head layout.
    _run_case(
        batch=1,
        seq=16,
        n_kv_heads=8,
        gqa=8,
        head_dim=64,
        sliding_window=128,
    )

    # GPT-OSS 120B-style head layout and longer context.
    _run_case(
        batch=1,
        seq=128,
        n_kv_heads=8,
        gqa=8,
        head_dim=64,
        sliding_window=128,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
