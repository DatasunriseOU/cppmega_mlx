"""SERIAL reference: full eager model fwd+bwd (mamba = serial scan) -> loss + 163 grads.

The flag-OFF descriptor-chain route is XPC-blocked in this environment on the
giant fused ``chain_12_13`` Metal kernel (the SERIAL mamba+residual mega-kernel
the chunked flag-ON path REPLACES). This harness produces the EQUIVALENT serial
ground truth WITHOUT that kernel: it runs the model's own eager forward (mamba on
the REFERENCE pure-MLX serial scan) + the SAME suffix loss
(``CE(norm(final_hidden), targets)``, mean over masked tokens) the route uses, and
takes full-model ``nn.value_and_grad``. The flag-ON route's 163 grads + loss must
match these (per-grad < 1e-3, loss within fp16-chunked tolerance).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=64)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

    # Match the route's prefix/eager mamba numerics: the direct-chain route's
    # prefix VJP (layers 0..start-1, incl. mamba layers 2/6) runs the model's
    # eager forward on the REFERENCE serial scan as well (the descriptor chunked
    # path only covers the in-region mamba at layer 10). Force REFERENCE so the
    # PREFIX mamba layers match the serial ground truth exactly.
    os.environ["CPPMEGA_KERNEL_PATH__MAMBA3_MIMO"] = "ref"

    from cppmega_mlx.recipes.model_factory import (
        build_local_gb10_quarter_tiny_smoke_model,
    )
    from cppmega_mlx.training.cut_cross_entropy import linear_cross_entropy

    mx.random.seed(0)
    model = build_local_gb10_quarter_tiny_smoke_model(
        hidden_size=64,
        num_attention_heads=1,
        mamba_expand=1,
        mamba_head_dim=64,
        mamba_state_dim=16,
        mamba_groups=1,
        mamba_chunk_size=64,
    )
    seq = int(args.seq)
    vocab = int(getattr(model, "vocab_size", 0) or 1024)
    ids = mx.array(np.random.randint(0, max(2, vocab), size=(1, seq)).astype(np.int32))
    tgt = mx.array(np.random.randint(0, max(2, vocab), size=(1, seq)).astype(np.int32))

    norm = model.norm
    norm_w = norm.weight
    head_w = model.lm_head.weight
    norm_eps = float(getattr(norm, "eps", 1e-5))
    mask = mx.ones(tgt.shape, dtype=mx.float32)

    def loss_fn(m):
        hidden = m.decoder_hidden_states(ids, apply_final_norm=False)
        inv_rms = mx.rsqrt(
            mx.mean(hidden * hidden, axis=-1, keepdims=True)
            + mx.array(norm_eps, dtype=hidden.dtype)
        )
        normed = hidden * inv_rms * norm_w
        token_losses = linear_cross_entropy(
            normed, head_w, tgt, reduction="none", chunk_rows=128, eval_chunks=False
        )
        ntokens = mask.sum()
        denom = mx.maximum(ntokens, mx.array(1.0, dtype=mx.float32))
        return (token_losses * mask).astype(mx.float32).sum() / denom

    t0 = time.perf_counter()
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    flat = [(str(n), v) for n, v in tree_flatten(grads) if isinstance(v, mx.array)]
    mx.eval(loss, *(v for _, v in flat))
    elapsed = time.perf_counter() - t0

    grad_dump = {}
    for n, v in flat:
        a = np.asarray(v.astype(mx.float32))
        grad_dump[n] = {
            "shape": list(a.shape),
            "absmax": float(np.abs(a).max()) if a.size else 0.0,
            "sum": float(a.sum()),
            "l2": float(np.sqrt((a.astype(np.float64) ** 2).sum())),
            "data": a.reshape(-1).tolist() if a.size <= 200000 else None,
        }

    receipt = {
        "mode": "serial_eager_reference",
        "seq": seq,
        "loss": float(loss.item()),
        "loss_finite": bool(mx.isfinite(loss).item()),
        "grad_count": len(flat),
        "elapsed_total_s": elapsed,
        "grads": grad_dump,
    }
    with open(args.out, "w") as f:
        json.dump(receipt, f)
    print(
        f"[serial-ref] loss={receipt['loss']:.6e} grads={len(flat)} "
        f"elapsed={elapsed:.3f}s wrote {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
