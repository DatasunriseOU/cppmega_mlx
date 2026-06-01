"""Numeric-equivalence check: Mamba3 seq-chunked backward vs monolithic.

RULE #1 guard for CPPMEGA_MAMBA3_BWD_SEQ_CHUNK: the chunked backward must give
the same loss + gradients as the unchunked backward (within fp32 reassociation
tolerance) on the real HybridTinyLM training loss.

Run: .venv/bin/python scripts/verify_mamba3_chunk_equiv_20260601.py
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.models.hybrid_lm import HybridTinyLM
from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile
from cppmega_mlx.training.loss import next_token_cut_cross_entropy


def build(depth: int) -> HybridTinyLM:
    p = local_gb10_quarter_profile()
    pat = p.pattern[:depth]
    cfg = p.hybrid_config(
        pattern=pat,
        depth=len(pat),
        dsa_a_layer_ranks=tuple(range(pat.count("A"))),
        grad_checkpoint=False,
    )
    m = HybridTinyLM(cfg, dtype=mx.float32)  # fp32 weights for a tight tolerance
    mx.eval(m.parameters())
    return m


def grads_for(model, batch, chunk: int | None):
    if chunk is None:
        os.environ.pop("CPPMEGA_MAMBA3_BWD_SEQ_CHUNK", None)
    else:
        os.environ["CPPMEGA_MAMBA3_BWD_SEQ_CHUNK"] = str(chunk)

    def loss_fn(m, b):
        return next_token_cut_cross_entropy(m, b, eval_chunks=False)

    (loss, ntok), grads = nn.value_and_grad(model, loss_fn)(model, batch)
    mx.eval(loss, ntok, grads)
    return float(loss.item()), grads


def flat(grads, prefix=""):
    out = {}
    if isinstance(grads, dict):
        for k, v in grads.items():
            out.update(flat(v, f"{prefix}{k}."))
    elif isinstance(grads, (list, tuple)):
        for i, v in enumerate(grads):
            out.update(flat(v, f"{prefix}{i}."))
    elif isinstance(grads, mx.array):
        out[prefix] = grads
    return out


def main() -> None:
    # Pattern must include at least one mamba3 (M) layer; use depth 4 -> AEME.
    depth = 4
    seq = 256
    model = build(depth)
    tokens = mx.random.randint(0, 65536, (1, seq + 1))
    batch = {"tokens": tokens}

    loss_ref, g_ref = grads_for(model, batch, None)
    fr = flat(g_ref)

    print(f"depth={depth} seq={seq} loss_ref={loss_ref:.6f} nparams={len(fr)}")
    ok_all = True
    for chunk in (64, 96, 128):
        loss_c, g_c = grads_for(model, batch, chunk)
        fc = flat(g_c)
        max_abs = 0.0
        max_rel = 0.0
        worst = ""
        for k in fr:
            a = fr[k].astype(mx.float32)
            b = fc[k].astype(mx.float32)
            ae = float(mx.max(mx.abs(a - b)).item())
            denom = float(mx.max(mx.abs(a)).item()) + 1e-8
            re = ae / denom
            if ae > max_abs:
                max_abs = ae
            if re > max_rel:
                max_rel = re
                worst = k
        loss_dl = abs(loss_c - loss_ref)
        ok = (max_rel < 1e-3) and (loss_dl < 1e-4)
        ok_all = ok_all and ok
        print(
            f"  chunk={chunk:4} loss={loss_c:.6f} dloss={loss_dl:.2e} "
            f"max_abs={max_abs:.2e} max_rel={max_rel:.2e} worst={worst} "
            f"{'OK' if ok else 'FAIL'}"
        )

    print("RESULT:", "PASS" if ok_all else "FAIL")
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
