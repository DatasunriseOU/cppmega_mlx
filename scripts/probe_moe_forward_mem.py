#!/usr/bin/env python3
"""Forward-only peak-memory probe for the 1B local_gb10_quarter model.

Builds the real model (bf16, grad-checkpoint) and runs a SINGLE forward pass at
the requested batch/seq, reporting MLX peak memory. Used to measure whether the
env-gated efficient MoE (CPPMEGA_MOE_EFFICIENT=1) brings seq=4096 under the OOM
wall before any backward/optimizer step is attempted.

Usage:
  python scripts/probe_moe_forward_mem.py --batch 1 --seq 4096 [--backward]

Set CPPMEGA_MOE_EFFICIENT=1 in the env to exercise the sparse-gather MoE.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np

from cppmega_mlx.recipes.model_factory import local_gb10_quarter


def _peak_gb() -> float:
    fn = getattr(mx, "get_peak_memory", None)
    if fn is None:
        metal = getattr(mx, "metal", None)
        fn = getattr(metal, "get_peak_memory", None) if metal else None
    return float(fn()) / (1024**3) if fn else float("nan")


def _reset_peak() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:
        metal = getattr(mx, "metal", None)
        if metal and hasattr(metal, "reset_peak_memory"):
            metal.reset_peak_memory()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--backward", action="store_true", help="also run one backward")
    ap.add_argument("--vocab", type=int, default=65_536)
    args = ap.parse_args()

    efficient = os.environ.get("CPPMEGA_MOE_EFFICIENT", "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }
    t0 = time.time()
    model = local_gb10_quarter(dtype=mx.bfloat16, grad_checkpoint=True)
    mx.eval(model.parameters())
    mx.synchronize()
    build_s = time.time() - t0
    after_params_gb = _peak_gb()

    rng = np.random.RandomState(0)
    ids = mx.array(rng.randint(0, args.vocab, size=(args.batch, args.seq)).astype(np.int32))

    _reset_peak()
    if args.backward:
        import mlx.nn as nn

        def loss_fn(m, x):
            out = m(x)
            logits = out.logits if hasattr(out, "logits") else out
            return logits.astype(mx.float32).mean()

        lg = nn.value_and_grad(model, loss_fn)
        t1 = time.time()
        l, g = lg(model, ids)
        mx.eval(l, g)
        mx.synchronize()
        run_s = time.time() - t1
        loss_val = float(l)
    else:
        t1 = time.time()
        out = model(ids)
        logits = out.logits if hasattr(out, "logits") else out
        mx.eval(logits)
        mx.synchronize()
        run_s = time.time() - t1
        loss_val = float(logits.astype(mx.float32).mean())

    peak_gb = _peak_gb()
    result = {
        "efficient_moe": efficient,
        "batch": args.batch,
        "seq": args.seq,
        "backward": bool(args.backward),
        "build_s": round(build_s, 2),
        "run_s": round(run_s, 2),
        "after_params_peak_gb": round(after_params_gb, 3),
        "peak_gb": round(peak_gb, 3),
        "loss_or_mean": loss_val,
    }
    print("PROBE_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
