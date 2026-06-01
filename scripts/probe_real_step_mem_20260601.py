"""Real-loss single fwd+bwd(+optimizer) memory probe for the 1B model.

Unlike probe_moe_forward_mem.py (naive logits.mean() loss that trips the SDPA
mask VJP), this uses the production CCE training loss and the real AdamW step,
and reports BOTH the MLX peak and the torch-CUDA peak (the Path-B mamba3 bwd
scratch lives in the torch allocator, invisible to mx.get_peak_memory). It also
samples /proc/self/status VmRSS so the true unified-memory footprint is visible.

Usage:
  python scripts/probe_real_step_mem_20260601.py --batch 1 --seq 4096 [--optimizer]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from cppmega_mlx.recipes.model_factory import local_gb10_quarter
from cppmega_mlx.training.loss import next_token_cut_cross_entropy


def _peak_gb() -> float:
    fn = getattr(mx, "get_peak_memory", None)
    return float(fn()) / (1024**3) if fn else float("nan")


def _reset_peak() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024**2)  # kB -> GB
    except Exception:
        pass
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=65_536)
    ap.add_argument("--optimizer", action="store_true", help="also run AdamW update")
    ap.add_argument("--adam8bit", action="store_true", help="use int8 AdamW state")
    ap.add_argument(
        "--split-eval",
        dest="split_eval",
        action="store_true",
        help="eval grads + free bwd graph before the optimizer update",
    )
    ap.add_argument(
        "--grad-checkpoint",
        dest="grad_checkpoint",
        action="store_true",
        default=False,
        help="enable per-layer mx.checkpoint (default: off)",
    )
    args = ap.parse_args()

    try:
        import torch

        have_torch = torch.cuda.is_available()
    except Exception:
        torch = None
        have_torch = False

    # Optional MLX memory-limit cap (GB). Caps the MLX cache pool so the unified
    # footprint tracks the live high-water instead of MLX's reservation
    # headroom.
    cap_gb = os.environ.get("CPPMEGA_MLX_MEMORY_LIMIT_GB", "").strip()
    if cap_gb and hasattr(mx, "set_memory_limit"):
        mx.set_memory_limit(int(float(cap_gb) * (1024**3)))

    t0 = time.time()
    model = local_gb10_quarter(dtype=mx.bfloat16, grad_checkpoint=args.grad_checkpoint)
    mx.eval(model.parameters())
    mx.synchronize()
    build_s = time.time() - t0
    after_params_gb = _peak_gb()

    rng = np.random.RandomState(0)
    tokens = mx.array(
        rng.randint(0, args.vocab, size=(args.batch, args.seq + 1)).astype(np.int32)
    )
    batch = {"tokens": tokens}

    optimizer = None
    if args.optimizer:
        if args.adam8bit:
            from cppmega_mlx.training.optimizers import make_adam8bit

            optimizer = make_adam8bit(
                learning_rate=1e-4,
                weight_decay=0.0,
                quant_scheme="dynamic_int8_v1",
                min_8bit_size=4096,
            )
        else:
            from cppmega_mlx.training.optimizers import make_adamw

            optimizer = make_adamw(learning_rate=1e-4, weight_decay=0.0)
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)

    if have_torch:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _reset_peak()

    def loss_fn(m, b):
        return next_token_cut_cross_entropy(m, b, eval_chunks=False)

    t1 = time.time()
    (loss, ntok), grads = nn.value_and_grad(model, loss_fn)(model, batch)
    if optimizer is not None:
        if args.split_eval:
            # Materialise loss+grads and free the backward graph BEFORE the
            # optimizer update, so the bwd activations and the optimizer-update
            # working set never coexist in a single eval. Peak becomes
            # max(bwd, update) instead of bwd+update.
            mx.eval(loss, ntok, grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss, ntok)
    else:
        mx.eval(loss, ntok, grads)
    mx.synchronize()
    run_s = time.time() - t1

    peak_gb = _peak_gb()
    torch_peak_gb = (
        round(float(torch.cuda.max_memory_allocated()) / (1024**3), 3)
        if have_torch
        else None
    )
    result = {
        "efficient_moe": os.environ.get("CPPMEGA_MOE_EFFICIENT"),
        "mamba3_bwd_seq_chunk": os.environ.get("CPPMEGA_MAMBA3_BWD_SEQ_CHUNK") or None,
        "batch": args.batch,
        "seq": args.seq,
        "grad_checkpoint": bool(args.grad_checkpoint),
        "optimizer": bool(args.optimizer),
        "build_s": round(build_s, 2),
        "run_s": round(run_s, 2),
        "after_params_peak_gb": round(after_params_gb, 3),
        "mlx_peak_gb": round(peak_gb, 3),
        "torch_cuda_peak_gb": torch_peak_gb,
        "rss_gb": round(_rss_gb(), 3),
        "loss": float(loss),
    }
    print("REALPROBE_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
