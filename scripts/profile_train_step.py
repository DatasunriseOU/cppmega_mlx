#!/usr/bin/env python3
"""Throughput/peak-memory profiler for the Stage-1 4x4096 bf16 train step.

REAL run (project RULE #1): builds the real ``DenseCppLM`` at the Stage-1 profile
and streams REAL ``clang_semantic_4k_v10`` rows through the SAME compiled
train-step construction used by ``train_eval_stage1.py`` (forward + backward +
AdamW update, single mx.eval boundary/step). It measures steady-state steps/s
(warmup excluded) and ``mx.get_peak_memory()`` for a chosen knob combination.

It does NOT checkpoint, eval, or run the compile probe; it isolates the train
step so we can compare configs cleanly. No mocks, no fabricated numbers.

Knobs (each independently togglable so we can run the matrix):
  --grad-checkpoint / (default off)
  --chunked-ce      / (default off)
  --clear-cache-every N  (0 = never call mx.clear_cache; default 0)
  --no-compile
  --no-ngram        (disable the optional ngram_hash side feature)

Usage example (one matrix cell):
  .venv/bin/python scripts/profile_train_step.py --bf16 --chunked-ce \
      --warmup 6 --measure 20 --batch 4 --seq-len 4096
"""

from __future__ import annotations

import argparse
import glob
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from scripts.train_eval_stage1 import (
    CHANNELS,
    DATA_GLOB,
    _batches,
    _iter_rows,
)


def _peak_gb() -> float:
    return float(mx.get_peak_memory()) / 1e9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=0, help="alias unused; see --measure")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--measure", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=1280)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--ffn", type=int, default=3456)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--chunked-ce", action="store_true")
    ap.add_argument("--ce-chunk-size", type=int, default=4096)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument(
        "--no-ngram",
        action="store_true",
        help="disable the optional ngram_hash side feature for this run",
    )
    ap.add_argument(
        "--clear-cache-every",
        type=int,
        default=0,
        help="call mx.clear_cache() every N steps (0 = never)",
    )
    args = ap.parse_args()

    all_shards = sorted(glob.glob(DATA_GLOB))
    if len(all_shards) < 2:
        raise FileNotFoundError(f"need >=2 shards; matched {len(all_shards)}")
    train_shards = all_shards[:-1]

    cfg = DenseCppLMConfig(
        vocab_size=65536,
        hidden_size=args.hidden,
        depth=args.depth,
        ffn_hidden_size=args.ffn,
        num_query_heads=20,
        num_kv_heads=4,
        head_dim=64,
        max_seq_length=max(4096, args.seq_len),
        grad_checkpoint=args.grad_checkpoint,
        chunked_ce=args.chunked_ce,
        ce_chunk_size=args.ce_chunk_size,
        ngram_hash_enabled=not args.no_ngram,
    )
    dtype = mx.bfloat16 if args.bf16 else mx.float32
    model = DenseCppLM(cfg, dtype=dtype if args.bf16 else None)
    nparams = model.num_parameters()

    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))

    def loss_fn(model, input_ids, targets, loss_mask, side):
        _, loss = model(input_ids, targets=targets, loss_mask=loss_mask, **side)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    def _step(input_ids, targets, loss_mask, side_vals):
        side = {dst: side_vals[i] for i, (_src, dst) in enumerate(CHANNELS)}
        loss, grads = loss_and_grad(model, input_ids, targets, loss_mask, side)
        grads, gnorm = optim.clip_grad_norm(grads, args.grad_clip)
        opt.update(model, grads)
        return loss, gnorm

    step_fn = _step if args.no_compile else mx.compile(_step, inputs=state, outputs=state)

    row_iter = _iter_rows(train_shards, args.seq_len, args.seed)
    batch_iter = _batches(row_iter, args.batch, args.seq_len)

    cfg_str = (
        f"bf16={args.bf16} grad_ckpt={args.grad_checkpoint} chunked_ce={args.chunked_ce} "
        f"compile={not args.no_compile} ngram={not args.no_ngram} "
        f"clear_cache_every={args.clear_cache_every} B={args.batch} S={args.seq_len}"
    )
    print(f"[profile] params={nparams/1e6:.1f}M  {cfg_str}", flush=True)

    # Warmup (excluded from timing; triggers compile + buffer pool growth).
    for w in range(args.warmup):
        input_ids, targets, loss_mask, side = next(batch_iter)
        side_vals = tuple(side[dst] for _src, dst in CHANNELS)
        opt.learning_rate = args.lr
        loss, gnorm = step_fn(input_ids, targets, loss_mask, side_vals)
        mx.eval(state, loss, gnorm)
        if args.clear_cache_every and (w + 1) % args.clear_cache_every == 0:
            mx.clear_cache()
    print(f"[profile] warmup done ({args.warmup} steps); resetting peak", flush=True)

    mx.reset_peak_memory()
    t0 = time.time()
    last_loss = float("nan")
    for m in range(args.measure):
        input_ids, targets, loss_mask, side = next(batch_iter)
        side_vals = tuple(side[dst] for _src, dst in CHANNELS)
        opt.learning_rate = args.lr
        loss, gnorm = step_fn(input_ids, targets, loss_mask, side_vals)
        mx.eval(state, loss, gnorm)
        last_loss = float(loss)
        if args.clear_cache_every and (m + 1) % args.clear_cache_every == 0:
            mx.clear_cache()
    elapsed = time.time() - t0
    sps = args.measure / elapsed
    spstep = elapsed / args.measure

    print(
        f"[RESULT] {cfg_str}\n"
        f"[RESULT] steps/s={sps:.4f}  s/step={spstep:.3f}  peak_gb={_peak_gb():.2f}  "
        f"last_loss={last_loss:.3f}  measured_steps={args.measure}",
        flush=True,
    )


if __name__ == "__main__":
    main()
