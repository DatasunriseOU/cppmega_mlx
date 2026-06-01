"""Backward-memory breakdown profiler for the seq=4096 train step.

Goal: validate WHY the full fwd+bwd+AdamW step bursts far above the
forward-only footprint, and whether per-layer mx.checkpoint actually reduces
the backward peak in this MLX build.

This builds a reduced-depth HybridTinyLM (real block types: attention / moe /
mamba3 / m2rnn) so it fits on a Mac, and measures peak memory for:

  A) forward-only (no grad)
  B) full value_and_grad + AdamW update, grad_checkpoint=OFF
  C) full value_and_grad + AdamW update, grad_checkpoint=ON (per-layer)
  D) same as C but with split eval (eval grads, free, then eval optimizer)

Run:  .venv/bin/python scripts/profile_bwd_mem_20260601.py --depth 4 --seq 1024
"""

from __future__ import annotations

import argparse
import gc

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.models.hybrid_lm import HybridTinyLM
from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile
from cppmega_mlx.training.loss import next_token_cut_cross_entropy
from cppmega_mlx.training.optimizers import make_adamw


def gb(x: int | None) -> float:
    return (x or 0) / (1024 ** 3)


def reset_peak() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    gc.collect()


def peak() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return 0


def build_model(depth: int, grad_checkpoint: bool) -> HybridTinyLM:
    profile = local_gb10_quarter_profile()
    # Take the leading `depth` symbols of the real pattern so we get a mix of
    # attention/moe/mamba3 blocks at full hidden width.
    pattern = profile.pattern[:depth]
    n_attn = pattern.count("A")
    dsa_ranks = tuple(range(n_attn))
    cfg = profile.hybrid_config(
        pattern=pattern,
        depth=len(pattern),
        dsa_a_layer_ranks=dsa_ranks,
        grad_checkpoint=grad_checkpoint,
    )
    model = HybridTinyLM(cfg, dtype=mx.bfloat16)
    mx.eval(model.parameters())
    return model


def make_batch(bs: int, seq: int) -> dict:
    vocab = 65536
    tokens = mx.random.randint(0, vocab, (bs, seq + 1))
    return {"tokens": tokens}


def run_forward_only(model, batch) -> int:
    reset_peak()
    loss, ntok = next_token_cut_cross_entropy(model, batch, eval_chunks=False)
    mx.eval(loss, ntok)
    return peak()


def run_full_step(model, batch, *, split_eval: bool) -> tuple[int, float]:
    optimizer = make_adamw(learning_rate=1e-4, weight_decay=0.0)
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)
    reset_peak()

    def loss_fn(m, b):
        return next_token_cut_cross_entropy(m, b, eval_chunks=False)

    lag = nn.value_and_grad(model, loss_fn)
    (loss, ntok), grads = lag(model, batch)
    if split_eval:
        mx.eval(loss, ntok, grads)
        peak_after_bwd = peak()
        optimizer.update(model, grads)
        del grads
        mx.eval(model.parameters(), optimizer.state)
        return max(peak_after_bwd, peak()), float(loss.item())
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss, ntok)
    return peak(), float(loss.item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=1)
    args = ap.parse_args()

    print(f"mlx={mx.__version__} depth={args.depth} seq={args.seq} bs={args.bs}")

    # A) forward-only, no checkpoint
    m = build_model(args.depth, grad_checkpoint=False)
    b = make_batch(args.bs, args.seq)
    fwd = run_forward_only(m, b)
    print(f"A) forward-only            peak={gb(fwd):7.3f} GB")
    del m
    reset_peak()

    # B) full step, checkpoint OFF
    m = build_model(args.depth, grad_checkpoint=False)
    p_b, loss_b = run_full_step(m, b, split_eval=False)
    print(f"B) full step ckpt=OFF      peak={gb(p_b):7.3f} GB  loss={loss_b:.4f}")
    del m
    reset_peak()

    # C) full step, checkpoint ON
    m = build_model(args.depth, grad_checkpoint=True)
    p_c, loss_c = run_full_step(m, b, split_eval=False)
    print(f"C) full step ckpt=ON       peak={gb(p_c):7.3f} GB  loss={loss_c:.4f}")
    del m
    reset_peak()

    # D) full step, checkpoint ON + split eval
    m = build_model(args.depth, grad_checkpoint=True)
    p_d, loss_d = run_full_step(m, b, split_eval=True)
    print(f"D) full step ckpt=ON+split peak={gb(p_d):7.3f} GB  loss={loss_d:.4f}")

    print()
    print(f"ckpt savings (B->C): {gb(p_b - p_c):+.3f} GB "
          f"({100*(p_b-p_c)/max(p_b,1):.1f}%)")
    print(f"split savings (C->D): {gb(p_c - p_d):+.3f} GB "
          f"({100*(p_c-p_d)/max(p_c,1):.1f}%)")


if __name__ == "__main__":
    main()
