"""PR-2 seq=4096 memory probe (FUSED-PIPELINE-ROADMAP §4/§5).

Measure the training-step peak memory of a real-sized hybrid LM at
seq=4096 with grad accumulation OFF vs ON, via mx.get_peak_memory and the
OS ``free``-style RSS. Used on gb10 to demonstrate the 116 GB -> 30-40 GB drop.

Usage:
  python pr2_seq4096_memory_probe.py --seq 4096 --batch 4 --grad-accum 8
  python pr2_seq4096_memory_probe.py --seq 4096 --batch 1 --grad-accum 1   # baseline
"""

from __future__ import annotations

import argparse
import gc
import time

import mlx.core as mx

from cppmega_mlx.data.batch import synthetic_token_batch
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM
from cppmega_mlx.training.loop import one_step_train
from cppmega_mlx.training.optimizers import make_adamw
from cppmega_mlx.training.optimizers_quantized import make_adam8bit


def _gib(n: int | None) -> float:
    return (n or 0) / (1024 ** 3)


def build_model(hidden: int, depth: int, vocab: int, seq: int, grad_ckpt: bool,
                pattern: str = "AEMR") -> HybridTinyLM:
    cfg = HybridTinyConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        pattern=pattern,
        depth=depth,
        num_attention_heads=max(1, hidden // 64),
        max_seq_length=seq,
        moe_num_experts=8,
        moe_top_k=2,
        moe_expert_hidden_size=hidden * 2,
        grad_checkpoint=grad_ckpt,
    )
    model = HybridTinyLM(config=cfg)
    mx.eval(model.parameters())
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--clear-cache", action="store_true")
    ap.add_argument("--opt", choices=["adamw", "adam8bit"], default="adamw")
    ap.add_argument("--pattern", type=str, default="AEMR")
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()

    mx.random.seed(0)
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    model = build_model(args.hidden, args.depth, args.vocab, args.seq, args.grad_ckpt, args.pattern)
    nparams = sum(v.size for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters()))
    print(f"after-model-build peak={_gib(mx.get_peak_memory() if hasattr(mx,'get_peak_memory') else None):.2f}GiB")
    opt = make_adam8bit(learning_rate=1e-4) if args.opt == "adam8bit" else make_adamw(learning_rate=1e-4)
    batch = synthetic_token_batch(batch_size=args.batch, seq_length=args.seq, vocab_size=args.vocab)

    print(f"config: seq={args.seq} batch={args.batch} grad_accum={args.grad_accum} "
          f"hidden={args.hidden} depth={args.depth} grad_ckpt={args.grad_ckpt} "
          f"opt={args.opt} clear_cache={args.clear_cache} params={nparams/1e6:.1f}M")

    for step in range(args.steps):
        t0 = time.perf_counter()
        r = one_step_train(
            model, opt, batch,
            grad_accum_steps=args.grad_accum,
            clear_cache=args.clear_cache,
        )
        peak = mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else None
        active = mx.get_active_memory() if hasattr(mx, "get_active_memory") else None
        print(f"step {step}: loss={r.loss:.5f} ntok={r.ntokens} "
              f"tok/s={r.tokens_per_second:.1f} "
              f"peak={_gib(peak):.2f}GiB active={_gib(active):.2f}GiB "
              f"wall={time.perf_counter()-t0:.2f}s")
        gc.collect()

    peak = mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else None
    print(f"FINAL peak_memory={_gib(peak):.2f}GiB")


if __name__ == "__main__":
    main()
