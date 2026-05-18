#!/usr/bin/env python3
"""V4 1B-scale fwd + bwd smoke on our parquet samples.

Builds a V4 stack via UnifiedSuperblockV4 sized close to ~1B params, loads
N batches from the GB10 clang_semantic parquet, runs fwd + bwd, measures
throughput.

Usage:
    ~/sources/nanochat/.venv/bin/python scripts/v4_1b_parquet_smoke.py \
        --seq-len 512 --batch-size 1 --steps 3 --hidden-size 2048 \
        --layers 12

Output: JSON receipt under reports/v4/v4_1b_parquet_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from cppmega_mlx.data.parquet_dataset import TokenParquetDataset  # noqa: E402

from cppmega_v4.models.unified_superblock_v4 import UnifiedSuperblockV4  # noqa: E402
from cppmega_v4.run_template import BlockSpec, RunTemplate  # noqa: E402


DEFAULT_PARQUET = (
    REPO_ROOT / "data" / "parquet_samples" / "gb10"
    / "clang_semantic_4k_v10" / "val_00000.parquet"
)


def _make_template(hidden_size: int, layers: int, vocab_size: int) -> RunTemplate:
    """Build a V4 template close to ~1B params at hidden=2048, layers=12.

    Real-factories version: includes GDN (Path A→C dispatchable), KDA,
    attention, MoE, NSA, CSA+HCA, Engram, MLP — the full V4 cohort.
    """
    n_heads = max(1, hidden_size // 64)
    h_dim = hidden_size // n_heads
    per_kind = max(1, layers // 6)  # spread layers across 6 attention-like kinds
    blocks = [
        BlockSpec(kind="engram", repeat=1, params={
            "num_ngram_layers": 2, "max_ngram_size": 4,
            "num_embed_table_per_ngram": 4, "embed_dim": 64,
            "embed_table_size": 1024,
        }),
        BlockSpec(kind="gdn", repeat=per_kind, params={
            "num_heads": n_heads, "head_dim": h_dim, "use_short_conv": False,
        }),
        BlockSpec(kind="kda", repeat=per_kind, params={
            "num_heads": n_heads, "head_dim": h_dim, "use_short_conv": False,
        }),
        BlockSpec(kind="attention", repeat=per_kind, params={
            "num_heads": n_heads, "head_dim": h_dim,
        }),
        BlockSpec(kind="nsa", repeat=per_kind, params={
            "num_heads": n_heads, "head_dim": h_dim,
            "compress_block_size": 64, "select_topk": 16,
            "sliding_window": 256,
        }),
        BlockSpec(kind="csa_hca", repeat=per_kind, params={
            "num_heads": n_heads, "head_dim": h_dim,
            "m_csa": 4, "m_hca": 16,
        }),
        BlockSpec(kind="moe", repeat=per_kind, params={
            "num_experts": 8, "top_k": 2,
            "expert_hidden_size": hidden_size * 2,
        }),
        BlockSpec(kind="mlp", repeat=max(1, layers - 6 * per_kind),
                  params={"intermediate_size": hidden_size * 4}),
    ]
    return RunTemplate(
        name=f"v4_smoke_h{hidden_size}_L{layers}",
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        blocks=blocks,
    )


@dataclass
class StepStat:
    step: int
    fwd_ms: float
    bwd_ms: float
    total_ms: float
    loss: float


@dataclass
class RunReceipt:
    template_name: str
    hidden_size: int
    layers: int
    seq_len: int
    batch_size: int
    steps: int
    parquet_path: str
    fwd_median_ms: float
    bwd_median_ms: float
    total_median_ms: float
    tokens_per_sec: float
    per_step: list = field(default_factory=list)


def _build_embed_and_head(vocab_size: int, hidden_size: int):
    embed = nn.Embedding(vocab_size, hidden_size)
    head = nn.Linear(hidden_size, vocab_size, bias=False)
    return embed, head


def _cross_entropy_loss(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    """Sum-of-per-token cross-entropy with mask, divided by mask.sum()."""
    logits_f32 = logits.astype(mx.float32)
    # log_softmax(x) = x - logsumexp(x)
    log_probs = logits_f32 - mx.logsumexp(logits_f32, axis=-1, keepdims=True)
    target_lp = mx.take_along_axis(
        log_probs, targets.astype(mx.int32)[..., None], axis=-1,
    )[..., 0]
    neg_lp = -target_lp * mask.astype(mx.float32)
    denom = mx.maximum(mask.astype(mx.float32).sum(), 1.0)
    return neg_lp.sum() / denom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "v4")
    args = parser.parse_args()

    if not args.parquet.exists():
        print(f"!! parquet not found: {args.parquet}", file=sys.stderr)
        return 1

    template = _make_template(args.hidden_size, args.layers, args.vocab_size)
    sb = UnifiedSuperblockV4(template)
    embed, head = _build_embed_and_head(args.vocab_size, args.hidden_size)

    ds = TokenParquetDataset(
        str(args.parquet), seq_len=args.seq_len, batch_size=args.batch_size,
        token_key="token_ids", loop=True,
    )
    print(f"== V4 1B smoke ==")
    print(f"   template:    {template.name} ({template.total_blocks()} blocks)")
    print(f"   hidden_size: {args.hidden_size}, layers: {args.layers}")
    print(f"   seq_len:     {args.seq_len}, batch_size: {args.batch_size}, steps: {args.steps}")
    print(f"   parquet:     {args.parquet}")
    print(f"   dataset:     {ds.num_samples} samples → {ds.num_batches} batches")

    stats: list[StepStat] = []
    iter_batches = ds.iter_batches()
    for step in range(args.steps):
        batch = next(iter_batches)
        # batch.inputs: [B, T] int32 tokens; batch.targets: [B, T]; batch.target_mask
        token_ids = mx.array(batch.inputs)
        target_ids = mx.array(batch.targets)
        target_mask = mx.array(batch.target_mask)
        # document_ids: structure_ids first column if present, else None.
        doc_ids = None

        def fwd_bwd_step():
            # Build the full computation graph for one step.
            def loss_fn(tok_ids):
                hidden = embed(tok_ids)
                out = sb(tok_ids, hidden, document_ids=doc_ids)
                logits = head(out)
                return _cross_entropy_loss(logits, target_ids, target_mask)
            loss, grad = mx.value_and_grad(loss_fn)(token_ids)
            mx.eval(loss, grad)
            return loss, grad

        # Forward-only timing.
        t0 = time.perf_counter()
        hidden = embed(token_ids)
        out = sb(token_ids, hidden, document_ids=doc_ids)
        logits = head(out)
        loss_only = _cross_entropy_loss(logits, target_ids, target_mask)
        mx.eval(loss_only)
        fwd_ms = (time.perf_counter() - t0) * 1000

        # Fwd+bwd timing.
        t1 = time.perf_counter()
        loss, grad = fwd_bwd_step()
        total_ms = (time.perf_counter() - t1) * 1000
        bwd_ms = max(0.0, total_ms - fwd_ms)

        ss = StepStat(
            step=step, fwd_ms=fwd_ms, bwd_ms=bwd_ms, total_ms=total_ms,
            loss=float(loss.item()),
        )
        stats.append(ss)
        print(f"   step {step}: loss={ss.loss:.4f}  fwd={fwd_ms:.1f}ms  "
              f"total={total_ms:.1f}ms  (bwd≈{bwd_ms:.1f}ms)")

    fwd_med = statistics.median(s.fwd_ms for s in stats)
    bwd_med = statistics.median(s.bwd_ms for s in stats)
    total_med = statistics.median(s.total_ms for s in stats)
    tok_per_step = args.batch_size * args.seq_len
    tok_per_sec = tok_per_step / (total_med / 1000) if total_med > 0 else 0.0

    receipt = RunReceipt(
        template_name=template.name,
        hidden_size=args.hidden_size, layers=args.layers,
        seq_len=args.seq_len, batch_size=args.batch_size, steps=args.steps,
        parquet_path=str(args.parquet),
        fwd_median_ms=fwd_med, bwd_median_ms=bwd_med, total_median_ms=total_med,
        tokens_per_sec=tok_per_sec,
        per_step=[asdict(s) for s in stats],
    )
    args.out.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = args.out / f"v4_parquet_smoke_h{args.hidden_size}_L{args.layers}_{ts}.json"
    out_path.write_text(json.dumps(asdict(receipt), indent=2))
    print()
    print(f"== summary ==")
    print(f"   fwd median:   {fwd_med:8.1f} ms")
    print(f"   bwd median:   {bwd_med:8.1f} ms")
    print(f"   total median: {total_med:8.1f} ms")
    print(f"   tokens/sec:   {tok_per_sec:8.1f}")
    print(f"   receipt:      {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
