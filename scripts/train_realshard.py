"""Streaming real-shard trainer for the Stage-1 dense C++ foundation LM.

Unlike the fixed-batch smoke test, this trainer STREAMS rows across every
``shard_*.parquet`` in ``clang_semantic_4k_v10`` and yields a FRESH batch of
``B`` rows x ``S`` tokens on every step, so a 1000-step run sees genuinely fresh
data (cycled + shuffled, not one repeated batch).

Each yielded row carries the token-aligned side channels the model consumes:
``token_structure_ids`` -> structure_ids, ``token_dep_levels`` -> dep_levels,
``token_ast_depth`` -> ast_depth_ids, ``token_sibling_index`` ->
sibling_index_ids, ``token_ast_node_type`` -> node_type_ids. There is NO
platform channel in this dataset (per task contract), so platform_ids is not
passed. ``loss_mask`` is ones (every target position contributes).

RULE #1 (fail fast / fail loud): every shape / dtype / bound violation RAISES
with WHERE + WHAT. No silent fallbacks, no silent model shrink.
"""

from __future__ import annotations

import argparse
import glob
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pyarrow.parquet as pq
from mlx.utils import tree_flatten

from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.training.stage1_production import (
    add_stage1_production_arguments,
    run_stage1_graph_domain_production,
)

DATA_GLOB = "/Users/dave/sources/parquet/clang_semantic_4k_v10/shard_*.parquet"

# Token-aligned side channels carried per row -> model kwarg name.
CHANNELS = (
    ("token_structure_ids", "structure_ids"),
    ("token_dep_levels", "dep_levels"),
    ("token_ast_depth", "ast_depth_ids"),
    ("token_sibling_index", "sibling_index_ids"),
    ("token_ast_node_type", "node_type_ids"),
)
TOKEN_COL = "token_ids"
READ_COLS = [TOKEN_COL] + [c for c, _ in CHANNELS]


def _iter_rows(shard_paths: list[str], seq_len: int, seed: int):
    """Infinitely yield single rows (>= seq_len+1 tokens) cycled & shuffled.

    Shards are visited in a shuffled order each epoch; within a shard, rows are
    shuffled. Truncation to ``seq_len`` happens in the batcher; here we only
    filter rows that are long enough to form input[:S] + target[1:S+1].
    """

    need = seq_len + 1
    rng = random.Random(seed)
    epoch = 0
    while True:
        order = list(range(len(shard_paths)))
        rng.shuffle(order)
        for si in order:
            table = pq.read_table(shard_paths[si], columns=READ_COLS)
            cols = {name: table[name].to_pylist() for name in READ_COLS}
            n = len(cols[TOKEN_COL])
            row_order = list(range(n))
            rng.shuffle(row_order)
            for ri in row_order:
                toks = cols[TOKEN_COL][ri]
                if toks is None or len(toks) < need:
                    continue
                row = {"token_ids": toks}
                for src, _dst in CHANNELS:
                    chan = cols[src][ri]
                    if chan is None or len(chan) < need:
                        raise ValueError(
                            f"_iter_rows: row {ri} in {shard_paths[si]} has "
                            f"{TOKEN_COL} len {len(toks)} but {src} len "
                            f"{0 if chan is None else len(chan)} (< {need}); "
                            f"token-aligned channels must match token length"
                        )
                    row[src] = chan
                yield row
        epoch += 1


def _stack(rows: list[list[int]], seq_len: int, offset: int) -> mx.array:
    """Stack a length-S slice (starting at ``offset``) of each row into (B, S)."""

    out = [r[offset : offset + seq_len] for r in rows]
    for i, r in enumerate(out):
        if len(r) != seq_len:
            raise ValueError(
                f"_stack: row {i} slice len {len(r)} != seq_len {seq_len}"
            )
    return mx.array(out, dtype=mx.int32)


def _batches(row_iter, batch: int, seq_len: int):
    """Yield fresh (input_ids, targets, loss_mask, side_channels) every step."""

    while True:
        rows = [next(row_iter) for _ in range(batch)]
        toks = [r["token_ids"] for r in rows]
        input_ids = _stack(toks, seq_len, 0)
        targets = _stack(toks, seq_len, 1)
        side = {}
        for src, dst in CHANNELS:
            side[dst] = _stack([r[src] for r in rows], seq_len, 0)
        loss_mask = mx.ones((batch, seq_len), dtype=mx.float32)
        yield input_ids, targets, loss_mask, side


def _peak_gb() -> float:
    return float(mx.get_peak_memory()) / 1e9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1280)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    add_stage1_production_arguments(ap)
    ap.add_argument(
        "--production-bucket",
        type=int,
        default=None,
        help="immutable bundle sequence-length bucket",
    )
    ap.add_argument(
        "--production-expected-bundle-id",
        default=None,
        help="exact immutable bundle ID expected by the restore receipt",
    )
    ap.add_argument(
        "--production-restore-receipt",
        type=Path,
        default=None,
        help="retained bundle-root restore_receipt.json",
    )
    args = ap.parse_args()

    production_bundle_args = {
        "--production-graph-domain-data": args.production_graph_domain_data,
        "--production-bucket": args.production_bucket,
        "--production-expected-bundle-id": args.production_expected_bundle_id,
        "--production-restore-receipt": args.production_restore_receipt,
    }
    production_mode = any(
        value is not None for value in production_bundle_args.values()
    )
    missing_bundle_args = [
        flag for flag, value in production_bundle_args.items() if value is None
    ]
    if production_mode and missing_bundle_args:
        ap.error(
            "production bundle mode requires explicit CLI provenance for all bundle "
            f"arguments; missing {', '.join(missing_bundle_args)}"
        )
    if production_mode:
        run_stage1_graph_domain_production(
            data_path=args.production_graph_domain_data,
            bucket=args.production_bucket,
            expected_bundle_id=args.production_expected_bundle_id,
            restore_receipt=args.production_restore_receipt,
            steps=args.steps,
            batch_size=args.batch,
            seq_len=args.seq_len,
            hidden_size=args.hidden,
            depth=args.depth,
            ffn_hidden_size=3456,
            learning_rate=3e-4,
            seed=args.seed,
            attention_mode=args.production_attention_mode,
            bf16=args.bf16,
        )
        return

    shard_paths = sorted(glob.glob(DATA_GLOB))
    if not shard_paths:
        raise FileNotFoundError(f"no shards matched {DATA_GLOB}")

    cfg = DenseCppLMConfig(
        vocab_size=65536,
        hidden_size=args.hidden,
        depth=args.depth,
        ffn_hidden_size=3456,
        num_query_heads=20,
        num_kv_heads=4,
        head_dim=64,
        max_seq_length=max(4096, args.seq_len),
    )
    dtype = mx.bfloat16 if args.bf16 else mx.float32
    model = DenseCppLM(cfg, dtype=dtype if args.bf16 else None)
    nparams = model.num_parameters()

    print(
        f"[config] hidden={cfg.hidden_size} depth={cfg.depth} ffn={cfg.ffn_hidden_size} "
        f"qh={cfg.num_query_heads} kvh={cfg.num_kv_heads} head_dim={cfg.head_dim} "
        f"vocab={cfg.vocab_size} dtype={'bf16' if args.bf16 else 'fp32'}",
        flush=True,
    )
    print(
        f"[config] steps={args.steps} batch={args.batch} seq_len={args.seq_len} "
        f"tokens/step={args.batch * args.seq_len}",
        flush=True,
    )
    print(f"[params] {nparams / 1e6:.2f}M", flush=True)

    opt = optim.AdamW(
        learning_rate=3e-4, weight_decay=0.1, betas=(0.9, 0.95)
    )
    warmup = 200
    peak_lr = 3e-4

    def lr_at(step: int) -> float:
        if step < warmup:
            return peak_lr * (step + 1) / warmup
        return peak_lr

    def loss_fn(model, input_ids, targets, loss_mask, side):
        _, loss = model(
            input_ids, targets=targets, loss_mask=loss_mask, **side
        )
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    row_iter = _iter_rows(shard_paths, args.seq_len, args.seed)
    batch_iter = _batches(row_iter, args.batch, args.seq_len)

    mx.reset_peak_memory()
    losses: list[tuple[int, float]] = []
    t0 = time.time()
    first_loss = None
    last_loss = None

    for step in range(args.steps):
        input_ids, targets, loss_mask, side = next(batch_iter)
        opt.learning_rate = lr_at(step)
        loss, grads = loss_and_grad(model, input_ids, targets, loss_mask, side)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        lval = float(loss)
        if first_loss is None:
            first_loss = lval
        last_loss = lval
        if step == 0 or (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed
            losses.append((step + 1, lval))
            print(
                f"[step {step + 1:>4}] loss={lval:.4f} lr={opt.learning_rate.item():.2e} "
                f"peak={_peak_gb():.2f}GB steps/s={sps:.3f}",
                flush=True,
            )

    elapsed = time.time() - t0
    sps = args.steps / elapsed
    peak = _peak_gb()

    print("=" * 70, flush=True)
    print("[SUMMARY]", flush=True)
    print(
        f"  config: hidden={cfg.hidden_size} depth={cfg.depth} ffn={cfg.ffn_hidden_size} "
        f"qh={cfg.num_query_heads} kvh={cfg.num_kv_heads} head_dim={cfg.head_dim} "
        f"vocab={cfg.vocab_size} dtype={'bf16' if args.bf16 else 'fp32'}",
        flush=True,
    )
    print(
        f"  batch={args.batch} seq_len={args.seq_len} tokens/step={args.batch * args.seq_len}",
        flush=True,
    )
    print(f"  params={nparams / 1e6:.2f}M", flush=True)
    print(f"  PEAK_MEMORY_GB={peak:.2f}", flush=True)
    print(f"  steps/s={sps:.3f}", flush=True)
    print(f"  loss_initial={first_loss:.4f} loss_final={last_loss:.4f}", flush=True)
    print(f"  loss_curve={[(s, round(l, 4)) for s, l in losses]}", flush=True)
    print(f"  PEAK_LE_40GB={'YES' if peak <= 40.0 else 'NO'}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
