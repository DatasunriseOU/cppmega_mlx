"""Stage-1 dense ~500M GQA C++ LM: real local streaming train + eval + compile probe.

This is a REAL local run (no mocks, no fabricated metrics — project RULE #1):

* Streaming bf16 training over every ``shard_*.parquet`` in
  ``clang_semantic_4k_v10`` EXCEPT a fixed held-out validation shard (the last
  shard) which is NEVER trained on. Fresh (B, S) batch every step.
* AdamW (lr 3e-4, wd 0.1, betas 0.9/0.95), grad-clip 1.0, linear warmup then
  cosine decay to 10% of peak. bf16 numerics are finite-checked every step and
  RAISE on NaN/Inf (fail-loud, no silent skip).
* mx peak memory tracked and logged.
* Checkpoint model + optimizer state every ``--ckpt-every`` (default 1000) steps
  to ``outputs/stage1_ckpts/``.
* EVAL every ``--eval-every`` (default 250) steps:
    (a) val loss + PERPLEXITY (exp(mean CE)) over a fixed held-out row set;
    (b) COMPILE PROBE: take K val prefixes, greedy/temperature-decode ~256
        tokens, decode to C++ text via the cppmega tokenizer, write each to a
        temp .cpp, run CodeVerifier.syntax_check (clang++ -fsyntax-only
        -std=c++17), and record the syntax-valid pass-rate + sample diagnostics.
  One log line per eval: step, train_loss, val_loss, val_ppl,
  compile_pass_rate, peak_gb.

All output goes to ``outputs/train_eval_stage1.log`` (and stdout).

Reuses: DenseCppLM (cppmega_mlx/models/dense_cpp_lm.py), the streaming loader
pattern from scripts/train_realshard.py, CodeVerifier.syntax_check
(cppmega_mlx/runtime/code_verifier.py), and the cppmega tokenizer.
"""

from __future__ import annotations

import argparse
import glob
import math
import random
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pyarrow.parquet as pq
from mlx.utils import tree_flatten

from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.runtime.code_verifier import CodeVerifier
from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
from cppmega_mlx.training.stage1_production import (
    add_stage1_production_arguments,
    run_stage1_graph_domain_production,
)

DATA_GLOB = "/Users/dave/sources/parquet/clang_semantic_4k_v10/shard_*.parquet"
OUT_DIR = Path("/Volumes/external/sources/cppmega.mlx/outputs")
CKPT_DIR = OUT_DIR / "stage1_ckpts"
LOG_PATH = OUT_DIR / "train_eval_stage1.log"
TOKENIZER_PATH = Path(
    "/Volumes/external/sources/cppmega.mlx/cppmega_mlx/tokenizer/tokenizer.json"
)

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

_LOG_FH = None


def log(msg: str) -> None:
    """Write a line to stdout AND the persistent log file."""
    global _LOG_FH
    print(msg, flush=True)
    if _LOG_FH is None:
        _LOG_FH = LOG_PATH.open("a", encoding="utf-8")
    _LOG_FH.write(msg + "\n")
    _LOG_FH.flush()


def _check_finite(name: str, value: float, step: int) -> None:
    if not math.isfinite(value):
        raise FloatingPointError(
            f"[train_eval_stage1] non-finite {name}={value} at step {step}; "
            f"bf16 numerics diverged (fail-loud, refusing to continue)"
        )


# --------------------------------------------------------------------------- #
def _iter_rows(shard_paths: list[str], seq_len: int, seed: int):
    """Infinitely yield single rows (>= seq_len+1 tokens) cycled & shuffled."""
    need = seq_len + 1
    rng = random.Random(seed)
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
                ok = True
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
                if ok:
                    yield row


def _stack(rows: list[list[int]], seq_len: int, offset: int) -> mx.array:
    out = [r[offset : offset + seq_len] for r in rows]
    for i, r in enumerate(out):
        if len(r) != seq_len:
            raise ValueError(f"_stack: row {i} slice len {len(r)} != seq_len {seq_len}")
    return mx.array(out, dtype=mx.int32)


def _batches(row_iter, batch: int, seq_len: int):
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


def _load_val_rows(val_shard: str, seq_len: int, max_rows: int) -> list[dict]:
    """Load a FIXED held-out validation row set (never trained on)."""
    need = seq_len + 1
    table = pq.read_table(val_shard, columns=READ_COLS)
    cols = {name: table[name].to_pylist() for name in READ_COLS}
    n = len(cols[TOKEN_COL])
    rows: list[dict] = []
    for ri in range(n):
        toks = cols[TOKEN_COL][ri]
        if toks is None or len(toks) < need:
            continue
        row = {"token_ids": toks}
        skip = False
        for src, _dst in CHANNELS:
            chan = cols[src][ri]
            if chan is None or len(chan) < need:
                skip = True
                break
            row[src] = chan
        if skip:
            continue
        rows.append(row)
        if len(rows) >= max_rows:
            break
    if not rows:
        raise ValueError(
            f"_load_val_rows: no rows with >= {need} tokens in held-out {val_shard}"
        )
    return rows


def _val_batch(rows: list[dict], idx: list[int], seq_len: int):
    toks = [rows[i]["token_ids"] for i in idx]
    input_ids = _stack(toks, seq_len, 0)
    targets = _stack(toks, seq_len, 1)
    side = {}
    for src, dst in CHANNELS:
        side[dst] = _stack([rows[i][src] for i in idx], seq_len, 0)
    loss_mask = mx.ones((len(idx), seq_len), dtype=mx.float32)
    return input_ids, targets, loss_mask, side


def _peak_gb() -> float:
    return float(mx.get_peak_memory()) / 1e9


# --------------------------------------------------------------------------- #
def evaluate_val(model, val_rows, batch, seq_len, step) -> tuple[float, float]:
    """Mean masked CE + perplexity over the full held-out set."""
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(val_rows), batch):
        idx = list(range(start, min(start + batch, len(val_rows))))
        if len(idx) < 1:
            continue
        input_ids, targets, loss_mask, side = _val_batch(val_rows, idx, seq_len)
        _, loss = model(input_ids, targets=targets, loss_mask=loss_mask, **side)
        mx.eval(loss)
        lval = float(loss)
        _check_finite("val_loss_batch", lval, step)
        total_loss += lval
        n_batches += 1
    mean_ce = total_loss / max(1, n_batches)
    ppl = math.exp(min(mean_ce, 50.0))  # cap exponent to avoid overflow on early steps
    return mean_ce, ppl


def _decode_continuation(model, prefix_ids, gen_tokens, seq_len, temperature):
    """Greedy (temperature<=0) / temperature-sampled autoregressive decode."""
    ctx = list(prefix_ids)
    generated: list[int] = []
    for _ in range(gen_tokens):
        window = ctx[-seq_len:]
        inp = mx.array([window], dtype=mx.int32)
        logits, _ = model(inp)
        last = logits[0, -1]
        if temperature and temperature > 0:
            probs = mx.softmax(last.astype(mx.float32) / temperature)
            nxt = int(mx.random.categorical(mx.log(probs + 1e-9)).item())
        else:
            nxt = int(mx.argmax(last).item())
        mx.eval(inp)
        generated.append(nxt)
        ctx.append(nxt)
    return generated


def compile_probe(model, tokenizer, verifier, val_rows, k, prefix_len,
                  gen_tokens, seq_len, temperature, step):
    """Decode K continuations, syntax-check each via clang++, return pass-rate."""
    k = min(k, len(val_rows))
    passes = 0
    samples = []
    with tempfile.TemporaryDirectory(prefix="stage1_probe_") as tmp:
        for i in range(k):
            full = val_rows[i]["token_ids"]
            prefix = full[:prefix_len]
            gen = _decode_continuation(
                model, prefix, gen_tokens, seq_len, temperature
            )
            text = tokenizer.decode(prefix + gen)
            cpp = Path(tmp) / f"probe_{step}_{i}.cpp"
            cpp.write_text(text, encoding="utf-8")
            outcome = verifier.syntax_check(str(cpp), std="c++17")
            if outcome.ok:
                passes += 1
            if len(samples) < 2:
                diag = outcome.diagnostics[0] if outcome.diagnostics else "(none)"
                samples.append(
                    f"    probe[{i}] ok={outcome.ok} exit={outcome.exit_code} "
                    f"diag0={diag[:160]!r}"
                )
    return passes / max(1, k), samples


# --------------------------------------------------------------------------- #
def save_ckpt(model, opt, step) -> Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    mpath = CKPT_DIR / f"model_step{step:06d}.safetensors"
    opath = CKPT_DIR / f"opt_step{step:06d}.safetensors"
    model.save_weights(str(mpath))
    opt_flat = {k: v for k, v in tree_flatten(opt.state) if isinstance(v, mx.array)}
    mx.save_safetensors(str(opath), opt_flat)
    return mpath


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=1280)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--ffn", type=int, default=3456)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--val-rows", type=int, default=64)
    ap.add_argument("--probe-k", type=int, default=8)
    ap.add_argument("--probe-prefix", type=int, default=256)
    ap.add_argument("--probe-gen", type=int, default=256)
    ap.add_argument("--probe-temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    # Activation-memory controls (opt-in; default path numerically unchanged).
    ap.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="per-DenseCppBlock gradient checkpointing (recompute activations "
        "in backward to cut peak memory)",
    )
    ap.add_argument(
        "--chunked-ce",
        action="store_true",
        help="streaming cross-entropy over vocab chunks (avoids materializing "
        "the full (B,S,V) logits tensor for backward)",
    )
    ap.add_argument(
        "--ce-chunk-size",
        type=int,
        default=16384,
        help="row chunk size (over flattened B*S) for --chunked-ce. Larger "
        "chunks = fewer Python loop iterations / kernel launches; 16384 measured "
        "fastest at 4x4096 (29.1GB peak, well under budget).",
    )
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="disable mx.compile of the train step (debugging)",
    )
    ap.add_argument(
        "--clear-cache-every",
        type=int,
        default=0,
        help="call mx.clear_cache() every N steps (0 = never). Memory is no "
        "longer tight at 4x4096 (~29GB of 128GB), so the default 0 skips the "
        "per-step cache flush, which measured +6%% steps/s with identical peak.",
    )
    add_stage1_production_arguments(ap)
    args = ap.parse_args()

    if args.production_graph_domain_data is not None:
        run_stage1_graph_domain_production(
            data_path=args.production_graph_domain_data,
            steps=args.steps,
            batch_size=args.batch,
            seq_len=args.seq_len,
            hidden_size=args.hidden,
            depth=args.depth,
            ffn_hidden_size=args.ffn,
            learning_rate=args.lr,
            seed=args.seed,
            attention_mode=args.production_attention_mode,
            compile=not args.no_compile,
            bf16=args.bf16,
        )
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_shards = sorted(glob.glob(DATA_GLOB))
    if len(all_shards) < 2:
        raise FileNotFoundError(
            f"need >=2 shards for train + held-out val; matched {len(all_shards)} "
            f"for {DATA_GLOB}"
        )
    val_shard = all_shards[-1]          # held out, NEVER trained on
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
    )
    dtype = mx.bfloat16 if args.bf16 else mx.float32
    model = DenseCppLM(cfg, dtype=dtype if args.bf16 else None)
    nparams = model.num_parameters()

    tokenizer = load_cppmega_tokenizer(TOKENIZER_PATH)
    verifier = CodeVerifier(repo_root=str(OUT_DIR))

    log("=" * 78)
    log(
        f"[config] hidden={cfg.hidden_size} depth={cfg.depth} ffn={cfg.ffn_hidden_size} "
        f"qh={cfg.num_query_heads} kvh={cfg.num_kv_heads} head_dim={cfg.head_dim} "
        f"vocab={cfg.vocab_size} dtype={'bf16' if args.bf16 else 'fp32'}"
    )
    log(
        f"[config] steps={args.steps} batch={args.batch} seq_len={args.seq_len} "
        f"tokens/step={args.batch * args.seq_len} lr={args.lr} wd={args.wd} "
        f"betas=(0.9,0.95) grad_clip={args.grad_clip} warmup={args.warmup} "
        f"cosine_decay=True"
    )
    log(
        f"[config] grad_checkpoint={args.grad_checkpoint} "
        f"chunked_ce={args.chunked_ce} ce_chunk_size={args.ce_chunk_size} "
        f"compile={not args.no_compile}"
    )
    log(
        f"[config] eval_every={args.eval_every} ckpt_every={args.ckpt_every} "
        f"val_rows={args.val_rows} probe_k={args.probe_k} "
        f"probe_prefix={args.probe_prefix} probe_gen={args.probe_gen} "
        f"probe_temp={args.probe_temp}"
    )
    log(
        f"[data] train_shards={len(train_shards)} held_out_val_shard={val_shard}"
    )
    log(f"[params] {nparams / 1e6:.2f}M")

    opt = optim.AdamW(
        learning_rate=args.lr, weight_decay=args.wd, betas=(0.9, 0.95)
    )
    peak_lr = args.lr
    warmup = args.warmup
    total = args.steps
    min_lr = peak_lr * 0.1

    def lr_at(step: int) -> float:
        if step < warmup:
            return peak_lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        prog = min(1.0, max(0.0, prog))
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * prog))

    def loss_fn(model, input_ids, targets, loss_mask, side):
        _, loss = model(input_ids, targets=targets, loss_mask=loss_mask, **side)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Compiled train step. ``mx.compile`` with state in/out lets MLX fuse the
    # forward+backward+optimizer update and (critically for memory) reuse
    # buffers across the graph. The optimizer + model parameters are the captured
    # state. Side channels are passed positionally as a fixed-arity tuple so the
    # compiled signature is stable across steps.
    state = [model.state, opt.state]

    def _step(input_ids, targets, loss_mask, side_vals):
        side = {dst: side_vals[i] for i, (_src, dst) in enumerate(CHANNELS)}
        loss, grads = loss_and_grad(model, input_ids, targets, loss_mask, side)
        grads, gnorm = optim.clip_grad_norm(grads, args.grad_clip)
        opt.update(model, grads)
        return loss, gnorm

    if args.no_compile:
        step_fn = _step
    else:
        step_fn = mx.compile(_step, inputs=state, outputs=state)

    val_rows = _load_val_rows(val_shard, args.seq_len, args.val_rows)
    log(f"[data] loaded {len(val_rows)} held-out val rows")

    row_iter = _iter_rows(train_shards, args.seq_len, args.seed)
    batch_iter = _batches(row_iter, args.batch, args.seq_len)

    mx.reset_peak_memory()
    t0 = time.time()
    last_train_loss = float("nan")

    for step in range(args.steps):
        input_ids, targets, loss_mask, side = next(batch_iter)
        # LR is updated outside the compiled step (it changes every step); MLX
        # picks up the new optimizer scalar via the captured state.
        opt.learning_rate = lr_at(step)
        side_vals = tuple(side[dst] for _src, dst in CHANNELS)
        loss, gnorm = step_fn(input_ids, targets, loss_mask, side_vals)
        # Single eval boundary per step: forces the compiled graph + optimizer
        # update to execute and lets MLX free transient activation buffers.
        mx.eval(state, loss, gnorm)
        last_train_loss = float(loss)
        _check_finite("train_loss", last_train_loss, step)
        _check_finite("grad_norm", float(gnorm), step)
        # Memory at 4x4096 is ~29GB of 128GB (not tight), so by default we do
        # NOT flush the freed-but-pooled buffer cache every step: keeping the
        # pool warm avoids re-allocation churn and measured +6%% steps/s with an
        # identical 28.7GB peak. ``--clear-cache-every N`` re-enables periodic
        # flushing if a future config runs closer to the memory ceiling.
        if args.clear_cache_every and (step + 1) % args.clear_cache_every == 0:
            mx.clear_cache()

        if step == 0 or (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed
            log(
                f"[step {step + 1:>5}] train_loss={last_train_loss:.4f} "
                f"lr={opt.learning_rate.item():.3e} gnorm={float(gnorm):.3f} "
                f"peak={_peak_gb():.2f}GB steps/s={sps:.3f}"
            )

        is_eval = (step + 1) % args.eval_every == 0 or step == 0
        if is_eval:
            t_eval = time.time()
            val_ce, val_ppl = evaluate_val(
                model, val_rows, args.batch, args.seq_len, step
            )
            pass_rate, samples = compile_probe(
                model, tokenizer, verifier, val_rows, args.probe_k,
                args.probe_prefix, args.probe_gen, args.seq_len,
                args.probe_temp, step,
            )
            log(
                f"[EVAL step={step + 1}] train_loss={last_train_loss:.4f} "
                f"val_loss={val_ce:.4f} val_ppl={val_ppl:.2f} "
                f"compile_pass_rate={pass_rate:.3f} peak_gb={_peak_gb():.2f} "
                f"eval_s={time.time() - t_eval:.1f}"
            )
            for s in samples:
                log(s)

        if (step + 1) % args.ckpt_every == 0:
            cp = save_ckpt(model, opt, step + 1)
            log(f"[ckpt step={step + 1}] saved {cp}")

    final_cp = save_ckpt(model, opt, args.steps)
    log(f"[DONE] steps={args.steps} final_ckpt={final_cp} peak_gb={_peak_gb():.2f}")
    log("=" * 78)


if __name__ == "__main__":
    main()
