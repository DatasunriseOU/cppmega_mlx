"""Stage-1 dense C++ LM production objective mixture + eval + compile probe.

This is a REAL local run (no mocks, no fabricated metrics — project RULE #1):

* Streaming bf16 training over typed tokenized-enriched shards, with exact
  deterministic eligibility-aware quotas for causal LM, FIM, AST-FIM, true
  IFIM, commit repair/transduction, and recovery objectives. The last shard is
  held out and NEVER trained on.
* Configured graph BCE/coverage is differentiated in the same scalar as LM loss;
  exact objective samples, input/loss tokens, and loss components are reported.
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
import json
import math
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pyarrow.parquet as pq
from mlx.utils import tree_flatten

from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.training.objective_data import (
    OBJECTIVE_SOURCE_COLUMNS,
    graph_targets_and_pair_mask,
    objective_source_from_tokenized_row,
    require_objective_source_columns,
)
from cppmega_mlx.training.objective_mixer import (
    EligibilityAwareTaskMixer,
    GraphAuxLossConfig,
    ObjectiveAccounting,
    ObjectiveSource,
    production_training_loss,
)
from cppmega_mlx.training.objectives import ObjectiveExample
from cppmega_mlx.training.task_mixer import STAGE1_DEFAULT_RATES, TaskKind
from cppmega_mlx.runtime.code_verifier import CodeVerifier
from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
from cppmega_mlx.training.stage1_production import (
    add_stage1_production_arguments,
    run_stage1_graph_domain_production,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_GLOB = "/Users/dave/sources/parquet/clang_semantic_4k_v10/shard_*.parquet"
OUT_DIR = _REPO_ROOT / "outputs"
CKPT_DIR = OUT_DIR / "stage1_ckpts"
LOG_PATH = OUT_DIR / "train_eval_stage1.log"
TOKENIZER_PATH = _REPO_ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json"

# Token-aligned side channels carried per row -> model kwarg name.
CHANNELS = (
    ("token_structure_ids", "structure_ids"),
    ("token_dep_levels", "dep_levels"),
    ("token_ast_depth", "ast_depth_ids"),
    ("token_sibling_index", "sibling_index_ids"),
    ("token_ast_node_type", "node_type_ids"),
)
TOKEN_COL = "token_ids"
PROVENANCE_COLUMNS = ("repo", "filepath", "commit_hash")
READ_COLS = list(
    dict.fromkeys((*OBJECTIVE_SOURCE_COLUMNS, *PROVENANCE_COLUMNS, "doc_ids"))
)

_ALIGNED_TASKS = frozenset(
    {
        TaskKind.CAUSAL_LM,
    }
)

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
def _iter_sources(shard_paths: list[str], seed: int):
    """Infinitely yield typed objective sources in deterministic shuffled order."""

    rng = random.Random(seed)
    source_index = 0
    while True:
        order = list(range(len(shard_paths)))
        rng.shuffle(order)
        for si in order:
            parquet_file = pq.ParquetFile(shard_paths[si])
            available = tuple(parquet_file.schema_arrow.names)
            require_objective_source_columns(available)
            selected = [column for column in READ_COLS if column in available]
            table = parquet_file.read(columns=selected)
            cols = {name: table[name].to_pylist() for name in selected}
            n = len(cols[TOKEN_COL])
            row_order = list(range(n))
            rng.shuffle(row_order)
            for ri in row_order:
                row = {name: values[ri] for name, values in cols.items()}
                yield objective_source_from_tokenized_row(
                    row, source_index=source_index
                )
                source_index += 1


def _stack(rows: list[list[int]], seq_len: int, offset: int) -> mx.array:
    out = [r[offset : offset + seq_len] for r in rows]
    for i, r in enumerate(out):
        if len(r) != seq_len:
            raise ValueError(f"_stack: row {i} slice len {len(r)} != seq_len {seq_len}")
    return mx.array(out, dtype=mx.int32)


@dataclass(frozen=True)
class ObjectiveBatch:
    task: TaskKind
    examples: tuple[ObjectiveExample, ...]
    input_ids: mx.array
    targets: mx.array
    loss_mask: mx.array
    document_ids: mx.array
    side_channels: dict[str, mx.array]
    block_bias: mx.array
    graph_targets: mx.array
    graph_pair_mask: mx.array
    graph_samples: int
    graph_edges: int

    @property
    def aligned(self) -> bool:
        return self.task in _ALIGNED_TASKS


def _pad(values: mx.array, length: int, *, fill: int = 0) -> list[int]:
    items = [int(value) for value in values.tolist()]
    if len(items) > length:
        raise ValueError(f"objective sequence length {len(items)} exceeds {length}")
    return items + [fill] * (length - len(items))


def _code_packet(source: ObjectiveSource, task: TaskKind) -> CodePacket | None:
    return None if task not in _ALIGNED_TASKS else source.code_packet


def _materialize_batch(
    task: TaskKind,
    entries: list[tuple[ObjectiveExample, ObjectiveSource]],
    *,
    seq_len: int,
    graph_relations: tuple[str, ...] = ("call", "type"),
) -> ObjectiveBatch:
    examples = tuple(example for example, _source in entries)
    input_ids = mx.array(
        [_pad(example.input_ids, seq_len) for example in examples], dtype=mx.int32
    )
    targets = mx.array(
        [_pad(example.target_ids, seq_len) for example in examples], dtype=mx.int32
    )
    loss_mask = mx.array(
        [_pad(example.loss_mask, seq_len) for example in examples], dtype=mx.float32
    )
    document_rows: list[list[int]] = []
    for example, source in entries:
        input_length = int(example.input_ids.shape[0])
        if task in _ALIGNED_TASKS:
            packet = _code_packet(source, task)
            assert packet is not None
            if packet.document_ids is None:
                raise ValueError(
                    f"{task.value}: required aligned channel document_ids is absent"
                )
            document_rows.append(
                _pad(packet.document_ids[:input_length], seq_len)
            )
        else:
            document_rows.append([1] * input_length + [0] * (seq_len - input_length))
    document_ids = mx.array(document_rows, dtype=mx.int32)

    side_channels: dict[str, mx.array] = {}
    if task in _ALIGNED_TASKS:
        packet_fields = {
            "structure_ids": "structure_ids",
            "dep_levels": "dep_levels",
            "ast_depth_ids": "ast_depth",
            "sibling_index_ids": "sibling_index",
            "node_type_ids": "ast_node_type",
        }
        for model_name, packet_name in packet_fields.items():
            rows: list[list[int]] = []
            for example, source in entries:
                packet = _code_packet(source, task)
                assert packet is not None
                values = getattr(packet, packet_name)
                if values is None:
                    raise ValueError(
                        f"{task.value}: required aligned channel {packet_name} is absent"
                    )
                rows.append(_pad(values[: int(example.input_ids.shape[0])], seq_len))
            side_channels[model_name] = mx.array(rows, dtype=mx.int32)

    batch_size = len(entries)
    graph_targets = np.zeros((batch_size, seq_len, seq_len), dtype=np.float32)
    graph_pair_mask = np.zeros_like(graph_targets)
    graph_samples = 0
    if task in _ALIGNED_TASKS:
        for batch_index, (example, source) in enumerate(entries):
            packet = _code_packet(source, task)
            assert packet is not None
            input_length = int(example.input_ids.shape[0])
            dense_targets, dense_pair_mask = graph_targets_and_pair_mask(
                packet,
                input_length=input_length,
                relations=graph_relations,
            )
            causal_edges = int(dense_targets.sum(dtype=np.float64))
            if causal_edges == 0:
                continue
            graph_targets[
                batch_index, :input_length, :input_length
            ] = dense_targets
            graph_pair_mask[
                batch_index, :input_length, :input_length
            ] = dense_pair_mask
            graph_samples += 1

    targets_array = mx.array(graph_targets)
    return ObjectiveBatch(
        task=task,
        examples=examples,
        input_ids=input_ids,
        targets=targets,
        loss_mask=loss_mask,
        document_ids=document_ids,
        side_channels=side_channels,
        block_bias=targets_array,
        graph_targets=targets_array,
        graph_pair_mask=mx.array(graph_pair_mask),
        graph_samples=graph_samples,
        graph_edges=int(graph_targets.sum(dtype=np.float64)),
    )


def _objective_batches(
    source_iter,
    mixer: EligibilityAwareTaskMixer,
    *,
    batch_size: int,
    seq_len: int,
    quota_window_samples: int,
    seed: int,
    graph_relations: tuple[str, ...] = ("call", "type"),
):
    quotas = mixer.quotas(quota_window_samples)
    if any(quota % batch_size for quota in quotas.values()):
        raise ValueError(
            "every objective quota must be divisible by batch size; got "
            + ", ".join(f"{task.value}={quota}" for task, quota in quotas.items())
        )
    rng = random.Random(seed)
    start_step = 0
    while True:
        sources = [next(source_iter) for _ in range(quota_window_samples)]
        realized = mixer.materialize_window(sources, start_step=start_step)
        grouped: dict[TaskKind, list[tuple[ObjectiveExample, ObjectiveSource]]] = {
            task: [] for task in quotas
        }
        for item in realized:
            grouped[item.task].append(
                (item.example, sources[item.source_index])
            )
        batches: list[ObjectiveBatch] = []
        for task, entries in grouped.items():
            for offset in range(0, len(entries), batch_size):
                batches.append(
                    _materialize_batch(
                        task,
                        entries[offset : offset + batch_size],
                        seq_len=seq_len,
                        graph_relations=graph_relations,
                    )
                )
        rng.shuffle(batches)
        yield from batches
        start_step += quota_window_samples


def _load_val_rows(val_shard: str, seq_len: int, max_rows: int) -> list[dict]:
    """Load a FIXED held-out validation row set (never trained on)."""
    need = seq_len + 1
    val_columns = [TOKEN_COL, "doc_ids", *(source for source, _target in CHANNELS)]
    table = pq.read_table(val_shard, columns=val_columns)
    cols = {name: table[name].to_pylist() for name in val_columns}
    n = len(cols[TOKEN_COL])
    rows: list[dict] = []
    for ri in range(n):
        toks = cols[TOKEN_COL][ri]
        if toks is None or len(toks) < need:
            continue
        row = {"token_ids": toks}
        doc_ids = cols["doc_ids"][ri]
        if doc_ids is None or len(doc_ids) < need:
            continue
        row["doc_ids"] = doc_ids
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
    docs = [rows[i]["doc_ids"] for i in idx]
    document_ids = _stack(docs, seq_len, 0)
    target_document_ids = _stack(docs, seq_len, 1)
    side = {}
    for src, dst in CHANNELS:
        side[dst] = _stack([rows[i][src] for i in idx], seq_len, 0)
    loss_mask = (document_ids == target_document_ids).astype(mx.float32)
    return input_ids, targets, loss_mask, document_ids, side


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
        input_ids, targets, loss_mask, document_ids, side = _val_batch(
            val_rows, idx, seq_len
        )
        block_bias = (
            mx.zeros((len(idx), seq_len, seq_len), dtype=mx.float32)
            if model.config.attention_mode == "dsa"
            else None
        )
        _, loss = model(
            input_ids,
            targets=targets,
            loss_mask=loss_mask,
            document_ids=document_ids,
            block_bias=block_bias,
            edge_kind_bias=None if block_bias is None else mx.zeros_like(block_bias),
            **side,
        )
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
        document_ids = mx.ones_like(inp)
        block_bias = (
            mx.zeros((1, len(window), len(window)), dtype=mx.float32)
            if model.config.attention_mode == "dsa"
            else None
        )
        logits, _ = model(
            inp,
            document_ids=document_ids,
            block_bias=block_bias,
            edge_kind_bias=None if block_bias is None else mx.zeros_like(block_bias),
        )
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
    ap.add_argument("--steps", type=int, default=10020)
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
    ap.add_argument(
        "--data-glob",
        default=DATA_GLOB,
        help="typed tokenized-enriched train/validation shard glob",
    )
    ap.add_argument(
        "--quota-window-samples",
        type=int,
        default=0,
        help="objective sample window (0 = 60 * batch; quotas must divide batch)",
    )
    ap.add_argument("--graph-aux-weight", type=float, default=1.0)
    ap.add_argument("--graph-indexer-weight", type=float, default=0.001)
    ap.add_argument("--graph-layer-weight", type=float, default=1.0)
    ap.add_argument("--graph-bce-weight", type=float, default=0.10)
    ap.add_argument("--graph-coverage-weight", type=float, default=0.05)
    ap.add_argument("--graph-topk", type=int, default=32)
    ap.add_argument(
        "--graph-relations",
        default="call,type",
        help="Comma-separated code-graph relations supervised by the indexer",
    )
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
            ffn_hidden_size=args.ffn,
            learning_rate=args.lr,
            seed=args.seed,
            attention_mode=args.production_attention_mode,
            compile=not args.no_compile,
            bf16=args.bf16,
        )
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_shards = sorted(glob.glob(args.data_glob))
    if len(all_shards) < 2:
        raise FileNotFoundError(
            f"need >=2 shards for train + held-out val; matched {len(all_shards)} "
            f"for {args.data_glob}"
        )
    val_shard = all_shards[-1]          # held out, NEVER trained on
    train_shards = all_shards[:-1]

    if args.steps < 1 or args.batch < 1:
        raise ValueError("--steps and --batch must be positive")
    if not math.isfinite(args.graph_aux_weight) or args.graph_aux_weight <= 0.0:
        raise ValueError("--graph-aux-weight must be finite and positive")
    for name, value in (
        ("--graph-indexer-weight", args.graph_indexer_weight),
        ("--graph-layer-weight", args.graph_layer_weight),
        ("--graph-bce-weight", args.graph_bce_weight),
        ("--graph-coverage-weight", args.graph_coverage_weight),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    graph_aux_enabled = True
    graph_relations = tuple(
        relation.strip()
        for relation in args.graph_relations.split(",")
        if relation.strip()
    )
    unsupported_graph_relations = sorted(
        set(graph_relations) - {"call", "type"}
    )
    if unsupported_graph_relations:
        raise ValueError(
            "local Stage-1 graph supervision supports call/type relations; got "
            f"{unsupported_graph_relations}"
        )
    if args.quota_window_samples < 0:
        raise ValueError("--quota-window-samples must be non-negative")
    quota_window_samples = args.quota_window_samples or (60 * args.batch)
    if quota_window_samples % args.batch:
        raise ValueError("quota window samples must be divisible by batch size")
    steps_per_quota_window = quota_window_samples // args.batch
    if args.steps % steps_per_quota_window:
        raise ValueError(
            f"--steps={args.steps} must be divisible by {steps_per_quota_window} "
            "to finish an exact objective quota window"
        )

    mixer = EligibilityAwareTaskMixer(
        STAGE1_DEFAULT_RATES,
        seed=args.seed,
        max_input_tokens=args.seq_len,
    )
    graph_config = GraphAuxLossConfig(
        relations=graph_relations,
        topk=args.graph_topk,
        global_weight=args.graph_aux_weight,
        indexer_weight=args.graph_indexer_weight,
        layer_weight=args.graph_layer_weight,
        bce_weight=args.graph_bce_weight,
        coverage_weight=args.graph_coverage_weight,
    )

    cfg = DenseCppLMConfig(
        vocab_size=65536,
        hidden_size=args.hidden,
        depth=args.depth,
        ffn_hidden_size=args.ffn,
        num_query_heads=20,
        num_kv_heads=4,
        head_dim=64,
        max_seq_length=max(4096, args.seq_len),
        attention_mode="dsa" if graph_aux_enabled else "gqa",
        attention_sparse_topk=args.graph_topk,
        require_graph_routes=graph_aux_enabled,
        graph_routes_enabled=graph_aux_enabled,
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
        f"[objectives] rates={{{', '.join(f'{task.value}:{rate:.8f}' for task, rate in mixer.rates.items())}}} "
        f"quota_window_samples={quota_window_samples} graph_aux_weight={args.graph_aux_weight} "
        f"graph_bce={args.graph_bce_weight} graph_coverage={args.graph_coverage_weight}"
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

    def _objective_loss(
        model,
        input_ids,
        targets,
        loss_mask,
        document_ids,
        side,
        block_bias,
        graph_targets,
        graph_pair_mask,
    ):
        return production_training_loss(
            model,
            input_ids,
            targets,
            loss_mask,
            side_channels=side,
            document_ids=document_ids,
            block_bias=block_bias if graph_aux_enabled else None,
            graph_targets=graph_targets if graph_aux_enabled else None,
            graph_pair_mask=graph_pair_mask if graph_aux_enabled else None,
            graph_config=graph_config if graph_aux_enabled else None,
            graph_weight=args.graph_aux_weight,
        )

    aligned_loss_and_grad = nn.value_and_grad(
        model,
        lambda input_ids, targets, loss_mask, document_ids, side_vals, block_bias,
        graph_targets, graph_pair_mask: _objective_loss(
            model,
            input_ids,
            targets,
            loss_mask,
            document_ids,
            {
                dst: side_vals[index]
                for index, (_src, dst) in enumerate(CHANNELS)
            },
            block_bias,
            graph_targets,
            graph_pair_mask,
        ),
    )

    def _reordered_lm_loss(
        model, input_ids, targets, loss_mask, document_ids, block_bias, graph_targets,
        graph_pair_mask,
    ):
        del graph_targets, graph_pair_mask
        _, lm_loss = model(
            input_ids,
            targets=targets,
            loss_mask=loss_mask,
            document_ids=document_ids,
            block_bias=block_bias,
            edge_kind_bias=mx.zeros_like(block_bias),
        )
        if lm_loss is None:
            raise RuntimeError("model returned no LM loss despite supplied targets")
        return lm_loss, lm_loss, mx.array(0.0, dtype=mx.float32)

    reordered_loss_and_grad = nn.value_and_grad(
        model,
        lambda input_ids, targets, loss_mask, document_ids, block_bias, graph_targets,
        graph_pair_mask: _reordered_lm_loss(
            model,
            input_ids,
            targets,
            loss_mask,
            document_ids,
            block_bias,
            graph_targets,
            graph_pair_mask,
        ),
    )

    # Compiled train step. ``mx.compile`` with state in/out lets MLX fuse the
    # forward+backward+optimizer update and (critically for memory) reuse
    # buffers across the graph. The optimizer + model parameters are the captured
    # state. Side channels are passed positionally as a fixed-arity tuple so the
    # compiled signature is stable across steps.
    state = [model.state, opt.state]

    def _finish_step(losses, grads):
        total_loss, lm_loss, graph_loss = losses
        grads, gnorm = optim.clip_grad_norm(grads, args.grad_clip)
        opt.update(model, grads)
        return total_loss, lm_loss, graph_loss, gnorm

    def _step_aligned(
        input_ids,
        targets,
        loss_mask,
        document_ids,
        side_vals,
        block_bias,
        graph_targets,
        graph_pair_mask,
    ):
        losses, grads = aligned_loss_and_grad(
            input_ids,
            targets,
            loss_mask,
            document_ids,
            side_vals,
            block_bias,
            graph_targets,
            graph_pair_mask,
        )
        return _finish_step(losses, grads)

    def _step_reordered(
        input_ids,
        targets,
        loss_mask,
        document_ids,
        block_bias,
        graph_targets,
        graph_pair_mask,
    ):
        losses, grads = reordered_loss_and_grad(
            input_ids,
            targets,
            loss_mask,
            document_ids,
            block_bias,
            graph_targets,
            graph_pair_mask,
        )
        return _finish_step(losses, grads)

    if args.no_compile:
        aligned_step_fn = _step_aligned
        reordered_step_fn = _step_reordered
    else:
        aligned_step_fn = mx.compile(_step_aligned, inputs=state, outputs=state)
        reordered_step_fn = mx.compile(_step_reordered, inputs=state, outputs=state)

    val_rows = _load_val_rows(val_shard, args.seq_len, args.val_rows)
    log(f"[data] loaded {len(val_rows)} held-out val rows")

    source_iter = _iter_sources(train_shards, args.seed)
    batch_iter = _objective_batches(
        source_iter,
        mixer,
        batch_size=args.batch,
        seq_len=args.seq_len,
        quota_window_samples=quota_window_samples,
        seed=args.seed,
        graph_relations=graph_config.relations,
    )
    accounting = ObjectiveAccounting(mixer.rates)
    lm_accounting = ObjectiveAccounting(mixer.rates)
    graph_aux_samples = 0
    graph_aux_edges = 0
    graph_aux_batches = 0
    graph_aux_loss_sum = 0.0

    mx.reset_peak_memory()
    t0 = time.time()
    last_train_loss = float("nan")

    for step in range(args.steps):
        objective_batch = next(batch_iter)
        # LR is updated outside the compiled step (it changes every step); MLX
        # picks up the new optimizer scalar via the captured state.
        opt.learning_rate = lr_at(step)
        if objective_batch.aligned:
            side_vals = tuple(
                objective_batch.side_channels[dst] for _src, dst in CHANNELS
            )
            loss, lm_loss, graph_loss, gnorm = aligned_step_fn(
                objective_batch.input_ids,
                objective_batch.targets,
                objective_batch.loss_mask,
                objective_batch.document_ids,
                side_vals,
                objective_batch.block_bias,
                objective_batch.graph_targets,
                objective_batch.graph_pair_mask,
            )
        else:
            loss, lm_loss, graph_loss, gnorm = reordered_step_fn(
                objective_batch.input_ids,
                objective_batch.targets,
                objective_batch.loss_mask,
                objective_batch.document_ids,
                objective_batch.block_bias,
                objective_batch.graph_targets,
                objective_batch.graph_pair_mask,
            )
        # Single eval boundary per step: forces the compiled graph + optimizer
        # update to execute and lets MLX free transient activation buffers.
        mx.eval(state, loss, lm_loss, graph_loss, gnorm)
        last_train_loss = float(loss)
        last_lm_loss = float(lm_loss)
        last_graph_loss = float(graph_loss)
        _check_finite("train_loss", last_train_loss, step)
        _check_finite("train_lm_loss", last_lm_loss, step)
        _check_finite("train_graph_loss", last_graph_loss, step)
        _check_finite("grad_norm", float(gnorm), step)
        for example in objective_batch.examples:
            accounting.record(objective_batch.task, example, loss=last_train_loss)
            lm_accounting.record(objective_batch.task, example, loss=last_lm_loss)
        graph_aux_samples += objective_batch.graph_samples
        graph_aux_edges += objective_batch.graph_edges
        graph_aux_batches += 1
        graph_aux_loss_sum += last_graph_loss
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
                f"lm_loss={last_lm_loss:.4f} graph_loss={last_graph_loss:.4f} "
                f"objective={objective_batch.task.value} "
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

    objective_report = accounting.report()
    lm_objective_report = lm_accounting.report()
    expected_quotas = mixer.quotas(args.steps * args.batch)
    realized_counts = {
        task: int(objective_report.get(task.value, {}).get("samples", 0))
        for task in expected_quotas
    }
    if realized_counts != expected_quotas:
        raise AssertionError(
            f"realized objective counts {realized_counts} != exact quotas "
            f"{expected_quotas}"
        )
    if graph_aux_enabled and (graph_aux_samples == 0 or graph_aux_edges == 0):
        raise AssertionError(
            "graph auxiliary loss was configured but no graph-eligible sample/edge "
            "entered training"
        )
    graph_report = {
        "configured_weight": args.graph_aux_weight,
        "batches": graph_aux_batches,
        "eligible_samples": graph_aux_samples,
        "positive_edges": graph_aux_edges,
        "raw_loss_sum": graph_aux_loss_sum,
        "raw_mean_loss": graph_aux_loss_sum / graph_aux_batches,
    }
    log(
        "[objectives] total_loss_accounting="
        f"{json.dumps(objective_report, sort_keys=True)}"
    )
    log(
        "[objectives] lm_loss_accounting="
        f"{json.dumps(lm_objective_report, sort_keys=True)}"
    )
    log(f"[objectives] graph_accounting={json.dumps(graph_report, sort_keys=True)}")

    final_cp = save_ckpt(model, opt, args.steps)
    log(f"[DONE] steps={args.steps} final_ckpt={final_cp} peak_gb={_peak_gb():.2f}")
    log("=" * 78)


if __name__ == "__main__":
    main()
