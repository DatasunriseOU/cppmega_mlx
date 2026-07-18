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
from collections import Counter
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
from cppmega_mlx.data.graph_packet import GraphBatch
from cppmega_mlx.nn.code_graph_routes import build_token_graph_biases
from cppmega_mlx.training.objective_data import (
    OBJECTIVE_GRAPH_RELATION_COLUMNS,
    OBJECTIVE_SOURCE_COLUMNS,
    exclude_objective_routes,
    graph_batch_from_objective_routes,
    objective_source_from_tokenized_row,
    remap_objective_routes,
    require_objective_source_columns,
)
from cppmega_mlx.training.objective_mixer import (
    EligibilityAwareTaskMixer,
    GraphAuxLossConfig,
    ObjectiveAccounting,
    ObjectiveSource,
    production_training_loss,
)
from cppmega_mlx.training.objective_schedule import (
    CanonicalObjectivePlanner,
    OBJECTIVE_SCHEDULE_ALGORITHM,
    OBJECTIVE_SCHEDULE_RECEIPT_SCHEMA,
    canonical_schedule_receipt_sha256,
)
from cppmega_mlx.data.graph_recipe import (
    STAGE1_GRAPH_RELATIONS,
    stage1_graph_config_kwargs,
)
from cppmega_mlx.training.objectives import (
    SOURCE_TOKEN_INDICES_METADATA_KEY,
    ObjectiveExample,
)
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
_TRANSFORMED_CODE_TASKS = frozenset(
    {
        TaskKind.FIM,
        TaskKind.AST_FIM,
        TaskKind.IFIM,
        TaskKind.SYMBOL_RECOVERY,
        TaskKind.TYPE_RECOVERY,
        TaskKind.CALLEE_RECOVERY,
    }
)
_COMMIT_TASKS = frozenset({TaskKind.COMMIT_DIFF, TaskKind.PRE_TO_POST})
_CHUNK_GRAPH_RELATIONS = frozenset({"call", "type"})
_COMMIT_GRAPH_EXCLUSION_REASON = (
    "independently_tokenized_commit_sections_have_no_exact_source_map"
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


def _parse_graph_relations(value: str) -> tuple[str, ...]:
    relations = tuple(
        relation.strip() for relation in value.split(",") if relation.strip()
    )
    unsupported = sorted(set(relations) - set(STAGE1_GRAPH_RELATIONS))
    if unsupported:
        raise ValueError(
            "local Stage-1 graph supervision received non-canonical relations: "
            f"{unsupported}"
        )
    return relations


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
    edge_kind_bias: mx.array
    graph_targets: mx.array
    graph_pair_mask: mx.array
    graph_samples: int
    graph_edges: int
    graph_route_receipts: tuple[dict[str, object], ...]
    graph_route_exclusion_reason: str | None
    schedule_window_receipt: dict[str, object] | None = None

    @property
    def aligned(self) -> bool:
        return self.task in _ALIGNED_TASKS


def _pad(values: mx.array, length: int, *, fill: int = 0) -> list[int]:
    items = [int(value) for value in values.tolist()]
    if len(items) > length:
        raise ValueError(f"objective sequence length {len(items)} exceeds {length}")
    return items + [fill] * (length - len(items))


def _code_packet(source: ObjectiveSource, task: TaskKind) -> CodePacket | None:
    return None if task in _COMMIT_TASKS else source.code_packet


def _objective_source_indices(
    task: TaskKind,
    example: ObjectiveExample,
) -> list[int]:
    raw = example.metadata.get(SOURCE_TOKEN_INDICES_METADATA_KEY)
    input_length = int(example.input_ids.shape[0])
    if not isinstance(raw, (tuple, list)) or len(raw) != input_length + 1:
        raise ValueError(
            f"{task.value}: exact source-token map must have length "
            f"{input_length + 1}, got {0 if not isinstance(raw, (tuple, list)) else len(raw)}"
        )
    source_indices: list[int] = []
    for output_index, raw_index in enumerate(raw[:input_length]):
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError(
                f"{task.value}: source-token map index {output_index} must be an integer"
            )
        source_indices.append(int(raw_index))
    return source_indices


def _packet_vector(
    packet: CodePacket,
    field: str,
    *,
    where: str,
) -> list[int]:
    raw = getattr(packet, field)
    if raw is None:
        raise ValueError(f"{where}: required CodePacket.{field} is absent")
    values = np.asarray(raw)
    source_length = int(packet.token_ids.shape[0])
    if values.ndim != 1 or int(values.shape[0]) != source_length:
        raise ValueError(
            f"{where}: CodePacket.{field} must have shape ({source_length},), "
            f"got {tuple(values.shape)}"
        )
    return [int(value) for value in values.tolist()]


def _mapped_structure_vector(
    packet: CodePacket,
    field: str,
    source_indices: list[int],
    *,
    where: str,
) -> list[int]:
    source_values = _packet_vector(packet, field, where=where)
    result: list[int] = []
    for output_index, source_index in enumerate(source_indices):
        if source_index < 0:
            result.append(0)
            continue
        if source_index >= len(source_values):
            raise ValueError(
                f"{where}: source-token map index {source_index} at output token "
                f"{output_index} is outside {len(source_values)} values"
            )
        result.append(source_values[source_index])
    return result


def _mapped_document_ids(
    packet: CodePacket,
    source_indices: list[int],
    *,
    where: str,
) -> list[int]:
    source_length = int(packet.token_ids.shape[0])
    if packet.document_ids is None:
        raise ValueError(f"{where}: required aligned channel document_ids is absent")
    source_values = _packet_vector(packet, "document_ids", where=where)
    output: list[int | None] = []
    for output_index, source_index in enumerate(source_indices):
        if source_index < 0:
            output.append(None)
            continue
        if source_index >= source_length:
            raise ValueError(
                f"{where}: source-token map index {source_index} at output token "
                f"{output_index} is outside {source_length} tokens"
            )
        document_id = source_values[source_index]
        if document_id <= 0:
            raise ValueError(
                f"{where}: CodePacket.document_ids[{source_index}] must be positive"
            )
        output.append(document_id)
    mapped_positions = [index for index, value in enumerate(output) if value is not None]
    if not mapped_positions:
        raise ValueError(f"{where}: source-token map contains no original tokens")
    previous: list[int | None] = [None] * len(output)
    following: list[int | None] = [None] * len(output)
    last: int | None = None
    for index, value in enumerate(output):
        if value is not None:
            last = index
        previous[index] = last
    last = None
    for index in range(len(output) - 1, -1, -1):
        if output[index] is not None:
            last = index
        following[index] = last
    for index, value in enumerate(output):
        if value is not None:
            continue
        left = previous[index]
        right = following[index]
        source_position = (
            left
            if right is None or (left is not None and index - left <= right - index)
            else right
        )
        if source_position is None:  # pragma: no cover - mapped_positions proves this
            raise AssertionError(f"{where}: cannot bind inserted-token document ID")
        output[index] = output[source_position]
    return [int(value) for value in output if value is not None]


def _merge_graph_batches(rows: list[GraphBatch]) -> GraphBatch:
    if not rows:
        raise ValueError("objective graph batch requires at least one row")
    if any(row.batch_size != 1 for row in rows):
        raise ValueError("objective route adapter must produce one graph row at a time")
    return GraphBatch(
        graphs=tuple(row.graphs[0] for row in rows),
        chunk_starts=tuple(row.chunk_starts[0] for row in rows),
        chunk_ends=tuple(row.chunk_ends[0] for row in rows),
        chunk_kinds=tuple(row.chunk_kinds[0] for row in rows),
        chunk_dep_levels=tuple(row.chunk_dep_levels[0] for row in rows),
        edge_kinds=tuple(row.edge_kinds[0] for row in rows),
    )


def _graph_target_row_mx(
    graph_batch: GraphBatch,
    row: int,
    *,
    seq_len: int,
    graph_relations: tuple[str, ...],
) -> mx.array:
    graph = graph_batch.graphs[row]
    starts = graph_batch.chunk_starts[row].astype(mx.int32)
    ends = graph_batch.chunk_ends[row].astype(mx.int32)
    chunk_count = int(starts.shape[0])
    positions = mx.arange(seq_len, dtype=mx.int32)
    membership = (
        (positions[:, None] >= starts[None, :])
        & (positions[:, None] < ends[None, :])
    ).astype(mx.float32)
    targets = mx.zeros((seq_len, seq_len), dtype=mx.float32)
    for relation in graph_relations:
        edge = graph.edge(relation)
        if edge is None:
            raise ValueError(
                f"objective graph targets require relation {relation!r}; "
                f"graph row {row} has {graph.relations}"
            )
        source = edge.src
        destination = edge.dst
        if edge.mask is not None:
            active = edge.mask > 0
            source = source[active]
            destination = destination[active]
        if int(source.shape[0]) == 0:
            continue
        bound = chunk_count if relation in _CHUNK_GRAPH_RELATIONS else seq_len
        invalid = mx.any(
            (source < 0)
            | (destination < 0)
            | (source >= bound)
            | (destination >= bound)
        )
        mx.eval(invalid)
        if bool(invalid.item()):
            source_values = np.asarray(source)
            destination_values = np.asarray(destination)
            bad = (
                (source_values < 0)
                | (destination_values < 0)
                | (source_values >= bound)
                | (destination_values >= bound)
            )
            first = int(np.flatnonzero(bad)[0])
            coordinate = "chunks" if relation in _CHUNK_GRAPH_RELATIONS else "tokens"
            raise ValueError(
                f"objective graph {relation} edge "
                f"({int(source_values[first])}, {int(destination_values[first])}) "
                f"is outside {bound} {coordinate}"
            )
        if relation in _CHUNK_GRAPH_RELATIONS:
            adjacency = mx.zeros((chunk_count, chunk_count), dtype=mx.float32)
            adjacency = adjacency.at[source, destination].add(
                mx.ones(source.shape, dtype=mx.float32)
            )
            targets = targets + membership @ adjacency @ membership.T
        else:
            targets = targets.at[source, destination].add(
                mx.ones(source.shape, dtype=mx.float32)
            )
    return (targets > 0).astype(mx.float32)


def _graph_targets_and_pair_mask_mx(
    graph_batch: GraphBatch,
    *,
    seq_len: int,
    input_lengths: list[int],
    document_ids: mx.array,
    graph_relations: tuple[str, ...],
) -> tuple[mx.array, mx.array, mx.array]:
    targets = mx.stack(
        [
            _graph_target_row_mx(
                graph_batch,
                row,
                seq_len=seq_len,
                graph_relations=graph_relations,
            )
            for row in range(graph_batch.batch_size)
        ],
        axis=0,
    )
    positions = mx.arange(seq_len, dtype=mx.int32)
    causal = positions[:, None] >= positions[None, :]
    valid_tokens = positions[None, :] < mx.array(
        input_lengths, dtype=mx.int32
    )[:, None]
    same_document = document_ids[:, :, None] == document_ids[:, None, :]
    pair_mask = (
        causal[None, :, :]
        & valid_tokens[:, :, None]
        & valid_tokens[:, None, :]
        & same_document
    )
    targets = (targets > 0) & pair_mask
    eligible_rows = mx.any(targets, axis=(1, 2))
    pair_mask = pair_mask & eligible_rows[:, None, None]
    return (
        targets.astype(mx.float32),
        pair_mask.astype(mx.float32),
        eligible_rows,
    )


def _materialize_batch(
    task: TaskKind,
    entries: list[tuple[ObjectiveExample, ObjectiveSource]],
    *,
    seq_len: int,
    graph_relations: tuple[str, ...] = ("call", "type"),
    require_route_sidecars: bool = True,
    schedule_window_receipt: dict[str, object] | None = None,
) -> ObjectiveBatch:
    if not entries:
        raise ValueError(f"{task.value}: objective batch entries must be non-empty")
    unsupported_relations = sorted(
        set(graph_relations) - set(OBJECTIVE_GRAPH_RELATION_COLUMNS)
    )
    if not graph_relations or unsupported_relations:
        raise ValueError(
            f"{task.value}: graph_relations must be a non-empty supported tuple; "
            f"unsupported={unsupported_relations}"
        )
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
    packet_fields = {
        "structure_ids": "structure_ids",
        "dep_levels": "dep_levels",
        "ast_depth_ids": "ast_depth",
        "sibling_index_ids": "sibling_index",
        "node_type_ids": "ast_node_type",
    }
    document_rows: list[list[int]] = []
    side_rows: dict[str, list[list[int]]] = {
        model_name: [] for model_name in packet_fields
    }
    graph_rows: list[GraphBatch] = []
    route_receipts: list[dict[str, object]] = []
    exclusion_reasons: set[str] = set()
    input_lengths: list[int] = []
    for row_index, (example, source) in enumerate(entries):
        input_length = int(example.input_ids.shape[0])
        input_lengths.append(input_length)
        source_indices = _objective_source_indices(task, example)
        packet = _code_packet(source, task)
        if task in _COMMIT_TASKS:
            document_rows.append([1] * input_length + [0] * (seq_len - input_length))
            for model_name in packet_fields:
                side_rows[model_name].append([0] * seq_len)
            route_remap = exclude_objective_routes(
                source.code_packet,
                where=f"{task.value}[{row_index}]",
                reason=_COMMIT_GRAPH_EXCLUSION_REASON,
                require_sidecars=require_route_sidecars,
            )
            exclusion_reasons.add(_COMMIT_GRAPH_EXCLUSION_REASON)
        elif task in _ALIGNED_TASKS or task in _TRANSFORMED_CODE_TASKS:
            if packet is None:
                raise ValueError(
                    f"{task.value}: code objective requires ObjectiveSource.code_packet"
                )
            mapped_documents = _mapped_document_ids(
                packet,
                source_indices,
                where=f"{task.value}[{row_index}]",
            )
            document_rows.append(
                mapped_documents + [0] * (seq_len - input_length)
            )
            for model_name, packet_name in packet_fields.items():
                mapped = _mapped_structure_vector(
                    packet,
                    packet_name,
                    source_indices,
                    where=f"{task.value}[{row_index}]",
                )
                side_rows[model_name].append(mapped + [0] * (seq_len - input_length))
            route_remap = remap_objective_routes(
                packet,
                source_token_indices=source_indices,
                where=f"{task.value}[{row_index}]",
                require_sidecars=require_route_sidecars,
                mode=("identity" if task in _ALIGNED_TASKS else "source_token_remap"),
            )
        else:  # pragma: no cover - TaskKind exhaustiveness guard
            raise ValueError(f"unsupported objective task {task.value}")
        graph_rows.append(
            graph_batch_from_objective_routes(
                route_remap.columns,
                input_length=input_length,
                where=f"{task.value}[{row_index}]",
            )
        )
        route_receipts.append(dict(route_remap.receipt))

    if len(exclusion_reasons) > 1:  # pragma: no cover - task-homogeneous batches
        raise ValueError(
            f"{task.value}: objective batch has conflicting graph exclusions "
            f"{sorted(exclusion_reasons)}"
        )
    document_ids = mx.array(document_rows, dtype=mx.int32)
    side_channels = {
        model_name: mx.array(rows, dtype=mx.int32)
        for model_name, rows in side_rows.items()
    }
    graph_batch = _merge_graph_batches(graph_rows)
    relation_bias, edge_kind_bias = build_token_graph_biases(
        graph_batch,
        batch_size=len(entries),
        seq_length=seq_len,
        document_ids=document_ids,
    )
    graph_targets, graph_pair_mask, eligible_rows = _graph_targets_and_pair_mask_mx(
        graph_batch,
        seq_len=seq_len,
        input_lengths=input_lengths,
        document_ids=document_ids,
        graph_relations=graph_relations,
    )
    graph_edges_array = mx.sum(graph_targets)
    graph_samples_array = mx.sum(eligible_rows.astype(mx.int32))
    mx.eval(graph_edges_array, graph_samples_array)
    return ObjectiveBatch(
        task=task,
        examples=examples,
        input_ids=input_ids,
        targets=targets,
        loss_mask=loss_mask,
        document_ids=document_ids,
        side_channels=side_channels,
        block_bias=relation_bias,
        edge_kind_bias=edge_kind_bias,
        graph_targets=graph_targets,
        graph_pair_mask=graph_pair_mask,
        graph_samples=int(graph_samples_array.item()),
        graph_edges=int(graph_edges_array.item()),
        graph_route_receipts=tuple(route_receipts),
        graph_route_exclusion_reason=(
            None if not exclusion_reasons else next(iter(exclusion_reasons))
        ),
        schedule_window_receipt=schedule_window_receipt,
    )


def _objective_batches(
    source_iter,
    mixer: EligibilityAwareTaskMixer,
    *,
    batch_size: int,
    seq_len: int,
    quota_window_samples: int,
    quota_lookahead_samples: int | None = None,
    seed: int,
    graph_relations: tuple[str, ...] = ("call", "type"),
    require_route_sidecars: bool = True,
    schedule_receipts: list[dict[str, object]] | None = None,
):
    if quota_lookahead_samples is None:
        quota_lookahead_samples = 3 * quota_window_samples
    if quota_lookahead_samples < 0:
        raise ValueError("quota_lookahead_samples must be non-negative")
    quotas = mixer.quotas(quota_window_samples)
    if any(quota % batch_size for quota in quotas.values()):
        raise ValueError(
            "every objective quota must be divisible by batch size; got "
            + ", ".join(f"{task.value}={quota}" for task, quota in quotas.items())
        )
    rng = random.Random(seed)
    planner = CanonicalObjectivePlanner(
        mixer=mixer,
        source_iter=source_iter,
        quota_window_samples=quota_window_samples,
        quota_lookahead_samples=quota_lookahead_samples,
        graph_relations=graph_relations,
        require_route_sidecars=require_route_sidecars,
    )
    start_step = 0
    while True:
        window = planner.plan_window(start_step=start_step)
        if schedule_receipts is not None:
            schedule_receipts.append(dict(window.receipt))
        grouped: dict[TaskKind, list[tuple[ObjectiveExample, ObjectiveSource]]] = {
            task: [] for task in quotas
        }
        for assignment in window.assignments:
            item = assignment.realized
            grouped[item.task].append(
                (item.example, assignment.source)
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
                        require_route_sidecars=require_route_sidecars,
                        schedule_window_receipt=window.receipt,
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
    ap.add_argument(
        "--quota-lookahead-samples",
        type=int,
        default=None,
        help=(
            "maximum extra source rows retained while satisfying one exact "
            "objective quota window (default: three quota windows)"
        ),
    )
    graph_recipe = stage1_graph_config_kwargs()
    ap.add_argument(
        "--graph-aux-weight", type=float, default=graph_recipe["global_weight"]
    )
    ap.add_argument(
        "--graph-indexer-weight", type=float, default=graph_recipe["indexer_weight"]
    )
    ap.add_argument(
        "--graph-layer-weight", type=float, default=graph_recipe["layer_weight"]
    )
    ap.add_argument(
        "--graph-bce-weight", type=float, default=graph_recipe["bce_weight"]
    )
    ap.add_argument(
        "--graph-coverage-weight", type=float, default=graph_recipe["coverage_weight"]
    )
    ap.add_argument("--graph-topk", type=int, default=graph_recipe["topk"])
    ap.add_argument(
        "--graph-relations",
        default=",".join(STAGE1_GRAPH_RELATIONS),
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
    graph_relations = _parse_graph_relations(args.graph_relations)
    if args.quota_window_samples < 0:
        raise ValueError("--quota-window-samples must be non-negative")
    quota_window_samples = args.quota_window_samples or (60 * args.batch)
    quota_lookahead_samples = (
        3 * quota_window_samples
        if args.quota_lookahead_samples is None
        else args.quota_lookahead_samples
    )
    if quota_lookahead_samples < 0:
        raise ValueError("--quota-lookahead-samples must be non-negative")
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
        edge_kind_bias,
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
            block_bias=block_bias,
            edge_kind_bias=edge_kind_bias,
            graph_targets=graph_targets,
            graph_pair_mask=graph_pair_mask,
            graph_config=graph_config,
            graph_weight=args.graph_aux_weight,
        )

    objective_loss_and_grad = nn.value_and_grad(
        model,
        lambda input_ids, targets, loss_mask, document_ids, side_vals, block_bias,
        edge_kind_bias, graph_targets, graph_pair_mask: _objective_loss(
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
            edge_kind_bias,
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

    def _step(
        input_ids,
        targets,
        loss_mask,
        document_ids,
        side_vals,
        block_bias,
        edge_kind_bias,
        graph_targets,
        graph_pair_mask,
    ):
        losses, grads = objective_loss_and_grad(
            input_ids,
            targets,
            loss_mask,
            document_ids,
            side_vals,
            block_bias,
            edge_kind_bias,
            graph_targets,
            graph_pair_mask,
        )
        return _finish_step(losses, grads)

    if args.no_compile:
        step_fn = _step
    else:
        step_fn = mx.compile(_step, inputs=state, outputs=state)

    val_rows = _load_val_rows(val_shard, args.seq_len, args.val_rows)
    log(f"[data] loaded {len(val_rows)} held-out val rows")

    source_iter = _iter_sources(train_shards, args.seed)
    schedule_receipts: list[dict[str, object]] = []
    batch_iter = _objective_batches(
        source_iter,
        mixer,
        batch_size=args.batch,
        seq_len=args.seq_len,
        quota_window_samples=quota_window_samples,
        quota_lookahead_samples=quota_lookahead_samples,
        seed=args.seed,
        graph_relations=graph_config.relations,
        require_route_sidecars=True,
        schedule_receipts=schedule_receipts,
    )
    accounting = ObjectiveAccounting(mixer.rates)
    lm_accounting = ObjectiveAccounting(mixer.rates)
    graph_aux_samples = 0
    graph_aux_edges = 0
    graph_aux_batches = 0
    graph_aux_loss_sum = 0.0
    graph_route_exclusions: Counter[str] = Counter()

    mx.reset_peak_memory()
    t0 = time.time()
    last_train_loss = float("nan")

    for step in range(args.steps):
        objective_batch = next(batch_iter)
        # LR is updated outside the compiled step (it changes every step); MLX
        # picks up the new optimizer scalar via the captured state.
        opt.learning_rate = lr_at(step)
        side_vals = tuple(
            objective_batch.side_channels[dst] for _src, dst in CHANNELS
        )
        loss, lm_loss, graph_loss, gnorm = step_fn(
            objective_batch.input_ids,
            objective_batch.targets,
            objective_batch.loss_mask,
            objective_batch.document_ids,
            side_vals,
            objective_batch.block_bias,
            objective_batch.edge_kind_bias,
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
        if objective_batch.graph_route_exclusion_reason is not None:
            graph_route_exclusions[objective_batch.graph_route_exclusion_reason] += len(
                objective_batch.examples
            )
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
        "route_exclusions": dict(graph_route_exclusions),
    }
    schedule_report = {
        "schema": OBJECTIVE_SCHEDULE_RECEIPT_SCHEMA,
        "algorithm": OBJECTIVE_SCHEDULE_ALGORITHM,
        "windows": len(schedule_receipts),
        "windows_sha256": canonical_schedule_receipt_sha256(schedule_receipts),
        "quota_window_samples": quota_window_samples,
        "quota_lookahead_samples": quota_lookahead_samples,
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
    log(
        f"[objectives] schedule_receipt={json.dumps(schedule_report, sort_keys=True)}"
    )

    final_cp = save_ckpt(model, opt, args.steps)
    log(f"[DONE] steps={args.steps} final_ckpt={final_cp} peak_gb={_peak_gb():.2f}")
    log("=" * 78)


if __name__ == "__main__":
    main()
