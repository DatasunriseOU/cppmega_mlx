#!/usr/bin/env python3
"""Materialize the typed objective schedule as Megatron-ready parquet.

The output rows use ``shifted_lm_document_v1``: ``input_ids`` contains
``[objective_input[0], *objective_targets]`` and ``loss_mask`` contains the
objective mask followed by a zero sentinel. The adjacent JSON receipt is
validated again by the root Megatron converter before indexed data is emitted.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cppmega_v4.data.doc_id_assignment import stable_doc_signature
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
    TOKEN_IDS_COLUMN,
    TOKEN_SOURCE_DOC_IDS_COLUMN,
)
from cppmega_mlx.training.megatron_objectives import (
    MaterializedMegatronDocument,
    OBJECTIVE_TOKEN_SIDE_CHANNELS,
    build_pre_materialized_objective_contract,
    materialize_megatron_document,
    write_objective_materialization_artifact,
)
from cppmega_mlx.training.objective_data import (
    OBJECTIVE_SOURCE_COLUMNS,
    objective_source_from_tokenized_row,
    require_megatron_objective_source_columns,
)
from cppmega_mlx.training.objective_mixer import (
    EligibilityAwareTaskMixer,
    GraphAuxLossConfig,
)
from cppmega_mlx.training.task_mixer import STAGE1_DEFAULT_RATES

_ARROW_DTYPES = {
    "uint8": pa.uint8(),
    "uint16": pa.uint16(),
    "uint32": pa.uint32(),
    "uint64": pa.uint64(),
}
TOKEN_SIDECAR_TYPES = {
    column: _ARROW_DTYPES[dtype]
    for column, dtype in OBJECTIVE_TOKEN_SIDE_CHANNELS
}
PAIR_TYPE = pa.struct([pa.field("from", pa.int32()), pa.field("to", pa.int32())])
TRIPLE_TYPE = pa.struct(
    [
        pa.field("from", pa.int32()),
        pa.field("to", pa.int32()),
        pa.field("kind", pa.int32()),
    ]
)


def materialized_schema() -> pa.Schema:
    fields = [
        pa.field("input_ids", pa.list_(pa.int32()), nullable=False),
        pa.field("valid_token_count", pa.uint32(), nullable=False),
        pa.field("objective_kind", pa.string(), nullable=False),
        *[
            pa.field(column, pa.list_(dtype), nullable=False)
            for column, dtype in TOKEN_SIDECAR_TYPES.items()
        ],
        pa.field(
            "source_platform_ids", pa.list_(pa.list_(pa.uint16())), nullable=False
        ),
        pa.field("token_call_edges", pa.list_(PAIR_TYPE), nullable=False),
        pa.field("token_type_edges", pa.list_(PAIR_TYPE), nullable=False),
        pa.field("token_domain_edges", pa.list_(TRIPLE_TYPE), nullable=False),
        pa.field("token_build_edges", pa.list_(TRIPLE_TYPE), nullable=False),
        pa.field("token_shell_edges", pa.list_(TRIPLE_TYPE), nullable=False),
        pa.field("token_diagnostic_edges", pa.list_(TRIPLE_TYPE), nullable=False),
        pa.field("token_cross_domain_edges", pa.list_(TRIPLE_TYPE), nullable=False),
        pa.field("token_chunk_starts", pa.list_(pa.uint32()), nullable=False),
        pa.field("token_chunk_ends", pa.list_(pa.uint32()), nullable=False),
        pa.field("token_chunk_kinds", pa.list_(pa.uint16()), nullable=False),
        pa.field("token_chunk_dep_levels", pa.list_(pa.uint16()), nullable=False),
    ]
    return pa.schema(fields)


def _pad(values: object, capacity: int, *, fill: int) -> list[int]:
    result = [int(value) for value in values]  # type: ignore[union-attr]
    if len(result) > capacity:
        raise ValueError(
            f"materialized token-aligned row length {len(result)} exceeds {capacity}"
        )
    return result + [fill] * (capacity - len(result))


def padded_row(
    document: MaterializedMegatronDocument, *, capacity: int
) -> dict[str, object]:
    row = dict(document.row)
    valid = int(row["valid_token_count"])
    if valid != len(document.token_ids) or valid > capacity:
        raise ValueError(
            f"invalid materialized valid_token_count={valid}, capacity={capacity}"
        )
    row["input_ids"] = _pad(row["input_ids"], capacity, fill=0)
    for column in TOKEN_SIDECAR_TYPES:
        fill = 0
        row[column] = _pad(row[column], capacity, fill=fill)
    return row


def _iter_sources(shards: list[str], *, seed: int):
    rng = random.Random(seed)
    source_index = 0
    signature_to_id: dict[str, int] = {}
    id_to_signature: dict[int, str] = {}
    identity_columns = (
        "source_doc_id",
        "source_document_id",
        "document_id",
        "doc_id",
        "repo_stable_id",
        "filepath_stable_id",
        "commit_hash",
        "file_local_commit_index",
        "text",
    )
    while True:
        shard_order = list(range(len(shards)))
        rng.shuffle(shard_order)
        for shard_index in shard_order:
            parquet = pq.ParquetFile(shards[shard_index])
            available = tuple(parquet.schema_arrow.names)
            require_megatron_objective_source_columns(available)
            selected = [
                column for column in OBJECTIVE_SOURCE_COLUMNS if column in available
            ]
            if "doc_ids" in available:
                selected.append("doc_ids")
            selected.extend(
                column
                for column in identity_columns
                if column in available and column not in selected
            )
            table = parquet.read(columns=selected)
            columns = {name: table[name].to_pylist() for name in selected}
            row_order = list(range(table.num_rows))
            rng.shuffle(row_order)
            for row_index in row_order:
                row = {name: values[row_index] for name, values in columns.items()}
                signature = stable_doc_signature(row)
                stable_source_id = signature_to_id.get(signature)
                if stable_source_id is None:
                    stable_source_id = deterministic_source_id(signature)
                    collision = id_to_signature.get(stable_source_id)
                    if collision is not None and collision != signature:
                        raise ValueError(
                            "stable source identity hash collision: "
                            f"id={stable_source_id} signatures={collision!r}, "
                            f"{signature!r}"
                        )
                    signature_to_id[signature] = stable_source_id
                    id_to_signature[stable_source_id] = signature
                token_count = len(row[TOKEN_IDS_COLUMN])
                raw_source_ids = [
                    int(value) for value in row[TOKEN_SOURCE_DOC_IDS_COLUMN]
                ]
                if len(raw_source_ids) != token_count:
                    raise ValueError(
                        f"{TOKEN_SOURCE_DOC_IDS_COLUMN} length "
                        f"{len(raw_source_ids)} != token count {token_count}"
                    )
                if any(value < 0 for value in raw_source_ids) or (
                    any(value == 0 for value in raw_source_ids)
                    and not all(value == 0 for value in raw_source_ids)
                ):
                    raise ValueError(
                        f"{TOKEN_SOURCE_DOC_IDS_COLUMN} mixes positive and "
                        "non-positive IDs"
                    )
                positive_source_ids = {
                    value for value in raw_source_ids if value > 0
                }
                if len(positive_source_ids) > 1:
                    raise ValueError(
                        "objective source row contains multiple logical source IDs; "
                        "materialize per-document rows before objective mixing"
                    )
                row[TOKEN_SOURCE_DOC_IDS_COLUMN] = [stable_source_id] * token_count
                yield objective_source_from_tokenized_row(
                    row, source_index=source_index
                )
                source_index += 1


def deterministic_source_id(signature: str) -> int:
    """Map one stable source signature to a non-zero uint32 identity."""

    if not isinstance(signature, str) or not signature:
        raise ValueError("stable source signature must be a non-empty string")
    identity = int.from_bytes(
        hashlib.sha256(signature.encode("utf-8")).digest()[:4], "big"
    )
    return identity or 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--quota-window-samples", type=int, default=60)
    parser.add_argument("--shard-rows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--graph-aux-weight", type=float, default=1.0)
    parser.add_argument("--graph-indexer-weight", type=float, default=0.001)
    parser.add_argument("--graph-layer-weight", type=float, default=1.0)
    parser.add_argument("--graph-bce-weight", type=float, default=0.10)
    parser.add_argument("--graph-coverage-weight", type=float, default=0.05)
    parser.add_argument("--graph-topk", type=int, default=8)
    parser.add_argument(
        "--graph-relations",
        default="call,type",
        help="Comma-separated graph relations included in the auxiliary loss",
    )
    args = parser.parse_args()

    if args.samples < 1 or args.samples % args.quota_window_samples:
        raise ValueError(
            "--samples must be positive and divisible by --quota-window-samples"
        )
    if args.seq_len < 2 or args.shard_rows < 1:
        raise ValueError("--seq-len must be >=2 and --shard-rows must be >=1")
    shards = sorted(glob.glob(args.data_glob))
    if not shards:
        raise FileNotFoundError(f"no parquet shards match {args.data_glob!r}")

    mixer = EligibilityAwareTaskMixer(
        STAGE1_DEFAULT_RATES,
        seed=args.seed,
        max_input_tokens=args.seq_len,
    )
    source_iter = _iter_sources(shards, seed=args.seed)
    documents: list[MaterializedMegatronDocument] = []
    start_step = 0
    while len(documents) < args.samples:
        sources = [next(source_iter) for _ in range(args.quota_window_samples)]
        realized = mixer.materialize_window(sources, start_step=start_step)
        documents.extend(
            materialize_megatron_document(
                item,
                sources[item.source_index],
                require_production_sidecars=True,
            )
            for item in realized
        )
        start_step += args.quota_window_samples

    graph_relations = tuple(
        relation.strip()
        for relation in args.graph_relations.split(",")
        if relation.strip()
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
    contract = build_pre_materialized_objective_contract(
        documents,
        rates=mixer.rates,
        seed=args.seed,
        quota_window_samples=args.quota_window_samples,
        graph_config=graph_config,
        graph_weight=args.graph_aux_weight,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    capacity = args.seq_len + 1
    schema = materialized_schema()
    parquet_paths: list[Path] = []
    for shard_index, offset in enumerate(range(0, len(documents), args.shard_rows)):
        rows = [
            padded_row(document, capacity=capacity)
            for document in documents[offset : offset + args.shard_rows]
        ]
        table = pa.Table.from_pylist(rows, schema=schema)
        path = args.output_dir / f"objectives_{shard_index:05d}.parquet"
        pq.write_table(table, path, compression="zstd")
        parquet_paths.append(path)
    artifact_path = write_objective_materialization_artifact(
        args.output_dir,
        contract=contract,
        parquet_paths=parquet_paths,
    )
    contract_path = args.output_dir / "objective_contract.json"
    print(
        json.dumps(
            {
                "documents": len(documents),
                "input_tokens": contract["totals"]["input_tokens"],  # type: ignore[index]
                "loss_tokens": contract["totals"]["loss_tokens"],  # type: ignore[index]
                "contract": str(contract_path),
                "artifact": str(artifact_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
