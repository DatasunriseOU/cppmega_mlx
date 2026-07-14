"""Fail-closed graph/domain recipe for production Stage-1 MLX training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from cppmega_mlx.data.batch import LMTokenBatch, ensure_lm_batch
from cppmega_mlx.data.domain_schema import DomainEdgeKind
from cppmega_mlx.data.graph_packet import GraphBatch
from cppmega_mlx.data.production_bundle import (
    ProductionMegatronDatasetMetadata,
    open_production_megatron_bundle,
)
from cppmega_mlx.models.dense_cpp_lm import (
    DenseCppLM,
    DenseCppLMConfig,
    GraphIndexedAttention,
)
from cppmega_mlx.training.compiled import (
    CompiledPretrainingStep,
    normalize_compiled_batch,
)


STAGE1_GRAPH_DOMAIN_RECIPE = "stage1_graph_domain_v1"
PRODUCTION_GRAPH_BETA = 1.0
PRODUCTION_DOMAIN_RESIDUAL_SCALE = 1.0
PRODUCTION_DOMAIN_FIELDS = ("domain_ids", "role_ids", "confidence_ids")
ProductionAttentionMode = Literal["gqa", "dsa"]


def stage1_production_config(
    *,
    attention_mode: ProductionAttentionMode = "gqa",
    **overrides: Any,
) -> DenseCppLMConfig:
    """Build the explicit Stage-1 graph/domain model contract."""

    if attention_mode not in ("gqa", "dsa"):
        raise ValueError(
            "Stage-1 production attention_mode must be 'gqa' or 'dsa'; "
            f"got {attention_mode!r}"
        )
    protected = {
        "graph_routes_enabled": True,
        "require_graph_routes": True,
        "graph_attention_bias_beta": PRODUCTION_GRAPH_BETA,
        "domain_residual_scale": PRODUCTION_DOMAIN_RESIDUAL_SCALE,
        "require_domain_routes": True,
    }
    conflicts = {
        key: value
        for key, value in protected.items()
        if key in overrides and overrides[key] != value
    }
    if conflicts:
        raise ValueError(
            "Stage-1 production graph/domain settings are fixed; conflicting "
            f"overrides: {conflicts}"
        )
    values = {**overrides, **protected, "attention_mode": attention_mode}
    return DenseCppLMConfig(**values)


def build_stage1_production_model(
    *,
    attention_mode: ProductionAttentionMode = "gqa",
    dtype: mx.Dtype | None = None,
    **overrides: Any,
) -> DenseCppLM:
    config = stage1_production_config(
        attention_mode=attention_mode,
        **overrides,
    )
    model = DenseCppLM(config, dtype=dtype)
    validate_stage1_production_model(model)
    return model


def validate_stage1_production_model(model: DenseCppLM) -> None:
    cfg = model.config
    if not cfg.graph_routes_enabled or not cfg.require_graph_routes:
        raise ValueError("production Stage-1 requires graph routes fail-closed")
    if cfg.graph_attention_bias_beta != PRODUCTION_GRAPH_BETA:
        raise ValueError(
            "production Stage-1 graph beta must be "
            f"{PRODUCTION_GRAPH_BETA}, got {cfg.graph_attention_bias_beta}"
        )
    if (
        cfg.domain_residual_scale != PRODUCTION_DOMAIN_RESIDUAL_SCALE
        or not cfg.require_domain_routes
    ):
        raise ValueError(
            "production Stage-1 requires all domain routes at residual scale "
            f"{PRODUCTION_DOMAIN_RESIDUAL_SCALE}"
        )
    if cfg.attention_mode == "dsa":
        if not all(
            isinstance(layer.attention, GraphIndexedAttention)
            and layer.attention.config.mode == "dsa"
            for layer in model.layers
        ):
            raise RuntimeError(
                "production DSA selected a non-indexed/dense attention implementation"
            )
    elif cfg.attention_mode == "gqa":
        if not all(
            not isinstance(layer.attention, GraphIndexedAttention)
            and layer.attention.config.mode == "gqa"
            and layer.attention.config.is_gqa
            for layer in model.layers
        ):
            raise RuntimeError("production GQA did not preserve grouped-KV semantics")
    else:
        raise ValueError(
            f"unsupported production attention mode {cfg.attention_mode!r}"
        )


def stage1_production_batch_receipt(
    batch: LMTokenBatch,
    *,
    config: DenseCppLMConfig,
) -> dict[str, Any]:
    """Validate one startup batch and return proof of live graph/domain signal."""

    _validate_stage1_production_config(config)
    lm_batch = ensure_lm_batch(batch)
    if lm_batch.graph_batch is None:
        raise ValueError(
            "production Stage-1 requires a typed graph_batch from graph sidecars"
        )

    graph_edges = sum(
        len(edge.to_pairs())
        for graph in lm_batch.graph_batch.graphs
        for edge in graph.edges.values()
    )
    if graph_edges <= 0:
        raise ValueError("production Stage-1 requires nonzero graph edges")
    edge_kind_edges, edge_kind_ids = _validate_edge_kind_categories(
        lm_batch.graph_batch
    )

    normalized = normalize_compiled_batch(
        lm_batch,
        graph_routes_enabled=True,
    )
    document_boundaries = _document_boundary_count(lm_batch)
    graph_bias = normalized["graph_attention_bias"]
    edge_kind_bias = normalized["graph_edge_kind_bias"]
    if not isinstance(graph_bias, mx.array):
        raise ValueError("production Stage-1 graph prior was not materialized")
    if not isinstance(edge_kind_bias, mx.array):
        raise ValueError("production Stage-1 edge-kind prior was not materialized")
    graph_prior_nonzero_array = mx.sum(graph_bias != 0)
    edge_kind_prior_nonzero_array = mx.sum(edge_kind_bias != 0)

    domain_arrays: dict[str, mx.array] = {}
    for name in PRODUCTION_DOMAIN_FIELDS:
        value = normalized[name]
        if not isinstance(value, mx.array):
            raise ValueError(
                "production Stage-1 requires all domain sidecars; "
                f"missing {name}"
            )
        domain_arrays[name] = value
    _validate_domain_arrays(domain_arrays, config=config)

    domain_tokens_array = mx.sum(domain_arrays["domain_ids"] != 0)
    mx.eval(
        graph_prior_nonzero_array,
        edge_kind_prior_nonzero_array,
        domain_tokens_array,
    )
    graph_prior_nonzero = int(graph_prior_nonzero_array.item())
    edge_kind_prior_nonzero = int(edge_kind_prior_nonzero_array.item())
    domain_tokens_nonzero = int(domain_tokens_array.item())
    if graph_prior_nonzero <= 0:
        raise ValueError("production Stage-1 requires a nonzero graph prior")
    if domain_tokens_nonzero <= 0:
        raise ValueError("production Stage-1 requires nonzero domain tokens")

    return {
        "recipe": STAGE1_GRAPH_DOMAIN_RECIPE,
        "attention_mode": config.attention_mode,
        "graph_edges": graph_edges,
        "edge_kind_edges": edge_kind_edges,
        "edge_kind_ids": edge_kind_ids,
        "graph_relations": sorted(
            {
                relation
                for graph in lm_batch.graph_batch.graphs
                for relation in graph.relations
            }
        ),
        "graph_prior_nonzero": graph_prior_nonzero,
        "edge_kind_prior_nonzero": edge_kind_prior_nonzero,
        "graph_attention_bias_beta": config.graph_attention_bias_beta,
        "domain_sidecars": sorted(domain_arrays),
        "domain_tokens_nonzero": domain_tokens_nonzero,
        "domain_residual_scale": config.domain_residual_scale,
        "domain_residual_enabled": config.domain_residual_scale > 0,
        "document_ids_present": True,
        "document_boundaries": document_boundaries,
        "batch_size": int(lm_batch.tokens.shape[0]),
        "sequence_length": int(lm_batch.tokens.shape[1]),
    }


def add_stage1_production_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--production-graph-domain-data",
        type=Path,
        default=None,
        help=(
            "restored immutable Megatron bundle root with required graph/domain "
            "sidecars; activates the fail-closed production recipe"
        ),
    )
    parser.add_argument(
        "--production-attention-mode",
        choices=("gqa", "dsa"),
        default="gqa",
    )


def run_stage1_graph_domain_production(
    *,
    data_path: Path,
    bucket: int,
    expected_bundle_id: str,
    restore_receipt: Path,
    steps: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    depth: int,
    ffn_hidden_size: int,
    learning_rate: float,
    seed: int,
    attention_mode: ProductionAttentionMode = "gqa",
    compile: bool = True,
    bf16: bool = False,
    bundle_hash_jobs: int = 4,
) -> dict[str, Any]:
    """Run Stage-1 only after immutable bundle and restore validation."""

    if steps < 1:
        raise ValueError("production Stage-1 steps must be positive")
    dataset = open_production_megatron_bundle(
        data_path,
        bucket,
        expected_bundle_id,
        restore_receipt=restore_receipt,
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        loop=True,
        hash_jobs=bundle_hash_jobs,
    )
    try:
        first_batch = next(dataset.iter_batches(loop=False))
    except StopIteration as error:
        raise ValueError(
            "production Stage-1 dataset has no complete deterministic startup batch"
        ) from error
    dtype = mx.bfloat16 if bf16 else None
    model = build_stage1_production_model(
        attention_mode=attention_mode,
        dtype=dtype,
        vocab_size=dataset.metadata.vocab_size,
        hidden_size=hidden_size,
        depth=depth,
        ffn_hidden_size=ffn_hidden_size,
        max_seq_length=seq_len,
        num_query_heads=20,
        num_kv_heads=4,
        head_dim=64,
    )
    if not isinstance(dataset.metadata, ProductionMegatronDatasetMetadata):
        raise RuntimeError("production Stage-1 dataset lost immutable provenance metadata")
    startup = {
        **stage1_production_batch_receipt(first_batch, config=model.config),
        **dataset.metadata.provenance_receipt(),
    }
    print("[stage1-production-startup] " + json.dumps(startup, sort_keys=True), flush=True)

    optimizer = optim.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )
    stepper = CompiledPretrainingStep(model, optimizer, compile=compile)
    batch_iter = dataset.iter_batches(loop=True)
    observed_edges = 0
    observed_edge_kind_edges = 0
    observed_graph_prior_nonzero = 0
    observed_edge_kind_prior_nonzero = 0
    observed_domain_tokens = 0
    observed_document_boundaries = 0
    last_loss = math.nan
    last_domain_residual_l1 = 0.0
    for _ in range(steps):
        current = next(batch_iter)
        counts = _batch_route_counts(current)
        observed_edges += counts["graph_edges"]
        observed_edge_kind_edges += counts["edge_kind_edges"]
        observed_graph_prior_nonzero += counts["graph_prior_nonzero"]
        observed_edge_kind_prior_nonzero += counts["edge_kind_prior_nonzero"]
        observed_domain_tokens += counts["domain_tokens_nonzero"]
        observed_document_boundaries += counts["document_boundaries"]
        last_loss = stepper(current).loss
        last_domain_residual_l1 = _domain_residual_l1(model, current)
    if (
        observed_edges <= 0
        or observed_edge_kind_edges <= 0
        or observed_graph_prior_nonzero <= 0
        or observed_domain_tokens <= 0
        or observed_document_boundaries <= 0
        or last_domain_residual_l1 <= 0.0
    ):
        raise RuntimeError(
            "production Stage-1 observed no live graph/kind/domain/document "
            "signal during training"
        )
    receipt = {
        **startup,
        "steps": steps,
        "compiled": compile,
        "observed_graph_edges": observed_edges,
        "observed_edge_kind_edges": observed_edge_kind_edges,
        "observed_graph_prior_nonzero": observed_graph_prior_nonzero,
        "observed_edge_kind_prior_nonzero": observed_edge_kind_prior_nonzero,
        "observed_domain_tokens_nonzero": observed_domain_tokens,
        "observed_document_boundaries": observed_document_boundaries,
        "last_domain_residual_l1": last_domain_residual_l1,
        "last_loss": last_loss,
    }
    print("[stage1-production-receipt] " + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def _validate_stage1_production_config(config: DenseCppLMConfig) -> None:
    if (
        not config.graph_routes_enabled
        or not config.require_graph_routes
        or config.graph_attention_bias_beta != PRODUCTION_GRAPH_BETA
        or config.domain_residual_scale != PRODUCTION_DOMAIN_RESIDUAL_SCALE
        or not config.require_domain_routes
    ):
        raise ValueError(
            "batch receipt requires the canonical Stage-1 graph/domain config"
        )


def _validate_edge_kind_categories(
    graph_batch: GraphBatch,
) -> tuple[int, list[int]]:
    if not graph_batch.edge_kinds:
        raise ValueError(
            "production Stage-1 requires edge-kind sidecars for graph triples"
        )
    known = {int(kind) for kind in DomainEdgeKind}
    observed: set[int] = set()
    edge_count = 0
    for row in graph_batch.edge_kinds:
        for values in row.values():
            raw = np.asarray(values)
            edge_count += int(raw.size)
            observed.update(int(value) for value in raw.tolist())
    if edge_count <= 0:
        raise ValueError(
            "production Stage-1 requires edge-kind sidecars for graph triples"
        )
    unknown = sorted(observed - known)
    if unknown:
        raise ValueError(
            f"production Stage-1 edge-kind sidecars contain unsupported IDs {unknown}"
        )
    return edge_count, sorted(observed)


def _validate_domain_arrays(
    arrays: dict[str, mx.array],
    *,
    config: DenseCppLMConfig,
) -> None:
    maxima = {
        "domain_ids": config.domain_num_domains - 1,
        "role_ids": config.domain_num_roles - 1,
        "confidence_ids": config.domain_num_confidences - 1,
    }
    expected_shape = tuple(arrays["domain_ids"].shape)
    for name, value in arrays.items():
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"production domain sidecar {name} shape {tuple(value.shape)} "
                f"does not match {expected_shape}"
            )
        if value.dtype not in (mx.int16, mx.int32, mx.int64, mx.uint16, mx.uint32):
            raise ValueError(f"production domain sidecar {name} must be integer")
        invalid = mx.any((value < 0) | (value > maxima[name]))
        mx.eval(invalid)
        if bool(invalid.item()):
            raise ValueError(
                f"production domain sidecar {name} contains out-of-range IDs"
            )


def _batch_route_counts(batch: LMTokenBatch) -> dict[str, int]:
    lm_batch = ensure_lm_batch(batch)
    if lm_batch.graph_batch is None:
        raise ValueError("production batch lost typed graph_batch")
    graph_edges = sum(
        len(edge.to_pairs())
        for graph in lm_batch.graph_batch.graphs
        for edge in graph.edges.values()
    )
    edge_kind_edges, _edge_kind_ids = _validate_edge_kind_categories(
        lm_batch.graph_batch
    )
    normalized = normalize_compiled_batch(lm_batch, graph_routes_enabled=True)
    relation_bias = normalized["graph_attention_bias"]
    kind_bias = normalized["graph_edge_kind_bias"]
    if not isinstance(relation_bias, mx.array) or not isinstance(kind_bias, mx.array):
        raise ValueError("production batch lost fixed graph priors")
    domain_routes = (
        {} if lm_batch.side_channels is None
        else lm_batch.side_channels.get("domain_routes", {})
    )
    domain_ids = lm_batch.domain_ids
    if domain_ids is None:
        domain_ids = domain_routes.get("domain_ids")
    if not isinstance(domain_ids, mx.array):
        raise ValueError("production batch lost domain_ids")
    count = mx.sum(domain_ids != 0)
    relation_count = mx.sum(relation_bias != 0)
    kind_count = mx.sum(kind_bias != 0)
    mx.eval(count, relation_count, kind_count)
    return {
        "graph_edges": graph_edges,
        "edge_kind_edges": edge_kind_edges,
        "graph_prior_nonzero": int(relation_count.item()),
        "edge_kind_prior_nonzero": int(kind_count.item()),
        "domain_tokens_nonzero": int(count.item()),
        "document_boundaries": _document_boundary_count(lm_batch),
    }


def _document_boundary_count(batch: LMTokenBatch) -> int:
    document_ids = batch.document_ids
    if not isinstance(document_ids, mx.array):
        raise ValueError("production Stage-1 requires document_ids sidecars")
    if document_ids.dtype not in (
        mx.int8,
        mx.int16,
        mx.int32,
        mx.int64,
        mx.uint8,
        mx.uint16,
        mx.uint32,
        mx.uint64,
    ):
        raise ValueError("production document_ids sidecar must be integer")
    invalid = mx.any(document_ids < 0)
    boundaries = mx.sum(document_ids[:, 1:] != document_ids[:, :-1])
    mx.eval(invalid, boundaries)
    if bool(invalid.item()):
        raise ValueError("production document_ids sidecar must be non-negative")
    return int(boundaries.item())


def _domain_residual_l1(model: DenseCppLM, batch: LMTokenBatch) -> float:
    if model.domain_embedding is None:
        raise RuntimeError("production model lost domain embedding")
    kwargs = ensure_lm_batch(batch).model_kwargs()
    residual = model.domain_embedding(
        domain_ids=kwargs.get("domain_ids"),
        role_ids=kwargs.get("role_ids"),
        confidence_ids=kwargs.get("confidence_ids"),
        target_dtype=model.token_embedding.weight.dtype,
    )
    value = mx.sum(mx.abs(residual.astype(mx.float32)))
    mx.eval(value)
    return float(value.item()) * float(model.config.domain_residual_scale)


__all__ = [
    "PRODUCTION_DOMAIN_RESIDUAL_SCALE",
    "PRODUCTION_GRAPH_BETA",
    "STAGE1_GRAPH_DOMAIN_RECIPE",
    "add_stage1_production_arguments",
    "build_stage1_production_model",
    "run_stage1_graph_domain_production",
    "stage1_production_batch_receipt",
    "stage1_production_config",
    "validate_stage1_production_model",
]
