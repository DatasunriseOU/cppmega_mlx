"""Fail-closed graph/domain recipe for production Stage-1 MLX training."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
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
from cppmega_mlx.data.graph_recipe import (
    STAGE1_GRAPH_BIAS_BETA,
    STAGE1_GRAPH_RELATIONS,
    STAGE1_GRAPH_SCORE_FORMULA,
    STAGE1_GRAPH_SCORE_STAGE,
    STAGE1_GRAPH_TOPK,
    stage1_graph_config_kwargs,
    stage1_graph_recipe_binding,
    validate_stage1_graph_config,
)
from cppmega_mlx.training.objective_mixer import (
    GraphAuxLossConfig,
    ProductionTrainingLossBreakdown,
    scheduled_production_training_loss_breakdown,
)
from cppmega_mlx.training.objective_schedule import (
    PREMATERIALIZED_OBJECTIVE,
    canonical_batch_shape_assignment_receipts,
    validate_objective_assignment_receipt,
)


STAGE1_GRAPH_DOMAIN_RECIPE = "stage1_graph_domain_v1"
STAGE1_GQA_ROUTE_OBJECTIVE = "gqa_route_only_lm_ce"
STAGE1_DSA_GRAPH_OBJECTIVE = "lm_ce_plus_dsa_indexer_graph_auxiliary"
PRODUCTION_GRAPH_BETA = float(STAGE1_GRAPH_BIAS_BETA)
PRODUCTION_DOMAIN_RESIDUAL_SCALE = 1.0
PRODUCTION_DOMAIN_FIELDS = ("domain_ids", "role_ids", "confidence_ids")
ProductionAttentionMode = Literal["gqa", "dsa"]
STAGE1_PRODUCTION_GRAPH_RELATIONS = STAGE1_GRAPH_RELATIONS
STAGE1_PRODUCTION_GRAPH_LOSS = GraphAuxLossConfig(
    **stage1_graph_config_kwargs(),
)


@dataclass(frozen=True)
class _Stage1ObjectiveBatch:
    input_ids: mx.array
    targets: mx.array
    loss_mask: mx.array
    document_ids: mx.array
    relation_bias: mx.array
    edge_kind_bias: mx.array
    graph_targets: mx.array
    graph_pair_mask: mx.array
    side_channels: Mapping[str, mx.array]


@dataclass(frozen=True)
class Stage1ProductionObjective:
    """Canonical Stage-1 CE objective with optional DSA graph supervision."""

    graph_config: GraphAuxLossConfig = STAGE1_PRODUCTION_GRAPH_LOSS
    schedule_assignments: tuple[Mapping[str, object], ...] | None = None

    def __post_init__(self) -> None:
        validate_stage1_graph_config(self.graph_config)

    def validate_batch(
        self,
        batch: LMTokenBatch | Mapping[str, Any] | mx.array,
    ) -> None:
        values = _stage1_objective_batch(batch)
        finite_relation = mx.all(mx.isfinite(values.relation_bias))
        finite_edge_kind = mx.all(mx.isfinite(values.edge_kind_bias))
        targets_outside_mask = mx.any(
            (values.graph_targets > 0) & (values.graph_pair_mask <= 0)
        )
        loss_tokens = mx.sum(values.loss_mask.astype(mx.float32))
        mx.eval(
            finite_relation,
            finite_edge_kind,
            targets_outside_mask,
            loss_tokens,
        )
        if not bool(finite_relation.item()):
            raise ValueError("production graph relation prior must be finite")
        if not bool(finite_edge_kind.item()):
            raise ValueError("production graph edge-kind prior must be finite")
        if bool(targets_outside_mask.item()):
            raise ValueError(
                "production Stage-1 graph target crosses a causal/document boundary"
            )
        if float(loss_tokens.item()) <= 0.0:
            raise ValueError("production Stage-1 batch has no LM loss tokens")

    def loss_breakdown(
        self,
        model: DenseCppLM,
        batch: LMTokenBatch | Mapping[str, Any] | mx.array,
    ) -> ProductionTrainingLossBreakdown:
        values = _stage1_objective_batch(batch)
        if (
            model.config.attention_mode == "dsa"
            and model.config.attention_sparse_topk != self.graph_config.topk
        ):
            raise ValueError(
                "production Stage-1 graph ranking topk differs from model DSA topk: "
                f"{self.graph_config.topk} != {model.config.attention_sparse_topk}"
            )
        graph_relations = (
            self.graph_config.relations
            if model.config.attention_mode == "dsa"
            else ()
        )
        schedule_assignments = self.schedule_assignments
        if schedule_assignments is None:
            schedule_assignments = canonical_batch_shape_assignment_receipts(
                objective=PREMATERIALIZED_OBJECTIVE,
                batch_size=int(values.input_ids.shape[0]),
                input_tokens=int(values.input_ids.shape[1]),
                loss_tokens=1,
                graph_relations=graph_relations,
            )
        if len(schedule_assignments) != int(values.input_ids.shape[0]):
            raise ValueError(
                "production Stage-1 schedule receipt count does not match batch"
            )
        for assignment in schedule_assignments:
            validate_objective_assignment_receipt(
                assignment,
                graph_relations=graph_relations,
            )
        if model.config.attention_mode == "gqa":
            # Dense GQA has no learned indexer score tensor. Graph routes still
            # condition the full-causal decoder, while DSA-only graph metrics
            # remain zero instead of re-labelling route eligibility as loss.
            return scheduled_production_training_loss_breakdown(
                model,
                values.input_ids,
                values.targets,
                values.loss_mask,
                side_channels=values.side_channels,
                document_ids=values.document_ids,
                block_bias=values.relation_bias,
                edge_kind_bias=values.edge_kind_bias,
                graph_targets=None,
                graph_pair_mask=None,
                graph_config=None,
                graph_weight=0.0,
                graph_relations=(),
                schedule_assignments=schedule_assignments,
                require_schedule_receipt=True,
            )
        if model.config.attention_mode != "dsa":
            raise ValueError(
                "production Stage-1 supports only attention_mode='gqa' or 'dsa'"
            )
        return scheduled_production_training_loss_breakdown(
            model,
            values.input_ids,
            values.targets,
            values.loss_mask,
            side_channels=values.side_channels,
            document_ids=values.document_ids,
            block_bias=values.relation_bias,
            edge_kind_bias=values.edge_kind_bias,
            graph_targets=values.graph_targets,
            graph_pair_mask=values.graph_pair_mask,
            graph_config=self.graph_config,
            graph_weight=self.graph_config.global_weight,
            graph_relations=self.graph_config.relations,
            schedule_assignments=schedule_assignments,
            require_schedule_receipt=True,
        )

    def __call__(
        self,
        model: DenseCppLM,
        batch: LMTokenBatch | Mapping[str, Any] | mx.array,
    ) -> tuple[mx.array, mx.array]:
        breakdown = self.loss_breakdown(model, batch)
        return breakdown.total, breakdown.ntokens

    def receipt(
        self,
        *,
        attention_mode: ProductionAttentionMode,
    ) -> dict[str, Any]:
        if attention_mode == "gqa":
            return {
                "name": STAGE1_GQA_ROUTE_OBJECTIVE,
                "attention_mode": attention_mode,
                "formula": "lm_ce",
                "graph_route": "dense_additive_bias_full_causal",
                "graph_supervision": "none",
                "graph_auxiliary_enabled": False,
                "route_only": True,
                "single_decoder_forward": True,
                "schedule": "canonical_objective_schedule_receipt",
                "total_loss_path": "scheduled_production_training_loss_breakdown",
            }
        if attention_mode != "dsa":
            raise ValueError(
                "production Stage-1 receipt requires attention_mode='gqa' or 'dsa'"
            )
        config = self.graph_config
        return {
            "name": STAGE1_DSA_GRAPH_OBJECTIVE,
            "attention_mode": attention_mode,
            "recipe": stage1_graph_recipe_binding(),
            "bias_beta": STAGE1_GRAPH_BIAS_BETA,
            "score_formula": STAGE1_GRAPH_SCORE_FORMULA,
            "score_stage": STAGE1_GRAPH_SCORE_STAGE,
            "formula": "lm_ce + graph_edge_bce + graph_ranking",
            "graph_route": "dsa_sparse_indexer",
            "graph_supervision": "dsa_indexer",
            "graph_auxiliary_enabled": True,
            "route_only": False,
            "relations": list(config.relations),
            "topk": config.topk,
            "global_weight": config.global_weight,
            "indexer_weight": config.indexer_weight,
            "layer_weight": config.layer_weight,
            "edge_bce_weight": config.bce_weight,
            "ranking_weight": config.coverage_weight,
            "positive_weight": config.pos_weight,
            "ranking_margin": config.margin,
            "single_decoder_forward": True,
            "schedule": "canonical_objective_schedule_receipt",
            "total_loss_path": "scheduled_production_training_loss_breakdown",
        }


def _stage1_objective_batch(
    batch: LMTokenBatch | Mapping[str, Any] | mx.array,
) -> _Stage1ObjectiveBatch:
    normalized = normalize_compiled_batch(batch, graph_routes_enabled=True)
    lm_batch = ensure_lm_batch(normalized)
    model_kwargs = lm_batch.model_kwargs()
    relation_bias = model_kwargs.pop("block_bias", None)
    edge_kind_bias = model_kwargs.pop("edge_kind_bias", None)
    document_ids = lm_batch.input_document_ids
    if not isinstance(relation_bias, mx.array):
        raise ValueError(
            "production Stage-1 objective requires a graph relation prior from "
            "ProductionMegatronDataset graph sidecars"
        )
    if not isinstance(edge_kind_bias, mx.array):
        raise ValueError(
            "production Stage-1 objective requires a graph edge-kind prior from "
            "ProductionMegatronDataset graph sidecars"
        )
    if not isinstance(document_ids, mx.array):
        raise ValueError("production Stage-1 objective requires document_ids")

    input_ids = lm_batch.inputs
    expected_graph_shape = (
        int(input_ids.shape[0]),
        int(input_ids.shape[1]),
        int(input_ids.shape[1]),
    )
    for name, value in (
        ("graph relation prior", relation_bias),
        ("graph edge-kind prior", edge_kind_bias),
    ):
        if tuple(value.shape) != expected_graph_shape:
            raise ValueError(
                f"production Stage-1 {name} must have shape {expected_graph_shape}, "
                f"got {tuple(value.shape)}"
            )

    sequence_length = int(input_ids.shape[1])
    positions = mx.arange(sequence_length, dtype=mx.int32)
    causal = positions[:, None] >= positions[None, :]
    same_document = document_ids[:, :, None] == document_ids[:, None, :]
    pair_mask = causal[None, :, :] & same_document
    if lm_batch.attention_mask is not None:
        token_mask = lm_batch.attention_mask
        if tuple(token_mask.shape) == tuple(lm_batch.tokens.shape):
            token_mask = token_mask[:, :sequence_length]
        elif tuple(token_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                "production Stage-1 attention_mask cannot align to model inputs: "
                f"{tuple(token_mask.shape)} vs {tuple(input_ids.shape)}"
            )
        active = token_mask > 0
        pair_mask = pair_mask & active[:, :, None] & active[:, None, :]

    graph_targets = (relation_bias != 0) & pair_mask
    eligible_rows = mx.any(graph_targets, axis=(1, 2))
    pair_mask = pair_mask & eligible_rows[:, None, None]

    return _Stage1ObjectiveBatch(
        input_ids=input_ids,
        targets=lm_batch.targets,
        loss_mask=lm_batch.target_mask,
        document_ids=document_ids,
        relation_bias=relation_bias,
        edge_kind_bias=edge_kind_bias,
        graph_targets=graph_targets.astype(mx.float32),
        graph_pair_mask=pair_mask.astype(mx.float32),
        side_channels={
            name: value
            for name, value in model_kwargs.items()
            if isinstance(value, mx.array)
        },
    )


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
        "attention_sparse_topk": STAGE1_GRAPH_TOPK,
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
    """Validate one startup batch and receipt route vs. active loss signals."""

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
    objective_values = _stage1_objective_batch(lm_batch)
    graph_route_positive_pairs_array = mx.sum(objective_values.graph_targets)

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
        graph_route_positive_pairs_array,
        domain_tokens_array,
    )
    graph_prior_nonzero = int(graph_prior_nonzero_array.item())
    edge_kind_prior_nonzero = int(edge_kind_prior_nonzero_array.item())
    graph_route_positive_pairs = int(graph_route_positive_pairs_array.item())
    domain_tokens_nonzero = int(domain_tokens_array.item())
    if domain_tokens_nonzero <= 0:
        raise ValueError("production Stage-1 requires nonzero domain tokens")

    return {
        "recipe": STAGE1_GRAPH_DOMAIN_RECIPE,
        "graph_recipe": stage1_graph_recipe_binding(),
        "bias_beta": STAGE1_GRAPH_BIAS_BETA,
        "score_formula": STAGE1_GRAPH_SCORE_FORMULA,
        "score_stage": STAGE1_GRAPH_SCORE_STAGE,
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
        "graph_route_positive_pairs": graph_route_positive_pairs,
        "graph_supervision_positive_pairs": (
            graph_route_positive_pairs if config.attention_mode == "dsa" else 0
        ),
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
        default="dsa",
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
    attention_mode: ProductionAttentionMode = "dsa",
    compile: bool = True,
    bf16: bool = False,
    bundle_hash_jobs: int = 4,
    graph_loss_config: GraphAuxLossConfig = STAGE1_PRODUCTION_GRAPH_LOSS,
) -> dict[str, Any]:
    """Run Stage-1 only after immutable bundle and restore validation."""

    if steps < 1:
        raise ValueError("production Stage-1 steps must be positive")
    objective = Stage1ProductionObjective(graph_loss_config)
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
        attention_sparse_topk=graph_loss_config.topk,
    )
    if not isinstance(dataset.metadata, ProductionMegatronDatasetMetadata):
        raise RuntimeError("production Stage-1 dataset lost immutable provenance metadata")
    startup = {
        **stage1_production_batch_receipt(first_batch, config=model.config),
        **dataset.metadata.provenance_receipt(),
        "loss_objective": objective.receipt(
            attention_mode=model.config.attention_mode
        ),
    }
    print("[stage1-production-startup] " + json.dumps(startup, sort_keys=True), flush=True)

    optimizer = optim.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )
    stepper = CompiledPretrainingStep(
        model,
        optimizer,
        loss_fn=objective,
        compile=compile,
    )
    batch_iter = dataset.iter_batches(loop=True)
    observed_edges = 0
    observed_graph_route_positive_pairs = 0
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
        observed_graph_route_positive_pairs += counts[
            "graph_route_positive_pairs"
        ]
        observed_edge_kind_edges += counts["edge_kind_edges"]
        observed_graph_prior_nonzero += counts["graph_prior_nonzero"]
        observed_edge_kind_prior_nonzero += counts["edge_kind_prior_nonzero"]
        observed_domain_tokens += counts["domain_tokens_nonzero"]
        observed_document_boundaries += counts["document_boundaries"]
        last_loss = stepper(current).loss
        last_domain_residual_l1 = _domain_residual_l1(model, current)
    if (
        observed_edges <= 0
        or observed_graph_route_positive_pairs <= 0
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
        "observed_graph_route_positive_pairs": observed_graph_route_positive_pairs,
        "observed_graph_supervision_positive_pairs": (
            observed_graph_route_positive_pairs
            if model.config.attention_mode == "dsa"
            else 0
        ),
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
        config.attention_mode not in ("gqa", "dsa")
        or not config.graph_routes_enabled
        or not config.require_graph_routes
        or config.graph_attention_bias_beta != PRODUCTION_GRAPH_BETA
        or config.domain_residual_scale != PRODUCTION_DOMAIN_RESIDUAL_SCALE
        or not config.require_domain_routes
    ):
        raise ValueError(
            "batch receipt requires the canonical Stage-1 graph/domain config "
            "with attention_mode='gqa' or 'dsa'"
        )


def _validate_edge_kind_categories(
    graph_batch: GraphBatch,
) -> tuple[int, list[int]]:
    graph_edges = sum(
        len(edge.to_pairs())
        for graph in graph_batch.graphs
        for edge in graph.edges.values()
    )
    if not graph_batch.edge_kinds:
        if graph_edges == 0:
            return 0, []
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
    if edge_count <= 0 and graph_edges > 0:
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
    objective_values = _stage1_objective_batch(lm_batch)
    graph_route_positive_pairs = mx.sum(objective_values.graph_targets)
    mx.eval(count, relation_count, kind_count, graph_route_positive_pairs)
    return {
        "graph_edges": graph_edges,
        "edge_kind_edges": edge_kind_edges,
        "graph_prior_nonzero": int(relation_count.item()),
        "graph_route_positive_pairs": int(graph_route_positive_pairs.item()),
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
    "STAGE1_DSA_GRAPH_OBJECTIVE",
    "STAGE1_GRAPH_DOMAIN_RECIPE",
    "STAGE1_GQA_ROUTE_OBJECTIVE",
    "STAGE1_PRODUCTION_GRAPH_LOSS",
    "STAGE1_PRODUCTION_GRAPH_RELATIONS",
    "Stage1ProductionObjective",
    "add_stage1_production_arguments",
    "build_stage1_production_model",
    "run_stage1_graph_domain_production",
    "stage1_production_batch_receipt",
    "stage1_production_config",
    "validate_stage1_production_model",
]
