"""Pydantic v2 schema for the Visual Builder JSON-RPC 2.0 contract.

Wire format follows VisualBuilderPlan.md §5.2. Every request/response
is a frozen, deterministic Pydantic model — the GUI emits JSON, the
backend consumes JSON, never .py codegen.

All payload shapes are GUI-stable: schema changes go through
SCHEMA_VERSION bump and a corresponding TS type regen in the frontend.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION: str = "1.1.1"

JsonRpcVersion = Literal["2.0"]


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 envelope.
# ---------------------------------------------------------------------------


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request envelope."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: JsonRpcVersion = "2.0"
    id: str | int
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object."""

    model_config = ConfigDict(extra="forbid")

    code: int
    message: str
    data: dict[str, Any] | None = None


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response envelope (success XOR error)."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: JsonRpcVersion = "2.0"
    id: str | int | None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


# ---------------------------------------------------------------------------
# Standard JSON-RPC error codes + our extensions.
# ---------------------------------------------------------------------------


class ErrorCode:
    """Reserved + project-specific JSON-RPC error codes."""

    # Standard JSON-RPC 2.0 reserved range -32700..-32000.
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # cppmega Visual Builder extensions (-32000..-32099).
    RESOLVE_ERROR = -32001
    BUILD_ERROR = -32002
    PROBE_ERROR = -32003
    PIPELINE_ERROR = -32004


# ---------------------------------------------------------------------------
# Domain payload shapes — graph, loss, optim, sharding.
# ---------------------------------------------------------------------------


class NodeSpec(BaseModel):
    """One brick node in the canvas graph."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class EdgeSpec(BaseModel):
    """One directed edge: src producer → dst consumer."""

    model_config = ConfigDict(extra="forbid")

    src: str
    dst: str


class GraphSpec(BaseModel):
    """Linear or parallel-block topology in wire form."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[NodeSpec]
    edges: list[EdgeSpec] = Field(default_factory=list)


class LossSpecPayload(BaseModel):
    """Wire form of LossSpec (matches buildspec.LossSpec)."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    head_outputs: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class ScheduleSpecPayload(BaseModel):
    """Wire form of ScheduleSpec (cppmega_v4.buildspec.schedules)."""

    model_config = ConfigDict(extra="forbid")

    kind: str = "constant"
    warmup_steps: int = 0
    total_steps: int | None = None
    min_lr_ratio: float = 0.1
    decay_steps: int | None = None
    power: float = 2.0


class ParamGroupPayload(BaseModel):
    """One optimizer param group with matcher + hyperparams."""

    model_config = ConfigDict(extra="forbid")

    matcher: str
    lr: float
    weight_decay: float = 0.0
    betas: tuple[float, float] | None = None
    ns_steps: int | None = None
    schedule: ScheduleSpecPayload | None = None


class OptimSpecPayload(BaseModel):
    """Wire form of OptimSpec."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    groups: list[ParamGroupPayload]
    # V5-G23: UI gradient_clip_norm passed through to backend OptimSpec
    # so stage_train can apply L2-norm clipping. None disables.
    gradient_clip_norm: float | None = 1.0
    mixed_precision: bool = True


def _default_optim_payload() -> OptimSpecPayload:
    return OptimSpecPayload(
        kind="adamw",
        groups=[
            ParamGroupPayload(
                matcher="all",
                lr=1e-3,
                weight_decay=0.01,
                betas=(0.9, 0.95),
            ),
        ],
    )


class TopologyPayload(BaseModel):
    """How to instantiate a DeviceTopology — factory + kwargs."""

    model_config = ConfigDict(extra="forbid")

    factory: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class AxisAssignmentPayload(BaseModel):
    """One axis of the parallelism plan."""

    model_config = ConfigDict(extra="forbid")

    axis_name: str
    kind: str
    degree: int


class ShardingSpecPayload(BaseModel):
    """Wire form of ShardingSpec."""

    model_config = ConfigDict(extra="forbid")

    topology: TopologyPayload
    axis_assignments: list[AxisAssignmentPayload]
    # V4-13: 'whole_model' is canonical for both UI TopBar + gotcha_checker
    # conditions. Schema previously rejected it as not in {regional, global,
    # off}, blocking UI from ever triggering the fsdp2_whole_compile /
    # megatron_tp_whole_compile error gotchas. 'global' kept as back-compat.
    compile_mode: Literal["regional", "global", "off", "whole_model"] = "regional"
    fp8_enabled: bool = False


class RewriterPayload(BaseModel):
    """One rewriter to apply, with parameters."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


SideChannelModePayload = Literal["off", "auto", "require", "if_available"]
SideChannelEmbeddingPayload = Literal[
    "categorical", "numeric_bucket", "span", "edge_bias", "none"
]
SideChannelFallbackPayload = Literal[
    "zeros", "unknown_id", "drop_family", "error"
]
InferenceEnrichmentSourcePayload = Literal[
    "none", "prompt_only", "parse_if_possible", "project_index", "auto"
]
InferenceFailPolicyPayload = Literal["drop_family", "text_only", "error"]
PackingPolicyPayload = Literal["sequential", "best_fit"]


class FamilySpecPayload(BaseModel):
    """Wire form of one side-channel family policy."""

    model_config = ConfigDict(extra="forbid")

    mode: SideChannelModePayload = "if_available"
    columns: list[str] = Field(default_factory=list)
    embedding: SideChannelEmbeddingPayload = "categorical"
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    residual_scale: float = Field(default=1.0, ge=0.0)
    fallback: SideChannelFallbackPayload = "drop_family"
    language_scope: list[str] = Field(default_factory=lambda: ["any"])


class InferenceEnrichmentSpecPayload(BaseModel):
    """Wire form of inference-time side-channel enrichment policy."""

    model_config = ConfigDict(extra="forbid")

    source: InferenceEnrichmentSourcePayload = "auto"
    fail_policy: InferenceFailPolicyPayload = "drop_family"
    timeout_ms: int = Field(default=500, ge=0)
    cache_enabled: bool = True


def _default_side_channel_family_payloads() -> dict[str, FamilySpecPayload]:
    return {
        "platform": FamilySpecPayload(
            mode="auto",
            columns=["platform_ids", "source_platform_ids"],
            embedding="categorical",
            dropout=0.10,
        ),
        "syntax": FamilySpecPayload(
            mode="if_available",
            columns=[
                "token_ast_depth",
                "token_sibling_index",
                "token_ast_node_type",
            ],
            embedding="categorical",
            dropout=0.25,
        ),
        "structure": FamilySpecPayload(
            mode="if_available",
            columns=[
                "token_structure_ids",
                "token_dep_levels",
                "token_chunk_starts",
                "token_chunk_ends",
                "token_chunk_kinds",
                "token_chunk_dep_levels",
            ],
            embedding="categorical",
            dropout=0.25,
        ),
        "semantic_graph": FamilySpecPayload(
            mode="if_available",
            columns=[
                "token_symbol_ids",
                "token_call_targets",
                "token_type_refs",
                "token_def_use",
                "token_call_edges",
                "token_type_edges",
            ],
            embedding="edge_bias",
            dropout=0.50,
        ),
        "temporal_diff": FamilySpecPayload(
            mode="off",
            columns=[
                "token_change_mask_pre",
                "token_change_mask_post",
                "hunk_id_per_token",
                "edit_op_per_token",
            ],
            embedding="categorical",
            dropout=0.0,
        ),
    }


class SideChannelSpecPayload(BaseModel):
    """Wire form of the generic side-channel conditioning policy."""

    model_config = ConfigDict(extra="forbid")

    mode: SideChannelModePayload = "auto"
    families: dict[str, FamilySpecPayload] = Field(
        default_factory=_default_side_channel_family_payloads
    )
    inference: InferenceEnrichmentSpecPayload = Field(
        default_factory=InferenceEnrichmentSpecPayload
    )


class DataMaterializationSpecPayload(BaseModel):
    """Wire form of packed-row parquet materialization policy."""

    model_config = ConfigDict(extra="forbid")

    packing_policy: PackingPolicyPayload = "best_fit"
    max_seq_len: int = Field(default=4096, gt=0)
    pad_to_max: bool = True
    include_provenance: bool = True
    required_token_fields: list[str] = Field(default_factory=lambda: [
        "input_ids",
        "target_ids",
        "loss_mask",
        "doc_ids",
        "pack_id",
        "valid_token_count",
        "num_docs",
    ])


# ---------------------------------------------------------------------------
# `verify` — primary endpoint, full validation pass.
# ---------------------------------------------------------------------------


class VerifyParams(BaseModel):
    """Input to ``verify`` — full spec snapshot from the GUI."""

    model_config = ConfigDict(extra="forbid")

    graph: GraphSpec
    dim_env: dict[str, int]
    loss: LossSpecPayload
    optim: OptimSpecPayload = Field(default_factory=_default_optim_payload)
    rewriters: list[RewriterPayload] = Field(default_factory=list)
    sharding: ShardingSpecPayload | None = None
    training: bool = True
    side_channels: SideChannelSpecPayload = Field(
        default_factory=SideChannelSpecPayload
    )
    data_materialization: DataMaterializationSpecPayload = Field(
        default_factory=DataMaterializationSpecPayload
    )
    available_side_channels: list[str] = Field(default_factory=lambda: ["doc_ids", "token_ids"])


class EdgeResolution(BaseModel):
    """One resolved edge with shape + match severity."""

    model_config = ConfigDict(extra="forbid")

    src: str
    dst: str
    shape: list[int]
    matched: bool
    severity: Literal["info", "warning", "error"]


class Diagnostic(BaseModel):
    """One diagnostic entry attached to a node/edge."""

    model_config = ConfigDict(extra="forbid")

    severity: Literal["info", "warning", "error"]
    component: str
    message: str
    suggested_fix: str | None = None


class ResolvedGraph(BaseModel):
    """Resolved-shapes view of the graph."""

    model_config = ConfigDict(extra="forbid")

    edges: list[EdgeResolution]
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    has_errors: bool = False


class PerBrickMemory(BaseModel):
    """Per-brick memory breakdown."""

    model_config = ConfigDict(extra="forbid")

    params_bytes: int
    activations_bytes: int
    kv_cache_bytes: int = 0


class WorstRankMemory(BaseModel):
    """Per-rank memory totals at the worst rank in the topology."""

    model_config = ConfigDict(extra="forbid")

    weights_bytes: int = 0
    grads_bytes: int = 0
    optimizer_state_bytes: int = 0
    activations_bytes: int = 0
    fsdp_allgather_peak_bytes: int = 0
    kv_cache_bytes: int = 0
    moe_routing_buffers_bytes: int = 0
    collective_workspace_bytes: int = 0
    framework_overhead_bytes: int = 0
    master_weights_bytes: int = 0
    total_bytes: int = 0


class DistributedMemoryPayload(BaseModel):
    """Wire form of DistributedMemoryReport."""

    model_config = ConfigDict(extra="forbid")

    worst_rank_idx: int = 0
    worst_rank: WorstRankMemory = Field(default_factory=WorstRankMemory)
    duplication_bytes: int = 0
    master_weights_overhead_bytes: int = 0
    kernel_boundary_materialisation_bytes: int = 0
    fits_on_topology: bool = True


class GotchaPayload(BaseModel):
    """One gotcha firing from the gotcha-checker table."""

    model_config = ConfigDict(extra="forbid")

    id: str
    severity: Literal["info", "warning", "error"]
    message: str
    reference: str | None = None


class FusionRegionPayload(BaseModel):
    """One fused/non-fused region in the plan."""

    model_config = ConfigDict(extra="forbid")

    brick_names: list[str]
    backend: str
    is_fused: bool
    estimated_savings_us: float = 0.0


class InferenceEntryPayload(BaseModel):
    """One dimension-inference entry (E7-2). Surfaces in the
    Dimensions sidebar tab so users see why num_heads=2 came out as it
    did (e.g. H=128/head_dim=64 → 2)."""

    model_config = ConfigDict(extra="forbid")

    brick: str
    param: str
    value: Any
    source: Literal["user", "auto"]
    reason: str


class VerifyResult(BaseModel):
    """Wire form of one ``verify`` response."""

    model_config = ConfigDict(extra="forbid")

    resolved: ResolvedGraph
    memory_per_brick: dict[str, PerBrickMemory]
    memory_distributed: DistributedMemoryPayload | None = None
    gotchas: list[GotchaPayload] = Field(default_factory=list)
    fusion_plan: list[FusionRegionPayload] = Field(default_factory=list)
    inference_log: list[InferenceEntryPayload] = Field(default_factory=list)
    elapsed_ms: float


# ---------------------------------------------------------------------------
# `suggest_sharding` — proposals for the current model.
# ---------------------------------------------------------------------------


class ShardingProposalPayload(BaseModel):
    """One sharding proposal with fit + reason."""

    model_config = ConfigDict(extra="forbid")

    strategy_name: str
    fits: bool
    estimated_per_rank_bytes: int
    reason: str
    num_errors: int = 0
    sharding: ShardingSpecPayload


class SuggestShardingParams(BaseModel):
    """Input to ``suggest_sharding``."""

    model_config = ConfigDict(extra="forbid")

    graph: GraphSpec
    dim_env: dict[str, int]
    loss: LossSpecPayload
    optim: OptimSpecPayload
    topology: TopologyPayload


class SuggestShardingResult(BaseModel):
    """List of ranked proposals."""

    model_config = ConfigDict(extra="forbid")

    proposals: list[ShardingProposalPayload]
    elapsed_ms: float


# ---------------------------------------------------------------------------
# `suggest_adapters` — for one mismatched edge.
# ---------------------------------------------------------------------------


class AdapterStepPayload(BaseModel):
    """One adapter step in a suggested chain."""

    model_config = ConfigDict(extra="forbid")

    rule: str
    description: str
    params: dict[str, Any] = Field(default_factory=dict)


class SuggestAdaptersParams(BaseModel):
    """Input — point at an edge + spec snapshot."""

    model_config = ConfigDict(extra="forbid")

    graph: GraphSpec
    dim_env: dict[str, int]
    producer: str
    consumer: str
    max_steps: int = 4


class SuggestAdaptersResult(BaseModel):
    """Suggested adapter chain for the edge."""

    model_config = ConfigDict(extra="forbid")

    producer: str
    consumer: str
    producer_shape: list[int]
    consumer_shape: list[int]
    chain: list[AdapterStepPayload]
    reason: str


# ---------------------------------------------------------------------------
# `build_preset_specs` — drag a preset into the canvas.
# ---------------------------------------------------------------------------


class BuildPresetSpecsParams(BaseModel):
    """Input — preset name + tiny hidden for instantiation preview."""

    model_config = ConfigDict(extra="forbid")

    preset_name: str
    hidden_size: int
    num_layers: int | None = None


class BuildPresetSpecsResult(BaseModel):
    """Output — wire-form specs (leaf or parallel-block)."""

    model_config = ConfigDict(extra="forbid")

    specs: list[dict[str, Any]]
    preset_name: str


# ---------------------------------------------------------------------------
# `probe.run` — bridge to Contract Probe.
# ---------------------------------------------------------------------------


class ProbeRunParams(BaseModel):
    """Input — full spec + paths to tokenizer + parquet."""

    model_config = ConfigDict(extra="forbid")

    graph: GraphSpec
    dim_env: dict[str, int]
    loss: LossSpecPayload
    optim: OptimSpecPayload
    tokenizer_source: str
    parquet_path: str
    probe_hidden_size: int = 64
    run_dry_forward: bool = True


# Result is the ContractProbeReport already JSON-shaped by
# cppmega_v4.probe.to_dict — surfaced opaque to keep the schemas independent.


class ProbeRunResult(BaseModel):
    """Opaque container — Contract Probe owns the inner shape."""

    model_config = ConfigDict(extra="allow")

    schema_version: str
    is_clean: bool
    elapsed_ms: float


# ---------------------------------------------------------------------------
# `pipeline.run` — orchestrator over stages.
# ---------------------------------------------------------------------------


class StageOptions(BaseModel):
    """Open per-stage options bag."""

    model_config = ConfigDict(extra="allow")


class PipelinePayload(BaseModel):
    """Pipeline manifest — what stages, in order, with options."""

    model_config = ConfigDict(extra="forbid")

    stages: list[str]
    stage_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    continue_on_failure: bool = False


class PipelineRunParams(BaseModel):
    """Input — full spec + pipeline."""

    model_config = ConfigDict(extra="forbid")

    spec: VerifyParams
    pipeline: PipelinePayload


class PipelineAbortParams(BaseModel):
    """Request cancellation of an in-flight pipeline train stage."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)


class PipelineAbortResult(BaseModel):
    """Acknowledgement that the train abort token has been set."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["abort_requested"] = "abort_requested"
    run_id: str


class StageResult(BaseModel):
    """One stage execution result."""

    model_config = ConfigDict(extra="allow")

    name: str
    status: Literal["ok", "skipped", "fail", "cancelled"]
    elapsed_ms: float
    warnings: int = 0
    errors: int = 0
    error: dict[str, Any] | None = None


class PipelineRunResult(BaseModel):
    """Pipeline-level rollup."""

    model_config = ConfigDict(extra="forbid")

    stages: list[StageResult]
    overall_status: Literal["ok", "fail", "cancelled"]
    total_elapsed_ms: float


# ---------------------------------------------------------------------------
# Event taxonomy — strings only; payloads handled per-method.
# ---------------------------------------------------------------------------


EVENT_TAXONOMY: frozenset[str] = frozenset({
    "node.move",
    "graph.mutate",
    "param.edit",
    "loss.update",
    "optim.update",
    "rewriter.add",
    "rewriter.remove",
    "rewriter.reorder",
    "sharding.update",
    "verify.request",
    "sharding.request",
    "build.request",
    "backend.status",
    "probe.run",
    "pipeline.run",
    "pipeline.abort",
})


METHOD_REGISTRY: frozenset[str] = frozenset({
    "verify",
    "suggest_sharding",
    "suggest_adapters",
    "build_preset_specs",
    "probe.run",
    "pipeline.run",
    "pipeline.abort",
    "backend.status",
    "tokenizer.encode_visualize",
    "tokenizer.list_presets",
    "data.preview_parquet",
    "catalog.explain",
    "catalog.list_options",
    "architectures.list_presets",
    "suggest_optim_groups",
    "data.roundtrip_check",
    "ablation.run",
    "ckpt.inspect",
    "dtype.cost_estimate",
    "pipeline.pause",
    "pipeline.resume",
    "gen.run",
})


class ArchitecturesListPresetsResult(BaseModel):
    """architectures.list_presets — sorted preset names from
    cppmega_v4.architectures.PRESETS. UI calls this once on mount to
    populate the preset launcher dropdown dynamically."""

    model_config = ConfigDict(extra="forbid")

    presets: list[str] = Field(default_factory=list)


class SuggestOptimGroupsParams(BaseModel):
    """suggest_optim_groups — auto-classify graph parameters into
    matcher-based optimizer groups (E7-4)."""

    model_config = ConfigDict(extra="forbid")

    graph: GraphSpec
    optim_kind: str = "muon_adamw_hybrid"
    hidden_size: int = 128


class ProposedGroupPayload(BaseModel):
    """Wire form of group_inference.ProposedGroup."""

    model_config = ConfigDict(extra="forbid")

    matcher: str
    optim_kind: str
    lr: float
    weight_decay: float
    betas: tuple[float, float] | None = None
    ns_steps: int | None = None
    param_count: int
    rationale: str


class SuggestOptimGroupsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[ProposedGroupPayload] = Field(default_factory=list)
    total_params: int = 0
    uncovered_params: int = 0


class CatalogExplainParams(BaseModel):
    """catalog.explain — fetch one ExplainEntry."""

    model_config = ConfigDict(extra="forbid")

    category: str
    name: str


class CatalogExplainEntryPayload(BaseModel):
    """Wire form of ExplainEntry."""

    model_config = ConfigDict(extra="forbid")

    category: str
    name: str
    summary: str
    when_to_use: str
    when_to_avoid: str
    recommended_params: dict[str, Any] = Field(default_factory=dict)
    paper_ref: str | None = None
    paper_url: str | None = None
    gotchas: list[str] = Field(default_factory=list)


class CatalogExplainResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry: CatalogExplainEntryPayload | None = None
    not_found_message: str | None = None


class CatalogListOptionsParams(BaseModel):
    """catalog.list_options — fetch every entry in a category."""

    model_config = ConfigDict(extra="forbid")

    category: str


class CatalogOptionSummary(BaseModel):
    """Compact summary used in dropdowns (no full entry to keep payload
    small when listing many options)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    summary: str
    paper_ref: str | None = None


class CatalogListOptionsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    options: list[CatalogOptionSummary] = Field(default_factory=list)
