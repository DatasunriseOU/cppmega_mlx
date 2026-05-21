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


SCHEMA_VERSION: str = "1.0.0"

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


class ParamGroupPayload(BaseModel):
    """One optimizer param group with matcher + hyperparams."""

    model_config = ConfigDict(extra="forbid")

    matcher: str
    lr: float
    weight_decay: float = 0.0
    betas: tuple[float, float] | None = None


class OptimSpecPayload(BaseModel):
    """Wire form of OptimSpec."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    groups: list[ParamGroupPayload]


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
    compile_mode: Literal["regional", "global", "off"] = "regional"
    fp8_enabled: bool = False


class RewriterPayload(BaseModel):
    """One rewriter to apply, with parameters."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# `verify` — primary endpoint, full validation pass.
# ---------------------------------------------------------------------------


class VerifyParams(BaseModel):
    """Input to ``verify`` — full spec snapshot from the GUI."""

    model_config = ConfigDict(extra="forbid")

    graph: GraphSpec
    dim_env: dict[str, int]
    loss: LossSpecPayload
    optim: OptimSpecPayload
    rewriters: list[RewriterPayload] = Field(default_factory=list)
    sharding: ShardingSpecPayload | None = None
    training: bool = True
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


class VerifyResult(BaseModel):
    """Wire form of one ``verify`` response."""

    model_config = ConfigDict(extra="forbid")

    resolved: ResolvedGraph
    memory_per_brick: dict[str, PerBrickMemory]
    memory_distributed: DistributedMemoryPayload | None = None
    gotchas: list[GotchaPayload] = Field(default_factory=list)
    fusion_plan: list[FusionRegionPayload] = Field(default_factory=list)
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


class StageResult(BaseModel):
    """One stage execution result."""

    model_config = ConfigDict(extra="allow")

    name: str
    status: Literal["ok", "skipped", "fail"]
    elapsed_ms: float
    warnings: int = 0
    errors: int = 0
    error: dict[str, Any] | None = None


class PipelineRunResult(BaseModel):
    """Pipeline-level rollup."""

    model_config = ConfigDict(extra="forbid")

    stages: list[StageResult]
    overall_status: Literal["ok", "fail"]
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
})


METHOD_REGISTRY: frozenset[str] = frozenset({
    "verify",
    "suggest_sharding",
    "suggest_adapters",
    "build_preset_specs",
    "probe.run",
    "pipeline.run",
    "backend.status",
})
