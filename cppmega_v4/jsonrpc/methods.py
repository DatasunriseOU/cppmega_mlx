"""Pure-Python JSON-RPC method handlers.

Each handler takes a Pydantic params model and returns a Pydantic
result model. No HTTP, no async — that's the server layer's job. This
split keeps the handlers fully unit-testable without spinning a server.

Cache integration: each handler accepts an optional ``cache`` argument.
When provided, the canonical SHA-256 of the params is looked up first
and the handler is short-circuited on hit. Mutations never go through
the cache.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from cppmega_v4.architectures.presets import (
    PRESETS,
    build_preset_specs as _build_preset_specs,
)
from cppmega_v4.buildspec import (
    LossKind,
    LossSpec,
    ModelBuildSpec,
    OptimKind,
    OptimSpec,
    ParamGroup,
)
from cppmega_v4.buildspec.schedules import ScheduleSpec
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.parallelism import (
    AxisAssignment,
    ParallelismKind,
    ShardingSpec,
    a100_8x,
    b100_8x,
    gb10_quarter,
    h100_8x,
    h200_8x,
    m3_ultra_solo,
    suggest_sharding as _suggest_sharding,
    tpu_v5p_4,
    tpu_v6e_8,
    verify_distributed_plan,
)
from cppmega_v4.probe import contract_probe as _contract_probe
from cppmega_v4.probe import to_dict as _probe_to_dict
from cppmega_v4.spec import (
    suggest_adapters as _suggest_adapters,
    verify_and_estimate,
)
from cppmega_v4.spec.resolver import resolve_shapes
from cppmega_v4.fusion.auto_planner import plan_fusion_regions

from cppmega_v4.jsonrpc.cache import LRUCache, canonical_sha256
from cppmega_v4.jsonrpc.schema import (
    AxisAssignmentPayload,
    BuildPresetSpecsParams,
    BuildPresetSpecsResult,
    Diagnostic,
    DistributedMemoryPayload,
    EdgeResolution,
    FusionRegionPayload,
    GotchaPayload,
    GraphSpec,
    LossSpecPayload,
    OptimSpecPayload,
    PerBrickMemory,
    ProbeRunParams,
    ProbeRunResult,
    ResolvedGraph,
    ShardingProposalPayload,
    ShardingSpecPayload,
    SuggestAdaptersParams,
    SuggestAdaptersResult,
    SuggestShardingParams,
    SuggestShardingResult,
    TopologyPayload,
    VerifyParams,
    VerifyResult,
    WorstRankMemory,
)


_TOPOLOGY_FACTORIES = {
    "h100_8x": h100_8x,
    "h200_8x": h200_8x,
    "a100_8x": a100_8x,
    "b100_8x": b100_8x,
    "gb10_quarter": gb10_quarter,
    "tpu_v5p_4": tpu_v5p_4,
    "tpu_v6e_8": tpu_v6e_8,
    "m3_ultra_solo": m3_ultra_solo,
}


# ---------------------------------------------------------------------------
# Coercion helpers: wire payload → backend dataclasses.
# ---------------------------------------------------------------------------


def _graph_to_specs(graph: GraphSpec) -> list[dict[str, Any]]:
    """Wire GraphSpec → from_block_specs-compatible list.

    Parallel-block detection: a node whose id has form ``parallel:<n>``
    flags a synthetic group with its outbound children. For v1 we accept
    only linear graphs (edges form a chain). Non-linear shapes are an
    error; the caller should pre-resolve parallel-block emit on the
    frontend.
    """
    return [
        {"kind": n.kind, "name": n.id, "params": dict(n.params)}
        for n in graph.nodes
    ]


def _make_loss(payload: LossSpecPayload) -> LossSpec:
    # V4-7: UI LossTab sends a flattened MTP shape (params={"k": K, "beta": B}
    # and a single head_outputs entry) so that the user doesn't have to spell
    # out beta_0..beta_{K-1} or pre-clone the head. Expand here so the
    # LossSpec __post_init__ contract is satisfied without making the UI
    # know about per-i betas.
    kind = LossKind(payload.kind)
    params = dict(payload.params)
    head_outputs = list(payload.head_outputs)
    if kind is LossKind.MTP_WEIGHTED:
        try:
            k = int(params.get("k", 2))
        except (TypeError, ValueError):
            k = 2
        params["k"] = k
        # If the UI sent a single `beta`, broadcast it to beta_0..beta_{k-1}.
        if "beta" in params and not any(f"beta_{i}" in params for i in range(k)):
            beta_val = float(params.pop("beta"))
            for i in range(k):
                params[f"beta_{i}"] = beta_val
        else:
            for i in range(k):
                params.setdefault(f"beta_{i}", 0.5)
        # Auto-extend head_outputs to length k by repeating the last entry —
        # MTPRewriter will materialise the k heads downstream; the UI only
        # needs to know the seed head.
        if not head_outputs:
            head_outputs = ["mlp"]
        while len(head_outputs) < k:
            head_outputs.append(head_outputs[-1])
        head_outputs = head_outputs[:k]
    return LossSpec(
        kind=kind,
        head_outputs=tuple(head_outputs),
        params=params,
        label_source="next_k_tokens" if kind is LossKind.MTP_WEIGHTED
                                     else "next_token",
    )


def _make_optim(payload: OptimSpecPayload) -> OptimSpec:
    kind = OptimKind(payload.kind)

    def _default_betas() -> tuple[float, float] | None:
        if kind in (OptimKind.LION, OptimKind.LION_8BIT):
            return (0.9, 0.99)
        if kind is OptimKind.ADAM_8BIT:
            return (0.9, 0.999)
        if kind in (OptimKind.ADAMW, OptimKind.MUON_ADAMW_HYBRID):
            return (0.9, 0.95)
        return None

    def _make_schedule(g) -> ScheduleSpec | None:
        schedule = g.schedule
        if schedule is None:
            return None
        return ScheduleSpec(
            kind=schedule.kind,
            warmup_steps=schedule.warmup_steps,
            total_steps=schedule.total_steps,
            min_lr_ratio=schedule.min_lr_ratio,
            decay_steps=schedule.decay_steps,
            power=schedule.power,
        )

    groups = tuple(
        ParamGroup(
            matcher=g.matcher,
            lr=g.lr,
            weight_decay=g.weight_decay,
            betas=g.betas if g.betas is not None else _default_betas(),
            ns_steps=(
                g.ns_steps
                if g.ns_steps is not None
                else (5 if kind is OptimKind.MUON else None)
            ),
            schedule=_make_schedule(g),
        )
        for g in payload.groups
    )
    return OptimSpec(
        kind=kind,
        groups=groups,
        gradient_clip_norm=payload.gradient_clip_norm,
        mixed_precision=payload.mixed_precision,
    )


def _make_topology(payload: TopologyPayload):
    factory = _TOPOLOGY_FACTORIES.get(payload.factory)
    if factory is None:
        raise ValueError(
            f"unknown topology factory {payload.factory!r}; "
            f"available: {sorted(_TOPOLOGY_FACTORIES)}"
        )
    return factory(**payload.kwargs)


def _make_sharding(payload: ShardingSpecPayload) -> ShardingSpec:
    topology = _make_topology(payload.topology)
    axes = tuple(
        AxisAssignment(
            axis_name=a.axis_name,
            kind=ParallelismKind(a.kind),
            degree=a.degree,
        )
        for a in payload.axis_assignments
    )
    return ShardingSpec(
        topology=topology,
        axis_assignments=axes,
        compile_mode=payload.compile_mode,
        fp8_enabled=payload.fp8_enabled,
    )


def _shape_to_list(shape) -> list[int]:
    if shape is None:
        return []
    return [int(d) if isinstance(d, int) else 0 for d in shape]


# ---------------------------------------------------------------------------
# Cache helpers.
# ---------------------------------------------------------------------------


def _cache_key(method: str, params_dict: Mapping[str, Any]) -> str:
    return f"{method}::{canonical_sha256(params_dict)}"


def _cache_lookup(cache: LRUCache | None, method: str, payload: Any):
    if cache is None:
        return None, None
    key = _cache_key(method, payload.model_dump(mode="json"))
    hit = cache.get(key)
    return key, hit


def _cache_store(cache: LRUCache | None, key: str | None, value: Any):
    if cache is not None and key is not None:
        cache.set(key, value)


def _side_channel_policy_gotchas(
    side_channels: Any,
    available: frozenset[str],
) -> list[GotchaPayload]:
    """Report GUI-visible contract errors for required side-channel families."""
    out: list[GotchaPayload] = []
    global_requires = side_channels.mode == "require"
    for family, policy in sorted(side_channels.families.items()):
        if policy.mode == "off":
            continue
        required = policy.mode == "require" or global_requires
        if not required:
            continue
        missing = [col for col in policy.columns if col not in available]
        if not missing:
            continue
        out.append(GotchaPayload(
            id=f"side_channel_required_{family}",
            severity="error",
            message=(
                f"required side-channel family {family!r} is missing "
                f"{', '.join(missing)}"
            ),
            reference="docs/side_channel_conditioning_plan.md#configuration-contract",
        ))
    return out


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def verify(params: VerifyParams, *, cache: LRUCache | None = None) -> VerifyResult:
    """Run verify_and_estimate + (optional) verify_distributed_plan."""
    key, hit = _cache_lookup(cache, "verify", params)
    if hit is not None:
        return hit

    t0 = time.perf_counter()
    specs = _graph_to_specs(params.graph)
    hidden = params.dim_env.get("H", 64)
    graph = from_block_specs(specs, hidden_size=hidden, instantiate=False)

    available = frozenset(params.available_side_channels)
    resolved = resolve_shapes(
        graph, params.dim_env,
        strict=False, available_side_channels=available,
    )
    fusion_plan = tuple(plan_fusion_regions(graph))

    result_one = verify_and_estimate(
        graph,
        dim_env=params.dim_env,
        training=params.training,
        available_side_channels=available,
    )

    mem = result_one.memory
    per_brick: dict[str, PerBrickMemory] = {}
    for name in (n.name for n in graph.nodes):
        rec = mem.per_brick.get(name)
        if rec is None:
            per_brick[name] = PerBrickMemory(params_bytes=0, activations_bytes=0)
        else:
            per_brick[name] = PerBrickMemory(
                params_bytes=int(rec.params_bytes),
                activations_bytes=int(rec.activations_bytes),
                kv_cache_bytes=int(rec.kv_cache_bytes),
            )

    distributed_payload: DistributedMemoryPayload | None = None
    gotcha_payloads: list[GotchaPayload] = []
    if params.sharding is not None:
        sharding = _make_sharding(params.sharding)
        loss = _make_loss(params.loss)
        optim = _make_optim(params.optim)
        build_spec = ModelBuildSpec(graph=graph, loss=loss, optim=optim)
        dverify = verify_distributed_plan(build_spec, sharding, training=params.training)
        d = dverify.memory
        wr = d.worst_rank if hasattr(d, "worst_rank") else None
        distributed_payload = DistributedMemoryPayload(
            worst_rank_idx=int(getattr(d, "worst_rank_idx", 0)),
            worst_rank=WorstRankMemory(
                weights_bytes=int(getattr(wr, "weights_bytes", 0)),
                grads_bytes=int(getattr(wr, "grads_bytes", 0)),
                optimizer_state_bytes=int(getattr(wr, "optimizer_state_bytes", 0)),
                activations_bytes=int(getattr(wr, "activations_bytes", 0)),
                fsdp_allgather_peak_bytes=int(getattr(wr, "fsdp_allgather_peak_bytes", 0)),
                kv_cache_bytes=int(getattr(wr, "kv_cache_bytes", 0)),
                moe_routing_buffers_bytes=int(getattr(wr, "moe_routing_buffers_bytes", 0)),
                collective_workspace_bytes=int(getattr(wr, "collective_workspace_bytes", 0)),
                framework_overhead_bytes=int(getattr(wr, "framework_overhead_bytes", 0)),
                master_weights_bytes=int(getattr(wr, "master_weights_bytes", 0)),
                total_bytes=int(getattr(wr, "total_bytes", 0)),
            ),
            duplication_bytes=int(getattr(d, "duplication_bytes", 0)),
            master_weights_overhead_bytes=int(getattr(d, "master_weights_overhead_bytes", 0)),
            kernel_boundary_materialisation_bytes=int(
                getattr(d, "kernel_boundary_materialisation_bytes", 0)
            ),
            fits_on_topology=bool(getattr(d, "fits_on_topology", True)),
        )
        gotcha_payloads = [
            GotchaPayload(
                id=g.gotcha_id,
                severity=g.severity.value if hasattr(g.severity, "value")
                                          else str(g.severity),
                message=g.message,
                reference=getattr(g, "reference", None),
            )
            for g in dverify.gotchas
        ]
    gotcha_payloads.extend(
        _side_channel_policy_gotchas(params.side_channels, available)
    )
    # V7-F56b: surface the symbolic-dim mismatch as a gotcha so the
    # vbgui GotchasTab + per-brick badge render it without needing
    # a separate WS channel.
    de = params.dim_env if isinstance(params.dim_env, dict) else (
        params.dim_env.model_dump()
        if hasattr(params.dim_env, "model_dump") else {}
    )
    f56b_H = de.get("H")
    f56b_nh = de.get("nh")
    f56b_hd = de.get("head_dim")
    if (f56b_H is not None and f56b_nh is not None and f56b_hd is not None
            and f56b_nh * f56b_hd != f56b_H):
        gotcha_payloads.append(GotchaPayload(
            id="v7_f56b_dim_env_mismatch",
            severity="warning",
            message=(
                f"dim_env.H={f56b_H} but nh*head_dim = "
                f"{f56b_nh}*{f56b_hd} = {f56b_nh * f56b_hd}. "
                "Attention still runs via internal Q projection, but "
                "this almost always means the architect mis-pinned a "
                "dim_env value."
            ),
            reference=None,
        ))

    edge_payloads: list[EdgeResolution] = []
    for re in resolved.edges:
        severity = "info" if re.matched else "error"
        edge_payloads.append(EdgeResolution(
            src=re.producer, dst=re.consumer,
            shape=_shape_to_list(re.producer_shape),
            matched=bool(re.matched), severity=severity,
        ))

    diags = [
        Diagnostic(
            severity=getattr(d.severity, "value", str(d.severity)),
            component=getattr(d, "component", "graph"),
            message=getattr(d, "message", ""),
            suggested_fix=getattr(d, "suggested_fix", None),
        )
        for d in (resolved.errors + resolved.warnings)
    ]

    fusion_payloads = [
        FusionRegionPayload(
            brick_names=list(fp.brick_names),
            backend=fp.backend,
            is_fused=bool(getattr(fp, "is_fused", True)),
            estimated_savings_us=float(getattr(fp, "estimated_savings_us", 0.0)),
        )
        for fp in fusion_plan
    ]

    # E7-2: build inference log so the UI Dimensions tab can show why
    # each per-brick parameter has the value it does (user-set vs
    # auto-derived from dim_env).
    from cppmega_v4.spec.inference_log import build_inference_log
    from cppmega_v4.jsonrpc.schema import InferenceEntryPayload
    inference_log_raw = build_inference_log(
        {"nodes": [{"id": n.id, "kind": n.kind, "params": dict(n.params)}
                   for n in params.graph.nodes]},
        dict(params.dim_env),
    )
    inference_log_payload = [
        InferenceEntryPayload(
            brick=e.brick, param=e.param, value=e.value,
            source=e.source, reason=e.reason,
        )
        for e in inference_log_raw
    ]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    out = VerifyResult(
        resolved=ResolvedGraph(
            edges=edge_payloads,
            diagnostics=diags,
            has_errors=bool(resolved.errors),
        ),
        memory_per_brick=per_brick,
        memory_distributed=distributed_payload,
        gotchas=gotcha_payloads,
        fusion_plan=fusion_payloads,
        inference_log=inference_log_payload,
        elapsed_ms=elapsed_ms,
    )
    _cache_store(cache, key, out)
    return out


# ---------------------------------------------------------------------------
# suggest_sharding
# ---------------------------------------------------------------------------


def suggest_sharding(
    params: SuggestShardingParams, *, cache: LRUCache | None = None,
) -> SuggestShardingResult:
    """Wrap cppmega_v4.parallelism.suggest_sharding for the GUI."""
    key, hit = _cache_lookup(cache, "suggest_sharding", params)
    if hit is not None:
        return hit

    t0 = time.perf_counter()
    specs = _graph_to_specs(params.graph)
    hidden = params.dim_env.get("H", 64)
    graph = from_block_specs(specs, hidden_size=hidden, instantiate=False)
    loss = _make_loss(params.loss)
    optim = _make_optim(params.optim)
    build_spec = ModelBuildSpec(graph=graph, loss=loss, optim=optim)
    topology = _make_topology(params.topology)

    proposals = _suggest_sharding(build_spec, topology)

    out_proposals: list[ShardingProposalPayload] = []
    for p in proposals:
        out_proposals.append(ShardingProposalPayload(
            strategy_name=p.strategy_name,
            fits=bool(p.fits),
            estimated_per_rank_bytes=int(p.estimated_per_rank_bytes),
            reason=p.reason,
            num_errors=int(p.num_errors),
            sharding=ShardingSpecPayload(
                topology=params.topology,
                axis_assignments=[
                    AxisAssignmentPayload(
                        axis_name=a.axis_name,
                        kind=a.kind.value if hasattr(a.kind, "value") else str(a.kind),
                        degree=int(a.degree),
                    )
                    for a in p.sharding.axis_assignments
                ],
                compile_mode=p.sharding.compile_mode,
                fp8_enabled=p.sharding.fp8_enabled,
            ),
        ))

    elapsed = (time.perf_counter() - t0) * 1000.0
    out = SuggestShardingResult(proposals=out_proposals, elapsed_ms=elapsed)
    _cache_store(cache, key, out)
    return out


# ---------------------------------------------------------------------------
# suggest_adapters
# ---------------------------------------------------------------------------


def suggest_adapters(
    params: SuggestAdaptersParams, *, cache: LRUCache | None = None,
) -> SuggestAdaptersResult:
    """Wrap cppmega_v4.spec.suggest_adapters for one mismatched edge."""
    key, hit = _cache_lookup(cache, "suggest_adapters", params)
    if hit is not None:
        return hit

    specs = _graph_to_specs(params.graph)
    hidden = params.dim_env.get("H", 64)
    graph = from_block_specs(specs, hidden_size=hidden, instantiate=False)
    resolved = resolve_shapes(graph, params.dim_env)
    proposal = _suggest_adapters(
        resolved, params.producer, params.consumer, max_steps=params.max_steps,
    )

    chain = [
        {"rule": getattr(s, "rule", ""), "description": getattr(s, "description", ""),
         "params": dict(getattr(s, "params", {}))}
        for s in (proposal.chain or [])
    ]
    out = SuggestAdaptersResult(
        producer=proposal.producer,
        consumer=proposal.consumer,
        producer_shape=_shape_to_list(proposal.producer_shape),
        consumer_shape=_shape_to_list(proposal.consumer_shape),
        chain=chain,
        reason=proposal.reason,
    )
    _cache_store(cache, key, out)
    return out


# ---------------------------------------------------------------------------
# build_preset_specs
# ---------------------------------------------------------------------------


def build_preset_specs(
    params: BuildPresetSpecsParams, *, cache: LRUCache | None = None,
) -> BuildPresetSpecsResult:
    """Expand a preset name into wire-form specs."""
    key, hit = _cache_lookup(cache, "build_preset_specs", params)
    if hit is not None:
        return hit

    if params.preset_name not in PRESETS:
        raise ValueError(
            f"unknown preset {params.preset_name!r}; "
            f"available: {sorted(PRESETS)}"
        )
    specs = _build_preset_specs(
        params.preset_name,
        hidden_size=params.hidden_size,
        num_layers=params.num_layers,
    )
    out = BuildPresetSpecsResult(
        specs=[dict(s) for s in specs],
        preset_name=params.preset_name,
    )
    _cache_store(cache, key, out)
    return out


# ---------------------------------------------------------------------------
# probe.run
# ---------------------------------------------------------------------------


def probe_run(params: ProbeRunParams, *, cache: LRUCache | None = None) -> ProbeRunResult:
    """Bridge to cppmega_v4.probe.contract_probe."""
    key, hit = _cache_lookup(cache, "probe.run", params)
    if hit is not None:
        return hit

    specs = _graph_to_specs(params.graph)
    hidden = params.dim_env.get("H", params.probe_hidden_size)
    graph = from_block_specs(specs, hidden_size=hidden, instantiate=False)
    loss = _make_loss(params.loss)
    optim = _make_optim(params.optim)
    build_spec = ModelBuildSpec(graph=graph, loss=loss, optim=optim)

    report = _contract_probe(
        build_spec,
        params.tokenizer_source,
        params.parquet_path,
        probe_hidden_size=params.probe_hidden_size,
        run_dry_forward=params.run_dry_forward,
    )
    report_dict = _probe_to_dict(report)
    out = ProbeRunResult(**report_dict)
    _cache_store(cache, key, out)
    return out
