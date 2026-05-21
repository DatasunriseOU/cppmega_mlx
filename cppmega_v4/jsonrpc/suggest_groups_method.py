"""JSON-RPC handler for suggest_optim_groups (E7-4)."""

from __future__ import annotations

from cppmega_v4.buildspec.group_inference import suggest_groups
from cppmega_v4.buildspec.optim_spec import OptimKind
from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.schema import (
    ProposedGroupPayload,
    SuggestOptimGroupsParams,
    SuggestOptimGroupsResult,
)


def suggest_optim_groups(
    params: SuggestOptimGroupsParams,
    *,
    cache: LRUCache | None = None,
) -> SuggestOptimGroupsResult:
    """Materialise the graph, classify every parameter, return proposals."""
    from cppmega_v4.fusion import from_block_specs

    try:
        kind = OptimKind(params.optim_kind)
    except ValueError as exc:
        raise ValueError(f"unknown optim_kind {params.optim_kind!r}") from exc

    # Materialise specs with instantiate=True so .module exposes
    # parameters().
    block_specs = []
    for n in params.graph.nodes:
        block_specs.append({
            "kind": n.kind,
            "name": n.id,
            "params": dict(n.params),
        })

    graph = from_block_specs(block_specs, hidden_size=params.hidden_size,
                             instantiate=True)

    # Flatten everyone's parameters into one dict keyed by brick.name.
    flat: dict[str, object] = {}
    for node in graph.nodes:
        if node.module is None:
            continue
        try:
            flat[node.name] = node.module.parameters()
        except Exception:
            continue

    res = suggest_groups(flat, kind)

    return SuggestOptimGroupsResult(
        proposals=[
            ProposedGroupPayload(
                matcher=p.matcher,
                optim_kind=p.optim_kind.value,
                lr=p.lr,
                weight_decay=p.weight_decay,
                betas=p.betas,
                ns_steps=p.ns_steps,
                param_count=p.param_count,
                rationale=p.rationale,
            )
            for p in res.proposals
        ],
        total_params=res.total_params,
        uncovered_params=res.uncovered_params,
    )
