"""V8-R04: ``architectures.auto_fit`` — given a preset, pick the best
``(hidden_size, num_layers)`` AND a sharding proposal for the detected
(or supplied) devbox in one shot.

Strategy:
  1. Resolve the target topology — either ``host_info.topology`` from
     the caller, or pick the first ``available_topologies`` entry from
     ``platform.get_info``.
  2. Compute ``target_bytes = total_hbm × headroom`` and run
     :func:`architectures.scale_down.scale_down` to land a model that
     fits the device.
  3. Call :func:`suggest_sharding` against the scaled spec with a
     minimal default ``loss=cross_entropy`` + ``optim=adamw`` so the
     planner has enough to produce ranked proposals.
  4. Return the bundle: scaled / sharding / fits / reason.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from cppmega_v4.architectures.scale_down import scale_down as _scale_down
from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.methods import (
    _cache_lookup, _cache_store, suggest_sharding,
)
from cppmega_v4.jsonrpc.schema import (
    ScaleDownFromCanonical, ScaleDownResultModel,
    SuggestShardingParams, SuggestShardingResult,
)
from cppmega_v4.parallelism import topology as _topo


__all__ = [
    "AutoFitParams",
    "AutoFitHostInfo",
    "AutoFitResult",
    "auto_fit",
    "TOPOLOGY_BUILDERS",
]


# Same table as memory_matrix_method but kept local to avoid a cycle.
TOPOLOGY_BUILDERS: dict[str, Any] = {
    "h100_8x":       lambda: _topo.h100_8x(),
    "h200_8x":       lambda: _topo.h200_8x(),
    "a100_8x":       lambda: _topo.a100_8x(),
    "b100_8x":       lambda: _topo.b100_8x(),
    "gb10_quarter":  lambda: _topo.gb10_quarter(),
    "tpu_v6e_8":     lambda: _topo.tpu_v6e_8(),
    "tpu_v5p_4":     lambda: _topo.tpu_v5p_4(),
    "m3_ultra_solo": lambda: _topo.m3_ultra_solo(),
}


class AutoFitHostInfo(BaseModel):
    """When supplied, overrides the server-side platform probe."""

    model_config = ConfigDict(extra="forbid")

    topology: str
    headroom: float = 0.9


class AutoFitParams(BaseModel):
    """Input — preset name + optional host overrides."""

    model_config = ConfigDict(extra="forbid")

    preset: str
    host_info: AutoFitHostInfo | None = None


class AutoFitResult(BaseModel):
    """Output — scale-down result + sharding proposals + fits verdict."""

    model_config = ConfigDict(extra="forbid")

    scaled: ScaleDownResultModel
    sharding: SuggestShardingResult
    fits: bool
    reason: str
    topology: str
    headroom: float


def _resolve_topology(params: AutoFitParams) -> tuple[str, float]:
    """Pick (topology_name, headroom) from host_info or platform probe."""
    if params.host_info is not None:
        if params.host_info.topology not in TOPOLOGY_BUILDERS:
            raise ValueError(
                f"unknown topology {params.host_info.topology!r}; "
                f"choose from {sorted(TOPOLOGY_BUILDERS)}")
        return params.host_info.topology, params.host_info.headroom
    from cppmega_v4.runtime.platform_probe import probe_platform
    info = probe_platform()
    available = info.get("available_topologies") or []
    for t in available:
        if t in TOPOLOGY_BUILDERS:
            return t, 0.9
    return "m3_ultra_solo", 0.9   # final fallback


def auto_fit(
    params: AutoFitParams, *, cache: LRUCache | None = None,
) -> AutoFitResult:
    """Chain scale_down + suggest_sharding for the detected/given devbox."""
    key, hit = _cache_lookup(cache, "architectures.auto_fit", params)
    if hit is not None:
        return hit

    topo_name, headroom = _resolve_topology(params)
    topo = TOPOLOGY_BUILDERS[topo_name]()
    total_hbm = topo.total_hbm_bytes
    target_bytes = int(total_hbm * headroom)
    scaled = _scale_down(params.preset, target_bytes)

    # Build minimal sharding query payload — use the scaled specs +
    # cross_entropy + adamw defaults. We feed the scaled hidden_size
    # so the suggester sees the right model size, and head_outputs to
    # the last brick so verify_build_spec accepts the request.
    graph_nodes = []
    graph_edges = []
    prev_name: str | None = None
    for spec in scaled.specs:
        if "parallel" in spec:
            # Each branch becomes a leaf for the suggester.
            for leaf in spec["parallel"]:
                name = leaf.get("name") or leaf.get("kind")
                graph_nodes.append({
                    "id": name, "kind": leaf["kind"],
                    "params": leaf.get("params", {}),
                })
                if prev_name is not None:
                    graph_edges.append({"src": prev_name, "dst": name})
                prev_name = name
            continue
        name = spec.get("name") or spec.get("kind")
        graph_nodes.append({
            "id": name, "kind": spec["kind"],
            "params": spec.get("params", {}),
        })
        if prev_name is not None:
            graph_edges.append({"src": prev_name, "dst": name})
        prev_name = name

    last = graph_nodes[-1]["id"] if graph_nodes else "out"
    sharding_params = SuggestShardingParams.model_validate({
        "graph": {"nodes": graph_nodes, "edges": graph_edges},
        "dim_env": {"H": scaled.hidden_size, "B": 1, "S": 256},
        "loss": {"kind": "cross_entropy", "head_outputs": [last]},
        "optim": {"kind": "adamw", "groups": [
            {"matcher": "all", "lr": 3e-4, "weight_decay": 0.01,
             "betas": [0.9, 0.95]},
        ], "gradient_clip_norm": 1.0, "mixed_precision": True},
        "topology": {"factory": topo_name, "kwargs": {}},
    })
    sharding = suggest_sharding(sharding_params, cache=cache)

    best_proposal = (
        sharding.proposals[0] if sharding.proposals else None)
    fits = bool(scaled.fits and (
        best_proposal is None or best_proposal.fits))
    axis_desc = (
        ",".join(f"{a.axis_name}×{a.degree}"
                 for a in best_proposal.sharding.axis_assignments)
        if best_proposal else "<no proposal>")
    reason = (
        f"hidden={scaled.hidden_size}, layers={scaled.num_layers}, "
        f"axis={axis_desc}, peak={scaled.estimated_bytes / 1e9:.2f} GB / "
        f"{total_hbm / 1e9:.0f} GB")
    out = AutoFitResult(
        scaled=ScaleDownResultModel(
            hidden_size=scaled.hidden_size,
            num_layers=scaled.num_layers,
            estimated_bytes=scaled.estimated_bytes,
            target_bytes=scaled.target_bytes,
            fits=scaled.fits,
            scaled_down_from=ScaleDownFromCanonical(
                hidden_size=scaled.scaled_down_from[0],
                num_layers=scaled.scaled_down_from[1],
            ),
            specs=scaled.specs,
        ),
        sharding=sharding,
        fits=fits,
        reason=reason,
        topology=topo_name,
        headroom=headroom,
    )
    _cache_store(cache, key, out)
    return out
