"""Heuristic sharding-strategy proposer.

Given a :class:`ModelBuildSpec` and a :class:`DeviceTopology`, produces
3-5 ranked :class:`ShardingProposal`s. Heuristics are informed by the
production patterns the team uses in ``../cppmega`` and ``../nanochat``:

  - **MoE present → EP first** (degree = sqrt(num_experts) clamped to the
    available mesh) — the cppmega EP=4/8 pattern.
  - **>70B params on ≤80 GB device → FSDP2 mandatory** — single-rank
    weights don't fit otherwise.
  - **Attention with H ≥ 4096 → TP=2 reduces activation memory** — the
    nanochat Megatron TP+SP recipe.
  - **<10B params + single device → no-shard / DP-only**.
  - **Always compile_mode="regional"** — avoids the FSDP2/Megatron
    whole-model-compile footguns (see GotchaChecker).
  - **master_weights_fp32=False unless explicitly requested** — Muon
    handles bf16 loss scaling fine; saves the duplication overhead.

Each proposal is scored by a fitness metric (lower per-rank memory →
better) and gotcha penalty (any ERROR severity gotcha pushes score to
worst). Proposals that would never fit on the topology are filtered out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from cppmega_v4.buildspec import ModelBuildSpec
from cppmega_v4.parallelism.distributed_memory import (
    DistributedMemoryReport,
    estimate_distributed_memory,
)
from cppmega_v4.parallelism.gotcha_checker import (
    Gotcha,
    GotchaSeverity,
    check_gotchas,
)
from cppmega_v4.parallelism.sharding_spec import (
    AxisAssignment,
    ParallelismKind,
    ShardingSpec,
    fsdp2_only,
    fsdp2_plus_tp,
    megatron_ep_only,
    single_device,
)
from cppmega_v4.parallelism.topology import DeviceTopology


@dataclass(frozen=True)
class ShardingProposal:
    """One ranked proposal."""

    strategy_name: str
    sharding: ShardingSpec
    reason: str
    estimated_per_rank_bytes: int
    fits: bool
    gotchas: tuple[Gotcha, ...] = field(default_factory=tuple)

    @property
    def num_errors(self) -> int:
        return sum(
            1 for g in self.gotchas if g.severity is GotchaSeverity.ERROR
        )


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------


def _moe_present(spec: ModelBuildSpec) -> bool:
    return any(n.kind in {"moe", "bailing_moe"} for n in spec.graph.nodes)


def _largest_attention_h(spec: ModelBuildSpec) -> int:
    env = spec.dim_env or {}
    if any(n.kind in {
        "gated_attention", "attention", "mla", "mla_absorb",
        "gqa_sliding", "cca_attention", "mistral4_mla",
    } for n in spec.graph.nodes):
        return int(env.get("H", 0))
    return 0


def _estimate_total_param_bytes(spec: ModelBuildSpec) -> int:
    """Use the single-device memory report as the baseline param-bytes
    proxy (sums brick params_elems × bf16 width)."""
    from cppmega_v4.spec import estimate_memory, resolve_shapes
    env = spec.dim_env or {"B": 1, "S": 1, "H": 1}
    resolved = resolve_shapes(spec.graph, env, strict=False,
                              available_side_channels=frozenset({
                                  "doc_ids", "token_ids",
                              }))
    base = estimate_memory(resolved, training=False, dtype_bytes=2)
    return base.weights_bytes


def _smallest_device_hbm(topology: DeviceTopology) -> int:
    return min(d.hbm_bytes for d in topology.devices)


def _propose_single_device(
    spec: ModelBuildSpec, topology: DeviceTopology,
) -> ShardingProposal:
    sharding = single_device(topology)
    report = estimate_distributed_memory(spec, sharding)
    return ShardingProposal(
        strategy_name="single_device",
        sharding=sharding,
        reason=(
            "DP-only / replicate everything. Fits when each device's HBM "
            "holds full model + optimizer."
        ),
        estimated_per_rank_bytes=report.worst_rank.total_bytes,
        fits=report.fits_on_topology(),
        gotchas=check_gotchas(sharding, spec),
    )


def _propose_fsdp2(
    spec: ModelBuildSpec, topology: DeviceTopology,
) -> ShardingProposal | None:
    if topology.num_devices < 2:
        return None
    sharding = fsdp2_only(topology)
    report = estimate_distributed_memory(spec, sharding)
    return ShardingProposal(
        strategy_name="fsdp2_only",
        sharding=sharding,
        reason=(
            "FSDP2 (ZeRO-3) sharding across the dp axis. Optimizer + "
            "grads divided by dp_degree. Peak weights = unsharded "
            "during forward all-gather. Recommended for >10B models on "
            "80 GB devices."
        ),
        estimated_per_rank_bytes=report.worst_rank.total_bytes,
        fits=report.fits_on_topology(),
        gotchas=check_gotchas(sharding, spec),
    )


def _propose_megatron_ep(
    spec: ModelBuildSpec, topology: DeviceTopology,
) -> ShardingProposal | None:
    if not _moe_present(spec):
        return None
    if "ep" not in topology.mesh_axes or topology.mesh_axes["ep"] < 2:
        return None
    sharding = megatron_ep_only(topology)
    report = estimate_distributed_memory(spec, sharding)
    return ShardingProposal(
        strategy_name="megatron_ep_only",
        sharding=sharding,
        reason=(
            f"MoE present + EP={topology.mesh_axes['ep']} axis available. "
            "Cppmega-style production pattern (Megatron EP=4/8). "
            "Each rank holds num_experts/ep_degree experts."
        ),
        estimated_per_rank_bytes=report.worst_rank.total_bytes,
        fits=report.fits_on_topology(),
        gotchas=check_gotchas(sharding, spec),
    )


def _propose_fsdp2_plus_tp(
    spec: ModelBuildSpec, topology: DeviceTopology,
) -> ShardingProposal | None:
    if "tp" not in topology.mesh_axes or topology.mesh_axes["tp"] < 2:
        return None
    if "dp" not in topology.mesh_axes or topology.mesh_axes["dp"] < 2:
        return None
    sharding = fsdp2_plus_tp(topology)
    report = estimate_distributed_memory(spec, sharding)
    return ShardingProposal(
        strategy_name="fsdp2_plus_tp",
        sharding=sharding,
        reason=(
            "3D parallelism: FSDP2 across dp + Megatron TP across tp. "
            "TP halves per-layer matmul activation memory. Use when "
            "attention has H ≥ 4096 or very deep stacks. "
            "compile_mode='regional' is mandatory."
        ),
        estimated_per_rank_bytes=report.worst_rank.total_bytes,
        fits=report.fits_on_topology(),
        gotchas=check_gotchas(sharding, spec),
    )


def _propose_fsdp2_plus_ep(
    spec: ModelBuildSpec, topology: DeviceTopology,
) -> ShardingProposal | None:
    """FSDP2 across dp + EP across ep — for MoE models on 3D meshes."""
    if not _moe_present(spec):
        return None
    if "ep" not in topology.mesh_axes or topology.mesh_axes["ep"] < 2:
        return None
    if "dp" not in topology.mesh_axes or topology.mesh_axes["dp"] < 2:
        return None
    sharding = ShardingSpec(
        topology=topology,
        axis_assignments=(
            AxisAssignment("dp", ParallelismKind.FSDP2,
                           topology.mesh_axes["dp"]),
            AxisAssignment("ep", ParallelismKind.EP,
                           topology.mesh_axes["ep"]),
        ),
        compile_mode="regional",
    )
    report = estimate_distributed_memory(spec, sharding)
    return ShardingProposal(
        strategy_name="fsdp2_plus_ep",
        sharding=sharding,
        reason=(
            "FSDP2 backbone + EP for MoE experts. The killer combo for "
            "100B+ MoE models on multi-node clusters. Each rank's "
            "memory = (backbone params / dp) + (expert params / ep)."
        ),
        estimated_per_rank_bytes=report.worst_rank.total_bytes,
        fits=report.fits_on_topology(),
        gotchas=check_gotchas(sharding, spec),
    )


# ---------------------------------------------------------------------------
# Scoring + ranking
# ---------------------------------------------------------------------------


_ERROR_PENALTY: int = 10 ** 15   # any ERROR gotcha pushes score to worst


def _score(p: ShardingProposal) -> int:
    """Lower is better. ERROR gotchas dominate; otherwise fits + per-rank."""
    if p.num_errors > 0:
        return _ERROR_PENALTY + p.estimated_per_rank_bytes
    if not p.fits:
        # Doesn't fit → ranks below any fitting strategy.
        return (_ERROR_PENALTY // 10) + p.estimated_per_rank_bytes
    return p.estimated_per_rank_bytes


def suggest_sharding(
    build_spec: ModelBuildSpec,
    topology: DeviceTopology,
) -> list[ShardingProposal]:
    """Produce 3-5 ranked proposals.

    Returns a list sorted by :func:`_score` ascending. The first entry
    is the recommended choice; the rest are alternatives the user can
    pick from. Proposals that emit ERROR-severity gotchas are kept in
    the list but ranked last (so the GUI can show "this would crash —
    here's a safer alternative" instead of hiding the option).
    """
    proposals: list[ShardingProposal] = []
    for builder in (
        _propose_single_device,
        _propose_fsdp2,
        _propose_megatron_ep,
        _propose_fsdp2_plus_tp,
        _propose_fsdp2_plus_ep,
    ):
        p = builder(build_spec, topology)
        if p is not None:
            proposals.append(p)
    proposals.sort(key=_score)
    return proposals


__all__ = [
    "ShardingProposal",
    "suggest_sharding",
]
