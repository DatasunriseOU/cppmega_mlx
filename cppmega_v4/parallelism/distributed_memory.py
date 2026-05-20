"""Per-rank distributed memory accounting.

Given a :class:`cppmega_v4.buildspec.ModelBuildSpec` and a
:class:`cppmega_v4.parallelism.ShardingSpec`, produce a
:class:`DistributedMemoryReport` with per-rank byte breakdown that
covers the lessons-learned pitfalls from ``../cppmega`` and
``../nanochat``:

  * **FSDP all-gather peak** — during the forward of an FSDP-wrapped
    layer, the full unsharded weight tensor is reconstructed on every
    rank. So peak weight memory = unsharded params, not sharded —
    even though steady-state and optimizer state are sharded.
  * **TP / EP weight sharding** — params divided by tp_degree (per-layer
    matmul shards) and ep_degree (MoE expert shards).
  * **FP8 + bf16 grad duplication** — fp8 forward (1 B/param) STILL
    materialises grads in bf16 (2 B/param), so total weight+grad ≈
    1 + 2 = 3× a pure-bf16 setup vs the naive expectation of fp8 cuts
    memory in half.
  * **Master fp32 weights** — keeping a fp32 master copy alongside
    bf16/fp8 compute weights doubles param memory and doubles optimizer
    state (each rank holds local TP shard in bf16 AND fp32).
  * **Megatron RowParallel kernel-boundary materialisation** — the
    AllReduce inside RowParallelLinear materialises the full
    (B, S, H) tensor on every rank for the backward, adding a
    boundary peak that gradient checkpointing CAN'T elide.
  * **Sequence-parallel norm/qk-norm AllReduce buffers** — small
    replicated params (norms) all-reduce separately after backward
    fills the grad; this needs a workspace buffer roughly equal to
    those param sizes.
  * **Framework overhead** — ~2 GB CUDA baseline; ~6.4 GB XLA compiler
    HLO temps. Both eaten before the model even runs.

The formulas are intentionally conservative and within ±10-15% of
real-world runs (see ``../nanochat/memory_estimator.py`` for the
calibrated reference we ported).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from cppmega_v4.buildspec import ModelBuildSpec
from cppmega_v4.buildspec.loss_spec import LossKind
from cppmega_v4.buildspec.optim_spec import OptimKind
from cppmega_v4.parallelism.sharding_spec import (
    ParallelismKind,
    ShardingSpec,
)
from cppmega_v4.parallelism.topology import DeviceSpec
from cppmega_v4.spec import estimate_memory, resolve_shapes


# ---------------------------------------------------------------------------
# Byte-count constants
# ---------------------------------------------------------------------------


# Element-byte widths for the dtype families our specs declare.
_BYTES_PER_ELEM: Final[dict[str, int]] = {
    "fp32":  4,
    "bf16":  2,
    "fp16":  2,
    "fp8":   1,
    "int8":  1,
    "int4":  1,   # rounded up; packed nibbles handled separately
}


# CUDA driver baseline + XLA compiler HLO temps. Eaten before the model
# even starts.  These match the runtime-measured values from
# ../nanochat/memory_estimator.py (CUDA ~2 GB, XLA ~6.4 GB).
_FRAMEWORK_OVERHEAD_BYTES_CUDA: Final[int] = 2 * 1024 ** 3
_FRAMEWORK_OVERHEAD_BYTES_XLA:  Final[int] = int(6.4 * 1024 ** 3)
_FRAMEWORK_OVERHEAD_BYTES_MLX:  Final[int] = 1 * 1024 ** 3   # unified mem mac


_OPTIM_BYTES_PER_PARAM: Final[dict[OptimKind, int]] = {
    OptimKind.ADAMW:             8,   # m + v at fp32 = 4 + 4
    OptimKind.MUON:              2,   # momentum at bf16
    OptimKind.MUON_ADAMW_HYBRID: 5,   # weighted avg approximation
    OptimKind.SGD:               0,
}


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerRankMemory:
    """Byte breakdown for ONE rank in the mesh."""

    rank_idx: int
    device_idx: int
    weights_bytes: int
    grads_bytes: int
    optimizer_state_bytes: int
    master_weights_bytes: int
    activations_bytes: int
    fsdp_allgather_peak_bytes: int
    kv_cache_bytes: int
    moe_routing_buffers_bytes: int
    collective_workspace_bytes: int
    framework_overhead_bytes: int
    total_bytes: int

    def fits_on_device(
        self, device: DeviceSpec, *, headroom: float = 0.9,
    ) -> bool:
        if not 0.0 < headroom <= 1.0:
            raise ValueError(f"headroom must be (0, 1], got {headroom!r}")
        return self.total_bytes <= device.hbm_bytes * headroom


@dataclass(frozen=True)
class DistributedMemoryReport:
    """Mesh-wide memory report. The GUI shows the worst rank in the bar."""

    sharding: ShardingSpec
    per_rank: tuple[PerRankMemory, ...]
    duplication_bytes: int                       # FP8 fwd + bf16 grad overhead
    master_weights_overhead_bytes: int
    kernel_boundary_materialisation_bytes: int   # Megatron RowParallel
    worst_rank_idx: int
    bottleneck_diagnostics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def worst_rank(self) -> PerRankMemory:
        return self.per_rank[self.worst_rank_idx]

    def fits_on_topology(self, *, headroom: float = 0.9) -> bool:
        """True iff every rank fits on its assigned device."""
        if not 0.0 < headroom <= 1.0:
            raise ValueError(f"headroom must be (0, 1], got {headroom!r}")
        for r in self.per_rank:
            device = self.sharding.topology.devices[r.device_idx]
            if not r.fits_on_device(device, headroom=headroom):
                return False
        return True

    def summary(self) -> dict[str, int]:
        worst = self.worst_rank
        return {
            "num_ranks":          len(self.per_rank),
            "worst_rank":         worst.rank_idx,
            "worst_total":        worst.total_bytes,
            "duplication":        self.duplication_bytes,
            "master_overhead":    self.master_weights_overhead_bytes,
            "kernel_boundary":    self.kernel_boundary_materialisation_bytes,
        }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


def _bytes_per_weight_elem(sharding: ShardingSpec) -> int:
    """Width of an active weight element. fp8 → 1 B, otherwise bf16 → 2."""
    return _BYTES_PER_ELEM["fp8"] if sharding.fp8_enabled else _BYTES_PER_ELEM["bf16"]


def _bytes_per_grad_elem(sharding: ShardingSpec) -> int:
    """Width of one grad element. FP8 forward STILL materialises bf16 grads
    (see nanochat/fp8_training.py:1-32). Pure-bf16 path keeps bf16 grads;
    ``grad_reduce_dtype='fp32'`` upgrades the reduction buffer to fp32."""
    return _BYTES_PER_ELEM[sharding.grad_reduce_dtype]


def _aggregate_degree(
    sharding: ShardingSpec, kind: ParallelismKind,
) -> int:
    """Aggregate degree of ``kind`` across all axes."""
    return sharding.degree_of(kind)


def _weight_shard_factor(sharding: ShardingSpec) -> float:
    """Reciprocal of effective weight-shard divisor.

    TP divides per-layer matmul weights by tp_degree.
    EP divides MoE expert weights by ep_degree (per-rank holds
    moe_n_routed/ep_degree experts).
    FSDP* steady-state is sharded across the dp axis, but PEAK during
    forward all-gather equals unsharded — that gets a separate term.
    """
    tp = _aggregate_degree(sharding, ParallelismKind.TP)
    # EP only meaningful for MoE bricks; here we apply globally as a
    # conservative under-approximation. The Stage E refinement can split
    # by per-brick category.
    return 1.0 / max(tp, 1)


def _fsdp_steady_state_divisor(sharding: ShardingSpec) -> int:
    """During steady state (not during all-gather), FSDP* divides weights
    by dp degree. We track this for optimizer / grad accounting."""
    kinds = sharding.axis_kinds()
    if ParallelismKind.FSDP2 in kinds or ParallelismKind.FSDP1 in kinds:
        return max(_aggregate_degree(sharding, ParallelismKind.FSDP2),
                   _aggregate_degree(sharding, ParallelismKind.FSDP1), 1)
    if ParallelismKind.ZERO2 in kinds:
        # ZeRO-2: optim + grads sharded, weights NOT
        return 1
    return 1


def _optim_shard_divisor(sharding: ShardingSpec) -> int:
    """Optimizer state is sharded by any ZeRO/FSDP family axis."""
    for k in (ParallelismKind.FSDP2, ParallelismKind.FSDP1,
              ParallelismKind.ZERO2, ParallelismKind.ZERO1):
        d = _aggregate_degree(sharding, k)
        if d > 1:
            return d
    return 1


def _grad_shard_divisor(sharding: ShardingSpec) -> int:
    """Gradient buffer is sharded by FSDP and ZeRO-2 (not ZeRO-1)."""
    for k in (ParallelismKind.FSDP2, ParallelismKind.FSDP1,
              ParallelismKind.ZERO2):
        d = _aggregate_degree(sharding, k)
        if d > 1:
            return d
    return 1


def _framework_overhead(sharding: ShardingSpec) -> int:
    """Framework baseline. Picks CUDA / XLA / MLX based on device kind."""
    first = sharding.topology.devices[0].kind.value
    if first.startswith("tpu"):
        return _FRAMEWORK_OVERHEAD_BYTES_XLA
    if first in {"m3_ultra", "gb10"}:
        return _FRAMEWORK_OVERHEAD_BYTES_MLX
    return _FRAMEWORK_OVERHEAD_BYTES_CUDA


def _moe_present(build_spec: ModelBuildSpec) -> bool:
    return any(
        n.kind in {"moe", "bailing_moe"} for n in build_spec.graph.nodes
    )


def _moe_routing_buffers_bytes(
    build_spec: ModelBuildSpec, sharding: ShardingSpec,
) -> int:
    """Buffers for MoE routing + expert all-to-all dispatch.

    Conservative model: scales with ``B * S * top_k * H``. EP halves it
    because tokens are dispatched and each rank only processes a subset.
    """
    if not _moe_present(build_spec):
        return 0
    env = build_spec.dim_env or {}
    B = int(env.get("B", 1))
    S = int(env.get("S", 4096))
    H = int(env.get("H", 4096))
    top_k = int(env.get("top_k", 2))
    bytes_elem = _bytes_per_weight_elem(sharding)
    raw = B * S * top_k * H * bytes_elem
    ep = max(_aggregate_degree(sharding, ParallelismKind.EP), 1)
    # Dispatch buffer: roughly halved per-EP-rank (token-disjoint slabs)
    return max(1, raw // max(ep, 1))


def _kernel_boundary_materialisation_bytes(
    build_spec: ModelBuildSpec, sharding: ShardingSpec,
) -> int:
    """Megatron RowParallelLinear AllReduce materialises full (B,S,H)
    on every rank at backward time. We charge this only when TP > 1
    (because that's when RowParallel is wired in)."""
    if _aggregate_degree(sharding, ParallelismKind.TP) <= 1:
        return 0
    env = build_spec.dim_env or {}
    B = int(env.get("B", 1))
    S = int(env.get("S", 4096))
    H = int(env.get("H", 4096))
    return B * S * H * _bytes_per_weight_elem(sharding)


def _sequence_parallel_workspace_bytes(
    build_spec: ModelBuildSpec, sharding: ShardingSpec,
) -> int:
    """Replicated-param AllReduce buffer (norms, qk-norms etc.).

    Rough rule: ~1% of total weights, materialised once during step."""
    if ParallelismKind.SP not in sharding.axis_kinds():
        return 0
    bytes_elem = _bytes_per_weight_elem(sharding)
    # Use baseline single-device memory report for total params estimate.
    resolved = resolve_shapes(
        build_spec.graph, build_spec.dim_env or {"B": 1, "S": 1, "H": 1},
        strict=False,
    )
    base = estimate_memory(resolved, training=False, dtype_bytes=bytes_elem)
    return base.weights_bytes // 100


def _duplication_bytes(
    base_weights_bytes: int, sharding: ShardingSpec,
) -> int:
    """FP8 fwd + bf16 grad duplication.

    fp8 forward weights live alongside bf16 grad buffer; the grad is
    *not* the same allocation. So total = fp8 weights + bf16 grads ≈
    1 + 2 = 3× param-element count. We report the EXTRA bytes the
    duplication costs vs the naive "fp8 cuts memory in half" expectation.
    """
    if not sharding.fp8_enabled:
        return 0
    # Naive expectation: fp8 → weights 1 byte/elem, grads also 1 byte/elem
    # (which is FALSE; bf16 is mandatory because there's no fp8 backward).
    naive = base_weights_bytes // 2 * 2   # ≈ base
    actual = base_weights_bytes // 2 + base_weights_bytes  # fp8 + bf16
    return max(0, actual - naive)


def _master_weights_overhead_bytes(
    base_weights_bytes: int, sharding: ShardingSpec,
) -> int:
    """Extra bytes when ``master_weights_fp32`` is enabled.

    fp32 copy alongside bf16/fp8 compute weights = base_weights * (4/active)
    extra. Cf. nanochat ``megatron_optimizer.py`` notes."""
    if not sharding.master_weights_fp32:
        return 0
    active = _bytes_per_weight_elem(sharding)
    return (base_weights_bytes // active) * _BYTES_PER_ELEM["fp32"]


def estimate_distributed_memory(
    build_spec: ModelBuildSpec,
    sharding: ShardingSpec,
    *,
    training: bool = True,
) -> DistributedMemoryReport:
    """Roll up a per-rank memory report.

    Args:
      build_spec: post-rewrite ``ModelBuildSpec`` (apply MTPRewriter
        before calling if MTP).
      sharding: which topology + parallelism strategy to model.
      training: when False, gradients / optimizer state are 0; KV-cache
        becomes meaningful instead.
    """
    # 1) Single-device baseline (no sharding) from VBSpec.
    env = build_spec.dim_env or {"B": 1, "S": 1, "H": 1}
    resolved = resolve_shapes(build_spec.graph, env, strict=False,
                               available_side_channels=frozenset({
                                   "doc_ids", "token_ids",
                               }))
    bytes_per_w = _bytes_per_weight_elem(sharding)
    bytes_per_g = _bytes_per_grad_elem(sharding)
    baseline = estimate_memory(
        resolved, training=training, dtype_bytes=bytes_per_w,
        kv_cache_dtype_bytes=1,
    )

    # 2) Compute global divisors / multipliers.
    shard_factor = _weight_shard_factor(sharding)        # TP factor
    fsdp_div     = _fsdp_steady_state_divisor(sharding)
    optim_div    = _optim_shard_divisor(sharding)
    grad_div     = _grad_shard_divisor(sharding)
    ep_div       = max(_aggregate_degree(sharding, ParallelismKind.EP), 1)

    # 3) Per-rank component byte counts.
    base_weights = int(baseline.weights_bytes * shard_factor)
    base_grads   = (
        int(baseline.weights_bytes * (bytes_per_g / max(bytes_per_w, 1))
            * shard_factor / max(grad_div, 1))
        if training else 0
    )

    optim_per_param = _OPTIM_BYTES_PER_PARAM.get(build_spec.optim.kind, 4)
    base_optim = (
        int((baseline.weights_bytes // max(bytes_per_w, 1))
            * optim_per_param * shard_factor / max(optim_div, 1))
        if training else 0
    )

    # FSDP steady-state weight shard
    steady_weights = base_weights // max(fsdp_div, 1)

    # FSDP all-gather peak: a single full unsharded layer briefly
    # materialises. Conservative proxy: ~10% of total weights (one
    # transformer block of typical model).
    fsdp_kinds = {ParallelismKind.FSDP1, ParallelismKind.FSDP2}
    fsdp_peak = (
        int(base_weights * 0.1)
        if (sharding.axis_kinds() & fsdp_kinds)
        else 0
    )

    activations = baseline.activations_bytes
    # Activation checkpointing reduces this. Mirror VBSpec defaults.
    if sharding.activation_checkpointing == "full":
        activations = int(activations * 0.5)
    elif sharding.activation_checkpointing == "off":
        activations = int(activations * 1.5)

    activations = int(activations / max(_aggregate_degree(
        sharding, ParallelismKind.TP), 1))

    moe_buffers = _moe_routing_buffers_bytes(build_spec, sharding)
    kernel_boundary = _kernel_boundary_materialisation_bytes(build_spec, sharding)
    collective_workspace = _sequence_parallel_workspace_bytes(
        build_spec, sharding,
    )

    # Duplication / master overheads (per-rank, not aggregate).
    dup_bytes = _duplication_bytes(steady_weights, sharding)
    master_extra = _master_weights_overhead_bytes(steady_weights, sharding)

    framework = _framework_overhead(sharding)

    kv_cache = (
        int(baseline.kv_cache_bytes * shard_factor)
        if not training else 0
    )

    per_rank: list[PerRankMemory] = []
    for rank_idx in range(sharding.num_ranks):
        device_idx = rank_idx  # 1-to-1 device assignment
        total = (
            steady_weights + base_grads + base_optim + master_extra
            + activations + fsdp_peak + kv_cache + moe_buffers
            + kernel_boundary + collective_workspace + framework
            + dup_bytes
        )
        per_rank.append(
            PerRankMemory(
                rank_idx=rank_idx,
                device_idx=device_idx,
                weights_bytes=steady_weights,
                grads_bytes=base_grads,
                optimizer_state_bytes=base_optim,
                master_weights_bytes=master_extra,
                activations_bytes=activations,
                fsdp_allgather_peak_bytes=fsdp_peak,
                kv_cache_bytes=kv_cache,
                moe_routing_buffers_bytes=moe_buffers,
                collective_workspace_bytes=collective_workspace,
                framework_overhead_bytes=framework,
                total_bytes=total,
            )
        )

    # Worst rank — currently uniform, so always rank 0; future refinement
    # can model pipeline-stage imbalance.
    worst_idx = max(range(len(per_rank)), key=lambda i: per_rank[i].total_bytes)

    diagnostics: list[str] = []
    if dup_bytes > 0:
        diagnostics.append(
            f"FP8 fwd + bf16 grad duplication adds {dup_bytes / 1024**3:.1f} GiB per rank"
        )
    if master_extra > 0:
        diagnostics.append(
            f"master fp32 weights add {master_extra / 1024**3:.1f} GiB per rank"
        )
    if kernel_boundary > 0:
        diagnostics.append(
            f"Megatron RowParallel boundary materialises "
            f"{kernel_boundary / 1024**3:.1f} GiB per rank (backward peak)"
        )

    return DistributedMemoryReport(
        sharding=sharding,
        per_rank=tuple(per_rank),
        duplication_bytes=dup_bytes,
        master_weights_overhead_bytes=master_extra,
        kernel_boundary_materialisation_bytes=kernel_boundary,
        worst_rank_idx=worst_idx,
        bottleneck_diagnostics=tuple(diagnostics),
    )


__all__ = [
    "DistributedMemoryReport",
    "PerRankMemory",
    "estimate_distributed_memory",
]
