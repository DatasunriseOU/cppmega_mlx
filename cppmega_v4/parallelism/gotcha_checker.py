"""Known-gotcha table — catches distributed-training footguns at planning
time so the GUI can flag them BEFORE training launches.

Every Gotcha record encodes one lesson learned by the team. The
``reference`` field points back to the file or doc in ``../nanochat``
or ``../cppmega`` where the original investigation lives, so future
engineers can audit / update the entry when the upstream gotcha is
fixed (or shifts).

The check is purely a function of (ShardingSpec, ModelBuildSpec) — no
runtime; suitable for the real-time GUI inner loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from cppmega_v4.buildspec import ModelBuildSpec
from cppmega_v4.parallelism.sharding_spec import (
    ParallelismKind,
    ShardingSpec,
)


class GotchaSeverity(str, Enum):
    """Mirrors cppmega_v4.spec.DiagnosticSeverity so GUIs share one legend."""

    INFO    = "info"
    WARNING = "warning"
    ERROR   = "error"


@dataclass(frozen=True)
class Gotcha:
    """One row in the gotcha table."""

    gotcha_id: str
    severity: GotchaSeverity
    condition: Callable[[ShardingSpec, ModelBuildSpec], bool]
    message: str
    reference: str            # file:line / doc anchor in nanochat / cppmega


# ---------------------------------------------------------------------------
# Helpers used in trigger predicates
# ---------------------------------------------------------------------------


def _kinds(s: ShardingSpec) -> frozenset[ParallelismKind]:
    return s.axis_kinds()


def _has(s: ShardingSpec, *kinds: ParallelismKind) -> bool:
    return any(k in _kinds(s) for k in kinds)


def _is_tpu(s: ShardingSpec) -> bool:
    return s.topology.devices[0].kind.value.startswith("tpu")


def _has_moe(b: ModelBuildSpec) -> bool:
    return any(n.kind in {"moe", "bailing_moe"} for n in b.graph.nodes)


def _has_attention(b: ModelBuildSpec) -> bool:
    return any(
        n.kind in {
            "attention", "gated_attention", "gqa_sliding", "cca_attention",
            "mla", "mla_absorb", "mistral4_mla", "bailing_mla",
            "dsv4_attention", "nsa", "csa_hca",
        }
        for n in b.graph.nodes
    )


def _num_experts(b: ModelBuildSpec) -> int:
    return int(b.dim_env.get("num_experts", 0))


def _is_apple_silicon(s: ShardingSpec) -> bool:
    kind = s.topology.devices[0].kind.value
    return kind in {"m3_ultra", "gb10"}


def _is_gpu(s: ShardingSpec) -> bool:
    kind = s.topology.devices[0].kind.value
    return kind in {"h100_80gb", "h200_141gb", "a100_40gb", "a100_80gb", "b100_80gb"}


def _is_nvlink_gpu_topology(s: ShardingSpec) -> bool:
    if not _is_gpu(s):
        return False
    return any("nvlink" in d.interconnect for d in s.topology.devices)



# ---------------------------------------------------------------------------
# The gotcha table — every entry has a real upstream provenance.
# ---------------------------------------------------------------------------


GOTCHAS: tuple[Gotcha, ...] = (
    Gotcha(
        gotcha_id="fsdp2_whole_compile",
        severity=GotchaSeverity.ERROR,
        condition=lambda s, b: (
            _has(s, ParallelismKind.FSDP2, ParallelismKind.FSDP1)
            and s.compile_mode == "whole_model"
        ),
        message=(
            "FSDP2 + whole-model torch.compile produces flat loss "
            "(gradients never sync; PyTorch #144376). Use "
            "compile_mode='regional' — compile each TransformerBlock "
            "BEFORE FSDP2-wrapping."
        ),
        reference="nanochat/CLAUDE.md (fsdp2_compile_section); "
                  "nanochat/fsdp_cuda.py",
    ),
    Gotcha(
        gotcha_id="megatron_tp_whole_compile",
        severity=GotchaSeverity.ERROR,
        condition=lambda s, b: (
            ParallelismKind.TP in _kinds(s)
            and s.compile_mode == "whole_model"
        ),
        message=(
            "Megatron TP + whole-model compile = NaN step 1 (hooks "
            "reorder, param.grad=None triggers recompile mid-backward, "
            "PyTorch #118435). Use compile_mode='regional'."
        ),
        reference="nanochat/scripts/base_train.py (regional_compile flag); "
                  "nanochat/megatron_tp.py:69-108",
    ),
    Gotcha(
        gotcha_id="fp8_grad_duplication",
        severity=GotchaSeverity.WARNING,
        condition=lambda s, b: s.fp8_enabled,
        message=(
            "FP8 forward forces bf16/fp32 grad copies — no FP8 backward "
            "kernel. Effective weight+grad ≈ 3× param-elements (fp8 weight "
            "+ bf16 grad), not the naive 0.5× FP8 promises. Use Muon "
            "(2 B/param momentum) to mitigate AdamW (8 B/param) overhead."
        ),
        reference="nanochat/fp8_training.py:1-32; "
                  "nanochat/CLAUDE.md (fp8_grad_section)",
    ),
    Gotcha(
        gotcha_id="master_fp32_duplication",
        severity=GotchaSeverity.WARNING,
        condition=lambda s, b: s.master_weights_fp32,
        message=(
            "master_weights_fp32=True keeps an fp32 master copy alongside "
            "bf16/fp8 compute weights. Doubles param memory + doubles "
            "optimizer state. Use Muon + bf16 loss scaling instead "
            "(nanochat production pattern)."
        ),
        reference="nanochat/megatron_optimizer.py (master_weight section)",
    ),
    Gotcha(
        gotcha_id="ep_more_than_16_experts_xla",
        severity=GotchaSeverity.WARNING,
        condition=lambda s, b: (
            _is_tpu(s)
            and _num_experts(b) > 16
            and ParallelismKind.EP in _kinds(s)
        ),
        message=(
            "TPU XLA has a 4 GB single-tensor limit. With >16 fused MoE "
            "experts the [N_experts, C_e_fused, D] tensor can exceed it "
            "and crash. Set --xla_tpu_rwb_fusion=false OR keep "
            "experts/rank ≤ 16."
        ),
        reference="nanochat/memory_estimator.py:LLO_4GB_section",
    ),
    Gotcha(
        gotcha_id="pp_comm_stream_broken",
        severity=GotchaSeverity.INFO,
        condition=lambda s, b: _has(s, ParallelismKind.PP, ParallelismKind.PP_VPP),
        message=(
            "PP comm-stream separation patch is BROKEN in torch.distributed."
            "pipelining (the NANOCHAT_PP_COMM_STREAM env var exists but is "
            "disabled by default). P2P comm serialises on the default "
            "stream — no overlap with compute. Expect ~10-20% throughput "
            "hit vs Megatron GPipe."
        ),
        reference="nanochat/pipeline_parallel.py:1-50",
    ),
    Gotcha(
        gotcha_id="megatron_row_parallel_boundary",
        severity=GotchaSeverity.INFO,
        condition=lambda s, b: ParallelismKind.TP in _kinds(s),
        message=(
            "Megatron RowParallelLinear AllReduce materialises the full "
            "(B,S,H) tensor on every rank at the layer's backward boundary. "
            "Gradient checkpointing CAN'T elide this — it must be saved "
            "for the backward computation."
        ),
        reference="nanochat/megatron_tp.py (RowParallel section); "
                  "cppmega/cppmega/megatron/memory_debug.py:258-302",
    ),
    Gotcha(
        gotcha_id="fsdp_allgather_peak_unsharded",
        severity=GotchaSeverity.INFO,
        condition=lambda s, b: _has(s, ParallelismKind.FSDP2, ParallelismKind.FSDP1),
        message=(
            "FSDP all-gather peak == unsharded parameter size. Steady-state "
            "memory is divided by dp_degree, but during the forward pass "
            "each block briefly materialises its full unsharded weight on "
            "every rank. Plan worst-case peak around one fully-gathered block."
        ),
        reference="nanochat/fsdp_cuda.py; "
                  "nanochat/memory_estimator.py (FSDP all-gather section)",
    ),
    Gotcha(
        gotcha_id="sp_replicated_param_allreduce_overhead",
        severity=GotchaSeverity.INFO,
        condition=lambda s, b: ParallelismKind.SP in _kinds(s),
        message=(
            "Sequence-parallel with TP: small replicated params (norms, "
            "qk-norms) all-reduce separately AFTER backward fills the grad. "
            "No overlap with GEMM. Use TE's communicate_next_backward to "
            "fuse with the next layer's GEMM."
        ),
        reference="nanochat/megatron_tp.py (_install_sequence_parallel_hooks)",
    ),
    Gotcha(
        gotcha_id="tp_master_weights_double_state",
        severity=GotchaSeverity.WARNING,
        condition=lambda s, b: (
            ParallelismKind.TP in _kinds(s)
            and s.master_weights_fp32
        ),
        message=(
            "TP + master_weights_fp32: each rank holds the local TP weight "
            "shard in BOTH bf16 (compute) AND fp32 (optimizer state). The "
            "duplication isn't solved at the library level. Drop master "
            "weights or accept the per-rank doubling."
        ),
        reference="nanochat/megatron_optimizer.py (TP master section)",
    ),
    Gotcha(
        gotcha_id="dp_no_optim_sharding",
        severity=GotchaSeverity.INFO,
        condition=lambda s, b: (
            ParallelismKind.DP in _kinds(s)
            and not _has(s, ParallelismKind.FSDP2, ParallelismKind.FSDP1,
                          ParallelismKind.ZERO1, ParallelismKind.ZERO2)
            and s.topology.num_devices > 1
        ),
        message=(
            "DP-only (no FSDP / no ZeRO) replicates the full optimizer state "
            "on every rank. For >5B-param models on 80 GB devices this is "
            "usually the bottleneck. Consider FSDP2 (ZeRO-3) — optim sharded "
            "by dp_degree at near-zero throughput cost."
        ),
        reference="cppmega/docs/memory_dtype_audit_2026_04_25.md",
    ),
    Gotcha(
        gotcha_id="grad_reduce_fp32_doubles_grad_buffer",
        severity=GotchaSeverity.WARNING,
        condition=lambda s, b: s.grad_reduce_dtype == "fp32",
        message=(
            "grad_reduce_dtype='fp32' doubles the gradient-buffer cost vs "
            "bf16 reduction. Use bf16 unless you're chasing the last 0.5% "
            "of numerical stability on very deep stacks."
        ),
        reference="cppmega/cppmega/megatron/memory_debug.py:258-302 "
                  "(--local-ddp-disable-contiguous-grad-buffer notes)",
    ),
    Gotcha(
        gotcha_id="ep_without_moe",
        severity=GotchaSeverity.WARNING,
        condition=lambda s, b: (
            ParallelismKind.EP in _kinds(s)
            and not _has_moe(b)
        ),
        message=(
            "EP axis declared but no MoE bricks in the model. EP has nothing "
            "to shard — wastes the mesh axis. Drop EP or add a 'moe' brick."
        ),
        reference="cppmega/cppmega/recipes/megatron_args.py (EP guard)",
    ),
    Gotcha(
        gotcha_id="checkpointing_off_with_large_seq",
        severity=GotchaSeverity.WARNING,
        condition=lambda s, b: (
            s.activation_checkpointing == "off"
            and int(b.dim_env.get("S", 0)) >= 4096
        ),
        message=(
            "activation_checkpointing='off' at S ≥ 4096: activations dominate "
            "memory and likely OOM. Use 'full' (only block boundaries) or "
            "'selective' (per-layer cherry-pick — Mamba-style)."
        ),
        reference="nanochat/memory_estimator.py:activations_section",
    ),
    Gotcha(
        gotcha_id="fp8_with_sgd_loses_precision",
        severity=GotchaSeverity.INFO,
        condition=lambda s, b: (
            s.fp8_enabled
            and b.optim.kind.value == "sgd"
        ),
        message=(
            "FP8 forward + SGD: no momentum to smooth out quantisation noise. "
            "Use AdamW (or Muon — preferred for bf16-state optimizer) when "
            "FP8 is enabled. SGD + FP8 typically diverges by step ~200."
        ),
        reference="cppmega/docs/memory_dtype_audit_2026_04_25.md "
                  "(precision-aware storage ladder)",
    ),
    Gotcha(
        gotcha_id="incompatible_comm_backend",
        severity=GotchaSeverity.ERROR,
        condition=lambda s, b: (
            (s.comm_backend == "nccl" and (_is_apple_silicon(s) or _is_tpu(s)))
            or (s.comm_backend == "pjrt" and (_is_apple_silicon(s) or _is_gpu(s)))
            or (s.comm_backend == "jaccl" and not _is_apple_silicon(s))
        ),
        message=(
            "Incompatible communication backend: 'nccl' requires Nvidia GPUs, "
            "'pjrt' requires TPUs, and 'jaccl' requires Apple Silicon hardware. "
            "Select a communication backend matching the hardware topology."
        ),
        reference="cppmega_v4/parallelism/sharding_spec.py; AGENTS.md",
    ),
    Gotcha(
        gotcha_id="slow_loopback_ring",
        severity=GotchaSeverity.WARNING,
        condition=lambda s, b: (
            s.comm_backend == "ring" and _is_nvlink_gpu_topology(s)
        ),
        message=(
            "Using MLX Ring over TCP ('ring') on an NVLink-interconnected GPU topology. "
            "This will incur significant communication latency bottlenecks. Use 'nccl' "
            "for NVLink inter-GPU collective acceleration."
        ),
        reference="cppmega_v4/parallelism/sharding_spec.py; gotcha_checker.py",
    ),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_gotchas(
    sharding: ShardingSpec,
    build_spec: ModelBuildSpec,
    *,
    gotchas: tuple[Gotcha, ...] | None = None,
) -> tuple[Gotcha, ...]:
    """Run every gotcha trigger; return all that fire.

    Pure function — no runtime, suitable for the GUI inner loop on every
    sharding-spec / model-spec mutation.
    """
    gotcha_table = GOTCHAS if gotchas is None else gotchas
    fired: list[Gotcha] = []
    for g in gotcha_table:
        try:
            if g.condition(sharding, build_spec):
                fired.append(g)
        except Exception:
            # Defensive: a trigger predicate should never raise; if it
            # does (e.g. a missing dim_env key) we treat as not-fired
            # rather than break the whole check.
            continue
    return tuple(fired)


__all__ = [
    "GOTCHAS",
    "Gotcha",
    "GotchaSeverity",
    "check_gotchas",
]
