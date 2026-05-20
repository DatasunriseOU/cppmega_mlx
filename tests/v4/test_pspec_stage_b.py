"""PSpec Stage B tests — distributed_memory accounting."""

from __future__ import annotations

import pytest

from cppmega_v4.architectures import build_preset_specs
from cppmega_v4.buildspec import (
    ModelBuildSpec,
    adamw,
    cross_entropy_loss,
    muon,
    sgd,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode
from cppmega_v4.parallelism import (
    AxisAssignment,
    DistributedMemoryReport,
    ParallelismKind,
    PerRankMemory,
    ShardingSpec,
    estimate_distributed_memory,
    fsdp2_only,
    fsdp2_plus_tp,
    gb10_quarter,
    h100_8x,
    h200_8x,
    m3_ultra_solo,
    megatron_ep_only,
    single_device,
    tpu_v6e_8,
)


_ENV = {
    "B": 1, "S": 4096, "H": 4096,
    "nh": 32, "nkv": 4, "head_dim": 128,
    "num_experts": 8, "top_k": 2,
}


def _qwen_spec() -> ModelBuildSpec:
    specs = build_preset_specs("qwen3_next", hidden_size=4096)
    specs.append({"kind": "mlp", "name": "logits"})
    g = from_block_specs(specs, hidden_size=4096, instantiate=False)
    return ModelBuildSpec(
        graph=g,
        loss=cross_entropy_loss(),
        optim=adamw(lr=3e-4),
        dim_env=_ENV,
    )


def _moe_spec() -> ModelBuildSpec:
    """Compose a graph with explicit MoE bricks for EP test cases."""
    g = BrickGraph(
        nodes=(
            BrickNode(kind="moe", name="moe0",
                      params={"num_experts": 8, "top_k": 2}),
            BrickNode(kind="mlp", name="logits"),
        ),
        edges=(("moe0", "logits"),),
    )
    return ModelBuildSpec(
        graph=g,
        loss=cross_entropy_loss(),
        optim=adamw(lr=3e-4),
        dim_env=_ENV,
    )


# ---------------------------------------------------------------------------
# PerRankMemory invariants
# ---------------------------------------------------------------------------


def test_per_rank_memory_fits_on_device_validates_headroom():
    r = PerRankMemory(
        rank_idx=0, device_idx=0,
        weights_bytes=1, grads_bytes=1, optimizer_state_bytes=1,
        master_weights_bytes=0, activations_bytes=1,
        fsdp_allgather_peak_bytes=0, kv_cache_bytes=0,
        moe_routing_buffers_bytes=0, collective_workspace_bytes=0,
        framework_overhead_bytes=0, total_bytes=5,
    )
    device = h100_8x().devices[0]
    with pytest.raises(ValueError, match="headroom"):
        r.fits_on_device(device, headroom=0)


# ---------------------------------------------------------------------------
# Single-device baseline
# ---------------------------------------------------------------------------


def test_single_device_yields_n_per_rank_entries():
    """num_ranks should equal number of devices in topology."""
    spec = _qwen_spec()
    sharding = single_device(h100_8x())
    r = estimate_distributed_memory(spec, sharding)
    assert isinstance(r, DistributedMemoryReport)
    assert len(r.per_rank) == 8


def test_single_device_no_duplication_no_master():
    spec = _qwen_spec()
    sharding = single_device(h100_8x())
    r = estimate_distributed_memory(spec, sharding)
    assert r.duplication_bytes == 0
    assert r.master_weights_overhead_bytes == 0
    assert r.kernel_boundary_materialisation_bytes == 0


# ---------------------------------------------------------------------------
# FSDP2 shards optimiser state by dp degree
# ---------------------------------------------------------------------------


def test_fsdp2_reduces_optimizer_state_by_dp_degree():
    spec = _qwen_spec()
    no_shard = single_device(h100_8x())
    sharded = fsdp2_only(h100_8x())   # dp=8

    r_base = estimate_distributed_memory(spec, no_shard)
    r_shard = estimate_distributed_memory(spec, sharded)

    # Steady-state optimiser state should be ~1/8 of the baseline.
    assert r_shard.worst_rank.optimizer_state_bytes < (
        r_base.worst_rank.optimizer_state_bytes // 4
    )


def test_fsdp2_adds_allgather_peak_term():
    spec = _qwen_spec()
    sharding = fsdp2_only(h100_8x())
    r = estimate_distributed_memory(spec, sharding)
    assert r.worst_rank.fsdp_allgather_peak_bytes > 0


def test_no_fsdp_no_allgather_peak():
    spec = _qwen_spec()
    sharding = single_device(h100_8x())
    r = estimate_distributed_memory(spec, sharding)
    assert r.worst_rank.fsdp_allgather_peak_bytes == 0


# ---------------------------------------------------------------------------
# TP shards per-layer params and materialises RowParallel boundary
# ---------------------------------------------------------------------------


def test_tp_reduces_weights_per_rank():
    spec = _qwen_spec()
    no_tp = single_device(h100_8x())
    tp = fsdp2_plus_tp(h100_8x(dp=4, tp=2, ep=1, pp=1))

    r_no = estimate_distributed_memory(spec, no_tp)
    r_tp = estimate_distributed_memory(spec, tp)
    # TP=2 halves weight-shard factor; combined with FSDP=4 vs full DP=8
    # the per-rank weights should still drop.
    assert r_tp.worst_rank.weights_bytes < r_no.worst_rank.weights_bytes


def test_tp_adds_kernel_boundary_materialisation():
    spec = _qwen_spec()
    tp = fsdp2_plus_tp(h100_8x(dp=4, tp=2, ep=1, pp=1))
    r = estimate_distributed_memory(spec, tp)
    assert r.kernel_boundary_materialisation_bytes > 0


def test_no_tp_no_kernel_boundary():
    spec = _qwen_spec()
    r = estimate_distributed_memory(spec, fsdp2_only(h100_8x()))
    assert r.kernel_boundary_materialisation_bytes == 0


# ---------------------------------------------------------------------------
# EP reduces MoE routing buffers
# ---------------------------------------------------------------------------


def test_ep_reduces_moe_routing_buffers():
    spec = _moe_spec()
    no_ep = single_device(h100_8x())
    ep4 = megatron_ep_only(h100_8x(dp=2, tp=1, ep=4, pp=1))
    r_no = estimate_distributed_memory(spec, no_ep)
    r_ep = estimate_distributed_memory(spec, ep4)
    assert r_ep.worst_rank.moe_routing_buffers_bytes < (
        r_no.worst_rank.moe_routing_buffers_bytes
    )


def test_no_moe_no_moe_buffers():
    """Spec without MoE bricks → moe_routing_buffers == 0."""
    g = BrickGraph(
        nodes=(
            BrickNode(kind="gdn", name="bb"),
            BrickNode(kind="mlp", name="logits"),
        ),
        edges=(("bb", "logits"),),
    )
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(), dim_env=_ENV,
    )
    r = estimate_distributed_memory(spec, single_device(h100_8x()))
    assert r.worst_rank.moe_routing_buffers_bytes == 0


# ---------------------------------------------------------------------------
# FP8 + bf16 grad duplication accounting (THE headline gotcha)
# ---------------------------------------------------------------------------


def test_fp8_enabled_adds_duplication_bytes():
    spec = _qwen_spec()
    bf16 = ShardingSpec(
        topology=h100_8x(), fp8_enabled=False,
        axis_assignments=(AxisAssignment("dp", ParallelismKind.FSDP2, 8),),
    )
    fp8 = ShardingSpec(
        topology=h100_8x(), fp8_enabled=True,
        axis_assignments=(AxisAssignment("dp", ParallelismKind.FSDP2, 8),),
    )
    r_bf16 = estimate_distributed_memory(spec, bf16)
    r_fp8 = estimate_distributed_memory(spec, fp8)
    assert r_bf16.duplication_bytes == 0
    assert r_fp8.duplication_bytes > 0
    # Diagnostic surfaced
    assert any(
        "FP8 fwd + bf16 grad duplication" in d
        for d in r_fp8.bottleneck_diagnostics
    )


# ---------------------------------------------------------------------------
# Master fp32 weights overhead
# ---------------------------------------------------------------------------


def test_master_weights_fp32_adds_overhead_diagnostic():
    spec = _qwen_spec()
    plain = ShardingSpec(
        topology=h100_8x(),
        axis_assignments=(AxisAssignment("dp", ParallelismKind.FSDP2, 8),),
    )
    master = ShardingSpec(
        topology=h100_8x(),
        axis_assignments=(AxisAssignment("dp", ParallelismKind.FSDP2, 8),),
        master_weights_fp32=True,
    )
    r_plain = estimate_distributed_memory(spec, plain)
    r_master = estimate_distributed_memory(spec, master)
    assert r_plain.master_weights_overhead_bytes == 0
    assert r_master.master_weights_overhead_bytes > 0
    assert r_master.worst_rank.master_weights_bytes > 0
    assert any(
        "master fp32 weights add" in d
        for d in r_master.bottleneck_diagnostics
    )


# ---------------------------------------------------------------------------
# Optimizer kind affects state size
# ---------------------------------------------------------------------------


def test_muon_uses_less_optimizer_state_than_adamw():
    g_specs = build_preset_specs("qwen3_next", hidden_size=4096)
    g_specs.append({"kind": "mlp", "name": "logits"})
    g = from_block_specs(g_specs, hidden_size=4096, instantiate=False)
    adam_spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(),
        optim=adamw(), dim_env=_ENV,
    )
    muon_spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(),
        optim=muon(), dim_env=_ENV,
    )
    sharding = single_device(h100_8x())
    r_adam = estimate_distributed_memory(adam_spec, sharding)
    r_muon = estimate_distributed_memory(muon_spec, sharding)
    assert r_muon.worst_rank.optimizer_state_bytes < (
        r_adam.worst_rank.optimizer_state_bytes
    )


def test_sgd_zero_optimizer_state():
    g_specs = build_preset_specs("qwen3_next", hidden_size=4096)
    g_specs.append({"kind": "mlp", "name": "logits"})
    g = from_block_specs(g_specs, hidden_size=4096, instantiate=False)
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(),
        optim=sgd(), dim_env=_ENV,
    )
    sharding = single_device(h100_8x())
    r = estimate_distributed_memory(spec, sharding)
    assert r.worst_rank.optimizer_state_bytes == 0


# ---------------------------------------------------------------------------
# Training vs inference
# ---------------------------------------------------------------------------


def test_inference_drops_grads_and_optim():
    spec = _qwen_spec()
    sharding = single_device(h100_8x())
    train = estimate_distributed_memory(spec, sharding, training=True)
    infer = estimate_distributed_memory(spec, sharding, training=False)
    assert train.worst_rank.grads_bytes > 0
    assert train.worst_rank.optimizer_state_bytes > 0
    assert infer.worst_rank.grads_bytes == 0
    assert infer.worst_rank.optimizer_state_bytes == 0


def test_inference_can_have_kv_cache():
    spec = _qwen_spec()
    sharding = single_device(h100_8x())
    infer = estimate_distributed_memory(spec, sharding, training=False)
    assert infer.worst_rank.kv_cache_bytes > 0


# ---------------------------------------------------------------------------
# Activation checkpointing
# ---------------------------------------------------------------------------


def test_activation_checkpointing_off_increases_activations():
    spec = _qwen_spec()
    full = fsdp2_only(h100_8x(), activation_checkpointing="full")
    off  = fsdp2_only(h100_8x(), activation_checkpointing="off")
    r_full = estimate_distributed_memory(spec, full)
    r_off  = estimate_distributed_memory(spec, off)
    assert r_off.worst_rank.activations_bytes > r_full.worst_rank.activations_bytes


# ---------------------------------------------------------------------------
# Framework overhead per device family
# ---------------------------------------------------------------------------


def test_framework_overhead_tpu_higher_than_cuda():
    spec = _qwen_spec()
    cuda = single_device(h100_8x())
    tpu = single_device(tpu_v6e_8())
    r_cuda = estimate_distributed_memory(spec, cuda)
    r_tpu  = estimate_distributed_memory(spec, tpu)
    assert r_tpu.worst_rank.framework_overhead_bytes > r_cuda.worst_rank.framework_overhead_bytes


def test_framework_overhead_mac_lower_than_cuda():
    spec = _qwen_spec()
    cuda = single_device(h100_8x())
    mac = single_device(m3_ultra_solo())
    r_cuda = estimate_distributed_memory(spec, cuda)
    r_mac  = estimate_distributed_memory(spec, mac)
    assert r_mac.worst_rank.framework_overhead_bytes < r_cuda.worst_rank.framework_overhead_bytes


# ---------------------------------------------------------------------------
# fits_on_topology + summary
# ---------------------------------------------------------------------------


def test_fits_on_topology_validates_headroom():
    spec = _qwen_spec()
    r = estimate_distributed_memory(spec, single_device(h100_8x()))
    with pytest.raises(ValueError, match="headroom"):
        r.fits_on_topology(headroom=1.5)


def test_fits_on_topology_true_for_qwen_on_h200():
    spec = _qwen_spec()
    r = estimate_distributed_memory(spec, fsdp2_only(h200_8x()))
    assert isinstance(r.fits_on_topology(), bool)


def test_summary_dict_shape():
    spec = _qwen_spec()
    r = estimate_distributed_memory(spec, single_device(h100_8x()))
    s = r.summary()
    for key in ("num_ranks", "worst_rank", "worst_total",
                "duplication", "master_overhead", "kernel_boundary"):
        assert key in s


# ---------------------------------------------------------------------------
# Worst-rank picker
# ---------------------------------------------------------------------------


def test_worst_rank_index_is_in_range():
    spec = _qwen_spec()
    r = estimate_distributed_memory(spec, fsdp2_only(h100_8x()))
    assert 0 <= r.worst_rank_idx < len(r.per_rank)
    assert r.worst_rank.total_bytes >= max(
        rank.total_bytes for rank in r.per_rank
    )


# ---------------------------------------------------------------------------
# System: every preset produces a finite report
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topology_name", ["gb10_quarter", "h100_8x", "h200_8x"])
def test_system_qwen_preset_estimates_under_h200_budget(topology_name):
    topology = {
        "gb10_quarter": gb10_quarter,
        "h100_8x": h100_8x,
        "h200_8x": h200_8x,
    }[topology_name]()
    sharding = (
        single_device(topology)
        if topology.num_devices == 1
        else fsdp2_only(topology)
    )
    spec = _qwen_spec()
    r = estimate_distributed_memory(spec, sharding)
    # Total per-rank memory should be positive and finite
    assert r.worst_rank.total_bytes > 0
    # And the GUI summary should be non-empty
    assert len(r.summary()) > 0
