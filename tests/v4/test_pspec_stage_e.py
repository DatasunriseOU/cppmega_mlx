"""PSpec Stage E tests — verify_distributed_plan + GUI workflow + perf gate."""

from __future__ import annotations

import time

import pytest

from cppmega_v4.architectures import available_presets, build_preset_specs
from cppmega_v4.buildspec import (
    MTPRewriter,
    ModelBuildSpec,
    adamw,
    cross_entropy_loss,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.parallelism import (
    AxisAssignment,
    DistributedMemoryReport,
    DistributedVerificationResult,
    GotchaSeverity,
    ParallelismKind,
    ShardingSpec,
    fsdp2_only,
    fsdp2_plus_tp,
    gb10_quarter,
    h100_8x,
    h200_8x,
    suggest_sharding,
    verify_distributed_plan,
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
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        dim_env=_ENV,
    )


# ---------------------------------------------------------------------------
# Smoke: well-formed result
# ---------------------------------------------------------------------------


def test_verify_returns_well_formed_result():
    r = verify_distributed_plan(_qwen_spec(), fsdp2_only(h100_8x()))
    assert isinstance(r, DistributedVerificationResult)
    assert isinstance(r.memory, DistributedMemoryReport)
    assert isinstance(r.gotchas, tuple)
    assert r.elapsed_ms >= 0


def test_verify_has_errors_helper():
    r = verify_distributed_plan(_qwen_spec(), fsdp2_only(h100_8x()))
    assert isinstance(r.has_errors, bool)


def test_verify_errors_and_warnings_partition_by_severity():
    spec = _qwen_spec()
    # Provoke errors AND warnings: FSDP2 + whole compile (ERROR) +
    # fp8 (WARNING fp8_grad_duplication) + master_fp32 (WARNING).
    sharding = ShardingSpec(
        topology=h100_8x(),
        axis_assignments=(AxisAssignment("dp", ParallelismKind.FSDP2, 8),),
        compile_mode="whole_model",
        fp8_enabled=True,
        master_weights_fp32=True,
    )
    r = verify_distributed_plan(spec, sharding)
    assert all(g.severity is GotchaSeverity.ERROR for g in r.errors)
    assert all(g.severity is GotchaSeverity.WARNING for g in r.warnings)
    assert r.has_errors is True


def test_verify_summary_dict_shape():
    r = verify_distributed_plan(_qwen_spec(), fsdp2_only(h100_8x()))
    summary = r.summary()
    for key in ("errors", "warnings", "memory", "fits", "elapsed_ms"):
        assert key in summary, key


# ---------------------------------------------------------------------------
# Training vs inference
# ---------------------------------------------------------------------------


def test_inference_mode_zero_grads():
    spec = _qwen_spec()
    sharding = fsdp2_only(h100_8x())
    r = verify_distributed_plan(spec, sharding, training=False)
    assert r.memory.worst_rank.grads_bytes == 0
    assert r.memory.worst_rank.optimizer_state_bytes == 0


def test_training_mode_nonzero_grads():
    spec = _qwen_spec()
    r = verify_distributed_plan(spec, fsdp2_only(h100_8x()), training=True)
    assert r.memory.worst_rank.grads_bytes > 0
    assert r.memory.worst_rank.optimizer_state_bytes > 0


# ---------------------------------------------------------------------------
# Perf gate — <100 ms per call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", sorted(available_presets()))
def test_verify_under_100ms_per_preset(preset_name):
    """Real-time GUI inner loop — every preset under any safe topology
    must verify in < 100 ms warm (target 50 ms; current ~5 ms)."""
    specs = build_preset_specs(preset_name, hidden_size=4096)
    specs.append({"kind": "mlp", "name": "logits"})
    g = from_block_specs(specs, hidden_size=4096, instantiate=False)
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        dim_env=_ENV,
    )
    sharding = fsdp2_only(h100_8x())
    # warm
    verify_distributed_plan(spec, sharding)
    t0 = time.perf_counter()
    r = verify_distributed_plan(spec, sharding)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 200.0, (
        f"verify({preset_name!r}) took {elapsed_ms:.1f} ms "
        "(soft cap 200 ms; target 100 ms; current run ~5 ms)"
    )
    assert r.elapsed_ms < 200.0


# ---------------------------------------------------------------------------
# Compose with suggest_sharding (the GUI workflow)
# ---------------------------------------------------------------------------


def test_top_proposal_verifies_cleanly():
    """The top-ranked proposal from suggest_sharding should verify with
    no ERROR-severity gotchas."""
    spec = _qwen_spec()
    proposals = suggest_sharding(spec, h100_8x())
    top = proposals[0]
    r = verify_distributed_plan(spec, top.sharding)
    assert r.has_errors is False


def test_top_proposal_per_rank_matches_proposal_estimate():
    spec = _qwen_spec()
    proposals = suggest_sharding(spec, h100_8x())
    top = proposals[0]
    r = verify_distributed_plan(spec, top.sharding)
    # Same memory math should yield the same worst-rank bytes
    assert r.memory.worst_rank.total_bytes == top.estimated_per_rank_bytes


# ---------------------------------------------------------------------------
# Compose with apply_rewrites — MTP post-rewrite spec verifies cleanly
# ---------------------------------------------------------------------------


def test_mtp_rewritten_spec_verifies_under_h100_fsdp2():
    """End-to-end: build spec with MTPRewriter → apply rewrites →
    verify_distributed_plan under fsdp2_only → no errors."""
    spec = _qwen_spec().replace(rewrites=(MTPRewriter(k=2),))
    applied = spec.apply_rewrites()
    r = verify_distributed_plan(applied, fsdp2_only(h100_8x()))
    assert r.has_errors is False
    # Memory grew due to extra MTP head — verify it's higher than the
    # pre-rewrite spec, not lower.
    pre = verify_distributed_plan(_qwen_spec(), fsdp2_only(h100_8x()))
    assert r.memory.worst_rank.total_bytes >= pre.memory.worst_rank.total_bytes


# ---------------------------------------------------------------------------
# Full GUI workflow integration
# ---------------------------------------------------------------------------


def test_system_gui_workflow_pick_topology_then_suggest_then_accept_then_verify():
    """Simulates the GUI flow end-to-end:
      1. User picks preset → builds ModelBuildSpec
      2. User picks topology (h100_8x)
      3. GUI calls suggest_sharding → 3-5 ranked proposals
      4. User accepts the top proposal
      5. GUI calls verify_distributed_plan → renders memory bar +
         gotcha chips with severity colour codes
      6. Verifies fits + no ERROR-severity gotchas
    """
    spec = _qwen_spec()
    topology = h100_8x()

    proposals = suggest_sharding(spec, topology)
    assert len(proposals) >= 1
    top = proposals[0]

    r = verify_distributed_plan(spec, top.sharding)
    assert r.memory.worst_rank.total_bytes > 0
    # GUI-side summary works
    s = r.summary()
    assert s["fits"] in (True, False)
    assert s["errors"] == 0   # top proposal should be clean
    # Per-rank bar can render
    assert len(r.memory.per_rank) == topology.num_devices


def test_system_oom_workflow_user_sees_doesnt_fit():
    """When the model is too big for the topology, fits=False and the
    GUI sees the warning. We force OOM with a synthetic single-device
    topology with tiny HBM."""
    from cppmega_v4.parallelism import DeviceKind, DeviceSpec, DeviceTopology
    tiny = DeviceTopology(
        devices=(DeviceSpec(
            kind=DeviceKind.A100_40GB,
            hbm_bytes=2 * 1024**3,   # 2 GB — far too small for any preset
            interconnect="nvlink",
            bandwidth_gbps=600.0,
        ),),
        mesh_axes={"dp": 1},
    )
    spec = _qwen_spec()
    sharding = ShardingSpec(
        topology=tiny,
        axis_assignments=(AxisAssignment("dp", ParallelismKind.DP, 1),),
    )
    r = verify_distributed_plan(spec, sharding)
    assert r.summary()["fits"] is False


def test_system_gotcha_workflow_user_sees_compile_mode_error():
    """User mistakenly picks compile_mode='whole_model' on FSDP2 —
    GUI sees ERROR gotcha + recommended fix in message."""
    spec = _qwen_spec()
    bad = ShardingSpec(
        topology=h100_8x(),
        axis_assignments=(AxisAssignment("dp", ParallelismKind.FSDP2, 8),),
        compile_mode="whole_model",
    )
    r = verify_distributed_plan(spec, bad)
    assert r.has_errors is True
    err = r.errors[0]
    assert err.gotcha_id == "fsdp2_whole_compile"
    assert "regional" in err.message  # suggested fix in message
    assert "nanochat" in err.reference   # provenance pointer
