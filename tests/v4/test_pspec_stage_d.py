"""PSpec Stage D tests — auto_shard heuristic proposer."""

from __future__ import annotations

import pytest

from cppmega_v4.architectures import build_preset_specs
from cppmega_v4.buildspec import (
    ModelBuildSpec,
    adamw,
    cross_entropy_loss,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode
from cppmega_v4.parallelism import (
    ParallelismKind,
    ShardingProposal,
    gb10_quarter,
    h100_8x,
    h200_8x,
    m3_ultra_solo,
    suggest_sharding,
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
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        dim_env=_ENV,
    )


def _dense_spec() -> ModelBuildSpec:
    """No MoE — for EP-skipping tests."""
    g = BrickGraph(
        nodes=(
            BrickNode(kind="gated_attention", name="attn",
                      params={"num_attention_heads": 4,
                              "num_key_value_heads": 2, "head_dim": 16}),
            BrickNode(kind="mlp", name="logits"),
        ),
        edges=(("attn", "logits"),),
    )
    return ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        dim_env=_ENV,
    )


# ---------------------------------------------------------------------------
# Output shape + ranking invariants
# ---------------------------------------------------------------------------


def test_suggest_returns_at_least_one_proposal():
    proposals = suggest_sharding(_qwen_spec(), h100_8x())
    assert len(proposals) >= 1
    assert all(isinstance(p, ShardingProposal) for p in proposals)


def test_proposals_sorted_by_score_ascending():
    proposals = suggest_sharding(_qwen_spec(), h100_8x())
    # Each proposal's score should be ≥ the previous one's
    scores: list[int] = []
    for p in proposals:
        if p.num_errors > 0:
            score = 10 ** 15 + p.estimated_per_rank_bytes
        elif not p.fits:
            score = (10 ** 15 // 10) + p.estimated_per_rank_bytes
        else:
            score = p.estimated_per_rank_bytes
        scores.append(score)
    assert scores == sorted(scores)


def test_top_proposal_is_recommended_choice():
    proposals = suggest_sharding(_qwen_spec(), h100_8x())
    top = proposals[0]
    # Top pick must either fit or have no ERROR gotchas
    assert top.fits or top.num_errors == 0


# ---------------------------------------------------------------------------
# Heuristic: small model on single device → single_device wins
# ---------------------------------------------------------------------------


def test_single_device_only_proposal_for_single_device_topology():
    """Single-device topology should produce only single_device proposal
    (no FSDP / EP / TP possible without ≥2 devices)."""
    proposals = suggest_sharding(_qwen_spec(), gb10_quarter())
    names = {p.strategy_name for p in proposals}
    assert names == {"single_device"}


def test_m3_ultra_solo_proposes_single_device_only():
    proposals = suggest_sharding(_qwen_spec(), m3_ultra_solo())
    names = {p.strategy_name for p in proposals}
    assert names == {"single_device"}


# ---------------------------------------------------------------------------
# Heuristic: MoE present → EP-family proposals appear
# ---------------------------------------------------------------------------


def test_moe_model_yields_ep_proposal_when_mesh_has_ep_axis():
    spec = _qwen_spec()   # qwen3_next includes moe brick
    topology = h100_8x(dp=2, tp=1, ep=4, pp=1)
    proposals = suggest_sharding(spec, topology)
    names = {p.strategy_name for p in proposals}
    assert "megatron_ep_only" in names


def test_dense_model_skips_ep_proposals():
    spec = _dense_spec()   # no MoE
    topology = h100_8x(dp=2, tp=1, ep=4, pp=1)
    proposals = suggest_sharding(spec, topology)
    names = {p.strategy_name for p in proposals}
    assert "megatron_ep_only" not in names
    assert "fsdp2_plus_ep" not in names


# ---------------------------------------------------------------------------
# Heuristic: 3D mesh → TP + FSDP appears
# ---------------------------------------------------------------------------


def test_3d_mesh_yields_fsdp2_plus_tp_proposal():
    spec = _qwen_spec()
    topology = h100_8x(dp=4, tp=2, ep=1, pp=1)
    proposals = suggest_sharding(spec, topology)
    names = {p.strategy_name for p in proposals}
    assert "fsdp2_plus_tp" in names


def test_2d_mesh_without_tp_axis_skips_tp_proposal():
    spec = _qwen_spec()
    topology = h100_8x()   # mesh = {"dp":8, "tp":1, ...} → no useful tp
    proposals = suggest_sharding(spec, topology)
    names = {p.strategy_name for p in proposals}
    assert "fsdp2_plus_tp" not in names


# ---------------------------------------------------------------------------
# Multi-device topology → FSDP2 always appears
# ---------------------------------------------------------------------------


def test_multi_device_topology_proposes_fsdp2():
    proposals = suggest_sharding(_qwen_spec(), h100_8x())
    names = {p.strategy_name for p in proposals}
    assert "fsdp2_only" in names


def test_h200_8x_yields_more_proposals_than_single_device():
    multi = suggest_sharding(_qwen_spec(), h200_8x())
    solo = suggest_sharding(_qwen_spec(), m3_ultra_solo())
    assert len(multi) >= len(solo)


# ---------------------------------------------------------------------------
# Compile-mode safety
# ---------------------------------------------------------------------------


def test_all_proposals_use_regional_compile_mode():
    """Every proposal returned by the suggester must use compile_mode=
    'regional' — that's the only safe mode for FSDP2 / Megatron TP per
    the gotcha checker."""
    for topology_factory in (
        lambda: h100_8x(dp=2, tp=2, ep=2, pp=1),
        lambda: h200_8x(),
        lambda: gb10_quarter(),
    ):
        proposals = suggest_sharding(_qwen_spec(), topology_factory())
        for p in proposals:
            assert p.sharding.compile_mode == "regional", (
                f"{p.strategy_name}: compile_mode={p.sharding.compile_mode!r}"
            )


def test_no_proposal_enables_master_weights_fp32_by_default():
    """Master fp32 doubles param+optim memory — should never be the
    default suggestion."""
    proposals = suggest_sharding(_qwen_spec(), h100_8x(dp=4, tp=2, ep=1, pp=1))
    for p in proposals:
        assert p.sharding.master_weights_fp32 is False


# ---------------------------------------------------------------------------
# Reason text + sanity
# ---------------------------------------------------------------------------


def test_every_proposal_has_human_readable_reason():
    proposals = suggest_sharding(_qwen_spec(), h100_8x(dp=2, tp=2, ep=2, pp=1))
    for p in proposals:
        assert p.reason  # non-empty string


def test_every_proposal_carries_gotcha_results():
    proposals = suggest_sharding(_qwen_spec(), h100_8x())
    for p in proposals:
        assert isinstance(p.gotchas, tuple)


def test_proposal_num_errors_helper():
    proposals = suggest_sharding(_qwen_spec(), h100_8x())
    for p in proposals:
        assert p.num_errors == sum(
            1 for g in p.gotchas if g.severity.value == "error"
        )


# ---------------------------------------------------------------------------
# fits flag matches the memory report's verdict
# ---------------------------------------------------------------------------


def test_fits_flag_consistent_with_per_rank_bytes_vs_hbm():
    proposals = suggest_sharding(_qwen_spec(), gb10_quarter())
    for p in proposals:
        device_hbm = p.sharding.topology.devices[0].hbm_bytes
        if p.estimated_per_rank_bytes <= int(device_hbm * 0.9):
            assert p.fits is True
        else:
            assert p.fits is False


# ---------------------------------------------------------------------------
# System: every preset gets at least one proposal across topologies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", [
    "qwen3_next", "kimi_linear", "kimi_k2", "deepseek_v3",
    "gemma4", "ling26", "nemotron3",
])
@pytest.mark.parametrize("topology_name", ["gb10_quarter", "h100_8x"])
def test_system_every_preset_yields_proposals_on_every_topology(
    preset_name, topology_name,
):
    specs = build_preset_specs(preset_name, hidden_size=4096)
    specs.append({"kind": "mlp", "name": "logits"})
    g = from_block_specs(specs, hidden_size=4096, instantiate=False)
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        dim_env=_ENV,
    )
    topology = {
        "gb10_quarter": gb10_quarter,
        "h100_8x": h100_8x,
    }[topology_name]()
    proposals = suggest_sharding(spec, topology)
    assert len(proposals) >= 1
    # At least one proposal should be a non-error candidate
    has_clean = any(p.num_errors == 0 for p in proposals)
    assert has_clean, (
        f"{preset_name!r} on {topology_name!r}: no clean proposal "
        f"({[p.strategy_name for p in proposals]})"
    )
