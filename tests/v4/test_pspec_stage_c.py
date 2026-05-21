"""PSpec Stage C tests — gotcha_checker table."""

from __future__ import annotations

from cppmega_v4.architectures import build_preset_specs
from cppmega_v4.buildspec import (
    ModelBuildSpec,
    adamw,
    cross_entropy_loss,
    sgd,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode
from cppmega_v4.parallelism import (
    AxisAssignment,
    GOTCHAS,
    Gotcha,
    GotchaSeverity,
    ParallelismKind,
    ShardingSpec,
    check_gotchas,
    fsdp2_only,
    fsdp2_plus_tp,
    h100_8x,
    megatron_ep_only,
    single_device,
    tpu_v6e_8,
)


_ENV = {
    "B": 1, "S": 4096, "H": 4096,
    "nh": 32, "nkv": 4, "head_dim": 128,
    "num_experts": 8, "top_k": 2,
}


def _qwen_spec(env_override: dict | None = None) -> ModelBuildSpec:
    specs = build_preset_specs("qwen3_next", hidden_size=4096)
    specs.append({"kind": "mlp", "name": "logits"})
    g = from_block_specs(specs, hidden_size=4096, instantiate=False)
    return ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        dim_env={**_ENV, **(env_override or {})},
    )


def _dense_spec(env_override: dict | None = None) -> ModelBuildSpec:
    """No-MoE graph for EP-without-MoE tests."""
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
        dim_env={**_ENV, **(env_override or {})},
    )


def _fired_ids(fired: tuple[Gotcha, ...]) -> set[str]:
    return {g.gotcha_id for g in fired}


# ---------------------------------------------------------------------------
# Table coverage + dataclass invariants
# ---------------------------------------------------------------------------


def test_every_gotcha_has_id_severity_message_reference():
    assert len(GOTCHAS) >= 10
    for g in GOTCHAS:
        assert g.gotcha_id
        assert isinstance(g.severity, GotchaSeverity)
        assert g.message
        assert g.reference
        assert callable(g.condition)


def test_gotcha_ids_are_unique():
    ids = [g.gotcha_id for g in GOTCHAS]
    assert len(ids) == len(set(ids))


def test_check_gotchas_returns_tuple_of_gotchas():
    spec = _qwen_spec()
    fired = check_gotchas(fsdp2_only(h100_8x()), spec)
    assert isinstance(fired, tuple)
    assert all(isinstance(g, Gotcha) for g in fired)


# ---------------------------------------------------------------------------
# Per-gotcha trigger tests
# ---------------------------------------------------------------------------


def test_fsdp2_whole_compile_fires_only_when_both_set():
    spec = _qwen_spec()
    bad = ShardingSpec(
        topology=h100_8x(),
        axis_assignments=(AxisAssignment("dp", ParallelismKind.FSDP2, 8),),
        compile_mode="whole_model",
    )
    good = ShardingSpec(
        topology=h100_8x(),
        axis_assignments=(AxisAssignment("dp", ParallelismKind.FSDP2, 8),),
        compile_mode="regional",
    )
    assert "fsdp2_whole_compile" in _fired_ids(check_gotchas(bad, spec))
    assert "fsdp2_whole_compile" not in _fired_ids(check_gotchas(good, spec))


def test_megatron_tp_whole_compile_fires():
    spec = _qwen_spec()
    bad = ShardingSpec(
        topology=h100_8x(dp=4, tp=2, ep=1, pp=1),
        axis_assignments=(
            AxisAssignment("dp", ParallelismKind.FSDP2, 4),
            AxisAssignment("tp", ParallelismKind.TP, 2),
        ),
        compile_mode="whole_model",
    )
    assert "megatron_tp_whole_compile" in _fired_ids(check_gotchas(bad, spec))


def test_megatron_tp_whole_compile_not_fired_when_regional():
    spec = _qwen_spec()
    good = fsdp2_plus_tp(h100_8x(dp=4, tp=2, ep=1, pp=1))
    assert "megatron_tp_whole_compile" not in _fired_ids(check_gotchas(good, spec))


def test_fp8_grad_duplication_fires_only_with_fp8():
    spec = _qwen_spec()
    fp8 = ShardingSpec(
        topology=h100_8x(),
        axis_assignments=(AxisAssignment("dp", ParallelismKind.FSDP2, 8),),
        fp8_enabled=True,
    )
    bf16 = fsdp2_only(h100_8x(), fp8_enabled=False)
    assert "fp8_grad_duplication" in _fired_ids(check_gotchas(fp8, spec))
    assert "fp8_grad_duplication" not in _fired_ids(check_gotchas(bf16, spec))


def test_master_fp32_duplication_fires():
    spec = _qwen_spec()
    master = ShardingSpec(
        topology=h100_8x(),
        axis_assignments=(AxisAssignment("dp", ParallelismKind.FSDP2, 8),),
        master_weights_fp32=True,
    )
    plain = fsdp2_only(h100_8x())
    assert "master_fp32_duplication" in _fired_ids(check_gotchas(master, spec))
    assert "master_fp32_duplication" not in _fired_ids(check_gotchas(plain, spec))


def test_ep_more_than_16_experts_xla_fires_on_tpu_only():
    spec = _qwen_spec({"num_experts": 32})
    tpu = ShardingSpec(
        topology=tpu_v6e_8(),
        axis_assignments=(AxisAssignment("dp", ParallelismKind.EP, 4),
                          AxisAssignment("tp", ParallelismKind.TP, 2)),
    )
    cuda = ShardingSpec(
        topology=h100_8x(dp=2, tp=1, ep=4, pp=1),
        axis_assignments=(AxisAssignment("ep", ParallelismKind.EP, 4),),
    )
    assert "ep_more_than_16_experts_xla" in _fired_ids(check_gotchas(tpu, spec))
    assert "ep_more_than_16_experts_xla" not in _fired_ids(check_gotchas(cuda, spec))


def test_ep_more_than_16_experts_xla_not_fired_when_few_experts():
    spec = _qwen_spec({"num_experts": 8})
    tpu = ShardingSpec(
        topology=tpu_v6e_8(),
        axis_assignments=(AxisAssignment("dp", ParallelismKind.EP, 4),
                          AxisAssignment("tp", ParallelismKind.TP, 2)),
    )
    assert "ep_more_than_16_experts_xla" not in _fired_ids(check_gotchas(tpu, spec))


def test_pp_comm_stream_broken_fires_when_pp_present():
    spec = _qwen_spec()
    pp = ShardingSpec(
        topology=h100_8x(dp=4, tp=1, ep=1, pp=2),
        axis_assignments=(
            AxisAssignment("dp", ParallelismKind.DP, 4),
            AxisAssignment("pp", ParallelismKind.PP, 2),
        ),
    )
    no_pp = fsdp2_only(h100_8x())
    assert "pp_comm_stream_broken" in _fired_ids(check_gotchas(pp, spec))
    assert "pp_comm_stream_broken" not in _fired_ids(check_gotchas(no_pp, spec))


def test_megatron_row_parallel_boundary_fires_with_tp():
    spec = _qwen_spec()
    tp = fsdp2_plus_tp(h100_8x(dp=4, tp=2, ep=1, pp=1))
    no_tp = fsdp2_only(h100_8x())
    assert "megatron_row_parallel_boundary" in _fired_ids(check_gotchas(tp, spec))
    assert "megatron_row_parallel_boundary" not in _fired_ids(check_gotchas(no_tp, spec))


def test_fsdp_allgather_peak_fires_with_fsdp():
    spec = _qwen_spec()
    fsdp = fsdp2_only(h100_8x())
    no_fsdp = single_device(h100_8x())
    assert "fsdp_allgather_peak_unsharded" in _fired_ids(check_gotchas(fsdp, spec))
    assert "fsdp_allgather_peak_unsharded" not in _fired_ids(check_gotchas(no_fsdp, spec))


def test_sp_replicated_param_allreduce_overhead_fires_with_sp():
    spec = _qwen_spec()
    with_sp = ShardingSpec(
        topology=h100_8x(dp=4, tp=2, ep=1, pp=1),
        axis_assignments=(
            AxisAssignment("dp", ParallelismKind.FSDP2, 4),
            AxisAssignment("tp", ParallelismKind.SP, 2),
        ),
    )
    no_sp = fsdp2_plus_tp(h100_8x(dp=4, tp=2, ep=1, pp=1))
    assert "sp_replicated_param_allreduce_overhead" in _fired_ids(check_gotchas(with_sp, spec))
    assert "sp_replicated_param_allreduce_overhead" not in _fired_ids(check_gotchas(no_sp, spec))


def test_tp_master_weights_double_state_fires_when_both():
    spec = _qwen_spec()
    bad = ShardingSpec(
        topology=h100_8x(dp=4, tp=2, ep=1, pp=1),
        axis_assignments=(
            AxisAssignment("dp", ParallelismKind.FSDP2, 4),
            AxisAssignment("tp", ParallelismKind.TP, 2),
        ),
        master_weights_fp32=True,
    )
    assert "tp_master_weights_double_state" in _fired_ids(check_gotchas(bad, spec))


def test_dp_no_optim_sharding_fires_for_dp_only_multi_device():
    spec = _qwen_spec()
    dp_only = single_device(h100_8x())   # uses ParallelismKind.DP
    assert "dp_no_optim_sharding" in _fired_ids(check_gotchas(dp_only, spec))
    # FSDP2 → not fired
    assert "dp_no_optim_sharding" not in _fired_ids(
        check_gotchas(fsdp2_only(h100_8x()), spec),
    )


def test_grad_reduce_fp32_warning_fires():
    spec = _qwen_spec()
    bad = ShardingSpec(
        topology=h100_8x(),
        axis_assignments=(AxisAssignment("dp", ParallelismKind.FSDP2, 8),),
        grad_reduce_dtype="fp32",
    )
    assert "grad_reduce_fp32_doubles_grad_buffer" in _fired_ids(
        check_gotchas(bad, spec)
    )


def test_ep_without_moe_fires():
    spec = _dense_spec()   # no MoE bricks
    ep = megatron_ep_only(h100_8x(dp=2, tp=1, ep=4, pp=1))
    assert "ep_without_moe" in _fired_ids(check_gotchas(ep, spec))


def test_ep_without_moe_NOT_fired_when_moe_present():
    spec = _qwen_spec()    # qwen3_next has moe brick
    ep = megatron_ep_only(h100_8x(dp=2, tp=1, ep=4, pp=1))
    assert "ep_without_moe" not in _fired_ids(check_gotchas(ep, spec))


def test_checkpointing_off_with_large_seq_fires():
    spec = _qwen_spec({"S": 4096})
    bad = fsdp2_only(h100_8x(), activation_checkpointing="off")
    good = fsdp2_only(h100_8x(), activation_checkpointing="full")
    assert "checkpointing_off_with_large_seq" in _fired_ids(check_gotchas(bad, spec))
    assert "checkpointing_off_with_large_seq" not in _fired_ids(check_gotchas(good, spec))


def test_checkpointing_off_NOT_fired_for_small_seq():
    spec = _qwen_spec({"S": 1024})
    short = fsdp2_only(h100_8x(), activation_checkpointing="off")
    assert "checkpointing_off_with_large_seq" not in _fired_ids(check_gotchas(short, spec))


def test_fp8_with_sgd_loses_precision_fires():
    g = BrickGraph(
        nodes=(BrickNode(kind="mlp", name="m"),),
    )
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=sgd(),
        dim_env=_ENV,
    )
    fp8_sgd = ShardingSpec(
        topology=h100_8x(),
        axis_assignments=(AxisAssignment("dp", ParallelismKind.FSDP2, 8),),
        fp8_enabled=True,
    )
    assert "fp8_with_sgd_loses_precision" in _fired_ids(check_gotchas(fp8_sgd, spec))


# ---------------------------------------------------------------------------
# Clean spec — no errors / no warnings
# ---------------------------------------------------------------------------


def test_clean_qwen_fsdp2_only_has_no_errors():
    """Default fsdp2_only on h100_8x with regional compile should NOT
    fire any ERROR gotcha."""
    spec = _qwen_spec()
    fired = check_gotchas(fsdp2_only(h100_8x()), spec)
    errors = [g for g in fired if g.severity is GotchaSeverity.ERROR]
    assert errors == []


def test_severity_distribution_is_balanced():
    """Sanity: we have a mix of severities across the table — not all
    ERROR or all INFO."""
    seen = {g.severity for g in GOTCHAS}
    assert {GotchaSeverity.ERROR, GotchaSeverity.WARNING,
            GotchaSeverity.INFO} <= seen


# ---------------------------------------------------------------------------
# Defensive: a misbehaving predicate doesn't crash the whole check
# ---------------------------------------------------------------------------


def test_check_gotchas_swallows_predicate_exceptions():
    """If a trigger predicate raises (e.g. missing dim_env key), the
    overall check should NOT propagate — that gotcha is treated as
    not-fired."""
    from cppmega_v4.parallelism import gotcha_checker as gc

    def boom(s, b):
        raise KeyError("missing 'B' from dim_env")

    gotchas = (
        Gotcha(
            gotcha_id="boom",
            severity=GotchaSeverity.ERROR,
            condition=boom,
            message="should never be visible",
            reference="test",
        ),
    )
    spec = _qwen_spec()
    fired = gc.check_gotchas(fsdp2_only(h100_8x()), spec, gotchas=gotchas)
    assert fired == ()
