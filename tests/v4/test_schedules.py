"""E7-9 tests: LR Schedules (cosine / linear_warmup / wsd / inv_sqrt /
   polynomial / constant) + integration with ParamGroup + stage_train."""

from __future__ import annotations

import math

import pytest

from cppmega_v4.buildspec.schedules import (
    SCHEDULE_BUILTINS,
    SCHEDULE_KINDS,
    ScheduleSpec,
    constant,
    cosine_annealing,
    inv_sqrt,
    linear_warmup_then_constant,
    polynomial,
    wsd,
)
from cppmega_v4.buildspec.optim_spec import ParamGroup, adamw


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_schedule_kinds_has_six_entries():
    assert set(SCHEDULE_KINDS) == {
        "constant", "cosine", "linear_warmup", "wsd",
        "inv_sqrt", "polynomial",
    }


def test_schedule_builtins_registers_all_six():
    for k in SCHEDULE_KINDS:
        assert k in SCHEDULE_BUILTINS


# ---------------------------------------------------------------------------
# constant + linear_warmup
# ---------------------------------------------------------------------------


def test_constant_returns_base_lr_every_step():
    s = constant().build(1e-3)
    for step in (0, 1, 100, 100_000):
        assert s(step) == 1e-3


def test_linear_warmup_ramps_from_zero_then_holds():
    s = linear_warmup_then_constant(warmup_steps=10).build(1e-3)
    assert s(0) == 0.0
    assert s(5) == pytest.approx(0.5e-3)
    assert s(10) == 1e-3
    assert s(100) == 1e-3


def test_linear_warmup_zero_raises():
    with pytest.raises(ValueError, match="warmup_steps"):
        linear_warmup_then_constant(warmup_steps=0)


# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------


def test_cosine_warmup_then_decay():
    s = cosine_annealing(total_steps=100, min_lr_ratio=0.1,
                         warmup_steps=10).build(1e-3)
    assert s(0) == 0.0
    assert s(5) == pytest.approx(0.5e-3)
    assert s(10) == pytest.approx(1e-3)
    # Mid-decay: between base and floor
    mid = s(55)
    assert 0.1e-3 < mid < 1e-3
    # End of decay: floor
    end = s(99)
    assert end == pytest.approx(1e-3 * 0.1, abs=1e-5)


def test_cosine_requires_total_steps():
    with pytest.raises(ValueError, match="requires total_steps"):
        ScheduleSpec(kind="cosine")


def test_cosine_monotonic_decay_after_warmup():
    s = cosine_annealing(total_steps=50, min_lr_ratio=0.05).build(1e-3)
    prev = s(0)
    for step in range(1, 50):
        cur = s(step)
        assert cur <= prev + 1e-9, f"step {step}: {cur} > {prev}"
        prev = cur


# ---------------------------------------------------------------------------
# wsd
# ---------------------------------------------------------------------------


def test_wsd_three_phases():
    s = wsd(warmup_steps=10, decay_steps=20, total_steps=100,
            min_lr_ratio=0.1).build(1e-3)
    # warmup: ramping up
    assert s(0) == 0.0
    assert s(10) == pytest.approx(1e-3)
    # steady: at base_lr
    assert s(50) == 1e-3
    assert s(79) == 1e-3  # last steady step (steady_end = 100-20 = 80)
    # decay: linear from base_lr at step 80 down to floor at step 100.
    # Progress(t) = (t - 80) / 20 → at t=80 progress=0, at t=100 progress=1.
    assert s(80) == pytest.approx(1e-3, abs=1e-5)
    # At t=99: progress=0.95 → lr = base + (floor-base)*0.95 = 1.45e-4
    assert s(99) == pytest.approx(1e-3 + (1e-4 - 1e-3) * 0.95, abs=1e-6)
    # At t=100: progress=1.0 → exactly floor
    assert s(100) == pytest.approx(1e-4, abs=1e-6)


def test_wsd_requires_decay_steps():
    with pytest.raises(ValueError, match="decay_steps"):
        ScheduleSpec(kind="wsd", warmup_steps=10, total_steps=100)


def test_wsd_warmup_plus_decay_must_fit():
    with pytest.raises(ValueError, match="warmup_steps.*decay_steps"):
        wsd(warmup_steps=60, decay_steps=60, total_steps=100)


# ---------------------------------------------------------------------------
# inv_sqrt
# ---------------------------------------------------------------------------


def test_inv_sqrt_warmup_then_decay():
    s = inv_sqrt(warmup_steps=100).build(1e-3)
    assert s(0) == 0.0
    assert s(100) == pytest.approx(1e-3)
    # After warmup: decays as 1/sqrt(step)
    expected_400 = 1e-3 * math.sqrt(100 / 400)
    assert s(400) == pytest.approx(expected_400)


def test_inv_sqrt_zero_warmup_raises():
    with pytest.raises(ValueError, match="warmup_steps"):
        inv_sqrt(warmup_steps=0)


# ---------------------------------------------------------------------------
# polynomial
# ---------------------------------------------------------------------------


def test_polynomial_decays_to_floor():
    s = polynomial(total_steps=100, power=2.0, min_lr_ratio=0.1).build(1e-3)
    assert s(0) == pytest.approx(1e-3)
    assert s(50) == pytest.approx(1e-3 * (0.1 + 0.9 * 0.25), abs=1e-5)
    assert s(100) == pytest.approx(1e-4, abs=1e-5)


def test_polynomial_zero_power_rejected():
    with pytest.raises(ValueError, match="power"):
        polynomial(total_steps=10, power=0)


# ---------------------------------------------------------------------------
# ScheduleSpec validation
# ---------------------------------------------------------------------------


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="kind="):
        ScheduleSpec(kind="invalid_kind")  # type: ignore[arg-type]


def test_negative_warmup_rejected():
    with pytest.raises(ValueError, match="warmup_steps"):
        ScheduleSpec(kind="constant", warmup_steps=-1)


def test_min_lr_ratio_out_of_range_rejected():
    with pytest.raises(ValueError, match="min_lr_ratio"):
        ScheduleSpec(kind="cosine", total_steps=10, min_lr_ratio=1.5)


def test_total_steps_less_than_warmup_rejected():
    with pytest.raises(ValueError, match="total_steps"):
        ScheduleSpec(kind="cosine", warmup_steps=100, total_steps=50)


def test_build_rejects_non_positive_base_lr():
    s = constant()
    with pytest.raises(ValueError, match="base_lr"):
        s.build(0)


# ---------------------------------------------------------------------------
# sample() — used by UI sparkline
# ---------------------------------------------------------------------------


def test_sample_returns_n_points():
    s = cosine_annealing(total_steps=100)
    pts = s.sample(1e-3, n_points=20)
    assert len(pts) == 20
    assert all(p >= 0 for p in pts)


def test_sample_constant_returns_flat_curve():
    s = constant()
    pts = s.sample(1e-3, n_points=10)
    assert all(p == 1e-3 for p in pts)


# ---------------------------------------------------------------------------
# ParamGroup integration
# ---------------------------------------------------------------------------


def test_param_group_accepts_schedule():
    g = ParamGroup(matcher="all", lr=1e-3, betas=(0.9, 0.95),
                   schedule=cosine_annealing(total_steps=100))
    fn = g.effective_lr_callable()
    assert fn(0) == 0.0 or fn(0) == pytest.approx(1e-3)  # no warmup
    assert fn(100) <= 1e-3


def test_param_group_without_schedule_constant_lr():
    g = ParamGroup(matcher="all", lr=2e-4, betas=(0.9, 0.95))
    fn = g.effective_lr_callable()
    for step in (0, 1, 1000):
        assert fn(step) == 2e-4


def test_param_group_rejects_non_schedulespec():
    with pytest.raises(TypeError, match="schedule must be ScheduleSpec"):
        ParamGroup(matcher="all", lr=1e-3, betas=(0.9, 0.95),
                   schedule="cosine")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Smoke: AdamW factory still works after ParamGroup extension
# ---------------------------------------------------------------------------


def test_adamw_factory_after_schedule_extension():
    spec = adamw(lr=3e-4)
    assert spec.groups[0].schedule is None
    fn = spec.groups[0].effective_lr_callable()
    assert fn(0) == 3e-4
    assert fn(99) == 3e-4


# ---------------------------------------------------------------------------
# stage_train integration: schedule actually drives the optimizer
# ---------------------------------------------------------------------------


def test_stage_train_honours_schedule_via_lr_trajectory():
    """stage_train must (a) sweep the schedule per step, (b) report
    lr_trajectory in extras, (c) report schedule_kind. End-to-end."""
    from cppmega_v4.runner import Pipeline, run_pipeline
    from cppmega_v4.jsonrpc.schema import VerifyParams
    from cppmega_v4.architectures import build_preset_specs

    specs = build_preset_specs("llama3_8b", hidden_size=128)
    graph = {
        "nodes": [
            {"id": s.get("name"), "kind": s["kind"],
             "params": s.get("params", {})}
            for s in specs
        ],
        "edges": [
            {"src": specs[i].get("name"), "dst": specs[i + 1].get("name")}
            for i in range(len(specs) - 1)
        ],
    }
    spec = VerifyParams.model_validate({
        "graph": graph,
        "dim_env": {"B": 1, "S": 8, "H": 128, "nh": 2, "nkv": 1,
                    "head_dim": 64, "num_experts": 4, "top_k": 2},
        "loss": {"kind": "cross_entropy",
                 "head_outputs": [specs[-1].get("name")]},
        "optim": {
            "kind": "adamw",
            "groups": [
                {
                    "matcher": "all",
                    "lr": 1e-3,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.95],
                    "schedule": {
                        "kind": "linear_warmup",
                        "warmup_steps": 4,
                    },
                },
            ],
        },
    })
    r = run_pipeline(spec, Pipeline.from_dict({"stages": [
        "parse", "verify_build_spec", "resolve_shapes",
        "build_model", "train",
    ]}))
    train = next(s for s in r.stages if s.name == "train")
    assert train.status == "ok", train.error
    extras = train.extras
    assert extras is not None
    assert extras["schedule_kind"] == "linear_warmup"
    trajectory = extras["lr_trajectory"]
    # default num_steps=2; with warmup=4, lr at step 0 = 0, step 1 = 0.25e-3
    assert len(trajectory) == 2
    assert trajectory[0] == 0.0
    assert trajectory[1] == pytest.approx(0.25e-3, abs=1e-6)


def test_stage_train_constant_when_no_schedule():
    """With no schedule attached, lr_trajectory must be flat and the
    reported schedule_kind must say constant."""
    from cppmega_v4.runner import Pipeline, run_pipeline
    from cppmega_v4.jsonrpc.schema import VerifyParams
    from cppmega_v4.architectures import build_preset_specs

    specs = build_preset_specs("llama3_8b", hidden_size=128)
    graph = {
        "nodes": [
            {"id": s.get("name"), "kind": s["kind"],
             "params": s.get("params", {})}
            for s in specs
        ],
        "edges": [
            {"src": specs[i].get("name"), "dst": specs[i + 1].get("name")}
            for i in range(len(specs) - 1)
        ],
    }
    spec = VerifyParams.model_validate({
        "graph": graph,
        "dim_env": {"B": 1, "S": 8, "H": 128, "nh": 2, "nkv": 1,
                    "head_dim": 64, "num_experts": 4, "top_k": 2},
        "loss": {"kind": "cross_entropy",
                 "head_outputs": [specs[-1].get("name")]},
        "optim": {
            "kind": "adamw",
            "groups": [{"matcher": "all", "lr": 5e-4,
                        "weight_decay": 0.01, "betas": [0.9, 0.95]}],
        },
    })
    r = run_pipeline(spec, Pipeline.from_dict({"stages": [
        "parse", "verify_build_spec", "build_model", "train",
    ]}))
    train = next(s for s in r.stages if s.name == "train")
    assert train.status == "ok", train.error
    extras = train.extras
    assert extras["schedule_kind"] == "constant"
    trajectory = extras["lr_trajectory"]
    assert all(lr == 5e-4 for lr in trajectory), trajectory
