from __future__ import annotations

import json
from pathlib import Path

import pytest

from cppmega_mlx.runtime.memory import (
    DEFAULT_METAL_RATIO,
    DEFAULT_PEAK_MEMORY_RATIO,
    DEFAULT_WIRED_RATIO,
    AppliedMemoryLimits,
    ClearCacheEvent,
    MemoryLimitApiStatus,
    MemoryLimitPlan,
    RuntimeStackEvidence,
    apply_memory_limit_plan,
    device_total_memory_bytes,
    maybe_clear_cache_after_step,
    memory_limit_api_status,
    memory_limit_plan,
    memory_limit_receipt,
    peak_memory_threshold_bytes,
    runtime_stack_evidence,
    should_clear_cache_after_step,
)


DEV_128_TOTAL_BYTES = 137_438_953_472
M06_RECEIPT_PATH = Path("bench/baselines/m06_memory.json")


def test_memory_limit_plan_uses_documented_default_ratios() -> None:
    plan = memory_limit_plan(1000)

    assert plan == MemoryLimitPlan(
        total_bytes=1000,
        wired_ratio=DEFAULT_WIRED_RATIO,
        metal_ratio=DEFAULT_METAL_RATIO,
        wired_limit_bytes=700,
        metal_limit_bytes=850,
    )
    assert plan.to_dict() == {
        "total_bytes": 1000,
        "wired_ratio": 0.7,
        "metal_ratio": 0.85,
        "wired_limit_bytes": 700,
        "metal_limit_bytes": 850,
    }


def test_memory_limit_plan_accepts_custom_ratios() -> None:
    plan = memory_limit_plan(
        1024,
        wired_ratio=0.5,
        metal_ratio=0.75,
    )

    assert plan.wired_limit_bytes == 512
    assert plan.metal_limit_bytes == 768


def test_memory_limit_plan_matches_documented_dev_128_math() -> None:
    plan = memory_limit_plan(DEV_128_TOTAL_BYTES)

    assert plan.wired_limit_bytes == 96_207_267_430
    assert plan.metal_limit_bytes == 116_823_110_451
    assert peak_memory_threshold_bytes(DEV_128_TOTAL_BYTES) == 103_079_215_104


@pytest.mark.parametrize("total_bytes", [0, -1])
def test_memory_limit_plan_rejects_non_positive_total(total_bytes: int) -> None:
    with pytest.raises(ValueError, match="total_bytes must be positive"):
        memory_limit_plan(total_bytes)


def test_memory_limit_plan_requires_integer_total_bytes() -> None:
    with pytest.raises(TypeError, match="total_bytes must be an integer"):
        memory_limit_plan(1024.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("wired_ratio", {"wired_ratio": 0.0}),
        ("wired_ratio", {"wired_ratio": 1.0}),
        ("metal_ratio", {"metal_ratio": -0.1}),
        ("metal_ratio", {"metal_ratio": 1.2}),
    ],
)
def test_memory_limit_plan_validates_ratio_bounds(
    name: str,
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be > 0 and < 1"):
        memory_limit_plan(1024, **kwargs)


def test_memory_limit_plan_requires_numeric_ratios() -> None:
    with pytest.raises(TypeError, match="wired_ratio must be a numeric ratio"):
        memory_limit_plan(1024, wired_ratio="0.7")  # type: ignore[arg-type]


class _FakeMetal:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def set_memory_limit(self, limit: int) -> int:
        self.calls.append(limit)
        return 456


class _FakeMLX:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.metal = _FakeMetal()
        self.clear_cache_calls = 0
        self.synchronize_calls = 0
        self.cache_memory = 64

    def set_wired_limit(self, limit: int) -> int:
        self.calls.append(limit)
        return 123

    def device_info(self) -> dict[str, int]:
        return {"memory_size": 4096}

    def default_device(self) -> str:
        return "Device(gpu, 0)"

    def get_cache_memory(self) -> int:
        return self.cache_memory

    def clear_cache(self) -> None:
        self.clear_cache_calls += 1
        self.cache_memory = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _FakeMLXWithRootMemoryLimit(_FakeMLX):
    def __init__(self) -> None:
        super().__init__()
        self.root_memory_calls: list[int] = []

    def set_memory_limit(self, limit: int) -> int:
        self.root_memory_calls.append(limit)
        return 789


class _FakeMLXWithRootMemoryOnly(_FakeMLXWithRootMemoryLimit):
    metal = None

    def __init__(self) -> None:
        super().__init__()
        self.metal = None


class _FakeMLXMissingMemorySetter(_FakeMLX):
    metal = None

    def __init__(self) -> None:
        super().__init__()
        self.metal = None


def test_apply_memory_limit_plan_is_dry_run_by_default() -> None:
    fake = _FakeMLX()
    plan = memory_limit_plan(1000)

    result = apply_memory_limit_plan(plan, mx_module=fake)

    assert result == AppliedMemoryLimits(plan=plan, applied=False)
    assert fake.calls == []
    assert fake.metal.calls == []
    assert result.to_dict()["applied"] is False


def test_apply_memory_limit_plan_sets_both_limits_when_explicit() -> None:
    fake = _FakeMLX()
    plan = memory_limit_plan(1000)

    result = apply_memory_limit_plan(plan, mx_module=fake, apply=True)

    assert result.applied is True
    assert result.previous_wired_limit_bytes == 123
    assert result.previous_metal_limit_bytes == 456
    assert result.metal_limit_api_path == "mx.metal.set_memory_limit"
    assert fake.calls == [700]
    assert fake.metal.calls == [850]


def test_apply_memory_limit_plan_prefers_metal_memory_limit_api() -> None:
    fake = _FakeMLXWithRootMemoryLimit()
    plan = memory_limit_plan(1000)

    result = apply_memory_limit_plan(plan, mx_module=fake, apply=True)

    assert result.previous_wired_limit_bytes == 123
    assert result.previous_metal_limit_bytes == 456
    assert result.metal_limit_api_path == "mx.metal.set_memory_limit"
    assert fake.calls == [700]
    assert fake.root_memory_calls == []
    assert fake.metal.calls == [850]


def test_apply_memory_limit_plan_rejects_missing_apis() -> None:
    plan = memory_limit_plan(1000)

    with pytest.raises(RuntimeError, match="set_wired_limit is unavailable"):
        apply_memory_limit_plan(plan, mx_module=object(), apply=True)


def test_apply_memory_limit_plan_uses_root_memory_limit_when_metal_path_is_missing() -> (
    None
):
    fake = _FakeMLXWithRootMemoryOnly()
    plan = memory_limit_plan(1000)

    result = apply_memory_limit_plan(plan, mx_module=fake, apply=True)

    assert result.previous_wired_limit_bytes == 123
    assert result.previous_metal_limit_bytes == 789
    assert result.metal_limit_api_path == "mx.set_memory_limit"
    assert fake.calls == [700]
    assert fake.root_memory_calls == [850]


def test_apply_memory_limit_plan_checks_memory_limit_api_before_wired_mutation() -> (
    None
):
    fake = _FakeMLXMissingMemorySetter()
    plan = memory_limit_plan(1000)

    with pytest.raises(RuntimeError, match="no supported MLX memory limit setter"):
        apply_memory_limit_plan(plan, mx_module=fake, apply=True)

    assert fake.calls == []


def test_memory_limit_api_status_reports_root_and_compat_paths() -> None:
    status = memory_limit_api_status(_FakeMLXWithRootMemoryLimit())

    assert status == MemoryLimitApiStatus(
        wired_limit_available=True,
        root_memory_limit_available=True,
        metal_memory_limit_available=True,
        preferred_memory_limit_api_path="mx.metal.set_memory_limit",
        supported_memory_limit_api_paths=(
            "mx.metal.set_memory_limit",
            "mx.set_memory_limit",
        ),
    )
    assert status.to_dict()["supported_memory_limit_api_paths"] == [
        "mx.metal.set_memory_limit",
        "mx.set_memory_limit",
    ]


def test_runtime_stack_evidence_records_mlx_metal_and_device_stack() -> None:
    stack = runtime_stack_evidence(_FakeMLX())

    assert stack.default_device == "Device(gpu, 0)"
    assert stack.device_info == {"memory_size": 4096}
    assert stack.to_dict()["device_info"] == {"memory_size": 4096}
    assert "platform_system" in stack.to_dict()


def test_device_total_memory_bytes_reads_fake_mlx_device_info() -> None:
    assert device_total_memory_bytes(_FakeMLX()) == 4096


def test_device_total_memory_bytes_handles_missing_device_info() -> None:
    assert device_total_memory_bytes(object()) is None


@pytest.mark.parametrize(
    ("step", "every_steps", "expected"),
    [
        (1, None, False),
        (1, 4, False),
        (4, 4, True),
        (8, 4, True),
    ],
)
def test_should_clear_cache_after_step_matches_cadence(
    step: int,
    every_steps: int | None,
    expected: bool,
) -> None:
    assert should_clear_cache_after_step(step, every_steps) is expected


@pytest.mark.parametrize("step", [0, -1])
def test_should_clear_cache_after_step_rejects_invalid_step(step: int) -> None:
    with pytest.raises(ValueError, match="step must be positive"):
        should_clear_cache_after_step(step, 1)


@pytest.mark.parametrize("every_steps", [0, -1])
def test_should_clear_cache_after_step_rejects_invalid_cadence(
    every_steps: int,
) -> None:
    with pytest.raises(ValueError, match="every_steps must be positive"):
        should_clear_cache_after_step(1, every_steps)


def test_maybe_clear_cache_after_step_skips_non_cadence_step() -> None:
    fake = _FakeMLX()

    event = maybe_clear_cache_after_step(1, 4, mx_module=fake)

    assert event is None
    assert fake.clear_cache_calls == 0
    assert fake.synchronize_calls == 0
    assert fake.cache_memory == 64


def test_maybe_clear_cache_after_step_records_cadence_event() -> None:
    fake = _FakeMLX()

    event = maybe_clear_cache_after_step(4, 4, mx_module=fake)

    assert event is not None
    assert event == ClearCacheEvent(
        step=4,
        every_steps=4,
        api_path="mx.clear_cache",
        cache_memory_before_bytes=64,
        cache_memory_after_bytes=0,
    )
    assert event.to_dict() == {
        "step": 4,
        "every_steps": 4,
        "api_path": "mx.clear_cache",
        "cache_memory_before_bytes": 64,
        "cache_memory_after_bytes": 0,
    }
    assert fake.clear_cache_calls == 1
    assert fake.synchronize_calls == 1


def test_maybe_clear_cache_after_step_fails_when_api_missing() -> None:
    with pytest.raises(RuntimeError, match="clear_cache is unavailable"):
        maybe_clear_cache_after_step(4, 4, mx_module=object())


def test_memory_limit_receipt_records_partial_m06_gate() -> None:
    plan = memory_limit_plan(DEV_128_TOTAL_BYTES)
    applied = AppliedMemoryLimits(
        plan=plan,
        applied=True,
        previous_wired_limit_bytes=0,
        previous_metal_limit_bytes=0,
        metal_limit_api_path="mx.set_memory_limit",
    )

    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=1,
        clear_cache_every_steps=None,
        model_profile="HybridTinyLM",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=False,
        plan=plan,
        applied_limits=applied,
        api_status=memory_limit_api_status(_FakeMLXWithRootMemoryLimit()),
        runtime_stack=runtime_stack_for_receipt(),
        source="bench/baselines/m04_train_step.json",
        blockers=({"id": "cppmega-mlx-t8f.4", "status": "open"},),
        notes=("tiny HybridTinyLM receipt only",),
    )

    assert receipt["receipt_scope"] == "local_mlx_m06_memory"
    assert receipt["status"] == "partial"
    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["peak_threshold_ratio"] == DEFAULT_PEAK_MEMORY_RATIO
    assert receipt["peak_threshold_bytes"] == 103_079_215_104
    assert receipt["peak_memory_below_threshold"] is True
    assert receipt["measured_steps_meet_gate"] is False
    assert receipt["model_profile_meets_gate"] is False
    assert receipt["optimizer_meets_gate"] is True
    assert receipt["grad_checkpoint_meets_gate"] is False
    assert receipt["run_command_recorded"] is False
    assert receipt["run_command_meets_gate"] is False
    assert receipt["runtime_stack_recorded"] is True
    assert receipt["runtime_stack_meets_gate"] is True
    assert receipt["runtime_stack"] == runtime_stack_for_receipt().to_dict()
    assert receipt["clear_cache_cadence_recorded"] is False
    assert receipt["clear_cache_event_count"] is None
    assert receipt["clear_cache_events_meet_gate"] is False
    assert receipt["memory_limits_applied_meet_gate"] is True
    assert receipt["memory_limit_ratios_meet_gate"] is True
    assert receipt["required_wired_ratio"] == DEFAULT_WIRED_RATIO
    assert receipt["required_metal_ratio"] == DEFAULT_METAL_RATIO
    assert receipt["memory_limit_plan"] == plan.to_dict()
    assert receipt["applied_memory_limits"] == applied.to_dict()
    assert receipt["memory_profile_source"] == "bench/baselines/m04_train_step.json"
    assert receipt["blockers"] == [{"id": "cppmega-mlx-t8f.4", "status": "open"}]


@pytest.mark.parametrize("peak_memory_bytes", [None, 103_079_215_104])
def test_memory_limit_receipt_does_not_accept_unknown_or_over_threshold_peak(
    peak_memory_bytes: int | None,
) -> None:
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=peak_memory_bytes,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=10,
        clear_cache_events=clear_cache_events_for_receipt(),
        model_profile="local_gb10_quarter",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=True,
        applied_limits=applied_limits_for_receipt(),
        api_status=memory_limit_api_status(_FakeMLX()),
        runtime_stack=runtime_stack_for_receipt(),
        full_acceptance_claim=True,
    )

    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["status"] == "partial"


def test_memory_limit_receipt_rejects_mismatched_plan_total() -> None:
    with pytest.raises(ValueError, match="plan.total_bytes must match total_bytes"):
        memory_limit_receipt(
            total_bytes=DEV_128_TOTAL_BYTES,
            peak_memory_bytes=151_213_010,
            measured_steps=100,
            clear_cache_every_steps=10,
            plan=memory_limit_plan(1024),
        )


def test_memory_limit_receipt_rejects_mismatched_applied_limit_plan() -> None:
    plan = memory_limit_plan(DEV_128_TOTAL_BYTES)
    applied_limits = AppliedMemoryLimits(
        plan=memory_limit_plan(1024),
        applied=True,
        previous_wired_limit_bytes=0,
        previous_metal_limit_bytes=0,
        metal_limit_api_path="mx.metal.set_memory_limit",
    )

    with pytest.raises(ValueError, match="applied_limits.plan must match plan"):
        memory_limit_receipt(
            total_bytes=DEV_128_TOTAL_BYTES,
            peak_memory_bytes=151_213_010,
            measured_steps=100,
            clear_cache_every_steps=10,
            plan=plan,
            applied_limits=applied_limits,
        )


def test_memory_limit_receipt_rejects_mismatched_clear_cache_event_count() -> None:
    with pytest.raises(
        ValueError,
        match="clear_cache_event_count must match clear_cache_events length",
    ):
        memory_limit_receipt(
            total_bytes=DEV_128_TOTAL_BYTES,
            peak_memory_bytes=151_213_010,
            measured_steps=100,
            clear_cache_every_steps=10,
            clear_cache_event_count=9,
            clear_cache_events=clear_cache_events_for_receipt(),
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        (
            {"peak_memory_bytes": -1},
            "peak_memory_bytes must be non-negative when provided",
        ),
        ({"measured_steps": -1}, "measured_steps must be non-negative"),
        (
            {"run_command": ("python", 1)},
            "run_command entries must be strings",
        ),
    ],
)
def test_memory_limit_receipt_rejects_invalid_m06_metrics(
    kwargs: dict[str, object],
    error: str,
) -> None:
    receipt_kwargs = {
        "total_bytes": DEV_128_TOTAL_BYTES,
        "peak_memory_bytes": 151_213_010,
        "measured_steps": 100,
        "clear_cache_every_steps": 10,
        **kwargs,
    }

    with pytest.raises((TypeError, ValueError), match=error):
        memory_limit_receipt(**receipt_kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    (
        "model_profile",
        "optimizer_name",
        "grad_checkpoint_enabled",
        "expected_failed_gate",
    ),
    [
        ("HybridTinyLM", "AdamW", True, "model_profile_meets_gate"),
        ("local_gb10_quarter", "Lion", True, "optimizer_meets_gate"),
        ("local_gb10_quarter", "AdamW", False, "grad_checkpoint_meets_gate"),
    ],
)
def test_memory_limit_receipt_requires_full_run_identity_for_acceptance(
    model_profile: str,
    optimizer_name: str,
    grad_checkpoint_enabled: bool,
    expected_failed_gate: str,
) -> None:
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=10,
        clear_cache_events=clear_cache_events_for_receipt(),
        model_profile=model_profile,
        optimizer_name=optimizer_name,
        grad_checkpoint_enabled=grad_checkpoint_enabled,
        applied_limits=applied_limits_for_receipt(),
        api_status=memory_limit_api_status(_FakeMLX()),
        runtime_stack=runtime_stack_for_receipt(),
        full_acceptance_claim=True,
    )

    assert receipt[expected_failed_gate] is False
    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["status"] == "partial"


def test_memory_limit_receipt_keeps_hybrid_tiny_100_step_smoke_partial() -> None:
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_278,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=10,
        clear_cache_events=clear_cache_events_for_receipt(),
        model_profile="HybridTinyLM",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=False,
        applied_limits=applied_limits_for_receipt(),
        api_status=memory_limit_api_status(_FakeMLX()),
        full_acceptance_claim=True,
        source="/tmp/cppmega_m04_m06_memory_receipt.json",
        blockers=(
            {
                "id": "cppmega-mlx-t8f.6",
                "status": "open",
                "impact": "No local_gb10_quarter AdamW + grad-checkpoint receipt exists yet.",
            },
        ),
    )

    assert receipt["measured_steps_meet_gate"] is True
    assert receipt["clear_cache_cadence_recorded"] is True
    assert receipt["clear_cache_event_count_meets_gate"] is True
    assert receipt["clear_cache_event_sequence_meets_gate"] is True
    assert receipt["clear_cache_events_meet_gate"] is True
    assert receipt["memory_limits_applied_meet_gate"] is True
    assert receipt["memory_limit_ratios_meet_gate"] is True
    assert receipt["peak_memory_below_threshold"] is True
    assert receipt["model_profile_meets_gate"] is False
    assert receipt["grad_checkpoint_meets_gate"] is False
    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["status"] == "partial"


def test_checked_in_m06_receipt_remains_partial_until_exact_gate_runs() -> None:
    payload = json.loads(M06_RECEIPT_PATH.read_text(encoding="utf-8"))

    assert payload["receipt_scope"] == "local_mlx_m06_memory"
    assert payload["status"] == "partial"
    assert payload["full_m0_6_acceptance_claim"] is False
    assert payload["required_model_profile"] == "local_gb10_quarter"
    assert payload["model_profile"] == "HybridTinyLM"
    assert payload["model_profile_meets_gate"] is False
    assert payload["required_optimizer_name"] == "AdamW"
    assert payload["optimizer_name"] == "AdamW"
    assert payload["optimizer_meets_gate"] is True
    assert payload["grad_checkpoint_enabled"] is False
    assert payload["grad_checkpoint_meets_gate"] is False
    assert payload["run_command_recorded"] is False
    assert payload["run_command_meets_gate"] is False
    assert payload["run_command"] is None
    assert payload["measured_profile"]["profile_claim_scope"] == (
        "m04_target_parquet_hybrid_tiny_memory_smoke_only"
    )
    assert payload["measured_profile"]["steps_completed"] == payload["measured_steps"]
    assert (
        payload["measured_profile"]["peak_memory_bytes"]
        == payload["peak_memory_bytes"]
    )
    assert (
        payload["runtime_stack"]["device_info"]["memory_size"]
        == payload["total_memory_bytes"]
    )
    assert payload["required_wired_ratio"] == DEFAULT_WIRED_RATIO
    assert payload["required_metal_ratio"] == DEFAULT_METAL_RATIO
    assert payload["memory_limit_ratios_meet_gate"] is True
    assert payload["memory_limits_applied_meet_gate"] is True
    assert payload["peak_memory_below_threshold"] is True
    blockers = {blocker["id"]: blocker for blocker in payload["blockers"]}
    assert blockers["cppmega-mlx-t8f.6"]["status"] == "open"
    assert "No 100-step local_gb10_quarter" in blockers["cppmega-mlx-t8f.6"]["impact"]
    assert any("not full M0.6 acceptance" in note for note in payload["notes"])
    assert any("No M4-vs-GB10 parity" in note for note in payload["notes"])

    expected_events = payload["measured_steps"] // payload["clear_cache_every_steps"]
    assert payload["expected_clear_cache_event_count"] == expected_events
    assert payload["expected_clear_cache_steps"] == [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    ]
    assert payload["clear_cache_event_count"] == expected_events
    assert payload["clear_cache_event_steps"] == [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    ]
    assert payload["clear_cache_event_api_paths"] == ["mx.clear_cache"] * 10
    assert payload["clear_cache_event_cadences"] == [10] * 10
    assert payload["clear_cache_event_count_meets_gate"] is True
    assert payload["clear_cache_event_sequence_meets_gate"] is True
    assert payload["clear_cache_events_meet_gate"] is True
    assert [event["step"] for event in payload["clear_cache_events"]] == [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    ]
    assert {event["api_path"] for event in payload["clear_cache_events"]} == {
        "mx.clear_cache"
    }


def test_memory_limit_receipt_accepts_only_complete_m06_gate() -> None:
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=10,
        clear_cache_events=clear_cache_events_for_receipt(),
        model_profile="local_gb10_quarter",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=True,
        applied_limits=applied_limits_for_receipt(),
        api_status=memory_limit_api_status(_FakeMLX()),
        runtime_stack=runtime_stack_for_receipt(),
        run_command=local_gb10_quarter_run_command(),
        full_acceptance_claim=True,
    )

    assert receipt["model_profile_meets_gate"] is True
    assert receipt["optimizer_meets_gate"] is True
    assert receipt["grad_checkpoint_meets_gate"] is True
    assert receipt["run_command_meets_gate"] is True
    assert receipt["runtime_stack_meets_gate"] is True
    assert receipt["clear_cache_events_meet_gate"] is True
    assert receipt["memory_limits_applied_meet_gate"] is True
    assert receipt["memory_limit_ratios_meet_gate"] is True
    assert receipt["full_m0_6_acceptance_claim"] is True
    assert receipt["status"] == "accepted"


def test_memory_limit_receipt_requires_runtime_memory_size_to_match_total() -> None:
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=10,
        clear_cache_events=clear_cache_events_for_receipt(),
        model_profile="local_gb10_quarter",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=True,
        applied_limits=applied_limits_for_receipt(),
        api_status=memory_limit_api_status(_FakeMLX()),
        runtime_stack=RuntimeStackEvidence(
            mlx_version="0.31.1",
            mlx_metal_version="0.31.1",
            platform_system="Darwin",
            platform_release="25.4.0",
            macos_version="26.4.1",
            machine="arm64",
            default_device="Device(gpu, 0)",
            metal_available=True,
            device_info={
                "device_name": "Apple M4 Max",
                "memory_size": DEV_128_TOTAL_BYTES - 1,
            },
        ),
        run_command=local_gb10_quarter_run_command(),
        full_acceptance_claim=True,
    )

    assert receipt["runtime_stack_meets_gate"] is False
    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["status"] == "partial"


def test_memory_limit_receipt_accepts_split_argv_run_command_tokens() -> None:
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=10,
        clear_cache_events=clear_cache_events_for_receipt(),
        model_profile="local_gb10_quarter",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=True,
        applied_limits=applied_limits_for_receipt(),
        api_status=memory_limit_api_status(_FakeMLX()),
        runtime_stack=runtime_stack_for_receipt(),
        run_command=(
            "./.venv/bin/python",
            "scripts/m04_train_step.py",
            "--model-profile",
            "local_gb10_quarter",
            "--grad-checkpoint",
            "--apply-memory-limit-plan",
            "--clear-cache-every-steps",
            "10",
        ),
        full_acceptance_claim=True,
    )

    assert receipt["run_command_meets_gate"] is True
    assert receipt["full_m0_6_acceptance_claim"] is True
    assert receipt["status"] == "accepted"


def test_memory_limit_receipt_requires_documented_ratios_for_acceptance() -> None:
    plan = memory_limit_plan(
        DEV_128_TOTAL_BYTES,
        wired_ratio=0.60,
        metal_ratio=0.80,
    )
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=10,
        clear_cache_events=clear_cache_events_for_receipt(),
        model_profile="local_gb10_quarter",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=True,
        plan=plan,
        applied_limits=AppliedMemoryLimits(
            plan=plan,
            applied=True,
            previous_wired_limit_bytes=0,
            previous_metal_limit_bytes=0,
            metal_limit_api_path="mx.metal.set_memory_limit",
        ),
        api_status=memory_limit_api_status(_FakeMLX()),
        runtime_stack=runtime_stack_for_receipt(),
        full_acceptance_claim=True,
    )

    assert receipt["memory_limits_applied_meet_gate"] is True
    assert receipt["memory_limit_ratios_meet_gate"] is False
    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["status"] == "partial"


@pytest.mark.parametrize(
    "run_command",
    [
        None,
        (
            "./.venv/bin/python",
            "-m",
            "scripts.m04_train_step",
            "--model",
            "HybridTinyLM",
        ),
        (
            "./.venv/bin/python",
            "-m",
            "scripts.train_local_gb10_quarter",
            "--model-profile=local_gb10_quarter",
            "--grad-checkpoint=false",
            "--apply-memory-limit-plan=false",
            "--clear-cache-every-steps=10",
        ),
    ],
)
def test_memory_limit_receipt_requires_acceptance_run_command_provenance(
    run_command: tuple[str, ...] | None,
) -> None:
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=10,
        clear_cache_events=clear_cache_events_for_receipt(),
        model_profile="local_gb10_quarter",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=True,
        applied_limits=applied_limits_for_receipt(),
        api_status=memory_limit_api_status(_FakeMLX()),
        runtime_stack=runtime_stack_for_receipt(),
        run_command=run_command,
        full_acceptance_claim=True,
    )

    assert receipt["model_profile_meets_gate"] is True
    assert receipt["grad_checkpoint_meets_gate"] is True
    assert receipt["run_command_meets_gate"] is False
    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["status"] == "partial"


@pytest.mark.parametrize(
    "applied_limits",
    [
        AppliedMemoryLimits(
            plan=memory_limit_plan(DEV_128_TOTAL_BYTES),
            applied=True,
            previous_wired_limit_bytes=None,
            previous_metal_limit_bytes=0,
            metal_limit_api_path="mx.metal.set_memory_limit",
        ),
        AppliedMemoryLimits(
            plan=memory_limit_plan(DEV_128_TOTAL_BYTES),
            applied=True,
            previous_wired_limit_bytes=0,
            previous_metal_limit_bytes=None,
            metal_limit_api_path="mx.metal.set_memory_limit",
        ),
        AppliedMemoryLimits(
            plan=memory_limit_plan(DEV_128_TOTAL_BYTES),
            applied=True,
            previous_wired_limit_bytes=0,
            previous_metal_limit_bytes=0,
            metal_limit_api_path=None,
        ),
        AppliedMemoryLimits(
            plan=memory_limit_plan(DEV_128_TOTAL_BYTES),
            applied=True,
            previous_wired_limit_bytes=0,
            previous_metal_limit_bytes=0,
            metal_limit_api_path="mx.unsupported.set_memory_limit",
        ),
    ],
)
def test_memory_limit_receipt_requires_applied_limit_evidence_for_acceptance(
    applied_limits: AppliedMemoryLimits,
) -> None:
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=10,
        clear_cache_events=clear_cache_events_for_receipt(),
        model_profile="local_gb10_quarter",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=True,
        applied_limits=applied_limits,
        api_status=memory_limit_api_status(_FakeMLX()),
        runtime_stack=runtime_stack_for_receipt(),
        full_acceptance_claim=True,
    )

    assert receipt["memory_limits_applied_meet_gate"] is False
    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["status"] == "partial"


def test_memory_limit_receipt_requires_wired_limit_api_status_for_acceptance() -> None:
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=10,
        clear_cache_events=clear_cache_events_for_receipt(),
        model_profile="local_gb10_quarter",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=True,
        applied_limits=AppliedMemoryLimits(
            plan=memory_limit_plan(DEV_128_TOTAL_BYTES),
            applied=True,
            previous_wired_limit_bytes=0,
            previous_metal_limit_bytes=0,
            metal_limit_api_path="mx.set_memory_limit",
        ),
        api_status=MemoryLimitApiStatus(
            wired_limit_available=False,
            root_memory_limit_available=True,
            metal_memory_limit_available=True,
            preferred_memory_limit_api_path="mx.metal.set_memory_limit",
            supported_memory_limit_api_paths=(
                "mx.metal.set_memory_limit",
                "mx.set_memory_limit",
            ),
        ),
        runtime_stack=runtime_stack_for_receipt(),
        full_acceptance_claim=True,
    )

    assert receipt["memory_limits_applied_meet_gate"] is False
    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["status"] == "partial"


def test_memory_limit_receipt_requires_clear_cache_event_for_acceptance() -> None:
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=0,
        model_profile="local_gb10_quarter",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=True,
        applied_limits=applied_limits_for_receipt(),
        api_status=memory_limit_api_status(_FakeMLX()),
        runtime_stack=runtime_stack_for_receipt(),
        full_acceptance_claim=True,
    )

    assert receipt["clear_cache_cadence_recorded"] is True
    assert receipt["clear_cache_events_meet_gate"] is False
    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["status"] == "partial"


def test_memory_limit_receipt_requires_complete_clear_cache_cadence_for_acceptance() -> (
    None
):
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=9,
        clear_cache_events=clear_cache_events_for_receipt(steps=90),
        model_profile="local_gb10_quarter",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=True,
        applied_limits=applied_limits_for_receipt(),
        api_status=memory_limit_api_status(_FakeMLX()),
        runtime_stack=runtime_stack_for_receipt(),
        full_acceptance_claim=True,
    )

    assert receipt["expected_clear_cache_event_count"] == 10
    assert receipt["clear_cache_event_count_meets_gate"] is False
    assert receipt["clear_cache_event_sequence_meets_gate"] is False
    assert receipt["clear_cache_events_meet_gate"] is False
    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["status"] == "partial"


def test_memory_limit_receipt_requires_clear_cache_event_cadence_metadata() -> None:
    mismatched_events = [
        {
            **event.to_dict(),
            "every_steps": 20,
        }
        for event in clear_cache_events_for_receipt()
    ]
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=10,
        clear_cache_events=mismatched_events,
        model_profile="local_gb10_quarter",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=True,
        applied_limits=applied_limits_for_receipt(),
        api_status=memory_limit_api_status(_FakeMLX()),
        runtime_stack=runtime_stack_for_receipt(),
        run_command=local_gb10_quarter_run_command(),
        full_acceptance_claim=True,
    )

    assert receipt["clear_cache_event_steps"] == [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    ]
    assert receipt["clear_cache_event_api_paths"] == ["mx.clear_cache"] * 10
    assert receipt["clear_cache_event_cadences"] == [20] * 10
    assert receipt["clear_cache_event_sequence_meets_gate"] is False
    assert receipt["clear_cache_events_meet_gate"] is False
    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["status"] == "partial"


@pytest.mark.parametrize(
    "runtime_stack",
    [
        RuntimeStackEvidence(
            mlx_version=None,
            mlx_metal_version="0.31.1",
            platform_system="Darwin",
            platform_release="25.4.0",
            macos_version="26.4.1",
            machine="arm64",
            default_device="Device(gpu, 0)",
            metal_available=True,
            device_info={
                "memory_size": DEV_128_TOTAL_BYTES,
                "device_name": "Apple M4 Max",
            },
        ),
        RuntimeStackEvidence(
            mlx_version="0.31.1",
            mlx_metal_version=None,
            platform_system="Darwin",
            platform_release="25.4.0",
            macos_version="26.4.1",
            machine="arm64",
            default_device="Device(gpu, 0)",
            metal_available=True,
            device_info={
                "memory_size": DEV_128_TOTAL_BYTES,
                "device_name": "Apple M4 Max",
            },
        ),
        RuntimeStackEvidence(
            mlx_version="0.31.1",
            mlx_metal_version="0.31.1",
            platform_system="Darwin",
            platform_release="25.4.0",
            macos_version="26.4.1",
            machine="arm64",
            default_device="Device(gpu, 0)",
            metal_available=False,
            device_info={
                "memory_size": DEV_128_TOTAL_BYTES,
                "device_name": "Apple M4 Max",
            },
        ),
        RuntimeStackEvidence(
            mlx_version="0.31.1",
            mlx_metal_version="0.31.1",
            platform_system="Darwin",
            platform_release="25.4.0",
            macos_version="26.4.1",
            machine="arm64",
            default_device="",
            metal_available=True,
            device_info={
                "memory_size": DEV_128_TOTAL_BYTES,
                "device_name": "Apple M4 Max",
            },
        ),
        RuntimeStackEvidence(
            mlx_version="0.31.1",
            mlx_metal_version="0.31.1",
            platform_system="Linux",
            platform_release="6.9.0",
            macos_version="",
            machine="arm64",
            default_device="Device(gpu, 0)",
            metal_available=True,
            device_info={
                "memory_size": DEV_128_TOTAL_BYTES,
                "device_name": "Apple M4 Max",
            },
        ),
        RuntimeStackEvidence(
            mlx_version="0.31.1",
            mlx_metal_version="0.31.1",
            platform_system="Darwin",
            platform_release="25.4.0",
            macos_version="26.4.1",
            machine="x86_64",
            default_device="Device(gpu, 0)",
            metal_available=True,
            device_info={"memory_size": DEV_128_TOTAL_BYTES},
        ),
        RuntimeStackEvidence(
            mlx_version="0.31.1",
            mlx_metal_version="0.31.1",
            platform_system="Darwin",
            platform_release="25.4.0",
            macos_version="26.4.1",
            machine="arm64",
            default_device="Device(gpu, 0)",
            metal_available=True,
            device_info={"memory_size": "unknown", "device_name": "Apple M4 Max"},
        ),
    ],
)
def test_memory_limit_receipt_requires_runtime_stack_for_acceptance(
    runtime_stack: RuntimeStackEvidence | None,
) -> None:
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=10,
        clear_cache_events=clear_cache_events_for_receipt(),
        model_profile="local_gb10_quarter",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=True,
        applied_limits=applied_limits_for_receipt(),
        api_status=memory_limit_api_status(_FakeMLX()),
        runtime_stack=runtime_stack,
        full_acceptance_claim=True,
    )

    assert receipt["runtime_stack_meets_gate"] is False
    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["status"] == "partial"


def test_memory_limit_receipt_fails_closed_when_auto_runtime_stack_is_incomplete() -> (
    None
):
    receipt = memory_limit_receipt(
        total_bytes=DEV_128_TOTAL_BYTES,
        peak_memory_bytes=151_213_010,
        measured_steps=100,
        clear_cache_every_steps=10,
        clear_cache_event_count=10,
        clear_cache_events=clear_cache_events_for_receipt(),
        model_profile="local_gb10_quarter",
        optimizer_name="AdamW",
        grad_checkpoint_enabled=True,
        applied_limits=applied_limits_for_receipt(),
        api_status=memory_limit_api_status(_FakeMLX()),
        mx_module=object(),
        full_acceptance_claim=True,
    )

    assert receipt["runtime_stack_recorded"] is True
    assert receipt["runtime_stack_meets_gate"] is False
    assert receipt["full_m0_6_acceptance_claim"] is False
    assert receipt["status"] == "partial"


def applied_limits_for_receipt() -> AppliedMemoryLimits:
    return AppliedMemoryLimits(
        plan=memory_limit_plan(DEV_128_TOTAL_BYTES),
        applied=True,
        previous_wired_limit_bytes=0,
        previous_metal_limit_bytes=0,
        metal_limit_api_path="mx.metal.set_memory_limit",
    )


def clear_cache_events_for_receipt(
    *,
    steps: int = 100,
    every_steps: int = 10,
) -> list[ClearCacheEvent]:
    return [
        ClearCacheEvent(
            step=step,
            every_steps=every_steps,
            api_path="mx.clear_cache",
            cache_memory_before_bytes=64,
            cache_memory_after_bytes=0,
        )
        for step in range(every_steps, steps + 1, every_steps)
    ]


def local_gb10_quarter_run_command() -> tuple[str, ...]:
    return (
        "./.venv/bin/python",
        "-m",
        "scripts.train_local_gb10_quarter",
        "--model-profile",
        "local_gb10_quarter",
        "--optimizer",
        "AdamW",
        "--grad-checkpoint",
        "--apply-memory-limit-plan",
        "--clear-cache-every-steps",
        "10",
    )


def runtime_stack_for_receipt() -> RuntimeStackEvidence:
    return RuntimeStackEvidence(
        mlx_version="0.31.1",
        mlx_metal_version="0.31.1",
        platform_system="Darwin",
        platform_release="25.4.0",
        macos_version="26.4.1",
        machine="arm64",
        default_device="Device(gpu, 0)",
        metal_available=True,
        device_info={
            "architecture": "applegpu_g16s",
            "device_name": "Apple M4 Max",
            "memory_size": DEV_128_TOTAL_BYTES,
        },
    )
