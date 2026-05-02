from __future__ import annotations

import pytest

from cppmega_mlx.runtime.memory import (
    DEFAULT_METAL_RATIO,
    DEFAULT_WIRED_RATIO,
    AppliedMemoryLimits,
    MemoryLimitPlan,
    apply_memory_limit_plan,
    device_total_memory_bytes,
    memory_limit_plan,
)


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

    def set_wired_limit(self, limit: int) -> int:
        self.calls.append(limit)
        return 123

    def device_info(self) -> dict[str, int]:
        return {"memory_size": 4096}


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
    assert fake.calls == [700]
    assert fake.metal.calls == [850]


def test_apply_memory_limit_plan_rejects_missing_apis() -> None:
    plan = memory_limit_plan(1000)

    with pytest.raises(RuntimeError, match="set_wired_limit is unavailable"):
        apply_memory_limit_plan(plan, mx_module=object(), apply=True)


def test_device_total_memory_bytes_reads_fake_mlx_device_info() -> None:
    assert device_total_memory_bytes(_FakeMLX()) == 4096


def test_device_total_memory_bytes_handles_missing_device_info() -> None:
    assert device_total_memory_bytes(object()) is None
