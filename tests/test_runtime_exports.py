from __future__ import annotations

import cppmega_mlx.runtime as runtime


EXPECTED_RUNTIME_EXPORTS = [
    "AppliedMemoryLimits",
    "DEFAULT_METAL_RATIO",
    "DEFAULT_WIRED_RATIO",
    "KernelPath",
    "MemoryLimitPlan",
    "RuntimeEnvironment",
    "apply_memory_limit_plan",
    "capture_rng_state",
    "clear_dispatch_log",
    "detect_runtime_environment",
    "device_total_memory_bytes",
    "get_dispatch_log",
    "memory_limit_plan",
    "mlx_rng_state_available",
    "record_dispatch",
    "restore_rng_state",
    "seed_all",
    "selected_path",
]


def test_runtime_public_exports_are_explicit_and_stable() -> None:
    assert runtime.__all__ == EXPECTED_RUNTIME_EXPORTS
    assert len(runtime.__all__) == len(set(runtime.__all__))

    for name in EXPECTED_RUNTIME_EXPORTS:
        assert getattr(runtime, name) is not None


def test_runtime_wildcard_surface_matches_all() -> None:
    wildcard_namespace: dict[str, object] = {}
    exec("from cppmega_mlx.runtime import *", {}, wildcard_namespace)

    assert sorted(wildcard_namespace) == sorted(EXPECTED_RUNTIME_EXPORTS)


def test_runtime_exports_do_not_leak_unstable_internals() -> None:
    assert all(not name.startswith("_") for name in runtime.__all__)
    assert "memory" not in runtime.__all__
    assert "seed" not in runtime.__all__
    assert "env" not in runtime.__all__
