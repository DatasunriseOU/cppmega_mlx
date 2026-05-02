from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_environment.py"

SCRIPT_SPEC = importlib.util.spec_from_file_location("check_environment", SCRIPT)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
check_environment = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = check_environment
SCRIPT_SPEC.loader.exec_module(check_environment)


def run_check_environment(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_json_report_is_read_only_and_has_required_sections() -> None:
    result = run_check_environment("--json", "--no-system-profiler")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["kind"] == "cppmega_mlx_environment_report"
    assert payload["schema_version"] == 1
    assert payload["read_only"] is True
    for key in (
        "python",
        "platform",
        "macos",
        "host",
        "mlx",
        "metal",
        "distributed",
        "thermal",
        "file_descriptors",
    ):
        assert key in payload
    assert "version" in payload["python"]
    assert "default_device" in payload["mlx"]
    assert "memory" in payload["metal"]
    assert "soft" in payload["file_descriptors"]


def test_text_report_includes_core_diagnostics() -> None:
    result = run_check_environment("--no-system-profiler")

    assert result.returncode == 0, result.stderr
    assert "cppmega.mlx environment report" in result.stdout
    assert "python:" in result.stdout
    assert "MLX:" in result.stdout
    assert "default device:" in result.stdout
    assert "file descriptors:" in result.stdout


def test_mlx_absence_degrades_without_importing_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_environment, "package_version", lambda name: None)

    payload = check_environment.collect_mlx_report()

    assert payload["installed"] is False
    assert payload["version"] is None
    assert payload["import_error"] == "package not installed"
    assert payload["default_device"] is None
    assert payload["memory"]["available"] is False


def test_mlx_report_feature_detects_memory_and_device_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mx = types.SimpleNamespace(
        default_device=lambda: "Device(gpu, 0)",
        device_info=lambda: {
            "device_name": "Apple M4 Max",
            "memory_size": 128,
        },
        get_active_memory=lambda: 4,
        get_peak_memory=lambda: 8,
        get_cache_memory=lambda: 2,
    )
    monkeypatch.setattr(
        check_environment,
        "package_version",
        lambda name: "0.31.1" if name == "mlx" else None,
    )
    monkeypatch.setattr(check_environment, "import_module", lambda name: fake_mx)

    payload = check_environment.collect_mlx_report()

    assert payload["installed"] is True
    assert payload["version"] == "0.31.1"
    assert payload["default_device"] == "Device(gpu, 0)"
    assert payload["device_info"]["device_name"] == "Apple M4 Max"
    assert payload["memory"]["active_bytes"] == 4
    assert payload["memory"]["peak_bytes"] == 8
    assert payload["memory"]["cache_bytes"] == 2
    assert payload["memory"]["available"] is True


def test_metal_report_does_not_call_mutating_limit_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_setter() -> None:
        raise AssertionError("mutating MLX Metal limit API was called")

    fake_metal = types.SimpleNamespace(
        is_available=lambda: True,
        set_memory_limit=forbidden_setter,
        set_wired_limit=forbidden_setter,
        set_cache_limit=forbidden_setter,
    )
    fake_mx = types.SimpleNamespace(
        metal=fake_metal,
        device_info=lambda: {"device_name": "Apple M4 Max"},
        get_active_memory=lambda: 10,
        get_peak_memory=lambda: 20,
        get_cache_memory=lambda: 5,
    )
    monkeypatch.setattr(
        check_environment,
        "package_version",
        lambda name: "0.31.1" if name == "mlx" else None,
    )
    monkeypatch.setattr(check_environment, "import_module", lambda name: fake_mx)

    payload = check_environment.collect_metal_report()

    assert payload["module_present"] is True
    assert payload["available"] is True
    assert payload["device_info_source"] == "mlx.core.device_info"
    assert payload["memory_source"] == "mlx.core"
    assert payload["memory"]["active_bytes"] == 10
    assert payload["memory"]["peak_bytes"] == 20
    assert payload["memory"]["cache_bytes"] == 5


def test_full_environment_report_does_not_mutate_memory_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_limit_setter(*args: object) -> None:
        raise AssertionError(f"mutating MLX memory-limit API was called: {args!r}")

    fake_metal = types.SimpleNamespace(
        is_available=lambda: True,
        set_memory_limit=forbidden_limit_setter,
    )
    fake_mx = types.SimpleNamespace(
        metal=fake_metal,
        set_wired_limit=forbidden_limit_setter,
        default_device=lambda: "Device(gpu, 0)",
        device_info=lambda: {"device_name": "Apple M4 Max", "memory_size": 128},
        get_active_memory=lambda: 10,
        get_peak_memory=lambda: 20,
        get_cache_memory=lambda: 5,
        distributed=types.SimpleNamespace(is_available=lambda backend: backend == "ring"),
    )
    monkeypatch.setattr(
        check_environment,
        "package_version",
        lambda name: "0.31.1" if name == "mlx" else None,
    )
    monkeypatch.setattr(check_environment, "import_module", lambda name: fake_mx)

    payload = check_environment.collect_environment(system_profiler=False)

    assert payload["read_only"] is True
    assert payload["mlx"]["memory"]["active_bytes"] == 10
    assert payload["metal"]["available"] is True
    assert payload["metal"]["memory"]["peak_bytes"] == 20
    assert payload["distributed"]["backend_available"]["ring"] is True


def test_distributed_report_handles_missing_mlx_distributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mx = types.SimpleNamespace()
    monkeypatch.setattr(
        check_environment,
        "package_version",
        lambda name: "0.31.1" if name == "mlx" else None,
    )
    monkeypatch.setattr(check_environment, "import_module", lambda name: fake_mx)

    payload = check_environment.collect_distributed_report()

    assert payload["module_available"] is False
    assert payload["jaccl_status"] == "mlx.core.distributed unavailable"


def test_distributed_report_uses_core_distributed_backend_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDistributed:
        def is_available(self, backend: str) -> bool:
            return backend in {"any", "ring"}

    fake_mx = types.SimpleNamespace(distributed=FakeDistributed())
    monkeypatch.setattr(
        check_environment,
        "package_version",
        lambda name: "0.31.1" if name == "mlx" else None,
    )
    monkeypatch.setattr(check_environment, "import_module", lambda name: fake_mx)

    payload = check_environment.collect_distributed_report()

    assert payload["module_available"] is True
    assert payload["api_source"] == "mlx.core.distributed"
    assert payload["backend_available"]["ring"] is True
    assert payload["backend_available"]["jaccl"] is False
    assert payload["jaccl_available"] is False
    assert payload["jaccl_status"] == "unavailable"


def test_file_descriptor_report_is_json_safe() -> None:
    payload = check_environment.file_descriptor_limits()

    json.dumps(payload)
    assert payload["recommended_min_soft"] == check_environment.FD_RECOMMENDED_MIN
    assert payload["available"] in {True, False}


def test_system_profiler_can_be_disabled() -> None:
    payload = check_environment.collect_system_profiler_gpu(0.1, enabled=False)

    assert payload["queried"] is False
    assert payload["available"] is False
    assert payload["error"] == "disabled"
