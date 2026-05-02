from __future__ import annotations

from typing import Any, cast

import pytest

import cppmega_mlx.runtime.env as env_runtime


def _patch_platform(
    monkeypatch: pytest.MonkeyPatch,
    *,
    system: str = "Darwin",
    machine: str = "arm64",
) -> None:
    monkeypatch.setattr(env_runtime.platform, "python_version", lambda: "3.13.0")
    monkeypatch.setattr(env_runtime.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(env_runtime.platform, "platform", lambda: "macOS-15.4-arm64-arm-64bit")
    monkeypatch.setattr(env_runtime.platform, "system", lambda: system)
    monkeypatch.setattr(env_runtime.platform, "release", lambda: "24.4.0")
    monkeypatch.setattr(env_runtime.platform, "version", lambda: "Darwin Kernel Version 24.4.0")
    monkeypatch.setattr(env_runtime.platform, "machine", lambda: machine)
    monkeypatch.setattr(env_runtime.platform, "processor", lambda: "arm")
    monkeypatch.setattr(env_runtime.platform, "mac_ver", lambda: ("15.4", ("", "", ""), machine))
    monkeypatch.setattr(env_runtime.platform, "architecture", lambda: ("64bit", ""))


def test_runtime_environment_reports_python_platform_and_conservative_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(monkeypatch)
    monkeypatch.setattr(env_runtime, "_package_version", lambda name: None)

    report = env_runtime.detect_runtime_environment()
    payload = cast(dict[str, Any], report.to_dict())

    assert payload["kind"] == "cppmega_mlx_runtime_environment"
    assert payload["read_only"] is True
    assert payload["hardware_tier"] == "apple_silicon_local"
    assert payload["python"]["version"] == "3.13.0"
    assert payload["platform"]["machine"] == "arm64"
    assert payload["macos"]["version"] == "15.4"
    assert payload["process"]["arch"] == "arm64"


def test_runtime_environment_degrades_when_mlx_package_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_import() -> object:
        raise AssertionError("MLX import should not be attempted when package is missing")

    _patch_platform(monkeypatch, system="Linux", machine="x86_64")
    monkeypatch.setattr(env_runtime, "_package_version", lambda name: None)
    monkeypatch.setattr(env_runtime, "_import_mlx_core", forbidden_import)

    payload = cast(dict[str, Any], env_runtime.detect_runtime_environment().to_dict())
    mlx = cast(dict[str, Any], payload["mlx"])

    assert payload["hardware_tier"] == "unknown"
    assert mlx["available"] is False
    assert mlx["version"] is None
    assert mlx["import_error"] == "package not installed"
    assert mlx["default_device"] is None
    assert mlx["device_info"] == {}


class _FakeDevice:
    def __str__(self) -> str:
        return "Device(gpu, 0)"


class _ForbiddenMetal:
    def __init__(self) -> None:
        self.mutating_calls: list[tuple[str, int]] = []

    def set_memory_limit(self, limit: int) -> int:
        self.mutating_calls.append(("set_memory_limit", limit))
        raise AssertionError("mutating MLX Metal memory-limit API was called")

    def set_wired_limit(self, limit: int) -> int:
        self.mutating_calls.append(("set_wired_limit", limit))
        raise AssertionError("mutating MLX Metal wired-limit API was called")


class _FakeMLX:
    def __init__(self) -> None:
        self.mutating_calls: list[tuple[str, int]] = []
        self.metal = _ForbiddenMetal()

    def default_device(self) -> _FakeDevice:
        return _FakeDevice()

    def device_info(self) -> dict[str, str | int]:
        return {
            "device_name": "Apple M4 Max",
            "architecture": "applegpu_g16p",
            "memory_size": 137_438_953_472,
        }

    def set_wired_limit(self, limit: int) -> int:
        self.mutating_calls.append(("set_wired_limit", limit))
        raise AssertionError("mutating MLX wired-limit API was called")


def test_runtime_environment_reads_fake_mlx_without_mutating_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mx = _FakeMLX()
    _patch_platform(monkeypatch)
    monkeypatch.setattr(env_runtime, "_package_version", lambda name: "0.31.1")

    payload = cast(
        dict[str, Any],
        env_runtime.detect_runtime_environment(mx_module=fake_mx).to_dict(),
    )
    mlx = cast(dict[str, Any], payload["mlx"])

    assert mlx["available"] is True
    assert mlx["version"] == "0.31.1"
    assert mlx["default_device"] == "Device(gpu, 0)"
    assert mlx["device_info"]["device_name"] == "Apple M4 Max"
    assert mlx["device_info"]["memory_size"] == 137_438_953_472
    assert fake_mx.mutating_calls == []
    assert fake_mx.metal.mutating_calls == []


def test_runtime_environment_handles_missing_mlx_read_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(monkeypatch)
    monkeypatch.setattr(env_runtime, "_package_version", lambda name: "0.31.1")

    payload = cast(
        dict[str, Any],
        env_runtime.detect_runtime_environment(mx_module=object()).to_dict(),
    )
    mlx = cast(dict[str, Any], payload["mlx"])

    assert mlx["available"] is True
    assert mlx["default_device"] is None
    assert mlx["default_device_error"] == "default_device unavailable"
    assert mlx["device_info"] == {}
    assert mlx["device_info_error"] == "device_info unavailable"
