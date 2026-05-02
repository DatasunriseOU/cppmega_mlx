"""Read-only runtime environment detection for reporting guardrails."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module, metadata
from typing import Any, cast
import platform

JsonScalar = bool | int | float | str | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Small, read-only runtime report for local MLX guardrails."""

    python: dict[str, JsonValue]
    platform: dict[str, JsonValue]
    macos: dict[str, JsonValue]
    process: dict[str, JsonValue]
    mlx: dict[str, JsonValue]
    hardware_tier: str
    schema_version: int = 1
    kind: str = "cppmega_mlx_runtime_environment"
    read_only: bool = True

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "read_only": self.read_only,
            "python": dict(self.python),
            "platform": dict(self.platform),
            "macos": dict(self.macos),
            "process": dict(self.process),
            "mlx": dict(self.mlx),
            "hardware_tier": self.hardware_tier,
        }


def detect_runtime_environment(*, mx_module: Any | None = None) -> RuntimeEnvironment:
    """Collect a read-only environment report without changing MLX limits."""

    python_payload = _python_report()
    platform_payload = _platform_report()
    macos_payload = _macos_report(system=str(platform_payload["system"]))
    process_payload = _process_report(machine=str(platform_payload["machine"]))
    mlx_payload = _collect_mlx_report(mx_module=mx_module)
    raw_device_info = mlx_payload.get("device_info")
    device_info = raw_device_info if isinstance(raw_device_info, Mapping) else {}

    return RuntimeEnvironment(
        python=python_payload,
        platform=platform_payload,
        macos=macos_payload,
        process=process_payload,
        mlx=mlx_payload,
        hardware_tier=_hardware_tier(
            system=str(platform_payload["system"]),
            machine=str(platform_payload["machine"]),
            device_info=device_info,
        ),
    )


def _python_report() -> dict[str, JsonValue]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }


def _platform_report() -> dict[str, JsonValue]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _macos_report(*, system: str) -> dict[str, JsonValue]:
    version, version_info, machine = platform.mac_ver()
    return {
        "is_macos": system == "Darwin",
        "version": version or None,
        "version_info": [part for part in version_info if part],
        "machine": machine or None,
    }


def _process_report(*, machine: str) -> dict[str, JsonValue]:
    bitness, linkage = platform.architecture()
    return {
        "arch": machine,
        "bitness": bitness or None,
        "linkage": linkage or None,
    }


def _collect_mlx_report(*, mx_module: Any | None) -> dict[str, JsonValue]:
    version = _package_version("mlx")
    report: dict[str, Any] = {
        "available": False,
        "version": version,
        "import_error": None,
        "default_device": None,
        "default_device_error": None,
        "device_info": {},
        "device_info_error": None,
    }

    if mx_module is None:
        if version is None:
            report["import_error"] = "package not installed"
            return cast(dict[str, JsonValue], _json_safe(report))
        try:
            mx_module = _import_mlx_core()
        except Exception as exc:  # pragma: no cover - depends on local install.
            report["import_error"] = f"{type(exc).__name__}: {exc}"
            return cast(dict[str, JsonValue], _json_safe(report))

    report["available"] = True

    default_device, default_device_error = _call_noarg(mx_module, "default_device")
    if default_device_error is None:
        report["default_device"] = str(default_device)
    else:
        report["default_device_error"] = default_device_error

    device_info, device_info_error = _call_noarg(mx_module, "device_info")
    if device_info_error is not None:
        report["device_info_error"] = device_info_error
    elif isinstance(device_info, Mapping):
        report["device_info"] = dict(device_info)
    else:
        report["device_info_error"] = "device_info returned non-mapping"

    return cast(dict[str, JsonValue], _json_safe(report))


def _hardware_tier(
    *,
    system: str,
    machine: str,
    device_info: Mapping[str, JsonValue],
) -> str:
    if system == "Darwin" and machine.casefold() in {"arm64", "aarch64"}:
        return "apple_silicon_local"
    device_name = str(device_info.get("device_name") or "")
    if system == "Darwin" and "apple" in device_name.casefold():
        return "apple_silicon_local"
    return "unknown"


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _import_mlx_core() -> Any:
    return import_module("mlx.core")


def _call_noarg(obj: Any, name: str) -> tuple[Any, str | None]:
    func = getattr(obj, name, None)
    if not callable(func):
        return None, f"{name} unavailable"
    try:
        return func(), None
    except Exception as exc:  # pragma: no cover - backend/version dependent.
        return None, f"{name} failed: {type(exc).__name__}: {exc}"


def _json_safe(value: Any) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return str(value)


__all__ = [
    "RuntimeEnvironment",
    "detect_runtime_environment",
]
