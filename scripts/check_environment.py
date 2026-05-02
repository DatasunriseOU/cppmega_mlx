#!/usr/bin/env python3
"""Read-only environment report for cppmega.mlx local MLX bring-up."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Mapping, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FD_RECOMMENDED_MIN = 65_536


JsonScalar = bool | int | float | str | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def json_safe(value: Any) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_safe(item) for item in value]
    return str(value)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def bytes_to_gib(value: int | None) -> float | None:
    return None if value is None else value / 1024**3


def call_noarg(obj: Any, name: str) -> tuple[Any, str | None]:
    func = getattr(obj, name, None)
    if func is None:
        return None, f"{name} unavailable"
    try:
        return func(), None
    except Exception as exc:  # pragma: no cover - backend/version dependent.
        return None, f"{name} failed: {exc}"


def run_command(args: list[str], *, timeout_s: float) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            args,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        return None, message
    return result.stdout.strip(), None


def sysctl_value(name: str, *, timeout_s: float = 1.0) -> str | None:
    if shutil.which("sysctl") is None:
        return None
    stdout, _ = run_command(["sysctl", "-n", name], timeout_s=timeout_s)
    return stdout if stdout else None


def system_ram_bytes() -> int | None:
    sysctl_ram = optional_int(sysctl_value("hw.memsize"))
    if sysctl_ram is not None:
        return sysctl_ram
    pages = getattr(os, "sysconf", None)
    if pages is None:
        return None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        return None
    return page_size * page_count


def file_descriptor_limits() -> dict[str, JsonValue]:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX fallback.
        return {
            "available": False,
            "soft": None,
            "hard": None,
            "recommended_min_soft": FD_RECOMMENDED_MIN,
            "meets_recommended_min": None,
            "error": "resource module unavailable",
        }

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError) as exc:
        return {
            "available": False,
            "soft": None,
            "hard": None,
            "recommended_min_soft": FD_RECOMMENDED_MIN,
            "meets_recommended_min": None,
            "error": str(exc),
        }
    hard_is_infinite = hard == resource.RLIM_INFINITY
    return {
        "available": True,
        "soft": int(soft),
        "hard": None if hard_is_infinite else int(hard),
        "hard_is_infinite": hard_is_infinite,
        "recommended_min_soft": FD_RECOMMENDED_MIN,
        "meets_recommended_min": int(soft) >= FD_RECOMMENDED_MIN,
        "error": None,
    }


def platform_report() -> dict[str, JsonValue]:
    mac_ver = platform.mac_ver()
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node(),
        },
        "macos": {
            "is_macos": platform.system() == "Darwin",
            "version": mac_ver[0] or None,
            "version_info": [part for part in mac_ver[1] if part],
        },
    }


def collect_mlx_report() -> dict[str, JsonValue]:
    version = package_version("mlx")
    report: dict[str, Any] = {
        "installed": version is not None,
        "version": version,
        "import_error": None,
        "default_device": None,
        "device_info": {},
        "memory": memory_report(None, prefix="mlx.core"),
    }
    if version is None:
        report["import_error"] = "package not installed"
        return json_safe(report)  # type: ignore[return-value]

    try:
        mx = import_module("mlx.core")
    except Exception as exc:
        report["installed"] = False
        report["import_error"] = str(exc)
        return json_safe(report)  # type: ignore[return-value]

    default_device, default_error = call_noarg(mx, "default_device")
    report["default_device"] = str(default_device) if default_error is None else None
    report["default_device_error"] = default_error

    device_info, device_error = call_noarg(mx, "device_info")
    report["device_info"] = device_info if device_error is None else {}
    report["device_info_error"] = device_error
    report["memory"] = memory_report(mx, prefix="mlx.core")
    return json_safe(report)  # type: ignore[return-value]


def memory_report(obj: Any, *, prefix: str) -> dict[str, JsonValue]:
    counters: dict[str, int | None] = {}
    errors: list[str] = []
    for label, api_name in (
        ("active_bytes", "get_active_memory"),
        ("peak_bytes", "get_peak_memory"),
        ("cache_bytes", "get_cache_memory"),
    ):
        value, error = (
            call_noarg(obj, api_name)
            if obj is not None
            else (None, f"{api_name} unavailable")
        )
        counters[label] = optional_int(value)
        if error is not None:
            errors.append(f"{prefix}.{error}")
    return cast(
        dict[str, JsonValue],
        json_safe({
            "active_bytes": counters["active_bytes"],
            "peak_bytes": counters["peak_bytes"],
            "cache_bytes": counters["cache_bytes"],
            "active_gib": bytes_to_gib(counters["active_bytes"]),
            "peak_gib": bytes_to_gib(counters["peak_bytes"]),
            "cache_gib": bytes_to_gib(counters["cache_bytes"]),
            "available": any(value is not None for value in counters.values()),
            "errors": errors,
        }),
    )


def collect_metal_report() -> dict[str, JsonValue]:
    report: dict[str, Any] = {
        "module_present": False,
        "available": None,
        "available_error": None,
        "device_info": {},
        "device_info_error": None,
        "device_info_source": None,
        "memory": memory_report(None, prefix="mlx.core.metal"),
        "memory_source": None,
    }
    if package_version("mlx") is None:
        return json_safe(report)  # type: ignore[return-value]
    try:
        mx = import_module("mlx.core")
    except Exception as exc:
        report["available_error"] = str(exc)
        return json_safe(report)  # type: ignore[return-value]
    metal = getattr(mx, "metal", None)
    if metal is None:
        return json_safe(report)  # type: ignore[return-value]

    report["module_present"] = True
    available, available_error = call_noarg(metal, "is_available")
    report["available"] = bool(available) if available_error is None else None
    report["available_error"] = available_error
    device_info, device_error = call_noarg(mx, "device_info")
    report["device_info"] = device_info if device_error is None else {}
    report["device_info_error"] = device_error
    report["device_info_source"] = "mlx.core.device_info"
    report["memory"] = memory_report(mx, prefix="mlx.core")
    report["memory_source"] = "mlx.core"
    return json_safe(report)  # type: ignore[return-value]


def collect_distributed_report() -> dict[str, JsonValue]:
    report: dict[str, Any] = {
        "module_available": False,
        "api_source": None,
        "backend_available": {},
        "jaccl_available": None,
        "jaccl_status": "unknown",
        "error": None,
    }
    if package_version("mlx") is None:
        report["jaccl_status"] = "MLX package unavailable"
        return json_safe(report)  # type: ignore[return-value]
    try:
        mx = import_module("mlx.core")
    except Exception as exc:
        report["jaccl_status"] = "import_failed"
        report["error"] = str(exc)
        return json_safe(report)  # type: ignore[return-value]

    dist = getattr(mx, "distributed", None)
    if dist is None:
        report["jaccl_status"] = "mlx.core.distributed unavailable"
        return json_safe(report)  # type: ignore[return-value]

    report["module_available"] = True
    report["api_source"] = "mlx.core.distributed"
    is_available = getattr(dist, "is_available", None)
    if not callable(is_available):
        report["jaccl_status"] = "mlx.core.distributed.is_available unavailable"
        return json_safe(report)  # type: ignore[return-value]

    backend_available: dict[str, bool | None] = {}
    errors: list[str] = []
    for backend in ("any", "ring", "jaccl", "mpi", "nccl"):
        try:
            backend_available[backend] = bool(is_available(backend))
        except Exception as exc:  # pragma: no cover - backend/version dependent.
            backend_available[backend] = None
            errors.append(f"{backend}: {exc}")

    jaccl_available = backend_available["jaccl"]
    report["backend_available"] = backend_available
    report["jaccl_available"] = jaccl_available
    report["jaccl_status"] = (
        "available"
        if jaccl_available is True
        else "unavailable"
        if jaccl_available is False
        else "unknown"
    )
    if errors:
        report["error"] = "; ".join(errors)
    return json_safe(report)  # type: ignore[return-value]


def collect_thermal_report() -> dict[str, JsonValue]:
    powermetrics = shutil.which("powermetrics")
    return {
        "powermetrics_path": powermetrics,
        "status": "not_sampled",
        "requires_privileged_sampling": True,
        "note": "Read-only check reports tool presence only; it does not run powermetrics.",
    }


def extract_gpu_summary(value: Any) -> dict[str, JsonValue]:
    entries: list[dict[str, JsonValue]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            gpu_bits: dict[str, JsonValue] = {}
            for key, item in node.items():
                key_text = str(key).casefold()
                if "chipset" in key_text or "model" in key_text or "name" in key_text:
                    gpu_bits[str(key)] = json_safe(item)
                if "core" in key_text:
                    gpu_bits[str(key)] = json_safe(item)
            if gpu_bits:
                entries.append(gpu_bits)
            for item in node.values():
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    return cast(
        dict[str, JsonValue],
        json_safe({
            "entries": entries,
            "gpu_core_values": [
                item
                for entry in entries
                for key, item in entry.items()
                if "core" in key.casefold()
            ],
        }),
    )


def collect_system_profiler_gpu(timeout_s: float, *, enabled: bool) -> dict[str, JsonValue]:
    if not enabled:
        return {
            "queried": False,
            "available": False,
            "error": "disabled",
            "summary": {"entries": [], "gpu_core_values": []},
        }
    if shutil.which("system_profiler") is None:
        return {
            "queried": False,
            "available": False,
            "error": "system_profiler unavailable",
            "summary": {"entries": [], "gpu_core_values": []},
        }
    stdout, error = run_command(
        ["system_profiler", "SPDisplaysDataType", "-json"],
        timeout_s=timeout_s,
    )
    if error is not None or not stdout:
        return {
            "queried": True,
            "available": False,
            "error": error or "empty output",
            "summary": {"entries": [], "gpu_core_values": []},
        }
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return {
            "queried": True,
            "available": False,
            "error": str(exc),
            "summary": {"entries": [], "gpu_core_values": []},
        }
    return {
        "queried": True,
        "available": True,
        "error": None,
        "summary": extract_gpu_summary(payload),
    }


def collect_environment(
    *,
    system_profiler: bool = True,
    system_profiler_timeout_s: float = 5.0,
) -> dict[str, JsonValue]:
    platform_payload = platform_report()
    ram = system_ram_bytes()
    mlx = collect_mlx_report()
    metal = collect_metal_report()
    device_info = mlx.get("device_info") if isinstance(mlx, dict) else {}
    device_info_map = device_info if isinstance(device_info, dict) else {}
    return {
        "schema_version": 1,
        "kind": "cppmega_mlx_environment_report",
        "read_only": True,
        **platform_payload,
        "host": {
            "ram_bytes": ram,
            "ram_gib": bytes_to_gib(ram),
            "cpu_brand": sysctl_value("machdep.cpu.brand_string"),
            "physical_cpu_count": optional_int(sysctl_value("hw.physicalcpu")),
            "gpu": {
                "mlx_device_name": device_info_map.get("device_name"),
                "mlx_architecture": device_info_map.get("architecture"),
                "system_profiler": collect_system_profiler_gpu(
                    system_profiler_timeout_s,
                    enabled=system_profiler,
                ),
            },
        },
        "mlx": mlx,
        "metal": metal,
        "distributed": collect_distributed_report(),
        "thermal": collect_thermal_report(),
        "file_descriptors": file_descriptor_limits(),
    }


def render_text(report: Mapping[str, JsonValue]) -> str:
    lines = [
        "cppmega.mlx environment report",
        f"python: {nested(report, 'python', 'version')} ({nested(report, 'python', 'implementation')})",
        f"platform: {nested(report, 'platform', 'platform')}",
        f"macOS: {nested(report, 'macos', 'version') or 'not macOS'}",
        f"RAM: {nested(report, 'host', 'ram_gib')} GiB",
        f"MLX: {nested(report, 'mlx', 'version') or 'not installed'}",
        f"default device: {nested(report, 'mlx', 'default_device') or 'unavailable'}",
        f"Metal available: {nested(report, 'metal', 'available')}",
        f"Metal active memory bytes: {nested(report, 'metal', 'memory', 'active_bytes')}",
        f"Metal peak memory bytes: {nested(report, 'metal', 'memory', 'peak_bytes')}",
        f"file descriptors: soft={nested(report, 'file_descriptors', 'soft')} hard={nested(report, 'file_descriptors', 'hard')}",
        f"JACCL: {nested(report, 'distributed', 'jaccl_status')}",
        f"thermal: {nested(report, 'thermal', 'status')}",
    ]
    return "\n".join(lines)


def nested(mapping: Mapping[str, JsonValue], *keys: str) -> JsonValue:
    current: JsonValue = dict(mapping)
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full report as JSON instead of a compact text summary.",
    )
    parser.add_argument(
        "--no-system-profiler",
        action="store_true",
        help="Skip the optional system_profiler GPU-core probe.",
    )
    parser.add_argument(
        "--system-profiler-timeout-s",
        type=float,
        default=5.0,
        help="Timeout for the optional system_profiler GPU probe.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = collect_environment(
        system_profiler=not args.no_system_profiler,
        system_profiler_timeout_s=args.system_profiler_timeout_s,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
