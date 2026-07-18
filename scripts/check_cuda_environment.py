#!/usr/bin/env python3
"""Fail-closed read-only preflight for the self-hosted CUDA lane."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from typing import Any


REQUIRED_MODULES = ("torch", "triton", "tilelang")


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _nvidia_smi() -> tuple[str | None, str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None, "nvidia-smi is not on PATH"
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"nvidia-smi failed: {exc}"
    if result.returncode != 0:
        return None, result.stderr.strip() or f"nvidia-smi exited {result.returncode}"
    value = result.stdout.strip()
    return (value, None) if value else (None, "nvidia-smi returned no GPUs")


def collect_cuda_environment() -> dict[str, Any]:
    report: dict[str, Any] = {
        "kind": "cppmega_mlx_cuda_environment",
        "schema_version": 1,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "nvidia_smi": None,
        "versions": {name: _version(name) for name in REQUIRED_MODULES},
        "torch": {
            "imported": False,
            "cuda_available": False,
            "cuda_version": None,
            "device_count": 0,
            "devices": [],
        },
        "errors": [],
    }

    smi, smi_error = _nvidia_smi()
    report["nvidia_smi"] = smi
    if smi_error:
        report["errors"].append(smi_error)

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        report["errors"].append(f"torch import failed: {type(exc).__name__}: {exc}")
    else:
        torch_report = report["torch"]
        torch_report["imported"] = True
        torch_report["cuda_version"] = getattr(torch.version, "cuda", None)
        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            report["errors"].append("torch.cuda is unavailable")
        else:
            try:
                torch_report["cuda_available"] = bool(cuda.is_available())
                torch_report["device_count"] = int(cuda.device_count())
                if torch_report["cuda_available"] and torch_report["device_count"]:
                    torch_report["devices"] = [
                        {
                            "index": index,
                            "name": str(cuda.get_device_name(index)),
                            "capability": list(cuda.get_device_capability(index)),
                        }
                        for index in range(torch_report["device_count"])
                    ]
            except Exception as exc:
                report["errors"].append(
                    f"torch CUDA probe failed: {type(exc).__name__}: {exc}"
                )

    for module_name in REQUIRED_MODULES[1:]:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            report["errors"].append(
                f"{module_name} import failed: {type(exc).__name__}: {exc}"
            )

    if not report["torch"]["cuda_available"]:
        report["errors"].append("a CUDA device is required but torch.cuda is unavailable")
    if not report["torch"]["cuda_version"]:
        report["errors"].append("Torch was built without a reported CUDA runtime")
    report["ok"] = not report["errors"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = collect_cuda_environment()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("PASS" if report["ok"] else "FAIL")
        for error in report["errors"]:
            print(f"ERROR: {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
