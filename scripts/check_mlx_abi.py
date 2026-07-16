#!/usr/bin/env python3
"""Fail-closed MLX/MLX-Metal runtime compatibility probe.

The local development setup can intentionally import ``mlx.core`` from a
workspace checkout through ``PYTHONPATH``.  That is a different mode from an
installed wheel and must not be compared to Homebrew by string equality: a
development build reports a ``.dev`` version even when its package contract is
the pinned release.  The probe still requires the matching ``mlx-metal`` data
package in either mode.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_environment_contract() -> Any:
    """Load the repository-owned ABI environment contract by exact path."""

    path = ROOT / "scripts" / "mlx_env_contract.py"
    spec = importlib.util.spec_from_file_location(
        "_cppmega_mlx_abi_environment_contract",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MLX environment contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_mlx_wheel_environment = _load_environment_contract().build_mlx_wheel_environment
DEFAULT_ENV_ROOT = ROOT.parent / ".venvs" / "cppmega.mlx"
DEFAULT_PYTHON = DEFAULT_ENV_ROOT / "bin" / "python"
RECEIPT_KIND = "cppmega_mlx_abi_receipt"
RECEIPT_SCHEMA_VERSION = 1
_RELEASE_RE = re.compile(r"^(\d+\.\d+\.\d+)")


@dataclass(frozen=True)
class AbiProbe:
    package_version: str | None
    module_version: str | None
    module_file: Path | None
    mlx_metal_version: str | None
    source_root: Path | None
    brew_version: str | None
    brew_prefix: Path | None = None
    brew_linked_keg: Path | None = None
    python_executable: Path | None = None
    python_prefix: Path | None = None
    distribution_root: Path | None = None
    loaded_libmlx: Path | None = None
    runtime_smoke_ok: bool = False
    runtime_smoke_detail: str | None = None
    platform_name: str = "unknown"


@dataclass(frozen=True)
class AbiEvaluation:
    ok: bool
    mode: str
    messages: tuple[str, ...]


def _release(version: str | None) -> str | None:
    if not version:
        return None
    match = _RELEASE_RE.match(version.strip())
    return match.group(1) if match else None


def _same_release(left: str | None, right: str | None) -> bool:
    return _release(left) is not None and _release(left) == _release(right)


def _inside(path: Path | None, root: Path | None) -> bool:
    if path is None or root is None:
        return False
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _same_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return left.expanduser().resolve() == right.expanduser().resolve()
    except OSError:
        return False


def _brew_version_from_library(path: Path | None, prefix: Path) -> str | None:
    if path is None:
        return None
    cellar = (prefix / "Cellar" / "mlx").resolve()
    try:
        relative = path.resolve().relative_to(cellar)
    except (OSError, ValueError):
        return None
    return relative.parts[0] if relative.parts else None


def evaluate_probe(
    probe: AbiProbe,
    *,
    expected_python: Path | None = None,
    expected_env_root: Path | None = None,
) -> AbiEvaluation:
    messages: list[str] = []
    if probe.module_file is None:
        return AbiEvaluation(False, "unavailable", ("mlx.core module file is unknown",))

    source_mode = _inside(probe.module_file, probe.source_root)
    mode = "workspace_source" if source_mode else "installed_wheel"
    effective_version = probe.module_version or probe.package_version
    if effective_version is None:
        messages.append("FAIL: mlx version is unavailable")
    if (
        probe.module_version
        and probe.package_version
        and not _same_release(probe.module_version, probe.package_version)
    ):
        messages.append(
            f"FAIL: imported mlx.core={probe.module_version!r} does not match "
            f"the selected environment distribution {probe.package_version!r}"
        )

    if probe.python_executable is None or probe.python_prefix is None:
        messages.append("FAIL: target interpreter identity is unavailable")
    if expected_python is not None and not _same_path(
        probe.python_executable, expected_python
    ):
        messages.append(
            "FAIL: runtime Python differs from the selected interpreter: "
            f"runtime={probe.python_executable}, selected={expected_python}"
        )
    if expected_env_root is not None and not _same_path(
        probe.python_prefix, expected_env_root
    ):
        messages.append(
            "FAIL: runtime Python prefix differs from the selected environment: "
            f"runtime={probe.python_prefix}, selected={expected_env_root}"
        )

    if source_mode:
        messages.append(
            "workspace source checkout: source mode was selected explicitly or "
            "through an editable install"
        )
    elif probe.distribution_root is None or not _inside(
        probe.module_file, probe.distribution_root
    ):
        messages.append(
            "FAIL: installed mlx.core does not resolve inside its distribution root: "
            f"module={probe.module_file}, distribution={probe.distribution_root}"
        )

    if not probe.runtime_smoke_ok:
        messages.append(
            "FAIL: MLX runtime smoke failed: "
            f"{probe.runtime_smoke_detail or 'no runtime detail'}"
        )
    else:
        messages.append(
            f"MLX runtime smoke: {probe.runtime_smoke_detail or 'mx.eval succeeded'}"
        )

    is_darwin = probe.platform_name.lower() == "darwin"
    if is_darwin and probe.loaded_libmlx is None:
        messages.append("FAIL: loaded libmlx.dylib origin is unavailable")
    elif is_darwin and probe.loaded_libmlx is not None:
        allowed_roots = tuple(
            root for root in (probe.source_root, probe.distribution_root) if root is not None
        )
        loaded_is_owned = any(_inside(probe.loaded_libmlx, root) for root in allowed_roots)
        brew_root = probe.brew_prefix or Path(
            os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
        )
        loaded_from_brew = _inside(probe.loaded_libmlx, brew_root)
        loaded_brew_version = _brew_version_from_library(
            probe.loaded_libmlx,
            brew_root,
        )
        active_brew_version = loaded_brew_version or probe.brew_version
        if not loaded_is_owned and not loaded_from_brew:
            messages.append(
                "FAIL: loaded libmlx.dylib is outside the selected source/distribution: "
                f"{probe.loaded_libmlx}"
            )
        elif loaded_from_brew and active_brew_version is None:
            messages.append(
                "FAIL: cannot identify the loaded Homebrew mlx version from "
                f"{probe.loaded_libmlx}"
            )
        elif loaded_from_brew and not _same_release(
            effective_version, active_brew_version
        ):
            messages.append(
                f"FAIL: loaded Homebrew libmlx={active_brew_version!r} does not match "
                f"imported mlx={effective_version!r}"
            )
        else:
            messages.append(f"loaded libmlx: {probe.loaded_libmlx}")

    if is_darwin:
        expected_metal = _release(probe.package_version or effective_version)
        actual_metal = _release(probe.mlx_metal_version)
        if actual_metal is None:
            messages.append("FAIL: mlx-metal is not installed or has no release version")
        elif expected_metal is None:
            messages.append(
                f"FAIL: cannot derive the expected mlx-metal release from mlx={effective_version!r}"
            )
        elif actual_metal != expected_metal:
            messages.append(
                f"FAIL: mlx-metal={probe.mlx_metal_version!r} does not match "
                f"the mlx release contract {expected_metal!r}"
            )
        else:
            messages.append(f"mlx-metal release contract: {actual_metal}")
    else:
        messages.append(
            f"mlx-metal contract is not applicable on platform {probe.platform_name}"
        )

    informational_brew_root = probe.brew_prefix or Path(
        os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
    )
    if probe.brew_version and probe.loaded_libmlx is not None and not _inside(
        probe.loaded_libmlx,
        informational_brew_root,
    ):
        messages.append(
            f"brew version is informational: Homebrew mlx={probe.brew_version!r} "
            "is not the loaded dylib"
        )
    elif (
        probe.brew_version
        and probe.loaded_libmlx is None
        and is_darwin
        and not source_mode
    ):
        messages.append(
            f"FAIL: cannot prove whether Homebrew mlx={probe.brew_version!r} is loaded"
        )

    ok = not any(message.startswith("FAIL:") for message in messages)
    return AbiEvaluation(ok, mode, tuple(messages))


def _probe_code() -> str:
    return r'''
import ctypes
import importlib.metadata as md
import json
from pathlib import Path
import platform
import sys

result = {}
result["platform"] = platform.system()
result["python_executable"] = str(Path(sys.executable))
result["python_prefix"] = str(Path(sys.prefix).resolve())
try:
    distribution = md.distribution("mlx")
    result["package_version"] = distribution.version
    result["distribution_root"] = str(Path(distribution.locate_file("")).resolve())
except md.PackageNotFoundError:
    result["package_version"] = None
    result["distribution_root"] = None
try:
    result["mlx_metal_version"] = md.version("mlx-metal")
except md.PackageNotFoundError:
    result["mlx_metal_version"] = None


def loaded_mlx_library():
    if sys.platform != "darwin":
        return None
    try:
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        for index in range(dyld._dyld_image_count()):
            raw = dyld._dyld_get_image_name(index)
            if raw and Path(raw.decode()).name == "libmlx.dylib":
                return str(Path(raw.decode()).resolve())
    except Exception as exc:
        result["loaded_libmlx_error"] = f"{type(exc).__name__}: {exc}"
    return None


try:
    import mlx.core as mx
    result["module_version"] = getattr(mx, "__version__", None)
    result["module_file"] = str(Path(mx.__file__).resolve()) if mx.__file__ else None
    result["loaded_libmlx"] = loaded_mlx_library()
    try:
        probe = mx.array([1, 2], dtype=mx.int32) + 1
        mx.eval(probe)
        synchronize = getattr(mx, "synchronize", None)
        if synchronize is not None:
            synchronize()
        result["runtime_smoke_ok"] = probe.tolist() == [2, 3]
        result["runtime_smoke_detail"] = "mx.eval([1, 2] + 1) produced [2, 3]"
    except Exception as exc:
        result["runtime_smoke_ok"] = False
        result["runtime_smoke_detail"] = f"{type(exc).__name__}: {exc}"
except Exception as exc:
    result["import_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, sort_keys=True))
'''


def _probe_runtime(
    python: Path,
    *,
    environment: dict[str, str] | None = None,
    source_root: Path | None = None,
) -> AbiProbe:
    # Ambient import and loader paths never participate in the selected
    # interpreter contract. Workspace-source mode remains an explicit
    # CPPMEGA_MLX_SOURCE_ROOT override; the default is the installed wheel.
    if environment is None:
        environment = build_mlx_wheel_environment(os.environ, python=python)
    else:
        environment = build_mlx_wheel_environment(environment, python=python)
    if source_root is not None:
        environment["CPPMEGA_MLX_ENV_MODE"] = "mlx-source"
        selected_source = source_root.expanduser().resolve()
        environment["CPPMEGA_MLX_SOURCE_ROOT"] = str(selected_source)
        environment["PYTHONPATH"] = str(
            selected_source / "python"
        )
    process = subprocess.run(
        [str(python), "-c", _probe_code()],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "unknown import failure"
        raise RuntimeError(f"cannot import mlx.core from {python}: {detail}")
    try:
        payload: dict[str, Any] = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"mlx probe returned invalid JSON from {python}: {process.stdout!r}"
        ) from exc
    if payload.get("import_error"):
        raise RuntimeError(f"cannot import mlx.core from {python}: {payload['import_error']}")

    module_file = payload.get("module_file")
    module_path = Path(module_file).resolve() if isinstance(module_file, str) else None
    source_root = _source_root_for(module_path, configured_source=source_root)
    brew_prefix = Path(environment.get("HOMEBREW_PREFIX", "/opt/homebrew"))
    return AbiProbe(
        package_version=payload.get("package_version"),
        module_version=payload.get("module_version"),
        module_file=module_path,
        mlx_metal_version=payload.get("mlx_metal_version"),
        source_root=source_root,
        brew_version=_brew_version(brew_prefix),
        brew_prefix=brew_prefix,
        brew_linked_keg=_brew_linked_keg(brew_prefix),
        python_executable=Path(payload["python_executable"])
        if isinstance(payload.get("python_executable"), str)
        else None,
        python_prefix=Path(payload["python_prefix"]).resolve()
        if isinstance(payload.get("python_prefix"), str)
        else None,
        distribution_root=Path(payload["distribution_root"]).resolve()
        if isinstance(payload.get("distribution_root"), str)
        else None,
        loaded_libmlx=Path(payload["loaded_libmlx"]).resolve()
        if isinstance(payload.get("loaded_libmlx"), str)
        else None,
        runtime_smoke_ok=bool(payload.get("runtime_smoke_ok", False)),
        runtime_smoke_detail=payload.get("runtime_smoke_detail"),
        platform_name=str(payload.get("platform") or platform.system()),
    )


def _source_root_for(
    module_file: Path | None,
    *,
    configured_source: Path | None = None,
) -> Path | None:
    configured = configured_source
    if configured:
        return configured.expanduser().resolve()
    if module_file is None:
        return None
    parts = module_file.parts
    try:
        index = parts.index("python")
    except ValueError:
        return None
    if index > 0 and parts[index - 1] == "mlx":
        return Path(*parts[: index - 1]).joinpath("mlx").resolve()
    return None


def _brew_linked_keg(prefix: Path | None = None) -> Path | None:
    selected_prefix = prefix or Path(os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew"))
    linked_keg = selected_prefix.expanduser() / "opt" / "mlx"
    if not linked_keg.exists():
        return None
    try:
        return linked_keg.resolve(strict=True)
    except (OSError, ValueError):
        return None


def _brew_version(prefix: Path | None = None) -> str | None:
    selected_prefix = prefix or Path(os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew"))
    cellar = selected_prefix.expanduser() / "Cellar" / "mlx"
    linked_keg = _brew_linked_keg(selected_prefix)
    if not cellar.is_dir() or linked_keg is None:
        return None
    try:
        relative = linked_keg.relative_to(cellar.resolve())
    except (OSError, ValueError):
        return None
    return relative.parts[0] if relative.parts else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-root",
        type=Path,
        default=Path(os.environ.get("CPPMEGA_MLX_ENV_ROOT", DEFAULT_ENV_ROOT)),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=(
            Path(os.environ["CPPMEGA_MLX_PYTHON"])
            if os.environ.get("CPPMEGA_MLX_PYTHON")
            else None
        ),
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def _probe_payload(probe: AbiProbe) -> dict[str, object]:
    payload = asdict(probe)
    for field_name, value in payload.items():
        if isinstance(value, Path):
            payload[field_name] = str(value)
    return payload


def _receipt(
    *,
    probe: AbiProbe,
    evaluation: AbiEvaluation,
    python: Path,
    env_root: Path,
    source_root: Path | None,
) -> dict[str, object]:
    return {
        "kind": RECEIPT_KIND,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "ok": evaluation.ok,
        "selected": {
            "python": str(python),
            "env_root": str(env_root),
            "source_root": str(source_root) if source_root is not None else None,
        },
        "probe": _probe_payload(probe),
        "evaluation": asdict(evaluation),
    }


def _unavailable_probe(
    *,
    python: Path,
    env_root: Path,
    source_root: Path | None,
    detail: str,
) -> AbiProbe:
    brew_prefix = Path(os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew"))
    return AbiProbe(
        package_version=None,
        module_version=None,
        module_file=None,
        mlx_metal_version=None,
        source_root=source_root,
        brew_version=_brew_version(brew_prefix),
        brew_prefix=brew_prefix,
        brew_linked_keg=_brew_linked_keg(brew_prefix),
        python_executable=python,
        python_prefix=env_root,
        runtime_smoke_ok=False,
        runtime_smoke_detail=detail,
        platform_name=platform.system(),
    )


def _emit_receipt(
    payload: dict[str, object],
    *,
    json_output: bool,
) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    evaluation = payload["evaluation"]
    assert isinstance(evaluation, dict)
    selected = payload["selected"]
    assert isinstance(selected, dict)
    probe = payload["probe"]
    assert isinstance(probe, dict)
    print(f"mode: {evaluation['mode']}")
    print(f"python: {selected['python']}")
    print(f"prefix: {probe['python_prefix']}")
    print(f"mlx module: {probe['module_file']}")
    print(f"mlx distribution: {probe['distribution_root']}")
    print(f"mlx package: {probe['package_version']}")
    print(f"mlx-metal: {probe['mlx_metal_version']}")
    print(f"libmlx: {probe['loaded_libmlx']}")
    messages = evaluation["messages"]
    assert isinstance(messages, (list, tuple))
    for message in messages:
        print(message)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Preserve a virtualenv's bin/python symlink. Resolving it to the base
    # interpreter discards pyvenv.cfg discovery and probes the wrong package
    # set.
    env_root = Path(os.path.abspath(args.env_root.expanduser()))
    selected_python = args.python or env_root / "bin" / "python"
    python = Path(os.path.abspath(selected_python.expanduser()))
    source_root = (
        args.source_root.expanduser().resolve()
        if args.source_root is not None
        else None
    )
    if not python.is_file() or not os.access(python, os.X_OK):
        detail = f"Python executable is unavailable: {python}"
        probe = _unavailable_probe(
            python=python,
            env_root=env_root,
            source_root=source_root,
            detail=detail,
        )
        evaluation = AbiEvaluation(False, "unavailable", (f"FAIL: {detail}",))
        _emit_receipt(
            _receipt(
                probe=probe,
                evaluation=evaluation,
                python=python,
                env_root=env_root,
                source_root=source_root,
            ),
            json_output=args.json,
        )
        if not args.json:
            print(f"FAIL: {detail}", file=sys.stderr)
        return 2
    try:
        probe = _probe_runtime(python, source_root=source_root)
        evaluation = evaluate_probe(
            probe,
            expected_python=python,
            expected_env_root=env_root,
        )
    except RuntimeError as exc:
        detail = str(exc)
        probe = _unavailable_probe(
            python=python,
            env_root=env_root,
            source_root=source_root,
            detail=detail,
        )
        evaluation = AbiEvaluation(False, "unavailable", (f"FAIL: {detail}",))
        _emit_receipt(
            _receipt(
                probe=probe,
                evaluation=evaluation,
                python=python,
                env_root=env_root,
                source_root=source_root,
            ),
            json_output=args.json,
        )
        if not args.json:
            print(f"FAIL: {detail}", file=sys.stderr)
        return 1

    payload = _receipt(
        probe=probe,
        evaluation=evaluation,
        python=python,
        env_root=env_root,
        source_root=source_root,
    )
    _emit_receipt(payload, json_output=args.json)
    return 0 if evaluation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
