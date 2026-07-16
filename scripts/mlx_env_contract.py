#!/usr/bin/env python3
"""Build hermetic MLX wheel and TileLang source environments.

The project checkout has a shared ``.venv`` symlink and developers commonly
have a different TileLang/TVM stack exported in their shell.  This module
keeps the two supported runtime contracts explicit:

``mlx-wheel``
    The selected interpreter imports the installed MLX wheel and no source
    runtime paths are inherited.

``path-c-source``
    MLX still comes from that installed wheel, while TileLang, TVM, and
    TVM-FFI are resolved from one checked-out sibling source stack.

No package manager is invoked here.  The returned mapping is suitable for a
probe subprocess or for ``os.execve`` by the Path-C launcher.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_ROOT = ROOT.parent / ".venvs" / "cppmega.mlx"
DEFAULT_TILELANG_ROOT = ROOT.parent / "tilelang"

_EXACT_CONTRACT_VARS = frozenset(
    {
        "CC",
        "CFLAGS",
        "CMAKE_PREFIX_PATH",
        "CPATH",
        "CPPFLAGS",
        "CPPMEGA_MLX_ENV_MODE",
        "CPPMEGA_MLX_ENV_ROOT",
        "CPPMEGA_MLX_PYTHON",
        "CPPMEGA_MLX_SOURCE_ROOT",
        "CPPMEGA_TILELANG_DEV_ROOT",
        "CPPMEGA_TILELANG_SOURCE_ROOT",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "CPLUS_INCLUDE_PATH",
        "CXX",
        "CXXFLAGS",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "LDFLAGS",
        "PYTHONHOME",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "PYTHONSAFEPATH",
        "PYTHONUSERBASE",
        "TILELANG_DEV_BUILD_ROOT",
        "TILELANG_DISABLE_CACHE",
        "TILELANG_ROOT",
        "TVM_FFI_DLPACK_INCLUDE_PATH",
        "TVM_FFI_INCLUDE_PATH",
        "TVM_HOME",
        "TVM_IMPORT_PYTHON_PATH",
        "TVM_LIBRARY_PATH",
        "TVM_LIBRARY_PATH_SELECTED",
        "TVM_ROOT",
        "TVM_SOURCE_DIR",
        "VIRTUAL_ENV",
        "VIRTUAL_ENV_PROMPT",
    }
)
_CONTRACT_PREFIXES = (
    "CMAKE_",
    "CONDA",
    "CPPMEGA_MLX_",
    "CPPMEGA_TILELANG_",
    "DYLD_",
    "LD_",
    "MLX_",
    "MTL_",
    "PIP_",
    "TL_",
    "TILELANG",
    "TVM_",
    "UV_",
)


def default_python(repo_root: Path = ROOT) -> Path:
    return repo_root.parent / ".venvs" / "cppmega.mlx" / "bin" / "python"


def default_tilelang_root(repo_root: Path = ROOT) -> Path:
    return repo_root.parent / "tilelang"


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _is_contract_var(name: str) -> bool:
    return name in _EXACT_CONTRACT_VARS or name.startswith(_CONTRACT_PREFIXES)


def _looks_like_virtualenv_bin(path: str) -> bool:
    try:
        parts = Path(path).parts
    except (OSError, TypeError):
        return False
    return any(
        part in {".venv", ".venvs", "venv", ".tox", "virtualenv"}
        for part in parts
    )


def _clean_path(python: Path, *, homebrew_prefix: str | None) -> str:
    values = [str(python.parent)]
    if homebrew_prefix:
        values.extend(
            (
                str(Path(homebrew_prefix).expanduser() / "bin"),
                str(Path(homebrew_prefix).expanduser() / "sbin"),
            )
        )
    values.extend(("/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin"))
    values.extend(value for value in os.defpath.split(os.pathsep) if value)
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return os.pathsep.join(result)


def _base_environment(
    environ: Mapping[str, str],
    *,
    python: Path,
) -> dict[str, str]:
    selected_python = _absolute(python)
    environment = {
        name: value for name, value in environ.items() if not _is_contract_var(name)
    }
    environment["PATH"] = _clean_path(
        selected_python,
        homebrew_prefix=environ.get("HOMEBREW_PREFIX"),
    )
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    environment["VIRTUAL_ENV"] = str(selected_python.parent.parent)
    environment["CPPMEGA_MLX_ENV_ROOT"] = str(selected_python.parent.parent)
    environment["CPPMEGA_MLX_PYTHON"] = str(selected_python)
    return environment


def build_mlx_wheel_environment(
    environ: Mapping[str, str] | None = None,
    *,
    python: Path,
) -> dict[str, str]:
    """Return an environment that can only select the MLX wheel contract."""

    source_environment = os.environ if environ is None else environ
    environment = _base_environment(source_environment, python=python)
    environment["CPPMEGA_MLX_ENV_MODE"] = "mlx-wheel"
    return environment


def _has_library(directory: Path, stem: str) -> bool:
    return any(
        (directory / f"{stem}{suffix}").is_file()
        for suffix in (".dylib", ".so", ".so.0")
    )


def _select_build_root(tilelang_root: Path) -> Path:
    candidates = (
        tilelang_root / "build",
        tilelang_root / "build-metal-m4-codex",
        tilelang_root / "build-appleclang-triton-llvm",
    )
    for candidate in candidates:
        lib = candidate / "lib"
        if _has_library(lib, "libtilelang") and _has_library(lib, "libtvm_runtime"):
            return candidate
    rendered = ", ".join(str(path / "lib") for path in candidates)
    raise ValueError(f"no usable TileLang source build found; checked {rendered}")


def _unique_existing(paths: tuple[Path, ...]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = _absolute(path)
        if resolved.is_dir() and resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def build_path_c_environment(
    environ: Mapping[str, str] | None = None,
    *,
    repo_root: Path = ROOT,
    python: Path,
    tilelang_root: Path,
) -> dict[str, str]:
    """Return the explicit MLX-wheel + pinned TileLang-source contract."""

    repo = _absolute(repo_root)
    tilelang = _absolute(tilelang_root)
    tvm = tilelang / "3rdparty" / "tvm"
    tvm_ffi = tvm / "3rdparty" / "tvm-ffi"
    required = (
        tilelang / "tilelang",
        tvm / "python",
        tvm_ffi / "python",
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise ValueError("TileLang source stack is incomplete: " + ", ".join(missing))

    build_root = _select_build_root(tilelang)
    build_lib = build_root / "lib"
    library_paths = _unique_existing(
        (
            build_lib,
            build_root / "tvm",
            tvm / "build" / "lib",
            tvm / "build",
            tvm_ffi / "build" / "lib",
        )
    )
    if not library_paths:
        raise ValueError(f"TileLang source build has no library directory: {build_root}")

    source_environment = os.environ if environ is None else environ
    environment = _base_environment(source_environment, python=python)
    python_paths = (
        repo,
        tilelang,
        tvm / "python",
        tvm_ffi / "python",
    )
    environment.update(
        {
            "CPPMEGA_MLX_ENV_MODE": "path-c-source",
            "CPPMEGA_TILELANG_SOURCE_ROOT": str(tilelang),
            "PYTHONPATH": os.pathsep.join(str(path) for path in python_paths),
            "TILELANG_ROOT": str(tilelang),
            "TILELANG_DEV_BUILD_ROOT": str(build_root),
            "TILELANG_DISABLE_CACHE": "1",
            "TVM_ROOT": str(tvm),
            "TVM_HOME": str(tvm),
            "TVM_SOURCE_DIR": str(tvm),
            "TVM_IMPORT_PYTHON_PATH": str(tvm / "python"),
            "TVM_LIBRARY_PATH": os.pathsep.join(str(path) for path in library_paths),
            "TVM_LIBRARY_PATH_SELECTED": os.pathsep.join(
                str(path) for path in library_paths
            ),
            "DYLD_LIBRARY_PATH": os.pathsep.join(str(path) for path in library_paths),
        }
    )
    if (tvm_ffi / "include").is_dir():
        environment["TVM_FFI_INCLUDE_PATH"] = str(tvm_ffi / "include")
    dlpack_include = tvm_ffi / "3rdparty" / "dlpack" / "include"
    if dlpack_include.is_dir():
        environment["TVM_FFI_DLPACK_INCLUDE_PATH"] = str(dlpack_include)
    return environment
