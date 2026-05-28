"""Optional local dependencies for Triton Path D probes.

Path D is a development bridge: cppmega_v4 owns the dispatch surface, while
the Triton frontend and FLA kernels live in sibling checkouts on dev hosts.
This module keeps those lookup rules explicit and cheap.
"""

from __future__ import annotations

import os
import sys
import importlib
import importlib.util
import subprocess
from pathlib import Path
from functools import lru_cache
from types import ModuleType

TRITON_FRONTEND_PATH_ENV = "CPPMEGA_MLX_TRITON_FRONTEND_PATH"
FLA_SOURCE_PATH_ENV = "CPPMEGA_MLX_FLA_SOURCE_PATH"
TRITON_FRONTEND_UNSAFE_IMPORT_ENV = "CPPMEGA_V4_ENABLE_UNSAFE_TRITON_IMPORT"
PATH_D_NATIVE_IMPORT_PROBE_ENV = "CPPMEGA_V4_PATH_D_NATIVE_IMPORT_PROBE"
PATH_D_NATIVE_IMPORT_PROBE_TIMEOUT_ENV = "CPPMEGA_V4_PATH_D_NATIVE_IMPORT_PROBE_TIMEOUT"

_TRITON_NATIVE_MODULES = ("triton._C", "triton._C.libtriton")
_TILELANG_LLVM_PEER_MODULES = (
    "tilelang",
    "tilelang_cython_wrapper",
    "tvm",
    "tvm.base",
)
_LLVM_PEER_MODULES = (
    "_triton_frontend_cxx",
    "jaxlib.mlir._mlir_libs",
    "jaxlib.mlir",
    *_TILELANG_LLVM_PEER_MODULES,
)


def _prepend_existing(root: Path) -> bool:
    if not root.exists():
        return False
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_candidates(name: str) -> list[Path]:
    raw = os.environ.get(name)
    if not raw:
        return []
    return [Path(part).expanduser() for part in raw.split(os.pathsep) if part]


def _workspace_candidates(*relative_roots: str) -> list[Path]:
    repo = _repo_root()
    candidates: list[Path] = []
    for base in (repo.parent, *repo.parents):
        for rel in relative_roots:
            candidates.append(base / rel)
    return candidates


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_falsey(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


def _package_root_from_spec(module_name: str, package_depth: int) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    if spec.origin is None or spec.origin == "namespace":
        locations = spec.submodule_search_locations or ()
        first = next(iter(locations), None)
        return Path(first).resolve() if first else None
    origin = Path(spec.origin).resolve()
    try:
        return origin.parents[package_depth]
    except IndexError:
        return None


def _root_for_marker(candidate: Path, marker_parts: tuple[str, ...]) -> Path | None:
    candidate = candidate.expanduser()
    if candidate.joinpath(*marker_parts).exists():
        return candidate
    if tuple(candidate.parts[-len(marker_parts):]) == marker_parts:
        try:
            return candidate.parents[len(marker_parts) - 1]
        except IndexError:
            return None
    return None


def _ensure_root(
    *,
    env_name: str,
    module_name: str,
    package_depth: int,
    marker_parts: tuple[str, ...],
    workspace_relatives: tuple[str, ...],
) -> str | None:
    candidates = _env_candidates(env_name)
    candidates.extend(_workspace_candidates(*workspace_relatives))
    spec_root = _package_root_from_spec(module_name, package_depth)
    if spec_root is not None:
        candidates.append(spec_root)

    for candidate in _dedupe(candidates):
        root = _root_for_marker(candidate, marker_parts)
        if root is None:
            continue
        _prepend_existing(root)
        return str(root)
    return None


def ensure_triton_frontend_root() -> str | None:
    """Make ``poc.triton_frontend`` importable from env/workspace/install."""

    return _ensure_root(
        env_name=TRITON_FRONTEND_PATH_ENV,
        module_name="poc.triton_frontend",
        package_depth=2,
        marker_parts=("poc", "triton_frontend"),
        workspace_relatives=("tilelang", "tl_poc_review"),
    )


def ensure_fla_root() -> str | None:
    """Make ``fla`` importable from env/workspace/install."""

    return _ensure_root(
        env_name=FLA_SOURCE_PATH_ENV,
        module_name="fla",
        package_depth=1,
        marker_parts=("fla",),
        workspace_relatives=(
            "rent_kernels/flash-linear-attention",
            "flash-linear-attention",
        ),
    )


def unsafe_triton_frontend_import_enabled() -> bool:
    """Return True only when the caller explicitly accepts unsafe Triton imports.

    Some local Triton checkouts abort the Python process during import instead
    of raising a Python exception. Path D status probes run in normal test and
    benchmark discovery, so they must fail closed unless the developer opts in.
    """

    return _env_truthy(TRITON_FRONTEND_UNSAFE_IMPORT_ENV)


def _triton_native_loaded() -> bool:
    return any(name in sys.modules for name in _TRITON_NATIVE_MODULES)


def _triton_native_symbols_are_local() -> bool:
    triton_module = sys.modules.get("triton")
    return bool(getattr(triton_module, "_NATIVE_DLOPEN_LOCAL", False))


def loaded_path_d_llvm_peer_modules() -> list[str]:
    """Return loaded native peers that can collide with Triton's LLVM image."""

    return [name for name in _LLVM_PEER_MODULES if name in sys.modules]


def path_d_native_import_block_reason() -> str | None:
    """Return why importing Triton/FLA in this interpreter is unsafe now."""

    if _triton_native_loaded() and _triton_native_symbols_are_local():
        return None
    peers = loaded_path_d_llvm_peer_modules()
    if sys.platform == "darwin" and (
        _triton_native_symbols_are_local() or not _triton_native_loaded()
    ):
        peers = [name for name in peers if name not in _TILELANG_LLVM_PEER_MODULES]
    if not peers:
        return None
    return (
        "Path D native imports blocked in this Python process because "
        + ", ".join(peers)
        + " already loaded an LLVM/MLIR native image; importing or calling "
        "triton._C.libtriton here can abort on duplicate LLVM cl::opt "
        "registration. Re-run this Path D check in a fresh Python process."
    )


def import_triton_with_local_symbols() -> ModuleType:
    """Import Triton without exporting its LLVM symbols process-wide.

    Local Triton builds carry a static LLVM image inside ``libtriton``. On
    Darwin, importing that extension with the interpreter's default dlopen
    flags can make LLVM symbols visible to later TileLang/TVM loads, which
    can abort the process during duplicate LLVM command-line option
    registration. ``RTLD_LOCAL`` keeps the native image isolated while still
    returning the normal Python module.
    """

    if not hasattr(sys, "getdlopenflags") or not hasattr(sys, "setdlopenflags"):
        triton = importlib.import_module("triton")
        setattr(triton, "_NATIVE_DLOPEN_LOCAL", False)
        return triton

    old_flags = sys.getdlopenflags()
    local_flags = getattr(os, "RTLD_NOW", old_flags) | getattr(os, "RTLD_LOCAL", 0)
    sys.setdlopenflags(local_flags)
    try:
        triton = importlib.import_module("triton")
        setattr(triton, "_NATIVE_DLOPEN_LOCAL", True)
        return triton
    finally:
        sys.setdlopenflags(old_flags)


def _native_probe_timeout() -> float:
    raw = os.environ.get(PATH_D_NATIVE_IMPORT_PROBE_TIMEOUT_ENV, "").strip()
    if not raw:
        return 30.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def _probe_pythonpath(*roots: str | None) -> str:
    entries = [str(_repo_root())]
    entries.extend(str(root) for root in roots if root)
    current = os.environ.get("PYTHONPATH")
    if current:
        entries.extend(part for part in current.split(os.pathsep) if part)
    return os.pathsep.join(str(path) for path in _dedupe([Path(entry) for entry in entries]))


@lru_cache(maxsize=8)
def path_d_native_import_preflight(mode: str = "frontend") -> tuple[bool, str]:
    """Crash-safe subprocess preflight for Path D native imports.

    Native LLVM duplicate registration aborts the interpreter, so the probe
    must run in a child process. A successful child proves the current checkout
    can import Triton with local symbols and then import the requested frontend
    / FLA module set without taking SIGABRT.
    """

    if _env_falsey(PATH_D_NATIVE_IMPORT_PROBE_ENV):
        return True, f"Path D native import subprocess preflight disabled by {PATH_D_NATIVE_IMPORT_PROBE_ENV}=0"

    triton_root = ensure_triton_frontend_root()
    fla_root = ensure_fla_root() if mode in {"gdn_fla", "kda_fla"} else None
    if triton_root is None:
        return False, "Path D native import preflight failed: poc.triton_frontend root not found"
    if mode in {"gdn_fla", "kda_fla"} and fla_root is None:
        return False, "Path D native import preflight failed: FLA source root not found"

    child_code = r"""
import sys
from cppmega_v4._tilelang._path_d_deps import (
    ensure_fla_root,
    ensure_triton_frontend_root,
    import_triton_with_local_symbols,
    path_d_native_import_block_reason,
)

mode = sys.argv[1]
ensure_triton_frontend_root()
if mode in {"gdn_fla", "kda_fla"}:
    ensure_fla_root()
block_reason = path_d_native_import_block_reason()
if block_reason is not None:
    raise SystemExit(block_reason)
import_triton_with_local_symbols()
from poc.triton_frontend import from_triton_kernel  # noqa: F401
if mode == "gdn_fla":
    from fla.ops.common.chunk_delta_h import (  # noqa: F401
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )
elif mode == "kda_fla":
    from fla.ops.kda.chunk import chunk_kda  # noqa: F401
print("ok")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = _probe_pythonpath(triton_root, fla_root)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", child_code, mode],
            cwd=str(_repo_root()),
            env=env,
            capture_output=True,
            text=True,
            timeout=_native_probe_timeout(),
        )
    except subprocess.TimeoutExpired as exc:
        return False, (
            "Path D native import preflight timed out after "
            f"{exc.timeout:g}s for mode={mode}"
        )
    if completed.returncode == 0:
        return True, f"Path D native import preflight passed for mode={mode}"
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    detail = stderr or stdout or "<no output>"
    return False, (
        "Path D native import preflight failed for "
        f"mode={mode}: exit={completed.returncode}; {detail[:1000]}"
    )


def path_d_imports_allowed(mode: str = "frontend") -> tuple[bool, str]:
    """Return whether Path D may import native Triton/FLA in this process."""

    block_reason = path_d_native_import_block_reason()
    if block_reason is not None:
        return False, block_reason
    if not unsafe_triton_frontend_import_enabled():
        if mode in {"gdn_fla", "kda_fla"}:
            return False, unsafe_fla_import_disabled_reason(None)
        return False, unsafe_triton_frontend_import_disabled_reason(None)
    ok, reason = path_d_native_import_preflight(mode)
    if not ok:
        return False, reason
    return True, (
        "Path D native imports explicitly enabled by "
        f"{TRITON_FRONTEND_UNSAFE_IMPORT_ENV}=1; {reason}"
    )


def unsafe_triton_frontend_import_disabled_reason(root: str | None) -> str:
    root_text = f" root={root}" if root else " root=<not found>"
    return (
        "unsafe triton frontend import disabled (not importable by default) because "
        f"{TRITON_FRONTEND_UNSAFE_IMPORT_ENV}=1 is not set; native import "
        "preflight did not authorize in-process import; Path D runtime "
        f"adapter not reached;{root_text}"
    )


def unsafe_fla_import_disabled_reason(root: str | None) -> str:
    root_text = f" root={root}" if root else " root=<not found>"
    return (
        "unsafe FLA import disabled (not importable by default) because "
        f"{TRITON_FRONTEND_UNSAFE_IMPORT_ENV}=1 is not set; native import "
        "preflight did not authorize in-process import; Path D runtime "
        f"adapter not reached;{root_text}"
    )


__all__ = [
    "FLA_SOURCE_PATH_ENV",
    "PATH_D_NATIVE_IMPORT_PROBE_ENV",
    "PATH_D_NATIVE_IMPORT_PROBE_TIMEOUT_ENV",
    "TRITON_FRONTEND_PATH_ENV",
    "TRITON_FRONTEND_UNSAFE_IMPORT_ENV",
    "ensure_fla_root",
    "ensure_triton_frontend_root",
    "import_triton_with_local_symbols",
    "loaded_path_d_llvm_peer_modules",
    "path_d_imports_allowed",
    "path_d_native_import_block_reason",
    "path_d_native_import_preflight",
    "unsafe_fla_import_disabled_reason",
    "unsafe_triton_frontend_import_disabled_reason",
    "unsafe_triton_frontend_import_enabled",
]
