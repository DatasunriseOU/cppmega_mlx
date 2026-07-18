#!/usr/bin/env python3
"""Verify and launch Path-C with MLX wheels plus source TileLang."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]


def _load_environment_contract() -> Any:
    """Load the repository-owned contract without consulting ``PYTHONPATH``."""

    path = ROOT / "scripts" / "mlx_env_contract.py"
    spec = importlib.util.spec_from_file_location(
        "_cppmega_mlx_environment_contract",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MLX environment contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ENVIRONMENT_CONTRACT = _load_environment_contract()
build_path_c_environment = _ENVIRONMENT_CONTRACT.build_path_c_environment
default_python = _ENVIRONMENT_CONTRACT.default_python
default_tilelang_root = _ENVIRONMENT_CONTRACT.default_tilelang_root
CONTRACT_MODE = "path_c_source"
RECEIPT_KIND = "cppmega_mlx_path_c_environment_receipt"
RECEIPT_SCHEMA_VERSION = 1
_PROBE_MARKER = "__CPPMEGA_PATH_C_PROBE__="
_RELEASE_RE = re.compile(r"^(\d+\.\d+\.\d+)")
_PROBE_TIMEOUT_SECONDS = 120
_PROBE_RECEIPT_FIELDS = (
    "platform",
    "python_executable",
    "python_prefix",
    "mlx_module_file",
    "mlx_module_version",
    "mlx_package_version",
    "mlx_distribution_root",
    "mlx_metal_version",
    "mlx_runtime_smoke_ok",
    "mlx_loaded_library",
    "tilelang_module_file",
    "tilelang_module_version",
    "tilelang_distribution_version",
    "tvm_module_file",
    "tvm_ffi_module_file",
    "tvm_ffi_core_file",
    "tilelang_lower_callable",
    "metal_target_ok",
    "source_runtime_libraries",
)
_SOURCE_COMPONENTS = ("tilelang", "tvm", "tvm_ffi")


@dataclass(frozen=True)
class PathCProbeEvaluation:
    ok: bool
    mode: str
    mlx_mode: str
    tilelang_mode: str
    messages: tuple[str, ...]


def _inside(path: object, root: Path) -> bool:
    if not isinstance(path, (str, os.PathLike)):
        return False
    try:
        Path(path).resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _same_path(left: object, right: Path) -> bool:
    if not isinstance(left, (str, os.PathLike)):
        return False
    try:
        return Path(left).expanduser().resolve() == right.expanduser().resolve()
    except OSError:
        return False


def _release(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = _RELEASE_RE.match(value.strip())
    return match.group(1) if match else None


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _source_stack_messages(
    source_stack: Mapping[str, object],
    *,
    tilelang_root: Path,
    allow_dirty_source_stack: bool,
) -> list[str]:
    tvm_root = tilelang_root / "3rdparty" / "tvm"
    expected_source_roots = {
        "tilelang": tilelang_root,
        "tvm": tvm_root,
        "tvm_ffi": tvm_root / "3rdparty" / "tvm-ffi",
    }
    messages: list[str] = []
    for component, expected_root in expected_source_roots.items():
        state = source_stack.get(component)
        if not isinstance(state, Mapping):
            messages.append(f"FAIL: {component} source Git state is unavailable")
            continue
        if not _same_path(state.get("root"), expected_root):
            messages.append(
                f"FAIL: {component} source Git root differs from selected stack: "
                f"{state.get('root')}"
            )
        if not isinstance(state.get("revision"), str) or not state.get("revision"):
            messages.append(f"FAIL: {component} source Git revision is unavailable")
        if not _valid_sha256(state.get("worktree_digest")):
            messages.append(
                f"FAIL: {component} source worktree digest is unavailable"
            )
        dirty = state.get("dirty")
        if not isinstance(dirty, bool):
            messages.append(f"FAIL: {component} source dirty state is unavailable")
        elif dirty and not allow_dirty_source_stack:
            messages.append(f"FAIL: dirty {component} source tree is not allowed")
        elif dirty:
            messages.append(
                f"dirty {component} source tree was explicitly allowed by launcher flag"
            )
    return messages


def evaluate_path_c_probe(
    probe: Mapping[str, object],
    *,
    python: Path,
    tilelang_root: Path,
    source_stack: Mapping[str, object],
    allow_dirty_source_stack: bool = False,
) -> PathCProbeEvaluation:
    """Evaluate flat probe data without importing MLX in the launcher process."""

    # Keep the virtualenv's lexical bin/python path. Resolving the symlink
    # would turn the selected venv into the base Homebrew interpreter.
    python = Path(python).expanduser().absolute()
    tilelang_root = Path(tilelang_root).expanduser().resolve()
    tvm_root = tilelang_root / "3rdparty" / "tvm"
    ffi_root = tvm_root / "3rdparty" / "tvm-ffi"
    env_root = python.parent.parent.resolve()
    messages: list[str] = []

    if probe.get("platform") != "Darwin":
        messages.append(
            "FAIL: Path-C currently requires Darwin/Metal; refusing an unproven "
            f"platform: {probe.get('platform')}"
        )

    messages.extend(
        _source_stack_messages(
            source_stack,
            tilelang_root=tilelang_root,
            allow_dirty_source_stack=allow_dirty_source_stack,
        )
    )

    if not _same_path(probe.get("python_executable"), python):
        messages.append(
            "FAIL: runtime Python differs from the selected interpreter: "
            f"runtime={probe.get('python_executable')} selected={python}"
        )
    if not _same_path(probe.get("python_prefix"), env_root):
        messages.append(
            "FAIL: runtime Python prefix differs from the dedicated environment: "
            f"runtime={probe.get('python_prefix')} selected={env_root}"
        )

    mlx_file = probe.get("mlx_module_file")
    mlx_dist = probe.get("mlx_distribution_root")
    mlx_dist_path = Path(mlx_dist) if isinstance(mlx_dist, (str, os.PathLike)) else None
    mlx_mode = (
        "installed_wheel"
        if _inside(mlx_file, env_root)
        and mlx_dist_path is not None
        and _inside(mlx_file, mlx_dist_path)
        else "workspace_source"
    )
    if mlx_mode != "installed_wheel":
        messages.append(
            "FAIL: MLX must resolve from the dedicated installed wheel: "
            f"module={mlx_file} distribution={mlx_dist}"
        )
    if mlx_dist_path is None or not _inside(mlx_dist_path, env_root):
        messages.append(
            "FAIL: MLX distribution is outside the dedicated environment: "
            f"{mlx_dist}"
        )
    for field in ("mlx_module_version", "mlx_package_version", "mlx_metal_version"):
        if not isinstance(probe.get(field), str) or not probe.get(field):
            messages.append(f"FAIL: {field} is unavailable")
    loaded_mlx = probe.get("mlx_loaded_library")
    if loaded_mlx is not None and not _inside(loaded_mlx, env_root):
        messages.append(
            "FAIL: loaded libmlx is outside the dedicated wheel environment: "
            f"{loaded_mlx}"
        )
    if not bool(probe.get("mlx_runtime_smoke_ok")):
        messages.append("FAIL: MLX runtime smoke failed")
    if probe.get("platform") == "Darwin" and probe.get("mlx_loaded_library") is None:
        messages.append("FAIL: loaded libmlx origin is unavailable")
    elif probe.get("mlx_loaded_library") is not None and not _inside(
        probe.get("mlx_loaded_library"), env_root
    ):
        messages.append(
            "FAIL: loaded libmlx is outside the dedicated wheel environment: "
            f"{probe.get('mlx_loaded_library')}"
        )
    mlx_release = _release(probe.get("mlx_package_version") or probe.get("mlx_module_version"))
    metal_release = _release(probe.get("mlx_metal_version"))
    if probe.get("platform") == "Darwin" and metal_release is None:
        messages.append("FAIL: mlx-metal release is unavailable")
    elif metal_release is not None and mlx_release != metal_release:
        messages.append(
            "FAIL: mlx/metal release contract differs: "
            f"mlx={probe.get('mlx_package_version') or probe.get('mlx_module_version')} "
            f"mlx-metal={probe.get('mlx_metal_version')}"
        )

    tilelang_file = probe.get("tilelang_module_file")
    tilelang_mode = (
        "workspace_source"
        if _inside(tilelang_file, tilelang_root)
        else "installed_wheel"
    )
    if tilelang_mode != "workspace_source":
        messages.append(
            "FAIL: Path-C requires TileLang source mode: "
            f"module={tilelang_file} root={tilelang_root}"
        )
    if not _inside(probe.get("tvm_module_file"), tvm_root / "python"):
        messages.append(f"FAIL: TVM did not resolve from source: {probe.get('tvm_module_file')}")
    if not _inside(probe.get("tvm_ffi_module_file"), ffi_root / "python"):
        messages.append(
            f"FAIL: TVM-FFI Python did not resolve from source: {probe.get('tvm_ffi_module_file')}"
        )
    if not _inside(probe.get("tvm_ffi_core_file"), ffi_root / "build"):
        messages.append(
            "FAIL: TVM-FFI native extension did not resolve from source: "
            f"{probe.get('tvm_ffi_core_file')}"
        )
    if probe.get("tilelang_lower_callable") is not True:
        messages.append("FAIL: tilelang.engine.lower is unavailable")
    if probe.get("metal_target_ok") is not True:
        messages.append("FAIL: TVM Metal target construction failed")

    tilelang_release = _release(probe.get("tilelang_module_version"))
    tilelang_distribution_release = _release(probe.get("tilelang_distribution_version"))
    if tilelang_release is None:
        messages.append("FAIL: TileLang source module release is unavailable")
    if tilelang_distribution_release is None:
        messages.append("FAIL: TileLang distribution release is unavailable")
    if (
        tilelang_release is not None
        and tilelang_distribution_release is not None
        and tilelang_release != tilelang_distribution_release
    ):
        messages.append(
            "FAIL: TileLang source/wheel release contract differs: "
            f"source={probe.get('tilelang_module_version')} "
            f"wheel={probe.get('tilelang_distribution_version')}"
        )

    source_libraries = probe.get("source_runtime_libraries")
    if not isinstance(source_libraries, list):
        messages.append(
            "FAIL: loaded TileLang/TVM source runtime library receipt is unavailable"
        )
    else:
        for library in source_libraries:
            if not _inside(library, tilelang_root):
                messages.append(
                    "FAIL: loaded TileLang/TVM library is outside source root: "
                    f"{library}"
                )
        if probe.get("platform") == "Darwin":
            library_names = {Path(str(path)).name for path in source_libraries}
            required_names = {
                "libtilelang.dylib",
                "libtvm_runtime.dylib",
                "libtvm_ffi.dylib",
            }
            missing_names = sorted(required_names - library_names)
            if missing_names:
                messages.append(
                    "FAIL: selected source stack did not load required libraries: "
                    + ", ".join(missing_names)
                )

    if not messages:
        messages.extend(
            (
                "MLX mode: installed wheel from dedicated environment",
                "TileLang mode: workspace source with vendored TVM/TVM-FFI",
            )
        )
    return PathCProbeEvaluation(
        ok=not any(message.startswith("FAIL:") for message in messages),
        mode=CONTRACT_MODE,
        mlx_mode=mlx_mode,
        tilelang_mode=tilelang_mode,
        messages=tuple(messages),
    )


_PROBE_CODE = r'''
import ctypes
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import sys


def module_file(module):
    value = getattr(module, "__file__", None)
    return str(Path(value).resolve()) if value else None


def distribution(name):
    try:
        return md.distribution(name)
    except md.PackageNotFoundError:
        return None


def loaded_images():
    if sys.platform != "darwin":
        return []
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    images = []
    for index in range(dyld._dyld_image_count()):
        raw = dyld._dyld_get_image_name(index)
        if raw:
            images.append(str(Path(raw.decode()).resolve()))
    return images


result = {
    "platform": platform.system(),
    "python_executable": str(Path(sys.executable)),
    "python_prefix": str(Path(sys.prefix).resolve()),
}
try:
    import mlx.core as mx
    import tilelang
    import tilelang.engine
    import tvm
    import tvm_ffi
    import tvm_ffi.core

    value = mx.array([1, 2], dtype=mx.int32) + 1
    mx.eval(value)
    synchronize = getattr(mx, "synchronize", None)
    if synchronize is not None:
        synchronize()
    images = loaded_images()
    mlx_dist = distribution("mlx")
    tilelang_dist = distribution("tilelang")
    mlx_file = module_file(mx)
    tilelang_file = module_file(tilelang)
    env_root = os.environ.get("CPPMEGA_MLX_ENV_ROOT")
    result.update(
        {
            "mlx_module_file": mlx_file,
            "mlx_module_version": getattr(mx, "__version__", None),
            "mlx_package_version": mlx_dist.version if mlx_dist else None,
            "mlx_distribution_root": (
                str(Path(mlx_dist.locate_file("")).resolve()) if mlx_dist else None
            ),
            "mlx_metal_version": (
                distribution("mlx-metal").version
                if distribution("mlx-metal")
                else None
            ),
            "mlx_runtime_smoke_ok": value.tolist() == [2, 3],
            "mlx_loaded_library": next(
                (path for path in images if Path(path).name == "libmlx.dylib"),
                None,
            ),
            "tilelang_module_file": tilelang_file,
            "tilelang_module_version": getattr(tilelang, "__version__", None),
            "tilelang_distribution_version": tilelang_dist.version if tilelang_dist else None,
            "tvm_module_file": module_file(tvm),
            "tvm_ffi_module_file": module_file(tvm_ffi),
            "tvm_ffi_core_file": module_file(tvm_ffi.core),
            "tilelang_lower_callable": callable(getattr(tilelang.engine, "lower", None)),
            "metal_target_ok": True,
            "source_runtime_libraries": [
                path for path in images
                if Path(path).name.startswith(("libtilelang", "libtvm", "libz3"))
            ],
        }
    )
    if sys.platform == "darwin":
        tvm.target.Target("metal")
except Exception as exc:
    result["probe_error"] = f"{type(exc).__name__}: {exc}"
print("__CPPMEGA_PATH_C_PROBE__=" + json.dumps(result, sort_keys=True))
'''


def run_probe(python: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    try:
        process = subprocess.run(
            [str(python), "-c", _PROBE_CODE],
            cwd=ROOT,
            env=dict(environment),
            capture_output=True,
            text=True,
            check=False,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Path-C probe exceeded {_PROBE_TIMEOUT_SECONDS}s and was terminated"
        ) from exc
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "unknown failure"
        raise RuntimeError(f"Path-C probe exited {process.returncode}: {detail}")
    line = next(
        (
            candidate
            for candidate in reversed(process.stdout.splitlines())
            if candidate.startswith(_PROBE_MARKER)
        ),
        None,
    )
    if line is None:
        raise RuntimeError(
            f"Path-C probe returned no receipt: stdout={process.stdout!r} stderr={process.stderr!r}"
        )
    payload = json.loads(line.removeprefix(_PROBE_MARKER))
    if not isinstance(payload, dict):
        raise RuntimeError("Path-C probe receipt is not a JSON object")
    if payload.get("probe_error"):
        raise RuntimeError(str(payload["probe_error"]))
    return payload


def sanitize_launch_environment(
    environment: Mapping[str, str], *, python: Path
) -> dict[str, str]:
    """Remove runtime/compiler tuning inherited from the caller shell."""

    clean = dict(environment)
    blocked_prefixes = ("CMAKE_", "MLX_", "MTL_", "PIP_", "UV_")
    blocked_names = {
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "CPPFLAGS",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "PYTHONHOME",
        "PYTHONUSERBASE",
    }
    for name in list(clean):
        if name.startswith(blocked_prefixes) or name in blocked_names:
            clean.pop(name, None)

    path_entries = (
        Path(python).parent,
        Path("/opt/homebrew/bin"),
        Path("/opt/homebrew/sbin"),
        Path("/usr/local/bin"),
        Path("/usr/bin"),
        Path("/bin"),
        Path("/usr/sbin"),
        Path("/sbin"),
    )
    clean["PATH"] = os.pathsep.join(dict.fromkeys(str(path) for path in path_entries))
    clean["PYTHONNOUSERSITE"] = "1"
    clean["PYTHONSAFEPATH"] = "1"
    return clean


def _worktree_digest(
    root: Path,
    environment: Mapping[str, str],
    status_text: str,
) -> str | None:
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD", "--"],
        cwd=root,
        env=dict(environment),
        capture_output=True,
        check=False,
    )
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=root,
        env=dict(environment),
        capture_output=True,
        check=False,
    )
    if diff.returncode != 0 or untracked.returncode != 0:
        return None
    digest = hashlib.sha256()
    digest.update(status_text.encode("utf-8", errors="surrogateescape"))
    digest.update(diff.stdout)
    for raw_path in sorted(
        path for path in untracked.stdout.split(b"\0") if path
    ):
        relative = os.fsdecode(raw_path)
        candidate = root / relative
        digest.update(raw_path)
        try:
            if candidate.is_file():
                with candidate.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            elif candidate.is_symlink():
                digest.update(os.fsencode(os.readlink(candidate)))
        except OSError:
            return None
    return digest.hexdigest()


def _git_state(root: Path, environment: Mapping[str, str]) -> dict[str, object]:
    top_level = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        cwd=root,
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
    )
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        cwd=root,
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
    )
    if top_level.returncode != 0 or revision.returncode != 0 or status.returncode != 0:
        return {
            "root": None,
            "revision": None,
            "dirty": None,
            "worktree_digest": None,
            "error": (
                top_level.stderr.strip()
                or revision.stderr.strip()
                or status.stderr.strip()
                or "git inspection failed"
            ),
        }
    return {
        "root": str(Path(top_level.stdout.strip()).resolve()),
        "revision": revision.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
        "worktree_digest": _worktree_digest(
            root,
            environment,
            status.stdout,
        ),
    }


def _validated_command(
    command: Sequence[str], *, verify_only: bool
) -> list[str]:
    normalized = list(command)
    if normalized and normalized[0] == "--":
        normalized.pop(0)
    if verify_only and normalized:
        raise ValueError(
            "--verify-only cannot be combined with a command; remove one of them"
        )
    return normalized


def _empty_probe() -> dict[str, object]:
    return {field: None for field in _PROBE_RECEIPT_FIELDS}


def _empty_source_stack() -> dict[str, object]:
    return {
        component: {
            "root": None,
            "revision": None,
            "dirty": None,
            "worktree_digest": None,
        }
        for component in _SOURCE_COMPONENTS
    }


def _receipt_environment(
    environment: Mapping[str, str],
    *,
    python: Path,
    tilelang_root: Path,
) -> dict[str, object]:
    tvm_root = tilelang_root / "3rdparty" / "tvm"
    return {
        "mode": environment.get("CPPMEGA_MLX_ENV_MODE"),
        "python": str(python),
        "env_root": environment.get("CPPMEGA_MLX_ENV_ROOT"),
        "tilelang_root": str(tilelang_root),
        "tvm_root": str(tvm_root),
        "pythonpath": environment.get("PYTHONPATH"),
        "library_path": environment.get("DYLD_LIBRARY_PATH"),
    }


def _receipt(
    *,
    probe: Mapping[str, object],
    evaluation: PathCProbeEvaluation,
    environment: Mapping[str, str],
    python: Path,
    tilelang_root: Path,
    source_stack: Mapping[str, object],
    allow_dirty_source_stack: bool,
    command: Sequence[str] = (),
) -> dict[str, object]:
    tvm_root = tilelang_root / "3rdparty" / "tvm"
    return {
        "kind": RECEIPT_KIND,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "contract": CONTRACT_MODE,
        "ok": evaluation.ok,
        "allow_dirty_source_stack": allow_dirty_source_stack,
        "command": list(command),
        "reproducibility": {
            "clean_source_stack_required": not allow_dirty_source_stack,
            "selected_python": str(python),
            "selected_tilelang_root": str(tilelang_root),
            "selected_tvm_root": str(tvm_root),
        },
        "environment": _receipt_environment(
            environment,
            python=python,
            tilelang_root=tilelang_root,
        ),
        "source_stack": dict(source_stack),
        "probe": dict(probe),
        "evaluation": asdict(evaluation),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(os.environ.get("CPPMEGA_MLX_PYTHON", default_python())),
    )
    parser.add_argument(
        "--tilelang-root",
        type=Path,
        default=Path(
            os.environ.get("CPPMEGA_TILELANG_SOURCE_ROOT", default_tilelang_root())
        ),
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--allow-dirty-source-stack",
        action="store_true",
        help="allow dirty TileLang/TVM/TVM-FFI trees and record the opt-in in the receipt",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _emit_receipt(receipt: Mapping[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return
    evaluation = receipt["evaluation"]
    assert isinstance(evaluation, Mapping)
    environment = receipt["environment"]
    assert isinstance(environment, Mapping)
    print(f"contract: {receipt['contract']}")
    print(f"python: {environment['python']}")
    print(f"mlx_mode: {evaluation['mlx_mode']}")
    print(f"tilelang_mode: {evaluation['tilelang_mode']}")
    messages = evaluation["messages"]
    assert isinstance(messages, (list, tuple))
    for message in messages:
        print(message)
    print("PASS" if receipt["ok"] else "FAIL", flush=True)


def _failure_evaluation(message: str) -> PathCProbeEvaluation:
    return PathCProbeEvaluation(
        ok=False,
        mode=CONTRACT_MODE,
        mlx_mode="unavailable",
        tilelang_mode="unavailable",
        messages=(f"FAIL: {message}",),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    python = Path(args.python).expanduser().absolute()
    tilelang_root = Path(args.tilelang_root).expanduser().absolute()
    try:
        command = _validated_command(args.command, verify_only=args.verify_only)
    except ValueError as exc:
        receipt = _receipt(
            probe=_empty_probe(),
            evaluation=_failure_evaluation(str(exc)),
            environment={
                "CPPMEGA_MLX_ENV_MODE": None,
                "CPPMEGA_MLX_ENV_ROOT": str(python.parent.parent),
            },
            python=python,
            tilelang_root=tilelang_root,
            source_stack=_empty_source_stack(),
            allow_dirty_source_stack=args.allow_dirty_source_stack,
            command=args.command,
        )
        _emit_receipt(receipt, json_output=args.json)
        if not args.json:
            print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    environment: dict[str, str] = {}
    source_stack: dict[str, object] = _empty_source_stack()
    probe: dict[str, object] = _empty_probe()
    evaluation: PathCProbeEvaluation
    if not python.is_file() or not os.access(python, os.X_OK):
        evaluation = _failure_evaluation(f"dedicated Python is unavailable: {python}")
        receipt = _receipt(
            probe=probe,
            evaluation=evaluation,
            environment={
                "CPPMEGA_MLX_ENV_MODE": None,
                "CPPMEGA_MLX_ENV_ROOT": str(python.parent.parent),
            },
            python=python,
            tilelang_root=tilelang_root,
            source_stack=source_stack,
            allow_dirty_source_stack=args.allow_dirty_source_stack,
            command=command,
        )
        _emit_receipt(receipt, json_output=args.json)
        if not args.json:
            print(f"FAIL: dedicated Python is unavailable: {python}", file=sys.stderr)
        return 2
    try:
        environment = build_path_c_environment(
            os.environ,
            repo_root=ROOT,
            python=python,
            tilelang_root=tilelang_root,
        )
        environment = sanitize_launch_environment(environment, python=python)
        environment["CPPMEGA_MLX_ENV_ROOT"] = str(python.parent.parent)
        tvm_root = tilelang_root / "3rdparty" / "tvm"
        source_stack = {
            "tilelang": _git_state(tilelang_root, environment),
            "tvm": _git_state(tvm_root, environment),
            "tvm_ffi": _git_state(tvm_root / "3rdparty" / "tvm-ffi", environment),
        }
        source_messages = _source_stack_messages(
            source_stack,
            tilelang_root=tilelang_root,
            allow_dirty_source_stack=args.allow_dirty_source_stack,
        )
        if any(message.startswith("FAIL:") for message in source_messages):
            evaluation = PathCProbeEvaluation(
                ok=False,
                mode=CONTRACT_MODE,
                mlx_mode="unavailable",
                tilelang_mode="workspace_source",
                messages=tuple(source_messages),
            )
        else:
            probe = run_probe(python, environment)
            evaluation = evaluate_path_c_probe(
                probe,
                python=python,
                tilelang_root=tilelang_root,
                source_stack=source_stack,
                allow_dirty_source_stack=args.allow_dirty_source_stack,
            )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        evaluation = _failure_evaluation(str(exc))

    receipt = _receipt(
        probe=probe,
        evaluation=evaluation,
        environment=environment,
        python=python,
        tilelang_root=tilelang_root,
        source_stack=source_stack,
        allow_dirty_source_stack=args.allow_dirty_source_stack,
        command=command,
    )
    _emit_receipt(receipt, json_output=args.json)
    if not args.json and not evaluation.ok:
        print("FAIL: Path-C verification failed", file=sys.stderr)
    if not evaluation.ok:
        return 1

    if args.verify_only or not command:
        return 0
    try:
        os.execvpe(command[0], command, environment)
    except FileNotFoundError:
        print(f"FAIL: command is unavailable: {command[0]}", file=sys.stderr)
        return 127
    except OSError as exc:
        print(f"FAIL: cannot launch command {command[0]}: {exc}", file=sys.stderr)
        return 126


if __name__ == "__main__":
    raise SystemExit(main())
