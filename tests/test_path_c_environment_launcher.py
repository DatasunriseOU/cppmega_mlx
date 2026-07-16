from __future__ import annotations

from pathlib import Path
import runpy
import subprocess
import os
import json
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
_VENVPY = Path("/venv/bin/python")
_VENVPREFIX = Path("/venv")
_DIST = Path("/venv/lib/python3.13/site-packages")


def _load_launcher() -> dict[str, object]:
    path = ROOT / "scripts" / "run_mlx_path_c.py"
    assert path.is_file(), "missing Path-C environment launcher"
    return runpy.run_path(str(path))


def test_launcher_loads_repository_contract_without_ambient_module_shadowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("scripts.mlx_env_contract")
    fake.default_python = lambda: Path("/wrong/python")
    fake.default_tilelang_root = lambda: Path("/wrong/tilelang")
    fake.build_path_c_environment = lambda *_args, **_kwargs: {
        "CPPMEGA_MLX_ENV_MODE": "wrong"
    }
    monkeypatch.setitem(sys.modules, "scripts.mlx_env_contract", fake)

    launcher = _load_launcher()

    loaded = launcher["_ENVIRONMENT_CONTRACT"]
    assert getattr(loaded, "__file__", "").endswith(
        "/scripts/mlx_env_contract.py"
    )
    assert launcher["default_python"]() != Path("/wrong/python")


def _source_probe(tilelang_root: Path) -> dict[str, object]:
    tvm_root = tilelang_root / "3rdparty" / "tvm"
    ffi_root = tvm_root / "3rdparty" / "tvm-ffi"
    return {
        "platform": "Darwin",
        "python_executable": str(_VENVPY),
        "python_prefix": str(_VENVPREFIX),
        "mlx_module_file": str(_DIST / "mlx/core.so"),
        "mlx_module_version": "0.32.0",
        "mlx_package_version": "0.32.0",
        "mlx_distribution_root": str(_DIST),
        "mlx_metal_version": "0.32.0",
        "mlx_runtime_smoke_ok": True,
        "mlx_loaded_library": str(_DIST / "mlx/lib/libmlx.dylib"),
        "tilelang_module_file": str(tilelang_root / "tilelang/__init__.py"),
        "tilelang_module_version": "0.1.9+gitfixture",
        "tilelang_distribution_version": "0.1.9",
        "tvm_module_file": str(tvm_root / "python/tvm/__init__.py"),
        "tvm_ffi_module_file": str(ffi_root / "python/tvm_ffi/__init__.py"),
        "tvm_ffi_core_file": str(ffi_root / "build/core.cpython-313-darwin.so"),
        "tilelang_lower_callable": True,
        "metal_target_ok": True,
        "source_runtime_libraries": [
            str(tilelang_root / "build/lib/libtilelang.dylib"),
            str(tilelang_root / "build/lib/libtvm_runtime.dylib"),
            str(tilelang_root / "3rdparty/tvm/3rdparty/tvm-ffi/build/libtvm_ffi.dylib"),
        ],
    }


def _source_stack_state(tilelang_root: Path, *, dirty: str | None = None) -> dict[str, object]:
    tvm_root = tilelang_root / "3rdparty" / "tvm"
    ffi_root = tvm_root / "3rdparty" / "tvm-ffi"
    return {
        "tilelang": {
            "root": str(tilelang_root),
            "revision": "tilelang-revision",
            "dirty": dirty == "tilelang",
            "worktree_digest": "1" * 64,
        },
        "tvm": {
            "root": str(tvm_root),
            "revision": "tvm-revision",
            "dirty": dirty == "tvm",
            "worktree_digest": "2" * 64,
        },
        "tvm_ffi": {
            "root": str(ffi_root),
            "revision": "tvm-ffi-revision",
            "dirty": dirty == "tvm_ffi",
            "worktree_digest": "3" * 64,
        },
    }


def test_path_c_probe_accepts_source_tilelang_with_installed_mlx_wheel(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    tilelang_root = tmp_path / "tilelang"
    probe = _source_probe(tilelang_root)

    result = launcher["evaluate_path_c_probe"](
        probe,
        python=_VENVPY,
        tilelang_root=tilelang_root,
        source_stack=_source_stack_state(tilelang_root),
    )

    assert result.ok
    assert result.mlx_mode == "installed_wheel"
    assert result.mode == "path_c_source"


def test_path_c_probe_rejects_tilelang_wheel_origin(tmp_path: Path) -> None:
    launcher = _load_launcher()
    tilelang_root = tmp_path / "tilelang"
    probe = _source_probe(tilelang_root)
    probe["tilelang_module_file"] = str(_DIST / "tilelang/__init__.py")

    result = launcher["evaluate_path_c_probe"](
        probe,
        python=_VENVPY,
        tilelang_root=tilelang_root,
        source_stack=_source_stack_state(tilelang_root),
    )

    assert not result.ok
    assert "tilelang" in "\n".join(result.messages).lower()
    assert "source" in "\n".join(result.messages).lower()


def test_path_c_probe_rejects_non_darwin_platform(tmp_path: Path) -> None:
    launcher = _load_launcher()
    tilelang_root = tmp_path / "tilelang"
    probe = _source_probe(tilelang_root)
    probe["platform"] = "Linux"

    result = launcher["evaluate_path_c_probe"](
        probe,
        python=_VENVPY,
        tilelang_root=tilelang_root,
        source_stack=_source_stack_state(tilelang_root),
    )

    assert not result.ok
    assert "requires darwin" in "\n".join(result.messages).lower()


@pytest.mark.parametrize(
    "missing_field",
    ("tilelang_module_version", "tilelang_distribution_version"),
)
def test_path_c_probe_rejects_missing_tilelang_release(
    tmp_path: Path, missing_field: str
) -> None:
    launcher = _load_launcher()
    tilelang_root = tmp_path / "tilelang"
    probe = _source_probe(tilelang_root)
    probe.pop(missing_field)

    result = launcher["evaluate_path_c_probe"](
        probe,
        python=_VENVPY,
        tilelang_root=tilelang_root,
        source_stack=_source_stack_state(tilelang_root),
    )

    assert not result.ok
    assert "release is unavailable" in "\n".join(result.messages).lower()


def test_path_c_probe_rejects_mismatched_tilelang_release(tmp_path: Path) -> None:
    launcher = _load_launcher()
    tilelang_root = tmp_path / "tilelang"
    probe = _source_probe(tilelang_root)
    probe["tilelang_distribution_version"] = "0.2.0"

    result = launcher["evaluate_path_c_probe"](
        probe,
        python=_VENVPY,
        tilelang_root=tilelang_root,
        source_stack=_source_stack_state(tilelang_root),
    )

    assert not result.ok
    assert "release contract differs" in "\n".join(result.messages).lower()


@pytest.mark.parametrize(
    "missing_field",
    ("tilelang_lower_callable", "metal_target_ok", "source_runtime_libraries"),
)
def test_path_c_probe_rejects_incomplete_runtime_receipt(
    tmp_path: Path,
    missing_field: str,
) -> None:
    launcher = _load_launcher()
    tilelang_root = tmp_path / "tilelang"
    probe = _source_probe(tilelang_root)
    probe.pop(missing_field)

    result = launcher["evaluate_path_c_probe"](
        probe,
        python=_VENVPY,
        tilelang_root=tilelang_root,
        source_stack=_source_stack_state(tilelang_root),
    )

    assert not result.ok
    assert "fail:" in "\n".join(result.messages).lower()


def test_path_c_probe_rejects_dirty_source_stack_by_default(tmp_path: Path) -> None:
    launcher = _load_launcher()
    tilelang_root = tmp_path / "tilelang"

    result = launcher["evaluate_path_c_probe"](
        _source_probe(tilelang_root),
        python=_VENVPY,
        tilelang_root=tilelang_root,
        source_stack=_source_stack_state(tilelang_root, dirty="tvm"),
    )

    assert not result.ok
    assert "dirty tvm source tree" in "\n".join(result.messages).lower()


def test_path_c_probe_rejects_missing_source_reproducibility_field(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    tilelang_root = tmp_path / "tilelang"
    source_stack = _source_stack_state(tilelang_root)
    source_stack["tvm"].pop("worktree_digest")

    result = launcher["evaluate_path_c_probe"](
        _source_probe(tilelang_root),
        python=_VENVPY,
        tilelang_root=tilelang_root,
        source_stack=source_stack,
    )

    assert not result.ok
    assert "worktree digest" in "\n".join(result.messages).lower()


def test_path_c_probe_allows_dirty_source_stack_only_explicitly(tmp_path: Path) -> None:
    launcher = _load_launcher()
    tilelang_root = tmp_path / "tilelang"

    result = launcher["evaluate_path_c_probe"](
        _source_probe(tilelang_root),
        python=_VENVPY,
        tilelang_root=tilelang_root,
        source_stack=_source_stack_state(tilelang_root, dirty="tilelang"),
        allow_dirty_source_stack=True,
    )

    assert result.ok
    assert "explicitly allowed" in "\n".join(result.messages).lower()


def test_path_c_command_rejects_verify_only_with_command() -> None:
    launcher = _load_launcher()

    with pytest.raises(ValueError, match="cannot be combined"):
        launcher["_validated_command"](
            ("--", "/usr/bin/false"), verify_only=True
        )


def test_path_c_probe_rejects_wrong_python_identity(tmp_path: Path) -> None:
    launcher = _load_launcher()
    tilelang_root = tmp_path / "tilelang"
    probe = _source_probe(tilelang_root)
    probe["python_executable"] = "/other/bin/python"

    result = launcher["evaluate_path_c_probe"](
        probe,
        python=_VENVPY,
        tilelang_root=tilelang_root,
        source_stack=_source_stack_state(tilelang_root),
    )

    assert not result.ok
    assert "selected interpreter" in "\n".join(result.messages).lower()


def test_launch_environment_drops_ambient_paths_and_build_tuning() -> None:
    launcher = _load_launcher()
    environment = launcher["sanitize_launch_environment"](
        {
            "PATH": "/attacker/bin:/usr/bin",
            "CPATH": "/attacker/include",
            "CMAKE_PREFIX_PATH": "/attacker/cmake",
            "MLX_DEFAULT_DEVICE": "cpu",
            "MTL_DEBUG_LAYER": "1",
            "CPPMEGA_MLX_ENV_MODE": "path-c-source",
        },
        python=_VENVPY,
    )

    assert environment["PATH"].split(os.pathsep)[0] == str(_VENVPY.parent)
    assert "/attacker" not in "\n".join(environment.values())
    assert environment["CPPMEGA_MLX_ENV_MODE"] == "path-c-source"


def test_git_state_records_actual_top_level_for_nested_checkout(tmp_path: Path) -> None:
    launcher = _load_launcher()
    repo = tmp_path / "repo"
    nested = repo / "nested"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=test@example.com", "-c", "user.name=test", "commit", "--allow-empty", "-m", "init"],
        check=True,
        capture_output=True,
        text=True,
    )

    state = launcher["_git_state"](nested, os.environ)

    assert state["root"] == str(repo.resolve())
    assert state["revision"]
    assert state["dirty"] is False
    assert len(state["worktree_digest"]) == 64

    tracked = repo / "tracked.txt"
    tracked.write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=test@example.com", "-c", "user.name=test", "commit", "-m", "tracked"],
        check=True,
        capture_output=True,
        text=True,
    )
    clean = launcher["_git_state"](nested, os.environ)
    tracked.write_text("two\n", encoding="utf-8")
    dirty = launcher["_git_state"](nested, os.environ)

    assert clean["dirty"] is False
    assert dirty["dirty"] is True
    assert dirty["worktree_digest"] != clean["worktree_digest"]


def test_path_c_json_failure_receipt_preserves_required_fields(
    tmp_path: Path,
) -> None:
    launcher_path = ROOT / "scripts" / "run_mlx_path_c.py"
    missing_python = tmp_path / "missing-python"
    result = subprocess.run(
        [
            os.fspath(Path(sys.executable)),
            os.fspath(launcher_path),
            "--json",
            "--verify-only",
            "--python",
            os.fspath(missing_python),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["kind"] == "cppmega_mlx_path_c_environment_receipt"
    assert payload["schema_version"] == 1
    assert payload["environment"]["python"] == str(missing_python)
    assert "mlx_module_file" in payload["probe"]
    assert set(payload["source_stack"]) == {"tilelang", "tvm", "tvm_ffi"}
    assert payload["evaluation"]["ok"] is False


def test_path_c_launcher_verifies_real_dedicated_stack_without_shell_overrides() -> None:
    launcher_path = ROOT / "scripts" / "run_mlx_path_c.py"
    python = Path("/Volumes/external/sources/.venvs/cppmega.mlx/bin/python")
    tilelang_root = Path("/Volumes/external/sources/tilelang")
    if not python.is_file() or not tilelang_root.is_dir():
        pytest.skip("dedicated MLX environment or sibling TileLang checkout is unavailable")

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": "/wrong/mlx:/wrong/tvm",
            "VIRTUAL_ENV": "/wrong/.venv",
            "TL_EXTERNAL_TVM_HOME": "/wrong/tvm",
            "TVM_ROOT": "/wrong/tvm",
            "TILELANG_ROOT": "/wrong/tilelang",
            "DYLD_LIBRARY_PATH": "/wrong/lib",
        }
    )
    result = subprocess.run(
        [
            str(python),
            str(launcher_path),
            "--json",
            "--verify-only",
            "--allow-dirty-source-stack",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["evaluation"]["ok"] is True
    assert payload["allow_dirty_source_stack"] is True
    assert payload["evaluation"]["mode"] == "path_c_source"
    assert payload["evaluation"]["mlx_mode"] == "installed_wheel"
    assert payload["environment"]["tilelang_root"] == str(tilelang_root)
    assert payload["probe"]["python_executable"] == str(python)
    assert payload["source_stack"]["tvm_ffi"]["root"] == str(
        tilelang_root / "3rdparty" / "tvm" / "3rdparty" / "tvm-ffi"
    )
    assert "/wrong" not in result.stdout


def test_path_c_launcher_executes_with_dedicated_python_and_source_modules() -> None:
    launcher_path = ROOT / "scripts" / "run_mlx_path_c.py"
    python = Path("/Volumes/external/sources/.venvs/cppmega.mlx/bin/python")
    tilelang_root = Path("/Volumes/external/sources/tilelang")
    if not python.is_file() or not tilelang_root.is_dir():
        pytest.skip("dedicated MLX environment or sibling TileLang checkout is unavailable")

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": "/wrong/mlx:/wrong/tvm",
            "VIRTUAL_ENV": "/wrong/.venv",
            "TL_EXTERNAL_TVM_HOME": "/wrong/tvm",
            "TVM_ROOT": "/wrong/tvm",
            "TILELANG_ROOT": "/wrong/tilelang",
            "DYLD_LIBRARY_PATH": "/wrong/lib",
        }
    )
    code = """
import json
import os
from pathlib import Path
import sys
import tilelang
import tvm
import tvm_ffi

print("__PATH_C_LAUNCH__=" + json.dumps({
    "mode": os.environ.get("CPPMEGA_MLX_ENV_MODE"),
    "prefix": str(Path(sys.prefix).resolve()),
    "tilelang": str(Path(tilelang.__file__).resolve()),
    "tvm": str(Path(tvm.__file__).resolve()),
    "tvm_ffi": str(Path(tvm_ffi.__file__).resolve()),
}, sort_keys=True))
"""
    result = subprocess.run(
        [
            str(python),
            str(launcher_path),
            "--allow-dirty-source-stack",
            "--",
            "python",
            "-c",
            code,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    receipt_line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("__PATH_C_LAUNCH__=")
    )
    payload = json.loads(receipt_line.removeprefix("__PATH_C_LAUNCH__="))
    assert payload["mode"] == "path-c-source"
    assert payload["prefix"] == str(python.parent.parent)
    assert Path(payload["tilelang"]).is_relative_to(tilelang_root)
    assert Path(payload["tvm"]).is_relative_to(tilelang_root / "3rdparty" / "tvm")
    assert Path(payload["tvm_ffi"]).is_relative_to(
        tilelang_root / "3rdparty" / "tvm" / "3rdparty" / "tvm-ffi"
    )
    assert "/wrong" not in receipt_line


def test_bench_contract_mode_ignores_ambient_tvm_root(tmp_path: Path) -> None:
    python = Path("/Volumes/external/sources/.venvs/cppmega.mlx/bin/python")
    if not python.is_file():
        pytest.skip("dedicated MLX environment is unavailable")
    selected = tmp_path / "selected-tilelang"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(ROOT),
            "CPPMEGA_MLX_ENV_MODE": "path-c-source",
            "CPPMEGA_TILELANG_SOURCE_ROOT": str(selected),
            "TILELANG_ROOT": "/wrong/tilelang",
            "TVM_ROOT": "/wrong/tvm",
        }
    )
    code = """
import json
from scripts import bench_tilelang_fp8_path_c as bench
print(json.dumps({
    "tilelang": str(bench._resolve_tilelang_root()),
    "tvm": str(bench._resolve_tvm_root()),
}, sort_keys=True))
"""
    result = subprocess.run(
        [str(python), "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "tilelang": str(selected),
        "tvm": str(selected / "3rdparty" / "tvm"),
    }
