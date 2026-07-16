from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types

import pytest

from scripts import check_mlx_abi
from scripts.check_mlx_abi import AbiProbe, evaluate_probe


ROOT = Path(__file__).resolve().parents[1]
_VENVPY = Path("/venv/bin/python")
_VENVPREFIX = Path("/venv")
_DIST = Path("/venv/lib/python3.13/site-packages")


def _load_env_contract() -> dict[str, object]:
    module_path = ROOT / "scripts" / "mlx_env_contract.py"
    assert module_path.is_file(), "missing MLX/TileLang environment contract module"
    return runpy.run_path(str(module_path))


def test_abi_script_loads_repository_contract_without_ambient_shadowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("scripts.mlx_env_contract")
    fake.build_mlx_wheel_environment = lambda *_args, **_kwargs: {
        "CPPMEGA_MLX_ENV_MODE": "wrong"
    }
    monkeypatch.setitem(sys.modules, "scripts.mlx_env_contract", fake)

    namespace = runpy.run_path(str(ROOT / "scripts" / "check_mlx_abi.py"))

    loaded = namespace["_load_environment_contract"]()
    assert getattr(loaded, "__file__", "").endswith(
        "/scripts/mlx_env_contract.py"
    )


def _fake_path_c_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo_root = tmp_path / "cppmega.mlx"
    repo_root.mkdir()
    python = tmp_path / ".venvs" / "cppmega.mlx" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    python.chmod(0o755)

    tilelang_root = tmp_path / "tilelang"
    tvm_root = tilelang_root / "3rdparty" / "tvm"
    tvm_ffi_root = tvm_root / "3rdparty" / "tvm-ffi"
    for path in (
        tilelang_root / "tilelang",
        tvm_root / "python" / "tvm",
        tvm_ffi_root / "python" / "tvm_ffi",
        tvm_ffi_root / "build" / "lib",
        tilelang_root / "build" / "lib",
    ):
        path.mkdir(parents=True)
    (tilelang_root / "build" / "lib" / "libtilelang.dylib").touch()
    (tilelang_root / "build" / "lib" / "libtvm_runtime.dylib").touch()
    return repo_root, python, tilelang_root


def test_wheel_environment_discards_ambient_source_overrides(tmp_path: Path) -> None:
    contract = _load_env_contract()
    _, python, _ = _fake_path_c_layout(tmp_path)
    environment = contract["build_mlx_wheel_environment"](
        {
            "PATH": "/bad/.venv/bin:/usr/bin",
            "PYTHONPATH": "/bad/mlx/python:/bad/tvm/python",
            "VIRTUAL_ENV": "/bad/.venv",
            "VIRTUAL_ENV_PROMPT": "bad",
            "CPPMEGA_MLX_SOURCE_ROOT": "/bad/mlx",
            "TILELANG_ROOT": "/bad/tilelang",
            "TVM_ROOT": "/bad/tvm",
            "TVM_LIBRARY_PATH": "/bad/lib",
            "TL_MLX_SOURCE_HOME": "/bad/mlx",
            "MLX_DEFAULT_DEVICE": "cpu",
            "MTL_DEBUG_LAYER": "1",
            "DYLD_LIBRARY_PATH": "/bad/lib",
        },
        python=python,
    )

    assert environment["CPPMEGA_MLX_ENV_MODE"] == "mlx-wheel"
    assert environment["VIRTUAL_ENV"] == str(python.parent.parent)
    assert environment["PATH"].split(os.pathsep)[0] == str(python.parent)
    for name in (
        "PYTHONPATH",
        "CPPMEGA_MLX_SOURCE_ROOT",
        "TILELANG_ROOT",
        "TVM_ROOT",
        "TVM_LIBRARY_PATH",
        "TL_MLX_SOURCE_HOME",
        "MLX_DEFAULT_DEVICE",
        "MTL_DEBUG_LAYER",
        "DYLD_LIBRARY_PATH",
        "VIRTUAL_ENV_PROMPT",
    ):
        assert name not in environment


def test_path_c_environment_pins_source_stack_without_mlx_source(
    tmp_path: Path,
) -> None:
    contract = _load_env_contract()
    repo_root, python, tilelang_root = _fake_path_c_layout(tmp_path)
    environment = contract["build_path_c_environment"](
        {
            "PATH": "/bad/.venv/bin:/opt/homebrew/bin:/usr/bin",
            "PYTHONPATH": "/bad/mlx/python:/bad/tvm/python",
            "VIRTUAL_ENV": "/bad/.venv",
            "CPPMEGA_MLX_SOURCE_ROOT": "/bad/mlx",
            "CPPMEGA_TILELANG_SOURCE_ROOT": "/bad/tilelang",
            "TILELANG_ROOT": "/bad/tilelang",
            "TVM_ROOT": "/bad/tvm",
            "TVM_HOME": "/bad/tvm",
            "TL_EXTERNAL_TVM_HOME": "/bad/tvm",
            "MLX_DEFAULT_DEVICE": "cpu",
            "MTL_DEBUG_LAYER": "1",
            "DYLD_LIBRARY_PATH": "/bad/lib",
        },
        repo_root=repo_root,
        python=python,
        tilelang_root=tilelang_root,
    )

    tvm_root = tilelang_root / "3rdparty" / "tvm"
    tvm_ffi_root = tvm_root / "3rdparty" / "tvm-ffi"
    assert environment["CPPMEGA_MLX_ENV_MODE"] == "path-c-source"
    assert environment["CPPMEGA_TILELANG_SOURCE_ROOT"] == str(tilelang_root)
    assert environment["TILELANG_ROOT"] == str(tilelang_root)
    assert environment["TVM_ROOT"] == str(tvm_root)
    assert environment["TVM_HOME"] == str(tvm_root)
    assert environment["VIRTUAL_ENV"] == str(python.parent.parent)
    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(repo_root),
        str(tilelang_root),
        str(tvm_root / "python"),
        str(tvm_ffi_root / "python"),
    ]
    assert environment["DYLD_LIBRARY_PATH"].split(os.pathsep) == [
        str(tilelang_root / "build" / "lib"),
        str(tvm_ffi_root / "build" / "lib"),
    ]
    assert "CPPMEGA_MLX_SOURCE_ROOT" not in environment
    assert "/bad" not in "\n".join(environment.values())


def _healthy_probe(**overrides: object) -> AbiProbe:
    values: dict[str, object] = {
        "package_version": "0.32.0",
        "module_version": "0.32.0",
        "module_file": _DIST / "mlx/core.so",
        "mlx_metal_version": "0.32.0",
        "source_root": None,
        "brew_version": None,
        "python_executable": _VENVPY,
        "python_prefix": _VENVPREFIX,
        "distribution_root": _DIST,
        "loaded_libmlx": _DIST / "mlx/lib/libmlx.dylib",
        "runtime_smoke_ok": True,
        "runtime_smoke_detail": "test smoke",
        "platform_name": "Darwin",
    }
    values.update(overrides)
    return AbiProbe(**values)


def test_workspace_source_probe_checks_metal_pin_instead_of_brew_version() -> None:
    probe = _healthy_probe(
        module_version="0.32.0.dev20260527+e2e46fb8a",
        module_file=Path("/Volumes/external/sources/mlx/python/mlx/core.so"),
        source_root=Path("/Volumes/external/sources/mlx"),
        loaded_libmlx=Path("/Volumes/external/sources/mlx/build/lib/libmlx.dylib"),
        brew_version="0.31.1",
    )

    result = evaluate_probe(probe)

    assert result.ok
    assert result.mode == "workspace_source"
    assert "workspace source checkout" in result.messages[0]
    assert "brew version is informational" in "\n".join(result.messages)


def test_workspace_source_probe_rejects_stale_mlx_metal() -> None:
    probe = _healthy_probe(
        module_version="0.32.0.dev20260527+e2e46fb8a",
        module_file=Path("/Volumes/external/sources/mlx/python/mlx/core.so"),
        source_root=Path("/Volumes/external/sources/mlx"),
        loaded_libmlx=Path("/Volumes/external/sources/mlx/build/lib/libmlx.dylib"),
        mlx_metal_version="0.31.1",
        brew_version="0.32.0",
    )

    result = evaluate_probe(probe)

    assert not result.ok
    assert "mlx-metal" in "\n".join(result.messages)


def test_installed_wheel_probe_rejects_brew_drift() -> None:
    probe = _healthy_probe(
        loaded_libmlx=Path("/opt/homebrew/Cellar/mlx/0.31.1/lib/libmlx.dylib"),
        brew_version="0.31.1",
    )

    result = evaluate_probe(probe)

    assert not result.ok
    assert "loaded Homebrew libmlx" in "\n".join(result.messages)


def test_loaded_brew_keg_takes_precedence_over_linked_keg() -> None:
    probe = _healthy_probe(
        loaded_libmlx=Path("/opt/homebrew/Cellar/mlx/0.31.1/lib/libmlx.dylib"),
        brew_version="0.32.0",
    )

    result = evaluate_probe(probe)

    assert not result.ok
    assert "loaded Homebrew libmlx='0.31.1'" in "\n".join(result.messages)


def test_brew_version_uses_linked_opt_keg(tmp_path: Path) -> None:
    prefix = tmp_path / "homebrew"
    cellar = prefix / "Cellar" / "mlx"
    old_keg = cellar / "0.31.1"
    linked_keg = cellar / "0.32.0"
    old_keg.mkdir(parents=True)
    linked_keg.mkdir()
    opt = prefix / "opt"
    opt.mkdir()
    (opt / "mlx").symlink_to(linked_keg)

    assert check_mlx_abi._brew_version(prefix) == "0.32.0"


def test_loaded_homebrew_opt_keg_is_checked_against_imported_release(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "homebrew"
    cellar = prefix / "Cellar" / "mlx"
    linked_keg = cellar / "0.32.0"
    loaded_keg = cellar / "0.31.1"
    linked_keg.mkdir(parents=True)
    loaded_keg.mkdir()
    opt = prefix / "opt"
    opt.mkdir()
    (opt / "mlx").symlink_to(linked_keg)
    loaded_path = opt / "mlx" / "lib" / "libmlx.dylib"
    loaded_path.parent.mkdir()
    loaded_path.unlink(missing_ok=True)
    loaded_path.symlink_to(loaded_keg / "libmlx.dylib")

    probe = _healthy_probe(
        loaded_libmlx=loaded_path,
        brew_version="0.32.0",
        brew_prefix=prefix,
    )

    result = evaluate_probe(probe)

    assert not result.ok
    assert "loaded Homebrew libmlx='0.31.1'" in "\n".join(result.messages)


def test_probe_rejects_python_identity_that_differs_from_selected_interpreter() -> None:
    result = evaluate_probe(
        _healthy_probe(python_executable=Path("/other/bin/python")),
        expected_python=_VENVPY,
        expected_env_root=_VENVPREFIX,
    )

    assert not result.ok
    assert "selected interpreter" in "\n".join(result.messages)


def test_json_failure_receipt_preserves_required_fields(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "check_mlx_abi.py"
    missing_python = tmp_path / "missing-python"
    result = subprocess.run(
        [
            str(Path(sys.executable)),
            str(script),
            "--json",
            "--python",
            str(missing_python),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["kind"] == "cppmega_mlx_abi_receipt"
    assert payload["schema_version"] == 1
    assert payload["selected"]["python"] == str(missing_python)
    assert payload["probe"]["python_executable"] == str(missing_python)
    assert payload["evaluation"]["ok"] is False


def test_missing_mlx_metal_is_fail_closed() -> None:
    probe = _healthy_probe(mlx_metal_version=None)

    result = evaluate_probe(probe)

    assert not result.ok
    assert "mlx-metal" in "\n".join(result.messages)


def test_linux_probe_does_not_require_mlx_metal() -> None:
    result = evaluate_probe(
        _healthy_probe(platform_name="Linux", mlx_metal_version=None, loaded_libmlx=None)
    )

    assert result.ok
    assert "not applicable" in "\n".join(result.messages)


def test_legacy_repair_refuses_unowned_environment(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "fix_mlx_abi.sh"
    dedicated_python = ROOT.parent / ".venvs" / "cppmega.mlx" / "bin" / "python"
    if not dedicated_python.is_file():
        pytest.skip("dedicated MLX environment is unavailable")
    environment = os.environ.copy()
    environment["CPPMEGA_MLX_PYTHON"] = str(dedicated_python)
    environment["CPPMEGA_MLX_ENV_ROOT"] = str(tmp_path / "unowned-env")
    result = subprocess.run(
        ["bash", str(script), "--apply"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "refusing to sync unowned environment" in result.stderr


def test_legacy_repair_refuses_shared_checkout_environment() -> None:
    script = ROOT / "scripts" / "fix_mlx_abi.sh"
    shared_python = ROOT / ".venv" / "bin" / "python"
    if not shared_python.is_file():
        pytest.skip("shared checkout environment is unavailable")
    environment = os.environ.copy()
    environment.update(
        {
            "CPPMEGA_MLX_PYTHON": str(shared_python),
            "CPPMEGA_MLX_ENV_ROOT": str(ROOT / ".venv"),
        }
    )
    result = subprocess.run(
        ["bash", str(script), "--apply"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "inside git checkout" in result.stderr


def test_legacy_repair_refuses_base_python(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "fix_mlx_abi.sh"
    env_root = tmp_path / "env"
    env_root.mkdir()
    fake_python = env_root / "bin" / "python"
    fake_python.parent.mkdir()
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n%s\\n' \"$FAKE_PREFIX\" \"$FAKE_PREFIX\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "CPPMEGA_MLX_PYTHON": str(fake_python),
            "CPPMEGA_MLX_ENV_ROOT": str(env_root),
            "FAKE_PREFIX": str(env_root),
        }
    )

    result = subprocess.run(
        ["bash", str(script), "--apply"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "base Python" in result.stderr


def test_legacy_repair_rejects_unexpected_apply_arguments() -> None:
    script = ROOT / "scripts" / "fix_mlx_abi.sh"
    result = subprocess.run(
        ["bash", str(script), "--apply", "--python", "/tmp/other"],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unexpected argument" in result.stderr.lower()


def test_legacy_repair_launcher_is_executable() -> None:
    script = ROOT / "scripts" / "fix_mlx_abi.sh"

    assert os.access(script, os.X_OK)
