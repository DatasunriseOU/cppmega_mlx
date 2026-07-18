from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_native_build_dependencies_are_declared() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    requirements = pyproject["build-system"]["requires"]

    assert "mlx==0.32.0" in requirements
    assert any(requirement.startswith("nanobind>=") for requirement in requirements)
    assert "mlx==0.32.0" in pyproject["project"]["dependencies"]
    assert "z3-solver>=4.15,<4.15.5" in pyproject["project"]["dependencies"]
    assert "python-multipart>=0.0.20" in pyproject["project"][
        "optional-dependencies"
    ]["gui"]
    assert "websockets>=12" in pyproject["project"]["optional-dependencies"]["gui"]


def test_mlx_cmake_probe_ignores_workspace_pythonpath() -> None:
    from setup import _mlx_probe_environment

    probe_env = _mlx_probe_environment(
        {
            "PYTHONPATH": "/workspace/mlx/python:/workspace/tilelang",
            "CPPMEGA_MLX_SOURCE_ROOT": "/workspace/mlx",
            "DYLD_LIBRARY_PATH": "/workspace/mlx/build/lib",
            "MLX_DEFAULT_DEVICE": "cpu",
            "TVM_ROOT": "/workspace/tvm",
            "PATH": "/usr/bin",
        }
    )

    assert "PYTHONPATH" not in probe_env
    assert "CPPMEGA_MLX_SOURCE_ROOT" not in probe_env
    assert "DYLD_LIBRARY_PATH" not in probe_env
    assert "MLX_DEFAULT_DEVICE" not in probe_env
    assert "TVM_ROOT" not in probe_env
    assert probe_env["PATH"] == "/usr/bin"


def test_native_build_pins_cmake_to_invoking_python() -> None:
    import setup as cppmega_setup

    cmake_args = cppmega_setup._cmake_build_contract(
        "-DEXISTING_OPTION=ON",
        python=sys.executable,
        nanobind_dir=Path("/nanobind"),
        mlx_root=Path("/mlx"),
    )

    assert cmake_args.startswith(
        f"-DEXISTING_OPTION=ON -DPython_EXECUTABLE={sys.executable}"
    )
    assert "-Dnanobind_DIR=/nanobind" in cmake_args
    assert "-DMLX_ROOT=/mlx" in cmake_args


def test_native_build_prefers_target_interpreter_mlx(tmp_path) -> None:
    import setup as cppmega_setup

    runtime_root = tmp_path / "runtime-mlx"
    config = runtime_root / "share" / "cmake" / "MLX" / "MLXConfig.cmake"
    config.parent.mkdir(parents=True)
    config.write_text("# fixture\n")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=f"{runtime_root}\n",
            stderr="",
        )

    assert cppmega_setup._target_mlx_cmake_root(
        environ={}, runner=fake_run
    ) == runtime_root.resolve()


def test_native_build_refuses_implicit_build_dependency_fallback() -> None:
    import setup as cppmega_setup

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="target MLX unavailable",
        )

    with pytest.raises(RuntimeError, match="CPPMEGA_MLX_CMAKE_ROOT"):
        cppmega_setup._target_mlx_cmake_root(environ={}, runner=fake_run)


def test_native_build_accepts_only_valid_explicit_mlx_root(tmp_path) -> None:
    import setup as cppmega_setup

    root = tmp_path / "mlx-root"
    with pytest.raises(RuntimeError, match="MLXConfig.cmake"):
        cppmega_setup._target_mlx_cmake_root(
            environ={"CPPMEGA_MLX_CMAKE_ROOT": str(root)}
        )

    config = root / "share" / "cmake" / "MLX" / "MLXConfig.cmake"
    config.parent.mkdir(parents=True)
    config.write_text("# fixture\n", encoding="utf-8")
    assert cppmega_setup._target_mlx_cmake_root(
        environ={"CPPMEGA_MLX_CMAKE_ROOT": str(root)}
    ) == root.resolve()


def test_source_manifest_includes_native_extension_sources() -> None:
    manifest = (Path(__file__).resolve().parents[1] / "MANIFEST.in").read_text(
        encoding="utf-8"
    )

    for filename in (
        "CMakeLists.txt",
        "bindings.cpp",
        "fused_8bit.cpp",
        "fused_8bit.h",
        "fused_8bit.metal",
    ):
        assert filename in manifest

    assert "domain_schema_v1.json" in manifest
    assert "tokenizer.json" in manifest
    assert "global-exclude *.so *.dylib *.metallib" in manifest


def test_package_discovery_is_anchored_to_setup_py(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, setup; print(json.dumps(setup._local_packages()))",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    packages = json.loads(result.stdout)
    assert "cppmega_mlx" in packages
    assert "cppmega_v4" in packages
