from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

from setuptools import Distribution


def test_native_build_dependencies_are_declared() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    requirements = pyproject["build-system"]["requires"]

    assert "mlx==0.32.0" in requirements
    assert any(requirement.startswith("nanobind>=") for requirement in requirements)
    assert "mlx==0.32.0" in pyproject["project"]["dependencies"]
    assert "z3-solver>=4.15" in pyproject["project"]["dependencies"]
    assert "python-multipart>=0.0.20" in pyproject["project"][
        "optional-dependencies"
    ]["gui"]
    assert "websockets>=12" in pyproject["project"]["optional-dependencies"]["gui"]


def test_mlx_cmake_probe_ignores_workspace_pythonpath() -> None:
    from setup import _mlx_probe_environment

    probe_env = _mlx_probe_environment(
        {
            "PYTHONPATH": "/workspace/mlx/python:/workspace/tilelang",
            "PATH": "/usr/bin",
        }
    )

    assert "PYTHONPATH" not in probe_env
    assert probe_env["PATH"] == "/usr/bin"


def test_native_build_pins_cmake_to_invoking_python(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))

    import setup as cppmega_setup

    captured: dict[str, str | None] = {}

    def capture_build_extension(_command, _extension) -> None:
        captured["cmake_args"] = os.environ.get("CMAKE_ARGS")

    monkeypatch.setattr(
        cppmega_setup.extension.CMakeBuild,
        "build_extension",
        capture_build_extension,
    )
    monkeypatch.setenv("CMAKE_ARGS", "-DEXISTING_OPTION=ON")

    command = cppmega_setup.CppmegaCMakeBuild(Distribution())
    command.build_extension(object())

    cmake_args = captured["cmake_args"]
    assert cmake_args is not None
    assert cmake_args.startswith(
        f"-DEXISTING_OPTION=ON -DPython_EXECUTABLE={sys.executable}"
    )
    assert "-Dnanobind_DIR=" in cmake_args
    assert "-DMLX_ROOT=" in cmake_args
    assert os.environ["CMAKE_ARGS"] == "-DEXISTING_OPTION=ON"


def test_native_build_prefers_target_interpreter_mlx(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))

    import setup as cppmega_setup

    runtime_root = tmp_path / "runtime-mlx"
    config = runtime_root / "share" / "cmake" / "MLX" / "MLXConfig.cmake"
    config.parent.mkdir(parents=True)
    config.write_text("# fixture\n")

    monkeypatch.setattr(
        cppmega_setup.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"{runtime_root}\n",
            stderr="",
        ),
    )

    assert cppmega_setup._target_mlx_cmake_root() == runtime_root.resolve()
