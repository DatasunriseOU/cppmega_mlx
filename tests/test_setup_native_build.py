from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

from setuptools import Distribution


def test_native_build_dependencies_are_declared() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    requirements = pyproject["build-system"]["requires"]

    assert any(requirement.startswith("mlx>=") for requirement in requirements)
    assert any(requirement.startswith("nanobind>=") for requirement in requirements)


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
