import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from setuptools import setup

import nanobind
from mlx import extension


def _mlx_probe_environment(environ: Mapping[str, str]) -> dict[str, str]:
    probe_env = dict(environ)
    probe_env.pop("PYTHONPATH", None)
    return probe_env


def _target_mlx_cmake_root() -> Path:
    probe = subprocess.run(
        [sys.executable, "-m", "mlx", "--cmake-dir"],
        check=False,
        capture_output=True,
        text=True,
        env=_mlx_probe_environment(os.environ),
    )
    if probe.returncode == 0 and probe.stdout.strip():
        runtime_root = Path(probe.stdout.strip()).resolve()
        runtime_config = runtime_root / "share" / "cmake" / "MLX" / "MLXConfig.cmake"
        if runtime_config.is_file():
            return runtime_root

    build_root = Path(extension.__file__).resolve().parent
    build_config = build_root / "share" / "cmake" / "MLX" / "MLXConfig.cmake"
    if not build_config.is_file():
        raise RuntimeError(
            "neither the target interpreter nor the isolated build dependency "
            "provides MLXConfig.cmake"
        )
    print(
        "cppmega build: target interpreter has no MLX CMake package; "
        f"using isolated build dependency at {build_root}",
        file=sys.stderr,
    )
    return build_root


class CppmegaCMakeBuild(extension.CMakeBuild):
    """Build native extensions with the interpreter that invoked setuptools."""

    def build_extension(self, ext: extension.CMakeExtension) -> None:
        previous_args = os.environ.get("CMAKE_ARGS")
        build_contract_args = (
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dnanobind_DIR={nanobind.cmake_dir()}",
            f"-DMLX_ROOT={_target_mlx_cmake_root()}",
        )
        os.environ["CMAKE_ARGS"] = " ".join(
            part for part in (previous_args, *build_contract_args) if part
        )
        try:
            super().build_extension(ext)
        finally:
            if previous_args is None:
                os.environ.pop("CMAKE_ARGS", None)
            else:
                os.environ["CMAKE_ARGS"] = previous_args


if __name__ == "__main__":
    setup(
        ext_modules=[
            extension.CMakeExtension(
                "cppmega_mlx.training.native_optim._ext",
                sourcedir="cppmega_mlx/training/native_optim",
            )
        ],
        cmdclass={"build_ext": CppmegaCMakeBuild},
        package_data={
            "cppmega_mlx.training.native_optim": [
                "*.so",
                "*.dylib",
                "*.metallib",
            ]
        },
        zip_safe=False,
    )
