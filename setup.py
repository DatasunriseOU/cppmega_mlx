import os
import sys
from pathlib import Path

from setuptools import setup

import nanobind
from mlx import extension


class CppmegaCMakeBuild(extension.CMakeBuild):
    """Build native extensions with the interpreter that invoked setuptools."""

    def build_extension(self, ext: extension.CMakeExtension) -> None:
        previous_args = os.environ.get("CMAKE_ARGS")
        build_contract_args = (
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dnanobind_DIR={nanobind.cmake_dir()}",
            f"-DMLX_ROOT={Path(extension.__file__).resolve().parent}",
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
