import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from setuptools import setup

import nanobind
from mlx import extension


SETUP_ROOT = Path(__file__).resolve().parent


def _local_packages() -> list[str]:
    """Return only the two Python package trees owned by this repository."""
    packages: list[str] = []
    for root_name in ("cppmega_mlx", "cppmega_v4"):
        root = SETUP_ROOT / root_name
        if not root.is_dir():
            raise RuntimeError(f"missing package root: {root}")
        for init_file in root.rglob("__init__.py"):
            package_path = init_file.parent
            packages.append(
                package_path.relative_to(SETUP_ROOT).as_posix().replace("/", ".")
            )
    return sorted(packages)


def _mlx_probe_environment(environ: Mapping[str, str]) -> dict[str, str]:
    probe_env = dict(environ)
    blocked = {
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
        "CPPMEGA_MLX_SOURCE_ROOT",
        "DYLD_LIBRARY_PATH",
        "MLX_DEFAULT_DEVICE",
        "TVM_ROOT",
    }
    for name in tuple(probe_env):
        if name in blocked or name.startswith(
            (
                "CPPMEGA_MLX_",
                "CPPMEGA_TILELANG_",
                "DYLD_",
                "LD_",
                "MLX_",
                "MTL_",
                "TL_",
                "TILELANG",
                "TVM_",
            )
        ):
            probe_env.pop(name, None)
    return probe_env


def _cmake_build_contract(
    existing_args: str | None,
    *,
    python: str,
    nanobind_dir: Path,
    mlx_root: Path,
) -> str:
    protected = ("-DPython_EXECUTABLE=", "-Dnanobind_DIR=", "-DMLX_ROOT=")
    inherited = shlex.split(existing_args) if existing_args else []
    inherited = [arg for arg in inherited if not arg.startswith(protected)]
    inherited.extend(
        (
            f"-DPython_EXECUTABLE={python}",
            f"-Dnanobind_DIR={nanobind_dir}",
            f"-DMLX_ROOT={mlx_root}",
        )
    )
    return shlex.join(inherited)


def _target_mlx_cmake_root(
    *,
    environ: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> Path:
    selected_environment = os.environ if environ is None else environ
    explicit_root = selected_environment.get("CPPMEGA_MLX_CMAKE_ROOT")
    if explicit_root:
        selected = Path(explicit_root).expanduser().resolve()
        config = selected / "share" / "cmake" / "MLX" / "MLXConfig.cmake"
        if not config.is_file():
            raise RuntimeError(
                "CPPMEGA_MLX_CMAKE_ROOT does not provide MLXConfig.cmake: "
                f"{config}"
            )
        return selected

    run = subprocess.run if runner is None else runner
    probe = run(
        [sys.executable, "-m", "mlx", "--cmake-dir"],
        check=False,
        capture_output=True,
        text=True,
        env=_mlx_probe_environment(selected_environment),
    )
    if probe.returncode == 0 and probe.stdout.strip():
        runtime_root = Path(probe.stdout.strip()).resolve()
        runtime_config = runtime_root / "share" / "cmake" / "MLX" / "MLXConfig.cmake"
        if runtime_config.is_file():
            return runtime_root
    detail = probe.stderr.strip() or probe.stdout.strip() or f"exit {probe.returncode}"
    raise RuntimeError(
        "the invoking Python did not provide a valid MLX CMake package: "
        f"{detail}. Select an intentional alternate root with "
        "CPPMEGA_MLX_CMAKE_ROOT."
    )


class CppmegaCMakeBuild(extension.CMakeBuild):
    """Build native extensions with the interpreter that invoked setuptools."""

    def build_extension(self, ext: extension.CMakeExtension) -> None:
        previous_args = os.environ.get("CMAKE_ARGS")
        os.environ["CMAKE_ARGS"] = _cmake_build_contract(
            previous_args,
            python=sys.executable,
            nanobind_dir=Path(nanobind.cmake_dir()),
            mlx_root=_target_mlx_cmake_root(),
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
        packages=_local_packages(),
        ext_modules=[
            extension.CMakeExtension(
                "cppmega_mlx.training.native_optim._ext",
                sourcedir=str(SETUP_ROOT / "cppmega_mlx/training/native_optim"),
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
