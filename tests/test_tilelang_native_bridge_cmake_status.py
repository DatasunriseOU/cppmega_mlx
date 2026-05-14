"""Status checks for the optional native TileLang MLX TVM-FFI CMake bridge."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BRIDGE_DIR = REPO_ROOT / "cppmega_mlx" / "nn" / "_tilelang" / "native_bridge"
TILELANG_ROOT = Path("/private/tmp/tl_apache_tvm_swap")


def _cache_value(cache_text: str, name: str) -> str:
    prefix = f"{name}:STRING="
    for line in cache_text.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    raise AssertionError(f"{name} missing from CMake cache")


def test_native_bridge_cmake_reports_missing_mlx_python_bridge(tmp_path: Path) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is not installed")

    missing_bridge = tmp_path / "missing" / "libmlx_python_bridge.dylib"
    build_dir = tmp_path / "build"

    result = subprocess.run(
        [
            cmake,
            "-S",
            str(NATIVE_BRIDGE_DIR),
            "-B",
            str(build_dir),
            f"-DTILELANG_ROOT={TILELANG_ROOT}",
            f"-DCPPMEGA_MLX_PYTHON_BRIDGE={missing_bridge}",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    cache_text = (build_dir / "CMakeCache.txt").read_text()
    assert (
        _cache_value(cache_text, "CPPMEGA_TILELANG_MLX_TVM_FFI_STATUS")
        == "blocked_missing_mlx_python_bridge"
    )
    todo = _cache_value(cache_text, "CPPMEGA_TILELANG_MLX_TVM_FFI_TODO")
    assert "libmlx_python_bridge.dylib" in todo
    assert "mlx_core_wrap_mx_array_move" in todo
    assert "dlsym" in todo
    assert "mx.fast.metal_kernel" in todo


def test_native_bridge_cmake_blocks_unaudited_c_api_version(tmp_path: Path) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is not installed")
    if not (TILELANG_ROOT / "build" / "lib" / "libtilelang_mlx_tvm_ffi_c_api.dylib").exists():
        pytest.skip("TileLang native MLX TVM-FFI C API library is not built")

    build_dir = tmp_path / "build"

    result = subprocess.run(
        [
            cmake,
            "-S",
            str(NATIVE_BRIDGE_DIR),
            "-B",
            str(build_dir),
            f"-DTILELANG_ROOT={TILELANG_ROOT}",
            "-DCPPMEGA_EXPECTED_TILELANG_MLX_TVM_FFI_C_API_VERSION=999",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    cache_text = (build_dir / "CMakeCache.txt").read_text()
    assert (
        _cache_value(cache_text, "CPPMEGA_TILELANG_MLX_TVM_FFI_STATUS")
        == "blocked_tilelang_c_api_version_mismatch"
    )
    assert "Audit cppmega native bridge" in _cache_value(
        cache_text,
        "CPPMEGA_TILELANG_MLX_TVM_FFI_TODO",
    )


def test_native_bridge_cmake_blocks_unaudited_c_api_abi_hash(tmp_path: Path) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is not installed")
    if not (TILELANG_ROOT / "build" / "lib" / "libtilelang_mlx_tvm_ffi_c_api.dylib").exists():
        pytest.skip("TileLang native MLX TVM-FFI C API library is not built")

    build_dir = tmp_path / "build"

    result = subprocess.run(
        [
            cmake,
            "-S",
            str(NATIVE_BRIDGE_DIR),
            "-B",
            str(build_dir),
            f"-DTILELANG_ROOT={TILELANG_ROOT}",
            "-DCPPMEGA_EXPECTED_TILELANG_MLX_TVM_FFI_C_API_ABI_HASH=wrong",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    cache_text = (build_dir / "CMakeCache.txt").read_text()
    assert (
        _cache_value(cache_text, "CPPMEGA_TILELANG_MLX_TVM_FFI_STATUS")
        == "blocked_tilelang_c_api_abi_mismatch"
    )
    assert "Audit cppmega native bridge" in _cache_value(
        cache_text,
        "CPPMEGA_TILELANG_MLX_TVM_FFI_TODO",
    )
