"""Local Metal capability probes for the FP8 upstream patch motivation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


XCRUN = shutil.which("xcrun")


def _requires_xcrun() -> None:
    if XCRUN is None:
        pytest.skip("xcrun is unavailable; Metal SDK capability probe is macOS-only")


def _metal_compile(tmp_path: Path, name: str, source: str) -> subprocess.CompletedProcess[str]:
    _requires_xcrun()
    path = tmp_path / name
    output = tmp_path / f"{Path(name).stem}.air"
    path.write_text(source, encoding="utf-8")
    return subprocess.run(
        [XCRUN or "xcrun", "--sdk", "macosx", "metal", "-c", str(path), "-o", str(output)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_metal_accepts_uint8_storage_for_fp8_bytes(tmp_path: Path) -> None:
    result = _metal_compile(
        tmp_path,
        "uchar_storage.metal",
        """
#include <metal_stdlib>
using namespace metal;

kernel void copy_bytes(device uchar* src [[buffer(0)]],
                       device uchar* dst [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
  dst[tid] = src[tid];
}
""",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("name", "source", "expected_errors"),
    [
        (
            "float8_t.metal",
            """
#include <metal_stdlib>
using namespace metal;

kernel void bad(device float8_t* src [[buffer(0)]],
                device float8_t* dst [[buffer(1)]],
                uint tid [[thread_position_in_grid]]) {
  dst[tid] = src[tid];
}
""",
            ("unknown type name 'float8_t'",),
        ),
        (
            "float8_scalar.metal",
            """
#include <metal_stdlib>
using namespace metal;

kernel void bad(device float8* src [[buffer(0)]],
                device float8* dst [[buffer(1)]],
                uint tid [[thread_position_in_grid]]) {
  dst[tid] = src[tid];
}
""",
            ("Do_not_use_float8",),
        ),
        (
            "simdgroup_uchar.metal",
            """
#include <metal_stdlib>
using namespace metal;

kernel void bad(uint tid [[thread_position_in_grid]]) {
  simdgroup_matrix<uchar, 8, 8> tile;
  (void)tile;
}
""",
            ("simdgroup_matrix_storage<metal::uchar", "simdgroup_matrix<uchar"),
        ),
    ],
)
def test_metal_rejects_native_fp8_and_uchar_simdgroup_matrix(
    tmp_path: Path,
    name: str,
    source: str,
    expected_errors: tuple[str, ...],
) -> None:
    result = _metal_compile(tmp_path, name, source)
    assert result.returncode != 0, result.stdout
    assert any(expected_error in result.stderr for expected_error in expected_errors), result.stderr


def test_metal_headers_do_not_advertise_fp8_tensor_datatype() -> None:
    _requires_xcrun()
    sdk = subprocess.run(
        [XCRUN or "xcrun", "--sdk", "macosx", "--show-sdk-path"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    header_root = Path(sdk) / "System/Library/Frameworks/Metal.framework/Headers"
    matches = []
    for header in header_root.rglob("*.h"):
        text = header.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in ("MTLTensorDataTypeFloat8", "MTLTensorDataTypeFP8", "Float8E4M3", "Float8E5M2")):
            matches.append(header)
    assert matches == []
