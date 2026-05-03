"""Verify the TVM_METAL_STORAGE_MODE env var.

Run two ways:

    # Default behaviour (Private; backward compatible).
    python test_metal_shared_storage.py
    # Opt-in to Shared (for zero-copy DLPack with MLX).
    TVM_METAL_STORAGE_MODE=shared python test_metal_shared_storage.py

Each invocation:
  1. Reads the resolved storage mode via the new
     ``metal.GetStorageMode`` FFI helper and verifies it matches the env var.
  2. Allocates a small device buffer (NDArray) and round-trips data
     through a trivial element-wise add kernel, asserting the numeric
     result is correct regardless of storage mode.

The MTLBuffer.storageMode property cannot be inspected from pure Python
without a custom ObjC bridge, so the live property check is performed
in C++ via the FFI helper above (which calls the same
GetMetalStorageOptions() that AllocDataSpace uses).
"""

from __future__ import annotations

import os
import sys

import pytest

try:
    import numpy as np
except ImportError as exc:
    np = None
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import tvm
except ImportError as exc:
    tvm = None
    _TVM_IMPORT_ERROR = exc
else:
    _TVM_IMPORT_ERROR = None

if tvm is not None:
    from tvm.script import ir as I
    try:
        from tvm.script import tirx as T
    except ImportError as exc:
        T = None
        _TIRX_IMPORT_ERROR = exc
    else:
        _TIRX_IMPORT_ERROR = None
else:
    I = None
    T = None
    _TIRX_IMPORT_ERROR = _TVM_IMPORT_ERROR


def _expected_mode() -> str:
    raw = os.environ.get("TVM_METAL_STORAGE_MODE", "").strip().lower()
    if raw in {"shared", "managed", "private"}:
        return raw
    return "private"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "private"),
        ("", "private"),
        ("shared", "shared"),
        ("Shared", "shared"),
        ("managed", "managed"),
        ("private", "private"),
        ("invalid", "private"),
    ],
)
def test_expected_mode_matches_runtime_parser_contract(monkeypatch, raw: str | None, expected: str) -> None:
    if raw is None:
        monkeypatch.delenv("TVM_METAL_STORAGE_MODE", raising=False)
    else:
        monkeypatch.setenv("TVM_METAL_STORAGE_MODE", raw)
    assert _expected_mode() == expected


def _check_storage_mode_ffi() -> str:
    get_mode = tvm.get_global_func("metal.GetStorageMode", allow_missing=False)
    actual = str(get_mode())
    expected = _expected_mode()
    assert actual == expected, (
        f"FFI metal.GetStorageMode returned '{actual}', "
        f"expected '{expected}' (TVM_METAL_STORAGE_MODE="
        f"{os.environ.get('TVM_METAL_STORAGE_MODE')!r})"
    )
    return actual


def _check_kernel_round_trip(dev) -> None:
    n = 1024

    @I.ir_module
    class Module:
        @T.prim_func
        def add(
            A: T.Buffer((n,), "float32"),
            B: T.Buffer((n,), "float32"),
            C: T.Buffer((n,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i in T.thread_binding(n, thread="threadIdx.x"):
                with T.sblock("C"):
                    v = T.axis.spatial(n, i)
                    T.reads(A[v], B[v])
                    T.writes(C[v])
                    C[v] = A[v] + B[v]

    fun = tvm.compile(Module, target="metal")

    a_np = np.random.RandomState(0).randn(n).astype("float32")
    b_np = np.random.RandomState(1).randn(n).astype("float32")
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.empty((n,), "float32", dev)
    fun(a, b, c)
    np.testing.assert_allclose(c.asnumpy(), a_np + b_np, rtol=1e-5, atol=1e-6)


def test_tvm_metal_storage_mode_env_matches_ffi_and_kernel_round_trips() -> None:
    if np is None:
        pytest.skip(f"requires NumPy: {_NUMPY_IMPORT_ERROR}")
    if tvm is None:
        pytest.skip(f"requires Apache TVM: {_TVM_IMPORT_ERROR}")
    if T is None:
        pytest.skip(f"requires apache TVM with tvm.script.tirx: {_TIRX_IMPORT_ERROR}")
    if not tvm.metal(0).exist:
        pytest.skip("Metal device unavailable on this host")
    mode = _check_storage_mode_ffi()
    assert mode == _expected_mode()
    _check_kernel_round_trip(tvm.metal(0))


def main() -> int:
    if np is None:
        print(f"NumPy unavailable, skipping: {_NUMPY_IMPORT_ERROR}")
        return 0
    if tvm is None:
        print(f"Apache TVM unavailable, skipping: {_TVM_IMPORT_ERROR}")
        return 0
    if T is None:
        print(f"Apache TVM tirx API unavailable, skipping: {_TIRX_IMPORT_ERROR}")
        return 0
    if not tvm.metal(0).exist:
        print("Metal device unavailable on this host, skipping.")
        return 0
    mode = _check_storage_mode_ffi()
    dev = tvm.metal(0)
    _check_kernel_round_trip(dev)
    print(f"OK: TVM_METAL_STORAGE_MODE resolves to '{mode}', kernel round-trip passes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
