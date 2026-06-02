"""Zero-copy MLX-CUDA -> TileLang(target='cuda') DLPack bridge (escape hatch).

MLX on a CUDA host (gb10 / sm_121) refuses to export a CUDA DLPack capsule:
``mx.array.__dlpack__()`` raises ``"CUDA DLPack export is not supported."`` and
``__dlpack_device__`` advertises ``kDLCUDAManaged (13)`` with ``device_id`` hardcoded
to 0 (see /Volumes/external/sources/mlx python/src/convert.cpp:~311 and array.cpp:522).

This module is the interim **Python escape hatch** described in
``cppmega.mlx/docs/DLPACK-CUDA-FIXES.md`` plan (B): instead of forking MLX C++, we
build a ``DLManagedTensor`` ourselves from the MLX array's device pointer
(``mx.array.data_ptr()``, MLX #3342) with the correct shape / strides / dtype and a
``DLDevice{kDLCUDA, device_id}``, wrap it in a PyCapsule named ``"dltensor"``, and
hand it straight to ``tvm.runtime.from_dlpack`` (tvm-ffi imports it zero-copy, no
host roundtrip). A C-callable deleter keeps a Python reference to the source MLX
array alive until the consumer releases the tensor, then drops it.

RULE #1 (fail loud, no silent fallback): every unsupported case RAISES with where +
what failed. There is no copy fallback inside this module — the caller chooses
between this zero-copy path and the eager-copy bridge explicitly (env-gated by
``CPPMEGA_TILELANG_CUDA_ZEROCOPY``). If the zero-copy capsule cannot be built or
imported, we raise so the bug is surfaced, never producing degraded/wrong output.

Device-type note: tvm-ffi ``from_dlpack`` accepts both ``kDLCUDA (2)`` and
``kDLCUDAManaged (13)``. We emit ``kDLCUDA (2)`` by default because the TileLang CUDA
codegen indexes a plain device pointer and the MLX CUDA allocation is addressable as
ordinary device memory on the unified GB10 part; this is overridable via
``CPPMEGA_TILELANG_CUDA_DLPACK_DEVICE_TYPE`` for A/B testing (2 vs 13).
"""

import ctypes
import os
from typing import Any

import mlx.core as mx

# ---------------------------------------------------------------------------
# DLPack C ABI (DLPack >= 0.5; matches tvm-ffi's expected layout)
# ---------------------------------------------------------------------------

kDLCPU = 1
kDLCUDA = 2
kDLCUDAHost = 3
kDLCUDAManaged = 13

# DLDataTypeCode
_kDLInt = 0
_kDLUInt = 1
_kDLFloat = 2
_kDLBfloat = 4


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


# DLManagedTensor: deleter(self) takes a DLManagedTensor*
_DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DLManagedTensorDeleter),
    ]


# Keep capsule-name buffers + per-capsule keepalive state alive for the program
# lifetime. Each managed tensor is held in this registry keyed by its address so
# the deleter (called from C with only the struct pointer) can find and release
# the Python keepalives (source array, shape/stride buffers, the struct itself).
_CAPSULE_NAME = b"dltensor"
_CAPSULE_NAME_USED = b"used_dltensor"
_REGISTRY: dict[int, dict[str, Any]] = {}


_MLX_DTYPE_TO_DL = {
    mx.float32: (_kDLFloat, 32),
    mx.float16: (_kDLFloat, 16),
    mx.bfloat16: (_kDLBfloat, 16),
    mx.int8: (_kDLInt, 8),
    mx.int16: (_kDLInt, 16),
    mx.int32: (_kDLInt, 32),
    mx.int64: (_kDLInt, 64),
    mx.uint8: (_kDLUInt, 8),
    mx.uint16: (_kDLUInt, 16),
    mx.uint32: (_kDLUInt, 32),
    mx.uint64: (_kDLUInt, 64),
    mx.bool_: (_kDLUInt, 8),
}


def zerocopy_enabled() -> bool:
    """Return whether the env opts into the zero-copy CUDA DLPack escape hatch."""

    return os.environ.get("CPPMEGA_TILELANG_CUDA_ZEROCOPY", "0") not in ("", "0", "false", "False")


def _dlpack_device_type() -> int:
    """Resolve the DLPack device_type to emit (kDLCUDA(2) default; 13 for A/B)."""

    raw = os.environ.get("CPPMEGA_TILELANG_CUDA_DLPACK_DEVICE_TYPE", "").strip()
    if not raw:
        return kDLCUDA
    val = int(raw)
    if val not in (kDLCUDA, kDLCUDAManaged):
        raise ValueError(
            f"_cuda_zerocopy: CPPMEGA_TILELANG_CUDA_DLPACK_DEVICE_TYPE={val!r} is not a CUDA "
            f"device type; expected {kDLCUDA} (kDLCUDA) or {kDLCUDAManaged} (kDLCUDAManaged)."
        )
    return val


def _cuda_device_id() -> int:
    """Resolve the CUDA ordinal the MLX array lives on.

    MLX hardcodes device_id=0 in ``__dlpack_device__`` (array.cpp:522), so MLX gives
    us nothing reliable. We trust torch's current CUDA device, overridable via
    ``CPPMEGA_TILELANG_CUDA_DEVICE_ID`` for multi-GPU. RAISES if neither resolves —
    a wrong ordinal would silently read another GPU's memory.
    """

    raw = os.environ.get("CPPMEGA_TILELANG_CUDA_DEVICE_ID", "").strip()
    if raw:
        return int(raw)
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
    except Exception as exc:  # noqa: BLE001 - report and fail loud
        raise RuntimeError(
            f"_cuda_zerocopy: cannot resolve CUDA device ordinal for the MLX array "
            f"(torch.cuda.current_device() failed: {type(exc).__name__}: {exc}); set "
            f"CPPMEGA_TILELANG_CUDA_DEVICE_ID explicitly."
        ) from exc
    raise RuntimeError(
        "_cuda_zerocopy: cannot resolve CUDA device ordinal (no torch.cuda); set "
        "CPPMEGA_TILELANG_CUDA_DEVICE_ID explicitly."
    )


def _synchronize_mlx_stream() -> None:
    """Flush MLX's CUDA stream so the device pointer holds materialized data.

    The DLPack stream contract requires the producer to make its writes visible to
    the consumer stream before handoff. MLX has no public per-array stream export
    and its CUDA backend uses a single work stream (PR #2075, sync model #2391), so
    the safe, correct handoff is a full ``mx.eval`` + ``mx.synchronize`` on the
    array's stream. tvm-ffi then launches on its own (default) CUDA stream; a device
    synchronize before handoff guarantees ordering without per-stream event plumbing
    (which MLX never exposed — issue #3548). RAISES on failure (no silent skip).
    """

    mx.synchronize()


def _build_managed_tensor(arr: "mx.array") -> int:
    """Build a heap ``DLManagedTensor`` for ``arr`` and return its address.

    Caller owns wrapping the returned address in a PyCapsule. The struct + shape /
    stride buffers + a reference to ``arr`` are held in ``_REGISTRY`` until the
    deleter fires.
    """

    if not isinstance(arr, mx.array):
        raise TypeError(f"_cuda_zerocopy: expected mlx.core.array, got {type(arr).__name__}")

    dl = _MLX_DTYPE_TO_DL.get(arr.dtype)
    if dl is None:
        raise TypeError(
            f"_cuda_zerocopy: unsupported MLX dtype {arr.dtype} for CUDA DLPack export."
        )
    code, bits = dl

    data_ptr = int(arr.data_ptr())
    if data_ptr == 0:
        raise RuntimeError(
            "_cuda_zerocopy: mx.array.data_ptr() returned NULL; cannot export a "
            "zero-copy CUDA DLPack capsule from an unallocated array."
        )

    shape = [int(d) for d in arr.shape]
    ndim = len(shape)
    # MLX strides are in elements; DLPack strides are in elements too.
    itemsize = bits // 8 if bits >= 8 else 1
    strides_elems = [int(s // itemsize) if itemsize else int(s) for s in arr.strides()] if hasattr(arr, "strides") else None
    if strides_elems is None or len(strides_elems) != ndim:
        # Row-major contiguous fallback strides (in elements).
        strides_elems = [0] * ndim
        acc = 1
        for i in range(ndim - 1, -1, -1):
            strides_elems[i] = acc
            acc *= shape[i]

    ShapeArr = ctypes.c_int64 * max(ndim, 1)
    shape_buf = ShapeArr(*shape) if ndim else ShapeArr(0)
    stride_buf = ShapeArr(*strides_elems) if ndim else ShapeArr(0)

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(data_ptr)
    managed.dl_tensor.device = _DLDevice(_dlpack_device_type(), _cuda_device_id())
    managed.dl_tensor.ndim = ndim
    managed.dl_tensor.dtype = _DLDataType(code, bits, 1)
    managed.dl_tensor.shape = ctypes.cast(shape_buf, ctypes.POINTER(ctypes.c_int64)) if ndim else None
    managed.dl_tensor.strides = ctypes.cast(stride_buf, ctypes.POINTER(ctypes.c_int64)) if ndim else None
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None

    addr = ctypes.addressof(managed)

    def _deleter(_self_ptr: int) -> None:
        # Drop all keepalives for this managed tensor. ``arr`` (and thus the MLX
        # device allocation) is released only here, after the consumer is done.
        _REGISTRY.pop(addr, None)

    c_deleter = _DLManagedTensorDeleter(_deleter)
    managed.deleter = c_deleter

    _REGISTRY[addr] = {
        "managed": managed,
        "shape_buf": shape_buf,
        "stride_buf": stride_buf,
        "deleter": c_deleter,
        "array": arr,  # keep the MLX allocation alive
    }
    return addr


# PyCapsule plumbing via the CPython C-API (ctypes).
_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def mlx_cuda_array_to_dlpack_capsule(arr: "mx.array") -> Any:
    """Build a ``"dltensor"`` PyCapsule wrapping ``arr``'s CUDA device buffer.

    The capsule is consumed by ``tvm.runtime.from_dlpack``. We do NOT install a
    capsule destructor: ownership/lifetime is managed by the ``DLManagedTensor``
    deleter (called by the consumer after import), which keeps ``arr`` alive until
    then. (If a consumer never imports the capsule, the keepalive lives for the
    process — acceptable for our per-call usage and avoids a double-free.)
    """

    _synchronize_mlx_stream()
    addr = _build_managed_tensor(arr)
    capsule = _PyCapsule_New(ctypes.c_void_p(addr), _CAPSULE_NAME, None)
    return capsule


def mlx_cuda_array_to_tvm_tensor(arr: "mx.array") -> Any:
    """Import an MLX-CUDA array as a TVM tensor view, zero-copy, via DLPack.

    RAISES (no copy fallback) so a broken zero-copy path is surfaced, per RULE #1.
    """

    from tilelang import tvm

    capsule = mlx_cuda_array_to_dlpack_capsule(arr)
    try:
        return tvm.runtime.from_dlpack(capsule)
    except Exception as exc:  # noqa: BLE001 - surface where + what failed
        raise RuntimeError(
            f"_cuda_zerocopy: tvm.runtime.from_dlpack rejected the MLX-CUDA capsule "
            f"(device_type={_dlpack_device_type()}, device_id={_cuda_device_id()}, "
            f"dtype={arr.dtype}, shape={tuple(int(d) for d in arr.shape)}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
