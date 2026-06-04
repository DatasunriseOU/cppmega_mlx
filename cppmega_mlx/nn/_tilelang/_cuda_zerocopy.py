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
# CUDA mempool release-threshold + trim shim (lever dlpack-fix).
#
# Root cause of the R1-e2e fp8-backward OOM (docs/RELAX-GRAPH-VS-MEGATRON.md §21):
# the error string ``cudaMallocAsync(&data, size, stream) failed: out of memory``
# is MLX's OWN allocator (mlx/backend/cuda/allocator.cpp:225) hitting genuine
# physical-memory exhaustion of the single GB10 117 GB unified pool. TWO
# stream-ordered cudaMallocAsync mempools (MLX's + torch/TE's) each reserve-and-
# retain unified memory that neither releases, because neither sets
# ``cudaMemPoolAttrReleaseThreshold`` (MLX only READS ReservedMemCurrent at
# allocator.cpp:247). The COMBINED reserved set exceeds physical memory.
#
# This shim (Python-only, NO MLX rebuild) sets the release threshold to 0 on the
# device mempool(s) and calls ``cudaMemPoolTrimTo(pool, 0)`` so reserved-but-idle
# unified memory is RETURNED to the OS before the competing allocation. It is
# called ONLY at a bridge boundary AFTER eval/sync (never mid-kernel) — see
# RISK 3 in the lever research. RULE #1: trimming only makes the zero-copy path
# FIT; it never introduces a host-copy / fp8->bf16 degrade. If libcudart cannot
# be loaded or a CUDA call fails, this RAISES with where+what (no silent skip) —
# the caller decides whether trimming is required for its budget.
# ---------------------------------------------------------------------------

_CUDA_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4  # cudaMemPoolAttrReleaseThreshold
_cudart = None  # lazily-loaded libcudart handle
_RELEASE_THRESHOLD_SET: set[int] = set()  # devices whose threshold we already set


def _load_cudart():
    """Load libcudart once; RAISE (RULE #1) with where+what if it cannot load."""

    global _cudart
    if _cudart is not None:
        return _cudart
    last_exc = None
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.13", "libcudart.so.11.0"):
        try:
            _cudart = ctypes.CDLL(name)
            break
        except OSError as exc:  # noqa: PERF203 - try the next soname
            last_exc = exc
    if _cudart is None:
        raise RuntimeError(
            "_cuda_zerocopy: cannot load libcudart for the CUDA mempool trim shim "
            f"(tried libcudart.so / .so.12 / .so.13 / .so.11.0): {last_exc}. The "
            "fp8 bridge needs cudaMemPoolTrimTo to fit the dual-pool budget."
        )
    # cudaGetDevice(int*), cudaDeviceGetDefaultMemPool(pool*, int dev),
    # cudaMemPoolSetAttribute(pool, attr, void*), cudaMemPoolTrimTo(pool, size_t)
    _cudart.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    _cudart.cudaGetDevice.restype = ctypes.c_int
    _cudart.cudaDeviceGetDefaultMemPool.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    _cudart.cudaDeviceGetDefaultMemPool.restype = ctypes.c_int
    _cudart.cudaMemPoolSetAttribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    _cudart.cudaMemPoolSetAttribute.restype = ctypes.c_int
    _cudart.cudaMemPoolTrimTo.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _cudart.cudaMemPoolTrimTo.restype = ctypes.c_int
    return _cudart


def _default_mempool(rt, device: int) -> ctypes.c_void_p:
    pool = ctypes.c_void_p()
    err = rt.cudaDeviceGetDefaultMemPool(ctypes.byref(pool), int(device))
    if err != 0:
        raise RuntimeError(
            f"_cuda_zerocopy: cudaDeviceGetDefaultMemPool(dev={device}) failed "
            f"(cudaError={err}); cannot trim the CUDA mempool for the fp8 bridge."
        )
    return pool


def trim_cuda_mempool(device: int | None = None) -> None:
    """Set ReleaseThreshold=0 and trim the default mempool to 0 on ``device``.

    Called at the fp8 bridge boundary (after eval/sync) to return reserved-but-idle
    unified memory so the competing MLX+torch+TE allocations fit the single GB10
    pool. RAISES (RULE #1) on any CUDA failure — never silently degrades. If the
    box genuinely has no idle reserved memory to trim this is a fast no-op (TrimTo
    returns cudaSuccess and reclaims nothing).
    """

    rt = _load_cudart()
    if device is None:
        dev = ctypes.c_int(0)
        err = rt.cudaGetDevice(ctypes.byref(dev))
        if err != 0:
            raise RuntimeError(
                f"_cuda_zerocopy: cudaGetDevice failed (cudaError={err}) resolving "
                "the device for the CUDA mempool trim."
            )
        device = int(dev.value)
    pool = _default_mempool(rt, device)
    # Set the release threshold to 0 ONCE per device so the driver does not retain
    # reserved memory across the trim (default threshold is UINT64_MAX = retain all).
    if device not in _RELEASE_THRESHOLD_SET:
        thresh = ctypes.c_uint64(0)
        err = rt.cudaMemPoolSetAttribute(
            pool,
            _CUDA_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            ctypes.cast(ctypes.byref(thresh), ctypes.c_void_p),
        )
        if err != 0:
            raise RuntimeError(
                f"_cuda_zerocopy: cudaMemPoolSetAttribute(ReleaseThreshold=0, "
                f"dev={device}) failed (cudaError={err}); cannot bound the CUDA "
                "mempool reservation for the fp8 bridge."
            )
        _RELEASE_THRESHOLD_SET.add(device)
    err = rt.cudaMemPoolTrimTo(pool, ctypes.c_size_t(0))
    if err != 0:
        raise RuntimeError(
            f"_cuda_zerocopy: cudaMemPoolTrimTo(dev={device}, 0) failed "
            f"(cudaError={err}); cannot return idle reserved unified memory."
        )

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


def relieve_bridge_memory_pressure(device: int | None = None) -> None:
    """Trim BOTH the MLX cache and the CUDA mempool at a bridge boundary.

    The fp8 backward crosses the MLX<->torch DLPack bridge while MLX's forward/scan
    working set and torch/TE's fp8 scratch both hold reserved unified memory. This
    helper, called immediately BEFORE a backward crossing (after the producer
    eval/sync), returns MLX's cached buffers (``mx.clear_cache``) and returns the
    CUDA mempool's reserved-but-idle unified memory (``trim_cuda_mempool``) so the
    next allocation fits the single GB10 pool. It is gated by
    ``CPPMEGA_FP8_BRIDGE_TRIM`` (default ON when unset, to fix the §21 OOM; set 0 to
    A/B the un-trimmed behavior). RULE #1: this only makes the zero-copy path FIT —
    it never host-copies or degrades. On a genuine CUDA failure it RAISES with
    where+what (no silent skip).
    """

    if os.environ.get("CPPMEGA_FP8_BRIDGE_TRIM", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    # MLX side: return cached (non-live) buffers so the unified pool shrinks.
    clear = getattr(mx, "clear_cache", None)
    if callable(clear):
        clear()
    # CUDA side: set ReleaseThreshold=0 + cudaMemPoolTrimTo(0) so the driver
    # returns reserved-but-idle unified memory shared with torch/TE's pool.
    trim_cuda_mempool(device)


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


def _mlx_native_cuda_dlpack_available(arr: "mx.array") -> bool:
    """Return True iff this MLX build natively exports a kDLCUDA DLPack capsule.

    The native C++ export (MLX ``feat(dlpack)`` / commit 6da6a0e4) advertises
    ``__dlpack_device__() == (kDLCUDA(2), id)`` and produces a real CUDA capsule
    from ``__dlpack__()``. Older MLX builds either lack ``__dlpack__`` or raise
    "CUDA DLPack export is not supported", in which case we fall through to the
    ``data_ptr()`` Python escape hatch below. Both are genuine zero-copy paths.
    """

    dl_dev = getattr(arr, "__dlpack_device__", None)
    if not callable(dl_dev) or not callable(getattr(arr, "__dlpack__", None)):
        return False
    try:
        dev = dl_dev()
    except Exception:
        return False
    return bool(dev) and int(dev[0]) in (kDLCUDA, kDLCUDAManaged)


def mlx_cuda_array_to_torch_tensor(arr: "mx.array") -> Any:
    """Import an MLX-CUDA array as a torch CUDA tensor view, zero-copy, via DLPack.

    This is the device-view (no host roundtrip) counterpart to the numpy-copy
    bridge in ``_cuda_eager._mlx_to_torch_cuda``. It feeds the TileLang
    torch-backend kernel interfaces (``sparse_mla_fwd_interface`` /
    ``sparse_mla_bwd``) a real device view of the MLX allocation — no copy, no
    host bounce.

    Two genuine zero-copy mechanisms, in preference order (NOT a silent
    degrade — both are real zero-copy device views):

    1. **Native MLX kDLCUDA export** (commit 6da6a0e4): hand the MLX array
       straight to ``torch.from_dlpack``, which calls its ``__dlpack__()``.
    2. **``data_ptr()`` Python escape hatch**: build the kDLCUDA
       ``DLManagedTensor`` capsule from ``mx.array.data_ptr()`` (MLX #3342) and
       import it. Used only when the native export is absent.

    RAISES (no copy fallback) so a broken zero-copy path is surfaced, per RULE #1.
    """

    import torch
    from torch.utils.dlpack import from_dlpack as _torch_from_dlpack

    if _mlx_native_cuda_dlpack_available(arr):
        # Native zero-copy: torch.from_dlpack consumes MLX's own __dlpack__()
        # kDLCUDA capsule. MLX must flush its CUDA stream before handoff.
        _synchronize_mlx_stream()
        try:
            t = torch.from_dlpack(arr)
        except Exception as exc:  # noqa: BLE001 - surface where + what failed
            raise RuntimeError(
                f"_cuda_zerocopy: torch.from_dlpack rejected the MLX native "
                f"kDLCUDA export (dtype={arr.dtype}, "
                f"shape={tuple(int(d) for d in arr.shape)}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    else:
        if not hasattr(arr, "data_ptr"):
            raise RuntimeError(
                "_cuda_zerocopy: this MLX build neither exports a native kDLCUDA "
                "DLPack capsule (__dlpack__) nor exposes mx.array.data_ptr(); "
                "the zero-copy MLX->torch bridge cannot run. Build MLX at the "
                "kDLCUDA-export commit (6da6a0e4) or a build with data_ptr() "
                "(MLX #3342)."
            )
        capsule = mlx_cuda_array_to_dlpack_capsule(arr)
        try:
            t = _torch_from_dlpack(capsule)
        except Exception as exc:  # noqa: BLE001 - surface where + what failed
            raise RuntimeError(
                f"_cuda_zerocopy: torch.utils.dlpack.from_dlpack rejected the "
                f"MLX-CUDA data_ptr() capsule (device_type={_dlpack_device_type()}, "
                f"device_id={_cuda_device_id()}, dtype={arr.dtype}, "
                f"shape={tuple(int(d) for d in arr.shape)}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    if not t.is_cuda:
        raise RuntimeError(
            "_cuda_zerocopy: torch imported the MLX-CUDA DLPack as a non-CUDA "
            f"tensor (device={t.device}); the zero-copy bridge requires a CUDA "
            "device view."
        )
    del torch
    return t


# ---------------------------------------------------------------------------
# OUTPUT side: torch CUDA result -> mx.array, zero-copy, no host bounce
# ---------------------------------------------------------------------------
#
# Symmetric counterpart to ``mlx_cuda_array_to_torch_tensor`` (the INPUT bridge).
# A TileLang/torch-CUDA kernel writes its result into a torch CUDA tensor; this
# imports it straight back into an ``mx.array`` via DLPack with NO ``.cpu()``
# host roundtrip. It relies on the native MLX CUDA *import* (DatasunriseOU fork:
# convert.cpp ``cuda_dlpack_to_mlx`` + cuda backend
# ``copy_external_to_mlx_buffer``), which copies the foreign CUDA buffer into a
# fresh MLX-owned GPU allocation with a single on-device ``cudaMemcpy`` (no PCIe
# host bounce, and DEADLOCK-FREE under repeated imports: no foreign buffer / no
# Python owner ever enters MLX's scheduler — see docs/SPARSE-MLA-PATHC-WIRED.md).
#
# RULE #1: if the running MLX build lacks the CUDA import (older binary that
# still raises "CUDA DLPack import is not supported"), we RAISE with where+what
# instead of silently host-bouncing. The caller (``_cuda_eager``) chooses this
# zero-copy writeback only when the build supports it.


def mlx_cuda_import_available() -> bool:
    """Return True iff the running MLX build can import a CUDA DLPack capsule.

    Probes the native import by round-tripping a tiny torch CUDA tensor through
    ``mx.array(t)`` (DLPack). Returns False (not raise) for the *capability*
    probe; actual writebacks raise on failure.
    """

    try:
        import torch
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.ones(4, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        a = mx.array(t)
        mx.eval(a)
        return bool(abs(float(a.sum().item()) - 4.0) < 1e-4)
    except Exception:
        return False


def torch_cuda_tensor_to_mlx(t, out_dtype=None):
    """Import a torch CUDA tensor into an ``mx.array`` zero-copy (no host bounce).

    Replaces the ``t.detach().cpu().numpy()`` writeback in
    ``_cuda_eager._torch_cuda_to_mlx``: the kernel result stays GPU-resident and
    crosses the torch->MLX boundary via DLPack (torch ``__dlpack__`` consumed by
    the native MLX CUDA import). A single device-side copy lands it in an
    MLX-owned buffer; there is no ``.cpu()`` / PCIe roundtrip.

    RAISES (RULE #1) if the tensor is not a CUDA tensor or the MLX build cannot
    import the CUDA DLPack capsule -- never silently degrades to a host copy.
    """

    import torch

    if not isinstance(t, torch.Tensor):
        raise TypeError(
            f"_cuda_zerocopy.torch_cuda_tensor_to_mlx: expected a torch.Tensor, "
            f"got {type(t).__name__}."
        )
    if not t.is_cuda:
        raise RuntimeError(
            f"_cuda_zerocopy.torch_cuda_tensor_to_mlx: tensor is on {t.device}, "
            f"not CUDA; the zero-copy MLX import requires a CUDA device tensor."
        )

    # DLPack producer-readiness contract: flush torch's CUDA stream so the
    # buffer is materialized before MLX reads it. MLX import does a full device
    # sync of its own before the device-side copy, but we sync the producer here
    # to honor the handoff ordering explicitly.
    src = t.detach().contiguous()
    torch.cuda.synchronize()

    try:
        # mx.array(torch_tensor) routes through MLX create_array ->
        # nd_array_to_mlx -> (DatasunriseOU) cuda_dlpack_to_mlx, consuming the
        # tensor's __dlpack__ kDLCUDA capsule with no host roundtrip.
        arr = mx.array(src)
    except Exception as exc:  # noqa: BLE001 - surface where + what failed
        raise RuntimeError(
            f"_cuda_zerocopy.torch_cuda_tensor_to_mlx: mx.array() rejected the "
            f"torch CUDA DLPack capsule (dtype={src.dtype}, "
            f"shape={tuple(int(d) for d in src.shape)}). This MLX build likely "
            f"lacks the native CUDA DLPack import (cuda_dlpack_to_mlx); rebuild "
            f"MLX-CUDA with the import patch. Underlying error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if out_dtype is not None and arr.dtype != out_dtype:
        arr = arr.astype(out_dtype)
    del torch
    return arr
