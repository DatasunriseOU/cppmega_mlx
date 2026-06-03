"""PR 2 of docs/RELAX-GRAPH-MEMORY-PATH.md: the physical-bank -> logical-buffer
DPS adapter that makes a REAL path_c PrimFunc usable as a Relax-graph leaf, so the
whole train step can be assembled as ONE Relax @R.function with global memory
planning (StaticPlanBlockMemory).

WHY AN EXTERNAL-FUNCTION (call_dps_packed) BOUNDARY, NOT A call_tir LEAF
-----------------------------------------------------------------------
PR 1 (scratch/test_call_tir_dps.py) found the 3 DPS mismatches that block the real
physical-ABI path_c prim from being an R.call_tir leaf. PR 2 MEASURED whether each
is fixable inside a generic-TIR adapter:

  (1) PARAM ORDER -- scalar ``path_c_run_backward: T.int32`` at param index 5 (the
      middle), not last.  ==> CLOSED by specialization: ``prim.specialize({run_backward:
      const})`` yields a 16-param prim with ZERO scalar params (no mid-param scalar).
      Verified: scratch/pr2_test_curry.py prints "scalars: []".

  (2) NO TRAILING OUTPUT BUFFER -- logical tensors are packed into disjoint RANGES
      of a few large shared physical dtype banks, read+written IN PLACE
      (``tilelang_out_idx = [0,2,3,4,6,...,16]`` -- nearly every param is an output).
      ==> CLOSED by the adapter ABI below: it presents logical inputs as read-only
      Relax tensors and the logical output as a single trailing Relax tensor; the
      physical-bank packing is an INTERNAL detail of the packed function, not the
      call ABI.

  (3) NOT A GENERIC-TIR KERNEL -- the TileLang ``T.Kernel(64, threads=1024)`` body
      guards ``T.alloc_shared`` accesses inside conditionals; generic relax/s_tir
      RAISES "Cannot insert syncs inside condition" (thread_storage_sync.cc:145).
      ==> This is a HARD WALL and it does NOT close by specialization: even after
      currying run_backward to a constant the s_tir build STILL raises (the row-chunk
      dispatch guards remain). MEASURED: scratch/pr2_test_curry.py.
      BUT the SAME prim lowers cleanly through ``tilelang.compile`` (target=metal)
      in ~2 s, emitting a 168 KB Metal kernel + a callable JITKernel.
      MEASURED: scratch/pr2_compile_full.py.

  DECISION (this is the deliverable's pinned outcome): because of (3), the real
  path_c kernel can ONLY be lowered via ``tilelang.compile`` -- it can NEVER be
  inlined into a generic-TIR ``R.call_tir`` leaf. The correct Relax-graph boundary
  is therefore an EXTERNAL FUNCTION: ``R.call_dps_packed("<name>", [logical_inputs],
  out_sinfo)``. The named packed function:
      * allocates / owns the physical banks,
      * packs the logical inputs into their bank sub-ranges (per the prim's
        ``tl.fusion.physical_abi.logical_to_physical`` map),
      * invokes the tilelang.compile'd kernel (the real path_c compute) on the banks,
      * unpacks the logical-output sub-range into the trailing DPS output tensor.
  This closes (1) (run_backward curried per-specialized-kernel), (2) (logical I/O is
  the ABI; banks are internal), and (3) (kernel is the tilelang artifact, not s_tir).

  call_dps_packed outputs ARE Relax-level tensors that CallTIRRewrite materialises as
  ``builtin.alloc_tensor`` and ``StaticPlanBlockMemory`` co-plans -- so the planner
  STILL sees and reuses each region's logical working set across the assembled
  @R.function, exactly as PR 1's call_tir leaves did. (The internal physical banks
  are NOT Relax-visible; co-planning those is the explicit further step noted in the
  doc -- expose banks as Relax tensors -- but the logical-output liveness, which is
  what drives the cross-layer fwd/bwd concurrency win, is fully planned here.)

This module provides:
  * ``PathCRegionLeaf`` -- a curried (fwd-only / bwd-only) real path_c region with
    its tilelang-compiled kernel, logical I/O signature, and physical-bank map.
  * ``register_region_dps_packed`` -- register the pack/kernel/unpack packed func.
  * ``emit_region_call`` -- emit ``R.call_dps_packed`` for the region into a
    BlockBuilder, returning the logical output Var.

RULE #1 (fail loud): every stage asserts; mismatched shapes / missing banks RAISE.

Run the self-test:
    TVM_LIBRARY_PATH=/Volumes/external/sources/tilelang/build/lib \\
    PYTHONPATH=/Volumes/external/sources/cppmega.mlx \\
    <python> -m cppmega_mlx.runtime.path_c_dps_adapter
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import tvm
import tvm_ffi
from tvm import relax, tir


# --------------------------------------------------------------------------- #
# PR-7 DEVICE-RESIDENT primitives (the host-bounce elimination).
#
# The doc's RELAX-GRAPH-VS-MEGATRON lever: the per-step hot path round-tripped
# every ~2028 MB physical bank to HOST NUMPY across the call_dps_packed boundary
# each call (``.numpy()`` on read, host->device ``copyto`` on writeback). That host
# bounce -- NOT the device kernels -- is 96.9% of the step time.
#
# These helpers keep the banks DEVICE-RESIDENT end to end: bank sub-range pack/unpack
# is a ZERO-COPY device VIEW (``Tensor._create_view`` with a byte offset) + a
# device->device ``copyto``; banks are allocated ONCE via ``tvm.runtime.empty`` on the
# device; the tilelang JITKernel is fed the device banks directly (the scout's call
# convention) and mutates them IN PLACE at ``kernel.out_idx``. No np.ndarray /
# np.from_dlpack / .numpy() / host copyto appears in the per-region compute path.
#
# RULE #1 (fail loud): every helper asserts dtype/size/device; mismatches RAISE.
# --------------------------------------------------------------------------- #
_DTYPE_ITEMSIZE = {"float32": 4, "float16": 2, "bfloat16": 2, "int32": 4, "int64": 8}


def _itemsize(dtype: str) -> int:
    if dtype not in _DTYPE_ITEMSIZE:
        raise RuntimeError(
            f"FAIL-LOUD: device-resident path needs a known itemsize for dtype "
            f"{dtype!r}; add it to _DTYPE_ITEMSIZE")
    return _DTYPE_ITEMSIZE[dtype]


def _dlpack_device_type(arg: Any) -> int | None:
    """The DLPack device type of a tvm tensor, read from ``__dlpack_device__()`` ->
    (device_type, device_id). DLDeviceType: kDLCPU=1, kDLCUDA=2, kDLMetal=8, ... The
    ``Tensor.device`` object on this build does NOT expose a usable ``device_type``
    attribute (only a ``dlpack_device_type`` method + name), so we read the DLPack pair
    directly -- the authoritative, build-stable device signal."""
    fn = getattr(arg, "__dlpack_device__", None)
    if fn is None:
        return None
    try:
        return int(fn()[0])
    except Exception:  # noqa: BLE001
        return None


def is_device_tensor(arg: Any) -> bool:
    """True iff ``arg`` is a tvm device tensor that supports the zero-copy device
    view + device->device copy ABI (``_create_view`` and ``copyto``). The Relax VM
    materialises every call_dps_packed arg as a ``tvm.runtime.Tensor`` (registered
    globally via _set_class_tensor), which carries both -- so on a CUDA VM the bank
    args arrive ready for the device-resident path, NO numpy needed."""

    return hasattr(arg, "_create_view") and hasattr(arg, "copyto")


def device_bank_view(bank: Any, offset: int, size: int, dtype: str) -> Any:
    """A ZERO-COPY flat (size,) device VIEW of ``bank[offset:offset+size]`` (same
    allocation, no copy). ``offset``/``size`` are ELEMENT counts; the view's byte
    offset is ``offset * itemsize``. RULE #1: out-of-range RAISES."""

    bank_numel = int(np.prod([int(d) for d in bank.shape]))
    if offset < 0 or offset + size > bank_numel:
        raise RuntimeError(
            f"FAIL-LOUD: bank sub-range [{offset}:{offset+size}] out of bank numel "
            f"{bank_numel}")
    return bank._create_view((size,), dtype, relative_byte_offset=offset * _itemsize(dtype))


def device_pack(bank: Any, src: Any, offset: int, size: int, dtype: str) -> None:
    """Pack a device tensor ``src`` (numel==size) into ``bank[offset:offset+size]`` via
    a device->device copy into a zero-copy view. NO host traffic. RULE #1: RAISES on
    size/device mismatch."""

    src_numel = int(np.prod([int(d) for d in src.shape]))
    if src_numel != size:
        raise RuntimeError(
            f"FAIL-LOUD: device_pack src numel {src_numel} != ABI sub-range size {size}")
    view = device_bank_view(bank, offset, size, dtype)
    # src is (S0,S1,..); view is flat (size,). copyto compares numel, both contiguous.
    src_flat = src._create_view((size,), dtype, relative_byte_offset=0)
    src_flat.copyto(view)


def device_unpack(bank: Any, dst: Any, offset: int, size: int, dtype: str) -> None:
    """Unpack ``bank[offset:offset+size]`` into device tensor ``dst`` (numel==size) via
    a device->device copy from a zero-copy view. NO host traffic. RULE #1: RAISES."""

    dst_numel = int(np.prod([int(d) for d in dst.shape]))
    if dst_numel != size:
        raise RuntimeError(
            f"FAIL-LOUD: device_unpack dst numel {dst_numel} != ABI sub-range size {size}")
    view = device_bank_view(bank, offset, size, dtype)
    dst_flat = dst._create_view((size,), dtype, relative_byte_offset=0)
    view.copyto(dst_flat)


def alloc_device_banks(bank_shapes: dict[str, int], device: Any,
                       dtype: str = "float32") -> dict[str, Any]:
    """Allocate the physical banks ONCE, device-resident (``tvm.runtime.empty`` on
    ``device``), zero-initialised. Returns {bank: tvm.runtime.Tensor}. The banks are
    reused across calls (the kernel mutates them in place), so this is hoisted OUT of
    the per-step hot path. NO numpy host array is materialised for the bank storage."""

    banks: dict[str, Any] = {}
    for b, n in bank_shapes.items():
        t = tvm.runtime.empty((int(n),), dtype, device=device)
        _zero_device_tensor(t)
        banks[b] = t
    return banks


_ZERO_HOST_CACHE: dict[tuple, np.ndarray] = {}


def _zero_device_tensor(t: Any) -> None:
    """Zero a device tensor in place WITHOUT a per-call host allocation: copy from a
    cached host-zero buffer (allocated once per shape). This runs ONCE per bank at
    setup (not in the hot path), so the one-time host->device zero-fill is amortised."""

    key = (tuple(int(d) for d in t.shape), str(t.dtype))
    z = _ZERO_HOST_CACHE.get(key)
    if z is None:
        z = np.zeros(key[0], dtype=np.dtype(key[1]))
        _ZERO_HOST_CACHE[key] = z
    t.copyfrom(z)


# --------------------------------------------------------------------------- #
# Physical-ABI introspection (reads the REAL prim's metadata -- no fabrication)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LogicalBufferMap:
    """One logical tensor's placement inside a physical bank sub-range."""

    name: str
    bank: str
    offset: int
    size: int
    logical_shape: tuple[int, ...]
    dtype: str


def parse_logical_to_physical(prim: tir.PrimFunc) -> dict[str, LogicalBufferMap]:
    """Parse the ``tl.fusion.physical_abi.logical_to_physical`` attr off a real
    path_c prim into a {logical_name: LogicalBufferMap} dict. RAISES if absent."""

    if prim.attrs is None or "tl.fusion.physical_abi.logical_to_physical" not in prim.attrs:
        raise RuntimeError(
            "FAIL-LOUD: prim carries no tl.fusion.physical_abi.logical_to_physical "
            "attr -- not a physical-ABI path_c prim"
        )
    raw = prim.attrs["tl.fusion.physical_abi.logical_to_physical"]
    table = json.loads(str(raw))
    out: dict[str, LogicalBufferMap] = {}
    for name, spec in table.items():
        out[name] = LogicalBufferMap(
            name=name,
            bank=str(spec["bank"]),
            offset=int(spec["offset"]),
            size=int(spec["size"]),
            logical_shape=tuple(int(d) for d in spec["logical_shape"]),
            dtype=str(spec["dtype"]),
        )
    return out


def parse_physical_bank_shapes(prim: tir.PrimFunc) -> dict[str, int]:
    """Parse ``tl.fusion.physical_abi.physical_buffer_shapes`` -> {bank: numel}."""

    raw = prim.attrs["tl.fusion.physical_abi.physical_buffer_shapes"]
    table = json.loads(str(raw))
    return {str(k): int(v[0]) for k, v in table.items()}


def prim_bank_param_order(prim: tir.PrimFunc) -> list[str]:
    """The bank/handle param names IN ORDER (the physical kernel ABI order),
    skipping the scalar gate param. The tilelang kernel is invoked positionally
    in this order."""

    order = []
    for p in prim.params:
        if p in prim.buffer_map:
            order.append(prim.buffer_map[p].name)
    return order


# --------------------------------------------------------------------------- #
# A real path_c region, curried + tilelang-compiled, as a Relax-graph leaf
# --------------------------------------------------------------------------- #
@dataclass
class PathCRegionLeaf:
    """A REAL path_c region specialized to fwd-only or bwd-only, with the real
    tilelang-compiled kernel and the logical I/O signature the adapter exposes."""

    name: str
    run_backward: int
    prim: tir.PrimFunc                 # the specialized (no-scalar) physical-ABI prim
    kernel: Any                         # tilelang JITKernel (real compiled path_c kernel)
    logical_map: dict[str, LogicalBufferMap]
    bank_shapes: dict[str, int]
    bank_param_order: list[str]
    logical_inputs: tuple[str, ...]    # logical input tensor names (read-only)
    logical_output: str                 # the single logical output tensor name (DPS)


# --------------------------------------------------------------------------- #
# The DPS packed function: logical I/O ABI over the physical banks + real kernel
# --------------------------------------------------------------------------- #
def make_region_dps_packed(leaf: PathCRegionLeaf) -> Callable[..., Any]:
    """Build the packed function implementing the logical->physical DPS adapter for
    ``leaf``. Signature (call_dps_packed ABI): (logical_in_0, ..., logical_in_k,
    logical_out) -- inputs first, single OUTPUT LAST, banks INTERNAL.

    Body: pack each logical input into its bank sub-range, invoke the real tilelang
    kernel on the banks (in the kernel's positional bank order), unpack the
    logical-output sub-range into ``logical_out``.

    TWO PATHS (RULE #1: one clear path each, no silent fallback):
      * DEVICE-RESIDENT (the gb10 path): when the call_dps_packed ABI tensors arrive
        as device tensors (a CUDA/Metal Relax VM materialises them as
        ``tvm.runtime.Tensor`` with ``_create_view``/``copyto``), pack/unpack is a
        ZERO-COPY device VIEW + device->device copy, the banks stay device-resident
        (allocated ONCE in the driver), and the tilelang JITKernel mutates them in
        place. NO numpy host array in the hot path -- this is the host-bounce removal.
      * NUMPY-REFERENCE (the CPU self-test path): when the ABI tensors are host
        tensors (a CPU/LLVM VM), pack/unpack is numpy slicing so the adapter plumbing
        is testable off-device. This is gated by the arg DEVICE, not a try/except.
    """

    lmap = leaf.logical_map
    bank_shapes = leaf.bank_shapes

    def _packed(*args: Any) -> None:
        if len(args) != len(leaf.logical_inputs) + 1:
            raise RuntimeError(
                f"FAIL-LOUD: {leaf.name} DPS packed expected "
                f"{len(leaf.logical_inputs)+1} args "
                f"(inputs {len(leaf.logical_inputs)} + 1 output), got {len(args)}"
            )
        out_tensor = args[-1]
        # Route by the OUTPUT tensor's device residency (a clear gate, not a fallback):
        # a device tensor (CUDA/Metal VM) takes the device-resident path; otherwise the
        # numpy reference path (CPU/LLVM VM self-test).
        if is_device_tensor(out_tensor) and _output_is_device(out_tensor):
            _packed_device_resident(leaf, args)
        else:
            _packed_numpy_reference(leaf, args, bank_shapes, lmap)

    return _packed


def _output_is_device(t: Any) -> bool:
    """True iff the tensor lives on a non-CPU device (so the device-resident path
    applies). CPU tensors (dlpack device_type 1) take the numpy reference path."""

    dt = _dlpack_device_type(t)
    # DLDeviceType.kDLCPU == 1. Anything else (CUDA=2, Metal=8, ...) is a real device.
    return dt is not None and dt != 1


def _packed_device_resident(leaf: PathCRegionLeaf, args: tuple) -> None:
    """DEVICE-RESIDENT body: pack logical inputs into the driver-owned device banks via
    zero-copy views, run the real kernel on the device banks (in place), unpack the
    logical output via a device view. NO numpy in the hot path."""

    lmap = leaf.logical_map
    out_tensor = args[-1]
    dev = out_tensor.device
    banks = _device_banks_for(leaf, dev)  # allocated ONCE, reused across calls

    for lname, arr in zip(leaf.logical_inputs, args[:-1]):
        m = lmap[lname]
        device_pack(banks[m.bank], arr, m.offset, m.size, m.dtype)

    _drive_region_compute_device(leaf, banks, dev)

    m = lmap[leaf.logical_output]
    device_unpack(banks[m.bank], out_tensor, m.offset, m.size, m.dtype)


def _packed_numpy_reference(leaf: PathCRegionLeaf, args: tuple,
                            bank_shapes: dict[str, int],
                            lmap: dict[str, "LogicalBufferMap"]) -> None:
    """NUMPY-REFERENCE body (CPU self-test ONLY): host numpy pack/unpack around the
    abstract region driver. Kept so the adapter plumbing is testable off gb10."""

    in_arrays = [_dps_arg_to_host_numpy(a) for a in args[:-1]]
    out_tensor = args[-1]
    banks = {b: np.zeros((n,), dtype=np.float32) for b, n in bank_shapes.items()}
    for lname, arr in zip(leaf.logical_inputs, in_arrays):
        m = lmap[lname]
        flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
        if flat.size != m.size:
            raise RuntimeError(
                f"FAIL-LOUD: logical input {lname} numel {flat.size} != "
                f"ABI sub-range size {m.size}")
        banks[m.bank][m.offset : m.offset + m.size] = flat
    _drive_region_compute(leaf, banks)
    m = lmap[leaf.logical_output]
    out_host_shape = tuple(int(d) for d in out_tensor.shape)
    out_flat = banks[m.bank][m.offset : m.offset + m.size]
    _dps_writeback_host_to_arg(out_tensor, out_flat.reshape(out_host_shape))


# Per-leaf device bank cache -- banks allocated ONCE per (leaf, device) and reused
# across every call (the kernel mutates them in place at out_idx, so device-resident
# banks ARE the rolling state). This is the hoist that removes the per-call np.zeros
# bank allocation the scout measured as the dominant non-H2D/D2H cost.
_DEVICE_BANKS: dict[tuple, dict[str, Any]] = {}


def _device_banks_for(leaf: PathCRegionLeaf, device: Any) -> dict[str, Any]:
    key = (id(leaf), repr(device))  # repr is stable: device(type='cuda', index=0)
    banks = _DEVICE_BANKS.get(key)
    if banks is None:
        banks = alloc_device_banks(leaf.bank_shapes, device)
        _DEVICE_BANKS[key] = banks
    return banks


def _dps_arg_to_host_numpy(arg: Any) -> np.ndarray:
    """Import a call_dps_packed ABI tensor to a host numpy array, device-agnostically.

    CPU/Metal DLPack-capable tensors import zero-copy via ``np.from_dlpack``; CUDA
    tensors (whose device DLPack export is rejected by numpy with "Unsupported
    device in DLTensor") are copied to host via the tensor's ``.numpy()`` method.
    RULE #1 (fail loud): if BOTH paths fail the original errors are surfaced."""

    try:
        return np.from_dlpack(arg)
    except Exception as dlpack_err:  # noqa: BLE001 -- CUDA: Unsupported device in DLTensor
        to_numpy = getattr(arg, "numpy", None)
        if to_numpy is None:
            raise RuntimeError(
                "FAIL-LOUD: DPS arg is neither host-DLPack-importable nor has a "
                f".numpy() host-copy method (type={type(arg).__name__}); "
                f"np.from_dlpack raised: {dlpack_err}"
            ) from dlpack_err
        return np.ascontiguousarray(to_numpy())


def _dps_writeback_host_to_arg(out_tensor: Any, host_result: np.ndarray) -> None:
    """Write a host numpy result back into a call_dps_packed output tensor,
    device-agnostically.

    For a CPU/Metal tensor that aliases a host numpy view we assign through that
    view; for a CUDA device tensor (which does NOT alias host memory) we build a
    same-device source tensor and ``copyto`` the device output in place. RULE #1:
    if neither path is available we RAISE."""

    host_result = np.ascontiguousarray(host_result, dtype=np.float32)
    # CPU/Metal: alias the output's host buffer and assign in place.
    try:
        view = np.from_dlpack(out_tensor)
        view[...] = host_result.reshape(view.shape)
        return
    except Exception:  # noqa: BLE001 -- CUDA: device DLPack rejected by numpy
        pass
    # CUDA (and any device tensor with a .device + copyto): host->device copy.
    dev = getattr(out_tensor, "device", None)
    copyto = getattr(out_tensor, "copyto", None)
    if dev is None or copyto is None:
        raise RuntimeError(
            "FAIL-LOUD: DPS output tensor is not host-DLPack-aliasable and lacks "
            f"a (.device, .copyto) device-writeback path (type={type(out_tensor).__name__})"
        )
    src = tvm.runtime.tensor(host_result, device=dev)
    src.copyto(out_tensor)


# Hook the region compute. Default = a transparent reference matching the region's
# logical semantics (so the adapter ABI is testable on CPU without a live GPU); the
# real path is set by ``set_region_kernel_driver`` to call ``leaf.kernel`` on device.
_REGION_DRIVER: Callable[[PathCRegionLeaf, dict[str, np.ndarray]], None] | None = None

# The DEVICE-RESIDENT region compute hook: takes the driver-owned DEVICE banks (dict of
# tvm.runtime.Tensor on the device) and the device, runs the real kernel on them IN
# PLACE. NO numpy. Set by ``set_region_device_driver`` (the gb10 path).
_REGION_DEVICE_DRIVER: Callable[[PathCRegionLeaf, dict[str, Any], Any], None] | None = None


def set_region_kernel_driver(
    fn: Callable[[PathCRegionLeaf, dict[str, np.ndarray]], None] | None,
) -> None:
    global _REGION_DRIVER
    _REGION_DRIVER = fn


def set_region_device_driver(
    fn: Callable[[PathCRegionLeaf, dict[str, Any], Any], None] | None,
) -> None:
    """Install the DEVICE-RESIDENT region-compute driver (banks stay device tensors)."""
    global _REGION_DEVICE_DRIVER
    _REGION_DEVICE_DRIVER = fn


def _drive_region_compute(leaf: PathCRegionLeaf, banks: dict[str, np.ndarray]) -> None:
    if _REGION_DRIVER is not None:
        _REGION_DRIVER(leaf, banks)
        return
    raise RuntimeError(
        "FAIL-LOUD: no region kernel driver set. Call set_region_kernel_driver(...) "
        "with either the on-device tilelang-kernel driver or a reference driver "
        "before invoking the DPS packed function."
    )


def _drive_region_compute_device(
    leaf: PathCRegionLeaf, banks: dict[str, Any], device: Any) -> None:
    if _REGION_DEVICE_DRIVER is not None:
        _REGION_DEVICE_DRIVER(leaf, banks, device)
        return
    raise RuntimeError(
        "FAIL-LOUD: no DEVICE-RESIDENT region driver set. Call "
        "set_region_device_driver(make_device_resident_kernel_driver(leaf, device)) "
        "before invoking the DPS packed function on a device VM.")


def register_region_dps_packed(leaf: PathCRegionLeaf, packed_name: str) -> None:
    """Register the region's DPS packed function under ``packed_name`` so Relax can
    ``R.call_dps_packed(packed_name, ...)`` it."""

    tvm_ffi.register_global_func(packed_name, make_region_dps_packed(leaf), override=True)


# --------------------------------------------------------------------------- #
# PR-3 (3): the REAL on-device tilelang-kernel driver.
#
# This is the production region-compute driver: it invokes ``leaf.kernel`` (the
# tilelang.compile'd JITKernel -- the real path_c compute, which can ONLY be lowered
# by tilelang, PR-2 mismatch #3) on a live device, driven through the physical banks.
# Proven end-to-end on Metal (scratch/pr3_real_kernel_driver.py): the real 17-param
# MR kernel runs THROUGH the call_dps_packed boundary, computing 14.68M nonzero
# activation outputs.
# --------------------------------------------------------------------------- #
def make_real_kernel_driver(
    leaf: PathCRegionLeaf, device: Any,
) -> Callable[[PathCRegionLeaf, dict[str, np.ndarray]], None]:
    """NUMPY-STAGED on-device driver (the REFERENCE path, kept for the CPU/numpy
    self-test and the host-vs-device equivalence check). Runs ``leaf.kernel`` (the real
    tilelang JITKernel) on ``device`` by staging the NUMPY bank dict H2D each call and
    reading the out banks D2H. This is the host-bounced path the device-resident driver
    below REPLACES.

    RULE #1: shape / param-count mismatches RAISE; no silent fallback."""

    if not getattr(device, "exist", False):
        raise RuntimeError(
            f"FAIL-LOUD: device {device} not present for the real tilelang-kernel driver")
    kparams = list(leaf.kernel.params)
    out_idx = set(int(x) for x in leaf.kernel.out_idx)
    # the leading 5 kernel params are the 5 physical banks, in bank_param_order[:5].
    bank_pos = {name: i for i, name in enumerate(leaf.bank_param_order[:5])}
    # locate the scalar gate param (zero-dim) -- typically param index 5.
    gate_pos = next((i for i, p in enumerate(kparams)
                     if len(list(p.shape)) == 0), None)
    if gate_pos is None:
        raise RuntimeError("FAIL-LOUD: no scalar gate param found in the kernel ABI")

    def driver(lf: PathCRegionLeaf, banks: dict[str, np.ndarray]) -> None:
        args: list[Any] = [None] * len(kparams)
        for name, pos in bank_pos.items():
            if name not in banks:
                raise RuntimeError(f"FAIL-LOUD: bank {name} missing for real driver")
            args[pos] = tvm.runtime.tensor(
                np.ascontiguousarray(banks[name], np.float32), device=device)
        args[gate_pos] = int(lf.run_backward)
        for i in range(len(kparams)):
            if args[i] is not None or i == gate_pos:
                continue
            shp = [int(d) for d in kparams[i].shape]
            args[i] = tvm.runtime.tensor(np.zeros(shp, np.float32), device=device)
        leaf.kernel(*args)
        device.sync()
        for name, pos in bank_pos.items():
            if pos in out_idx:
                banks[name] = args[pos].numpy().reshape(-1)

    return driver


# --------------------------------------------------------------------------- #
# PR-7 (the rework): the DEVICE-RESIDENT real-kernel driver.
#
# Banks arrive as DEVICE tensors (the driver owns them, allocated once via
# alloc_device_banks). The kernel's positional arg list is built ONCE: the 5 physical
# banks bound at their positions (the SAME device tensor objects the adapter packs
# into), the curried run_backward scalar at the gate, and zero-filled DEVICE scratch
# for every auxiliary route-buffer param (allocated ONCE, reused). Per call we just run
# ``kernel(*args)`` -- the kernel MUTATES the device banks IN PLACE at ``out_idx``, so
# there is NOTHING to read back: the device bank tensors ARE the output. NO numpy, NO
# H2D, NO D2H in the hot path. This is the host-bounce elimination.
# --------------------------------------------------------------------------- #
def make_device_resident_kernel_driver(
    leaf: PathCRegionLeaf, device: Any,
) -> Callable[[PathCRegionLeaf, dict[str, Any], Any], None]:
    """Build the DEVICE-RESIDENT driver. Signature matches the device-driver hook:
    ``(leaf, device_banks, device) -> None``, where ``device_banks`` is the dict of the
    driver-owned device-resident bank tensors. RULE #1: mismatches RAISE."""

    if not getattr(device, "exist", False):
        raise RuntimeError(
            f"FAIL-LOUD: device {device} not present for the device-resident driver")
    kparams = list(leaf.kernel.params)
    bank_pos = {name: i for i, name in enumerate(leaf.bank_param_order[:5])}
    gate_pos = next((i for i, p in enumerate(kparams)
                     if len(list(p.shape)) == 0), None)
    if gate_pos is None:
        raise RuntimeError("FAIL-LOUD: no scalar gate param found in the kernel ABI")

    # Allocate the auxiliary route-buffer scratch ONCE (device-resident, zeroed). These
    # are not part of THIS region's logical ABI; the kernel reads them as zero scratch.
    scratch: dict[int, Any] = {}
    for i in range(len(kparams)):
        if i == gate_pos or i in set(bank_pos.values()):
            continue
        shp = tuple(int(d) for d in kparams[i].shape)
        t = tvm.runtime.empty(shp if shp else (1,),
                              str(kparams[i].dtype), device=device)
        _zero_device_tensor(t)
        scratch[i] = t

    def driver(lf: PathCRegionLeaf, device_banks: dict[str, Any], dev: Any) -> None:
        args: list[Any] = [None] * len(kparams)
        for name, pos in bank_pos.items():
            if name not in device_banks:
                raise RuntimeError(
                    f"FAIL-LOUD: device bank {name} missing for device-resident driver")
            args[pos] = device_banks[name]  # the SAME device tensor (in-place output)
        args[gate_pos] = int(lf.run_backward)
        for i, t in scratch.items():
            args[i] = t
        lf.kernel(*args)
        dev.sync()
        # kernel wrote the out banks IN PLACE at out_idx -> device_banks already updated.

    return driver


# --------------------------------------------------------------------------- #
# Relax emission: the region as a call_dps_packed leaf
# --------------------------------------------------------------------------- #
def emit_region_call(
    bb: relax.BlockBuilder,
    packed_name: str,
    inputs: list[relax.Expr],
    out_sinfo: relax.TensorStructInfo,
) -> relax.Var:
    """Emit ``R.call_dps_packed(packed_name, inputs, out_sinfo)`` and return the
    logical-output Var. This is the real path_c region as a Relax-graph leaf."""

    return bb.emit(relax.call_dps_packed(packed_name, inputs, out_sinfo))


if __name__ == "__main__":
    # Self-test lives in the measurement module (path_c_relax_step) which assembles
    # several real region leaves and runs StaticPlanBlockMemory; running this module
    # directly just sanity-checks introspection on the real MR prim.
    from cppmega_mlx.runtime.path_c_fusion import (
        build_path_c_aot_autograd_region,
        build_path_c_model_region_from_route_symbols,
    )
    from cppmega_mlx.runtime.path_c_fusion_schedules import (
        path_c_fusion_schedule_template,
    )
    from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile

    cfg = local_gb10_quarter_profile().hybrid_config()
    region = build_path_c_model_region_from_route_symbols(
        region_name="mr_path_c", route_symbols=("M", "R"), model_config=cfg,
    )
    prim = path_c_fusion_schedule_template(build_path_c_aot_autograd_region(region))
    lmap = parse_logical_to_physical(prim)
    banks = parse_physical_bank_shapes(prim)
    order = prim_bank_param_order(prim)
    print(f"real MR prim: {len(prim.params)} params, "
          f"{len(lmap)} logical tensors, {len(banks)} physical banks")
    print("physical banks:", banks)
    print("bank param order:", order)
    print("sample logical maps:")
    for k in list(lmap)[:6]:
        m = lmap[k]
        print(f"  {k:40s} -> {m.bank} [{m.offset}:{m.offset+m.size}] shape={m.logical_shape}")
    sys.exit(0)
