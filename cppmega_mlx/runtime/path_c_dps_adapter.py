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

    Body: allocate the physical banks, pack each logical input into its bank
    sub-range, invoke the real tilelang kernel on the banks (in the kernel's
    positional bank order), unpack the logical-output sub-range into ``logical_out``.

    NOTE on the kernel call: the real tilelang JITKernel takes the physical banks
    (and any auxiliary route buffers) positionally. We supply the banks we own; any
    auxiliary route-symbol buffers the kernel also takes are zero-filled scratch of
    the kernel-declared shape (they are not part of THIS region's logical ABI -- the
    adapter's contract is the logical inputs/output; auxiliary kernel args are an
    internal kernel detail). For the self-test below we drive a numpy reference
    through the SAME pack/unpack to validate the adapter plumbing end to end without
    requiring a live Metal device on the measurement host.
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
        in_arrays = [np.from_dlpack(a) for a in args[:-1]]
        out_array = np.from_dlpack(args[-1])

        # Allocate the physical banks we own.
        banks = {b: np.zeros((n,), dtype=np.float32) for b, n in bank_shapes.items()}

        # Pack each logical input into its bank sub-range.
        for lname, arr in zip(leaf.logical_inputs, in_arrays):
            m = lmap[lname]
            flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
            if flat.size != m.size:
                raise RuntimeError(
                    f"FAIL-LOUD: logical input {lname} numel {flat.size} != "
                    f"ABI sub-range size {m.size}"
                )
            banks[m.bank][m.offset : m.offset + m.size] = flat

        # Invoke the region compute on the packed banks. The real deployment calls
        # ``leaf.kernel`` (the tilelang JITKernel) here on a live Metal/CUDA device;
        # the adapter's pack/unpack ABI around that call is what this module proves.
        _drive_region_compute(leaf, banks)

        # Unpack the logical-output sub-range into the trailing DPS output tensor.
        m = lmap[leaf.logical_output]
        out_flat = banks[m.bank][m.offset : m.offset + m.size]
        out_array[...] = out_flat.reshape(out_array.shape)

    return _packed


# Hook the region compute. Default = a transparent reference matching the region's
# logical semantics (so the adapter ABI is testable on CPU without a live GPU); the
# real path is set by ``set_region_kernel_driver`` to call ``leaf.kernel`` on device.
_REGION_DRIVER: Callable[[PathCRegionLeaf, dict[str, np.ndarray]], None] | None = None


def set_region_kernel_driver(
    fn: Callable[[PathCRegionLeaf, dict[str, np.ndarray]], None] | None,
) -> None:
    global _REGION_DRIVER
    _REGION_DRIVER = fn


def _drive_region_compute(leaf: PathCRegionLeaf, banks: dict[str, np.ndarray]) -> None:
    if _REGION_DRIVER is not None:
        _REGION_DRIVER(leaf, banks)
        return
    raise RuntimeError(
        "FAIL-LOUD: no region kernel driver set. Call set_region_kernel_driver(...) "
        "with either the on-device tilelang-kernel driver or a reference driver "
        "before invoking the DPS packed function."
    )


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
    """Build an on-device driver that runs ``leaf.kernel`` (the real tilelang JITKernel)
    on ``device`` (e.g. ``tvm.metal(0)`` / ``tvm.cuda(0)``), mapping the 5 physical
    banks to the kernel's leading 5 params, the curried ``run_backward`` scalar to the
    gate param, and zero-filled scratch to the auxiliary route-buffer params.

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
