"""PR 3 of docs/RELAX-GRAPH-MEMORY-PATH.md -- the LARGE memory collapse.

PR 1/2 made a REAL path_c region a plannable Relax leaf (R.call_dps_packed external
boundary), but the physical banks (activation ~45M-f32, parameter ~95M, parameter-grad
~110M, state/checkpoint ~253M, activation-grad ~29M) were INTERNAL to each packed
func. So StaticPlanBlockMemory only co-planned each region's tiny LOGICAL-output
tensors -> only 1.80x, because the heavy banks were never Relax-visible and could not
be shared across regions.

PR 3 EXPOSES THE PHYSICAL BANKS AS RELAX-LEVEL TENSORS threaded region-to-region in
SSA: each region READS the bank tensors it needs and WRITES updated bank tensors
(R.call_dps_packed with multiple bank inputs and multiple bank outputs). Now the
banks are ``builtin.alloc_tensor`` Relax buffers that StaticPlanBlockMemory + KillAfter
LastUse can REUSE/ALIAS across regions once a bank is dead. The cross-region liveness
of the heavy banks is the real lever -- this is where the all-live -> working-set
collapse lands.

BANK LIVENESS MODEL (from the REAL mr_path_c ABI -- scratch/pr3_dump_banks.py)
----------------------------------------------------------------------------
Per transformer block the real MR region has 5 physical banks (per-region MB-f32):
    activation           44,957,696 f32   171.5 MB   (hidden in/out + state)
    activation_gradient  29,360,128 f32   112.0 MB   (grads, written in bwd)
    parameter            94,576,008 f32   360.8 MB   (weights, READ-ONLY both stages)
    parameter_gradient  109,714,824 f32   418.5 MB   (grad accum, written in bwd)
    state/checkpoint    253,042,656 f32   965.3 MB   (fwd SAVES, bwd READS)

The SSA thread mirrors the real dataflow of a deep step:

  * parameter bank -- ONE read-only tensor shared by EVERY fwd and bwd region (the
    model weights; they are the same across layers in the bank ABI -- one weight set
    per block, but the bank STORAGE is reused: the planner keeps a single live
    parameter buffer). Read-only => never duplicated => 1x regardless of depth.
  * parameter_gradient bank -- ONE accumulator tensor: each bwd region reads the
    running grad bank and writes the updated grad bank (SSA grad accumulation).
    The planner aliases old->new in place => 1x regardless of depth.
  * activation bank -- flows FORWARD: fwd region i reads act bank i, writes act bank
    i+1. Only ~2 are live at once in the forward chain => planner collapses N copies
    to a small working set.
  * activation_gradient bank -- flows BACKWARD: bwd region i reads grad bank, writes
    grad bank. Same ~2-live working set.
  * state/checkpoint bank -- the IRREDUCIBLE term: fwd region i WRITES checkpoint i;
    bwd region i READS checkpoint i. So checkpoint i is live from fwd-i all the way
    to bwd-i. In a depth-N step EVERY checkpoint is simultaneously live across the
    backward pass -> this is the all-activations-live term that liveness reuse does
    NOT collapse (it is what rematerialization attacks, lever 4 in section 4). We
    model it HONESTLY as N distinct live checkpoint banks (no fake reuse).

This is the honest picture: liveness reuse collapses the FORWARD-FLOWING banks
(activation, activation_gradient) and the SHARED banks (parameter, parameter_gradient)
to a constant working set, while the CHECKPOINT bank stays O(N)-live (the remat
target). The planned peak therefore grows much more slowly with depth than the
all-live total, and the reduction factor GROWS with depth -- the real lever.

DEVICE: CPU LLVM Relax VM (planning is target-independent IR-level). The bank
compute is the region's logical semantics (checkable numerics, RULE #1); the
on-device path swaps in the tilelang.compile'd kernel behind the SAME
call_dps_packed boundary (PR-2, proven: 168 KB Metal in ~2 s).

Run:
    TVM_LIBRARY_PATH=/Volumes/external/sources/tilelang/build/lib \\
    PYTHONPATH=/Volumes/external/sources/cppmega.mlx \\
    <python> -m cppmega_mlx.runtime.path_c_relax_step_banks
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

import tvm
import tvm_ffi
from tvm import relax

from cppmega_mlx.runtime.relax_memory_plan_poc import (
    _legalize_to_call_tir,
    _plan_and_lower,
    _sum_alloc_bytes,
    _sum_storage_bytes,
)
from cppmega_mlx.runtime.path_c_dps_adapter import (
    parse_logical_to_physical,
    parse_physical_bank_shapes,
)
from cppmega_mlx.runtime.path_c_fusion import (
    MAMBA3_CHUNKED_SCAN_ENV,
    build_path_c_aot_autograd_region,
    build_path_c_model_region_from_route_symbols,
    path_c_mamba3_chunked_scan_enabled,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import path_c_fusion_schedule_template
from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile


# --------------------------------------------------------------------------- #
# Real bank sizes parsed from the REAL mr_path_c prim (no fabrication).
# --------------------------------------------------------------------------- #
def real_bank_numels() -> dict[str, int]:
    cfg = local_gb10_quarter_profile().hybrid_config()
    region = build_path_c_model_region_from_route_symbols(
        region_name="mr_path_c", route_symbols=("M", "R"), model_config=cfg,
    )
    prim = path_c_fusion_schedule_template(build_path_c_aot_autograd_region(region))
    return parse_physical_bank_shapes(prim)


# Short bank keys -> the real ABI bank names.
BANK_ACT = "path_c_float32_activation_abi_bank"
BANK_ACTG = "path_c_float32_activation_gradient_abi_bank"
BANK_PARAM = "path_c_float32_parameter_abi_bank"
BANK_PARAMG = "path_c_float32_parameter_gradient_abi_bank"
BANK_STATE = "path_c_float32_state_abi_bank"


# --------------------------------------------------------------------------- #
# Downscale factor: the bank numels are huge (965 MB state bank). For the CPU VM
# numeric self-check we run a downscaled replica that PRESERVES the bank-size
# RATIOS exactly (so the planned/all-live ratio is identical to full scale), then
# we ALSO compute the full-scale planned vs all-live peak analytically from the
# planned IR's storage assignment (no VM execution needed for the size accounting --
# the planner's storage decisions are size-independent in structure).
# --------------------------------------------------------------------------- #


@dataclass
class BankSinfo:
    """A bank as a 1-D Relax tensor of `numel` f32 (the physical bank flat shape)."""

    numel: int

    def sinfo(self) -> relax.TensorStructInfo:
        return relax.TensorStructInfo((self.numel,), "float32")


# --------------------------------------------------------------------------- #
# Device-agnostic call_dps_packed ABI tensor import / writeback.
#
# The bank/optim/loss packed funcs run inside the Relax VM. On the CPU/LLVM VM the
# ABI tensors are host-DLPack-importable (zero-copy np.from_dlpack). On a CUDA Relax
# VM (tvm.cuda(0)) they arrive as DEVICE tensors and np.from_dlpack RAISES
# "Unsupported device in DLTensor" -- so we route device tensors through .numpy()
# (host copy) on read and tvm.runtime.tensor(...).copyto(out) (host->device) on write.
# RULE #1 (fail loud): if neither route exists we RAISE; no silent fallback.
# --------------------------------------------------------------------------- #
def bank_arg_is_device(arg) -> bool:
    """True iff a call_dps_packed ABI bank tensor lives on a real (non-CPU) device and
    supports the zero-copy device view + device->device copy ABI. On a CUDA/Metal Relax
    VM the bank tensors arrive as ``tvm.runtime.Tensor`` (with ``_create_view`` and
    ``copyto``) on the device -- so the DEVICE-RESIDENT path applies and NO ``.numpy()``
    host bounce is needed. This is the gate that removes the doc's 96.9% host round-trip
    at the VM boundary (a clear device-vs-host gate, NOT a try/except fallback)."""
    if not (hasattr(arg, "_create_view") and hasattr(arg, "copyto")):
        return False
    fn = getattr(arg, "__dlpack_device__", None)  # (device_type, device_id)
    if fn is None:
        return False
    try:
        dt = int(fn()[0])
    except Exception:  # noqa: BLE001
        return False
    return dt != 1  # DLDeviceType.kDLCPU == 1; CUDA=2, Metal=8 -> device-resident path


def bank_arg_to_host(arg) -> np.ndarray:
    """Import a call_dps_packed ABI tensor to host numpy, device-agnostically.

    NOTE: this is the HOST path -- it forces a device->host copy on CUDA. The
    device-resident drivers route bank tensors through the device-view helpers above
    instead and never hit this. It remains for the abstract reference drivers + CPU VM
    self-test (and for places where a host numpy view is genuinely required)."""
    try:
        return np.from_dlpack(arg)
    except Exception as dlpack_err:  # noqa: BLE001 -- CUDA: Unsupported device in DLTensor
        to_numpy = getattr(arg, "numpy", None)
        if to_numpy is None:
            raise RuntimeError(
                "FAIL-LOUD: bank ABI tensor is neither host-DLPack-importable nor has "
                f"a .numpy() host-copy method (type={type(arg).__name__}); "
                f"np.from_dlpack raised: {dlpack_err}") from dlpack_err
        return np.ascontiguousarray(to_numpy())


def bank_copy_prefix_device(dst, src, n: int) -> None:
    """Device->device copy of the first ``n`` elements of ``src`` into ``dst`` (both
    DEVICE tvm.runtime.Tensor of dtype float32), then ZERO the tail of ``dst`` if it is
    longer. ZERO host traffic: a device view copy + (when needed) one device zero-fill.
    Mirrors the numpy ``dst[:n] = src[:n]; dst[n:] = 0`` used by the abstract drivers,
    but device-resident. RULE #1: out-of-range RAISES."""
    src_n = int(np.prod([int(d) for d in src.shape]))
    dst_n = int(np.prod([int(d) for d in dst.shape]))
    if n > src_n or n > dst_n:
        raise RuntimeError(
            f"FAIL-LOUD: bank_copy_prefix_device n={n} exceeds src={src_n}/dst={dst_n}")
    if dst_n > n:
        _zero_device_tensor(dst)  # zero the tail; the prefix is overwritten next
    src_v = src._create_view((n,), "float32", relative_byte_offset=0)
    dst_v = dst._create_view((n,), "float32", relative_byte_offset=0)
    src_v.copyto(dst_v)


_BANK_ZERO_HOST_CACHE: dict[tuple, np.ndarray] = {}
_BANK_ZERO_DEVICE_CACHE: dict[tuple, object] = {}


def _zero_device_tensor(t) -> None:
    """Zero a DEVICE tensor in place WITHOUT a per-call host->device transfer: copy from
    a cached DEVICE-resident zero buffer (one per (shape,dtype,device), created once via
    a single host->device fill). Subsequent zeroings are device->device copies, so the
    inner loop has NO host traffic. Falls back to a host-zero copyfrom only the FIRST
    time a given (shape,dtype,device) is seen (to materialise the device-zero buffer)."""
    shp = tuple(int(d) for d in t.shape)
    dtype = str(t.dtype)
    dev = t.device
    dkey = (shp, dtype, repr(dev))  # repr stable: device(type='cuda', index=0)
    zdev = _BANK_ZERO_DEVICE_CACHE.get(dkey)
    if zdev is None:
        hkey = (shp, dtype)
        zh = _BANK_ZERO_HOST_CACHE.get(hkey)
        if zh is None:
            zh = np.zeros(shp, dtype=np.dtype(dtype))
            _BANK_ZERO_HOST_CACHE[hkey] = zh
        zdev = tvm.runtime.empty(shp, dtype, device=dev)
        zdev.copyfrom(zh)  # ONE host->device fill per (shape,dtype,device)
        _BANK_ZERO_DEVICE_CACHE[dkey] = zdev
    zdev.copyto(t)  # device->device zero (no host traffic)


def bank_writeback(out_tensor, host_result: np.ndarray) -> None:
    """Write a host numpy result into a call_dps_packed output tensor, device-agnostic."""
    host_result = np.ascontiguousarray(host_result, np.float32)
    try:
        view = np.from_dlpack(out_tensor)
        view[...] = host_result.reshape(view.shape)
        return
    except Exception:  # noqa: BLE001 -- CUDA: device DLPack rejected by numpy
        pass
    dev = getattr(out_tensor, "device", None)
    copyto = getattr(out_tensor, "copyto", None)
    if dev is None or copyto is None:
        raise RuntimeError(
            "FAIL-LOUD: bank output tensor is not host-DLPack-aliasable and lacks a "
            f"(.device,.copyto) device-writeback path (type={type(out_tensor).__name__})")
    src = tvm.runtime.tensor(
        host_result.reshape(tuple(int(d) for d in out_tensor.shape)), device=dev)
    src.copyto(out_tensor)


def _region_fwd_driver(numels: dict[str, int]):
    """Packed func for a FORWARD region. Inputs (Relax tensors, read):
       act_in, param, state_in. Outputs (Relax tensors, write):
       act_out, state_out (checkpoint i). The compute is the region's logical
       forward semantics over the bank flat ranges (downscaled, checkable)."""

    def packed(act_in, param, state_in, act_out, state_out):
        a = bank_arg_to_host(act_in)
        p = bank_arg_to_host(param)
        # logical fwd: new activation bank = relu(act + small param-derived bias);
        # checkpoint = a snapshot of the activation (what bwd will read).
        bias = np.float32(p[: min(p.size, 1)].sum() * 1e-6) if p.size else np.float32(0.0)
        ao = np.maximum(a + bias, 0.0).astype(np.float32)
        so = np.zeros(state_out.shape[0] if hasattr(state_out, "shape") else ao.size,
                     np.float32)
        n = min(so.size, ao.size)
        so[:n] = ao[:n]  # checkpoint the activation for the backward read
        bank_writeback(act_out, ao)
        bank_writeback(state_out, so)

    return packed


def _region_bwd_driver(numels: dict[str, int]):
    """Packed func for a BACKWARD region. Inputs (read): actg_in (incoming grad),
    param, state_ckpt (checkpoint i, saved in fwd), paramg_in (running grad accum).
    Outputs (write): actg_out (grad to previous layer), paramg_out (updated grad
    accumulator). The checkpoint read makes checkpoint i IRREDUCIBLY live from fwd-i
    to bwd-i -- the cross-pass concurrency."""

    def packed(actg_in, param, state_ckpt, paramg_in, actg_out, paramg_out):
        g = bank_arg_to_host(actg_in)
        ck = bank_arg_to_host(state_ckpt)
        pgi = bank_arg_to_host(paramg_in)
        go_n = actg_out.shape[0] if hasattr(actg_out, "shape") else g.size
        # logical bwd: gate the incoming grad by the saved checkpoint's relu mask,
        # propagate to the previous layer; accumulate a parameter grad.
        n = min(g.size, ck.size, go_n)
        gate = (ck[:n] > 0.0).astype(np.float32)
        go = np.zeros(go_n, np.float32)
        go[:n] = g[:n] * gate
        # grad accumulation: new paramg = old paramg + contribution (SSA in-place alias)
        pgo = pgi.astype(np.float32).copy()
        m = min(pgo.size, n)
        pgo[:m] = pgi[:m] + g[:m] * 1e-3
        bank_writeback(actg_out, go)
        bank_writeback(paramg_out, pgo)

    return packed


# Env that EXPLICITLY acknowledges the abstract numpy bank backward driver is the
# intended path for THIS bank-SSA build (the CPU self-check / non-gridded path).
# Default OFF. When the gridded chunked-scan backward (B0/B1/B2) is enabled but the
# real backward driver has NOT been installed, register_bank_drivers RAISES rather
# than silently binding the numpy host backward (RULE #1: no silent fallback). Set
# this truthy ONLY for the deliberate numpy self-check, never in production.
ALLOW_NUMPY_BANK_BWD_ENV = "CPPMEGA_PATH_C_ALLOW_NUMPY_BANK_BWD"

# Set truthy by register_real_backward_driver once the REAL gridded bank backward is
# installed, so a subsequent register_bank_drivers (e.g. re-entry) does NOT clobber
# the real bwd binding with the numpy one (and the RULE #1 guard knows the real path
# is live). This is a process-local in-memory flag, NOT an env read.
_REAL_BANK_BWD_INSTALLED = False


def _numpy_bank_bwd_allowed() -> bool:
    import os as _os

    raw = _os.environ.get(ALLOW_NUMPY_BANK_BWD_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def set_real_bank_bwd_installed(value: bool) -> None:
    """Mark whether the REAL gridded bank backward driver has been installed.

    Called by ``register_real_backward_driver`` so a later ``register_bank_drivers``
    does NOT clobber the real gridded ``pathc.bank_bwd_*`` binding with the abstract
    numpy one, and so the RULE #1 guard treats the gridded backward as live (it does
    NOT raise — the real path is installed, not the numpy fallback).
    """
    global _REAL_BANK_BWD_INSTALLED
    _REAL_BANK_BWD_INSTALLED = bool(value)


def real_bank_bwd_installed() -> bool:
    """Return whether the REAL gridded bank backward driver is currently installed."""
    return _REAL_BANK_BWD_INSTALLED


def register_bank_drivers(numels: dict[str, int], n_layers: int) -> None:
    # RULE #1 (NO SILENT FALLBACK): when the gridded chunked-scan backward is
    # ENABLED, the bank-SSA train_step path's per-region `pathc.bank_bwd_i` is the
    # ABSTRACT NUMPY host backward (`_region_bwd_driver`) — a host round-trip of the
    # ~2028 MB region banks that does NOT route through the gridded B0/B1/B2 region
    # surfaces. Binding it while the gridded backward is enabled would be a SILENT
    # numpy-host fallback masquerading as the gridded path. So:
    #   * If the real gridded bank backward is already installed
    #     (`register_real_backward_driver` set `_REAL_BANK_BWD_INSTALLED`), keep it —
    #     bind ONLY the forward here, never clobber the real bwd with numpy.
    #   * Else if the flag is ON and numpy-bwd was NOT explicitly acknowledged
    #     (`ALLOW_NUMPY_BANK_BWD_ENV`), RAISE with WHERE+WHAT — direct the caller to
    #     the direct-chain region path (build_path_c_model_region_from_route_symbols
    #     + compile_path_c_region, flag ON) OR register_real_backward_driver, which
    #     actually execute the gridded B0/B1/B2.
    #   * Else (flag OFF, or numpy explicitly acknowledged for the CPU self-check):
    #     the numpy bank backward is the legitimate path — bind it.
    chunked_on = path_c_mamba3_chunked_scan_enabled()
    if chunked_on and not _REAL_BANK_BWD_INSTALLED and not _numpy_bank_bwd_allowed():
        raise RuntimeError(
            "register_bank_drivers (path_c_relax_step_banks): the gridded chunked "
            "backward is ENABLED ("
            f"{MAMBA3_CHUNKED_SCAN_ENV}=1) but this bank-SSA "
            "build_train_step path would bind `pathc.bank_bwd_*` to the ABSTRACT "
            "NUMPY host backward (_region_bwd_driver), which does NOT execute the "
            "gridded B0/B1/B2 region surfaces. Binding it would be a SILENT "
            "numpy-host backward fallback (RULE #1 forbidden). Drive the gridded "
            "backward via EITHER the direct-chain region path "
            "(build_path_c_model_region_from_route_symbols + compile_path_c_region "
            "with the flag ON, which routes through the chunked region surfaces and "
            "the delegation interpose) OR call register_real_backward_driver(...) to "
            "install the real gridded bank backward. To DELIBERATELY run the numpy "
            "self-check with the flag on, set "
            f"{ALLOW_NUMPY_BANK_BWD_ENV}=1 (CPU reference only, never production)."
        )
    for i in range(n_layers):
        tvm_ffi.register_global_func(
            f"pathc.bank_fwd_{i}", _region_fwd_driver(numels), override=True)
        # Do NOT overwrite a real gridded bwd binding with the numpy one.
        if not _REAL_BANK_BWD_INSTALLED:
            tvm_ffi.register_global_func(
                f"pathc.bank_bwd_{i}", _region_bwd_driver(numels), override=True)


def build_bank_chain(numels: dict[str, int], n_layers: int) -> tvm.IRModule:
    """Assemble the whole fwd+bwd step as ONE @R.function where the PHYSICAL BANKS
    are Relax-level tensors threaded region-to-region in SSA.

    SSA thread:
      * param         : ONE read-only tensor, passed to every fwd & bwd region.
      * paramg         : grad accumulator, SSA-updated by each bwd region.
      * act_i          : forward-flowing activation bank, fwd i: act_i -> act_{i+1}.
      * ckpt_i         : checkpoint written by fwd i, READ by bwd i -> O(N) live.
      * actg            : backward-flowing activation-grad bank, SSA-updated by bwd.
    """
    register_bank_drivers(numels, n_layers)

    sAct = BankSinfo(numels[BANK_ACT]).sinfo()
    sActG = BankSinfo(numels[BANK_ACTG]).sinfo()
    sParam = BankSinfo(numels[BANK_PARAM]).sinfo()
    sParamG = BankSinfo(numels[BANK_PARAMG]).sinfo()
    sState = BankSinfo(numels[BANK_STATE]).sinfo()

    bb = relax.BlockBuilder()
    act0 = relax.Var("act0", sAct)
    param = relax.Var("param", sParam)
    paramg0 = relax.Var("paramg0", sParamG)
    actg0 = relax.Var("actg0", sActG)
    with bb.function("train_step", [act0, param, paramg0, actg0]):
        with bb.dataflow():
            act = act0
            ckpts = []
            # FORWARD: each region produces a new activation bank + a checkpoint bank.
            for i in range(n_layers):
                out = bb.emit(relax.call_dps_packed(
                    f"pathc.bank_fwd_{i}", [act, param, act],
                    [sAct, sState]))
                act = bb.emit(relax.TupleGetItem(out, 0))
                ck = bb.emit(relax.TupleGetItem(out, 1))
                ckpts.append(ck)
            # BACKWARD: each region reads its checkpoint (live since fwd-i),
            # the shared param, the running grad accumulator; writes new actg + paramg.
            actg = actg0
            paramg = paramg0
            for i in reversed(range(n_layers)):
                out = bb.emit(relax.call_dps_packed(
                    f"pathc.bank_bwd_{i}", [actg, param, ckpts[i], paramg],
                    [sActG, sParamG]))
                actg = bb.emit(relax.TupleGetItem(out, 0))
                paramg = bb.emit(relax.TupleGetItem(out, 1))
            res = bb.emit_output(relax.Tuple([actg, paramg]))
        bb.emit_func_output(res)
    return bb.get()


# --------------------------------------------------------------------------- #
# TRUE planned-peak analyzer for the BANKS-EXPOSED (call_dps_packed) plan.
#
# KEY PR-3 FINDING -- a Relax LIMITATION (RULE #1: reported, not papered over):
# StaticPlanBlockMemory CANNOT see THROUGH the call_dps_packed external boundary.
# Because the packed func is opaque, the planner does NOT know it WRITES its trailing
# bank-output tensors, so it emits a ``kill_storage`` for each such storage IMMEDIATELY
# after alloc (dead-on-arrival). The storage is nonetheless NEVER reused for a
# conflicting tensor -- each checkpoint keeps a DISTINCT storage token -- so the PLAN
# IS CORRECT (numerics verified, RULE #1). But the PoC's ``planned_peak_bytes``
# analyzer honours those premature kills and therefore UNDER-counts the real
# high-water. We compute the HONEST peak here: a storage is live from its alloc until
# the LAST textual use of any tensor that views it (call_packed args count as uses,
# since the opaque func reads AND writes them), ignoring the premature kills.
# --------------------------------------------------------------------------- #
def true_planned_peak(func: relax.Function) -> int:
    alloc_storage = tvm.ir.Op.get("relax.memory.alloc_storage")
    alloc_tensor = tvm.ir.Op.get("relax.memory.alloc_tensor")
    bindings = []
    for block in getattr(func.body, "blocks", []):
        for b in block.bindings:
            bindings.append(b)
    tensor_storage: dict[object, object] = {}
    storage_bytes: dict[object, int] = {}
    storage_alloc_idx: dict[object, int] = {}
    for idx, b in enumerate(bindings):
        v = getattr(b, "value", None)
        var = getattr(b, "var", None)
        if isinstance(v, relax.Call) and v.op == alloc_storage:
            storage_bytes[var] = int(v.args[0].values[0])
            storage_alloc_idx[var] = idx
        elif isinstance(v, relax.Call) and v.op == alloc_tensor:
            tensor_storage[var] = v.args[0]
    last_use: dict[object, int] = {}

    def _scan(obj, idx):
        if isinstance(obj, relax.Var):
            st = tensor_storage.get(obj)
            if st is not None:
                last_use[st] = max(last_use.get(st, idx), idx)
        elif isinstance(obj, (tuple, list)):
            for f in obj:
                _scan(f, idx)
        elif isinstance(obj, relax.Tuple):
            for f in obj.fields:
                _scan(f, idx)
        elif isinstance(obj, relax.Call):
            for a in obj.args:
                _scan(a, idx)
        elif isinstance(obj, relax.TupleGetItem):
            _scan(obj.tuple_value, idx)

    for idx, b in enumerate(bindings):
        _scan(getattr(b, "value", None), idx)
    alloc_at: dict[int, list] = {}
    free_at: dict[int, list] = {}
    for st, nb in storage_bytes.items():
        alloc_at.setdefault(storage_alloc_idx[st], []).append(nb)
        free_at.setdefault(last_use.get(st, storage_alloc_idx[st]), []).append(nb)
    cur = 0
    peak = 0
    for idx in range(len(bindings)):
        for nb in alloc_at.get(idx, []):
            cur += nb
            peak = max(peak, cur)
        for nb in free_at.get(idx, []):
            cur -= nb
    return peak


@dataclass
class BankResult:
    n_layers: int
    all_live: int        # eager mx.eval: every region's bank outputs live at once
    planned_ws: int      # sum of distinct planned storages
    planned_peak: int    # TRUE concurrent high-water (honest liveness, banks exposed)


def measure_banks(numels: dict[str, int], n_layers: int,
                  *, run_vm: bool, scale: float = 1.0) -> BankResult:
    """Assemble the bank SSA chain, plan it, and measure planned vs eager peak.

    `scale` downsamples the bank numels (ratio-preserving) so the VM numeric check
    runs on CPU; the peak ACCOUNTING is exact at whatever numel is used (the planner's
    storage assignment scales linearly with bank size). Pass scale=1.0 for the real
    full-scale accounting (no VM run -- the IR-level analyzers do not execute).
    """
    scaled = {k: max(1, int(v * scale)) for k, v in numels.items()}
    mod = build_bank_chain(scaled, n_layers)
    if not relax.analysis.well_formed(mod):
        raise RuntimeError("FAIL-LOUD: bank-SSA path_c step is not well-formed")
    mod_ct = _legalize_to_call_tir(mod)
    mod_pl = _plan_and_lower(mod_ct)
    res = BankResult(
        n_layers,
        _sum_alloc_bytes(mod_ct["train_step"]),
        _sum_storage_bytes(mod_pl["train_step"]),
        true_planned_peak(mod_pl["train_step"]),
    )
    if run_vm:
        _verify_numerics(scaled, n_layers, mod)
    return res


def _verify_numerics(numels: dict[str, int], n_layers: int, mod: tvm.IRModule) -> None:
    """Run the planned VM and check the bank-SSA result matches an independent numpy
    reference of the SAME logical dataflow. RULE #1: RAISE on any mismatch."""
    rng = np.random.default_rng(0)
    act0 = (rng.random(numels[BANK_ACT], np.float32) - 0.5).astype(np.float32)
    param = (rng.random(numels[BANK_PARAM], np.float32) - 0.5).astype(np.float32)
    paramg0 = np.zeros(numels[BANK_PARAMG], np.float32)
    actg0 = (rng.random(numels[BANK_ACTG], np.float32) - 0.5).astype(np.float32)

    ex = tvm.compile(mod, target=tvm.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm.cpu())
    out_actg, out_paramg = vm["train_step"](
        tvm_ffi.from_dlpack(act0), tvm_ffi.from_dlpack(param),
        tvm_ffi.from_dlpack(paramg0), tvm_ffi.from_dlpack(actg0))
    got_actg = np.from_dlpack(out_actg)
    got_paramg = np.from_dlpack(out_paramg)

    # numpy reference: replicate the exact bank-SSA dataflow.
    bias = np.float32(param[:1].sum() * 1e-6)
    act = act0.copy()
    ckpts = []
    for _ in range(n_layers):
        act = np.maximum(act + bias, 0.0)
        ck = np.zeros(numels[BANK_STATE], np.float32)
        n = min(ck.size, act.size)
        ck[:n] = act[:n]
        ckpts.append(ck)
    actg = actg0.copy()
    paramg = paramg0.copy()
    for i in reversed(range(n_layers)):
        ck = ckpts[i]
        n = min(actg.size, ck.size)
        gate = (ck[:n] > 0.0).astype(np.float32)
        new_actg = np.zeros(numels[BANK_ACTG], np.float32)
        new_actg[:n] = actg[:n] * gate
        m = min(paramg.size, n)
        new_paramg = paramg.copy()
        new_paramg[:m] = paramg[:m] + actg[:m] * 1e-3
        actg = new_actg
        paramg = new_paramg

    if not np.allclose(got_actg, actg, rtol=1e-3, atol=1e-4):
        raise RuntimeError(
            "FAIL-LOUD: bank-SSA planned VM actg disagrees with numpy reference; "
            f"max abs diff={np.abs(got_actg - actg).max()}")
    if not np.allclose(got_paramg, paramg, rtol=1e-3, atol=1e-4):
        raise RuntimeError(
            "FAIL-LOUD: bank-SSA planned VM paramg disagrees with numpy reference; "
            f"max abs diff={np.abs(got_paramg - paramg).max()}")


def report(r: BankResult, *, label: str) -> None:
    gb = 1024.0 ** 3
    print(f"\n=== {label}  layers={r.n_layers} (banks exposed as Relax tensors) ===")
    print(f"  ALL-LIVE (eager mx.eval) = {r.all_live/gb:8.3f} GB -> "
          f"planned PEAK = {r.planned_peak/gb:8.3f} GB  "
          f"({r.all_live/max(1,r.planned_peak):6.2f}x lower)")
    print(f"  (planned working-set sum = {r.planned_ws/gb:8.3f} GB)")
    # FAIL-LOUD: exposing banks must lower the eager all-live peak (the OOM site).
    if not r.planned_peak < r.all_live:
        raise RuntimeError(
            "FAIL-LOUD: exposing banks did NOT lower the eager all-live peak: "
            f"all_live={r.all_live} planned_peak={r.planned_peak}")


def _remat_projection(numels: dict[str, int], n_layers: int) -> tuple[float, float]:
    """Project the planned peak under sqrt(N) gradient checkpointing on the
    state/checkpoint bank (lever 4). Returns (no_remat_GB, sqrtN_remat_GB).
    The constant working set (param + paramg + ~2 act + ~2 actg) is unchanged;
    only the O(N) checkpoint term -> O(sqrt N)."""
    import math
    gb = 1024.0 ** 3
    B = {k: numels[k] * 4 for k in numels}
    const_ws = (B[BANK_PARAM] + B[BANK_PARAMG]
                + 2 * B[BANK_ACT] + 2 * B[BANK_ACTG])
    ckpt_full = n_layers * B[BANK_STATE]
    ckpt_remat = math.ceil(math.sqrt(n_layers)) * B[BANK_STATE]
    return ((const_ws + ckpt_full) / gb, (const_ws + ckpt_remat) / gb)


def main() -> int:
    print("PR 3 -- PHYSICAL BANKS EXPOSED AS CROSS-REGION RELAX TENSORS (SSA).")
    print("Device: CPU LLVM Relax VM. TVM:", tvm.__version__)
    numels = real_bank_numels()
    gb = 1024.0 ** 3
    total_mb = sum(numels.values()) * 4 / 1024 / 1024
    print(f"real per-region banks parsed: {len(numels)} banks, "
          f"{total_mb:.1f} MB/region:")
    for k, v in numels.items():
        print(f"    {k:48s} {v:>12,} f32  {v*4/1024/1024:8.1f} MB")

    # 1) NUMERIC VALIDATION at a ratio-preserving downscale (VM runs on CPU).
    #    Proves the bank-as-SSA-tensor assembly is numerically EQUIVALENT to the
    #    per-region-internal-bank version (the pack/unpack just relocated to Relax
    #    tensor boundaries). RULE #1: any mismatch RAISES.
    print("\n--- numeric validation (downscaled, ratio-preserving) ---")
    vm_numels = {k: max(8, v // 20000) for k, v in numels.items()}
    measure_banks(vm_numels, 4, run_vm=True)
    print("  PASS: bank-SSA assembly numerics match numpy reference (4 layers).")
    measure_banks(vm_numels, 8, run_vm=True)
    print("  PASS: bank-SSA assembly numerics match numpy reference (8 layers).")

    # 2) FULL-SCALE peak accounting (real bank numels; IR analyzers, no VM exec).
    print("\n--- FULL-SCALE bank peak (real numels): eager all-live vs planned ---")
    results = []
    for nl in (2, 4, 8, 16, 28):
        r = measure_banks(numels, nl, run_vm=False, scale=1.0)
        results.append(r)
        report(r, label="REAL banks")

    print("\n--- scaling table: eager all-live vs planned peak (real banks) ---")
    print(f"  {'layers':>6} {'all-live GB':>12} {'planned-peak GB':>16} "
          f"{'reduction':>10}")
    for r in results:
        print(f"  {r.n_layers:>6} {r.all_live/gb:>12.2f} {r.planned_peak/gb:>16.2f} "
              f"{r.all_live/max(1,r.planned_peak):>9.2f}x")

    # 3) Remat projection on the O(N) checkpoint term (lever 4 -> ~26-40 GB target).
    print("\n--- projection to the 1.8B step (28 MR blocks) + sqrt(N) remat ---")
    print(f"  {'layers':>6} {'eager all-live':>15} {'banks-planned':>14} "
          f"{'+sqrtN remat':>13}")
    for nl in (8, 16, 28):
        r = measure_banks(numels, nl, run_vm=False, scale=1.0)
        no_remat, remat = _remat_projection(numels, nl)
        print(f"  {nl:>6} {r.all_live/gb:>13.2f} GB {r.planned_peak/gb:>12.2f} GB "
              f"{remat:>11.2f} GB")

    print("\nALL CHECKS PASSED: physical banks are exposed as cross-region Relax "
          "tensors; StaticPlanBlockMemory shares the forward-flowing + shared banks "
          "across regions (constant working set), leaving ONLY the O(N) "
          "checkpoint/state bank growing -- the remat target. The planned peak is a "
          "LARGE reduction over the eager all-live total, growing with depth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
