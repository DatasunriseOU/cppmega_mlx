"""MXFP8 TN-adapter (backward NN/NT -> TN rewrite) + NVFP4 RtN path_c GEMM option.

This module is **lever 4** of the path_c fp8 effort. It lifts two MEASURED
prior-art pieces from the original (Megatron-targeted) ``cppmega`` stack and the
``cppmega.mlx`` nvfp4 route into a single, package-resident path_c surface,
gated default-OFF behind env flags, with RULE #1 fail-loud semantics (when a
surface is ENABLED it is the ONE path and RAISES on failure -- never a silent
bf16 / host-copy downgrade).

Two distinct surfaces (kept separate on purpose; do NOT conflate them):

1. ``CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1`` -- the **MXFP8 TN adapter**. Ported from
   ``cppmega/scripts/cppmega_fp8_shim.py`` (the working backward TN-adapter core
   at lines 1392-1500 ``_cppmega_mxfp8_colwise_as_rowwise_transpose`` +
   2059-2122 ``_cppmega_try_mxfp8_tn_adapter``). cuBLASLt on cc12.x returns
   ``CUBLAS_STATUS_NOT_SUPPORTED`` for the backward NN (dgrad) and NT (wgrad)
   MXFP8 GEMM layouts; the adapter retargets them to the supported **TN** layout
   by transposing the columnwise MXFP8 payload + scale (or consuming a TE-emitted
   rowwise-transpose sidecar) and then calling the *original* TE ``general_gemm``.
   This rides stock TE's ``general_gemm`` -- no new CUDA kernel. MEASURED on gb10
   in cppmega: lm_loss 1.566, ``bf16_fallback=0``, ~4501 ms/iter (on par with
   tensorwise, numerically stable).

   In path_c the MXFP8 quantized operands and the TE ``general_gemm`` module live
   on the torch/TE side; this module exposes :func:`try_mxfp8_tn_adapter` which a
   torch-side caller (e.g. a TE ``general_gemm`` monkey-patch, mirroring the
   cppmega shim's ``_cppmega_wrap_general_gemm``) invokes to perform the
   NN/NT->TN rewrite. It does NOT itself monkey-patch TE here -- it is the
   reusable adapter core; lever 1's ``fp8_te_linear`` backward (or a future
   general_gemm wrapper) is the call site.

2. ``CPPMEGA_NVFP4_PATH_C_GEMM=1`` -- the **NVFP4 RtN GEMM** as a path_c GEMM
   option. Wires the reduced round-to-nearest NVFP4 recipe
   (``NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True)``,
   ported from nanochat, MEASURED on gb10 sm_121: fwd rel_err 0.147, dgrad 0.1465,
   wgrad 0.1343) as a real GEMM that takes MLX operands, bridges them zero-copy to
   torch via lever 1's ``_cuda_zerocopy`` DLPack bridge, runs the FP4 GEMM through
   a ``te.Linear`` under the RtN autocast, and bridges the result back. The recipe
   construction + numeric gate live in ``scripts/_nvfp4_route.py`` (already
   committed); this module imports them lazily and only adds the *executable*
   GEMM that the route file lacked.

DLPack dependency (lever 1): the tensor handoff uses the SAME
``cppmega_mlx.nn._tilelang._cuda_zerocopy`` bridge that lever 1 fixes
(``torch.from_dlpack`` of the MLX native kDLCUDA export). This module assumes the
bridge is fixed; it imports ``mlx_cuda_array_to_torch_tensor`` /
``torch_cuda_tensor_to_mlx`` exactly as ``fp8_te_linear`` does. A bridge failure
RAISES with where+what -- it is never papered over.

RULE #1: every ENABLED path is the ONE path. On any failure (bridge reject, TE
unavailable, recipe not the reduced RtN one, numeric gate fail, unsupported
layout, missing operand sidecar) this RAISES with WHERE + WHAT. There is no
``try/except -> bf16``, no ``except: pass``, no clamp-instead-of-raise. Both
surfaces are default-OFF, so the dense path is entirely unaffected until a flag
is explicitly set.

NO GPU on the dev Mac: torch / transformer_engine / tilelang are imported lazily
inside the functions that need them, so this module imports + ``py_compile``s
cleanly off-gb10 (the TE/torch/CUDA work only fires when a flag is on AND the
function is called on the live host).
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

# RULE #1: the NVRTC loader fix MUST be applied before torch / TE are imported
# anywhere in this process (gb10). Mirrors fp8_te_linear / fp8_matmul_path_c.
# Safe no-op off-gb10 (returns "noop-not-gb10").
from cppmega_mlx._gb10_nvrtc_env import ensure_nvrtc_builtins_path

ensure_nvrtc_builtins_path()


# ---------------------------------------------------------------------------
# Env gates (default-OFF). Each gate selects exactly one surface; neither is on
# by default, so the dense path is unaffected.
# ---------------------------------------------------------------------------
MXFP8_TN_ADAPTER_ENV = "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER"
NVFP4_PATH_C_GEMM_ENV = "CPPMEGA_NVFP4_PATH_C_GEMM"
_TRUE = {"1", "true", "yes", "on"}

# MXFP8 backward layouts cuBLASLt cannot do on cc12.x; the TN adapter rewrites
# these to "TN". (dgrad is the NN GEMM weight.T @ dy; wgrad is the NT GEMM
# x.T @ dy.T -- both expressed via columnwise->rowwise-transpose of operands.)
_MXFP8_BWD_LAYOUTS = ("NN", "NT")


def mxfp8_tn_adapter_enabled() -> bool:
    """True iff the MXFP8 backward TN-adapter gate is on (default OFF)."""

    return os.environ.get(MXFP8_TN_ADAPTER_ENV, "").strip().lower() in _TRUE


def nvfp4_path_c_gemm_enabled() -> bool:
    """True iff the NVFP4 RtN path_c GEMM gate is on (default OFF)."""

    return os.environ.get(NVFP4_PATH_C_GEMM_ENV, "").strip().lower() in _TRUE


# ---------------------------------------------------------------------------
# Surface 1: MXFP8 TN adapter core (ported from cppmega_fp8_shim.py).
#
# These helpers operate on TE MXFP8Tensor objects (the torch-side quantized
# operands). They are imported/called by the torch-side TE general_gemm wrapper
# (lever 1's backward, or a future general_gemm monkey-patch). The MXFP8Tensor
# is duck-typed by class name ("MXFP8" in type(_x).__name__) so this module does
# NOT import transformer_engine at module load (keeps the Mac import clean).
# ---------------------------------------------------------------------------
def is_mxfp8_tensor(x: Any) -> bool:
    """True iff ``x`` is a TE MXFP8Tensor (duck-typed by class name).

    Mirrors ``_cppmega_is_mxfp8_tensor`` from the shim -- the same name-based
    check, so it works without importing transformer_engine on this Mac.
    """

    return "MXFP8" in type(x).__name__


def is_mxfp8_rowwise_transpose_operand(x: Any) -> bool:
    """True iff ``x`` already carries a GEMM-ready rowwise-transpose marker.

    Mirrors ``_cppmega_is_mxfp8_rowwise_transpose_operand``. Such an operand is a
    TE-emitted transpose sidecar (forward already produced the rowwise layout for
    backward) and can be used directly as the TN operand without a copy-transpose.
    """

    return bool(
        getattr(x, "_te_rowwise_transpose_for_backward_operand", False)
        or getattr(x, "_cppmega_mxfp8_rowwise_transpose_operand", False)
    )


def _mxfp8_debug_desc(x: Any) -> str:
    """Compact MXFP8Tensor descriptor for RULE #1 RAISE messages."""

    def _shape(maybe_tensor: Any) -> Any:
        shp = getattr(maybe_tensor, "shape", None)
        return tuple(shp) if shp is not None else None

    return (
        f"type={type(x).__name__}, shape={getattr(x, 'shape', None)}, "
        f"dtype={getattr(x, 'dtype', None)}, "
        f"fp8_dtype={getattr(x, '_fp8_dtype', None)}, "
        f"rowwise_data={_shape(getattr(x, '_rowwise_data', None))}, "
        f"columnwise_data={_shape(getattr(x, '_columnwise_data', None))}, "
        f"with_gemm_swizzled_scales={getattr(x, '_with_gemm_swizzled_scales', None)}, "
        f"is_param={getattr(x, '_is_param', None)}"
    )


def mxfp8_colwise_as_rowwise_transpose(x: Any) -> Any:
    """Return a rowwise-transpose MXFP8 operand suitable for a TN-layout GEMM.

    This is the heart of the TN adapter (ported from
    ``cppmega_fp8_shim._cppmega_mxfp8_colwise_as_rowwise_transpose``, lines
    1392-1500). cuBLASLt MXFP8 cannot do the backward NN/NT layouts on cc12.x, but
    it CAN do TN. The transpose of the columnwise MXFP8 payload + scale yields the
    rowwise layout the TN GEMM wants:

      * If ``x`` is already a rowwise-transpose operand (TE emitted a transpose
        sidecar in forward), return it directly -- no copy.
      * Else, take the compact (non-swizzled) columnwise payload/scale, transpose
        both, and wrap them in a fresh rowwise-only MXFP8Tensor.

    RAISES (RULE #1) if ``x`` is not MXFP8, if the scales are GEMM-swizzled
    (the adapter requires compact, non-swizzled scales), if the columnwise
    payload/scales are missing or mis-shaped, or if ``_fp8_dtype`` is absent.
    There is no copy-to-bf16 fallback.

    NOTE: this materializes a real transpose copy (genuine memory cost) -- the
    adapter is "on par", not a win; the win would be a native columnwise loader
    (cutlass_native), which is out of scope for this lever.
    """

    if not is_mxfp8_tensor(x):
        raise TypeError(
            f"mxfp8_colwise_as_rowwise_transpose: expected MXFP8 tensor, got "
            f"{type(x).__name__}"
        )

    # Fast path: TE already emitted a rowwise-transpose sidecar/operand.
    if is_mxfp8_rowwise_transpose_operand(x):
        return x

    # Compact, non-swizzled columnwise scales are REQUIRED. GEMM-swizzled scales
    # cannot be transposed compactly; surface that rather than guess.
    if getattr(x, "_with_gemm_swizzled_scales", False):
        raise ValueError(
            "mxfp8_colwise_as_rowwise_transpose: MXFP8 TN adapter requires "
            "compact, non-swizzled scales but operand uses GEMM-swizzled scales; "
            f"{_mxfp8_debug_desc(x)}"
        )

    data = getattr(x, "_columnwise_data", None)
    scale = getattr(x, "_columnwise_scale_inv", None)
    if data is None or scale is None:
        raise ValueError(
            "mxfp8_colwise_as_rowwise_transpose: MXFP8 TN adapter requires "
            f"columnwise data and scales; {_mxfp8_debug_desc(x)}"
        )
    if getattr(data, "dim", lambda: 0)() < 2:
        raise ValueError(
            "mxfp8_colwise_as_rowwise_transpose: MXFP8 TN adapter requires "
            f"matrix-like columnwise data; {_mxfp8_debug_desc(x)}"
        )
    if scale.dim() != 2:
        raise ValueError(
            "mxfp8_colwise_as_rowwise_transpose: MXFP8 TN adapter requires 2D "
            f"compact columnwise scales; {_mxfp8_debug_desc(x)}"
        )

    fp8_dtype = getattr(x, "_fp8_dtype", None)
    if fp8_dtype is None:
        raise ValueError(
            "mxfp8_colwise_as_rowwise_transpose: MXFP8 tensor is missing "
            f"_fp8_dtype; {_mxfp8_debug_desc(x)}"
        )

    # transformer_engine is imported lazily here (only when an MXFP8 operand is
    # actually being adapted on the live torch host) -- never at module load.
    try:
        from transformer_engine.pytorch.tensor import (  # noqa: PLC0415
            MXFP8Quantizer as _TE_MXFP8Quantizer,
        )
        from transformer_engine.pytorch.tensor.mxfp8_tensor import (  # noqa: PLC0415
            MXFP8Tensor as _TE_MXFP8Tensor,
        )
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what
        raise RuntimeError(
            "mxfp8_colwise_as_rowwise_transpose: the MXFP8 TN adapter was "
            "SELECTED but transformer_engine MXFP8 tensor internals are "
            f"unavailable: {type(exc).__name__}: {exc}. RULE #1: a selected "
            "adapter RAISES -- it does NOT fall back to bf16."
        ) from exc

    data_2d = data.reshape(-1, data.shape[-1])
    rowwise_data = data_2d.t().contiguous()
    rowwise_scale = scale.t().contiguous()
    fake_dtype = getattr(x, "_dtype", getattr(x, "dtype", None))

    quantizer = _TE_MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=False)
    quantizer.internal = True
    quantizer.optimize_for_gemm = False
    return _TE_MXFP8Tensor(
        shape=rowwise_data.shape,
        dtype=fake_dtype,
        fp8_dtype=fp8_dtype,
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale,
        columnwise_data=None,
        columnwise_scale_inv=None,
        quantizer=quantizer,
        requires_grad=False,
        with_gemm_swizzled_scales=False,
    )


def _has_comm_overlap(kwargs: dict) -> bool:
    """True iff this GEMM uses comm/compute overlap (not covered by the adapter).

    Mirrors ``_cppmega_has_comm_overlap``. The TN rewrite does not cover the
    Userbuffers / bulk-overlap GEMMs, so the adapter declines them (the caller
    must RAISE, not silently bf16, per RULE #1).
    """

    return (
        kwargs.get("ub", None) is not None
        or kwargs.get("ub_type", None) is not None
        or kwargs.get("extra_output", None) is not None
        or bool(kwargs.get("bulk_overlap", False))
    )


def try_mxfp8_tn_adapter(orig_general_gemm: Any, args: Any, kwargs: dict):
    """Rewrite a backward MXFP8 NN/NT GEMM to TN and run it; (ok, result_or_reason).

    Ported from ``cppmega_fp8_shim._cppmega_try_mxfp8_tn_adapter`` (lines
    2059-2122). ``orig_general_gemm`` is TE's *original* (unwrapped)
    ``general_gemm``; ``args``/``kwargs`` are its call args (``args[0]``,
    ``args[1]`` are the two operands; ``kwargs['layout']`` is the requested
    layout; ``kwargs['grad']`` marks a backward GEMM).

    Returns ``(False, reason)`` when this GEMM is not an adapter target (adapter
    disabled, not a backward GEMM, unsupported layout, comm-overlap, non-MXFP8
    operands). Returns ``(True, result)`` after rewriting NN/NT -> TN (transposing
    the relevant columnwise operand(s)) and calling ``orig_general_gemm``.

    RULE #1: this NEVER bf16-falls-back. If a target GEMM cannot be adapted (e.g.
    a required columnwise sidecar is missing), :func:`mxfp8_colwise_as_rowwise_transpose`
    RAISES and that exception propagates -- the caller MUST surface it, not
    swallow it into a bf16 path. The ``(False, reason)`` returns are ONLY for
    GEMMs that are legitimately not adapter targets (the dense/forward path), not
    for failures of an adapter target.
    """

    layout = kwargs.get("layout")
    if not mxfp8_tn_adapter_enabled():
        return False, "adapter_disabled"
    if not kwargs.get("grad", False):
        return False, "not_backward_gemm"
    if layout not in _MXFP8_BWD_LAYOUTS:
        return False, f"unsupported_layout:{layout}"
    if not isinstance(args, (list, tuple)) or len(args) < 2:
        return False, "missing_operands"
    if _has_comm_overlap(kwargs):
        return False, "comm_overlap_not_covered"
    if not (is_mxfp8_tensor(args[0]) and is_mxfp8_tensor(args[1])):
        return False, "non_mxfp8_operands"

    new_args = list(args)
    new_kwargs = dict(kwargs)
    new_kwargs["layout"] = "TN"
    # The GB10 probe validated the adapter with compact scales and non-split
    # accumulation. Native MXFP8 NN/NT fails regardless of split-accumulator, so
    # force the known-good TN mode (ported verbatim from the cppmega shim).
    new_kwargs["use_split_accumulator"] = False

    if layout == "NN":
        # dgrad: transpose operand 0 (the weight) columnwise->rowwise.
        new_args[0] = mxfp8_colwise_as_rowwise_transpose(new_args[0])
    else:  # "NT" -> wgrad: transpose both operands.
        new_args[0] = mxfp8_colwise_as_rowwise_transpose(new_args[0])
        new_args[1] = mxfp8_colwise_as_rowwise_transpose(new_args[1])

    return True, orig_general_gemm(*new_args, **new_kwargs)


def run_mxfp8_general_gemm_with_tn_adapter(
    orig_general_gemm: Any, args: Any, kwargs: dict
):
    """Drive ``orig_general_gemm`` with the TN adapter applied to backward MXFP8.

    Thin wrapper mirroring the dispatch body of the cppmega shim's
    ``_cppmega_wrap_general_gemm`` (lines 2371-2476) but RULE #1-strict: when the
    adapter is enabled AND this is a backward MXFP8 NN/NT GEMM, the adapter is the
    ONE path. If the adapter declines a *target* GEMM (returns False for a
    backward MXFP8 NN/NT with MXFP8 operands -- e.g. comm-overlap), this RAISES
    rather than passing through to the (cuBLASLt-unsupported, on cc12.x) native
    NN/NT GEMM. Non-target GEMMs (forward, non-MXFP8, dense) pass straight
    through to ``orig_general_gemm`` untouched.

    A general_gemm monkey-patch (the torch-side call site) can use this directly:
    ``module.general_gemm = lambda *a, **k: run_mxfp8_general_gemm_with_tn_adapter(orig, a, k)``.
    """

    layout = kwargs.get("layout")
    is_backward_mxfp8_target = (
        mxfp8_tn_adapter_enabled()
        and kwargs.get("grad", False)
        and layout in _MXFP8_BWD_LAYOUTS
        and isinstance(args, (list, tuple))
        and len(args) >= 2
        and is_mxfp8_tensor(args[0])
        and is_mxfp8_tensor(args[1])
    )
    if not is_backward_mxfp8_target:
        return orig_general_gemm(*args, **kwargs)

    op_kind = "dgrad" if layout == "NN" else "wgrad"
    adapted_ok, result_or_reason = try_mxfp8_tn_adapter(
        orig_general_gemm, args, kwargs
    )
    if adapted_ok:
        return result_or_reason
    # A backward MXFP8 NN/NT target that the adapter declined (e.g.
    # comm-overlap). RULE #1: do NOT pass through to the native NN/NT GEMM
    # (CUBLAS_STATUS_NOT_SUPPORTED on cc12.x) and do NOT bf16-fallback.
    raise RuntimeError(
        f"run_mxfp8_general_gemm_with_tn_adapter: MXFP8 TN adapter could not "
        f"adapt a backward {op_kind} (layout={layout}) GEMM that IS an adapter "
        f"target: {result_or_reason}. RULE #1: refusing the cuBLASLt-unsupported "
        f"native MXFP8 {layout} GEMM on cc12.x and refusing a bf16 fallback."
    )


# ---------------------------------------------------------------------------
# Surface 2: NVFP4 RtN GEMM as a path_c GEMM option.
#
# Wires the reduced round-to-nearest NVFP4 recipe (built + gated in
# scripts/_nvfp4_route.py) into an executable GEMM that takes MLX operands and
# runs the FP4 forward (and, on the VJP, the RtN dgrad/wgrad) through a te.Linear
# over the _cuda_zerocopy DLPack bridge. The recipe machinery is imported lazily
# from scripts._nvfp4_route so we neither fork nor duplicate it.
# ---------------------------------------------------------------------------
def _import_nvfp4_route():
    """Import the NVFP4 route module (recipe builder + autocast + gate).

    Lives in ``scripts/_nvfp4_route.py`` (already committed). RAISES (RULE #1) if
    it cannot be imported -- a selected NVFP4 GEMM does not silently fall back.
    """

    try:
        from scripts import _nvfp4_route  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what
        raise RuntimeError(
            "mxfp8_tn_adapter: the NVFP4 path_c GEMM was SELECTED "
            f"({NVFP4_PATH_C_GEMM_ENV} on) but scripts._nvfp4_route is "
            f"unavailable: {type(exc).__name__}: {exc}. Ensure the repo root is "
            "on sys.path (m04_train_step imports it the same way). RULE #1: a "
            "selected NVFP4 GEMM RAISES -- it does NOT fall back to bf16."
        ) from exc
    return _nvfp4_route


# NVFP4 (e2m1 + 16-element block scale) requires K to be a multiple of the block
# size; surface a config bug rather than let TE produce garbage.
NVFP4_BLOCK_SIZE = 16


def _check_nvfp4_shape(k_in: int, n_out: int) -> None:
    if k_in % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"nvfp4_path_c_gemm: K={k_in} must be a multiple of the NVFP4 block "
            f"size {NVFP4_BLOCK_SIZE} (e2m1 + per-block-{NVFP4_BLOCK_SIZE} scale)."
        )
    if n_out % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"nvfp4_path_c_gemm: N={n_out} must be a multiple of the NVFP4 block "
            f"size {NVFP4_BLOCK_SIZE}."
        )


_NVFP4_RTN_SUPPORTED: list[bool | None] = [None]


def _assert_nvfp4_rtn_backward_supported(route_mod: Any) -> None:
    """Gate: the reduced RtN NVFP4 backward MUST pass its numeric probe.

    Caches the boolean for the process. RAISES (RULE #1) if the gate fails -- the
    NVFP4 backward GEMM is only the ONE path when its dgrad/wgrad have been proven
    finite + within tolerance vs bf16 on THIS device. Never returns a degraded
    path. The default (RHT+SR) recipe is datacenter-only and is NOT used here.
    """

    if _NVFP4_RTN_SUPPORTED[0] is True:
        return
    supported = bool(route_mod.nvfp4_backward_rtn_supported())
    _NVFP4_RTN_SUPPORTED[0] = supported
    if not supported:
        raise RuntimeError(
            "nvfp4_path_c_gemm(bwd): the reduced round-to-nearest NVFP4 backward "
            "did not pass its numeric gate on this host "
            "(nvfp4_backward_rtn_supported()==False). RULE #1: refusing to run a "
            "VJP whose FP4 dgrad/wgrad is not proven finite + within tolerance, "
            "and refusing a bf16 fallback. See "
            "scripts/_nvfp4_route.nvfp4_te_backward_rtn_probe for the detail."
        )


def _mlx_dtype_to_torch(dtype, torch_mod):
    if dtype == mx.bfloat16:
        return torch_mod.bfloat16
    if dtype == mx.float16:
        return torch_mod.float16
    if dtype == mx.float32:
        return torch_mod.float32
    raise TypeError(
        f"nvfp4_path_c_gemm: unsupported MLX dtype {dtype} for the NVFP4 RtN "
        f"GEMM; expected bfloat16/float16/float32 master operands."
    )


def nvfp4_path_c_gemm(x: mx.array, weight: mx.array) -> mx.array:
    """Forward NVFP4 RtN GEMM as a path_c GEMM option: y = x @ weight.T (FP4).

    ``x`` is ``(.., K)`` MLX, ``weight`` is ``(N, K)`` MLX (torch ``nn.Linear``
    convention). Bridges both to torch CUDA tensors zero-copy via lever 1's
    ``_cuda_zerocopy`` bridge, runs the FP4 forward GEMM through a ``te.Linear``
    under the reduced RtN NVFP4 autocast (built + checked in
    ``scripts/_nvfp4_route``), and bridges the bf16 result back to MLX.

    RULE #1: gated by :func:`nvfp4_path_c_gemm_enabled` (default OFF) at the call
    site; when invoked it is the ONE path and RAISES on any failure (bridge
    reject, TE/torch unavailable, recipe-not-RtN, shape ineligible) -- never a
    silent bf16 / host-copy downgrade.
    """

    from cppmega_mlx.nn._tilelang._cuda_zerocopy import (  # noqa: PLC0415
        mlx_cuda_array_to_torch_tensor,
        torch_cuda_tensor_to_mlx,
    )

    if weight.ndim != 2:
        raise ValueError(
            f"nvfp4_path_c_gemm: weight must be 2-D (N, K), got "
            f"{tuple(weight.shape)}."
        )
    n_out, k_in = int(weight.shape[0]), int(weight.shape[1])
    if int(x.shape[-1]) != k_in:
        raise ValueError(
            f"nvfp4_path_c_gemm: x last dim {int(x.shape[-1])} != weight K "
            f"{k_in} (x {tuple(x.shape)}, weight {tuple(weight.shape)})."
        )
    _check_nvfp4_shape(k_in, n_out)

    route_mod = _import_nvfp4_route()

    try:
        import torch  # noqa: PLC0415
        import transformer_engine.pytorch as te  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what
        raise RuntimeError(
            "nvfp4_path_c_gemm: the NVFP4 RtN GEMM was SELECTED but "
            f"torch/transformer_engine is unavailable: {type(exc).__name__}: "
            f"{exc}. RULE #1: a selected NVFP4 GEMM RAISES -- no bf16 fallback."
        ) from exc

    out_dtype = x.dtype
    lead = tuple(int(d) for d in x.shape[:-1])
    x2 = x.reshape((-1, k_in)) if x.ndim != 2 else x

    try:
        x_t = mlx_cuda_array_to_torch_tensor(x2)
        w_t = mlx_cuda_array_to_torch_tensor(weight)
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what
        raise RuntimeError(
            "nvfp4_path_c_gemm: MLX->torch zero-copy bridge failed for the NVFP4 "
            f"forward (x {tuple(x2.shape)}, w {tuple(weight.shape)}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    torch_dtype = _mlx_dtype_to_torch(out_dtype, torch)
    recipe = route_mod.build_nvfp4_rtn_recipe()

    lin = te.Linear(k_in, n_out, bias=False, params_dtype=torch_dtype).to("cuda")
    with torch.no_grad():
        if tuple(lin.weight.shape) != tuple(w_t.shape):
            raise RuntimeError(
                f"nvfp4_path_c_gemm: te.Linear weight shape "
                f"{tuple(lin.weight.shape)} != MLX weight {tuple(w_t.shape)}."
            )
        lin.weight.copy_(w_t.to(lin.weight.dtype))
        with route_mod._te_autocast(te, recipe):
            out_t = lin(x_t.to(torch_dtype))

    out = torch_cuda_tensor_to_mlx(out_t, out_dtype=out_dtype)
    if lead:
        out = out.reshape((*lead, n_out))
    return out


def nvfp4_path_c_gemm_vjp(
    x: mx.array, weight: mx.array, cotangent: mx.array
) -> tuple[mx.array, mx.array]:
    """NVFP4 RtN backward: (dgrad wrt x, wgrad wrt weight) as MLX arrays.

    Re-runs the FP4 forward with torch autograd enabled under the reduced RtN
    autocast and backpropagates the MLX ``cotangent`` to recover dgrad (x.grad)
    and wgrad (weight.grad) -- both computed by TE in FP4 RtN (the MEASURED
    dgrad/wgrad GEMMs: rel_err ~0.1465 / ~0.1343 vs bf16 on gb10 sm_121). Gated by
    the numeric probe :func:`_assert_nvfp4_rtn_backward_supported`.

    RULE #1: when invoked it is the ONE path; RAISES on any failure (gate fail,
    bridge reject, TE/torch unavailable) -- never bf16-falls-back.
    """

    from cppmega_mlx.nn._tilelang._cuda_zerocopy import (  # noqa: PLC0415
        mlx_cuda_array_to_torch_tensor,
        torch_cuda_tensor_to_mlx,
    )

    if weight.ndim != 2:
        raise ValueError(
            f"nvfp4_path_c_gemm_vjp: weight must be 2-D (N, K), got "
            f"{tuple(weight.shape)}."
        )
    n_out, k_in = int(weight.shape[0]), int(weight.shape[1])
    _check_nvfp4_shape(k_in, n_out)

    route_mod = _import_nvfp4_route()

    try:
        import torch  # noqa: PLC0415
        import transformer_engine.pytorch as te  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what
        raise RuntimeError(
            "nvfp4_path_c_gemm_vjp: the NVFP4 RtN backward was SELECTED but "
            f"torch/transformer_engine is unavailable: {type(exc).__name__}: "
            f"{exc}. RULE #1: a selected NVFP4 backward RAISES -- no bf16 fallback."
        ) from exc

    # Numeric gate FIRST: the FP4 dgrad/wgrad must be proven on this host.
    _assert_nvfp4_rtn_backward_supported(route_mod)

    out_dtype = x.dtype
    lead = tuple(int(d) for d in x.shape[:-1])
    x2 = x.reshape((-1, k_in)) if x.ndim != 2 else x
    g2 = cotangent.reshape((-1, n_out)) if cotangent.ndim != 2 else cotangent

    try:
        x_t = mlx_cuda_array_to_torch_tensor(x2)
        w_t = mlx_cuda_array_to_torch_tensor(weight)
        g_t = mlx_cuda_array_to_torch_tensor(g2)
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what
        raise RuntimeError(
            "nvfp4_path_c_gemm_vjp: MLX->torch zero-copy bridge failed for the "
            f"NVFP4 backward: {type(exc).__name__}: {exc}"
        ) from exc

    torch_dtype = _mlx_dtype_to_torch(out_dtype, torch)
    recipe = route_mod.build_nvfp4_rtn_recipe()
    lin = te.Linear(k_in, n_out, bias=False, params_dtype=torch_dtype).to("cuda")
    with torch.no_grad():
        if tuple(lin.weight.shape) != tuple(w_t.shape):
            raise RuntimeError(
                f"nvfp4_path_c_gemm_vjp: te.Linear weight shape "
                f"{tuple(lin.weight.shape)} != MLX weight {tuple(w_t.shape)}."
            )
        lin.weight.copy_(w_t.to(lin.weight.dtype))
    if lin.weight.grad is not None:
        lin.weight.grad = None

    x_leaf = x_t.to(torch_dtype).detach().requires_grad_(True)
    with route_mod._te_autocast(te, recipe):
        y = lin(x_leaf)
    y.backward(g_t.to(y.dtype))

    if x_leaf.grad is None:
        raise RuntimeError(
            "nvfp4_path_c_gemm_vjp: te.Linear backward produced no x.grad under "
            "the NVFP4 RtN autocast; the FP4 dgrad GEMM did not run."
        )
    if lin.weight.grad is None:
        raise RuntimeError(
            "nvfp4_path_c_gemm_vjp: te.Linear backward produced no weight.grad "
            "under the NVFP4 RtN autocast; the FP4 wgrad GEMM did not run."
        )

    dgrad = torch_cuda_tensor_to_mlx(x_leaf.grad, out_dtype=out_dtype)
    wgrad = torch_cuda_tensor_to_mlx(lin.weight.grad, out_dtype=out_dtype)
    if lead:
        dgrad = dgrad.reshape((*lead, k_in))
    return dgrad, wgrad


__all__ = [
    "MXFP8_TN_ADAPTER_ENV",
    "NVFP4_PATH_C_GEMM_ENV",
    "NVFP4_BLOCK_SIZE",
    "mxfp8_tn_adapter_enabled",
    "nvfp4_path_c_gemm_enabled",
    "is_mxfp8_tensor",
    "is_mxfp8_rowwise_transpose_operand",
    "mxfp8_colwise_as_rowwise_transpose",
    "try_mxfp8_tn_adapter",
    "run_mxfp8_general_gemm_with_tn_adapter",
    "nvfp4_path_c_gemm",
    "nvfp4_path_c_gemm_vjp",
]
