# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Path C FP8 amax + quantize via TileLang DSL lowering.

This module is the TileLang-DSL counterpart to the CUDA-only Triton kernels
``_amax_kernel`` and ``_quantize_kernel`` defined in
``cppmega/cppmega/megatron/fp8_activations.py``. It is the Tier-1 PoC of the
unified fused-kernel pipeline: a single TileLang source compiles for both
CUDA and Apple Metal SIMDgroup targets, replacing the ``tensor.is_cuda``-gated
Triton path on CUDA hosts and providing the previously-missing Metal path.

Source attribution
------------------

The reference Triton kernels ported here live in:

* cppmega/cppmega/megatron/fp8_activations.py (``_amax_kernel`` line ~287,
  ``_quantize_kernel`` line ~310). Original style guide:
  ``cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py``.

The CUDA emission of the resulting TileLang PrimFunc is numerically equivalent
to the Triton kernel; the Metal emission relies on ``tvm_thread_allreduce``
plus a global ``T.atomic_max`` for the cross-block reduction (atomic_max on
fp32 is implemented via a CAS loop on Metal, matching CUDA's atomicMax fp32).

Deferred features (NOT implemented in this PoC)
-----------------------------------------------

* fp8 e5m2 variant -- the Triton reference quantizes only to e4m3fn
  (``tl.float8e4nv``); an e5m2 variant would need a separate kernel + scale
  semantics.
* bf16 input -- the kernels here accept fp16 input (and the wrapper auto-casts
  to fp16 for bf16). A native bf16 path would let us skip the upcast inside
  the kernel, but the Triton reference already widens to fp32 so the saving
  would be modest.
* Stochastic rounding for quantize -- both Triton and this port use IEEE
  round-to-nearest-even via ``T.cast(..., "float8_e4m3")``. RNE is the FP8
  default in PyTorch; SR is a follow-up.
* Fused amax + quantize -- the current pipeline keeps the Triton two-pass
  shape (launch 1: amax; host syncs; launch 2: quantize). A single-launch
  fused kernel would save one device synchronization at the cost of an
  extra round-trip through global memory for the amax broadcast.

API surface
-----------

* :func:`fp8_amax_tilelang` -- compute per-tensor abs-max (fp32 scalar) of a
  fp16/bf16/fp32 tensor via a TileLang block-reduce + atomic_max kernel.
* :func:`fp8_quantize_tilelang` -- given an inv_scale, scale + clamp + cast to
  fp8 e4m3fn via a TileLang elementwise kernel.
* :func:`tilelang_supports` -- runtime gate the patched
  ``cppmega/megatron/fp8_activations.py`` uses to decide between TileLang,
  Triton and the unfused PyTorch fallback.
* :func:`fp8_amax_path_c_status` -- importability + reason (mirrors the
  ``fp8_vecmat_path_c_status`` style).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import torch


# ---------------------------------------------------------------------------
# Kernel-shape defaults -- TileLang resolves these globals while decorating
# the nested @T.prim_func, mirroring fp8_vecmat_path_c.py's ``_FP8_VM_*``.
#
# Per-target (BLOCK_SIZE, THREADS) defaults
# -----------------------------------------
#
# +---------+-----------+----------+--------------------------------+
# | target  | BLOCK     | THREADS  | rationale                      |
# +---------+-----------+----------+--------------------------------+
# | cuda    | 1024      | 128      | warp=32; 4 warps; vector-frnd. |
# | hip     | 1024      | 256      | warp=64; 4 warps                |
# | metal   | 256       |  64      | simdgroup=32; 2 simdgroups      |
# | <other> | 1024      | 128      | conservative fallback           |
# +---------+-----------+----------+--------------------------------+
#
# INVARIANT: BLOCK % THREADS == 0 (the strided inner ``T.Parallel`` loop
# requires a clean stride; a non-divisible pair leaves a partial tail
# uncovered on the last block). Enforced in the kernel builders.
# ---------------------------------------------------------------------------

_FP8_AMAX_BLOCK_SIZE = 1024  # legacy default; retained for API compatibility.
_FP8_AMAX_THREADS = 128
_FP8_QUANT_BLOCK_SIZE = 1024
_FP8_QUANT_THREADS = 128

_BLOCK_SIZE_TABLE: dict[str, tuple[int, int]] = {
    "cuda": (1024, 128),
    "hip": (1024, 256),
    "rocm": (1024, 256),  # hip alias
    "metal": (256, 64),
}


def _target_family(target: str) -> str:
    if target.startswith("metal"):
        return "metal"
    if target.startswith("hip") or target.startswith("rocm"):
        return "hip"
    if target.startswith("cuda") or target.startswith("nvptx"):
        return "cuda"
    return "cuda"  # conservative default


def _pick_block_size(target: str, n_elements: int) -> tuple[int, int]:
    """Return ``(block_size, threads)`` for *target* and *n_elements*.

    Enforces ``block_size % threads == 0`` and ``block_size >= threads``.
    Shrinks to the next power-of-two when ``n_elements`` is smaller than the
    default block (avoids a fully-masked block on tiny inputs).
    """

    block, threads = _BLOCK_SIZE_TABLE.get(_target_family(target), (1024, 128))
    if n_elements > 0 and n_elements < block:
        # round n_elements up to next pow2, clamped to >= threads
        snapped = 1 << max(1, n_elements - 1).bit_length()
        block = max(threads, snapped)
    if block % threads != 0:  # pragma: no cover - table invariant
        # Snap block up to the next multiple of threads.
        block = ((block + threads - 1) // threads) * threads
    return block, threads


def _bucket_n(n: int, block_size: int) -> int:
    """Round *n* up to a power-of-two bucket >= ``block_size``.

    Two close call shapes (e.g. ``N=4097`` and ``N=5000``) bucket to the
    same kernel (``8192``), eliminating per-shape JIT compile thrashing.
    The cost is a single allocation + copy of zeros for the tail; the
    amax of a zero tail is the identity element of ``max`` so the result
    is unchanged.
    """

    if n <= block_size:
        return block_size
    return 1 << (n - 1).bit_length()

# Triton's _quantize_kernel hard-codes fp8 e4m3fn; max representable value is
# 448.0 (= torch.finfo(torch.float8_e4m3fn).max).
_FP8_E4M3_MAX: float = 448.0


@dataclass(frozen=True)
class FP8AmaxPathCStatus:
    """Runtime/lowering status for the Path C TileLang FP8 amax/quantize kernels."""

    available: bool
    reason: str
    cuda_target: str = "cuda"
    metal_target: str = "metal"


def _tilelang_available() -> tuple[bool, str]:
    try:
        import tilelang  # noqa: F401
        from tilelang import tvm as _tvm  # noqa: F401
        import tilelang.language as _T  # noqa: F401
    except Exception as exc:  # pragma: no cover - hosts without TileLang
        return False, f"tilelang import failed: {exc}"
    return True, "tilelang importable"


def fp8_amax_path_c_status() -> FP8AmaxPathCStatus:
    """Return whether TileLang is importable for Path C amax/quantize lowering."""

    ok, reason = _tilelang_available()
    if not ok:
        return FP8AmaxPathCStatus(available=False, reason=reason)
    return FP8AmaxPathCStatus(
        available=True,
        reason="FP8 amax/quantize Path C TileLang DSL lowering is available",
    )


def tilelang_supports_with_reason(
    device: torch.device | str | None,
) -> tuple[bool, str]:
    """Return ``(supported, reason)`` for *device*.

    Every return is a 2-tuple ``(bool, str)`` -- both the success and the
    failure paths carry a human-readable reason for diagnostics. The
    boolean-only :func:`tilelang_supports` is a thin wrapper over this for
    backward compatibility with the ``cppmega/megatron/fp8_activations.py``
    dispatch gate.
    """

    ok, reason = _tilelang_available()
    if not ok:
        return False, reason
    if device is None:
        return False, "device is None"
    if isinstance(device, str):
        try:
            dev_type = torch.device(device).type
        except (RuntimeError, ValueError) as exc:
            return False, f"unparseable device string {device!r}: {exc}"
    else:
        dev_type = device.type
    if dev_type == "cuda":
        if torch.cuda.is_available():
            return True, "cuda available"
        return False, "cuda requested but torch.cuda.is_available() is False"
    if dev_type == "mps":
        has_mps = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if has_mps:
            return True, "mps available"
        return False, "mps requested but torch.backends.mps.is_available() is False"
    return False, f"unsupported device type: {dev_type!r}"


def tilelang_supports(device: torch.device | str | None) -> bool:
    """Return True when the TileLang amax/quantize port can dispatch on *device*.

    The TileLang JIT supports CUDA (``cuda``) and Apple Metal (``mps`` /
    ``metal``) targets. CPU tensors must continue to use the unfused
    ``tensor.abs().amax()`` PyTorch fallback.

    Thin wrapper over :func:`tilelang_supports_with_reason` that drops the
    diagnostic reason. Kept as a stable bool API for the
    ``cppmega/megatron/fp8_activations.py`` dispatch gate.
    """

    return tilelang_supports_with_reason(device)[0]


def _resolve_target(device: torch.device) -> str:
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        # TileLang's Metal backend uses the same target string family as
        # fp8_vecmat_path_c.py; explicit warp size keeps codegen aligned with
        # Apple SIMDgroup width.
        return "metal -thread_warp_size=32"
    raise ValueError(f"fp8_amax_tilelang: unsupported device type {device.type!r}")


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------


def make_fp8_amax_kernel(
    *,
    n_elements: int,
    in_dtype: str = "float16",
    block_size: int = _FP8_AMAX_BLOCK_SIZE,
    threads: int = _FP8_AMAX_THREADS,
) -> Any:
    """Build a shape-specialized abs-max reducer.

    Mirrors the Triton ``_amax_kernel`` reference: each block reduces
    ``block_size`` elements via :func:`T.reduce_max` over ``T.abs(x)``, then a
    single thread issues a global :func:`T.atomic_max` against a fp32 scalar
    output. The pre-zeroed output buffer is the caller's responsibility (matches
    the Triton call-site which passes ``torch.zeros(1, dtype=torch.float32)``).

    Inputs:
        ``X``: ``(N,)`` ``in_dtype`` (fp16 / bf16 / fp32).
        ``Amax``: ``(1,)`` fp32, pre-zeroed.

    The TileLang DSL surface used here is intentionally narrow so the same
    PrimFunc lowers to both CUDA and Metal:

    * ``T.alloc_shared`` -- per-block staging buffer, fp16/bf16/fp32.
    * ``T.alloc_fragment`` -- per-block fp32 scratch for the reduce output.
    * ``T.copy`` -- coalesced global -> shared load (vectorizes on CUDA, maps
      to threadgroup loads on Metal).
    * ``T.reduce_max`` -- the standard tilelang block reducer.
    * ``T.atomic_max`` -- scalar fp32 atomic_max against the global counter.
    """

    if n_elements <= 0:
        raise ValueError(f"n_elements must be positive; got {n_elements}")
    if block_size <= 0 or threads <= 0:
        raise ValueError(f"block_size/threads must be positive; got {block_size}, {threads}")
    if block_size % threads != 0:
        raise RuntimeError(
            f"fp8_amax: BLOCK_SIZE={block_size} not divisible by THREADS={threads} "
            f"(N={n_elements}); the strided ``T.Parallel(BLOCK)`` inner loop "
            f"requires a clean stride, otherwise the last block emits a partial "
            f"tail that is not covered by any thread. Pick block_size as a "
            f"multiple of threads (e.g. {((block_size + threads - 1) // threads) * threads})."
        )

    import tilelang.language as T

    T = cast(Any, T)

    N = n_elements
    BLOCK = block_size

    @T.prim_func
    def fp8_amax_reduce(
        X: T.Tensor((N,), in_dtype),
        Amax: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK), threads=threads) as bx:
            X_abs = T.alloc_fragment((BLOCK,), "float32")
            local_amax = T.alloc_fragment((1,), "float32")

            # Single-pass load + abs + cast to fp32. Loading directly from
            # global into the per-block fragment removes the prior
            # global -> shared -> fragment double-pass (two T.Parallel loops
            # without an intervening barrier was a shared-memory data race
            # in addition to the wasted bandwidth). The mask handles the
            # last partial block; T.Parallel emits vectorized / coalesced
            # loads on both CUDA and Metal.
            for i in T.Parallel(BLOCK):
                gi = bx * BLOCK + i
                if gi < N:
                    X_abs[i] = T.abs(T.cast(X[gi], "float32"))
                else:
                    X_abs[i] = T.cast(0, "float32")

            T.reduce_max(X_abs, local_amax, dim=0, clear=True)

            # Single thread writes the block-local amax via atomic_max into
            # the global fp32 scalar. TileLang lowers this to atomicMax on
            # CUDA and to a CAS loop on Metal (atomicMax on fp32 is not a
            # native MSL primitive).
            if T.get_thread_binding(0) == 0:
                T.atomic_max(Amax, local_amax[0])

    return fp8_amax_reduce


def make_fp8_quantize_kernel(
    *,
    n_elements: int,
    in_dtype: str = "float16",
    block_size: int = _FP8_QUANT_BLOCK_SIZE,
    threads: int = _FP8_QUANT_THREADS,
) -> Any:
    """Build a shape-specialized quantize-to-fp8 e4m3fn kernel.

    Mirrors the Triton ``_quantize_kernel`` reference: load fp16/bf16/fp32,
    multiply by ``inv_scale`` (= ``fp8_max / amax``), clamp to
    ``[-fp8_max, fp8_max]``, then cast to fp8 e4m3fn with IEEE
    round-to-nearest-even (PyTorch / TVM default for ``T.cast(..., "float8_e4m3")``).

    Inputs:
        ``X``: ``(N,)`` ``in_dtype``.
        ``Y``: ``(N,)`` fp8_e4m3 output (uint8 storage at the torch boundary).
        ``InvScale``: ``(1,)`` fp32 scalar -- pre-computed on the host as
            ``fp8_max / amax``. Passed as a buffer to keep the PrimFunc
            signature device-portable; the alternative scalar kernel argument
            shape differs subtly between CUDA and Metal launch glue.
    """

    if n_elements <= 0:
        raise ValueError(f"n_elements must be positive; got {n_elements}")
    if block_size <= 0 or threads <= 0:
        raise ValueError(f"block_size/threads must be positive; got {block_size}, {threads}")
    if block_size % threads != 0:
        raise RuntimeError(
            f"fp8_quantize: BLOCK_SIZE={block_size} not divisible by THREADS={threads} "
            f"(N={n_elements}); see fp8_amax.py BLOCK_SIZE_TABLE invariant."
        )

    import tilelang.language as T

    T = cast(Any, T)

    N = n_elements
    BLOCK = block_size
    FP8_MAX = _FP8_E4M3_MAX

    @T.prim_func
    def fp8_quantize_e4m3(
        X: T.Tensor((N,), in_dtype),
        InvScale: T.Tensor((1,), "float32"),
        Y: T.Tensor((N,), "float8_e4m3"),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK), threads=threads) as bx:
            for i in T.Parallel(BLOCK):
                gi = bx * BLOCK + i
                if gi < N:
                    v = T.cast(X[gi], "float32") * InvScale[0]
                    # T.max / T.min compose a clamp; this matches the Triton
                    # ``tl.clamp`` semantics (NaN-passthrough is undefined in
                    # both references).
                    v = T.max(v, T.cast(-FP8_MAX, "float32"))
                    v = T.min(v, T.cast(FP8_MAX, "float32"))
                    Y[gi] = T.cast(v, "float8_e4m3")

    return fp8_quantize_e4m3


# ---------------------------------------------------------------------------
# JIT cache + torch dispatch
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _amax_kernel_for(bucket_n: int, in_dtype: str, target: str) -> Any:
    """Build, JIT-compile, and cache the amax kernel keyed on the *bucket* N.

    ``bucket_n`` is :func:`_bucket_n`'s power-of-two rounding so close call
    shapes share a single compiled kernel. The dispatcher pads the input
    with zeros up to ``bucket_n`` before launching; ``amax(0) = 0`` is the
    identity for ``max`` so the result is unchanged.
    """

    import tilelang

    block, threads = _pick_block_size(target, bucket_n)
    prim = make_fp8_amax_kernel(
        n_elements=bucket_n,
        in_dtype=in_dtype,
        block_size=block,
        threads=threads,
    )
    return tilelang.compile(prim, target=target, out_idx=None)


@lru_cache(maxsize=256)
def _quantize_kernel_for(n_elements: int, in_dtype: str, target: str) -> Any:
    """Build, JIT-compile, and cache the quantize kernel for a (shape, dtype, target).

    Quantize is keyed on the *exact* shape -- output sizing and the per-
    element ``Y[gi]`` write means the kernel must own exactly ``n_elements``
    output slots. Cache size is bumped to ``256`` to cover realistic LLM
    training rotations without thrashing.
    """

    import tilelang

    block, threads = _pick_block_size(target, n_elements)
    prim = make_fp8_quantize_kernel(
        n_elements=n_elements,
        in_dtype=in_dtype,
        block_size=block,
        threads=threads,
    )
    return tilelang.compile(prim, target=target, out_idx=None)


_TORCH_DTYPE_TO_TL: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def _resolve_in_dtype(tensor: torch.Tensor) -> str:
    tl_dtype = _TORCH_DTYPE_TO_TL.get(tensor.dtype)
    if tl_dtype is None:
        raise TypeError(
            f"fp8_amax_tilelang: unsupported input dtype {tensor.dtype!r}; "
            "expected one of fp16/bf16/fp32"
        )
    return tl_dtype


def fp8_amax_tilelang(x: torch.Tensor) -> torch.Tensor:
    """Compute the fp32 abs-max of *x* via the TileLang Path C kernel.

    Matches the contract of the Triton ``_amax_kernel`` invocation in
    ``_triton_fp8_pack``: returns a ``(1,)`` fp32 tensor on the same device as
    *x*, pre-initialized to zero and updated atomically by the kernel. Caller
    is responsible for converting it to a host scalar (``.item()``) when
    computing ``inv_scale`` for the quantize launch.

    Single TileLang source -- target string is selected from ``x.device``.
    """

    if x.numel() == 0:
        return torch.zeros(1, dtype=torch.float32, device=x.device)

    flat = x.reshape(-1).contiguous()
    n_actual = flat.numel()
    in_dtype = _resolve_in_dtype(flat)
    target = _resolve_target(flat.device)

    # Bucket cache key by next-pow2 to avoid per-shape JIT thrashing.
    # ``amax(zeros) == 0`` is the identity element for ``max`` so padding
    # the tail with zeros leaves the result unchanged.
    block, _threads = _pick_block_size(target, n_actual)
    bucket_n = _bucket_n(n_actual, block)

    if bucket_n != n_actual:
        padded = torch.zeros(bucket_n, dtype=flat.dtype, device=flat.device)
        padded[:n_actual] = flat
        flat = padded

    kernel = _amax_kernel_for(bucket_n, in_dtype, target)

    amax = torch.zeros(1, dtype=torch.float32, device=flat.device)
    kernel(flat, amax)
    return amax


def precompile_amax_kernel(
    n_elements: int,
    *,
    in_dtype: str = "float16",
    target: str = "cuda",
) -> None:
    """Warm the amax-kernel cache for a known shape/dtype/target.

    Useful for LLM training where the set of activation shapes is known
    ahead of time -- precompile once at startup, never pay the JIT cost
    on the hot path. Resolves to the same bucket key as the dispatcher
    so a single warm-up covers nearby shapes.
    """

    block, _threads = _pick_block_size(target, n_elements)
    bucket_n = _bucket_n(n_elements, block)
    _amax_kernel_for(bucket_n, in_dtype, target)


def fp8_quantize_tilelang(
    x: torch.Tensor,
    inv_scale: torch.Tensor | float,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize *x* to fp8 e4m3fn via the TileLang Path C kernel.

    Matches the contract of the Triton ``_quantize_kernel`` invocation in
    ``_triton_fp8_pack``: writes ``(N,)`` fp8 output (torch.float8_e4m3fn)
    where ``N == x.numel()``. ``inv_scale`` may be a Python float or a fp32
    ``(1,)`` tensor; either way it is broadcast as a single scalar.

    The output preserves the input shape unless *out* is provided.
    """

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("torch.float8_e4m3fn is not available in this PyTorch build")

    if x.numel() == 0:
        if out is not None:
            return out
        return torch.empty(x.shape, dtype=fp8_dtype, device=x.device)

    flat = x.reshape(-1).contiguous()
    n_elements = flat.numel()
    in_dtype = _resolve_in_dtype(flat)
    target = _resolve_target(flat.device)

    if isinstance(inv_scale, torch.Tensor):
        inv_scale_buf = inv_scale.to(dtype=torch.float32, device=flat.device).reshape((1,))
    else:
        inv_scale_buf = torch.tensor([float(inv_scale)], dtype=torch.float32, device=flat.device)

    kernel = _quantize_kernel_for(n_elements, in_dtype, target)

    if out is None:
        out_flat = torch.empty(n_elements, dtype=fp8_dtype, device=flat.device)
        kernel(flat, inv_scale_buf, out_flat)
        return out_flat.reshape(x.shape)

    # Caller-supplied out: TileLang lowering requires contiguous storage for
    # the FP8 write. If `out` is already contiguous, .contiguous() is a no-op
    # view; otherwise we write to a contiguous buffer and copy back so the
    # public contract (`out` is the returned tensor) holds.
    out_flat_view = out.reshape(-1)
    if out_flat_view.is_contiguous():
        kernel(flat, inv_scale_buf, out_flat_view)
        return out
    out_flat = torch.empty(n_elements, dtype=fp8_dtype, device=flat.device)
    kernel(flat, inv_scale_buf, out_flat)
    out_flat_view.copy_(out_flat)
    return out


def fp8_pack_tilelang(tensor: torch.Tensor, *, clamp: bool = False):
    """Drop-in TileLang replacement for ``_triton_fp8_pack``.

    Two kernel launches mirroring the Triton path:
        1. :func:`fp8_amax_tilelang` -- block-reduce + atomic_max.
        2. :func:`fp8_quantize_tilelang` -- scale + clamp + fp8 cast.

    The host-side scale computation between the two launches matches the
    Triton implementation byte-for-byte (including the ``amax_val > 0``
    fallback to scale=1.0).
    """

    import math

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("torch.float8_e4m3fn is not available in this PyTorch build")

    if tensor.numel() == 0:
        fp8_out = torch.empty(tensor.shape, dtype=fp8_dtype, device=tensor.device)
        scale = torch.tensor(1.0, dtype=torch.float32, device=tensor.device)
        return (fp8_out, scale, tensor.dtype)

    if clamp:
        tensor = tensor.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)

    amax_buf = fp8_amax_tilelang(tensor)
    amax_val = amax_buf.item()
    # NaN/Inf in input ``tensor`` propagates through ``T.abs`` -> ``T.reduce_max``
    # -> ``T.atomic_max`` and yields a non-finite ``amax_val``. Falling through
    # to ``inv_scale = fp8_max / amax_val`` would produce 0/NaN and silently
    # poison every output element. Fail loudly so the caller gets a clear
    # diagnostic instead of garbage FP8 weights downstream.
    if not math.isfinite(amax_val):
        raise FloatingPointError(
            f"fp8_pack_tilelang: input contains non-finite values "
            f"(amax={amax_val!r}); refuse to derive a degenerate scale. "
            f"Check the upstream tensor for NaN/Inf before quantization."
        )
    if amax_val > 0:
        scale_val = amax_val / _FP8_E4M3_MAX
        inv_scale_val = _FP8_E4M3_MAX / amax_val
    else:
        scale_val = 1.0
        inv_scale_val = 1.0

    fp8_out = fp8_quantize_tilelang(tensor, inv_scale_val)
    scale = torch.tensor(scale_val, dtype=torch.float32, device=tensor.device)
    return (fp8_out, scale, tensor.dtype)


__all__ = [
    "FP8AmaxPathCStatus",
    "fp8_amax_path_c_status",
    "fp8_amax_tilelang",
    "fp8_pack_tilelang",
    "fp8_quantize_tilelang",
    "make_fp8_amax_kernel",
    "make_fp8_quantize_kernel",
    "precompile_amax_kernel",
    "tilelang_supports",
    "tilelang_supports_with_reason",
]
