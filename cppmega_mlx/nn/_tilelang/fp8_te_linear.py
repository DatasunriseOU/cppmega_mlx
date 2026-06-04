"""TE DelayedScaling(E4M3) fp8 GEMM seam for the model's transformer-block Linears.

This wires the **MEASURED** 1.57-1.83x NVIDIA Transformer-Engine
``DelayedScaling(Format.E4M3)`` cuBLASLt fp8 tensor-core MMA (docs/RELAX-GRAPH-VS-
MEGATRON.md §19/§20, R1 ``tensorwise``) into the MLX model's transformer-block GEMMs
(attention q/k/v/out projections, MoE gate/up/down projections) for BOTH the forward
``y = x @ W.T`` and the backward ``dgrad`` + ``wgrad`` — through ONE persistent
``te.Linear`` module per call site.

The model is pure MLX (``mlx.core``/``mlx.nn``); TE is torch-only. The seam is the
proven in-tree pattern (cppmega_mlx/nn/mamba3.py): an ``@mx.custom_function`` whose
forward and ``.vjp`` cross the MLX<->torch CUDA boundary **zero-copy** via the
``_cuda_zerocopy`` DLPack bridge (no host bounce), run TE's fp8 GEMM in torch, and
return the result/cotangents back into MLX autograd.

Design (matches res_r1e2e PATH A):
  * The bf16 ``nn.Linear`` weight remains the master parameter — the optimizer keeps
    updating bf16. fp8 is cast inside TE at the GEMM only (fwd + dgrad/wgrad). MLX
    holds the bf16 activations/weights; fp8 halves only the GEMM operands at call
    time.
  * ``DelayedScaling`` keeps the amax history in the ``te.Linear`` MODULE state, so
    the module MUST be persistent per call site (cached by id+shape+dtype), never
    rebuilt per call — otherwise the delayed-scaling warmup never converges.
  * The weight is refreshed into the cached ``te.Linear`` each forward (cheap copy of
    a bf16 buffer) so optimizer updates to the MLX master weight are reflected.

RULE #1 (NO silent fallback): when the env gate is ON and a GEMM is selected for fp8,
this is the ONE path. Any TE import / bridge / NVRTC failure RAISES with where+what —
it NEVER silently degrades to the bf16 ``nn.Linear``. The gate default is OFF, so the
bf16 path stays byte-identical when fp8 is not requested. A GEMM that cannot run fp8
(e.g. an odd inner dim TE rejects) RAISES; it does not fall back.

Env gates:
  * ``CPPMEGA_FP8_LINEAR``  — master on/off (default OFF). 1/true/yes/on enables.
  * ``CPPMEGA_FP8_LINEAR_MIN_K`` / ``_MIN_N`` / ``_MIN_M`` — minimum GEMM dims for a
    Linear to be ELIGIBLE for fp8. A Linear smaller than the floor is run bf16 (it is
    NOT an fp8-selected GEMM, so this is not a fallback — it is never selected). TE
    fp8 needs the contracting/free dims to be multiples of 16; tiny-hidden MoE experts
    below the floor would not saturate the fp8 tensor cores. Defaults: K>=512, N>=512,
    M>=512 (all divisible-by-16-friendly). Set the floors to 0 to force fp8 on every
    Linear (used by the A/B harness to stress the small-shape case).
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

# RULE #1: the NVRTC loader fix MUST be applied before torch / TE are imported
# anywhere in this process (gb10). ensure_nvrtc_builtins_path() is a safe no-op
# off-gb10 (returns "noop-not-gb10"); on gb10 it re-execs with the corrected
# LD_LIBRARY_PATH so NVRTC builtins resolve when TE later dlopens them.
from cppmega_mlx._gb10_nvrtc_env import ensure_nvrtc_builtins_path

ensure_nvrtc_builtins_path()


FP8_LINEAR_ENV = "CPPMEGA_FP8_LINEAR"
_FP8_LINEAR_TRUE = {"1", "true", "yes", "on"}

_MIN_K_ENV = "CPPMEGA_FP8_LINEAR_MIN_K"
_MIN_N_ENV = "CPPMEGA_FP8_LINEAR_MIN_N"
_MIN_M_ENV = "CPPMEGA_FP8_LINEAR_MIN_M"
_DEFAULT_MIN_K = 512
_DEFAULT_MIN_N = 512
_DEFAULT_MIN_M = 512


def fp8_linear_enabled() -> bool:
    """Return True iff the master fp8-Linear env gate is on (default OFF)."""

    return os.environ.get(FP8_LINEAR_ENV, "").strip().lower() in _FP8_LINEAR_TRUE


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # RULE #1: a malformed gate is a config bug, not silent.
        raise ValueError(
            f"fp8_te_linear: env {name}={raw!r} is not an integer "
            f"(expected a minimum GEMM dimension)."
        ) from exc


def _fp8_shape_eligible(m: int, k: int, n: int) -> bool:
    """True iff (M=m, K=k, N=n) clears the fp8 eligibility floor.

    Not a fallback: a Linear below the floor is simply never SELECTED for fp8 (its
    one path is bf16). TE fp8 also requires K and N to be multiples of 16; a Linear
    that is selected but has an fp8-incompatible inner dim RAISES inside TE (RULE #1),
    it is not silently demoted here.
    """

    return (
        m >= _env_int(_MIN_M_ENV, _DEFAULT_MIN_M)
        and k >= _env_int(_MIN_K_ENV, _DEFAULT_MIN_K)
        and n >= _env_int(_MIN_N_ENV, _DEFAULT_MIN_N)
    )


# ---------------------------------------------------------------------------
# Persistent te.Linear cache (DelayedScaling amax history lives in the module).
# Keyed by the python id() of the owning MLX weight array's underlying buffer
# is NOT stable across MLX evals, so we key by call-site identity (id of the
# owning nn.Linear module) + (K, N, torch dtype). The amax history therefore
# persists for the life of the process per site, which is exactly the
# DelayedScaling contract.
# ---------------------------------------------------------------------------
_TE_LINEAR_CACHE: dict[Any, Any] = {}
_TE_RECIPE_CACHE: dict[str, Any] = {}


def _mlx_dtype_to_torch(dtype, torch_mod):
    if dtype == mx.bfloat16:
        return torch_mod.bfloat16
    if dtype == mx.float16:
        return torch_mod.float16
    if dtype == mx.float32:
        return torch_mod.float32
    raise TypeError(
        f"fp8_te_linear: unsupported MLX dtype {dtype} for the TE fp8 GEMM; "
        f"expected bfloat16/float16/float32 master weights."
    )


def _get_te_recipe():
    key = "delayed_e4m3"
    cached = _TE_RECIPE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import transformer_engine.common.recipe as te_recipe
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what
        raise RuntimeError(
            "fp8_te_linear: cannot import transformer_engine.common.recipe; the "
            "fp8 GEMM path is enabled (CPPMEGA_FP8_LINEAR) but Transformer-Engine "
            f"is unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    recipe = te_recipe.DelayedScaling(fp8_format=te_recipe.Format.E4M3)
    _TE_RECIPE_CACHE[key] = recipe
    return recipe


def _get_te_linear(site_key: Any, k: int, n: int, torch_dtype) -> Any:
    """Return a persistent ``te.Linear(K=k, N=n)`` for this call site.

    Created once per (site_key, k, n, torch_dtype); the DelayedScaling amax history
    accumulates in the module across steps. RAISES on any TE construction failure.
    """

    cache_key = (site_key, k, n, str(torch_dtype))
    cached = _TE_LINEAR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        import transformer_engine.pytorch as te
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what
        raise RuntimeError(
            "fp8_te_linear: cannot import transformer_engine.pytorch; the fp8 GEMM "
            "path is enabled (CPPMEGA_FP8_LINEAR) but Transformer-Engine is "
            f"unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        lin = te.Linear(k, n, bias=False, params_dtype=torch_dtype).cuda()
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what
        raise RuntimeError(
            f"fp8_te_linear: te.Linear(K={k}, N={n}, params_dtype={torch_dtype}) "
            f"construction failed (call site {site_key!r}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    _TE_LINEAR_CACHE[cache_key] = lin
    return lin


def _run_te_fp8_forward(x_t, w_t, lin):
    """Run TE fp8 forward ``y = x @ W.T`` with the cached te.Linear under autocast.

    x_t: (M, K) torch CUDA. w_t: (N, K) torch CUDA (the MLX master weight view).
    Refreshes the te.Linear weight from w_t (so optimizer updates are reflected),
    then calls the module under te.fp8_autocast(DelayedScaling(E4M3)).
    """

    import torch
    import transformer_engine.pytorch as te

    recipe = _get_te_recipe()
    with torch.no_grad():
        if lin.weight.shape != w_t.shape:
            raise RuntimeError(
                f"fp8_te_linear: cached te.Linear weight shape {tuple(lin.weight.shape)} "
                f"!= MLX master weight shape {tuple(w_t.shape)} — call-site/shape "
                f"cache key collision (RULE #1: refusing to GEMM mismatched shapes)."
            )
        lin.weight.copy_(w_t.to(lin.weight.dtype))
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        return lin(x_t)


@mx.custom_function
def fp8_linear(x: mx.array, weight: mx.array) -> mx.array:
    """fp8 ``y = x @ weight.T`` via TE DelayedScaling(E4M3), MLX-autograd-wrapped.

    ``x``: (..., K) MLX. ``weight``: (N, K) MLX (the ``nn.Linear`` master weight).
    Returns (..., N) MLX. Forward only here; the registered ``.vjp`` runs TE's torch
    backward for dgrad+wgrad.

    The call-site identity is carried implicitly: this top-level ``fp8_linear`` uses a
    process-global per-(shape,dtype) te.Linear. For per-site amax isolation the model
    wires ``maybe_fp8_linear_call`` (below) which passes a stable site key. RAISES on
    any TE/bridge failure (RULE #1).
    """

    return _fp8_linear_impl(x, weight, site_key=("global", int(weight.shape[0]), int(weight.shape[1])))


@fp8_linear.vjp
def _fp8_linear_vjp(primals, cotangent, output):
    del output
    x, weight = primals
    return _fp8_linear_bwd(x, weight, cotangent, site_key=("global", int(weight.shape[0]), int(weight.shape[1])))


def _fp8_linear_impl(x: mx.array, weight: mx.array, *, site_key: Any) -> mx.array:
    from cppmega_mlx.nn._tilelang._cuda_zerocopy import (
        mlx_cuda_array_to_torch_tensor,
        torch_cuda_tensor_to_mlx,
    )

    if weight.ndim != 2:
        raise ValueError(
            f"fp8_te_linear: weight must be 2-D (N, K), got shape {tuple(weight.shape)}."
        )
    n_out, k_in = int(weight.shape[0]), int(weight.shape[1])
    if int(x.shape[-1]) != k_in:
        raise ValueError(
            f"fp8_te_linear: x last dim {int(x.shape[-1])} != weight K {k_in} "
            f"(x shape {tuple(x.shape)}, weight shape {tuple(weight.shape)})."
        )

    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what (no degrade)
        raise RuntimeError(
            "fp8_te_linear: the fp8 GEMM was SELECTED (CPPMEGA_FP8_LINEAR on, shape "
            f"M*x.. K={k_in} N={n_out} above floor) but torch is unavailable in this "
            f"process: {type(exc).__name__}: {exc}. RULE #1: a selected fp8 GEMM "
            "RAISES — it does NOT fall back to bf16."
        ) from exc

    out_dtype = x.dtype
    # Flatten leading dims to a 2-D (M, K) GEMM; restore afterwards.
    lead = tuple(int(d) for d in x.shape[:-1])
    x2 = x.reshape((-1, k_in)) if x.ndim != 2 else x

    try:
        x_t = mlx_cuda_array_to_torch_tensor(x2)
        w_t = mlx_cuda_array_to_torch_tensor(weight)
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what
        raise RuntimeError(
            f"fp8_te_linear: MLX->torch zero-copy bridge failed for the fp8 forward "
            f"(site {site_key!r}, x {tuple(x2.shape)}, w {tuple(weight.shape)}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    torch_dtype = _mlx_dtype_to_torch(out_dtype, torch)
    lin = _get_te_linear(site_key, k_in, n_out, torch_dtype)
    with torch.no_grad():
        out_t = _run_te_fp8_forward(x_t.to(torch_dtype), w_t, lin)
    out = torch_cuda_tensor_to_mlx(out_t, out_dtype=out_dtype)
    if lead:
        out = out.reshape((*lead, n_out))
    return out


def _fp8_linear_bwd(
    x: mx.array,
    weight: mx.array,
    cotangent: mx.array,
    *,
    site_key: Any,
) -> tuple[mx.array, mx.array]:
    """TE fp8 backward: returns (dgrad wrt x, wgrad wrt weight) as MLX arrays.

    Re-runs the TE fp8 forward with torch autograd enabled and backpropagates the MLX
    cotangent to recover dgrad (x.grad) and wgrad (weight.grad) — both computed by TE
    in fp8 (the measured fp8 dgrad/wgrad GEMMs). The wgrad lands on the MLX bf16 master
    weight cotangent (the optimizer updates the bf16 master). RAISES on any failure.
    """

    from cppmega_mlx.nn._tilelang._cuda_zerocopy import (
        mlx_cuda_array_to_torch_tensor,
        relieve_bridge_memory_pressure,
        torch_cuda_tensor_to_mlx,
    )

    try:
        import torch
        import transformer_engine.pytorch as te
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what (no degrade)
        raise RuntimeError(
            "fp8_te_linear(bwd): the fp8 dgrad/wgrad GEMM was SELECTED but "
            f"torch/transformer_engine is unavailable: {type(exc).__name__}: {exc}. "
            "RULE #1: a selected fp8 backward RAISES — it does NOT fall back to bf16."
        ) from exc

    n_out, k_in = int(weight.shape[0]), int(weight.shape[1])
    out_dtype = x.dtype
    lead = tuple(int(d) for d in x.shape[:-1])
    x2 = x.reshape((-1, k_in)) if x.ndim != 2 else x
    g2 = cotangent.reshape((-1, n_out)) if cotangent.ndim != 2 else cotangent

    # Materialize the three bridge operands and flush MLX's stream BEFORE trimming.
    # trim_cuda_mempool must run only at a bridge boundary after eval/sync (RISK 3:
    # trimming mid-kernel could reclaim memory live tensors depend on). After this
    # eval, x2/w/g2 are the live set; the cache + idle reserved memory are not.
    mx.eval(x2, weight, g2)
    mx.synchronize()
    # Relieve the dual-pool (MLX + torch/TE) cudaMallocAsync reservation contention
    # that causes the §21 fp8-backward OOM, so the zero-copy crossing FITS. RULE #1:
    # this does not host-copy — it returns idle reserved unified memory and the
    # bridge below still RAISES on a genuinely-uncrossable tensor.
    try:
        relieve_bridge_memory_pressure()
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what
        raise RuntimeError(
            f"fp8_te_linear(bwd): CUDA mempool trim before the fp8-backward bridge "
            f"crossing failed (site {site_key!r}): {type(exc).__name__}: {exc}. The "
            "zero-copy bridge cannot be made to fit; this is a hard memory boundary, "
            "not a fallback point."
        ) from exc

    try:
        x_t = mlx_cuda_array_to_torch_tensor(x2)
        w_t = mlx_cuda_array_to_torch_tensor(weight)
        g_t = mlx_cuda_array_to_torch_tensor(g2)
    except Exception as exc:  # noqa: BLE001 - RULE #1 surface where+what
        raise RuntimeError(
            f"fp8_te_linear: MLX->torch zero-copy bridge failed for the fp8 backward "
            f"(site {site_key!r}): {type(exc).__name__}: {exc}"
        ) from exc

    torch_dtype = _mlx_dtype_to_torch(out_dtype, torch)
    lin = _get_te_linear(site_key, k_in, n_out, torch_dtype)
    recipe = _get_te_recipe()

    # Drive te.Linear's torch autograd: leaf x requires grad, the te.Linear weight
    # is the differentiable parameter. One backward yields dgrad (x.grad) and
    # wgrad (lin.weight.grad), both produced by TE's fp8 GEMMs.
    x_leaf = x_t.to(torch_dtype).detach().requires_grad_(True)
    with torch.no_grad():
        if lin.weight.shape != w_t.shape:
            raise RuntimeError(
                f"fp8_te_linear(bwd): cached te.Linear weight shape "
                f"{tuple(lin.weight.shape)} != MLX master weight {tuple(w_t.shape)}."
            )
        lin.weight.copy_(w_t.to(lin.weight.dtype))
    if lin.weight.grad is not None:
        lin.weight.grad = None
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        y = lin(x_leaf)
    y.backward(g_t.to(y.dtype))

    if x_leaf.grad is None:
        raise RuntimeError(
            "fp8_te_linear(bwd): te.Linear backward produced no dgrad (x.grad is "
            "None) — TE autograd did not run (RULE #1)."
        )
    if lin.weight.grad is None:
        raise RuntimeError(
            "fp8_te_linear(bwd): te.Linear backward produced no wgrad "
            "(weight.grad is None) — TE autograd did not run (RULE #1)."
        )

    dgrad = torch_cuda_tensor_to_mlx(x_leaf.grad, out_dtype=out_dtype)
    wgrad = torch_cuda_tensor_to_mlx(lin.weight.grad, out_dtype=weight.dtype)
    if lead:
        dgrad = dgrad.reshape((*lead, k_in))
    return dgrad, wgrad


# ---------------------------------------------------------------------------
# Call-site helper used by attention.py / moe.py. Keeps the bf16 nn.Linear as
# the master; routes its GEMM through fp8 ONLY when the gate is on AND the shape
# clears the floor. RULE #1: when fp8 IS selected, a failure raises (no bf16
# degrade); when the gate is off or the shape is below floor, the byte-identical
# bf16 nn.Linear path runs (it was never an fp8 GEMM).
# ---------------------------------------------------------------------------
def maybe_fp8_linear_call(linear_module, x: mx.array):
    """Run ``linear_module(x)`` either bf16 (default) or via the TE fp8 GEMM.

    ``linear_module`` is an ``mlx.nn.Linear`` (weight ``(N, K)``, optional bias).
    fp8 is taken iff: the master gate is on, the module has no bias (TE fp8 path here
    is bias-free; a biased Linear is below scope and runs bf16 — never an fp8 GEMM),
    and (M, K, N) clears the eligibility floor. Otherwise the exact bf16
    ``linear_module(x)`` runs (byte-identical to the un-wired model).
    """

    if not fp8_linear_enabled():
        return linear_module(x)

    weight = linear_module["weight"]
    has_bias = "bias" in linear_module
    n_out, k_in = int(weight.shape[0]), int(weight.shape[1])
    m = 1
    for d in x.shape[:-1]:
        m *= int(d)

    # A biased Linear is not in the fp8 GEMM scope (TE fp8 module here is bias-free);
    # it is never SELECTED for fp8, so running it bf16 is its one path, not a fallback.
    if has_bias or not _fp8_shape_eligible(m, k_in, n_out):
        return linear_module(x)

    # SELECTED for fp8 -> the ONE path. Any failure inside raises (RULE #1).
    site_key = (id(linear_module), n_out, k_in)
    return _fp8_linear_custom(x, weight, site_key)


# A per-site custom_function dispatcher: we cannot pass the non-array site_key
# through mx.custom_function's array-only primal interface, so we close over it
# via a tiny per-site cached custom_function wrapper.
_SITE_FN_CACHE: dict[Any, Any] = {}


def _fp8_linear_custom(x: mx.array, weight: mx.array, site_key: Any) -> mx.array:
    fn = _SITE_FN_CACHE.get(site_key)
    if fn is None:
        fn = _make_site_fn(site_key)
        _SITE_FN_CACHE[site_key] = fn
    return fn(x, weight)


def _make_site_fn(site_key: Any):
    @mx.custom_function
    def _site_fp8_linear(x: mx.array, weight: mx.array) -> mx.array:
        return _fp8_linear_impl(x, weight, site_key=site_key)

    @_site_fp8_linear.vjp
    def _site_fp8_linear_vjp(primals, cotangent, output):
        del output
        x, weight = primals
        return _fp8_linear_bwd(x, weight, cotangent, site_key=site_key)

    return _site_fp8_linear


__all__ = [
    "FP8_LINEAR_ENV",
    "fp8_linear",
    "fp8_linear_enabled",
    "maybe_fp8_linear_call",
]
