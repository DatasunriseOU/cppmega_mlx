"""Tiny correctness-first MLX reference for cppmega Mamba3-like blocks."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal, overload

import mlx.core as mx
import mlx.nn as nn


DEFAULT_CHUNK_SIZE = 128
MAMBA3_PATH_C_BWD_ENV = "CPPMEGA_MAMBA3_PATH_C_BWD"
_MAMBA3_PATH_C_PATH_B_BWD_VALUES = {
    "path_b",
    "b",
    "metal",
    "fwd_path_b_bwd",
    "path_c_fwd_path_b_bwd",
}
# Explicit proving-phase selector for the chunked B2->B1->B0 backward behind the
# Path C DSL forward. Selectable ONLY via the env; the receipt keeps AUTO on
# path_b until a no-worse bwd receipt is checked in. RULE #1: when selected this
# is the ONE clear path — its failures RAISE, there is no try/except->MSL.
_MAMBA3_PATH_C_CHUNKED_BWD_VALUES = {
    "path_c_chunked_bwd",
    "path_c_fwd_path_c_bwd",
    "chunked",
    "chunked_bwd",
}


def _mamba3_path_c_uses_path_b_backward() -> bool:
    return (
        os.environ.get(MAMBA3_PATH_C_BWD_ENV, "").strip().lower()
        in _MAMBA3_PATH_C_PATH_B_BWD_VALUES
    )


def _mamba3_path_c_uses_chunked_backward() -> bool:
    return (
        os.environ.get(MAMBA3_PATH_C_BWD_ENV, "").strip().lower()
        in _MAMBA3_PATH_C_CHUNKED_BWD_VALUES
    )


@mx.custom_function
def _mamba3_mimo_apply_with_state(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    """Differentiable Path B Mamba3 forward returning ``(y, h_last)``.

    Wraps :func:`cppmega_mlx.nn._tilelang.mamba3_mimo_fwd_metal` and reuses
    its existing manual VJP (via :func:`mamba3_mimo_bwd_metal`). Gradients
    flow only through ``y``; the cotangent for ``h_last`` is treated as
    zero. This matches the production model contract: the loss path uses
    ``y``, the cache path uses ``h_last`` for inference only.
    """

    from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_fwd_metal

    return mamba3_mimo_fwd_metal(x, B, C, z, A, dt, D, h0)


@_mamba3_mimo_apply_with_state.vjp
def _mamba3_mimo_apply_with_state_vjp(
    primals: tuple[mx.array, ...],
    cotangent: tuple[mx.array, mx.array],
    output: tuple[mx.array, mx.array],
) -> tuple[mx.array, ...]:
    """VJP that ignores the ``h_last`` cotangent (always zero in practice)."""

    from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal

    del output
    x, B, C, z, A, dt, D, h0 = primals
    dy = cotangent[0]
    return mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)


@mx.custom_function
def _mamba3_mimo_apply_with_state_cuda_eager(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    """Differentiable Path B Mamba3 forward on a CUDA host (no Metal).

    Device-aware Path B analog of :func:`_mamba3_mimo_apply_with_state`: when
    the live MLX device is the CUDA gpu (Metal unavailable), the hand-MSL Path
    B Metal kernel cannot run, so the same Mamba3 MIMO recurrence is routed
    through the host ``_cuda_eager`` bridge (``mamba3_mimo_fwd_cuda_eager`` —
    the exact CUDA-safe prim_func Mamba3 Path C uses on CUDA). Gradients flow
    only through ``y`` (the ``h_last`` cotangent is zero in practice), matching
    the Metal Path B contract.

    Kernel unavailability / launch failure RAISES (RULE #1) — never a silent
    fall-through to the Metal kernel (which cannot run on CUDA) or to a
    different path.
    """

    from cppmega_mlx.nn._tilelang._cuda_eager import (
        cuda_eager_available,
        mamba3_mimo_fwd_cuda_eager,
    )

    cuda_ok, cuda_reason = cuda_eager_available()
    if not cuda_ok:
        raise RuntimeError(
            "mamba3_mimo: Path B Metal kernel unavailable on this CUDA host "
            f"and the TileLang-CUDA EAGER forward is also unavailable: {cuda_reason}"
        )
    result = mamba3_mimo_fwd_cuda_eager(x, B, C, z, A, dt, D, h0)
    if result is None:
        raise RuntimeError(
            "mamba3_mimo: Path B TileLang-CUDA EAGER forward returned None "
            "(cuda_eager_available() was True — this is a bug)"
        )
    return result


@_mamba3_mimo_apply_with_state_cuda_eager.vjp
def _mamba3_mimo_apply_with_state_cuda_eager_vjp(
    primals: tuple[mx.array, ...],
    cotangent: tuple[mx.array, mx.array],
    output: tuple[mx.array, mx.array],
) -> tuple[mx.array, ...]:
    """CUDA-eager Path B VJP — ignores the ``h_last`` cotangent (zero)."""

    from cppmega_mlx.nn._tilelang._cuda_eager import (
        cuda_eager_available,
        mamba3_mimo_bwd_cuda_eager,
    )

    del output
    x, B, C, z, A, dt, D, h0 = primals
    dy = cotangent[0]
    cuda_ok, cuda_reason = cuda_eager_available()
    if not cuda_ok:
        raise RuntimeError(
            "mamba3_mimo: Path B Metal backward unavailable on this CUDA host "
            f"and the TileLang-CUDA EAGER backward is also unavailable: {cuda_reason}"
        )
    grads = mamba3_mimo_bwd_cuda_eager(dy, x, B, C, z, A, dt, D, h0)
    if grads is None:
        raise RuntimeError(
            "mamba3_mimo: Path B TileLang-CUDA EAGER backward returned None "
            "(cuda_eager_available() was True — this is a bug)"
        )
    return grads


@dataclass(frozen=True)
class Mamba3Config:
    """Local MLX config using cppmega-facing Author Mamba3 names where possible."""

    d_model: int
    expand: int = 2
    headdim: int = 64
    d_state: int = 16
    ngroups: int = 1
    mimo_rank: int = 1
    is_mimo: bool = False
    d_conv: int = 4
    chunk_size: int = DEFAULT_CHUNK_SIZE
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_floor: float = 0.01
    rope_fraction: float = 0.5

    def __post_init__(self) -> None:
        _require_positive_int("d_model", self.d_model)
        _require_positive_int("expand", self.expand)
        _require_positive_int("headdim", self.headdim)
        _require_positive_int("d_state", self.d_state)
        _require_positive_int("ngroups", self.ngroups)
        _require_positive_int("mimo_rank", self.mimo_rank)
        _require_positive_int("d_conv", self.d_conv)
        _require_positive_int("chunk_size", self.chunk_size)
        _require_positive_float("dt_min", self.dt_min)
        _require_positive_float("dt_max", self.dt_max)
        _require_positive_float("dt_init_floor", self.dt_init_floor)
        _require_positive_float("A_floor", self.A_floor)
        if self.d_inner % self.headdim != 0:
            raise ValueError(
                f"d_model * expand ({self.d_inner}) must be divisible by headdim "
                f"({self.headdim})"
            )
        if self.nheads % self.ngroups != 0:
            raise ValueError(f"ngroups ({self.ngroups}) must divide nheads ({self.nheads})")
        if self.rope_fraction not in (0.5, 1.0):
            raise ValueError("rope_fraction must be 0.5 or 1.0 for source-compatible Mamba3")
        if self.dt_min > self.dt_max:
            raise ValueError("dt_min must be <= dt_max")
        if self.dt_init_floor > self.dt_max:
            raise ValueError("dt_init_floor must be <= dt_max")
        split_tensor_size = int(self.d_state * self.rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        if split_tensor_size <= 0:
            raise ValueError(
                f"num_rope_angles must be positive for d_state={self.d_state}, "
                f"rope_fraction={self.rope_fraction}"
            )

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def nheads(self) -> int:
        if self.d_inner % self.headdim != 0:
            raise ValueError(
                f"d_model * expand ({self.d_inner}) must be divisible by headdim "
                f"({self.headdim})"
            )
        return self.d_inner // self.headdim

    @property
    def effective_mimo_rank(self) -> int:
        return self.mimo_rank if self.is_mimo else 1


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}")


@dataclass(frozen=True)
class Mamba3InProjDims:
    """Dimensions for the Author Mamba3 packed projection layout."""

    d_inner: int
    d_bc: int
    nheads: int
    num_rope_angles: int

    @property
    def total(self) -> int:
        return 2 * self.d_inner + 2 * self.d_bc + 3 * self.nheads + self.num_rope_angles

    @property
    def split_sizes(self) -> list[int]:
        return [
            self.d_inner,
            self.d_inner,
            self.d_bc,
            self.d_bc,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_angles,
        ]


@dataclass(frozen=True)
class Mamba3CacheState:
    """Batch-shaped local mirror of source Mamba3 (angle_dt, ssm, k, v) cache."""

    angle_dt: mx.array
    ssm: mx.array
    k: mx.array
    v: mx.array


def compute_num_rope_angles(d_state: int, rope_fraction: float) -> int:
    split_tensor_size = int(d_state * rope_fraction)
    if split_tensor_size % 2 != 0:
        split_tensor_size -= 1
    num_rope_angles = split_tensor_size // 2
    if num_rope_angles <= 0:
        raise ValueError(
            f"num_rope_angles must be positive for d_state={d_state}, "
            f"rope_fraction={rope_fraction}"
        )
    return num_rope_angles


def compute_mamba3_in_proj_dims(config: Mamba3Config) -> Mamba3InProjDims:
    return Mamba3InProjDims(
        d_inner=config.d_inner,
        d_bc=config.d_state * config.ngroups * config.effective_mimo_rank,
        nheads=config.nheads,
        num_rope_angles=compute_num_rope_angles(config.d_state, config.rope_fraction),
    )


def _rms_norm_last(x: mx.array, eps: float = 1e-5) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)


def _split_by_sizes(x: mx.array, sizes: list[int]) -> tuple[mx.array, ...]:
    indices: list[int] = []
    total = 0
    for size in sizes[:-1]:
        total += size
        indices.append(total)
    return tuple(mx.split(x, indices, axis=-1))


def _broadcast_groups_to_heads(x: mx.array, nheads: int, name: str) -> mx.array:
    groups = x.shape[2]
    if groups <= 0:
        raise ValueError(f"{name} group count must be positive, got {groups}")
    if nheads % groups != 0:
        raise ValueError(f"{name} group count {groups} must divide nheads {nheads}")
    if groups == nheads:
        return x
    return mx.repeat(x, repeats=nheads // groups, axis=2)


def _compute_trapezoidal_scale(dt: mx.array, trap: mx.array) -> mx.array:
    """Author Mamba3 trapezoidal input scale for B/K, shaped (B,S,H)."""

    if dt.ndim != 3:
        raise ValueError(f"dt must be shaped (B,S,H), got {dt.shape}")
    if trap.shape != dt.shape:
        raise ValueError(f"trap must have shape {dt.shape}, got {trap.shape}")

    sig_trap = mx.sigmoid(trap)
    if dt.shape[1] == 0:
        return dt
    dt_shifted = mx.concatenate([dt[:, 1:, :], mx.zeros_like(dt[:, :1, :])], axis=1)
    sig_shifted = mx.concatenate(
        [sig_trap[:, 1:, :], mx.zeros_like(sig_trap[:, :1, :]) + 0.5],
        axis=1,
    )
    return dt_shifted * (1.0 - sig_shifted) + dt * sig_trap


def _heads_to_group_scale(scale: mx.array, ngroups: int) -> mx.array:
    """Average per-head scales down to grouped B/C layout."""

    if scale.ndim != 3:
        raise ValueError(f"scale must be shaped (B,S,H), got {scale.shape}")
    nheads = scale.shape[2]
    if ngroups <= 0:
        raise ValueError(f"ngroups must be positive, got {ngroups}")
    if nheads % ngroups != 0:
        raise ValueError(f"ngroups {ngroups} must divide nheads {nheads}")
    return mx.mean(scale.reshape(scale.shape[0], scale.shape[1], ngroups, nheads // ngroups), axis=-1)


def _apply_rope_on_state_dim(tensor: mx.array, angles_cumsum: mx.array) -> mx.array:
    """Apply source-compatible complex RoPE to B/C over the state dimension."""

    if tensor.ndim != 4:
        raise ValueError(f"tensor must be shaped (B,S,G,N), got {tensor.shape}")
    if angles_cumsum.ndim != 4:
        raise ValueError(
            f"angles_cumsum must be shaped (B,S,H,R), got {angles_cumsum.shape}"
        )
    batch, seq, groups, d_state = tensor.shape
    if angles_cumsum.shape[0] != batch or angles_cumsum.shape[1] != seq:
        raise ValueError(
            f"angles_cumsum leading dims must be {(batch, seq)}, got {angles_cumsum.shape[:2]}"
        )
    if angles_cumsum.shape[2] < groups:
        raise ValueError(
            f"angles_cumsum head count {angles_cumsum.shape[2]} must cover groups {groups}"
        )

    n_rope = min(angles_cumsum.shape[-1], d_state // 2)
    if n_rope <= 0:
        return tensor
    rot_dim = 2 * n_rope
    angles = angles_cumsum[:, :, :groups, :n_rope]
    cos_a = mx.cos(angles)
    sin_a = mx.sin(angles)

    rotated_part = tensor[..., :rot_dim].reshape(batch, seq, groups, n_rope, 2)
    t0 = rotated_part[..., 0]
    t1 = rotated_part[..., 1]
    rotated = mx.stack([t0 * cos_a - t1 * sin_a, t0 * sin_a + t1 * cos_a], axis=-1)
    rotated = rotated.reshape(batch, seq, groups, rot_dim)
    if rot_dim == d_state:
        return rotated
    return mx.concatenate([rotated, tensor[..., rot_dim:]], axis=-1)


def _expand_mimo_rank_to_heads(tensor: mx.array, nheads: int, name: str) -> mx.array:
    """Expand grouped MIMO B/C tensors from (B,S,R,G,N) to (B,S,R,H,N)."""

    if tensor.ndim != 5:
        raise ValueError(f"{name} must be shaped (B,S,R,G,N), got {tensor.shape}")
    groups = tensor.shape[3]
    if groups <= 0:
        raise ValueError(f"{name} group count must be positive, got {groups}")
    if nheads % groups != 0:
        raise ValueError(f"{name} group count {groups} must divide nheads {nheads}")
    if groups == nheads:
        return tensor
    return mx.repeat(tensor, repeats=nheads // groups, axis=3)


def causal_depthwise_conv1d(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    """Causal depthwise Conv1d over MLX NLC tensors."""

    if x.ndim != 3:
        raise ValueError(f"x must be shaped (B,S,C), got {x.shape}")
    channels = x.shape[-1]
    if weight.shape[0] != channels or weight.shape[-1] != 1:
        raise ValueError(f"weight must be shaped (C,K,1) for C={channels}, got {weight.shape}")

    kernel_size = weight.shape[1]
    padded = mx.pad(x, [(0, 0), (kernel_size - 1, 0), (0, 0)])
    y = mx.conv1d(padded, weight, stride=1, padding=0, dilation=1, groups=channels)
    if bias is not None:
        y = y + bias
    return y


def _chunked_mamba3_diagonal_scan(
    log_decay: mx.array,
    inp: mx.array,
    C: mx.array,
    x: mx.array,
    z: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    chunk_size: int,
) -> tuple[mx.array, mx.array]:
    """Chunked diagonal SSM scan for the local Mamba3 reference block.

    The recurrence is h[t] = exp(log_decay[t]) * h[t-1] + inp[t].
    Carries are evaluated in source order to avoid stale lazy graph reuse across
    chunk boundaries on current MLX/Metal.
    """

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if log_decay.ndim != 5:
        raise ValueError(f"log_decay must be shaped (B,S,H,1,1), got {log_decay.shape}")
    if inp.ndim != 5:
        raise ValueError(f"inp must be shaped (B,S,H,P,N), got {inp.shape}")

    batch, seq, nheads, headdim, d_state = inp.shape
    if log_decay.shape != (batch, seq, nheads, 1, 1):
        raise ValueError(
            f"log_decay must have shape {(batch, seq, nheads, 1, 1)}, got {log_decay.shape}"
        )
    if C.shape != (batch, seq, nheads, d_state):
        raise ValueError(f"C must have shape {(batch, seq, nheads, d_state)}, got {C.shape}")
    if x.shape != (batch, seq, nheads, headdim):
        raise ValueError(f"x must have shape {(batch, seq, nheads, headdim)}, got {x.shape}")
    if z.shape != x.shape:
        raise ValueError(f"z must have shape {x.shape}, got {z.shape}")
    if D.shape == (nheads,):
        D_skip = D[:, None]
    elif D.shape == (nheads, headdim):
        D_skip = D
    else:
        raise ValueError(
            f"D must have shape {(nheads,)} or {(nheads, headdim)}, got {D.shape}"
        )
    if h0.shape != (batch, nheads, headdim, d_state):
        raise ValueError(
            f"h0 must have shape {(batch, nheads, headdim, d_state)}, got {h0.shape}"
        )

    if seq == 0:
        return mx.zeros((batch, 0, nheads, headdim), dtype=inp.dtype), h0

    h = h0
    outputs: list[mx.array] = []
    scan_chunk_size = min(chunk_size, 32)
    n_chunks = math.ceil(seq / scan_chunk_size)
    for chunk in range(n_chunks):
        start = chunk * scan_chunk_size
        end = min(start + scan_chunk_size, seq)

        # MLX 0.31 can miscompile/materialize the vectorized cumsum closed form
        # after dtype-changing modules run earlier in-process. Keep chunking for
        # bounded Python work, but compute the recurrence in source order.
        for step in range(start, end):
            h = mx.exp(log_decay[:, step]) * h + inp[:, step]
            y = mx.sum(h * C[:, step, :, None, :], axis=-1)
            y = y + D_skip.astype(y.dtype) * x[:, step]
            outputs.append(nn.silu(z[:, step]) * y)

    return mx.stack(outputs, axis=1), h


def _dispatch_mamba3_scan(
    *,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    chunk_size: int,
) -> tuple[mx.array, mx.array]:
    """Route the Mamba3 selective scan according to :class:`KernelPath`.

    AUTO may promote receipt-covered FP32 shapes to Path C TileLang DSL. The
    receipt gate runs before any TileLang availability probe so AUTO does not
    compile Path C candidate kernels when the profiled decision is Path B.
    REFERENCE always uses the pure-MLX chunked scan. PATH_C routes to the
    lowered TileLang DSL fwd+bwd kernel and fails closed if unavailable.
    """

    from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_metal_status
    from cppmega_mlx.runtime.kernel_policy import (
        KernelPath,
        record_dispatch,
        selected_path,
    )

    path = selected_path("mamba3_mimo")

    if path is KernelPath.PATH_C:
        from cppmega_mlx.nn._tilelang.mamba3_path_c import (
            mamba3_mimo_apply_with_state_path_c,
            mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd,
            mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd,
            mamba3_mimo_path_c_status,
        )

        status = mamba3_mimo_path_c_status()
        if not status.available:
            raise RuntimeError(f"mamba3_mimo: Path C kernel unavailable ({status.reason})")
        if _mamba3_path_c_uses_chunked_backward():
            # EXPLICIT proving-phase selector: Path C DSL fwd + chunked
            # B2->B1->B0 backward. RULE #1: the ONE clear path once selected —
            # the custom_function's .vjp RAISES (no try/except->MSL fallback) on
            # any chunked-chain failure / out-of-regime A. The receipt keeps AUTO
            # on path_b; this branch is reached only when CPPMEGA_MAMBA3_PATH_C_BWD
            # names the chunked mode.
            y, h_last, *_stash = mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(
                x, B, C, z, A, dt, D, h0
            )
            record_dispatch(
                "mamba3_mimo", path, "path_c_tilelang_dsl_fwd_path_c_chunked_bwd"
            )
            return y, h_last
        if _mamba3_path_c_uses_path_b_backward():
            y, h_last = mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd(
                x, B, C, z, A, dt, D, h0
            )
            record_dispatch(
                "mamba3_mimo", path, "path_c_tilelang_dsl_fwd_path_b_bwd"
            )
            return y, h_last
        y, h_last = mamba3_mimo_apply_with_state_path_c(x, B, C, z, A, dt, D, h0)
        record_dispatch("mamba3_mimo", path, "path_c_tilelang_dsl")
        return y, h_last

    if path is KernelPath.REFERENCE:
        record_dispatch("mamba3_mimo", path, "reference_pure_mlx")
        return _reference_scan(
            x=x, B=B, C=C, z=z, A=A, dt=dt, D=D, h0=h0, chunk_size=chunk_size,
        )

    # AUTO + PATH_B share the Metal-availability check.
    status = mamba3_mimo_metal_status(x)
    if path is KernelPath.PATH_B and not status.available:
        # Device-aware Path B: on a CUDA host (Metal unavailable, default
        # device is the CUDA gpu) the hand-MSL Path B Metal kernel cannot run.
        # Route the same Mamba3 MIMO recurrence through the host _cuda_eager
        # bridge (the exact CUDA-safe prim_func Path C uses on CUDA). Mirrors
        # the GDN/KDA Path B device switch. Unavailability/failure RAISES
        # (RULE #1) — never a silent fall-through.
        from cppmega_mlx.kernels.metal_ops import can_run_metal

        if not can_run_metal():
            from cppmega_mlx.nn._tilelang._cuda_eager import cuda_eager_available

            cuda_ok, cuda_reason = cuda_eager_available()
            if not cuda_ok:
                raise RuntimeError(
                    f"mamba3_mimo: Path B kernel unavailable ({status.reason}) "
                    f"and the TileLang-CUDA EAGER path is also unavailable: "
                    f"{cuda_reason}. Use path_c on CUDA."
                )
            y, h_last = _mamba3_mimo_apply_with_state_cuda_eager(
                x, B, C, z, A, dt, D, h0
            )
            record_dispatch("mamba3_mimo", path, "cuda_eager_fwd_v1")
            return y, h_last
        raise RuntimeError(
            f"mamba3_mimo: Path B kernel unavailable ({status.reason})"
        )
    if path is KernelPath.AUTO and status.available:
        from cppmega_mlx.nn._tilelang.mamba3_path_c import (
            mamba3_path_c_auto_mode_for_inputs,
        )

        auto_mode = mamba3_path_c_auto_mode_for_inputs(x, B, C, z, A, dt, D, h0)
        if auto_mode in {"path_c_fwd_bwd", "path_c_fwd_path_b_bwd"}:
            from cppmega_mlx.nn._tilelang.mamba3_path_c import (
                mamba3_mimo_apply_with_state_path_c,
                mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd,
                mamba3_mimo_path_c_status,
            )

            path_c_status = mamba3_mimo_path_c_status()
            if not path_c_status.available:
                record_dispatch("mamba3_mimo", path, "metal_kernel_fwd_v1")
                y, h_last = _mamba3_mimo_apply_with_state(x, B, C, z, A, dt, D, h0)
                return y, h_last
            try:
                if auto_mode == "path_c_fwd_bwd":
                    y, h_last = mamba3_mimo_apply_with_state_path_c(
                        x, B, C, z, A, dt, D, h0
                    )
                else:
                    y, h_last = mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd(
                        x, B, C, z, A, dt, D, h0
                    )
            except RuntimeError as exc:
                # RULE #1 (no automated/silent fallbacks): a RUNTIME crash in
                # the selected Mamba3 Path C kernel is a bug in that clear
                # path. Silently dropping to Path B / the pure-MLX reference
                # would return a degraded result and hide the bug. RAISE so
                # the graph/DLPack boundary failure surfaces and we fix the
                # root cause. (An UNAVAILABLE Path C is handled before this
                # try-block by the status check, not here.)
                raise RuntimeError(
                    "mamba3_mimo dispatch: selected Path C "
                    f"({auto_mode!r}) crashed at runtime "
                    f"({type(exc).__name__}: {exc}). Refusing to silently fall "
                    "back to Path B / pure-MLX reference (RULE #1) — this "
                    "points at a real bug in the Path C graph/DLPack boundary."
                ) from exc
            else:
                record_dispatch(
                    "mamba3_mimo",
                    path,
                    (
                        "path_c_tilelang_dsl"
                        if auto_mode == "path_c_fwd_bwd"
                        else "path_c_tilelang_dsl_fwd_path_b_bwd"
                    ),
                )
                return y, h_last
    if status.available:
        record_dispatch("mamba3_mimo", path, "metal_kernel_fwd_v1")
        y, h_last = _mamba3_mimo_apply_with_state(x, B, C, z, A, dt, D, h0)
        return y, h_last

    # AUTO fallback: pure-MLX reference.
    record_dispatch("mamba3_mimo", path, "reference_pure_mlx")
    return _reference_scan(
        x=x, B=B, C=C, z=z, A=A, dt=dt, D=D, h0=h0, chunk_size=chunk_size,
    )


def _reference_scan(
    *,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    chunk_size: int,
) -> tuple[mx.array, mx.array]:
    """Adapter that builds (log_decay, inp) and runs the chunked diagonal scan."""

    log_decay = (A * dt)[:, :, :, None, None]
    inp = x[:, :, :, :, None] * B[:, :, :, None, :]
    return _chunked_mamba3_diagonal_scan(
        log_decay,
        inp,
        C,
        x,
        z,
        D,
        h0,
        chunk_size=chunk_size,
    )


class Mamba3ReferenceBlock(nn.Module):
    """Minimal trainable Mamba3-like MLX block.

    This preserves the cppmega packed projection contract and causal state-space
    flavor, but intentionally avoids Megatron, Transformer Engine, CUDA, and
    custom Metal kernels.
    """

    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        self.dims = compute_mamba3_in_proj_dims(config)

        self.in_proj = nn.Linear(config.d_model, self.dims.total, bias=False)
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)
        conv_channels = config.d_inner + 2 * self.dims.d_bc
        self.conv_weight = self._init_conv_weight(conv_channels, config.d_conv)
        self.conv_bias = mx.zeros((conv_channels,))
        self.dt_bias = self._init_dt_bias(config)
        bc_shape = (config.effective_mimo_rank, config.ngroups, config.d_state)
        self.B_norm_weight = mx.ones(bc_shape)
        self.C_norm_weight = mx.ones(bc_shape)
        self.B_bias = mx.zeros(bc_shape)
        self.C_bias = mx.zeros(bc_shape)
        self.D = mx.ones((config.nheads,))

    @staticmethod
    def _init_conv_weight(channels: int, kernel_size: int) -> mx.array:
        if kernel_size <= 0:
            raise ValueError(f"d_conv must be positive, got {kernel_size}")
        scale = math.sqrt(1 / (channels * kernel_size))
        return mx.random.uniform(-scale, scale, (channels, kernel_size, 1))

    @staticmethod
    def _init_dt_bias(config: Mamba3Config) -> mx.array:
        dt_min = max(config.dt_min, config.dt_init_floor)
        if not (0 < dt_min <= config.dt_max):
            raise ValueError(
                f"expected 0 < max(dt_min, dt_init_floor) <= dt_max, got "
                f"{dt_min} and {config.dt_max}"
            )
        dt = mx.exp(mx.random.uniform(math.log(dt_min), math.log(config.dt_max), (config.nheads,)))
        return dt + mx.log(-mx.expm1(-dt))

    def split_in_proj(
        self, projected: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        if projected.shape[-1] != self.dims.total:
            raise ValueError(f"projected last dim must be {self.dims.total}, got {projected.shape}")
        return _split_by_sizes(projected, self.dims.split_sizes)  # type: ignore[return-value]

    def transform_bc(self, B: mx.array, C: mx.array) -> tuple[mx.array, mx.array]:
        """Apply the source Mamba3 B/C QK-norm contract."""

        B = _rms_norm_last(B) * self.B_norm_weight.astype(B.dtype) + self.B_bias.astype(B.dtype)
        C = _rms_norm_last(C) * self.C_norm_weight.astype(C.dtype) + self.C_bias.astype(C.dtype)
        return B, C

    def mamba_state_shapes_per_request(self) -> tuple[tuple[int, ...], ...]:
        """Return local inference-state shapes matching cppmega's Mamba3 cache contract."""

        cfg = self.config
        angle_shape = (cfg.nheads, self.dims.num_rope_angles)
        ssm_shape = (cfg.nheads, cfg.headdim, cfg.d_state)
        k_shape = (cfg.effective_mimo_rank, cfg.nheads, cfg.d_state)
        v_shape = (cfg.nheads, cfg.headdim)
        return (angle_shape, ssm_shape, k_shape, v_shape)

    def initial_h0(self, batch: int, dtype: mx.Dtype) -> mx.array:
        """Return the zero scan state used when no caller h0 is supplied."""

        _require_positive_int("batch", batch)
        cfg = self.config
        return mx.zeros(
            (batch, cfg.nheads, cfg.headdim, cfg.d_state),
            dtype=dtype,
        )

    def zero_cache_state(
        self,
        batch: int,
        *,
        dtype: mx.Dtype = mx.float32,
    ) -> Mamba3CacheState:
        """Return a batch-shaped zero cache matching the source per-request contract."""

        _require_positive_int("batch", batch)
        angle_shape, ssm_shape, k_shape, v_shape = self.mamba_state_shapes_per_request()
        return Mamba3CacheState(
            angle_dt=mx.zeros((batch, *angle_shape), dtype=dtype),
            ssm=mx.zeros((batch, *ssm_shape), dtype=dtype),
            k=mx.zeros((batch, *k_shape), dtype=dtype),
            v=mx.zeros((batch, *v_shape), dtype=dtype),
        )

    def _validate_cache_state(
        self,
        cache: Mamba3CacheState,
        *,
        batch: int,
        dtype: mx.Dtype,
    ) -> None:
        angle_shape, ssm_shape, k_shape, v_shape = self.mamba_state_shapes_per_request()
        expected_shapes = {
            "angle_dt": (batch, *angle_shape),
            "ssm": (batch, *ssm_shape),
            "k": (batch, *k_shape),
            "v": (batch, *v_shape),
        }
        values = {
            "angle_dt": cache.angle_dt,
            "ssm": cache.ssm,
            "k": cache.k,
            "v": cache.v,
        }
        for name, value in values.items():
            expected = expected_shapes[name]
            if value.shape != expected:
                raise ValueError(f"cache.{name} must have shape {expected}, got {value.shape}")
            if value.dtype != dtype:
                raise ValueError(f"cache.{name} must have dtype {dtype}, got {value.dtype}")

    @overload
    def __call__(
        self,
        hidden_states: mx.array,
        *,
        h0: mx.array | None = None,
        cache: Mamba3CacheState | None = None,
        return_cache: Literal[False] = False,
    ) -> tuple[mx.array, mx.array]: ...

    @overload
    def __call__(
        self,
        hidden_states: mx.array,
        *,
        h0: mx.array | None = None,
        cache: Mamba3CacheState | None = None,
        return_cache: Literal[True],
    ) -> tuple[mx.array, Mamba3CacheState]: ...

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        h0: mx.array | None = None,
        cache: Mamba3CacheState | None = None,
        return_cache: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, Mamba3CacheState]:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be shaped (B,S,D), got {hidden_states.shape}")
        if h0 is not None and cache is not None:
            raise ValueError("pass either h0 or cache, not both")

        cfg = self.config
        batch, seq, _ = hidden_states.shape
        if cache is not None:
            self._validate_cache_state(cache, batch=batch, dtype=hidden_states.dtype)

        z, x, B, C, dd_dt, dd_A, trap, angles = self.split_in_proj(self.in_proj(hidden_states))

        xBC = mx.concatenate([x, B, C], axis=-1)
        xBC = causal_depthwise_conv1d(
            xBC,
            self.conv_weight.astype(xBC.dtype),
            self.conv_bias.astype(xBC.dtype),
        )
        x, B, C = _split_by_sizes(nn.silu(xBC), [cfg.d_inner, self.dims.d_bc, self.dims.d_bc])
        x = x.reshape(batch, seq, cfg.nheads, cfg.headdim)
        z = z.reshape(batch, seq, cfg.nheads, cfg.headdim)

        B_mimo = B.reshape(batch, seq, cfg.effective_mimo_rank, cfg.ngroups, cfg.d_state)
        C_mimo = C.reshape(batch, seq, cfg.effective_mimo_rank, cfg.ngroups, cfg.d_state)
        B_mimo, C_mimo = self.transform_bc(B_mimo, C_mimo)
        B = mx.mean(B_mimo, axis=2)
        C = mx.mean(C_mimo, axis=2)

        dt = nn.softplus(dd_dt + self.dt_bias.astype(dd_dt.dtype))
        trap_scale = _compute_trapezoidal_scale(dt, trap)
        B = B * _heads_to_group_scale(trap_scale, cfg.ngroups)[:, :, :, None]

        angles = mx.broadcast_to(
            angles[:, :, None, :],
            (batch, seq, cfg.nheads, self.dims.num_rope_angles),
        )
        angles_cumsum = mx.cumsum(angles * dt[:, :, :, None], axis=1)
        if cache is not None:
            angles_cumsum = angles_cumsum + cache.angle_dt[:, None, :, :].astype(angles_cumsum.dtype)
        B = _apply_rope_on_state_dim(B, angles_cumsum)
        C = _apply_rope_on_state_dim(C, angles_cumsum)

        B_mimo_heads = _expand_mimo_rank_to_heads(B_mimo, cfg.nheads, "B_mimo")
        k_cache_source = _apply_rope_on_state_dim(
            B_mimo_heads.reshape(
                batch,
                seq * cfg.effective_mimo_rank,
                cfg.nheads,
                cfg.d_state,
            ),
            mx.repeat(angles_cumsum, repeats=cfg.effective_mimo_rank, axis=1),
        ).reshape(
            batch,
            seq,
            cfg.effective_mimo_rank,
            cfg.nheads,
            cfg.d_state,
        )
        B = _broadcast_groups_to_heads(B, cfg.nheads, "B")
        C = _broadcast_groups_to_heads(C, cfg.nheads, "C")

        A = mx.minimum(-nn.softplus(dd_A), -cfg.A_floor)

        if cache is not None:
            h = cache.ssm
        elif h0 is None:
            h = self.initial_h0(batch, hidden_states.dtype)
        else:
            expected = (batch, cfg.nheads, cfg.headdim, cfg.d_state)
            if h0.shape != expected:
                raise ValueError(f"h0 must have shape {expected}, got {h0.shape}")
            if h0.dtype != hidden_states.dtype:
                raise TypeError(
                    f"h0 dtype {h0.dtype} must match hidden_states dtype {hidden_states.dtype}"
                )
            h = h0

        y, h = _dispatch_mamba3_scan(
            x=x,
            B=B,
            C=C,
            z=z,
            A=A,
            dt=dt,
            D=self.D,
            h0=h,
            chunk_size=cfg.chunk_size,
        )
        y = y.reshape(batch, seq, cfg.d_inner)
        out = self.out_proj(y)
        if not return_cache:
            return out, h

        if seq == 0:
            if cache is not None:
                next_cache = cache
            else:
                next_cache = self.zero_cache_state(batch, dtype=hidden_states.dtype)
                if h0 is not None:
                    next_cache = Mamba3CacheState(
                        angle_dt=next_cache.angle_dt,
                        ssm=h,
                        k=next_cache.k,
                        v=next_cache.v,
                    )
        else:
            next_cache = Mamba3CacheState(
                angle_dt=mx.remainder(angles_cumsum[:, -1], 2.0 * math.pi),
                ssm=h,
                k=k_cache_source[:, -1],
                v=x[:, -1],
            )
        return out, next_cache


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "MAMBA3_PATH_C_BWD_ENV",
    "Mamba3CacheState",
    "Mamba3Config",
    "Mamba3InProjDims",
    "Mamba3ReferenceBlock",
    "_apply_rope_on_state_dim",
    "causal_depthwise_conv1d",
    "_compute_trapezoidal_scale",
    "compute_mamba3_in_proj_dims",
    "compute_num_rope_angles",
]
