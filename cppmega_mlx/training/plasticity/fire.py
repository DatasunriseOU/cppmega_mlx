"""FIRE: Newton-Schulz orthogonalization of 2D weight matrices.

Port of ``nanochat/nanochat/fire.py::newton_schulz`` and ``apply_fire``.
Used at phase transitions (default once at step 5000) to project weights
onto the nearest orthogonal matrix, reinjecting plasticity without
destroying learned scale.

GPU-native: spectral norm is estimated via power iteration (avoids MLX's
CPU-only ``linalg.svd``). Newton-Schulz body uses only matmul + add, which
``mx.compile`` happily fuses into Metal kernels.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import mlx.core as mx
import mlx.nn as nn

DEFAULT_NS_ITERS = 15
DEFAULT_POWER_ITERS = 12
DEFAULT_SKIP_KEYWORDS = ("wte", "lm_head", "embedding", "embed_tokens")


def _spectral_norm_power_iter(W: mx.array, iters: int = DEFAULT_POWER_ITERS) -> mx.array:
    """Estimate ``sigma_1(W)`` via power iteration on ``W^T W``.

    Returns a 0-D float32 scalar living on the same stream as ``W``. The
    estimate is a lower bound on the true spectral norm; a few extra iters
    cheaply tighten it. Fully GPU-friendly (no SVD, no host sync).
    """
    d_in = W.shape[1]
    v = mx.random.normal((d_in,), dtype=mx.float32)
    eps = mx.array(1e-12, dtype=mx.float32)
    for _ in range(iters):
        u = W @ v
        u = u / mx.maximum(mx.linalg.norm(u), eps)
        v = W.T @ u
        v_norm = mx.maximum(mx.linalg.norm(v), eps)
        v = v / v_norm
    return v_norm


def newton_schulz(
    W: mx.array,
    iters: int = DEFAULT_NS_ITERS,
    power_iters: int = DEFAULT_POWER_ITERS,
) -> mx.array:
    """Approximate polar decomposition via the Newton-Schulz cubic iteration.

    Projects ``W`` onto the nearest orthogonal matrix and rescales the result
    back to the original Frobenius norm so downstream activations preserve
    their variance. Float32 internally; emits original dtype.
    """
    if W.ndim != 2:
        raise ValueError(f"newton_schulz requires a 2D matrix, got {W.ndim}D")

    orig_dtype = W.dtype
    W_f32 = W.astype(mx.float32)
    orig_fro = mx.linalg.norm(W_f32)

    spectral = mx.maximum(
        _spectral_norm_power_iter(W_f32, iters=power_iters),
        mx.array(1e-8, dtype=mx.float32),
    )
    # Bump by a small factor so all singular values are strictly inside the
    # NS basin (0, sqrt(3)); power iteration may slightly underestimate.
    X = W_f32 / (spectral * 1.05)

    transposed = W.shape[0] < W.shape[1]
    if transposed:
        X = X.T

    a = mx.array(1.5, dtype=mx.float32)
    b = mx.array(-0.5, dtype=mx.float32)
    for _ in range(iters):
        A = X.T @ X
        X = a * X + b * (X @ A)

    if transposed:
        X = X.T

    new_fro = mx.maximum(mx.linalg.norm(X), mx.array(1e-8, dtype=mx.float32))
    X = X * (orig_fro / new_fro)
    return X.astype(orig_dtype)


def _flatten_named_params(
    params: Any, prefix: str = ""
) -> Iterable[tuple[str, mx.array]]:
    if isinstance(params, dict):
        for key, value in params.items():
            full = f"{prefix}.{key}" if prefix else key
            yield from _flatten_named_params(value, full)
    elif isinstance(params, (list, tuple)):
        for idx, value in enumerate(params):
            full = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _flatten_named_params(value, full)
    elif isinstance(params, mx.array):
        yield prefix, params


def apply_fire(
    model: nn.Module,
    *,
    target_keywords: tuple[str, ...] | None = None,
    skip_keywords: tuple[str, ...] = DEFAULT_SKIP_KEYWORDS,
    iters: int = DEFAULT_NS_ITERS,
    power_iters: int = DEFAULT_POWER_ITERS,
) -> set[str]:
    """Orthogonalize all 2D weight matrices in ``model`` in-place.

    Returns the dotted parameter paths modified so the caller can wipe Adam
    moments selectively via ``reset_optimizer_states_for_fired_keys``.
    """
    modified: set[str] = set()
    updates: dict[str, Any] = {}

    for name, param in _flatten_named_params(model.parameters()):
        if param.ndim != 2:
            continue
        if any(skip in name for skip in skip_keywords):
            continue
        if target_keywords is not None and not any(kw in name for kw in target_keywords):
            continue
        new_param = newton_schulz(param, iters=iters, power_iters=power_iters)
        _set_nested(updates, name.split("."), new_param)
        modified.add(name)

    if updates:
        model.update(updates)
        mx.eval(model.parameters())
    return modified


def _set_nested(root: dict[str, Any], path: list[str], value: mx.array) -> None:
    cursor: Any = root
    for piece in path[:-1]:
        next_cursor: Any
        if piece.isdigit():
            idx = int(piece)
            if not isinstance(cursor, list):
                raise TypeError("array path expects list cursor")
            while len(cursor) <= idx:
                cursor.append({})
            next_cursor = cursor[idx]
            if next_cursor is None or not isinstance(next_cursor, (dict, list)):
                next_cursor = {}
                cursor[idx] = next_cursor
        else:
            if not isinstance(cursor, dict):
                raise TypeError("dict path expects dict cursor")
            next_cursor = cursor.get(piece)
            if next_cursor is None or not isinstance(next_cursor, (dict, list)):
                next_cursor = {}
                cursor[piece] = next_cursor
        cursor = next_cursor
    last = path[-1]
    if last.isdigit() and isinstance(cursor, list):
        idx = int(last)
        while len(cursor) <= idx:
            cursor.append(None)
        cursor[idx] = value
    elif isinstance(cursor, dict):
        cursor[last] = value
    else:
        raise TypeError("unable to set leaf in nested update")


def reset_optimizer_states_for_fired_keys(
    optimizer_state: Any,
    fired_keys: Iterable[str],
    state_keys: tuple[str, ...] = ("m", "v"),
    reset_value: Callable[[mx.array], mx.array] | None = None,
) -> int:
    """Wipe specific buffers (``m``, ``v``) for FIRE'd params in MLX optimizer state.

    The MLX optimizer state is a pytree mirroring the model parameter tree.
    For each fired key (e.g. ``layers.0.attn.q_proj.weight``) we zero out
    the matching entries inside ``state_keys`` (Adam-style moment buffers).
    Returns the number of entries reset.
    """
    if reset_value is None:
        reset_value = mx.zeros_like

    fired = set(fired_keys)
    if not fired:
        return 0

    reset_count = 0
    for state_name in state_keys:
        substate = optimizer_state.get(state_name) if isinstance(optimizer_state, dict) else None
        if substate is None:
            continue
        for key, _ in _flatten_named_params(substate):
            if key in fired:
                _overwrite_in_pytree(
                    substate,
                    key.split("."),
                    reset_value,
                )
                reset_count += 1
    return reset_count


def _overwrite_in_pytree(
    root: Any, path: list[str], reset_value: Callable[[mx.array], mx.array]
) -> None:
    cursor: Any = root
    for piece in path[:-1]:
        cursor = cursor[int(piece)] if piece.isdigit() else cursor[piece]
    last = path[-1]
    if last.isdigit():
        idx = int(last)
        cursor[idx] = reset_value(cursor[idx])
    else:
        cursor[last] = reset_value(cursor[last])


__all__ = [
    "DEFAULT_NS_ITERS",
    "DEFAULT_POWER_ITERS",
    "DEFAULT_SKIP_KEYWORDS",
    "apply_fire",
    "newton_schulz",
    "reset_optimizer_states_for_fired_keys",
]
