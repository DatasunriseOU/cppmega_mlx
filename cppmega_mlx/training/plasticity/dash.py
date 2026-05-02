"""DASH: Direction-Aware SHrinking, per-neuron weight tempering.

Port of ``nanochat/nanochat/fire.py::dash_step``. For each row of a 2D
weight matrix, compute cosine similarity with the corresponding row of the
gradient. Rows whose cos-sim exceeds ``alpha`` are shrunk by ``shrink_rate``
(clamped to ``[0.5, 1.0]``). Reduces stability traps without uniformly
damping the spectrum.

In nanochat the call site is ``base_train.py:17342`` — runs every
``--dash_every`` steps BEFORE ``zero_grad``, so the in-flight gradient
tensors are still alive. Skips Muon-managed params because Muon already
removes the parallel component during its update.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import mlx.core as mx
import mlx.nn as nn

DEFAULT_ALPHA = 0.05
DEFAULT_SHRINK_RATE = 0.01
_MIN_SHRINK = 0.5
_MAX_SHRINK = 1.0


def dash_step(
    weight: mx.array,
    grad: mx.array,
    *,
    alpha: float = DEFAULT_ALPHA,
    shrink_rate: float = DEFAULT_SHRINK_RATE,
) -> mx.array:
    """Apply DASH shrinking to a single 2D weight matrix.

    Returns the new weight (the caller is responsible for assigning it back
    to the model). Pure functional: easy to wrap in ``mx.compile``.
    """
    if weight.ndim != 2 or grad.ndim != 2:
        raise ValueError(
            f"dash_step requires 2D arrays, got weight={weight.ndim}D grad={grad.ndim}D"
        )
    if weight.shape != grad.shape:
        raise ValueError(
            f"weight/grad shape mismatch: {weight.shape} vs {grad.shape}"
        )

    eps = mx.array(1e-8, dtype=weight.dtype)
    w_norm = mx.maximum(mx.linalg.norm(weight, axis=1), eps)
    g_norm = mx.maximum(mx.linalg.norm(grad, axis=1), eps)
    cos_sim = (weight * grad).sum(axis=1) / (w_norm * g_norm)
    penalty = mx.maximum(cos_sim - alpha, mx.array(0.0, dtype=weight.dtype))
    shrink = mx.clip(
        mx.array(1.0, dtype=weight.dtype) - shrink_rate * penalty,
        _MIN_SHRINK,
        _MAX_SHRINK,
    )
    return weight * shrink[:, None]


def _flatten_named_params(params: Any, prefix: str = "") -> Iterable[tuple[str, mx.array]]:
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


def _lookup_in_pytree(root: Any, path: list[str]) -> mx.array | None:
    cursor: Any = root
    for piece in path:
        if cursor is None:
            return None
        if piece.isdigit():
            idx = int(piece)
            if not isinstance(cursor, (list, tuple)) or idx >= len(cursor):
                return None
            cursor = cursor[idx]
        elif isinstance(cursor, dict):
            cursor = cursor.get(piece)
        else:
            return None
    return cursor if isinstance(cursor, mx.array) else None


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


def dash_step_tree(
    model: nn.Module,
    grads: Any,
    *,
    alpha: float = DEFAULT_ALPHA,
    shrink_rate: float = DEFAULT_SHRINK_RATE,
    skip_keys: Iterable[str] = (),
) -> set[str]:
    """Apply DASH across every 2D weight in ``model`` for which we have grads.

    Mutates ``model`` in-place. Returns the set of keys touched.
    """
    skip = set(skip_keys)
    updates: dict[str, Any] = {}
    touched: set[str] = set()

    for name, param in _flatten_named_params(model.parameters()):
        if name in skip or param.ndim != 2:
            continue
        grad = _lookup_in_pytree(grads, name.split("."))
        if grad is None or grad.shape != param.shape:
            continue
        new_param = dash_step(
            param, grad, alpha=alpha, shrink_rate=shrink_rate
        )
        _set_nested(updates, name.split("."), new_param)
        touched.add(name)

    if updates:
        model.update(updates)
        mx.eval(model.parameters())
    return touched


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_SHRINK_RATE",
    "dash_step",
    "dash_step_tree",
]
