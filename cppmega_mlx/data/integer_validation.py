"""Fail-closed host validation for integer-valued MLX side channels."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np


def validated_integer_array(
    value: Any,
    *,
    where: str,
    min_value: int | None = None,
    max_value: int | None = None,
    allow_integral_float: bool = True,
) -> np.ndarray:
    """Return a host array after proving every value is a bounded integer.

    Validation happens before any integer cast, so fractional and non-finite
    floats cannot be truncated and wide integer values cannot wrap.
    """

    if isinstance(value, mx.array):
        array = np.asarray(value)
    elif isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    else:
        raise TypeError(
            f"{where}: expected mx.array/np.ndarray/list/tuple, "
            f"got {type(value).__name__}"
        )

    kind = array.dtype.kind
    if kind in {"i", "u"}:
        pass
    elif kind == "f" and allow_integral_float:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{where}: integer values must be finite")
        if not np.all(array == np.floor(array)):
            raise ValueError(f"{where}: integer values must not be fractional")
    else:
        raise TypeError(
            f"{where}: expected an integer-valued numeric dtype, got {array.dtype}"
        )

    if min_value is not None and np.any(array < min_value):
        bad = array[array < min_value][:8].tolist()
        raise ValueError(
            f"{where}: values must be >= {min_value}; offending values {bad}"
        )
    if max_value is not None and np.any(array > max_value):
        bad = array[array > max_value][:8].tolist()
        raise ValueError(
            f"{where}: values must be <= {max_value}; offending values {bad}"
        )
    return array


def as_int32_array(value: Any, *, where: str) -> mx.array:
    """Convert to MLX int32 only after proving the values fit exactly."""

    array = validated_integer_array(
        value,
        where=where,
        min_value=int(np.iinfo(np.int32).min),
        max_value=int(np.iinfo(np.int32).max),
    )
    return mx.array(array.astype(np.int32, copy=False))


__all__ = ["as_int32_array", "validated_integer_array"]
