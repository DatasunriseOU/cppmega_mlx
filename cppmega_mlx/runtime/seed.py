"""Single-process RNG capture and restore helpers.

This module records Python's process-global ``random`` state, NumPy's legacy
process-global ``np.random`` state, and MLX's process-global ``mx.random.state``
when the installed MLX build exposes a compatible state list.

It intentionally does not claim distributed determinism, per-rank seed policy,
or restoration for independently created NumPy ``Generator`` instances.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import random

import mlx.core as mx
import numpy as np

_SNAPSHOT_VERSION = 1
_MLX_DTYPE_BY_NAME = {
    "uint32": mx.uint32,
    "mlx.core.uint32": mx.uint32,
}


def seed_all(seed: int) -> None:
    """Seed Python, NumPy's global RNG, and MLX's global RNG if available."""

    _require_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    seed_fn = getattr(mx.random, "seed", None)
    if callable(seed_fn):
        seed_fn(seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture process-global Python, NumPy, and MLX RNG states.

    The returned payload is JSON-serializable. MLX state is marked unavailable
    instead of raising when the installed MLX API does not expose the expected
    state-list shape.
    """

    return {
        "version": _SNAPSHOT_VERSION,
        "scope": "single_process_local",
        "python_random": _capture_python_random(),
        "numpy_random": _capture_numpy_random(),
        "mlx_random": _capture_mlx_random(),
    }


def restore_rng_state(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Restore a snapshot produced by :func:`capture_rng_state`.

    Python and NumPy state are restored from validated payloads. MLX restore is
    skipped with a status reason when the snapshot or installed API reports that
    MLX state capture is unavailable.
    """

    _require_snapshot_version(snapshot)
    random.setstate(_python_random_state_from_payload(snapshot["python_random"]))
    np.random.set_state(_numpy_random_state_from_payload(snapshot["numpy_random"]))
    mlx_result = _restore_mlx_random(snapshot["mlx_random"])
    return {
        "python_random": "restored",
        "numpy_random": "restored",
        "mlx_random": mlx_result,
    }


def mlx_rng_state_available() -> bool:
    """Return whether the installed MLX build exposes serializable RNG state."""

    payload = _capture_mlx_random()
    return payload.get("available") is True


def _require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _require_snapshot_version(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("version") != _SNAPSHOT_VERSION:
        raise ValueError(
            f"unsupported RNG snapshot version {snapshot.get('version')!r}; "
            f"expected {_SNAPSHOT_VERSION}"
        )
    for key in ("python_random", "numpy_random", "mlx_random"):
        if key not in snapshot:
            raise ValueError(f"RNG snapshot missing {key!r}")


def _capture_python_random() -> dict[str, Any]:
    version, state, gauss_next = random.getstate()
    return {
        "version": version,
        "state": list(state),
        "gauss_next": gauss_next,
    }


def _python_random_state_from_payload(payload: Any) -> tuple[int, tuple[int, ...], float | None]:
    if not isinstance(payload, Mapping):
        raise ValueError("python_random must be an object")
    version = payload.get("version")
    state = payload.get("state")
    gauss_next = payload.get("gauss_next")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("python_random.version must be an integer")
    if not isinstance(state, Sequence) or isinstance(state, (str, bytes)):
        raise ValueError("python_random.state must be a sequence")
    parsed_state = tuple(_require_int(value, name="python_random.state") for value in state)
    if gauss_next is not None and not isinstance(gauss_next, (int, float)):
        raise ValueError("python_random.gauss_next must be numeric or null")
    return (version, parsed_state, None if gauss_next is None else float(gauss_next))


def _capture_numpy_random() -> dict[str, Any]:
    bit_generator, state, pos, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": bit_generator,
        "state": np.asarray(state, dtype=np.uint32).tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _numpy_random_state_from_payload(
    payload: Any,
) -> tuple[str, np.ndarray, int, int, float]:
    if not isinstance(payload, Mapping):
        raise ValueError("numpy_random must be an object")
    bit_generator = payload.get("bit_generator")
    state = payload.get("state")
    pos = payload.get("pos")
    has_gauss = payload.get("has_gauss")
    cached_gaussian = payload.get("cached_gaussian")
    if not isinstance(bit_generator, str):
        raise ValueError("numpy_random.bit_generator must be a string")
    if not isinstance(state, Sequence) or isinstance(state, (str, bytes)):
        raise ValueError("numpy_random.state must be a sequence")
    parsed_state = np.array(
        [_require_int(value, name="numpy_random.state") for value in state],
        dtype=np.uint32,
    )
    parsed_pos = _require_int(pos, name="numpy_random.pos")
    parsed_has_gauss = _require_int(has_gauss, name="numpy_random.has_gauss")
    if not isinstance(cached_gaussian, (int, float)):
        raise ValueError("numpy_random.cached_gaussian must be numeric")
    return (
        bit_generator,
        parsed_state,
        parsed_pos,
        parsed_has_gauss,
        float(cached_gaussian),
    )


def _capture_mlx_random() -> dict[str, Any]:
    state = getattr(mx.random, "state", None)
    if not isinstance(state, list):
        return {
            "available": False,
            "reason": "mx.random.state is not exposed as a mutable list",
        }
    entries: list[dict[str, Any]] = []
    try:
        for item in state:
            arr = np.array(item)
            entries.append(
                {
                    "dtype": _mlx_dtype_name(item),
                    "shape": list(item.shape),
                    "values": arr.tolist(),
                }
            )
    except Exception as exc:  # pragma: no cover - depends on installed MLX build.
        return {
            "available": False,
            "reason": f"mx.random.state could not be serialized: {type(exc).__name__}",
        }
    return {
        "available": True,
        "state": entries,
    }


def _restore_mlx_random(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("mlx_random must be an object")
    if payload.get("available") is not True:
        return {
            "restored": False,
            "reason": str(payload.get("reason", "MLX RNG state unavailable")),
        }

    state = getattr(mx.random, "state", None)
    if not isinstance(state, list):
        return {
            "restored": False,
            "reason": "mx.random.state is not exposed as a mutable list",
        }
    entries = payload.get("state")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("mlx_random.state must be a sequence")

    restored = [_mlx_array_from_payload(entry) for entry in entries]
    try:
        state[:] = restored
        mx.eval(state)
    except Exception as exc:  # pragma: no cover - depends on installed MLX build.
        return {
            "restored": False,
            "reason": f"mx.random.state could not be restored: {type(exc).__name__}",
        }
    return {
        "restored": True,
    }


def _mlx_array_from_payload(payload: Any) -> mx.array:
    if not isinstance(payload, Mapping):
        raise ValueError("mlx_random.state entries must be objects")
    dtype_name = payload.get("dtype")
    shape = payload.get("shape")
    values = payload.get("values")
    if dtype_name not in _MLX_DTYPE_BY_NAME:
        raise ValueError(f"unsupported MLX RNG state dtype {dtype_name!r}")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise ValueError("mlx_random.state.shape must be a sequence")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("mlx_random.state.values must be a sequence")
    parsed_shape = tuple(_require_int(dim, name="mlx_random.state.shape") for dim in shape)
    arr = mx.array(list(values), dtype=_MLX_DTYPE_BY_NAME[dtype_name])
    if tuple(arr.shape) != parsed_shape:
        raise ValueError(
            f"MLX RNG state shape {tuple(arr.shape)!r} does not match {parsed_shape!r}"
        )
    return arr


def _mlx_dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype == mx.uint32:
        return "uint32"
    return str(dtype)


def _require_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must contain integers")
    return value
