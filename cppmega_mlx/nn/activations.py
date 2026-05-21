"""Activation registry — GUI-facing dispatch for per-brick activation choice.

The Visual Builder lets users pick an activation per mlp / gated_mlp /
moe brick. This module is the single source of truth for what names
exist, which are gated (require a `gate` companion projection), and
how each name maps to an MLX implementation.

Six entries are registered here (E7-12 minimum); E7-13 will extend
to ten (adding geglu/reglu/mish/xielu).

Backward-compat: the existing :mod:`cppmega_mlx.nn.moe` module declares
its own narrow ``Literal["gelu","relu2","swiglu"]``. We import that
narrow set from there and re-export under the wider name so callers
can opt into the full set incrementally.
"""

from __future__ import annotations

from typing import Final, Literal

import mlx.core as mx
import mlx.nn as nn


ActivationName = Literal[
    "gelu", "relu", "relu2", "sqrelu", "silu", "swiglu",
]
"""Recognised activation names — E7-12 set (6 entries).

Gated entries (require a ``gate`` companion projection): swiglu.
Dense entries (single projection input): gelu, relu, relu2, sqrelu, silu.
"""


IS_GATED: Final[dict[str, bool]] = {
    "gelu":   False,
    "relu":   False,
    "relu2":  False,
    "sqrelu": False,
    "silu":   False,
    "swiglu": True,
}


# Public listing used by GUI dropdowns and validation.
ACTIVATION_NAMES: Final[tuple[str, ...]] = tuple(IS_GATED.keys())


def is_gated(name: str) -> bool:
    """Return True if the activation needs a ``gate`` companion projection."""
    if name not in IS_GATED:
        raise ValueError(
            f"unknown activation {name!r}; choose from {ACTIVATION_NAMES}"
        )
    return IS_GATED[name]


def apply_activation(
    name: str,
    x: mx.array,
    gate: mx.array | None = None,
) -> mx.array:
    """Apply ``name`` to ``x`` (with ``gate`` for gated activations).

    Raises:
      ValueError: unknown name, or gated activation without gate, or
        dense activation called with a gate (caller bug).
    """
    if name not in IS_GATED:
        raise ValueError(
            f"unknown activation {name!r}; choose from {ACTIVATION_NAMES}"
        )
    needs_gate = IS_GATED[name]
    if needs_gate and gate is None:
        raise ValueError(
            f"{name!r} is gated and requires the `gate` argument"
        )
    if not needs_gate and gate is not None:
        raise ValueError(
            f"{name!r} is dense and rejects a `gate` argument"
        )

    if name == "gelu":
        return nn.gelu_approx(x)
    if name == "relu":
        return mx.maximum(x, 0)
    if name == "relu2":
        return mx.square(mx.maximum(x, 0))
    if name == "sqrelu":
        # Equivalent to relu2 in math; named separately so future Metal
        # kernel impl (training-aware) can route here without disturbing
        # callers that already say "relu2".
        return mx.square(mx.maximum(x, 0))
    if name == "silu":
        return nn.silu(x)
    if name == "swiglu":
        assert gate is not None  # narrowed for type-checkers
        return nn.silu(gate) * x

    raise AssertionError(f"unreachable: {name!r}")


__all__ = [
    "ACTIVATION_NAMES",
    "ActivationName",
    "IS_GATED",
    "apply_activation",
    "is_gated",
]
