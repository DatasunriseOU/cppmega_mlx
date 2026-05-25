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
    "glu", "gelu", "relu", "relu2", "sqrelu", "silu", "mish",
    "swiglu", "geglu", "reglu", "xielu",
]
"""Recognised activation names — E7-12 + E7-13 set + glu (11 entries).

Gated entries (require a ``gate`` companion projection):
  glu, swiglu, geglu, reglu, xielu.
Dense entries (single projection input):
  gelu, relu, relu2, sqrelu, silu, mish.
"""


IS_GATED: Final[dict[str, bool]] = {
    "glu":    True,
    "gelu":   False,
    "relu":   False,
    "relu2":  False,
    "sqrelu": False,
    "silu":   False,
    "mish":   False,
    "swiglu": True,
    "geglu":  True,
    "reglu":  True,
    "xielu":  True,
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

    if name == "glu":
        assert gate is not None
        return mx.sigmoid(gate) * x
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
    if name == "mish":
        # x * tanh(softplus(x)); softplus(z) = log(1 + exp(z))
        # Use log1p for numerical stability.
        return x * mx.tanh(mx.log1p(mx.exp(x)))
    if name == "swiglu":
        assert gate is not None
        return nn.silu(gate) * x
    if name == "geglu":
        assert gate is not None
        return nn.gelu_approx(gate) * x
    if name == "reglu":
        assert gate is not None
        return mx.maximum(gate, 0) * x
    if name == "xielu":
        # Extended xGLU variant: gelu(gate) * silu(x); not a single
        # published paper but appears in some Megatron-style ablations.
        assert gate is not None
        return nn.gelu_approx(gate) * nn.silu(x)

    raise AssertionError(f"unreachable: {name!r}")


__all__ = [
    "ACTIVATION_NAMES",
    "ActivationName",
    "IS_GATED",
    "apply_activation",
    "is_gated",
]
