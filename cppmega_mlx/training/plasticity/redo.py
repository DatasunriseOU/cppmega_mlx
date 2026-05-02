"""ReDo: Recycling Dormant Neurons for MLP layers.

Port of ``nanochat/nanochat/fire.py::ReDoDiagnostics`` and
``recycle_dormant_neurons``. Tracks per-neuron post-activation magnitude
via an EMA across batches; periodically reinitializes neurons whose mean
absolute activation falls below ``tau`` of the layer mean.

MLX has no torch-style forward hooks. Instead, the diagnostic captures
post-activation tensors via a tiny ``probe`` callback installed on the MLP
module. The model code path threads activations through the probe only
when it is enabled, so steady-state training pays no cost.

GPU-native: all stats / surgery ops run on the default stream (no SVD,
no host sync). The recycle decision uses ``mx.where`` rather than boolean
indexing, mirroring the upstream XLA-safe variant.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import mlx.core as mx
import mlx.nn as nn

DEFAULT_REDO_TAU = 0.025
DEFAULT_EMA_ALPHA = 0.1


def _default_redo_act(x: mx.array) -> mx.array:
    """Primer relu^2 — the activation profile under which dormancy is judged."""

    return nn.relu(x) ** 2


class ReDoDiagnostics:
    """Collect per-neuron post-activation EMA across MLP layers.

    Usage:

        diag = ReDoDiagnostics()
        diag.attach({"block_0_mlp": block.mlp, "block_1_mlp": block_1.mlp})
        # forward passes update diag.stats automatically
        ratios = diag.get_dormant_ratio(tau=0.025)
        n_recycled = recycle_dormant_neurons(layer_map, diag.stats, tau=0.025)
        diag.detach()

    The ``attach`` method installs a thin callable on each MLP module via
    ``module._redo_probe = self._record_factory(name)``. The MLP forward must
    call ``self._redo_probe(post_activation)`` if the attribute is set. This
    keeps the steady-state forward path branch-free and unhindered.
    """

    def __init__(self, ema_alpha: float = DEFAULT_EMA_ALPHA) -> None:
        self.stats: dict[str, mx.array] = {}
        self._ema_alpha = mx.array(ema_alpha, dtype=mx.float32)
        self._attached: dict[str, nn.Module] = {}
        self._act_fn: Callable[[mx.array], mx.array] = _default_redo_act

    def attach(
        self,
        modules: dict[str, nn.Module],
        *,
        act_fn: Callable[[mx.array], mx.array] | None = None,
    ) -> int:
        """Install probes on the given MLP modules.

        Each ``modules[name]`` must be an ``nn.Module`` whose forward path
        already calls ``self._redo_probe(post_activation)`` if present.
        """

        self.detach()
        if act_fn is not None:
            self._act_fn = act_fn

        for name, module in modules.items():
            module._redo_probe = self._record_factory(name)  # type: ignore[attr-defined]
            self._attached[name] = module
        return len(modules)

    def detach(self) -> None:
        """Remove all installed probes."""

        for module in self._attached.values():
            if hasattr(module, "_redo_probe"):
                delattr(module, "_redo_probe")
        self._attached.clear()

    def get_stats(self) -> dict[str, mx.array]:
        return self.stats

    def get_dormant_ratio(self, *, tau: float = DEFAULT_REDO_TAU) -> dict[str, float]:
        """Return per-layer fraction of dormant neurons (sync forces eval)."""

        out: dict[str, float] = {}
        for name, stats in self.stats.items():
            layer_mean = mx.maximum(stats.mean(), mx.array(1e-8, dtype=stats.dtype))
            is_dormant = (stats / layer_mean) < tau
            out[name] = float(is_dormant.sum().item()) / float(stats.size)
        return out

    def _record_factory(self, name: str) -> Callable[[mx.array], None]:
        ema = self._ema_alpha

        def _record(post_activation: mx.array) -> None:
            activated = self._act_fn(post_activation)
            reduce_axes = tuple(range(activated.ndim - 1))
            mean_abs = mx.abs(activated).mean(axis=reduce_axes).astype(mx.float32)
            prev = self.stats.get(name)
            if prev is None:
                self.stats[name] = mean_abs
            else:
                self.stats[name] = (1 - ema) * prev + ema * mean_abs

        return _record


def recycle_dormant_neurons(
    layer_map: dict[str, tuple[nn.Module | tuple[nn.Module, ...], nn.Module]],
    stats: dict[str, mx.array],
    *,
    tau: float = DEFAULT_REDO_TAU,
    rng_key: mx.array | None = None,
) -> int:
    """Reinit neurons whose activation EMA falls below ``tau * layer_mean``.

    For each ``(in_modules, out_module)`` pair:

    * Identifies dormant neurons via ``stats[name] / layer_mean < tau``.
    * Replaces incoming columns of ``out_module.weight`` with weakened
      Gaussian noise (``std * 0.1``).
    * Replaces outgoing rows of each ``in_modules`` weight (and zeroes
      its bias if any) with weakened Gaussian noise.

    Mutates the modules in place. Returns the total dormant neurons
    replaced. ``mx.where`` keeps the surgery branch-free for ``mx.compile``.
    """

    total = 0
    for name, (in_modules, out_module) in layer_map.items():
        if name not in stats:
            continue
        s = stats[name]
        layer_mean = mx.maximum(s.mean(), mx.array(1e-8, dtype=s.dtype))
        is_dormant = (s / layer_mean) < tau  # bool [hidden]
        n_dormant = int(is_dormant.sum().item())
        if n_dormant == 0:
            continue

        # Reinit incoming weights of fc-out (the column is per dormant neuron)
        out_weight = out_module.weight
        std_out = mx.maximum(out_weight.std(), mx.array(1e-8, dtype=out_weight.dtype))
        out_noise = mx.random.normal(out_weight.shape, dtype=out_weight.dtype) * (
            std_out * 0.1
        )
        # mask is broadcast across dim=0 (rows): dormant column index, all rows
        mask_out = mx.broadcast_to(is_dormant[None, :], out_weight.shape)
        out_module.weight = mx.where(mask_out, out_noise, out_weight)

        # Reinit outgoing weights of fc-in (rows correspond to dormant neurons)
        ins: tuple[nn.Module, ...] = (
            in_modules if isinstance(in_modules, tuple) else (in_modules,)
        )
        for in_mod in ins:
            in_weight = in_mod.weight
            std_in = mx.maximum(
                in_weight.std(), mx.array(1e-8, dtype=in_weight.dtype)
            )
            in_noise = mx.random.normal(in_weight.shape, dtype=in_weight.dtype) * (
                std_in * 0.1
            )
            mask_in = mx.broadcast_to(is_dormant[:, None], in_weight.shape)
            in_mod.weight = mx.where(mask_in, in_noise, in_weight)
            in_bias = getattr(in_mod, "bias", None)
            if in_bias is not None:
                in_mod.bias = mx.where(is_dormant, mx.zeros_like(in_bias), in_bias)

        # Reset stats so the recycled neurons are not flagged again next step.
        stats[name] = mx.where(is_dormant, layer_mean, s)
        total += n_dormant

    if total > 0:
        mx.eval(stats)
    return total


__all__ = [
    "DEFAULT_EMA_ALPHA",
    "DEFAULT_REDO_TAU",
    "ReDoDiagnostics",
    "recycle_dormant_neurons",
]
