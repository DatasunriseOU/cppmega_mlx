"""Repo-local optimizer helpers for MLX training."""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_merge, tree_unflatten

from cppmega_mlx.training._quantize_8bit import (
    DEFAULT_BLOCK_SIZE as MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE,
    QUANT_SCHEME_DYNAMIC,
    QUANT_SCHEME_SYMMETRIC,
    QUANT_SCHEMES,
    dequantize_blockwise,
    dequantize_dynamic_blockwise,
    num_blocks as _quant_num_blocks,
    quantize_blockwise,
    quantize_dynamic_blockwise,
)
from cppmega_mlx.training.optimizers_quantized import (
    ADAM8BIT_CLASS,
    ADAM8BIT_QUANT_KIND,
    ADAM8BIT_QUANT_SCHEMES,
    ADAM8BIT_SOURCE,
    LION8BIT_CLASS,
    LION8BIT_QUANT_KIND,
    LION8BIT_SOURCE,
    Adam8bit,
    Adam8bitQuantScheme,
    Lion8bit,
    Lion8bitQuantScheme,
    make_adam8bit,
    make_lion8bit,
)


ADAMW_FP32_MOMENTS_CLASS = "cppmega_mlx.training.optimizers.AdamWFP32Moments"
ADAMW_FP32_MOMENTS_SOURCE = "cppmega_mlx.training.optimizers.make_adamw"
ADAMW_BASE_CLASS = "mlx.optimizers.AdamW"
ADAMW_MOMENT_STATE_KEYS = ("m", "v")

MUON_SCALAR_OPTIMIZERS = ("adamw", "adam8bit", "lion", "lion8bit")
MuonScalarOptimizer = Literal["adamw", "adam8bit", "lion", "lion8bit"]

MUON_BASE_CLASS = "mlx.optimizers.Muon"
MUON_ADAMW_MULTI_CLASS = "cppmega_mlx.training.optimizers.MuonAdamWMulti"
MUON_ADAMW_MULTI_SOURCE = "cppmega_mlx.training.optimizers.make_muon"
MUON_NS_CARRIER_ENV = "CPPMEGA_MUON_NS_CARRIER"
MUON_NS_CARRIERS = ("fp32", "bf16")
MuonNSCarrier = Literal["fp32", "bf16"]

MUON_QUANTIZED_MOMENTUM_SCHEME = "symmetric_int8_v1"
"""Default codec identifier for the Muon momentum mapping.

Mirrors cppmega CUDA's
``quantized_muon_momentum_update_multi_and_normalize_groups_`` per-256-block
absmax layout in ``megatron/core/optimizer/emerging_optimizers.py``. The
default :class:`QuantizedMuonWithNSCarrier` keeps this symmetric int8 path
for backwards compatibility; passing ``quant_scheme="dynamic_int8_v1"``
swaps in the bitsandbytes-style dynamic LUT codec on the same uint8 +
fp32-absmax layout.
"""

MUON_QUANTIZED_MOMENTUM_SCHEMES: tuple[str, ...] = QUANT_SCHEMES
"""All accepted ``quant_scheme`` strings for
:class:`QuantizedMuonWithNSCarrier` and :func:`make_muon`. Re-exported from
``_quantize_8bit`` so callers have a single import."""

MuonQuantScheme = Literal["symmetric_int8_v1", "dynamic_int8_v1"]
"""Type alias for the accepted ``quant_scheme`` strings on the Muon
momentum buffer."""

EMBEDDING_LIKE_NAME_HINTS = ("embed", "embedding", "lm_head", "wte", "wpe")
MAMBA_SCALAR_LEAVES = frozenset(
    {
        "A_log",
        "dt_bias",
        "D",
        "B_bias",
        "C_bias",
        "B_norm_weight",
        "C_norm_weight",
        "mimo_x",
        "mimo_z",
        "mimo_o",
    }
)


class AdamWFP32Moments(optim.AdamW):
    """AdamW that keeps moment state in fp32 while preserving parameter dtype."""

    def init_single(self, parameter: mx.array, state: dict[str, Any]) -> None:
        state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)
        state["v"] = mx.zeros(parameter.shape, dtype=mx.float32)

    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: dict[str, Any],
    ) -> mx.array:
        lr = self.learning_rate.astype(mx.float32)
        decayed_parameter = parameter.astype(mx.float32) * (1 - lr * self.weight_decay)
        updated = optim.Adam.apply_single(
            self,
            gradient.astype(mx.float32),
            decayed_parameter,
            state,
        )
        return updated.astype(parameter.dtype)


def make_adamw(
    *,
    learning_rate: float | Callable[[mx.array], mx.array] = 1e-3,
    weight_decay: float = 0.01,
    betas: list[float] | None = None,
    eps: float = 1e-8,
    bias_correction: bool = False,
) -> AdamWFP32Moments:
    """Construct the repo default AdamW with fp32 moments for bf16 training."""

    return AdamWFP32Moments(
        learning_rate=learning_rate,
        betas=[0.9, 0.999] if betas is None else betas,
        eps=eps,
        weight_decay=weight_decay,
        bias_correction=bias_correction,
    )


class LionFP32Moments(optim.Lion):
    """Lion with fp32 momentum state for bf16-weight training.

    Lion only carries one momentum buffer per parameter (vs Adam's two), so
    optimizer state cost is 0.5x AdamW. Param updates are sign-based and
    quantization-friendly.
    """

    def init_single(self, parameter: mx.array, state: dict[str, Any]) -> None:
        state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)

    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: dict[str, Any],
    ) -> mx.array:
        lr = self.learning_rate.astype(mx.float32)
        decayed_parameter = parameter.astype(mx.float32) * (1 - lr * self.weight_decay)
        updated = optim.Lion.apply_single(
            self,
            gradient.astype(mx.float32),
            decayed_parameter,
            state,
        )
        return updated.astype(parameter.dtype)


def make_lion(
    *,
    learning_rate: float | Callable[[mx.array], mx.array] = 1e-4,
    weight_decay: float = 0.01,
    betas: list[float] | None = None,
) -> LionFP32Moments:
    """Construct the repo default Lion with fp32 moment for bf16 training.

    Default LR is 3-10x smaller than AdamW because Lion's sign updates do not
    rescale by gradient magnitude; matches Chen et al. 2302.06675 guidance.
    """

    return LionFP32Moments(
        learning_rate=learning_rate,
        betas=[0.9, 0.99] if betas is None else betas,
        weight_decay=weight_decay,
    )


def is_muon_compatible(name: str, param: mx.array) -> bool:
    """Mirror of Megatron's ``_is_nonlinear_or_embedding`` predicate, inverted so
    that True means the parameter belongs to the Muon group.

    The Muon group keeps only 2-D weight matrices that are not embedding tables,
    not LM head projections, and not Mamba scalar leaves (``A_log``, ``D``,
    ``dt_bias`` and friends). Everything else (1-D vectors, 3-D+ tensors,
    embeddings, lm_head, Mamba scalars, RMSNorm weights) is routed to the AdamW
    fallback group. This matches ``cppmega.cuda``'s
    ``_is_nonlinear_or_embedding`` Megatron emerging_optimizers gate.
    """

    if param.ndim != 2:
        return False
    leaf = name.rsplit(".", 1)[-1]
    if leaf in MAMBA_SCALAR_LEAVES:
        return False
    name_lower = name.lower()
    if any(hint in name_lower for hint in EMBEDDING_LIKE_NAME_HINTS):
        return False
    return True


def split_param_groups(
    model_params: Any,
    *,
    predicate: Callable[[str, mx.array], bool] = is_muon_compatible,
) -> tuple[Any, Any]:
    """Walk a parameter pytree and partition it into ``(muon, other)`` pytrees.

    Each returned pytree contains only the leaves that match its group; the
    other group's leaves are simply absent (no zero-sized placeholders, which
    keeps the underlying optimizers from initialising state for the wrong
    parameters). The two pytrees together cover every leaf in the input
    pytree exactly once.
    """

    flat = tree_flatten(model_params)
    muon_pairs: list[tuple[str, Any]] = []
    other_pairs: list[tuple[str, Any]] = []
    for name, value in flat:
        if isinstance(value, mx.array) and predicate(name, value):
            muon_pairs.append((name, value))
        else:
            other_pairs.append((name, value))
    muon_tree = tree_unflatten(muon_pairs) if muon_pairs else {}
    other_tree = tree_unflatten(other_pairs) if other_pairs else {}
    return muon_tree, other_tree


def _muon_supported_kwargs() -> set[str]:
    """Return the keyword argument names accepted by ``mlx.optimizers.Muon``.

    Used so we can adapt to whatever subset of knobs the installed MLX exposes
    instead of fail-closing if a kwarg is missing.
    """

    try:
        signature = inspect.signature(optim.Muon.__init__)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return set()
    params = signature.parameters
    return {name for name in params if name != "self"}


def _normalize_muon_ns_carrier(ns_carrier: str) -> MuonNSCarrier:
    normalized = ns_carrier.strip().lower()
    if normalized not in MUON_NS_CARRIERS:
        allowed = ", ".join(MUON_NS_CARRIERS)
        raise ValueError(f"ns_carrier must be one of {allowed}; got {ns_carrier!r}")
    return normalized  # type: ignore[return-value]


def _muon_ns_carrier_dtype(ns_carrier: MuonNSCarrier) -> mx.Dtype:
    return mx.bfloat16 if ns_carrier == "bf16" else mx.float32


def _muon_state_dtype() -> mx.Dtype:
    return mx.float32


def _muon_zeropower_newtonschulz5(
    update: mx.array,
    *,
    steps: int,
    ns_carrier: MuonNSCarrier,
    output_dtype: mx.Dtype,
) -> mx.array:
    """Newton-Schulz matrix-sign iteration with an explicit carrier dtype.

    This mirrors MLX's public Muon algorithm but keeps the carrier policy
    repo-local so we do not depend on MLX's private underscore helper.
    """

    if update.ndim != 2:
        raise ValueError(
            "Expected a 2D array for Newton-Schulz iteration, "
            f"got shape {update.shape} instead."
        )
    a, b, c = (3.4445, -4.7750, 2.0315)
    transpose_needed = update.shape[-2] > update.shape[-1]

    x = update.astype(mx.float32)
    if transpose_needed:
        x = x.T

    x = x / (mx.linalg.norm(x, keepdims=True) + 1e-7)
    carrier_dtype = _muon_ns_carrier_dtype(ns_carrier)
    x = x.astype(carrier_dtype)

    for _ in range(steps):
        gram = x @ x.T
        basis = mx.addmm(b * gram, gram, gram, beta=1.0, alpha=c)
        x = mx.addmm(a * x, basis, x, beta=1.0, alpha=1.0).astype(carrier_dtype)

    if transpose_needed:
        x = x.T
    return x.astype(output_dtype)


class MuonWithNSCarrier(optim.Muon):
    """MLX Muon with a configurable Newton-Schulz carrier dtype.

    ``ns_carrier="fp32"`` keeps the momentum/update path in fp32 and uses fp32
    NS iterations. ``ns_carrier="bf16"`` mirrors cppmega's GB10 knob by running
    only the NS polynomial on bf16 carrier tensors; the optimizer momentum and
    update state remain fp32.
    """

    def __init__(
        self,
        learning_rate: float | Callable[[mx.array], mx.array],
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
        *,
        ns_carrier: str = "fp32",
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        self.ns_carrier: MuonNSCarrier = _normalize_muon_ns_carrier(ns_carrier)

    def init_single(self, parameter: mx.array, state: dict[str, Any]) -> None:
        state["v"] = mx.zeros(parameter.shape, dtype=_muon_state_dtype())

    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: dict[str, Any],
    ) -> mx.array:
        state_dtype = state["v"].dtype
        gradient_for_state = gradient.astype(state_dtype)
        if self.weight_decay != 0:
            gradient_for_state = gradient_for_state + self.weight_decay * parameter.astype(
                state_dtype
            )

        v = self.momentum * state["v"]
        v = v + (1 - self.momentum) * gradient_for_state
        state["v"] = v

        if self.nesterov:
            update = gradient_for_state * (1 - self.momentum) + v * self.momentum
        else:
            update = v

        lr = self.learning_rate.astype(update.dtype)
        if update.ndim >= 2:
            original_shape = update.shape
            reshape_needed = update.ndim > 2

            if reshape_needed:
                update = mx.reshape(update, (update.shape[0], -1))

            update = _muon_zeropower_newtonschulz5(
                update,
                steps=self.ns_steps,
                ns_carrier=self.ns_carrier,
                output_dtype=update.dtype,
            )

            if reshape_needed:
                update = mx.reshape(update, original_shape)

            lr *= max(1, update.shape[-2] / update.shape[-1]) ** 0.5

        updated = parameter.astype(update.dtype) - lr * update
        return updated.astype(parameter.dtype)


class QuantizedMuonWithNSCarrier(MuonWithNSCarrier):
    """Muon variant that stores the persistent momentum buffer as ``uint8``
    payload + per-256-block fp32 absmax.

    Mirrors cppmega CUDA's
    ``quantized_muon_momentum_update_multi_and_normalize_groups_`` from
    ``megatron/core/optimizer/emerging_optimizers.py``. Memory cost on the
    Muon group drops from 4 B/param (fp32 momentum) to ~1.0156 B/param
    (1 B uint8 + 4/256 B absmax). On a 1.797B-param model with ~73% of the
    parameters routed to Muon (~1.31B), that is ~5.24 GiB -> ~1.33 GiB,
    about 4 GiB freed for activations + reserve.

    Critical invariant: only the *persistent* momentum buffer is quantized.
    The Newton-Schulz orthogonalization carrier inside ``apply_single``
    stays fp32 (or bf16, controlled by ``ns_carrier``) because the
    iterative matrix-sign step needs that fidelity. The fp32 fast-path
    inside the kernel is preserved: dequantize at start of the step,
    perform the standard Muon math + NS in fp32, re-quantize at the end.

    Codec is selectable via ``quant_scheme``:

    * ``"symmetric_int8_v1"`` (default, ``MUON_QUANTIZED_MOMENTUM_SCHEME``):
      uint8 with +128 bias, the existing M0-grade symmetric path.
    * ``"dynamic_int8_v1"``: opt-in bitsandbytes-style dynamic LUT (denser
      bins near zero). Same uint8 + fp32-absmax memory layout, just a
      different codec.
    """

    def __init__(
        self,
        learning_rate: float | Callable[[mx.array], mx.array],
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
        *,
        ns_carrier: str = "fp32",
        block_size: int = MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE,
        quant_scheme: MuonQuantScheme = QUANT_SCHEME_SYMMETRIC,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_carrier=ns_carrier,
        )
        self.block_size = int(block_size)
        if quant_scheme not in QUANT_SCHEMES:
            raise ValueError(
                f"quant_scheme must be one of {QUANT_SCHEMES}; got {quant_scheme!r}"
            )
        self.quant_scheme: str = quant_scheme

    def init_single(self, parameter: mx.array, state: dict[str, Any]) -> None:
        # For symmetric int8 the +128 bias maps to signed 0 -> all-zero
        # momentum after dequant. For the dynamic LUT scheme the canonical
        # zero entry sits at byte index 127 (LUT[127] == 0.0 from the
        # ``data.append(0)`` step in ``create_dynamic_map``). Either way an
        # all-zero absmax forces dequant to 0 regardless of the byte chosen,
        # but we still want the byte to be zero-valued so the moment stays
        # zero on the very first step before the per-block absmax updates.
        nb = _quant_num_blocks(int(parameter.size), self.block_size)
        zero_byte = 128 if self.quant_scheme == QUANT_SCHEME_SYMMETRIC else 127
        state["v_quant"] = mx.full(parameter.shape, zero_byte, dtype=mx.uint8)
        state["v_absmax"] = mx.zeros((nb,), dtype=mx.float32)

    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: dict[str, Any],
    ) -> mx.array:
        # 1) Dequantize persistent momentum to fp32 for the inner math,
        # using whichever codec the optimizer was configured for.
        v_prev = dequantize_blockwise(
            state["v_quant"],
            state["v_absmax"],
            scheme=self.quant_scheme,
            out_dtype=mx.float32,
        )

        gradient_for_state = gradient.astype(mx.float32)
        if self.weight_decay != 0:
            gradient_for_state = gradient_for_state + self.weight_decay * parameter.astype(
                mx.float32
            )

        # 2) Standard Muon momentum update in fp32.
        v_new = self.momentum * v_prev + (1.0 - self.momentum) * gradient_for_state

        if self.nesterov:
            update = (
                gradient_for_state * (1.0 - self.momentum) + v_new * self.momentum
            )
        else:
            update = v_new

        # 3) Re-quantize the *persistent* momentum buffer (and only that).
        # The NS update path below operates on the un-quantized fp32 ``update``
        # tensor so the matrix-sign iteration retains fp32 carrier fidelity.
        v_q, v_absmax = quantize_blockwise(
            v_new, self.block_size, scheme=self.quant_scheme
        )
        state["v_quant"] = v_q
        state["v_absmax"] = v_absmax

        # 4) Newton-Schulz on the fp32 update; carrier stays fp32 (or bf16
        # via ns_carrier) -- never quantized. This matches cppmega CUDA.
        lr = self.learning_rate.astype(mx.float32)
        if update.ndim >= 2:
            original_shape = update.shape
            reshape_needed = update.ndim > 2

            if reshape_needed:
                update = mx.reshape(update, (update.shape[0], -1))

            update = _muon_zeropower_newtonschulz5(
                update,
                steps=self.ns_steps,
                ns_carrier=self.ns_carrier,
                output_dtype=mx.float32,
            )

            if reshape_needed:
                update = mx.reshape(update, original_shape)

            lr = lr * (max(1, update.shape[-2] / update.shape[-1]) ** 0.5)

        # 5) Apply the orthogonalized update in fp32, cast back to the
        # parameter's dtype (bf16 in production).
        updated = parameter.astype(mx.float32) - lr * update
        return updated.astype(parameter.dtype)


class MuonAdamWMulti(optim.Optimizer):
    """Composite optimizer that delegates 2-D weights to Muon and the rest to
    AdamW, matching Megatron emerging_optimizers' ``_is_nonlinear_or_embedding``
    routing.

    The wrapper deliberately does not subclass :class:`mlx.optimizers.MultiOptimizer`
    because the audit tooling needs a state surface with explicit ``muon`` and
    ``adamw`` buckets rather than the upstream ``{"states": [...]}`` list.
    """

    def __init__(
        self,
        muon_optimizer: optim.Optimizer,
        adamw_optimizer: optim.Optimizer,
        *,
        predicate: Callable[[str, mx.array], bool] = is_muon_compatible,
    ) -> None:
        super().__init__()
        self._muon = muon_optimizer
        self._adamw = adamw_optimizer
        self._predicate = predicate

    @property
    def muon(self) -> optim.Optimizer:
        return self._muon

    @property
    def adamw(self) -> optim.Optimizer:
        return self._adamw

    @property
    def predicate(self) -> Callable[[str, mx.array], bool]:
        return self._predicate

    def _split(self, tree: Any) -> tuple[Any, Any]:
        return split_param_groups(tree, predicate=self._predicate)

    def init(self, parameters: Any) -> None:
        muon_params, adamw_params = self._split(parameters)
        self._muon.init(muon_params)
        self._adamw.init(adamw_params)
        self._initialized = True

    def apply_gradients(self, gradients: Any, parameters: Any) -> Any:
        muon_grads, adamw_grads = self._split(gradients)
        muon_params, adamw_params = self._split(parameters)
        merged: Any = {}
        if muon_grads:
            merged = tree_merge(
                merged,
                self._muon.apply_gradients(muon_grads, muon_params),
            )
        if adamw_grads:
            merged = tree_merge(
                merged,
                self._adamw.apply_gradients(adamw_grads, adamw_params),
            )
        return merged

    def update(self, model: nn.Module, gradients: Any) -> None:
        model.update(self.apply_gradients(gradients, model))

    @property
    def state(self) -> dict[str, Any]:
        return {"muon": self._muon.state, "adamw": self._adamw.state}

    @state.setter
    def state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or "muon" not in state or "adamw" not in state:
            raise ValueError(
                "MuonAdamWMulti state must be a dict with 'muon' and 'adamw' buckets"
            )
        self._muon.state = state["muon"]
        self._adamw.state = state["adamw"]

    @property
    def learning_rate(self) -> mx.array:
        return self._muon.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float | mx.array) -> None:
        self._muon.learning_rate = learning_rate
        self._adamw.learning_rate = learning_rate


def _normalize_muon_scalar_optimizer(scalar_optimizer: str) -> MuonScalarOptimizer:
    normalized = scalar_optimizer.strip().lower()
    if normalized not in MUON_SCALAR_OPTIMIZERS:
        allowed = ", ".join(MUON_SCALAR_OPTIMIZERS)
        raise ValueError(
            f"scalar_optimizer must be one of {allowed}; got {scalar_optimizer!r}"
        )
    return normalized  # type: ignore[return-value]


def make_muon(
    *,
    lr_muon: float = 2e-3,
    lr_adamw: float = 1e-4,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    betas_adamw: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    cppmega_cuda_parity: bool = False,
    ns_carrier: str = "fp32",
    scalar_optimizer: str = "adamw",
    quantize_momentum: bool = False,
    quantize_momentum_scheme: MuonQuantScheme = QUANT_SCHEME_SYMMETRIC,
    adam8bit_quant_scheme: Adam8bitQuantScheme = QUANT_SCHEME_SYMMETRIC,
    lion8bit_quant_scheme: Lion8bitQuantScheme = QUANT_SCHEME_SYMMETRIC,
) -> MuonAdamWMulti:
    """Construct a Muon + AdamW chained optimizer with cppmega CUDA-style routing.

    Defaults follow Keller Jordan's reference implementation: Muon at ``lr=2e-3``
    with ``momentum=0.95`` Nesterov updates and 5 Newton-Schulz steps, AdamW at
    ``lr=1e-4`` with ``betas=(0.9, 0.95)``. The Muon group covers 2-D linear
    weights only; the AdamW group catches embeddings, lm_head, RMSNorm scalars,
    Mamba ``A_log``/``D``/``dt_bias`` leaves, and any 3-D+ tensors — exactly the
    inversion of Megatron emerging_optimizers' ``_is_nonlinear_or_embedding``.

    When ``cppmega_cuda_parity`` is True, both groups are forced to share
    ``lr=1e-4`` and ``betas_adamw=(0.9, 0.999)``, which matches the gb10 CUDA
    runner that takes a single ``--lr 1e-4`` flag and is the parity-trace
    configuration used by the audit tooling.

    ``ns_carrier`` mirrors cppmega's ``CPPMEGA_MUON_NS_CARRIER`` knob. The
    factory uses a repo-local Muon subclass so the carrier policy does not
    depend on MLX's private Newton-Schulz helper.

    ``scalar_optimizer`` selects the optimizer used for the scalar/embedding
    group (embeddings, RMSNorm, biases, Mamba scalars, 3-D+ tensors). The
    default ``"adamw"`` keeps the fp32-moments AdamW used everywhere else in
    the repo. Other choices mirror cppmega CUDA's ``muon_scalar_optimizer``
    knob from the ``Megatron emerging_optimizers`` registry:

    * ``"adam8bit"`` swaps in :class:`Adam8bit` (uint8 m, v + per-256-block
      fp32 absmax). Cuts the scalar-group state from ~8 B/param to ~2 B/param.
    * ``"lion"`` swaps in :class:`LionFP32Moments` (single fp32 m buffer).
      Cuts the scalar-group state from ~8 B/param to ~4 B/param.
    * ``"lion8bit"`` swaps in :class:`Lion8bit` (single uint8 m + per-256-block
      fp32 absmax). Cuts the scalar-group state from ~8 B/param to ~1 B/param,
      mirroring ``bitsandbytes.optim.Lion8bit`` per
      ``cppmega/docs/lion8bit_ab_2026_04_25.md``.

    ``quantize_momentum`` mirrors cppmega CUDA's
    ``quantized_muon_momentum_update_multi_and_normalize_groups_`` knob.
    When True, the persistent Muon momentum buffer is stored as ``uint8``
    payload + per-256-block fp32 absmax (~1.0156 B/param) instead of fp32
    (~4 B/param). The Newton-Schulz orthogonalization carrier stays fp32
    regardless -- only the persistent state is quantized.

    ``quantize_momentum_scheme`` selects the Muon-momentum codec
    (``"symmetric_int8_v1"`` default; ``"dynamic_int8_v1"`` for the bnb LUT).
    ``adam8bit_quant_scheme`` does the same for the Adam8bit scalar group
    when ``scalar_optimizer="adam8bit"``. ``lion8bit_quant_scheme`` does the
    same for the Lion8bit scalar group when ``scalar_optimizer="lion8bit"``.
    All three default to symmetric for backwards compatibility.
    """

    ns_carrier = _normalize_muon_ns_carrier(
        os.environ.get(MUON_NS_CARRIER_ENV, ns_carrier)
    )
    scalar_optimizer = _normalize_muon_scalar_optimizer(scalar_optimizer)

    if cppmega_cuda_parity:
        lr_muon = 1e-4
        lr_adamw = 1e-4
        nesterov = False
        betas_adamw = (0.9, 0.999)

    muon_kwargs: dict[str, Any] = {"learning_rate": lr_muon}
    accepted = _muon_supported_kwargs() | {"ns_carrier"}
    if "momentum" in accepted:
        muon_kwargs["momentum"] = momentum
    if "nesterov" in accepted:
        muon_kwargs["nesterov"] = nesterov
    if "ns_steps" in accepted:
        muon_kwargs["ns_steps"] = ns_steps
    if "weight_decay" in accepted:
        muon_kwargs["weight_decay"] = weight_decay
    if "ns_carrier" in accepted:
        muon_kwargs["ns_carrier"] = ns_carrier

    if quantize_momentum_scheme not in QUANT_SCHEMES:
        raise ValueError(
            f"quantize_momentum_scheme must be one of {QUANT_SCHEMES}; "
            f"got {quantize_momentum_scheme!r}"
        )
    if adam8bit_quant_scheme not in QUANT_SCHEMES:
        raise ValueError(
            f"adam8bit_quant_scheme must be one of {QUANT_SCHEMES}; "
            f"got {adam8bit_quant_scheme!r}"
        )
    if lion8bit_quant_scheme not in QUANT_SCHEMES:
        raise ValueError(
            f"lion8bit_quant_scheme must be one of {QUANT_SCHEMES}; "
            f"got {lion8bit_quant_scheme!r}"
        )
    if quantize_momentum:
        muon_optimizer: optim.Optimizer = QuantizedMuonWithNSCarrier(
            **muon_kwargs, quant_scheme=quantize_momentum_scheme
        )
    else:
        muon_optimizer = MuonWithNSCarrier(**muon_kwargs)
    if scalar_optimizer == "adam8bit":
        adamw_optimizer: optim.Optimizer = make_adam8bit(
            learning_rate=lr_adamw,
            weight_decay=weight_decay,
            betas=list(betas_adamw),
            eps=eps,
            quant_scheme=adam8bit_quant_scheme,
        )
    elif scalar_optimizer == "lion":
        # Lion ignores eps; betas are Lion's (b1, b2) per Chen et al. The
        # scalar-group LR is shared with AdamW (lr_adamw); callers wanting
        # the Chen-recommended 3-10x smaller Lion LR should pass it via
        # lr_adamw explicitly. The default lr_adamw=1e-4 is already in that
        # ballpark for Lion.
        adamw_optimizer = make_lion(
            learning_rate=lr_adamw,
            weight_decay=weight_decay,
            betas=list(betas_adamw),
        )
    elif scalar_optimizer == "lion8bit":
        adamw_optimizer = make_lion8bit(
            learning_rate=lr_adamw,
            weight_decay=weight_decay,
            betas=list(betas_adamw),
            quant_scheme=lion8bit_quant_scheme,
        )
    else:
        adamw_optimizer = make_adamw(
            learning_rate=lr_adamw,
            weight_decay=weight_decay,
            betas=list(betas_adamw),
            eps=eps,
        )
    return MuonAdamWMulti(muon_optimizer, adamw_optimizer)


def collect_adamw_moment_dtypes(state: Any) -> dict[str, str]:
    moment_dtypes: dict[str, str] = {}

    def walk(path: tuple[str, ...], value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk((*path, str(key)), item)
            return
        if isinstance(value, list | tuple):
            for index, item in enumerate(value):
                walk((*path, str(index)), item)
            return
        if isinstance(value, mx.array) and path and path[-1] in ADAMW_MOMENT_STATE_KEYS:
            moment_dtypes["/".join(path)] = dtype_name(value)

    walk((), state)
    return moment_dtypes


def adamw_moment_dtypes_ok(
    state: Any,
    *,
    required_dtype: str = "float32",
) -> bool:
    moment_dtypes = collect_adamw_moment_dtypes(state)
    return bool(moment_dtypes) and all(
        dtype == required_dtype for dtype in moment_dtypes.values()
    )


def dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", value)
    return str(dtype).removeprefix("mlx.core.")


__all__ = [
    "ADAM8BIT_CLASS",
    "ADAM8BIT_QUANT_KIND",
    "ADAM8BIT_QUANT_SCHEMES",
    "ADAM8BIT_SOURCE",
    "ADAMW_BASE_CLASS",
    "ADAMW_FP32_MOMENTS_CLASS",
    "ADAMW_FP32_MOMENTS_SOURCE",
    "ADAMW_MOMENT_STATE_KEYS",
    "Adam8bit",
    "Adam8bitQuantScheme",
    "AdamWFP32Moments",
    "EMBEDDING_LIKE_NAME_HINTS",
    "LION8BIT_CLASS",
    "LION8BIT_QUANT_KIND",
    "LION8BIT_SOURCE",
    "Lion8bit",
    "LionFP32Moments",
    "MAMBA_SCALAR_LEAVES",
    "MUON_ADAMW_MULTI_CLASS",
    "MUON_ADAMW_MULTI_SOURCE",
    "MUON_BASE_CLASS",
    "MUON_NS_CARRIER_ENV",
    "MUON_NS_CARRIERS",
    "MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE",
    "MUON_QUANTIZED_MOMENTUM_SCHEME",
    "MUON_QUANTIZED_MOMENTUM_SCHEMES",
    "MUON_SCALAR_OPTIMIZERS",
    "MuonAdamWMulti",
    "MuonNSCarrier",
    "MuonQuantScheme",
    "MuonScalarOptimizer",
    "MuonWithNSCarrier",
    "QuantizedMuonWithNSCarrier",
    "adamw_moment_dtypes_ok",
    "collect_adamw_moment_dtypes",
    "dtype_name",
    "is_muon_compatible",
    "make_adam8bit",
    "make_adamw",
    "make_lion",
    "make_lion8bit",
    "make_muon",
    "split_param_groups",
]
