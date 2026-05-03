"""Repo-local optimizer helpers for MLX training."""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_merge, tree_unflatten


ADAMW_FP32_MOMENTS_CLASS = "cppmega_mlx.training.optimizers.AdamWFP32Moments"
ADAMW_FP32_MOMENTS_SOURCE = "cppmega_mlx.training.optimizers.make_adamw"
ADAMW_BASE_CLASS = "mlx.optimizers.AdamW"
ADAMW_MOMENT_STATE_KEYS = ("m", "v")

MUON_BASE_CLASS = "mlx.optimizers.Muon"
MUON_ADAMW_MULTI_CLASS = "cppmega_mlx.training.optimizers.MuonAdamWMulti"
MUON_ADAMW_MULTI_SOURCE = "cppmega_mlx.training.optimizers.make_muon"
MUON_NS_CARRIER_ENV = "CPPMEGA_MUON_NS_CARRIER"
MUON_NS_CARRIERS = ("fp32", "bf16")
MuonNSCarrier = Literal["fp32", "bf16"]

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
        merged: Any = {}
        if muon_grads:
            merged = tree_merge(merged, self._muon.apply_gradients(muon_grads, parameters))
        if adamw_grads:
            merged = tree_merge(merged, self._adamw.apply_gradients(adamw_grads, parameters))
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
    """

    ns_carrier = _normalize_muon_ns_carrier(
        os.environ.get(MUON_NS_CARRIER_ENV, ns_carrier)
    )

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

    muon_optimizer = MuonWithNSCarrier(**muon_kwargs)
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
    "ADAMW_BASE_CLASS",
    "ADAMW_FP32_MOMENTS_CLASS",
    "ADAMW_FP32_MOMENTS_SOURCE",
    "ADAMW_MOMENT_STATE_KEYS",
    "AdamWFP32Moments",
    "EMBEDDING_LIKE_NAME_HINTS",
    "LionFP32Moments",
    "MAMBA_SCALAR_LEAVES",
    "MUON_ADAMW_MULTI_CLASS",
    "MUON_ADAMW_MULTI_SOURCE",
    "MUON_BASE_CLASS",
    "MUON_NS_CARRIER_ENV",
    "MUON_NS_CARRIERS",
    "MuonAdamWMulti",
    "MuonNSCarrier",
    "MuonWithNSCarrier",
    "adamw_moment_dtypes_ok",
    "collect_adamw_moment_dtypes",
    "dtype_name",
    "is_muon_compatible",
    "make_adamw",
    "make_lion",
    "make_muon",
    "split_param_groups",
]
