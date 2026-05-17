"""Compiled MLX pretraining step utilities.

The shape mirrors the current MLX-LM trainer pattern: compute loss with
nn.value_and_grad, update the optimizer, and explicitly mx.eval the
model/optimizer state.  The compiled path captures model.state and
optimizer.state so fixed-shape batches can be replayed efficiently.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Literal, Mapping, TypeVar, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import average_gradients
import mlx.optimizers as optim
from mlx.utils import tree_map

from cppmega_mlx.data.batch import LMTokenBatch, ensure_lm_batch
from cppmega_mlx.training.loss import next_token_cross_entropy


LossFn = Callable[
    [nn.Module, LMTokenBatch | Mapping[str, mx.array] | mx.array],
    tuple[mx.array, mx.array],
]
CompileTarget = Literal[
    "mamba3_pre",
    "data_dep_a",
    "rmsnorm",
    "rmsnorm_gated",
    "moe_router",
]
F = TypeVar("F", bound=Callable[..., Any])

REGIONAL_COMPILE_TARGETS: Mapping[CompileTarget, bool] = {
    "mamba3_pre": True,
    "data_dep_a": True,
    "rmsnorm": False,
    "rmsnorm_gated": False,
    "moe_router": False,
}

STABLE_BATCH_KEYS = (
    "tokens",
    "attention_mask",
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)

CompiledBatch = dict[str, mx.array | None]
CompiledBatchSignature = tuple[tuple[str, tuple[int, ...] | None, str | None], ...]


def should_compile_region(target: CompileTarget) -> bool:
    """Return the fail-closed regional compile decision for a known target."""

    try:
        return REGIONAL_COMPILE_TARGETS[target]
    except KeyError as exc:
        raise ValueError(f"unknown regional compile target: {target!r}") from exc


def regional_compile(
    target: CompileTarget,
    fn: F | None = None,
    **compile_kwargs: Any,
) -> F | Callable[[F], F]:
    """Compile only regions that cppmega benchmarks allow.

    This is deliberately separate from CompiledPretrainingStep's full-step
    compile path.  It codifies the measured per-op allow/deny matrix so local
    call sites do not blanket-compile small regions that are known slowdowns.
    """

    def decorate(inner: F) -> F:
        if not should_compile_region(target):
            return inner
        compiled = mx.compile(inner, **compile_kwargs)
        return cast(F, compiled)

    if fn is None:
        return decorate
    return decorate(fn)


def maybe_compile_region(
    target: CompileTarget,
    fn: F,
    **compile_kwargs: Any,
) -> F:
    """Function-call form of regional_compile for dynamic call sites."""

    compiled = regional_compile(target, fn, **compile_kwargs)
    return cast(F, compiled)


def normalize_compiled_batch(
    batch: LMTokenBatch | Mapping[str, mx.array | None] | mx.array,
) -> CompiledBatch:
    """Return the fixed-key batch pytree used by compiled train steps.

    mx.compile keys off the Python input structure as well as array shapes
    and dtypes.  Keep every optional side channel present in the dict and use
    None for absent fields so callers do not alternate between different
    dict key sets when switching between plain token batches and structured
    batches.
    """

    batch_dict = ensure_lm_batch(batch).as_dict()
    return {key: batch_dict.get(key) for key in STABLE_BATCH_KEYS}


@dataclass
class PretrainingState:
    """Python-side resume cursor for a local pretraining run."""

    step: int = 0
    trained_tokens: int = 0

    def advance(self, ntokens: int) -> None:
        self.step += 1
        self.trained_tokens += ntokens

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PretrainingState":
        return cls(
            step=_require_non_negative_int(data.get("step", 0), name="step"),
            trained_tokens=_require_non_negative_int(
                data.get("trained_tokens", 0),
                name="trained_tokens",
            ),
        )


@dataclass(frozen=True)
class PretrainingMetrics:
    loss: float
    ntokens: int
    step: int
    trained_tokens: int
    updated: bool
    seconds: float
    tokens_per_second: float
    compiled: bool


class CompiledPretrainingStep:
    """Small stateful train-step wrapper with eager fallback.

    Batches are normalized to one fixed-key dict before entering the compiled
    function so optional side-channel presence does not create Python-level key
    churn.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        *,
        state: PretrainingState | Mapping[str, int] | None = None,
        loss_fn: LossFn = next_token_cross_entropy,
        compile: bool = True,
        grad_accum_steps: int = 1,
        split_grad_update_eval: bool = False,
    ):
        if not isinstance(compile, bool):
            raise TypeError("compile must be a boolean")
        if not isinstance(split_grad_update_eval, bool):
            raise TypeError("split_grad_update_eval must be a boolean")
        grad_accum_steps = _require_positive_int(
            grad_accum_steps,
            name="grad_accum_steps",
        )
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.compile = compile
        self.split_grad_update_eval = split_grad_update_eval
        self.grad_accum_steps = grad_accum_steps
        self.state = (
            state
            if isinstance(state, PretrainingState)
            else PretrainingState.from_dict(state or {})
        )
        self._compiled_step: Callable[..., tuple[mx.array, mx.array, Any]] | None = None
        self._compiled_batch_signature: CompiledBatchSignature | None = None
        self._grad_accum: Any = None
        self._pending_microbatches = 0
        self._loss_and_grad = nn.value_and_grad(self.model, self.loss_fn)

    def __call__(
        self,
        batch: LMTokenBatch | Mapping[str, mx.array] | mx.array,
    ) -> PretrainingMetrics:
        self.model.train()
        batch_dict = normalize_compiled_batch(batch)
        if self.compile:
            self._check_compiled_batch_signature(batch_dict)
        pending_microbatches = self._pending_microbatches + 1
        do_update = pending_microbatches == self.grad_accum_steps

        start = time.perf_counter()
        if self.compile:
            if self._compiled_step is None:
                self._compiled_step = self._build_compiled_step()
            loss, ntokens, self._grad_accum = self._compiled_step(
                batch_dict,
                self._grad_accum,
                do_update,
            )
        else:
            loss, ntokens, self._grad_accum = self._eager_step(
                batch_dict,
                self._grad_accum,
                do_update,
            )
        mx.eval(
            self.model.state,
            self.optimizer.state,
            mx.random.state,
            loss,
            ntokens,
            self._grad_accum,
        )
        elapsed = time.perf_counter() - start

        tokens = int(ntokens.item())
        self._pending_microbatches = 0 if do_update else pending_microbatches
        self.state.advance(tokens)
        return PretrainingMetrics(
            loss=float(loss.item()),
            ntokens=tokens,
            step=self.state.step,
            trained_tokens=self.state.trained_tokens,
            updated=do_update,
            seconds=elapsed,
            tokens_per_second=tokens / elapsed if elapsed > 0 else float("inf"),
            compiled=self.compile,
        )

    @property
    def gradient_accumulator(self) -> Any:
        """Gradient accumulator tree needed for exact mid-accumulation resume."""

        return self._grad_accum

    def state_dict(self) -> dict[str, Any]:
        """Return all Python-side state needed to resume this train-step wrapper."""

        return {
            "state": self.state.to_dict(),
            "grad_accum_steps": self.grad_accum_steps,
            "pending_microbatches": self._pending_microbatches,
            "gradient_accumulator_present": self._grad_accum is not None,
            "compiled": self.compile,
        }

    def load_state_dict(
        self,
        data: Mapping[str, Any],
        *,
        gradient_accumulator: Any = None,
    ) -> None:
        """Restore Python-side state from state_dict metadata.

        Optimizer/model tensors are restored by checkpoint loading.  Pending
        gradient accumulation is explicit: a non-zero pending count must be
        paired with the serialized gradient accumulator tree.
        """

        state_payload = data.get("state", data)
        if not isinstance(state_payload, Mapping):
            raise ValueError("training state must contain a state object")

        grad_accum_steps = _require_positive_int(
            data.get("grad_accum_steps", self.grad_accum_steps),
            name="grad_accum_steps",
        )
        if grad_accum_steps != self.grad_accum_steps:
            raise ValueError(
                "checkpoint grad_accum_steps "
                f"{grad_accum_steps} does not match runner {self.grad_accum_steps}"
            )

        pending_microbatches = _require_non_negative_int(
            data.get("pending_microbatches", 0),
            name="pending_microbatches",
        )
        if pending_microbatches < 0 or pending_microbatches >= self.grad_accum_steps:
            raise ValueError(
                "pending_microbatches must be in "
                f"[0, {self.grad_accum_steps})"
            )

        expects_accumulator = _require_bool(
            data.get("gradient_accumulator_present", False),
            name="gradient_accumulator_present",
        )
        if pending_microbatches > 0 and gradient_accumulator is None:
            raise ValueError(
                "pending_microbatches requires a gradient_accumulator for exact resume"
            )
        if expects_accumulator and gradient_accumulator is None:
            raise ValueError("checkpoint metadata expects a gradient_accumulator")
        if pending_microbatches == 0 and gradient_accumulator is not None:
            raise ValueError("gradient_accumulator cannot be restored at an update boundary")

        self.state = PretrainingState.from_dict(cast(Mapping[str, Any], state_payload))
        self._pending_microbatches = pending_microbatches
        self._grad_accum = gradient_accumulator
        self._compiled_step = None
        self._compiled_batch_signature = None

    def _check_compiled_batch_signature(self, batch: CompiledBatch) -> None:
        signature = _compiled_batch_signature(batch)
        if self._compiled_batch_signature is None:
            self._compiled_batch_signature = signature
            return
        if signature != self._compiled_batch_signature:
            raise ValueError(
                "compiled training step requires a fixed batch shape/dtype/field "
                "signature; create a new CompiledPretrainingStep for a new shape"
            )

    def _accumulate_or_update(
        self,
        grads: Any,
        prev_grad: Any,
        do_update: bool,
    ) -> Any:
        if prev_grad is not None:
            grads = tree_map(lambda x, y: x + y, grads, prev_grad)

        if do_update:
            grads = average_gradients(grads)
            if self.grad_accum_steps > 1:
                grads = tree_map(lambda x: x / self.grad_accum_steps, grads)
            self.optimizer.update(self.model, grads)
            return None

        return grads

    def _eager_step(
        self,
        batch: CompiledBatch,
        prev_grad: Any,
        do_update: bool,
    ) -> tuple[mx.array, mx.array, Any]:
        loss_batch = cast(Mapping[str, mx.array], batch)
        (loss, ntokens), grads = self._loss_and_grad(self.model, loss_batch)
        if self.split_grad_update_eval:
            mx.eval(loss, ntokens, grads)
        grads = self._accumulate_or_update(grads, prev_grad, do_update)
        return loss, ntokens, grads

    def _build_compiled_step(
        self,
    ) -> Callable[..., tuple[mx.array, mx.array, Any]]:
        captured_state = [self.model.state, self.optimizer.state, mx.random.state]

        @partial(mx.compile, inputs=captured_state, outputs=captured_state)
        def step(
            batch: CompiledBatch,
            prev_grad: Any,
            do_update: bool,
        ) -> tuple[mx.array, mx.array, Any]:
            loss_batch = cast(Mapping[str, mx.array], batch)
            (loss, ntokens), grads = self._loss_and_grad(self.model, loss_batch)
            grads = self._accumulate_or_update(grads, prev_grad, do_update)
            return loss, ntokens, grads

        return step


def _compiled_batch_signature(batch: CompiledBatch) -> CompiledBatchSignature:
    signature: list[tuple[str, tuple[int, ...] | None, str | None]] = []
    for key in STABLE_BATCH_KEYS:
        value = batch[key]
        if value is None:
            signature.append((key, None, None))
        else:
            signature.append((key, tuple(int(dim) for dim in value.shape), str(value.dtype)))
    return tuple(signature)


def _require_non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


__all__ = [
    "CompileTarget",
    "CompiledPretrainingStep",
    "REGIONAL_COMPILE_TARGETS",
    "maybe_compile_region",
    "normalize_compiled_batch",
    "PretrainingMetrics",
    "PretrainingState",
    "regional_compile",
    "should_compile_region",
    "STABLE_BATCH_KEYS",
]
