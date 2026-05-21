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
from typing import Any, Callable, Literal, Mapping, Sequence, TypeVar, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import average_gradients
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

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
PathCGradientProbe = Callable[[Mapping[str, Any]], None]
PathCTrainingRuntime = Any
PATH_C_DIRECT_FUSION_VALUE_AND_GRAD_CONTRACT = (
    "path_c_direct_fusion_value_and_grad_v1"
)
PATH_C_TRAINING_VALUE_AND_GRAD_CONTRACT = (
    PATH_C_DIRECT_FUSION_VALUE_AND_GRAD_CONTRACT
)
PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_CONTRACT = (
    "path_c_fused_train_block_training_runtime_v1"
)
PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT = (
    "path_c_fused_train_block_value_and_grad_v1"
)

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
    "platform_ids",
)

CompiledBatch = dict[str, mx.array | None]
CompiledBatchSignature = tuple[tuple[str, tuple[int, ...] | None, str | None], ...]


class PathCGradientBufferCapture:
    """Opt-in Path C gradient owner that stores references, not copies."""

    def __init__(
        self,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
        *,
        owner_name: str = "CompiledPretrainingStep.path_c_gradient_capture",
    ) -> None:
        self.owner_name = owner_name
        self.aliases = {
            str(source): (str(target),)
            if isinstance(target, str)
            else tuple(str(item) for item in target)
            for source, target in (aliases or {}).items()
        }
        self.buffers: dict[str, mx.array] = {}
        self.events: list[Mapping[str, Any]] = []

    def __call__(self, event: Mapping[str, Any]) -> None:
        tensor = event.get("tensor")
        if not isinstance(tensor, mx.array):
            return
        logical_names = tuple(str(name) for name in event.get("logical_names", ()))
        for name in logical_names:
            self.buffers[name] = tensor
            for alias in self.aliases.get(name, ()):
                self.buffers[alias] = tensor
        self.events.append(_path_c_capture_event_metadata(event))

    def clear(self) -> None:
        self.buffers.clear()
        self.events.clear()


def _path_c_capture_event_metadata(event: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in event.items():
        if isinstance(value, mx.array):
            metadata[f"{key}_shape"] = tuple(int(dim) for dim in value.shape)
            metadata[f"{key}_dtype"] = str(value.dtype)
        else:
            metadata[key] = value
    return metadata


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
        path_c_gradient_probe: PathCGradientProbe | None = None,
        path_c_training_runtime: PathCTrainingRuntime | None = None,
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
        self._path_c_gradient_probe: PathCGradientProbe | None = None
        if path_c_gradient_probe is not None:
            self.attach_path_c_gradient_probe(path_c_gradient_probe)
        self._path_c_training_runtime: PathCTrainingRuntime | None = None
        if path_c_training_runtime is not None:
            self.attach_path_c_training_runtime(path_c_training_runtime)

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

    def attach_path_c_gradient_probe(self, probe: PathCGradientProbe) -> None:
        """Install an explicit zero-copy gradient probe for eager train steps."""

        if self.compile:
            raise ValueError("Path C gradient probe requires compile=False")
        if not callable(probe):
            raise TypeError("probe must be callable")
        self._path_c_gradient_probe = probe

    def detach_path_c_gradient_probe(self) -> None:
        """Remove the explicit Path C gradient probe."""

        self._path_c_gradient_probe = None

    def attach_path_c_training_runtime(self, runtime: PathCTrainingRuntime) -> None:
        """Install an explicit runtime that owns eager value-and-grad execution."""

        if self.compile:
            raise ValueError("Path C training runtime requires compile=False")
        value_and_grad = getattr(runtime, "value_and_grad", None)
        if not callable(value_and_grad):
            raise TypeError("Path C training runtime must define value_and_grad")
        bind = getattr(runtime, "bind_training_graph", None)
        bound = False
        if callable(bind):
            binding = {
                "owner": "CompiledPretrainingStep",
                "uses_forward_hook": True,
                "uses_backward_or_vjp_hook": True,
            }
            if _path_c_training_runtime_uses_fused_train_block(runtime):
                binding["uses_fused_train_block_runtime"] = True
            else:
                binding["uses_direct_chain_runtime"] = True
            bind(**binding)
            bound = True
        try:
            value_and_grad_contract = _path_c_training_runtime_value_and_grad_contract(
                runtime
            )
            if value_and_grad_contract.get("status") != "ok":
                raise ValueError(
                    "Path C training runtime value_and_grad_contract is incomplete: "
                    f"{value_and_grad_contract.get('status')}"
                )
        except Exception:
            if bound:
                unbind = getattr(runtime, "unbind_training_graph", None)
                if callable(unbind):
                    unbind(owner="CompiledPretrainingStep")
            raise
        self._path_c_training_runtime = runtime

    def detach_path_c_training_runtime(self) -> None:
        """Remove the explicit Path C training runtime."""

        runtime = self._path_c_training_runtime
        unbind = getattr(runtime, "unbind_training_graph", None)
        if callable(unbind):
            unbind(owner="CompiledPretrainingStep")
        self._path_c_training_runtime = None

    def state_dict(self) -> dict[str, Any]:
        """Return all Python-side state needed to resume this train-step wrapper."""

        return {
            "state": self.state.to_dict(),
            "grad_accum_steps": self.grad_accum_steps,
            "pending_microbatches": self._pending_microbatches,
            "gradient_accumulator_present": self._grad_accum is not None,
            "compiled": self.compile,
            "path_c_training_runtime_installed": (
                self._path_c_training_runtime is not None
            ),
            "path_c_training_runtime_class": type(
                self._path_c_training_runtime
            ).__name__
            if self._path_c_training_runtime is not None
            else None,
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

    def _emit_path_c_gradients(self, grads: Any) -> None:
        probe = self._path_c_gradient_probe
        if probe is None:
            return
        for parameter_name, tensor in tree_flatten(grads):
            if not isinstance(tensor, mx.array):
                continue
            probe(
                {
                    "name": "gradient",
                    "parameter_name": parameter_name,
                    "logical_names": (f"{parameter_name}_grad",),
                    "tensor": tensor,
                    "phase": "value_and_grad",
                }
            )

    def _eager_step(
        self,
        batch: CompiledBatch,
        prev_grad: Any,
        do_update: bool,
    ) -> tuple[mx.array, mx.array, Any]:
        loss_batch = cast(Mapping[str, mx.array], batch)
        runtime = self._path_c_training_runtime
        if runtime is None:
            (loss, ntokens), grads = self._loss_and_grad(self.model, loss_batch)
        else:
            (loss, ntokens), grads = runtime.value_and_grad(
                self.model,
                loss_batch,
                self._loss_and_grad,
            )
        self._emit_path_c_gradients(grads)
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


def _path_c_training_runtime_value_and_grad_contract(
    runtime: PathCTrainingRuntime,
) -> dict[str, Any]:
    raw_contract = getattr(runtime, "value_and_grad_contract", None)
    if callable(raw_contract):
        raw_contract = raw_contract()
    if not isinstance(raw_contract, Mapping):
        return {
            "status": "missing",
            "contract": PATH_C_TRAINING_VALUE_AND_GRAD_CONTRACT,
        }
    payload = dict(raw_contract)
    contract = str(payload.get("contract", ""))
    owner = str(payload.get("owner", ""))
    uses_direct_chain_runtime = bool(payload.get("uses_direct_chain_runtime"))
    uses_fused_train_block_runtime = bool(
        payload.get("uses_fused_train_block_runtime")
    )
    direct_contract = contract == PATH_C_DIRECT_FUSION_VALUE_AND_GRAD_CONTRACT
    fused_contract = contract == PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT
    uses_runtime = bool(
        (direct_contract and uses_direct_chain_runtime)
        or (fused_contract and uses_fused_train_block_runtime)
    )
    uses_forward = bool(payload.get("uses_forward_hook"))
    uses_reverse = bool(payload.get("uses_backward_or_vjp_hook"))
    returns_model_grads = bool(payload.get("returns_model_grads"))
    returns_full_model_grads = bool(payload.get("returns_full_model_grads", False))
    loss_cotangent_bridge_ready = bool(payload.get("loss_cotangent_bridge_ready"))
    model_gradient_tree_ready = bool(payload.get("model_gradient_tree_ready"))
    delegates_to_eager = bool(payload.get("delegates_to_eager_loss_and_grad", True))
    hidden_packing = bool(payload.get("hidden_packing_performed", False))
    status = (
        "ok"
        if (direct_contract or fused_contract)
        and owner == "CompiledPretrainingStep"
        and uses_runtime
        and uses_forward
        and uses_reverse
        and returns_model_grads
        and returns_full_model_grads
        and loss_cotangent_bridge_ready
        and model_gradient_tree_ready
        and not delegates_to_eager
        and not hidden_packing
        else "incomplete"
    )
    return {
        **payload,
        "status": status,
        "contract": contract or PATH_C_TRAINING_VALUE_AND_GRAD_CONTRACT,
        "owner": owner or None,
        "uses_direct_chain_runtime": uses_direct_chain_runtime,
        "uses_fused_train_block_runtime": uses_fused_train_block_runtime,
        "uses_forward_hook": uses_forward,
        "uses_backward_or_vjp_hook": uses_reverse,
        "returns_model_grads": returns_model_grads,
        "returns_full_model_grads": returns_full_model_grads,
        "loss_cotangent_bridge_ready": loss_cotangent_bridge_ready,
        "model_gradient_tree_ready": model_gradient_tree_ready,
        "delegates_to_eager_loss_and_grad": delegates_to_eager,
        "hidden_packing_performed": hidden_packing,
    }


def _path_c_training_runtime_uses_fused_train_block(
    runtime: PathCTrainingRuntime,
) -> bool:
    return bool(
        getattr(runtime, "uses_fused_train_block_runtime", False)
        or str(getattr(runtime, "contract", ""))
        == PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_CONTRACT
    )


__all__ = [
    "CompileTarget",
    "CompiledPretrainingStep",
    "PATH_C_DIRECT_FUSION_VALUE_AND_GRAD_CONTRACT",
    "PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_CONTRACT",
    "PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT",
    "PATH_C_TRAINING_VALUE_AND_GRAD_CONTRACT",
    "PathCGradientBufferCapture",
    "PathCGradientProbe",
    "PathCTrainingRuntime",
    "REGIONAL_COMPILE_TARGETS",
    "maybe_compile_region",
    "normalize_compiled_batch",
    "PretrainingMetrics",
    "PretrainingState",
    "regional_compile",
    "should_compile_region",
    "STABLE_BATCH_KEYS",
]
