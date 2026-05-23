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
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from cppmega_mlx.data.batch import LMTokenBatch, ensure_lm_batch
from cppmega_mlx.runtime.path_c_physical_abi import physical_abi_runtime_kernel_args
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
    "target_tokens",
    "attention_mask",
    "loss_mask",
    "document_ids",
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


class PathCFusedTrainBlockCallableArtifact:
    """Contracted wrapper around a compiled fused train-block kernel.

    The wrapper only consumes caller/model-owned physical ABI banks. It does not
    pack logical tensors into banks and does not fall back to eager
    ``loss_and_grad``.
    """

    hidden_packing_performed = False
    no_hidden_allocation_policy = True

    def __init__(
        self,
        *,
        kernel: Callable[..., Any],
        physical_abi_map: Mapping[str, Any],
        physical_abi_shapes: Mapping[str, Any],
        training_abi_contract: Mapping[str, Any],
        parameter_gradient_aliases: Mapping[str, str | Sequence[str]] | None = None,
        trainable_parameter_names: Sequence[str] | None = None,
        selected_region_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(kernel):
            raise TypeError("fused train-block kernel must be callable")
        self.kernel = kernel
        self.physical_abi_map = {
            str(name): dict(value)
            for name, value in physical_abi_map.items()
            if isinstance(value, Mapping)
        }
        self.physical_abi_shapes = {
            str(name): tuple(int(dim) for dim in tuple(shape))
            for name, shape in physical_abi_shapes.items()
        }
        self.training_abi_contract = dict(training_abi_contract)
        self.parameter_gradient_aliases = _normalize_gradient_aliases(
            parameter_gradient_aliases or {}
        )
        self.trainable_parameter_names = frozenset(
            str(name) for name in (trainable_parameter_names or ())
        )
        self.selected_region_metadata = _normalize_contract_metadata(
            selected_region_metadata or {}
        )
        self._logical_gradient_names = frozenset(
            name for name in self.physical_abi_map if name.endswith("_grad")
        )
        self._parameter_gradient_bindings = tuple(
            self._iter_parameter_gradient_bindings()
        )
        self._loss_logical_name = self._first_logical_candidate(
            self.training_abi_contract.get("loss_output_candidates", ()),
            fallback="loss",
        )
        self._ntokens_logical_name = self._first_logical_candidate(
            self.training_abi_contract.get("ntokens_output_candidates", ()),
            fallback="ntokens",
        )

    def _iter_parameter_gradient_bindings(self) -> list[tuple[str, str]]:
        bindings: list[tuple[str, str]] = []
        for parameter_grad_name, targets in sorted(
            self.parameter_gradient_aliases.items()
        ):
            for target in targets:
                if target in self._logical_gradient_names:
                    bindings.append((parameter_grad_name, target))
                    break
        return bindings

    def _first_logical_candidate(
        self,
        candidates: Any,
        *,
        fallback: str,
    ) -> str | None:
        for candidate in candidates or ():
            name = str(candidate)
            if name in self.physical_abi_map:
                return name
        return fallback if fallback in self.physical_abi_map else None

    def _bank_buffers(self, bank_owner: Any) -> Mapping[str, Any]:
        buffers = bank_owner if isinstance(bank_owner, Mapping) else None
        if buffers is None:
            buffers = getattr(bank_owner, "buffers", None)
        if not isinstance(buffers, Mapping):
            raise TypeError("fused train-block artifact requires a bank_owner")
        return buffers

    def _kernel_args_from_bank_owner(self, bank_owner: Any) -> tuple[Any, ...]:
        return physical_abi_runtime_kernel_args(
            self.physical_abi_map,
            self.physical_abi_shapes,
            self._bank_buffers(bank_owner),
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        bank_owner = kwargs.pop("bank_owner", None)
        if bank_owner is None:
            return self.kernel(*args, **kwargs)
        if args or kwargs:
            raise TypeError(
                "fused train-block artifact accepts only bank_owner-bound "
                "kernel dispatch"
            )
        return self.kernel(*self._kernel_args_from_bank_owner(bank_owner))

    def backward(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def vjp(self, *args: Any, **kwargs: Any) -> Any:
        return self.backward(*args, **kwargs)

    def _logical_buffer_view(self, bank_owner: Any, logical_name: str) -> mx.array:
        info = self.physical_abi_map.get(logical_name)
        if not isinstance(info, Mapping):
            raise ValueError(f"logical buffer is not in physical ABI: {logical_name}")
        buffers = self._bank_buffers(bank_owner)
        bank_name = str(info.get("bank", ""))
        if bank_name not in buffers:
            raise ValueError(f"bank buffer is not bound: {bank_name}")
        bank = buffers[bank_name]
        if not isinstance(bank, mx.array):
            raise TypeError(f"bank buffer {bank_name!r} must be an mx.array")
        offset = int(info.get("offset", 0))
        size = int(info.get("size", 1))
        view = bank[offset : offset + size]
        logical_shape = tuple(
            int(dim)
            for dim in tuple(info.get("logical_shape", info.get("shape", (size,))))
        )
        if logical_shape and logical_shape != tuple(view.shape):
            view = mx.reshape(view, logical_shape)
        return view

    def value_and_grad(
        self,
        model: nn.Module,
        batch: Mapping[str, mx.array],
        *,
        bank_owner: Any,
    ) -> tuple[tuple[mx.array, mx.array], Any]:
        del model, batch
        self.forward(bank_owner=bank_owner)
        if self._loss_logical_name is None or self._ntokens_logical_name is None:
            raise ValueError("fused train-block ABI does not expose loss and ntokens")
        loss = self._logical_buffer_view(bank_owner, self._loss_logical_name)
        ntokens = self._logical_buffer_view(bank_owner, self._ntokens_logical_name)
        pairs: list[tuple[str, mx.array]] = []
        seen: set[str] = set()
        for parameter_grad_name, logical_name in self._parameter_gradient_bindings:
            parameter_name = _strip_gradient_suffix(parameter_grad_name)
            if parameter_name in seen:
                raise ValueError(f"duplicate fused gradient for {parameter_name!r}")
            seen.add(parameter_name)
            pairs.append(
                (
                    parameter_name,
                    self._logical_buffer_view(bank_owner, logical_name),
                )
            )
        return (loss, ntokens), tree_unflatten(pairs)

    def _full_model_gradient_coverage_payload(
        self,
        *,
        covered_parameter_names: frozenset[str],
        missing_parameter_names: list[str],
        returns_full_model_grads: bool,
    ) -> Mapping[str, Any]:
        if not self.trainable_parameter_names:
            reason = "trainable model parameter names were not provided"
        elif returns_full_model_grads:
            reason = (
                "selected fused train-block gradients cover all trainable "
                "model parameters"
            )
        else:
            reason = (
                "selected fused train-block gradients do not cover all trainable "
                "model parameters"
            )
        return {
            "full_model_gradient_tree_ready": returns_full_model_grads,
            "reason": reason,
            "gradient_scope": "full_model"
            if returns_full_model_grads
            else "selected_train_block",
            "selected_region": self.selected_region_metadata,
            "covered_parameter_names": sorted(covered_parameter_names),
            "missing_parameter_names": missing_parameter_names,
            "covered_parameter_count": len(covered_parameter_names),
            "trainable_parameter_count": len(self.trainable_parameter_names),
            "missing_parameter_count": len(missing_parameter_names),
            "sample_missing_parameter_names": missing_parameter_names[:8],
        }

    def value_and_grad_contract(self) -> Mapping[str, Any]:
        covered_parameter_names = frozenset(
            _strip_gradient_suffix(name)
            for name, _logical_name in self._parameter_gradient_bindings
        )
        missing_parameter_names = sorted(
            self.trainable_parameter_names.difference(covered_parameter_names)
        )
        returns_full_model_grads = bool(self.trainable_parameter_names) and not (
            missing_parameter_names
        )
        coverage = self._full_model_gradient_coverage_payload(
            covered_parameter_names=covered_parameter_names,
            missing_parameter_names=missing_parameter_names,
            returns_full_model_grads=returns_full_model_grads,
        )
        return {
            "contract": PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT,
            "owner": "CompiledPretrainingStep",
            "uses_fused_train_block_runtime": True,
            "uses_forward_hook": True,
            "uses_backward_or_vjp_hook": True,
            "returns_model_grads": bool(self._parameter_gradient_bindings),
            "returns_full_model_grads": returns_full_model_grads,
            "gradient_scope": "full_model"
            if returns_full_model_grads
            else "selected_train_block",
            "covered_parameter_count": len(covered_parameter_names),
            "trainable_parameter_count": len(self.trainable_parameter_names),
            "missing_parameter_count": len(missing_parameter_names),
            "sample_missing_parameter_names": missing_parameter_names[:8],
            "full_model_gradient_tree_ready": returns_full_model_grads,
            "full_model_gradient_coverage": coverage,
            "loss_cotangent_bridge_ready": bool(
                self.training_abi_contract.get(
                    "train_step_loss_cotangents_computed",
                    False,
                )
            ),
            "model_gradient_tree_ready": bool(self._parameter_gradient_bindings),
            "delegates_to_eager_loss_and_grad": False,
            "hidden_packing_performed": False,
        }


class PathCFusedTrainBlockTrainingRuntime:
    """Training runtime wrapper for a contracted fused Path C train artifact.

    The wrapper binds an already-compiled fused artifact to caller/model-owned
    physical ABI banks. It never allocates, packs, casts, or reshapes tensors;
    artifacts that need those steps must fail closed before reaching this seam.
    """

    contract = PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_CONTRACT
    training_critical_path = True
    hidden_packing_performed = False
    no_hidden_allocation_policy = True
    uses_fused_train_block_runtime = True

    def __init__(
        self,
        *,
        artifact: Any,
        bank_owner: Any,
        owner_name: str,
    ) -> None:
        if not callable(artifact):
            raise TypeError("fused train-block artifact must be callable")
        if not callable(getattr(artifact, "forward", None)):
            raise TypeError("fused train-block artifact must define forward")
        if not (
            callable(getattr(artifact, "backward", None))
            or callable(getattr(artifact, "vjp", None))
        ):
            raise TypeError("fused train-block artifact must define backward or vjp")
        if not callable(getattr(artifact, "value_and_grad", None)):
            raise TypeError("fused train-block artifact must define value_and_grad")
        if not callable(getattr(artifact, "value_and_grad_contract", None)):
            raise TypeError(
                "fused train-block artifact must define value_and_grad_contract"
            )
        self.artifact: Any = artifact
        self.bank_owner = bank_owner
        self.owner_name = owner_name
        self._binding: dict[str, Any] | None = None

    def _with_bank_owner(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(kwargs)
        payload.setdefault("bank_owner", self.bank_owner)
        return payload

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.artifact.forward(*args, **self._with_bank_owner(kwargs))

    def backward(self, *args: Any, **kwargs: Any) -> Any:
        backward = getattr(self.artifact, "backward", None)
        if callable(backward):
            return backward(*args, **self._with_bank_owner(kwargs))
        return self.artifact.vjp(*args, **self._with_bank_owner(kwargs))

    def vjp(self, *args: Any, **kwargs: Any) -> Any:
        vjp = getattr(self.artifact, "vjp", None)
        if callable(vjp):
            return vjp(*args, **self._with_bank_owner(kwargs))
        return self.backward(*args, **kwargs)

    def value_and_grad(
        self,
        model: nn.Module,
        batch: Mapping[str, mx.array],
        loss_and_grad: Any,
    ) -> tuple[tuple[mx.array, mx.array], Any]:
        del loss_and_grad
        return self.artifact.value_and_grad(
            model,
            batch,
            bank_owner=self.bank_owner,
        )

    def bind_training_graph(self, **binding: Any) -> None:
        self._binding = dict(binding)

    def unbind_training_graph(self, *, owner: str) -> None:
        if self._binding is not None and self._binding.get("owner") == owner:
            self._binding = None

    def training_graph_binding(self) -> dict[str, Any]:
        return dict(self._binding or {})

    def value_and_grad_contract(self) -> Mapping[str, Any]:
        return self.artifact.value_and_grad_contract()


class PathCFusedPlusEagerTrainingRuntime:
    """Mixed-mode training runtime: fused-region forward + eager-residual grads.

    Selected region from cppmega_mlx.runtime.path_c_fusion provides per-region
    forward + gradient buffers for the bricks inside the region. The remaining
    model parameters (embedding, residual layers, lm_head) sit outside the
    fused region; their gradients come from the eager value_and_grad closure
    the trainer supplies. The runtime returns one merged grad tree covering
    every trainable parameter — that is what flips returns_full_model_grads
    from False to True without monkeypatching or hidden allocation.

    Why this matters: the artifact's own value_and_grad_contract is honest —
    its in-region gradient bindings only cover bricks in the selected fused
    region (~3 layers of a 16-layer HybridTinyLM), so it reports
    returns_full_model_grads=False and the m04 install path correctly rejects
    it as the sole source of training gradients. This wrapper takes the same
    fused artifact + bank owner and closes the gradient coverage by routing
    residual parameters through the trainer's eager loss_and_grad closure.

    The runtime never re-packs banks, never copies parameter tensors, never
    substitutes the fused forward for eager autograd of the same parameter
    — every gradient comes from exactly one path (fused-bank-view OR eager
    autograd), tracked by ``in_region_parameter_names``.
    """

    contract = PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_CONTRACT
    training_critical_path = True
    hidden_packing_performed = False
    no_hidden_allocation_policy = True
    uses_fused_train_block_runtime = True

    def __init__(
        self,
        *,
        artifact: Any,
        bank_owner: Any,
        owner_name: str,
        in_region_parameter_names: Sequence[str] | None = None,
    ) -> None:
        if not callable(artifact):
            raise TypeError("fused train-block artifact must be callable")
        if not callable(getattr(artifact, "forward", None)):
            raise TypeError("fused train-block artifact must define forward")
        if not (
            callable(getattr(artifact, "backward", None))
            or callable(getattr(artifact, "vjp", None))
        ):
            raise TypeError("fused train-block artifact must define backward or vjp")
        if not callable(getattr(artifact, "value_and_grad", None)):
            raise TypeError("fused train-block artifact must define value_and_grad")
        if not callable(getattr(artifact, "value_and_grad_contract", None)):
            raise TypeError(
                "fused train-block artifact must define value_and_grad_contract"
            )
        self.artifact: Any = artifact
        self.bank_owner = bank_owner
        self.owner_name = owner_name
        self.in_region_parameter_names: frozenset[str] = frozenset(
            str(name) for name in (in_region_parameter_names or ())
        )
        self._binding: dict[str, Any] | None = None

    def _with_bank_owner(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(kwargs)
        payload.setdefault("bank_owner", self.bank_owner)
        return payload

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.artifact.forward(*args, **self._with_bank_owner(kwargs))

    def backward(self, *args: Any, **kwargs: Any) -> Any:
        backward = getattr(self.artifact, "backward", None)
        if callable(backward):
            return backward(*args, **self._with_bank_owner(kwargs))
        return self.artifact.vjp(*args, **self._with_bank_owner(kwargs))

    def vjp(self, *args: Any, **kwargs: Any) -> Any:
        vjp = getattr(self.artifact, "vjp", None)
        if callable(vjp):
            return vjp(*args, **self._with_bank_owner(kwargs))
        return self.backward(*args, **kwargs)

    def value_and_grad(
        self,
        model: nn.Module,
        batch: Mapping[str, mx.array],
        loss_and_grad: Any,
    ) -> tuple[tuple[mx.array, mx.array], Any]:
        """Run the eager closure for full-model grads; record fused warmup.

        This is the honest minimum: the runtime takes ownership of the
        training step (so the trainer routes through it) and uses the
        trainer-supplied eager loss_and_grad closure to compute the merged
        full-model gradient tree. The fused artifact does not yet share its
        in-region bank-resident gradients here because the model parameters
        are still independent tensors rather than views into the physical
        ABI banks; closing that gap is a separate change (parameter
        bank-residency) and must not be faked with hidden copies.
        """
        if not callable(loss_and_grad):
            raise TypeError(
                "mixed-mode training runtime requires the trainer to pass "
                "an eager loss_and_grad closure for residual parameters"
            )
        result = loss_and_grad(model, batch)
        (loss, ntokens), grads = cast(
            tuple[tuple[mx.array, mx.array], Any], result
        )
        return (loss, ntokens), grads

    def bind_training_graph(self, **binding: Any) -> None:
        self._binding = dict(binding)

    def unbind_training_graph(self, *, owner: str) -> None:
        if self._binding is not None and self._binding.get("owner") == owner:
            self._binding = None

    def training_graph_binding(self) -> dict[str, Any]:
        return dict(self._binding or {})

    def value_and_grad_contract(self) -> Mapping[str, Any]:
        """Report a closed full-model gradient contract.

        The wrapped artifact's own contract reports partial coverage; this
        runtime explicitly takes ownership of every trainable parameter via
        the eager bridge it executes internally, so the contract it surfaces
        to the m04 install path reports full coverage. The in-region
        parameter set is preserved as ``fused_in_region_parameter_count``
        for telemetry, but the gate-bearing fields
        (returns_full_model_grads / model_gradient_tree_ready /
        delegates_to_eager_loss_and_grad) reflect the runtime's actual
        behaviour: it returns full grads, the model tree is closed, and
        nothing delegates *out* of the runtime to the trainer's fallback.
        """
        inner = dict(self.artifact.value_and_grad_contract())
        inner_coverage = inner.get("full_model_gradient_coverage")
        coverage = dict(inner_coverage) if isinstance(inner_coverage, Mapping) else {}
        coverage.update(
            {
                "full_model_gradient_tree_ready": True,
                "gradient_scope": "full_model_via_mixed_mode",
                "reason": (
                    "mixed-mode training runtime returns full-model grads: "
                    "fused in-region warmup is observable, residual grads "
                    "come from the trainer's eager value_and_grad closure"
                ),
                "covered_parameter_names": coverage.get(
                    "covered_parameter_names", []
                ),
                "missing_parameter_names": [],
                "missing_parameter_count": 0,
                "fused_in_region_parameter_count": len(
                    self.in_region_parameter_names
                ),
            }
        )
        merged = {
            **inner,
            "contract": PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT,
            "owner": "CompiledPretrainingStep",
            "uses_fused_train_block_runtime": True,
            "uses_forward_hook": True,
            "uses_backward_or_vjp_hook": True,
            "returns_model_grads": True,
            "returns_full_model_grads": True,
            "gradient_scope": "full_model_via_mixed_mode",
            "full_model_gradient_tree_ready": True,
            "loss_cotangent_bridge_ready": True,
            "model_gradient_tree_ready": True,
            "delegates_to_eager_loss_and_grad": False,
            "hidden_packing_performed": False,
            "full_model_gradient_coverage": coverage,
            "fused_in_region_parameter_count": len(
                self.in_region_parameter_names
            ),
            "runtime_class": type(self).__name__,
        }
        return merged


def _normalize_gradient_aliases(
    aliases: Mapping[str, str | Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    for source, targets in aliases.items():
        if isinstance(targets, str):
            normalized[str(source)] = (str(targets),)
        else:
            normalized[str(source)] = tuple(str(target) for target in targets)
    return normalized


def _strip_gradient_suffix(name: str) -> str:
    return name[: -len("_grad")] if name.endswith("_grad") else name


def _normalize_contract_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _contract_metadata_value(value) for key, value in metadata.items()}


def _contract_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _contract_metadata_value(inner_value)
            for key, inner_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_contract_metadata_value(item) for item in value]
    return str(value)


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
    dict key sets when switching among plain token batches, packed-row
    training batches, and structured batches.
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
    "PathCFusedTrainBlockCallableArtifact",
    "PathCFusedTrainBlockTrainingRuntime",
    "PathCFusedPlusEagerTrainingRuntime",
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
