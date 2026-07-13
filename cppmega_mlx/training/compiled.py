"""Compiled MLX pretraining step utilities.

The shape mirrors the current MLX-LM trainer pattern: compute loss with
nn.value_and_grad, update the optimizer, and explicitly mx.eval the
model/optimizer state.  The compiled path captures model.state and
optimizer.state so fixed-shape batches can be replayed efficiently.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Literal, Mapping, Sequence, TypeVar, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import average_gradients
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from cppmega_mlx.data.batch import (
    LMTokenBatch,
    ensure_lm_batch,
    prevalidated_batch_values,
)
from cppmega_mlx.runtime.path_c_physical_abi import (
    physical_abi_runtime_kernel_args,
)
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
PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS: Mapping[str, int] = {
    "path_c_run_backward": 1,
    "path_c_row_chunk_index": 0,
    "path_c_row_subchunk_index": 0,
    "path_c_backward_stage_index": 0,
}

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
    # A base (non-staged) artifact wraps ONE compiled TileLang/Metal kernel that
    # lowers the whole forward and the full recurrent backward internally and only
    # gates on ``path_c_run_backward``; it is dispatched grid-chunked (launched once
    # per forward / backward). Multi-kernel stage artifacts override this to True.
    generated_stage_artifact = False

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
        kernel_buffer_order: Sequence[str] | None = None,
        kernel_buffer_shapes: Mapping[str, Sequence[int]] | None = None,
        backward_gate_param: str | None = None,
        row_chunk_count: int | None = None,
        row_chunk_index_param: str | None = None,
        row_subchunk_count: int | None = None,
        row_subchunk_index_param: str | None = None,
        rows_per_kernel_launch: int | None = None,
        backward_stage_count: int | None = None,
        backward_stage_index_param: str | None = None,
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
        self.kernel_buffer_order = (
            tuple(str(name) for name in kernel_buffer_order)
            if kernel_buffer_order is not None
            else None
        )
        self.kernel_buffer_shapes = {
            str(name): tuple(int(dim) for dim in tuple(shape))
            for name, shape in (kernel_buffer_shapes or {}).items()
        }
        self.backward_gate_param = str(backward_gate_param or "path_c_run_backward")
        self.row_chunk_count = (
            max(1, int(row_chunk_count)) if row_chunk_count is not None else None
        )
        self.row_chunk_index_param = (
            str(row_chunk_index_param) if row_chunk_index_param else None
        )
        self.row_subchunk_count = (
            max(1, int(row_subchunk_count))
            if row_subchunk_count is not None
            else None
        )
        self.row_subchunk_index_param = (
            str(row_subchunk_index_param) if row_subchunk_index_param else None
        )
        self.rows_per_kernel_launch = (
            max(1, int(rows_per_kernel_launch))
            if rows_per_kernel_launch is not None
            else None
        )
        self.backward_stage_count = (
            max(1, int(backward_stage_count))
            if backward_stage_count is not None
            else None
        )
        self.backward_stage_index_param = (
            str(backward_stage_index_param) if backward_stage_index_param else None
        )
        self._cppmega_path_c_backward_gate_param = self.backward_gate_param
        self._cppmega_path_c_row_chunk_count = self.row_chunk_count
        self._cppmega_path_c_row_chunk_index_param = self.row_chunk_index_param
        self._cppmega_path_c_row_subchunk_count = self.row_subchunk_count
        self._cppmega_path_c_row_subchunk_index_param = self.row_subchunk_index_param
        self._cppmega_path_c_rows_per_kernel_launch = self.rows_per_kernel_launch
        self._cppmega_path_c_backward_stage_count = self.backward_stage_count
        self._cppmega_path_c_backward_stage_index_param = (
            self.backward_stage_index_param
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

    def _kernel_args_from_bank_owner(
        self,
        bank_owner: Any,
        *,
        kernel_scalar_params: Mapping[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        scalar_params = {
            str(name): value for name, value in (kernel_scalar_params or {}).items()
        }
        if self.kernel_buffer_order is not None:
            owner_buffers = self._bank_buffers(bank_owner)
            physical_abi_runtime_kernel_args(
                self.physical_abi_map,
                self.physical_abi_shapes,
                {
                    name: owner_buffers[name]
                    for name in self.physical_abi_shapes
                    if name in owner_buffers
                },
            )
            buffers = self._kernel_exact_buffers(owner_buffers)
            missing: list[str] = []
            args: list[Any] = []
            for name in self.kernel_buffer_order:
                if name in buffers:
                    args.append(buffers[name])
                    continue
                if name in scalar_params:
                    args.append(scalar_params[name])
                    continue
                if name in PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS:
                    args.append(PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS[name])
                    continue
                missing.append(name)
            if missing:
                raise ValueError(
                    "physical ABI runtime bindings are not executable: "
                    + "; ".join(
                        f"{name}: missing caller-owned kernel buffer"
                        for name in missing
                    )
                )
            return tuple(args)
        return physical_abi_runtime_kernel_args(
            self.physical_abi_map,
            self.physical_abi_shapes,
            self._bank_buffers(bank_owner),
        )

    def _kernel_exact_buffers(self, buffers: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.kernel_buffer_shapes:
            return buffers
        exact = dict(buffers)
        for name, expected_shape in self.kernel_buffer_shapes.items():
            value = exact.get(name)
            if value is None:
                continue
            actual_shape = tuple(int(dim) for dim in tuple(getattr(value, "shape", ())))
            if actual_shape == expected_shape:
                continue
            expected_size = 1
            for dim in expected_shape:
                expected_size *= int(dim)
            actual_size = int(getattr(value, "size", 0) or 0)
            if actual_size < expected_size:
                raise ValueError(
                    "physical ABI runtime bindings are not executable: "
                    f"{name}: caller-owned kernel buffer shape {actual_shape} "
                    f"has {actual_size} elements, expected at least "
                    f"{expected_shape} ({expected_size} elements)"
                )
            view = mx.reshape(value, (-1,))[:expected_size]
            if expected_shape != (expected_size,):
                view = mx.reshape(view, expected_shape)
            exact[name] = view
        return exact

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        bank_owner = kwargs.pop("bank_owner", None)
        kernel_scalar_params = kwargs.pop("kernel_scalar_params", None)
        if bank_owner is None:
            return self.kernel(*args, **kwargs)
        if args or kwargs:
            raise TypeError(
                "fused train-block artifact accepts only bank_owner-bound "
                "kernel dispatch"
            )
        return self.kernel(
            *self._kernel_args_from_bank_owner(
                bank_owner,
                kernel_scalar_params=kernel_scalar_params,
            )
        )

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
                    "replay_cotangent_boundary_available",
                    False,
                )
                or self.training_abi_contract.get(
                    "train_step_loss_cotangents_computed",
                    False,
                )
            ),
            "model_gradient_tree_ready": bool(self._parameter_gradient_bindings),
            "delegates_to_eager_loss_and_grad": False,
            "hidden_packing_performed": False,
            "generated_stage_artifact": False,
            "row_chunk_count": self.row_chunk_count,
            "row_chunk_index_param": self.row_chunk_index_param,
            "row_subchunk_count": self.row_subchunk_count,
            "row_subchunk_index_param": self.row_subchunk_index_param,
            "rows_per_kernel_launch": self.rows_per_kernel_launch,
            "backward_stage_count": self.backward_stage_count,
            "backward_stage_index_param": self.backward_stage_index_param,
        }


class PathCGeneratedStageTrainBlockCallableArtifact(PathCFusedTrainBlockCallableArtifact):
    """Contracted Path C artifact backed by generated forward/backward stages.

    This keeps the runtime contract at the train-block boundary: callers bind
    one physical ABI bank owner and select forward/backward with the same ABI
    scalar gate.  The implementation detail is that the descriptor generator
    emitted smaller phase kernels over the same bank map, avoiding a monolithic
    Metal function without falling back to a Python chain of logical kernels.
    """

    generated_stage_artifact = True

    def __init__(
        self,
        *,
        forward_kernel: Callable[..., Any],
        backward_kernel: Callable[..., Any] | None,
        forward_kernels: Sequence[Callable[..., Any]] | None = None,
        backward_kernels: Sequence[Callable[..., Any]] | None = None,
        physical_abi_map: Mapping[str, Any],
        physical_abi_shapes: Mapping[str, Any],
        training_abi_contract: Mapping[str, Any],
        parameter_gradient_aliases: Mapping[str, str | Sequence[str]] | None = None,
        trainable_parameter_names: Sequence[str] | None = None,
        selected_region_metadata: Mapping[str, Any] | None = None,
        forward_kernel_buffer_order: Sequence[str] | None = None,
        backward_kernel_buffer_order: Sequence[str] | None = None,
        forward_kernel_buffer_orders: Sequence[Sequence[str]] | None = None,
        backward_kernel_buffer_orders: Sequence[Sequence[str]] | None = None,
        forward_kernel_buffer_shapes: Mapping[str, Sequence[int]] | None = None,
        backward_kernel_buffer_shapes: Mapping[str, Sequence[int]] | None = None,
        forward_kernel_buffer_shapes_by_stage: Sequence[
            Mapping[str, Sequence[int]]
        ] | None = None,
        backward_kernel_buffer_shapes_by_stage: Sequence[
            Mapping[str, Sequence[int]]
        ] | None = None,
        backward_gate_param: str = "path_c_run_backward",
        row_chunk_count: int | None = None,
        row_chunk_index_param: str | None = None,
        row_subchunk_count: int | None = None,
        row_subchunk_index_param: str | None = None,
        rows_per_kernel_launch: int | None = None,
        forward_stage_row_launches: Sequence[Mapping[str, Any]] | None = None,
        backward_stage_row_launches: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        forward_kernel_sequence = tuple(forward_kernels or (forward_kernel,))
        if not forward_kernel_sequence:
            raise ValueError("generated Path C artifact needs a forward stage")
        backward_kernel_sequence = tuple(
            backward_kernels
            if backward_kernels is not None
            else ((backward_kernel,) if backward_kernel is not None else ())
        )
        raw_forward_orders = (
            forward_kernel_buffer_orders
            or (
                (tuple(str(name) for name in forward_kernel_buffer_order),)
                if forward_kernel_buffer_order is not None
                else (None,) * len(forward_kernel_sequence)
            )
        )
        forward_order_sequence = tuple(
            None if order is None else tuple(str(name) for name in order)
            for order in raw_forward_orders
        )
        if len(forward_order_sequence) != len(forward_kernel_sequence):
            raise ValueError("forward stage kernel/order count mismatch")
        raw_backward_orders = (
            backward_kernel_buffer_orders
            or (
                (tuple(str(name) for name in backward_kernel_buffer_order),)
                if backward_kernel_buffer_order is not None
                else (None,) * len(backward_kernel_sequence)
            )
        )
        backward_order_sequence = tuple(
            None if order is None else tuple(str(name) for name in order)
            for order in raw_backward_orders
        )
        if len(backward_order_sequence) != len(backward_kernel_sequence):
            raise ValueError("backward stage kernel/order count mismatch")
        raw_forward_shape_sequence = forward_kernel_buffer_shapes_by_stage
        if raw_forward_shape_sequence is None:
            raw_forward_shape_sequence = (
                (forward_kernel_buffer_shapes or {}),
            ) * len(forward_kernel_sequence)
        raw_forward_shape_sequence = tuple(raw_forward_shape_sequence)
        if len(raw_forward_shape_sequence) != len(forward_kernel_sequence):
            raise ValueError("forward stage kernel/shape count mismatch")
        raw_backward_shape_sequence = backward_kernel_buffer_shapes_by_stage
        if raw_backward_shape_sequence is None:
            raw_backward_shape_sequence = (
                (backward_kernel_buffer_shapes or {}),
            ) * len(backward_kernel_sequence)
        raw_backward_shape_sequence = tuple(raw_backward_shape_sequence)
        if len(raw_backward_shape_sequence) != len(backward_kernel_sequence):
            raise ValueError("backward stage kernel/shape count mismatch")
        super().__init__(
            kernel=forward_kernel_sequence[0],
            physical_abi_map=physical_abi_map,
            physical_abi_shapes=physical_abi_shapes,
            training_abi_contract={
                **dict(training_abi_contract),
                "generated_stage_artifact": True,
            },
            parameter_gradient_aliases=parameter_gradient_aliases,
            trainable_parameter_names=trainable_parameter_names,
            selected_region_metadata=selected_region_metadata,
            kernel_buffer_order=forward_order_sequence[0],
        )
        self.forward_stages = tuple(
            PathCFusedTrainBlockCallableArtifact(
                kernel=kernel,
                physical_abi_map=physical_abi_map,
                physical_abi_shapes=physical_abi_shapes,
                training_abi_contract=self.training_abi_contract,
                parameter_gradient_aliases=parameter_gradient_aliases,
                trainable_parameter_names=trainable_parameter_names,
                selected_region_metadata=selected_region_metadata,
                kernel_buffer_order=order,
                kernel_buffer_shapes=shape_map,
            )
            for kernel, order, shape_map in zip(
                forward_kernel_sequence,
                forward_order_sequence,
                raw_forward_shape_sequence,
                strict=True,
            )
        )
        self.forward_stage = self.forward_stages[0]
        self.backward_stages = tuple(
            PathCFusedTrainBlockCallableArtifact(
                kernel=kernel,
                physical_abi_map=physical_abi_map,
                physical_abi_shapes=physical_abi_shapes,
                training_abi_contract=self.training_abi_contract,
                parameter_gradient_aliases=parameter_gradient_aliases,
                trainable_parameter_names=trainable_parameter_names,
                selected_region_metadata=selected_region_metadata,
                kernel_buffer_order=order,
                kernel_buffer_shapes=shape_map,
            )
            for kernel, order, shape_map in zip(
                backward_kernel_sequence,
                backward_order_sequence,
                raw_backward_shape_sequence,
                strict=True,
            )
        )
        self.backward_stage = self.backward_stages[0] if self.backward_stages else None
        self.backward_gate_param = str(backward_gate_param or "path_c_run_backward")
        self.row_chunk_count = (
            max(1, int(row_chunk_count)) if row_chunk_count is not None else None
        )
        self.row_chunk_index_param = (
            str(row_chunk_index_param) if row_chunk_index_param else None
        )
        self.row_subchunk_count = (
            max(1, int(row_subchunk_count))
            if row_subchunk_count is not None
            else None
        )
        self.row_subchunk_index_param = (
            str(row_subchunk_index_param) if row_subchunk_index_param else None
        )
        self.rows_per_kernel_launch = (
            max(1, int(rows_per_kernel_launch))
            if rows_per_kernel_launch is not None
            else None
        )
        self.forward_stage_row_launches = self._normalise_stage_row_launches(
            forward_stage_row_launches,
            len(forward_kernel_sequence),
        )
        self.backward_stage_row_launches = self._normalise_stage_row_launches(
            backward_stage_row_launches,
            len(backward_kernel_sequence),
        )
        self.forward_kernel_buffer_shapes_by_stage = tuple(
            {
                str(name): tuple(int(dim) for dim in tuple(shape))
                for name, shape in raw_shapes.items()
            }
            for raw_shapes in raw_forward_shape_sequence
        )
        self.forward_kernel_buffer_shapes = self.forward_kernel_buffer_shapes_by_stage[0]
        self.backward_kernel_buffer_shapes_by_stage = tuple(
            {
                str(name): tuple(int(dim) for dim in tuple(shape))
                for name, shape in raw_shapes.items()
            }
            for raw_shapes in raw_backward_shape_sequence
        )
        self.backward_kernel_buffer_shapes = (
            self.backward_kernel_buffer_shapes_by_stage[0]
            if self.backward_kernel_buffer_shapes_by_stage
            else {}
        )

    def _run_backward_from_scalars(
        self,
        kernel_scalar_params: Mapping[str, Any] | None,
    ) -> bool:
        scalar_params = kernel_scalar_params or {}
        raw_gate = scalar_params.get(
            self.backward_gate_param,
            PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS.get(self.backward_gate_param, 1),
        )
        try:
            return int(raw_gate) == 1
        except Exception:
            return bool(raw_gate)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        bank_owner = kwargs.get("bank_owner", None)
        kernel_scalar_params = kwargs.get("kernel_scalar_params", None)
        raw_max_stages = kwargs.get("max_stages", None)
        max_stages: int | None = None
        if raw_max_stages is not None:
            max_stages = max(0, int(raw_max_stages))
        if bank_owner is None:
            stage_kwargs = {
                key: value for key, value in kwargs.items() if key != "max_stages"
            }
            return self.forward_stage.forward(*args, **stage_kwargs)
        run_backward = self._run_backward_from_scalars(kernel_scalar_params)
        stages = self.backward_stages if run_backward else self.forward_stages
        if run_backward and not self.backward_stages:
            raise ValueError("generated Path C artifact has no backward stage")
        if max_stages is not None:
            stages = stages[:max_stages]
        stage_row_launches = (
            self.backward_stage_row_launches
            if run_backward
            else self.forward_stage_row_launches
        )
        if max_stages is not None:
            stage_row_launches = stage_row_launches[:max_stages]
        stage_base_kwargs = {
            key: value for key, value in kwargs.items() if key != "max_stages"
        }
        result: Any = None
        for index, (stage, stage_row_launch) in enumerate(
            zip(stages, stage_row_launches, strict=True)
        ):
            stage_shapes = (
                self.backward_kernel_buffer_shapes_by_stage[index]
                if run_backward
                else self.forward_kernel_buffer_shapes_by_stage[index]
            )
            for stage_scalar_params in self._stage_scalar_param_variants(
                kernel_scalar_params,
                stage_row_launch,
            ):
                stage_kwargs = dict(stage_base_kwargs)
                stage_kwargs["kernel_scalar_params"] = stage_scalar_params
                if stage_shapes:
                    stage_kwargs["bank_owner"] = self._stage_exact_bank_owner(
                        bank_owner,
                        stage_shapes,
                    )
                try:
                    result = stage.forward(*args, **stage_kwargs)
                except Exception as exc:
                    stage_kind = "backward" if run_backward else "forward"
                    raise RuntimeError(
                        f"generated Path C {stage_kind} stage {index} failed"
                    ) from exc
        return result

    def _normalise_stage_row_launches(
        self,
        raw_stage_row_launches: Sequence[Mapping[str, Any]] | None,
        stage_count: int,
    ) -> tuple[dict[str, Any], ...]:
        default = {
            "row_chunk_count": self.row_chunk_count,
            "row_chunk_index_param": self.row_chunk_index_param,
            "row_subchunk_count": self.row_subchunk_count,
            "row_subchunk_index_param": self.row_subchunk_index_param,
            "rows_per_kernel_launch": self.rows_per_kernel_launch,
        }
        if raw_stage_row_launches is None:
            return tuple(dict(default) for _ in range(stage_count))
        specs = tuple(raw_stage_row_launches)
        if len(specs) != stage_count:
            raise ValueError("stage row launch spec/kernel count mismatch")
        normalised: list[dict[str, Any]] = []
        for spec in specs:
            merged = dict(default)
            for key, value in dict(spec).items():
                if value is not None:
                    merged[str(key)] = value
            for int_key in (
                "row_chunk_count",
                "row_subchunk_count",
                "rows_per_kernel_launch",
            ):
                if merged.get(int_key) is not None:
                    merged[int_key] = max(1, int(merged[int_key]))
            for str_key in ("row_chunk_index_param", "row_subchunk_index_param"):
                if merged.get(str_key):
                    merged[str_key] = str(merged[str_key])
                else:
                    merged[str_key] = None
            normalised.append(merged)
        return tuple(normalised)

    def _stage_scalar_param_variants(
        self,
        kernel_scalar_params: Mapping[str, Any] | None,
        stage_row_launch: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        base_params = dict(kernel_scalar_params or {})
        artifact_subchunk_param = self.row_subchunk_index_param
        stage_subchunk_param = stage_row_launch.get("row_subchunk_index_param")
        if not artifact_subchunk_param or not stage_subchunk_param:
            return (base_params,)
        artifact_rows = self.rows_per_kernel_launch
        stage_rows = stage_row_launch.get("rows_per_kernel_launch")
        if artifact_rows is None or stage_rows is None:
            return (base_params,)
        artifact_rows = max(1, int(artifact_rows))
        stage_rows = max(1, int(stage_rows))
        artifact_subchunk = int(base_params.get(artifact_subchunk_param, 0))
        if stage_rows == artifact_rows:
            if stage_subchunk_param != artifact_subchunk_param:
                base_params[str(stage_subchunk_param)] = artifact_subchunk
            return (base_params,)
        if stage_rows > artifact_rows or artifact_rows % stage_rows != 0:
            raise ValueError(
                "stage rows_per_kernel_launch must divide the artifact "
                "coordinator rows_per_kernel_launch"
            )
        refine_factor = artifact_rows // stage_rows
        stage_subchunk_count = stage_row_launch.get("row_subchunk_count")
        variants: list[dict[str, Any]] = []
        for offset in range(refine_factor):
            fine_subchunk = artifact_subchunk * refine_factor + offset
            if (
                stage_subchunk_count is not None
                and fine_subchunk >= int(stage_subchunk_count)
            ):
                continue
            params = dict(base_params)
            params[str(stage_subchunk_param)] = fine_subchunk
            stage_chunk_param = stage_row_launch.get("row_chunk_index_param")
            if stage_chunk_param and self.row_chunk_index_param:
                params[str(stage_chunk_param)] = int(
                    base_params.get(self.row_chunk_index_param, 0)
                )
            variants.append(params)
        return tuple(variants) or (base_params,)

    def _stage_exact_bank_owner(
        self,
        bank_owner: Any,
        stage_shapes: Mapping[str, tuple[int, ...]],
    ) -> Mapping[str, Any]:
        buffers = dict(self._bank_buffers(bank_owner))
        physical_banks = set(self.physical_abi_shapes)
        for name, expected_shape in stage_shapes.items():
            if name in physical_banks:
                continue
            value = buffers.get(name)
            if value is None:
                continue
            actual_shape = tuple(int(dim) for dim in tuple(getattr(value, "shape", ())))
            if actual_shape == expected_shape:
                continue
            expected_size = 1
            for dim in expected_shape:
                expected_size *= int(dim)
            actual_size = int(getattr(value, "size", 0) or 0)
            if actual_size < expected_size:
                continue
            view = mx.reshape(value, (-1,))[:expected_size]
            if expected_shape != (expected_size,):
                view = mx.reshape(view, expected_shape)
            buffers[name] = view
        return buffers

    def value_and_grad_contract(self) -> Mapping[str, Any]:
        payload = dict(super().value_and_grad_contract())
        payload["generated_stage_artifact"] = True
        payload["generated_stage_count"] = len(self.forward_stages) + len(
            self.backward_stages
        )
        payload["row_chunk_count"] = self.row_chunk_count
        payload["row_chunk_index_param"] = self.row_chunk_index_param
        payload["row_subchunk_count"] = self.row_subchunk_count
        payload["row_subchunk_index_param"] = self.row_subchunk_index_param
        payload["rows_per_kernel_launch"] = self.rows_per_kernel_launch
        payload["forward_stage_row_launches"] = list(self.forward_stage_row_launches)
        payload["backward_stage_row_launches"] = list(self.backward_stage_row_launches)
        payload["delegates_to_eager_loss_and_grad"] = False
        return payload


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
        in_region_parameter_bank_aliases: Mapping[str, Mapping[str, Any]] | None = None,
        model_bank_sync_callable: Callable[[], Mapping[str, Any]] | None = None,
        fused_suffix_loss_fn: Callable[
            [nn.Module, Mapping[str, Any]], tuple[mx.array, mx.array]
        ] | None = None,
        fused_replay_loss_fn: Callable[
            [nn.Module, Mapping[str, Any]], tuple[mx.array, mx.array]
        ] | None = None,
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
        self.in_region_parameter_bank_aliases: dict[str, dict[str, Any]] = {
            str(name): {str(k): v for k, v in dict(info).items()}
            for name, info in (in_region_parameter_bank_aliases or {}).items()
        }
        if self.in_region_parameter_bank_aliases:
            # If aliases are supplied, the parameter-name set must agree.
            alias_names = frozenset(self.in_region_parameter_bank_aliases.keys())
            if self.in_region_parameter_names and (
                self.in_region_parameter_names != alias_names
            ):
                raise ValueError(
                    "in_region_parameter_names and "
                    "in_region_parameter_bank_aliases name sets disagree"
                )
            self.in_region_parameter_names = alias_names
        self.model_bank_sync_callable = (
            model_bank_sync_callable if callable(model_bank_sync_callable) else None
        )
        self.fused_suffix_loss_fn = (
            fused_suffix_loss_fn if callable(fused_suffix_loss_fn) else None
        )
        self.fused_replay_loss_fn = (
            fused_replay_loss_fn if callable(fused_replay_loss_fn) else None
        )
        self._binding: dict[str, Any] | None = None
        self.last_fused_warmup_payload: dict[str, Any] = {
            "attempted": False,
            "completed": False,
            "elapsed_ns": 0,
            "error": None,
        }
        self.last_fused_value_and_grad_payload: dict[str, Any] = {
            "attempted": False,
            "completed": False,
            "elapsed_ns": 0,
            "merged_parameter_count": 0,
            "merged_parameter_names": (),
            "missing_parameter_names": (),
            "bank_sync_status": None,
            "error": None,
            "suffix_bypass_active": False,
            "replay_cotangent_bridge_active": False,
        }
        self.last_parameter_bank_bind_report: dict[str, Any] | None = None

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

    def _merge_fused_grads_into_eager(
        self,
        eager_grads: Any,
        fused_grads: Any,
    ) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
        """Replace in-region grad entries in the eager tree with bank views.

        For every parameter in ``in_region_parameter_bank_aliases``, the
        bank-resident gradient view from ``fused_grads`` overwrites the
        corresponding entry in the eager gradient tree. Parameters that
        the fused tree does not carry (missing) are reported in the return
        tuple so the caller can record them without silently dropping
        gradient coverage.
        """

        if not self.in_region_parameter_bank_aliases:
            return eager_grads, (), ()
        flat_eager = dict(tree_flatten(eager_grads))
        flat_fused = dict(tree_flatten(fused_grads))
        merged_names: list[str] = []
        missing_names: list[str] = []
        for parameter_name in sorted(self.in_region_parameter_bank_aliases):
            fused_value = flat_fused.get(parameter_name)
            if not isinstance(fused_value, mx.array):
                missing_names.append(parameter_name)
                continue
            flat_eager[parameter_name] = fused_value
            merged_names.append(parameter_name)
        merged_grads = tree_unflatten(sorted(flat_eager.items()))
        return merged_grads, tuple(merged_names), tuple(missing_names)

    def value_and_grad(
        self,
        model: nn.Module,
        batch: Mapping[str, mx.array],
        loss_and_grad: Any,
    ) -> tuple[tuple[mx.array, mx.array], Any]:
        """Run fused + eager value_and_grad with the strongest available bridge.

        Behaviour depends on the runtime's installed state:

        * **Suffix-bypass mode (fused_suffix_loss_fn + aliases set)** —
          short-circuit eager autograd for the entire in-region suffix.
          The runtime builds ``nn.value_and_grad(model, fused_suffix_loss_fn)``
          and runs it once: the prefix (embedding + layers before the
          fused region) runs eagerly, then the fused custom-function
          executes the TileLang artifact and exposes bank-resident
          cotangents to MLX autograd. Because the suffix never enters
          the autograd graph, the eager forward / backward cost drops to
          the prefix only — this is the structural Path C compute /
          memory win. The trainer-supplied ``loss_and_grad`` closure is
          intentionally ignored in this mode: every gradient (prefix
          plus in-region) comes from one autograd pass that consumes the
          fused suffix as a custom function.

        * **Merged mode (aliases set, no suffix bypass)** — sync
          in-region parameter tensors back into the model-owned
          physical-ABI bank slots, launch the fused TileLang
          ``artifact.value_and_grad`` (forward + backward in one shot,
          returning a bank-resident gradient tree), then run the
          trainer's eager closure to produce a full-model gradient
          tree, and overwrite the in-region entries of the eager tree
          with the bank-resident fused gradients. The returned ``loss``
          / ``ntokens`` come from the eager closure (so trainer-side
          loss accounting is unchanged). Telemetry lands in
          ``self.last_fused_warmup_payload`` and
          ``self.last_fused_value_and_grad_payload``.

        * **Warmup mode (no aliases)** — keep the previous behaviour:
          run the fused ``artifact.forward(bank_owner=…)`` as a warmup
          pass (so the JIT cache stays warm and downstream telemetry
          can measure the kernel cost) and delegate the gradient tree
          to the eager closure. The fused grads are not merged because
          the parameters are still independent tensors. This mode is
          the honest fallback when bank-residency binding has not been
          installed; it does not pretend the runtime closed the
          gradient-coverage gap.
        """
        if not callable(loss_and_grad):
            raise TypeError(
                "mixed-mode training runtime requires the trainer to pass "
                "an eager loss_and_grad closure for residual parameters"
            )

        replay_bridge = bool(
            self.in_region_parameter_bank_aliases
            and self.fused_replay_loss_fn is not None
        )
        suffix_bypass = bool(
            self.in_region_parameter_bank_aliases
            and self.fused_suffix_loss_fn is not None
        )
        if replay_bridge or suffix_bypass:
            bank_sync_status: dict[str, Any] | None = None
            if self.model_bank_sync_callable is not None:
                try:
                    raw_status = self.model_bank_sync_callable()
                    bank_sync_status = (
                        dict(raw_status) if isinstance(raw_status, Mapping) else None
                    )
                except Exception as exc:
                    bank_sync_status = {
                        "status": "error",
                        "error": repr(exc),
                    }

            vg_payload: dict[str, Any] = {
                "attempted": True,
                "completed": False,
                "elapsed_ns": 0,
                "merged_parameter_count": len(
                    self.in_region_parameter_bank_aliases
                ),
                "merged_parameter_names": tuple(
                    sorted(self.in_region_parameter_bank_aliases)
                ),
                "missing_parameter_names": (),
                "bank_sync_status": bank_sync_status,
                "error": None,
                "suffix_bypass_active": bool(suffix_bypass and not replay_bridge),
                "replay_cotangent_bridge_active": bool(replay_bridge),
            }
            try:
                vg_start = time.perf_counter_ns()
                fused_loss_fn = (
                    self.fused_replay_loss_fn
                    if replay_bridge
                    else self.fused_suffix_loss_fn
                )
                if fused_loss_fn is None:  # pragma: no cover - guarded above
                    raise RuntimeError("fused bridge missing loss function")
                fused_value_and_grad = nn.value_and_grad(model, fused_loss_fn)
                (loss, ntokens), grads = fused_value_and_grad(model, batch)
                vg_payload["elapsed_ns"] = (
                    time.perf_counter_ns() - vg_start
                )
                vg_payload["completed"] = True
                self.last_fused_value_and_grad_payload = vg_payload
                # Warmup payload is meaningless in bypass mode (no separate
                # warmup launch happens); keep the default sentinel so
                # receipts can distinguish the modes.
                return (loss, ntokens), grads
            except Exception as exc:
                vg_payload["error"] = repr(exc)
                self.last_fused_value_and_grad_payload = vg_payload
                raise

        # Bank residency alone is not enough to consume fused gradients:
        # the fused train-block also needs per-batch runtime inputs
        # (hidden_entry, target_ids, target_mask) written into the ABI banks.
        # Without the suffix-bypass custom-function surface, the artifact's
        # value_and_grad would run against stale/zero runtime inputs and would
        # silently overwrite correct eager grads with bogus bank values. Fail
        # closed to the original warmup+eager behavior unless suffix-bypass
        # handled the step above.
        merged_mode = False

        # In merged mode, push optimizer-replaced parameters into bank slots
        # before the fused kernel reads them. The first step's sync is a no-op
        # (parameters were already written into the bank at bind time), but
        # subsequent steps need the sync because optimizer.update replaced the
        # bank-view attribute with a fresh tensor.
        bank_sync_status: dict[str, Any] | None = None
        if merged_mode and self.model_bank_sync_callable is not None:
            try:
                raw_status = self.model_bank_sync_callable()
                bank_sync_status = (
                    dict(raw_status) if isinstance(raw_status, Mapping) else None
                )
            except Exception as exc:
                bank_sync_status = {
                    "status": "error",
                    "error": repr(exc),
                }

        warmup_payload: dict[str, Any] = {
            "attempted": False,
            "completed": False,
            "elapsed_ns": 0,
            "error": None,
        }
        vg_payload: dict[str, Any] = {
            "attempted": False,
            "completed": False,
            "elapsed_ns": 0,
            "merged_parameter_count": 0,
            "merged_parameter_names": (),
            "missing_parameter_names": (),
            "bank_sync_status": bank_sync_status,
            "error": None,
            "suffix_bypass_active": False,
            "replay_cotangent_bridge_active": False,
        }

        fused_grads: Any | None = None
        if merged_mode:
            # Merged mode: run fused.value_and_grad so the artifact populates
            # in-region grad slots in the bank, then bind them as views into
            # the eager grad tree below.
            try:
                vg_payload["attempted"] = True
                vg_start = time.perf_counter_ns()
                (_fused_loss, _fused_ntokens), fused_grads = (
                    self.artifact.value_and_grad(
                        model, batch, bank_owner=self.bank_owner
                    )
                )
                mx.eval(*self.bank_owner.buffers.values())
                vg_payload["elapsed_ns"] = (
                    time.perf_counter_ns() - vg_start
                )
                vg_payload["completed"] = True
            except Exception as exc:
                vg_payload["error"] = repr(exc)
                fused_grads = None
        else:
            try:
                warmup_payload["attempted"] = True
                warmup_start = time.perf_counter_ns()
                self.artifact.forward(bank_owner=self.bank_owner)
                mx.eval(*self.bank_owner.buffers.values())
                warmup_payload["elapsed_ns"] = (
                    time.perf_counter_ns() - warmup_start
                )
                warmup_payload["completed"] = True
            except Exception as exc:
                warmup_payload["error"] = repr(exc)

        self.last_fused_warmup_payload = warmup_payload

        # Eager closure runs for the full model. In merged mode its in-region
        # grad entries are overwritten by the fused bank views below.
        result = loss_and_grad(model, batch)
        (loss, ntokens), grads = cast(
            tuple[tuple[mx.array, mx.array], Any], result
        )

        if merged_mode and fused_grads is not None:
            try:
                merged_grads, merged_names, missing_names = (
                    self._merge_fused_grads_into_eager(grads, fused_grads)
                )
                grads = merged_grads
                vg_payload["merged_parameter_count"] = len(merged_names)
                vg_payload["merged_parameter_names"] = merged_names
                vg_payload["missing_parameter_names"] = missing_names
            except Exception as exc:
                vg_payload["error"] = repr(exc)
        self.last_fused_value_and_grad_payload = vg_payload

        return (loss, ntokens), grads

    def bind_training_graph(self, **binding: Any) -> None:
        self._binding = dict(binding)

    def unbind_training_graph(self, *, owner: str) -> None:
        if self._binding is not None and self._binding.get("owner") == owner:
            self._binding = None

    def training_graph_binding(self) -> dict[str, Any]:
        return dict(self._binding or {})

    def value_and_grad_contract(self) -> Mapping[str, Any]:
        """Report whether the fused train block owns the training gradient path."""
        inner = dict(self.artifact.value_and_grad_contract())
        inner_coverage = inner.get("full_model_gradient_coverage")
        coverage = dict(inner_coverage) if isinstance(inner_coverage, Mapping) else {}
        bank_residency_ready = bool(self.in_region_parameter_bank_aliases)
        suffix_bypass_available = bool(
            self.fused_suffix_loss_fn is not None
            and self.in_region_parameter_bank_aliases
        )
        replay_cotangent_bridge_available = bool(
            self.fused_replay_loss_fn is not None
            and self.in_region_parameter_bank_aliases
        )
        fused_bridge_available = (
            replay_cotangent_bridge_available or suffix_bypass_available
        )
        returns_full_model_grads = fused_bridge_available
        gradient_scope = (
            "full_model_via_fused_replay_cotangent_bridge"
            if replay_cotangent_bridge_available
            else (
            "full_model_via_fused_suffix_bypass"
            if suffix_bypass_available
            else "selected_train_block_warmup_only"
            )
        )
        reason = (
            "mixed-mode training runtime returns full-model grads via "
            "replay/cotangent bridge: the generated TileLang artifact owns "
            "only the fused M/R/A block, while MLX eager prefix/suffix "
            "autograd supplies boundary cotangents and residual gradients"
            if replay_cotangent_bridge_available
            else (
            "mixed-mode training runtime returns full-model grads via "
            "suffix-bypass: in-region parameters and hidden-entry cotangents "
            "come from the fused custom-function VJP, residual/prefix grads "
            "come from MLX eager autograd"
            if suffix_bypass_available
            else (
                "mixed-mode training runtime returns full-model grads from "
                "the trainer's eager value_and_grad closure; bank-resident "
                "fused gradients are disabled because no verified runtime "
                "input writer / suffix-bypass bridge is active"
                if bank_residency_ready
                else (
                    "mixed-mode training runtime returns full-model grads: "
                    "fused in-region warmup is observable, residual grads "
                    "come from the trainer's eager value_and_grad closure"
                )
            )
            )
        )
        if not fused_bridge_available:
            reason = (
                "fused train-block gradient overlay is disabled; the runtime "
                "can run as warmup only, while the actual full-model gradients "
                "still come from the trainer's eager value_and_grad closure, "
                "so this is not a Path C training critical path"
            )
        overlay_parameter_count = (
            len(self.in_region_parameter_bank_aliases)
            if fused_bridge_available
            else 0
        )
        overlay_parameter_names = (
            tuple(sorted(self.in_region_parameter_bank_aliases))
            if fused_bridge_available
            else ()
        )
        coverage.update(
            {
                "full_model_gradient_tree_ready": returns_full_model_grads,
                "gradient_scope": gradient_scope,
                "reason": reason,
                "covered_parameter_names": coverage.get(
                    "covered_parameter_names", []
                ),
                "missing_parameter_names": (
                    []
                    if fused_bridge_available
                    else coverage.get("missing_parameter_names", [])
                ),
                "missing_parameter_count": (
                    0
                    if fused_bridge_available
                    else coverage.get("missing_parameter_count", 0)
                ),
                "fused_in_region_parameter_count": len(
                    self.in_region_parameter_names
                ),
                "parameter_bank_residency_active": bank_residency_ready,
                "bank_grad_overlay_active": fused_bridge_available,
                "merged_bank_resident_parameter_count": overlay_parameter_count,
                "merged_bank_resident_parameter_names": overlay_parameter_names,
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
            "returns_full_model_grads": returns_full_model_grads,
            "gradient_scope": gradient_scope,
            "full_model_gradient_tree_ready": returns_full_model_grads,
            "loss_cotangent_bridge_ready": fused_bridge_available,
            "model_gradient_tree_ready": returns_full_model_grads,
            "delegates_to_eager_loss_and_grad": not fused_bridge_available,
            "hidden_packing_performed": False,
            "full_model_gradient_coverage": coverage,
            "fused_in_region_parameter_count": len(
                self.in_region_parameter_names
            ),
            "parameter_bank_residency_active": bool(
                self.in_region_parameter_bank_aliases
            ),
            "bank_grad_overlay_active": fused_bridge_available,
            "merged_bank_resident_parameter_count": overlay_parameter_count,
            "suffix_bypass_available": suffix_bypass_available,
            "replay_cotangent_bridge_available": replay_cotangent_bridge_available,
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
        # Records whether the fused Path C runtime fell back to the eager
        # loss_and_grad path (CUDA safety net; see ``_eager_step``).
        self._path_c_runtime_fallback_count: int = 0
        self._path_c_runtime_fallback_error: str | None = None
        # Latched True once the fused runtime times out, so the remaining steps
        # skip it entirely and use the eager baseline.
        self._path_c_runtime_disabled: bool = False
        if path_c_training_runtime is not None:
            self.attach_path_c_training_runtime(path_c_training_runtime)

    def __call__(
        self,
        batch: LMTokenBatch | Mapping[str, mx.array] | mx.array,
    ) -> PretrainingMetrics:
        self.model.train()
        batch_dict = normalize_compiled_batch(batch)
        batch_is_prevalidated = False
        if self.compile:
            validator = getattr(self.model, "validate_compiled_batch", None)
            if callable(validator):
                validator(batch_dict)
                batch_is_prevalidated = True
            self._check_compiled_batch_signature(batch_dict)
        pending_microbatches = self._pending_microbatches + 1
        do_update = pending_microbatches == self.grad_accum_steps

        start = time.perf_counter()
        if self.compile:
            validation_context = (
                prevalidated_batch_values()
                if batch_is_prevalidated
                else nullcontext()
            )
            with validation_context:
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

    def _run_fused_value_and_grad_guarded(
        self,
        runtime: PathCTrainingRuntime,
        loss_batch: Mapping[str, mx.array],
    ) -> tuple[tuple[mx.array, mx.array], Any] | None:
        """Run the fused runtime value_and_grad under an exception (+ time) guard.

        Returns the ``((loss, ntokens), grads)`` tuple on success, or ``None`` to
        signal the caller should fall back to the eager loss_and_grad path.

        RULE #1 (fail fast, fail loud): a fused-runtime crash is a *real bug* in
        the selected Path C kernel / bridge. By DEFAULT this method therefore
        RAISES with where+what instead of silently substituting an eager result
        that looks fine but hides the broken fused path. Surfacing the bug is the
        priority.

        The eager degrade is retained ONLY as an explicit, opt-in bisection
        escape, enabled by setting ``MLX_PATH_C_FUSED_ALLOW_EAGER_DEGRADE`` to a
        truthy value (``1``/``true``/``yes``/``on``). When opted in, a fused
        crash is *loudly* degraded to eager — a ``RuntimeWarning`` is emitted and
        the receipt fields (``_path_c_runtime_fallback_count`` /
        ``_path_c_runtime_fallback_error``, surfaced as
        ``path_c_fused_runtime_evidence`` in the train-step report) record that
        the run used a degraded path. This is for deliberate bisection only; it
        is never the silent default.

        An OPTIONAL wall-clock watchdog is enabled only when
        ``MLX_PATH_C_FUSED_STEP_TIMEOUT_S`` is set to a positive number. It runs
        the fused call on a daemon worker thread and, on timeout, treats the
        runaway fused kernel exactly like a crash: RAISE by default, or (with the
        opt-in degrade env) permanently disable the fused runtime and fall back
        to eager. The watchdog is opt-in because a CUDA kernel that is launched
        but does not return cannot be cancelled in-process: the abandoned worker
        keeps the runaway kernel resident on the GPU. The clean default therefore
        runs the fused call inline (no leaked kernel) and relies on the exception
        guard. The first failure / timeout is recorded for reporting.
        """

        import os
        import warnings

        degrade_env = os.environ.get("MLX_PATH_C_FUSED_ALLOW_EAGER_DEGRADE", "")
        allow_eager_degrade = degrade_env.strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        timeout_env = os.environ.get("MLX_PATH_C_FUSED_STEP_TIMEOUT_S")
        timeout_s = 0.0
        if timeout_env:
            try:
                timeout_s = float(timeout_env)
            except (TypeError, ValueError):
                timeout_s = 0.0

        error: BaseException | None = None
        if timeout_s > 0:
            import threading

            result_box: dict[str, Any] = {}

            def _worker() -> None:
                try:
                    result_box["value"] = runtime.value_and_grad(
                        self.model,
                        loss_batch,
                        self._loss_and_grad,
                    )
                except BaseException as exc:  # noqa: BLE001 - to main thread
                    result_box["error"] = exc

            worker = threading.Thread(
                target=_worker, name="path-c-fused-value-and-grad", daemon=True
            )
            worker.start()
            worker.join(timeout=timeout_s)

            if worker.is_alive():
                # Timed out — a runaway / impractically slow fused kernel. This is
                # a real bug in the fused Path C kernel/bridge.
                timeout_detail = (
                    f"fused value_and_grad exceeded {timeout_s:g}s budget"
                )
                if self._path_c_runtime_fallback_error is None:
                    self._path_c_runtime_fallback_error = timeout_detail
                self._path_c_runtime_fallback_count += 1
                if not allow_eager_degrade:
                    # RULE #1: surface the runaway fused kernel instead of
                    # silently degrading to eager.
                    raise RuntimeError(
                        "Path C fused direct-chain runtime.value_and_grad timed out "
                        f"in CompiledPretrainingStep._run_fused_value_and_grad_guarded "
                        f"({timeout_detail}). Refusing to silently fall back to the "
                        "eager loss_and_grad path (RULE #1) — this points at a runaway "
                        "or pathologically slow fused Path C kernel/bridge. Set "
                        "MLX_PATH_C_FUSED_ALLOW_EAGER_DEGRADE=1 to opt into a loud "
                        "eager degrade for bisection only."
                    )
                # Opt-in bisection escape: abandon the worker (daemon) and
                # permanently disable the fused runtime so every subsequent step
                # uses the eager baseline. LOUD (warning + receipt fields).
                self._path_c_runtime_disabled = True
                warnings.warn(
                    "Path C fused direct-chain runtime.value_and_grad exceeded the "
                    f"{timeout_s:g}s budget; MLX_PATH_C_FUSED_ALLOW_EAGER_DEGRADE is "
                    "set, so disabling the fused runtime and degrading to the eager "
                    "loss_and_grad path for the rest of the run (bisection mode). The "
                    "underlying fused-kernel bug is NOT fixed; the receipt field "
                    "path_c_fused_runtime_evidence records this degraded run.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None

            error = result_box.get("error")
            if error is None:
                return result_box.get("value")
        else:
            try:
                return runtime.value_and_grad(
                    self.model,
                    loss_batch,
                    self._loss_and_grad,
                )
            except Exception as exc:  # pragma: no cover - hardware-specific path
                error = exc

        if error is not None:
            if self._path_c_runtime_fallback_error is None:
                self._path_c_runtime_fallback_error = (
                    f"{type(error).__name__}: {error}"
                )
            self._path_c_runtime_fallback_count += 1
            if not allow_eager_degrade:
                # RULE #1: a fused-runtime crash is a real bug in the selected
                # Path C kernel/bridge. Surface it instead of silently producing
                # an eager result that masks the broken fused path.
                raise RuntimeError(
                    "Path C fused direct-chain runtime.value_and_grad crashed in "
                    "CompiledPretrainingStep._run_fused_value_and_grad_guarded "
                    f"({type(error).__name__}: {error}). Refusing to silently fall "
                    "back to the eager loss_and_grad path (RULE #1) — this points at "
                    "a real bug in the fused Path C kernel/bridge (e.g. the MLX<->CUDA "
                    "buffer handoff, incomplete fused gradient extraction, or a CUDA "
                    "launch/exec error). Set MLX_PATH_C_FUSED_ALLOW_EAGER_DEGRADE=1 to "
                    "opt into a loud eager degrade for bisection only."
                ) from error
            # Opt-in bisection escape: LOUD degrade (warning + receipt fields).
            warnings.warn(
                "Path C fused direct-chain runtime.value_and_grad failed; "
                "MLX_PATH_C_FUSED_ALLOW_EAGER_DEGRADE is set, so degrading to the "
                "eager loss_and_grad path (bisection mode). The underlying fused "
                "bug is NOT fixed; path_c_fused_runtime_evidence records this "
                f"degraded run: {type(error).__name__}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        # Unreachable in practice: both the timeout and inline branches return on
        # success above. If we somehow reach here, fail loud rather than silently
        # degrade to eager (RULE #1).
        raise RuntimeError(
            "CompiledPretrainingStep._run_fused_value_and_grad_guarded reached an "
            "unreachable state: the fused runtime neither returned a result nor "
            "recorded an error. This is an internal control-flow bug."
        )

    def _eager_step(
        self,
        batch: CompiledBatch,
        prev_grad: Any,
        do_update: bool,
    ) -> tuple[mx.array, mx.array, Any]:
        loss_batch = cast(Mapping[str, mx.array], batch)
        runtime = self._path_c_training_runtime
        if runtime is None or self._path_c_runtime_disabled:
            (loss, ntokens), grads = self._loss_and_grad(self.model, loss_batch)
        else:
            outcome = self._run_fused_value_and_grad_guarded(runtime, loss_batch)
            if outcome is not None:
                (loss, ntokens), grads = outcome
            else:
                # OPT-IN BISECTION DEGRADE ONLY: this branch is reachable solely
                # when MLX_PATH_C_FUSED_ALLOW_EAGER_DEGRADE is set. By default
                # _run_fused_value_and_grad_guarded RAISES on a fused crash/timeout
                # (RULE #1 — surface the bug) instead of returning None. When the
                # operator has explicitly opted into the loud bisection degrade,
                # the fused Path C direct-chain runtime raised or exceeded its
                # wall-clock budget; we degrade to the eager value_and_grad path
                # (already warned + recorded in path_c_fused_runtime_evidence) so a
                # deliberate bisection run completes rather than aborting.
                (loss, ntokens), grads = self._loss_and_grad(
                    self.model, loss_batch
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
    "PathCGeneratedStageTrainBlockCallableArtifact",
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
