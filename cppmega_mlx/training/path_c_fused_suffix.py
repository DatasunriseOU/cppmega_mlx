"""Fused-suffix custom_function bridge for the Path C training runtime.

This module builds a TileLang artifact + bank-owner backed
:class:`mlx.core.custom_function` that the merged-mode training runtime
plugs in front of the trainer's loss-fn. Inside the loss-fn, the model
runs the **prefix** (embedding + side-channel embeddings + layers before
the fused region) eagerly, then hands off the entry hidden state and the
in-region trainable parameters to the custom function, which:

1.  Writes each primal into its bank slot (in place, no allocation).
2.  Launches the fused TileLang artifact's forward pass — the same kernel
    already wires forward + backward in a single dispatch, so the
    parameter grad slots and the hidden-entry grad slot are populated as
    a side effect when ``forward`` returns.
3.  Returns the bank-resident ``loss`` and ``ntokens`` views.

The function's VJP returns bank-view cotangents in the same primal order
so MLX autograd carries the fused suffix gradients out to:

  * the prefix (via ``hidden_entry_grad``);
  * the optimizer (via the in-region parameter grads).

Because the suffix never appears in the MLX autograd graph, eager
autograd skips layers in the fused region, the final-norm, the
``lm_head`` projection, and the cross-entropy loss reduction. That is
where Path C wins compute / memory over Path B.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import mlx.core as mx

from cppmega_mlx.runtime.path_c_physical_abi import (
    logical_bank_view,
    write_into_bank_slot,
)


FusedSuffixCallable = Callable[..., Any]
_MX_DTYPE_BY_ABI_NAME = {
    "bool": mx.bool_,
    "uint8": mx.uint8,
    "int8": mx.int8,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "uint16": mx.uint16,
    "int16": mx.int16,
    "float32": mx.float32,
    "uint32": mx.uint32,
    "int32": mx.int32,
    "float64": mx.float64,
    "uint64": mx.uint64,
    "int64": mx.int64,
}


def _zeros_like_with_cotangent_dtype(value: mx.array) -> mx.array:
    """Return zeros shaped like ``value`` matching its cotangent dtype.

    MLX's custom_function VJP expects each returned cotangent to be a
    real-valued array; integer primals receive a zero-valued float
    cotangent so the autograd plumbing does not break, even though the
    upstream consumer never reads them.
    """
    if value.dtype in (mx.int8, mx.int16, mx.int32, mx.int64,
                        mx.uint8, mx.uint16, mx.uint32, mx.uint64):
        return mx.zeros(value.shape, dtype=mx.float32)
    return mx.zeros_like(value)


def _target_mask_for_abi(
    abi_map: Mapping[str, Any],
    logical_name: str,
    value: mx.array,
) -> mx.array:
    info = abi_map.get(logical_name)
    if not isinstance(info, Mapping):
        return value
    expected_dtype = _MX_DTYPE_BY_ABI_NAME.get(str(info.get("dtype", "")))
    if expected_dtype is None or value.dtype == expected_dtype:
        return value
    if value.dtype not in (mx.float16, mx.bfloat16, mx.float32, mx.float64):
        return value
    return value.astype(expected_dtype)


def build_fused_suffix_custom_function(
    *,
    artifact: Any,
    bank_owner: Any,
    abi_map: Mapping[str, Any],
    hidden_entry_logical_name: str,
    target_ids_logical_name: str,
    target_mask_logical_name: str,
    loss_logical_name: str,
    ntokens_logical_name: str,
    in_region_parameter_bank_aliases: Mapping[str, Mapping[str, Any]],
    parameter_order: Sequence[str],
) -> FusedSuffixCallable:
    """Compose a closure that wraps the fused artifact in an MLX custom_function.

    The returned callable's positional signature is::

        f(hidden_entry, target_ids, target_mask, *params_in_order)

    Each ``params_in_order`` entry is the parameter tensor reachable from
    the model tree, in the canonical order described by
    ``parameter_order``. The function returns ``(loss, ntokens)`` where
    both are scalar :class:`mlx.core.array` views into the model-owned
    physical-ABI banks.
    """

    bank_buffers = (
        bank_owner if isinstance(bank_owner, Mapping)
        else getattr(bank_owner, "buffers", None)
    )
    if not isinstance(bank_buffers, Mapping):
        raise TypeError(
            "fused-suffix custom function requires a bank owner with buffers"
        )
    parameter_order_tuple = tuple(parameter_order)
    parameter_alias_table = tuple(
        in_region_parameter_bank_aliases[name]
        for name in parameter_order_tuple
    )

    def _write_runtime_state(
        hidden_entry: mx.array,
        target_ids: mx.array,
        target_mask: mx.array,
        params: tuple[mx.array, ...],
    ) -> None:
        # Hidden entry, target_ids and target_mask are runtime inputs that
        # the fused kernel reads from bank slots; the in-region parameter
        # primals are the trainable tensors the kernel folds with its
        # forward / backward graph. Every write is a slice-assignment
        # into pre-allocated bank storage — no implicit allocation, no
        # large staging tensor, exactly the explicit caller-visible bridge
        # the project's tensor-memory rule requires.
        write_into_bank_slot(
            abi_map, bank_buffers, hidden_entry_logical_name, hidden_entry
        )
        write_into_bank_slot(
            abi_map, bank_buffers, target_ids_logical_name, target_ids
        )
        target_mask = _target_mask_for_abi(
            abi_map,
            target_mask_logical_name,
            target_mask,
        )
        write_into_bank_slot(
            abi_map, bank_buffers, target_mask_logical_name, target_mask
        )
        for info, value in zip(parameter_alias_table, params, strict=True):
            write_into_bank_slot(
                abi_map,
                bank_buffers,
                str(info["logical_name"]),
                value,
            )

    def _launch_fused_kernel() -> None:
        # ``artifact.forward(bank_owner=...)`` runs the fused PrimFunc
        # forward + backward in a single dispatch; the train-block
        # PrimFunc was compiled with ``include_backward=True`` so the
        # parameter grad slots and the hidden-entry grad slot are filled
        # as a side effect by the time forward returns.
        artifact.forward(bank_owner=bank_owner)
        # Force evaluation so the bank slots are observable to the VJP
        # without the autograd graph re-tracing the kernel call.
        mx.eval(*bank_buffers.values())

    @mx.custom_function
    def fused_suffix(
        hidden_entry: mx.array,
        target_ids: mx.array,
        target_mask: mx.array,
        *params: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if len(params) != len(parameter_order_tuple):
            raise ValueError(
                "fused-suffix custom function got "
                f"{len(params)} parameter primals, expected "
                f"{len(parameter_order_tuple)}"
            )
        _write_runtime_state(hidden_entry, target_ids, target_mask, params)
        _launch_fused_kernel()
        loss = logical_bank_view(
            abi_map, bank_buffers, loss_logical_name
        )
        ntokens = logical_bank_view(
            abi_map, bank_buffers, ntokens_logical_name
        )
        return mx.squeeze(loss), mx.squeeze(ntokens)

    @fused_suffix.vjp
    def fused_suffix_vjp(
        primals: tuple[mx.array, ...],
        cotangents: tuple[mx.array, ...] | mx.array,
        output: Any,
    ) -> tuple[mx.array, ...]:
        del output
        # MLX delivers cotangents either as a tuple (one per output) or
        # as a single array when the function returns a single value.
        # ``(loss, ntokens)`` produces a 2-tuple; we use only the loss
        # cotangent (ntokens is auxiliary and contributes no gradient).
        if isinstance(cotangents, tuple):
            loss_cotan = cotangents[0]
        else:
            loss_cotan = cotangents
        hidden_entry_primal = primals[0]
        target_ids_primal = primals[1]
        target_mask_primal = primals[2]
        param_primals = primals[3:]
        if len(param_primals) != len(parameter_order_tuple):
            raise ValueError(
                "fused-suffix VJP received "
                f"{len(param_primals)} parameter cotangents, expected "
                f"{len(parameter_order_tuple)}"
            )

        hidden_entry_grad_name = f"{hidden_entry_logical_name}_grad"
        hidden_entry_cotan = logical_bank_view(
            abi_map, bank_buffers, hidden_entry_grad_name
        )
        # Reshape the bank-flat view to the prefix-supplied primal shape
        # (the prefix may broadcast over batch / sequence; we honor the
        # caller's exact shape).
        if (
            tuple(int(dim) for dim in tuple(hidden_entry_cotan.shape))
            != tuple(int(dim) for dim in tuple(hidden_entry_primal.shape))
        ):
            hidden_entry_cotan = mx.reshape(
                hidden_entry_cotan, hidden_entry_primal.shape
            )
        hidden_entry_cotan = hidden_entry_cotan * loss_cotan

        target_ids_cotan = _zeros_like_with_cotangent_dtype(
            target_ids_primal
        )
        target_mask_cotan = _zeros_like_with_cotangent_dtype(
            target_mask_primal
        )

        param_cotans: list[mx.array] = []
        for info, primal in zip(parameter_alias_table, param_primals, strict=True):
            grad_name = str(info["logical_grad_name"])
            cotan = logical_bank_view(abi_map, bank_buffers, grad_name)
            if (
                tuple(int(dim) for dim in tuple(cotan.shape))
                != tuple(int(dim) for dim in tuple(primal.shape))
            ):
                cotan = mx.reshape(cotan, primal.shape)
            cotan = cotan * loss_cotan
            param_cotans.append(cotan)

        return (
            hidden_entry_cotan,
            target_ids_cotan,
            target_mask_cotan,
            *param_cotans,
        )

    return fused_suffix


__all__ = ["build_fused_suffix_custom_function", "FusedSuffixCallable"]
