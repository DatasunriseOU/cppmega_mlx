"""MLX custom-function bridge for Path C fused-block replay.

The bridge keeps the generated TileLang artifact scoped to the selected
M/R/A train block. Prefix and suffix/loss work stay in MLX; the custom VJP
only writes suffix cotangents into ABI grad slots and replays the fused
block backward through the same model-owned physical banks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx

from cppmega_mlx.runtime.path_c_physical_abi import (
    logical_bank_view,
    write_into_bank_slot,
)

FusedReplayBoundaryCallable = Callable[..., tuple[mx.array, ...]]


def _bank_buffers_from_owner(bank_owner: Any) -> Mapping[str, Any]:
    buffers = bank_owner if isinstance(bank_owner, Mapping) else None
    if buffers is None:
        buffers = getattr(bank_owner, "buffers", None)
    if not isinstance(buffers, Mapping):
        raise TypeError("fused replay custom function requires bank_owner buffers")
    return buffers


def _reshape_like(value: mx.array, primal: mx.array) -> mx.array:
    if tuple(int(dim) for dim in tuple(value.shape)) == tuple(
        int(dim) for dim in tuple(primal.shape)
    ):
        return value
    return mx.reshape(value, primal.shape)


def _logical_views(
    abi_map: Mapping[str, Any],
    bank_buffers: Mapping[str, Any],
    logical_names: Sequence[str],
) -> tuple[mx.array, ...]:
    return tuple(
        logical_bank_view(abi_map, bank_buffers, str(name))
        for name in logical_names
    )


_ABI_DTYPE_MAP: Mapping[str, mx.Dtype] = {
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


def _cast_boundary_cotangent_for_abi(
    abi_map: Mapping[str, Any],
    logical_name: str,
    value: mx.array,
) -> mx.array:
    """Return a cotangent seed in the dtype declared by its ABI grad slot."""

    info = abi_map.get(logical_name)
    if not isinstance(info, Mapping):
        raise ValueError(
            f"logical cotangent buffer is not in physical ABI: {logical_name!r}"
        )
    expected_dtype = str(info.get("dtype", ""))
    mx_dtype = _ABI_DTYPE_MAP.get(expected_dtype)
    if mx_dtype is None or value.dtype == mx_dtype:
        return value
    return value.astype(mx_dtype)


def replay_launch_scalar_params(
    *,
    run_backward: bool,
    launch_count: int,
    subchunk_count: int,
    gate_param: str,
    row_chunk_index_param: str = "",
    row_subchunk_index_param: str = "",
) -> tuple[dict[str, int], ...]:
    """Return scalar launch params for a fused replay pass."""

    chunk_param = str(row_chunk_index_param or "")
    subchunk_param = str(row_subchunk_index_param or "")
    launches: list[dict[str, int]] = []
    gate_value = 1 if run_backward else 0
    for chunk_index in range(max(1, int(launch_count))):
        for subchunk_index in range(max(1, int(subchunk_count))):
            scalar_params: dict[str, int] = {str(gate_param): gate_value}
            if chunk_param:
                scalar_params[chunk_param] = int(chunk_index)
            if subchunk_param:
                scalar_params[subchunk_param] = int(subchunk_index)
            launches.append(scalar_params)
    return tuple(launches)


def build_fused_replay_boundary_custom_function(
    *,
    artifact: Any,
    bank_owner: Any,
    abi_map: Mapping[str, Any],
    hidden_entry_logical_name: str,
    boundary_output_logical_names: Sequence[str],
    boundary_cotangent_logical_names: Sequence[str],
    in_region_parameter_bank_aliases: Mapping[str, Mapping[str, Any]],
    parameter_order: Sequence[str],
    row_chunk_count: int | None = None,
    row_chunk_index_param: str | None = None,
    row_subchunk_count: int | None = None,
    row_subchunk_index_param: str | None = None,
    backward_stage_count: int | None = None,
    backward_stage_index_param: str | None = None,
    backward_gate_param: str = "path_c_run_backward",
) -> FusedReplayBoundaryCallable:
    """Return ``f(hidden_entry, *params) -> boundary outputs``.

    Forward writes the prefix hidden entry and in-region parameter primals into
    physical ABI banks, runs the fused block with ``path_c_run_backward=0``,
    and returns bank views for the suffix boundary tensors. The VJP receives
    suffix cotangents for those boundary tensors, writes them into matching
    ``*_grad`` ABI slots, replays the generated backward with
    ``path_c_run_backward=1``, and returns bank-view cotangents for the prefix
    hidden entry and in-region parameters.
    """

    if not callable(getattr(artifact, "forward", None)):
        raise TypeError("fused replay requires an artifact.forward method")
    bank_buffers = _bank_buffers_from_owner(bank_owner)
    parameter_order_tuple = tuple(str(name) for name in parameter_order)
    boundary_outputs = tuple(str(name) for name in boundary_output_logical_names)
    boundary_cotangents = tuple(str(name) for name in boundary_cotangent_logical_names)
    if not boundary_outputs:
        raise ValueError("fused replay requires at least one boundary output")
    if len(boundary_outputs) != len(boundary_cotangents):
        raise ValueError(
            "boundary output and cotangent logical-name counts must match"
        )
    parameter_alias_table = tuple(
        dict(in_region_parameter_bank_aliases[name]) for name in parameter_order_tuple
    )
    chunk_param = str(row_chunk_index_param or "")
    launch_count = max(1, int(row_chunk_count or 1)) if chunk_param else 1
    subchunk_param = str(row_subchunk_index_param or "")
    subchunk_count = (
        max(1, int(row_subchunk_count or 1)) if subchunk_param else 1
    )
    gate_param = str(backward_gate_param or "path_c_run_backward")
    stage_param = str(backward_stage_index_param or "")
    stage_count = max(1, int(backward_stage_count or 1)) if stage_param else 1
    backward_sync_outputs = (
        f"{hidden_entry_logical_name}_grad",
        *(
            str(info["logical_grad_name"])
            for info in parameter_alias_table
        ),
    )

    def _write_primal_state(hidden_entry: mx.array, params: tuple[mx.array, ...]) -> None:
        write_into_bank_slot(
            abi_map,
            bank_buffers,
            hidden_entry_logical_name,
            hidden_entry,
        )
        for info, value in zip(parameter_alias_table, params, strict=True):
            write_into_bank_slot(
                abi_map,
                bank_buffers,
                str(info["logical_name"]),
                value,
            )

    def _launch_all(*, run_backward: bool) -> None:
        for scalar_params in replay_launch_scalar_params(
            run_backward=run_backward,
            launch_count=launch_count,
            subchunk_count=subchunk_count,
            gate_param=gate_param,
            row_chunk_index_param=chunk_param,
            row_subchunk_index_param=subchunk_param,
        ):
            stage_range = range(stage_count) if run_backward else range(1)
            for stage_index in stage_range:
                stage_scalar_params = dict(scalar_params)
                if run_backward and stage_param:
                    stage_scalar_params[stage_param] = int(stage_index)
                try:
                    artifact.forward(
                        bank_owner=bank_owner,
                        kernel_scalar_params=stage_scalar_params,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "fused replay artifact launch failed with "
                        f"kernel_scalar_params={stage_scalar_params!r}"
                    ) from exc
        sync_outputs = backward_sync_outputs if run_backward else boundary_outputs
        mx.eval(*_logical_views(abi_map, bank_buffers, sync_outputs))

    @mx.custom_function
    def fused_replay_boundary(
        hidden_entry: mx.array,
        *params: mx.array,
    ) -> tuple[mx.array, ...]:
        if len(params) != len(parameter_order_tuple):
            raise ValueError(
                "fused replay got "
                f"{len(params)} parameter primals, expected "
                f"{len(parameter_order_tuple)}"
            )
        _write_primal_state(hidden_entry, params)
        _launch_all(run_backward=False)
        return _logical_views(abi_map, bank_buffers, boundary_outputs)

    @fused_replay_boundary.vjp
    def fused_replay_boundary_vjp(
        primals: tuple[mx.array, ...],
        cotangents: tuple[mx.array, ...] | mx.array,
        output: Any,
    ) -> tuple[mx.array, ...]:
        del output
        hidden_entry_primal = primals[0]
        param_primals = primals[1:]
        cotangent_tuple = (
            cotangents if isinstance(cotangents, tuple) else (cotangents,)
        )
        if len(cotangent_tuple) != len(boundary_cotangents):
            raise ValueError(
                "fused replay VJP got "
                f"{len(cotangent_tuple)} boundary cotangents, expected "
                f"{len(boundary_cotangents)}"
            )
        if len(param_primals) != len(parameter_order_tuple):
            raise ValueError(
                "fused replay VJP received "
                f"{len(param_primals)} parameter primals, expected "
                f"{len(parameter_order_tuple)}"
            )
        for logical_name, value in zip(
            boundary_cotangents,
            cotangent_tuple,
            strict=True,
        ):
            write_into_bank_slot(
                abi_map,
                bank_buffers,
                logical_name,
                _cast_boundary_cotangent_for_abi(
                    abi_map,
                    logical_name,
                    value,
                ),
            )
        _launch_all(run_backward=True)

        hidden_entry_grad_name = f"{hidden_entry_logical_name}_grad"
        hidden_entry_cotan = logical_bank_view(
            abi_map,
            bank_buffers,
            hidden_entry_grad_name,
        )
        hidden_entry_cotan = _reshape_like(hidden_entry_cotan, hidden_entry_primal)

        param_cotans: list[mx.array] = []
        for info, primal in zip(parameter_alias_table, param_primals, strict=True):
            grad_name = str(info["logical_grad_name"])
            cotan = logical_bank_view(abi_map, bank_buffers, grad_name)
            param_cotans.append(_reshape_like(cotan, primal))
        return (hidden_entry_cotan, *param_cotans)

    return fused_replay_boundary


__all__ = [
    "FusedReplayBoundaryCallable",
    "build_fused_replay_boundary_custom_function",
    "replay_launch_scalar_params",
]
