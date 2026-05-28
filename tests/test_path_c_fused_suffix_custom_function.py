"""Unit tests for the fused-suffix custom function bridge."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import pytest

from cppmega_mlx.training.path_c_fused_suffix import (
    build_fused_suffix_custom_function,
)
from cppmega_mlx.training.path_c_fused_replay import (
    build_fused_replay_boundary_custom_function,
    replay_launch_scalar_params,
)


class _RecordingArtifact:
    """In-memory artifact stand-in that records forward calls and reads bank."""

    def __init__(self, sentinel_loss: float, sentinel_ntokens: float) -> None:
        self.forward_calls: list[Mapping[str, Any]] = []
        self.sentinel_loss = sentinel_loss
        self.sentinel_ntokens = sentinel_ntokens

    def forward(self, *, bank_owner: Any) -> None:
        self.forward_calls.append({"bank_owner": bank_owner})
        # Pretend the kernel populated loss / ntokens / grad slots. The
        # ABI map below places loss at offset 100, ntokens at 101, and
        # hidden_entry_grad at offset 0. We use mx.array assignment to
        # mutate the bank in place — the same shape of side-effect the
        # real artifact has.
        bank = bank_owner.buffers["f32"]
        bank[100:101] = mx.array([self.sentinel_loss], dtype=mx.float32)
        bank[101:102] = mx.array([self.sentinel_ntokens], dtype=mx.float32)
        # hidden_entry_grad sentinel — bank[200:212] for a (1, 3, 4) view.
        bank[200:212] = mx.full((12,), 7.0, dtype=mx.float32)
        # parameter grad sentinels
        bank[212:216] = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        bank[216:220] = mx.array([5.0, 6.0, 7.0, 8.0], dtype=mx.float32)


class _FakeBankOwner:
    def __init__(self) -> None:
        self.buffers = {"f32": mx.zeros((300,), dtype=mx.float32)}


class _RecordingReplayArtifact:
    """Artifact stand-in for fused block forward/replay-backward gates."""

    def __init__(self) -> None:
        self.forward_calls: list[Mapping[str, Any]] = []
        self.backward_cotangent_snapshots: list[dict[str, list[float]]] = []

    def forward(
        self,
        *,
        bank_owner: Any,
        kernel_scalar_params: Mapping[str, Any] | None = None,
    ) -> None:
        scalar_params = dict(kernel_scalar_params or {})
        self.forward_calls.append(
            {"bank_owner": bank_owner, "kernel_scalar_params": scalar_params}
        )
        bank = bank_owner.buffers["f32"]
        run_backward = int(scalar_params.get("path_c_run_backward", -1))
        if run_backward == 0:
            bank[80:92] = mx.full((12,), 2.0, dtype=mx.float32)
            bank[92:104] = mx.full((12,), 5.0, dtype=mx.float32)
        elif run_backward == 1:
            self.backward_cotangent_snapshots.append(
                {
                    "boundary_a_grad": bank[220:232].tolist(),
                    "boundary_b_grad": bank[232:244].tolist(),
                }
            )
            bank[200:212] = mx.full((12,), 7.0, dtype=mx.float32)
            bank[212:216] = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
            bank[216:220] = mx.array([5.0, 6.0, 7.0, 8.0], dtype=mx.float32)
        else:
            raise AssertionError(f"unexpected path_c_run_backward={run_backward}")


def _make_abi_map() -> dict[str, dict[str, Any]]:
    return {
        "hidden_entry": {
            "bank": "f32", "dtype": "float32", "offset": 0, "size": 12,
            "shape": (12,), "logical_shape": (1, 3, 4),
        },
        "hidden_entry_grad": {
            "bank": "f32", "dtype": "float32", "offset": 200, "size": 12,
            "shape": (12,), "logical_shape": (1, 3, 4),
        },
        "target_ids": {
            "bank": "f32", "dtype": "float32", "offset": 50, "size": 3,
            "shape": (3,), "logical_shape": (3,),
        },
        "target_mask": {
            "bank": "f32", "dtype": "float32", "offset": 53, "size": 3,
            "shape": (3,), "logical_shape": (3,),
        },
        "loss": {
            "bank": "f32", "dtype": "float32", "offset": 100, "size": 1,
            "shape": (1,), "logical_shape": (1,),
        },
        "ntokens": {
            "bank": "f32", "dtype": "float32", "offset": 101, "size": 1,
            "shape": (1,), "logical_shape": (1,),
        },
        "param_a": {
            "bank": "f32", "dtype": "float32", "offset": 60, "size": 4,
            "shape": (4,), "logical_shape": (4,),
        },
        "param_a_grad": {
            "bank": "f32", "dtype": "float32", "offset": 212, "size": 4,
            "shape": (4,), "logical_shape": (4,),
        },
        "param_b": {
            "bank": "f32", "dtype": "float32", "offset": 64, "size": 4,
            "shape": (4,), "logical_shape": (4,),
        },
        "param_b_grad": {
            "bank": "f32", "dtype": "float32", "offset": 216, "size": 4,
            "shape": (4,), "logical_shape": (4,),
        },
        "boundary_a": {
            "bank": "f32", "dtype": "float32", "offset": 80, "size": 12,
            "shape": (12,), "logical_shape": (1, 3, 4),
        },
        "boundary_b": {
            "bank": "f32", "dtype": "float32", "offset": 92, "size": 12,
            "shape": (12,), "logical_shape": (1, 3, 4),
        },
        "boundary_a_grad": {
            "bank": "f32", "dtype": "float32", "offset": 220, "size": 12,
            "shape": (12,), "logical_shape": (1, 3, 4),
        },
        "boundary_b_grad": {
            "bank": "f32", "dtype": "float32", "offset": 232, "size": 12,
            "shape": (12,), "logical_shape": (1, 3, 4),
        },
    }


def _parameter_aliases() -> dict[str, dict[str, Any]]:
    return {
        "p_a": {
            "logical_name": "param_a",
            "logical_grad_name": "param_a_grad",
            "bank": "f32", "dtype": "float32",
            "offset": 60, "size": 4, "logical_shape": (4,),
        },
        "p_b": {
            "logical_name": "param_b",
            "logical_grad_name": "param_b_grad",
            "bank": "f32", "dtype": "float32",
            "offset": 64, "size": 4, "logical_shape": (4,),
        },
    }


def test_fused_replay_boundary_value_and_grad_seeds_cotangents() -> None:
    artifact = _RecordingReplayArtifact()
    bank_owner = _FakeBankOwner()
    abi_map = _make_abi_map()
    f = build_fused_replay_boundary_custom_function(
        artifact=artifact,
        bank_owner=bank_owner,
        abi_map=abi_map,
        hidden_entry_logical_name="hidden_entry",
        boundary_output_logical_names=("boundary_a", "boundary_b"),
        boundary_cotangent_logical_names=("boundary_a_grad", "boundary_b_grad"),
        in_region_parameter_bank_aliases=_parameter_aliases(),
        parameter_order=("p_a", "p_b"),
    )

    hidden_entry = mx.full((1, 3, 4), 9.0, dtype=mx.float32)
    param_a = mx.array([10.0, 11.0, 12.0, 13.0], dtype=mx.float32)
    param_b = mx.array([20.0, 21.0, 22.0, 23.0], dtype=mx.float32)

    def loss_fn(hidden_entry, param_a, param_b):
        boundary_a, boundary_b = f(hidden_entry, param_a, param_b)
        return boundary_a.sum() + boundary_b.sum() * mx.array(3.0, dtype=mx.float32)

    value, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))(
        hidden_entry,
        param_a,
        param_b,
    )
    mx.eval(value, *grads)

    assert float(value) == (12 * 2.0) + (12 * 5.0 * 3.0)
    assert [
        call["kernel_scalar_params"]["path_c_run_backward"]
        for call in artifact.forward_calls
    ] == [0, 1]
    assert artifact.backward_cotangent_snapshots == [
        {
            "boundary_a_grad": [1.0] * 12,
            "boundary_b_grad": [3.0] * 12,
        }
    ]
    assert grads[0].shape == hidden_entry.shape
    assert grads[0].tolist() == [[[7.0] * 4] * 3]
    assert grads[1].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert grads[2].tolist() == [5.0, 6.0, 7.0, 8.0]


def test_fused_replay_boundary_casts_suffix_cotangent_to_grad_bank_dtype() -> None:
    artifact = _RecordingReplayArtifact()
    bank_owner = _FakeBankOwner()
    f = build_fused_replay_boundary_custom_function(
        artifact=artifact,
        bank_owner=bank_owner,
        abi_map=_make_abi_map(),
        hidden_entry_logical_name="hidden_entry",
        boundary_output_logical_names=("boundary_a", "boundary_b"),
        boundary_cotangent_logical_names=("boundary_a_grad", "boundary_b_grad"),
        in_region_parameter_bank_aliases=_parameter_aliases(),
        parameter_order=("p_a", "p_b"),
    )

    def loss_fn(hidden_entry, param_a, param_b):
        boundary_a, boundary_b = f(hidden_entry, param_a, param_b)
        return (
            boundary_a.astype(mx.bfloat16).sum()
            + boundary_b.astype(mx.bfloat16).sum() * mx.array(2.0, dtype=mx.bfloat16)
        )

    value, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))(
        mx.full((1, 3, 4), 9.0, dtype=mx.float32),
        mx.array([10.0, 11.0, 12.0, 13.0], dtype=mx.float32),
        mx.array([20.0, 21.0, 22.0, 23.0], dtype=mx.float32),
    )
    mx.eval(value, *grads)

    assert artifact.backward_cotangent_snapshots == [
        {
            "boundary_a_grad": [1.0] * 12,
            "boundary_b_grad": [2.0] * 12,
        }
    ]
    assert bank_owner.buffers["f32"].dtype == mx.float32


def test_fused_replay_grid_chunked_artifact_launches_once_without_chunk_param() -> None:
    artifact = _RecordingReplayArtifact()
    bank_owner = _FakeBankOwner()
    f = build_fused_replay_boundary_custom_function(
        artifact=artifact,
        bank_owner=bank_owner,
        abi_map=_make_abi_map(),
        hidden_entry_logical_name="hidden_entry",
        boundary_output_logical_names=("boundary_a", "boundary_b"),
        boundary_cotangent_logical_names=("boundary_a_grad", "boundary_b_grad"),
        in_region_parameter_bank_aliases=_parameter_aliases(),
        parameter_order=("p_a", "p_b"),
        row_chunk_count=64,
        row_chunk_index_param=None,
    )

    outputs = f(
        mx.full((1, 3, 4), 9.0, dtype=mx.float32),
        mx.array([10.0, 11.0, 12.0, 13.0], dtype=mx.float32),
        mx.array([20.0, 21.0, 22.0, 23.0], dtype=mx.float32),
    )
    mx.eval(*outputs)

    assert len(artifact.forward_calls) == 1
    assert artifact.forward_calls[0]["kernel_scalar_params"] == {
        "path_c_run_backward": 0,
    }


def test_fused_replay_launcher_chunked_artifact_launches_all_chunks() -> None:
    launch_params = replay_launch_scalar_params(
        run_backward=False,
        launch_count=2,
        subchunk_count=3,
        gate_param="path_c_run_backward",
        row_chunk_index_param="path_c_row_chunk_index",
        row_subchunk_index_param="path_c_row_subchunk_index",
    )

    expected_forward = [
        {
            "path_c_run_backward": 0,
            "path_c_row_chunk_index": chunk,
            "path_c_row_subchunk_index": subchunk,
        }
        for chunk in range(2)
        for subchunk in range(3)
    ]
    assert list(launch_params) == expected_forward
    assert list(
        replay_launch_scalar_params(
            run_backward=True,
            launch_count=1,
            subchunk_count=1,
            gate_param="path_c_run_backward",
        )
    ) == [{"path_c_run_backward": 1}]


def test_fused_replay_launcher_stage_selector_launches_backward_stages() -> None:
    artifact = _RecordingReplayArtifact()
    bank_owner = _FakeBankOwner()
    f = build_fused_replay_boundary_custom_function(
        artifact=artifact,
        bank_owner=bank_owner,
        abi_map=_make_abi_map(),
        hidden_entry_logical_name="hidden_entry",
        boundary_output_logical_names=("boundary_a", "boundary_b"),
        boundary_cotangent_logical_names=("boundary_a_grad", "boundary_b_grad"),
        in_region_parameter_bank_aliases=_parameter_aliases(),
        parameter_order=("p_a", "p_b"),
        row_chunk_count=1,
        row_chunk_index_param="path_c_row_chunk_index",
        row_subchunk_count=2,
        row_subchunk_index_param="path_c_row_subchunk_index",
        backward_stage_count=3,
        backward_stage_index_param="path_c_backward_stage_index",
    )

    def loss_fn(hidden_entry, param_a, param_b):
        boundary_a, boundary_b = f(hidden_entry, param_a, param_b)
        return boundary_a.sum() + boundary_b.sum()

    value, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))(
        mx.full((1, 3, 4), 9.0, dtype=mx.float32),
        mx.array([10.0, 11.0, 12.0, 13.0], dtype=mx.float32),
        mx.array([20.0, 21.0, 22.0, 23.0], dtype=mx.float32),
    )
    mx.eval(value, *grads)

    assert [
        call["kernel_scalar_params"]
        for call in artifact.forward_calls
    ] == [
        {
            "path_c_run_backward": 0,
            "path_c_row_chunk_index": 0,
            "path_c_row_subchunk_index": 0,
        },
        {
            "path_c_run_backward": 0,
            "path_c_row_chunk_index": 0,
            "path_c_row_subchunk_index": 1,
        },
        {
            "path_c_run_backward": 1,
            "path_c_row_chunk_index": 0,
            "path_c_row_subchunk_index": 0,
            "path_c_backward_stage_index": 0,
        },
        {
            "path_c_run_backward": 1,
            "path_c_row_chunk_index": 0,
            "path_c_row_subchunk_index": 0,
            "path_c_backward_stage_index": 1,
        },
        {
            "path_c_run_backward": 1,
            "path_c_row_chunk_index": 0,
            "path_c_row_subchunk_index": 0,
            "path_c_backward_stage_index": 2,
        },
        {
            "path_c_run_backward": 1,
            "path_c_row_chunk_index": 0,
            "path_c_row_subchunk_index": 1,
            "path_c_backward_stage_index": 0,
        },
        {
            "path_c_run_backward": 1,
            "path_c_row_chunk_index": 0,
            "path_c_row_subchunk_index": 1,
            "path_c_backward_stage_index": 1,
        },
        {
            "path_c_run_backward": 1,
            "path_c_row_chunk_index": 0,
            "path_c_row_subchunk_index": 1,
            "path_c_backward_stage_index": 2,
        },
    ]


def test_fused_suffix_forward_writes_inputs_and_returns_loss() -> None:
    artifact = _RecordingArtifact(sentinel_loss=2.5, sentinel_ntokens=11.0)
    bank_owner = _FakeBankOwner()
    abi_map = _make_abi_map()
    in_region_aliases = {
        "p_a": {
            "logical_name": "param_a",
            "logical_grad_name": "param_a_grad",
            "bank": "f32", "dtype": "float32",
            "offset": 60, "size": 4, "logical_shape": (4,),
        },
        "p_b": {
            "logical_name": "param_b",
            "logical_grad_name": "param_b_grad",
            "bank": "f32", "dtype": "float32",
            "offset": 64, "size": 4, "logical_shape": (4,),
        },
    }
    f = build_fused_suffix_custom_function(
        artifact=artifact,
        bank_owner=bank_owner,
        abi_map=abi_map,
        hidden_entry_logical_name="hidden_entry",
        target_ids_logical_name="target_ids",
        target_mask_logical_name="target_mask",
        loss_logical_name="loss",
        ntokens_logical_name="ntokens",
        in_region_parameter_bank_aliases=in_region_aliases,
        parameter_order=("p_a", "p_b"),
    )

    hidden_entry = mx.full((1, 3, 4), 9.0, dtype=mx.float32)
    target_ids = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    target_mask = mx.array([1.0, 0.0, 1.0], dtype=mx.float32)
    param_a = mx.array([10.0, 11.0, 12.0, 13.0], dtype=mx.float32)
    param_b = mx.array([20.0, 21.0, 22.0, 23.0], dtype=mx.float32)

    loss, ntokens = f(hidden_entry, target_ids, target_mask, param_a, param_b)
    mx.eval(loss, ntokens)
    assert float(loss) == 2.5
    assert float(ntokens) == 11.0
    assert len(artifact.forward_calls) == 1

    # Verify the bank actually carries the written values for inputs.
    bank = bank_owner.buffers["f32"]
    assert bank[0:12].tolist() == [9.0] * 12   # hidden_entry
    assert bank[50:53].tolist() == [1.0, 2.0, 3.0]
    assert bank[53:56].tolist() == [1.0, 0.0, 1.0]
    assert bank[60:64].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert bank[64:68].tolist() == [20.0, 21.0, 22.0, 23.0]


def test_fused_suffix_casts_small_target_mask_to_abi_dtype() -> None:
    artifact = _RecordingArtifact(sentinel_loss=2.5, sentinel_ntokens=11.0)
    bank_owner = _FakeBankOwner()
    bank_owner.buffers["bf16"] = mx.zeros((8,), dtype=mx.bfloat16)
    abi_map = _make_abi_map()
    abi_map["target_mask"] = {
        "bank": "bf16", "dtype": "bfloat16", "offset": 0, "size": 3,
        "shape": (3,), "logical_shape": (3,),
    }
    in_region_aliases = {
        "p_a": {
            "logical_name": "param_a",
            "logical_grad_name": "param_a_grad",
            "bank": "f32", "dtype": "float32",
            "offset": 60, "size": 4, "logical_shape": (4,),
        },
    }
    f = build_fused_suffix_custom_function(
        artifact=artifact,
        bank_owner=bank_owner,
        abi_map=abi_map,
        hidden_entry_logical_name="hidden_entry",
        target_ids_logical_name="target_ids",
        target_mask_logical_name="target_mask",
        loss_logical_name="loss",
        ntokens_logical_name="ntokens",
        in_region_parameter_bank_aliases=in_region_aliases,
        parameter_order=("p_a",),
    )

    loss, ntokens = f(
        mx.full((1, 3, 4), 9.0, dtype=mx.float32),
        mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
        mx.array([1.0, 0.0, 1.0], dtype=mx.float32),
        mx.array([10.0, 11.0, 12.0, 13.0], dtype=mx.float32),
    )
    mx.eval(loss, ntokens)

    assert bank_owner.buffers["bf16"][0:3].dtype == mx.bfloat16
    assert bank_owner.buffers["bf16"][0:3].tolist() == [1.0, 0.0, 1.0]


def test_fused_suffix_value_and_grad_returns_bank_view_cotangents() -> None:
    artifact = _RecordingArtifact(sentinel_loss=3.0, sentinel_ntokens=11.0)
    bank_owner = _FakeBankOwner()
    abi_map = _make_abi_map()
    in_region_aliases = {
        "p_a": {
            "logical_name": "param_a", "logical_grad_name": "param_a_grad",
            "bank": "f32", "dtype": "float32",
            "offset": 60, "size": 4, "logical_shape": (4,),
        },
        "p_b": {
            "logical_name": "param_b", "logical_grad_name": "param_b_grad",
            "bank": "f32", "dtype": "float32",
            "offset": 64, "size": 4, "logical_shape": (4,),
        },
    }
    f = build_fused_suffix_custom_function(
        artifact=artifact, bank_owner=bank_owner, abi_map=abi_map,
        hidden_entry_logical_name="hidden_entry",
        target_ids_logical_name="target_ids",
        target_mask_logical_name="target_mask",
        loss_logical_name="loss",
        ntokens_logical_name="ntokens",
        in_region_parameter_bank_aliases=in_region_aliases,
        parameter_order=("p_a", "p_b"),
    )

    hidden_entry = mx.full((1, 3, 4), 1.0, dtype=mx.float32)
    target_ids = mx.zeros((3,), dtype=mx.float32)
    target_mask = mx.ones((3,), dtype=mx.float32)
    param_a = mx.zeros((4,), dtype=mx.float32)
    param_b = mx.zeros((4,), dtype=mx.float32)

    def loss_fn(hidden_entry, param_a, param_b):
        loss, _ntokens = f(hidden_entry, target_ids, target_mask, param_a, param_b)
        return loss

    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))
    val, grads = grad_fn(hidden_entry, param_a, param_b)
    mx.eval(val, *grads)

    assert float(val) == 3.0
    # Three grads — hidden_entry, param_a, param_b. Each is the
    # corresponding bank-view sentinel the artifact wrote.
    assert grads[0].shape == hidden_entry.shape
    assert grads[0].tolist() == [[[7.0]*4]*3]
    assert grads[1].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert grads[2].tolist() == [5.0, 6.0, 7.0, 8.0]


def test_fused_suffix_vjp_scales_bank_view_cotangents_by_loss_cotangent() -> None:
    artifact = _RecordingArtifact(sentinel_loss=3.0, sentinel_ntokens=11.0)
    bank_owner = _FakeBankOwner()
    abi_map = _make_abi_map()
    in_region_aliases = {
        "p_a": {
            "logical_name": "param_a", "logical_grad_name": "param_a_grad",
            "bank": "f32", "dtype": "float32",
            "offset": 60, "size": 4, "logical_shape": (4,),
        },
        "p_b": {
            "logical_name": "param_b", "logical_grad_name": "param_b_grad",
            "bank": "f32", "dtype": "float32",
            "offset": 64, "size": 4, "logical_shape": (4,),
        },
    }
    f = build_fused_suffix_custom_function(
        artifact=artifact, bank_owner=bank_owner, abi_map=abi_map,
        hidden_entry_logical_name="hidden_entry",
        target_ids_logical_name="target_ids",
        target_mask_logical_name="target_mask",
        loss_logical_name="loss",
        ntokens_logical_name="ntokens",
        in_region_parameter_bank_aliases=in_region_aliases,
        parameter_order=("p_a", "p_b"),
    )

    hidden_entry = mx.full((1, 3, 4), 1.0, dtype=mx.float32)
    target_ids = mx.zeros((3,), dtype=mx.float32)
    target_mask = mx.ones((3,), dtype=mx.float32)
    param_a = mx.zeros((4,), dtype=mx.float32)
    param_b = mx.zeros((4,), dtype=mx.float32)

    def scaled_loss_fn(hidden_entry, param_a, param_b):
        loss, _ntokens = f(hidden_entry, target_ids, target_mask, param_a, param_b)
        return loss * mx.array(3.0, dtype=mx.float32)

    grad_fn = mx.value_and_grad(scaled_loss_fn, argnums=(0, 1, 2))
    val, grads = grad_fn(hidden_entry, param_a, param_b)
    mx.eval(val, *grads)

    assert float(val) == 9.0
    assert grads[0].tolist() == [[[21.0]*4]*3]
    assert grads[1].tolist() == [3.0, 6.0, 9.0, 12.0]
    assert grads[2].tolist() == [15.0, 18.0, 21.0, 24.0]


def test_fused_suffix_rejects_wrong_primal_count() -> None:
    artifact = _RecordingArtifact(0.0, 0.0)
    bank_owner = _FakeBankOwner()
    abi_map = _make_abi_map()
    in_region_aliases = {
        "p_a": {
            "logical_name": "param_a", "logical_grad_name": "param_a_grad",
            "bank": "f32", "dtype": "float32",
            "offset": 60, "size": 4, "logical_shape": (4,),
        },
    }
    f = build_fused_suffix_custom_function(
        artifact=artifact, bank_owner=bank_owner, abi_map=abi_map,
        hidden_entry_logical_name="hidden_entry",
        target_ids_logical_name="target_ids",
        target_mask_logical_name="target_mask",
        loss_logical_name="loss",
        ntokens_logical_name="ntokens",
        in_region_parameter_bank_aliases=in_region_aliases,
        parameter_order=("p_a",),
    )
    with pytest.raises(ValueError, match="expected 1"):
        f(
            mx.zeros((1, 3, 4)),
            mx.zeros((3,)),
            mx.zeros((3,)),
            mx.zeros((4,)),
            mx.zeros((4,)),  # one too many
        )
