"""Unit tests for the fused-suffix custom function bridge."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import pytest

from cppmega_mlx.training.path_c_fused_suffix import (
    build_fused_suffix_custom_function,
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
    }


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
