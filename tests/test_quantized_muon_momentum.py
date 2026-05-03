"""Tests for the int8-quantized Muon momentum buffer.

Mirrors cppmega CUDA's
``quantized_muon_momentum_update_multi_and_normalize_groups_`` from
``megatron/core/optimizer/emerging_optimizers.py``: the persistent Muon
momentum is stored as ``uint8`` payload + per-256-block fp32 absmax. The
Newton-Schulz orthogonalization carrier stays fp32 -- only the persistent
state is quantized -- so we verify that round-trip error stays small,
state dtypes match the codec, and the loss trajectory tracks the fp32
baseline within a few percent.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.training._quantize_8bit import (
    DEFAULT_BLOCK_SIZE,
    dequantize_dynamic_blockwise,
    quantize_dynamic_blockwise,
)
from cppmega_mlx.training.optimizers import (
    MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE,
    MUON_QUANTIZED_MOMENTUM_SCHEME,
    MuonAdamWMulti,
    MuonWithNSCarrier,
    QuantizedMuonWithNSCarrier,
    make_muon,
)


def _flatten_arrays(tree: Any) -> list[mx.array]:
    return [v for _, v in tree_flatten(tree) if isinstance(v, mx.array)]


def _state_bytes(tree: Any) -> int:
    return sum(int(arr.size * arr.dtype.size) for arr in _flatten_arrays(tree))


class _MuonOnlyModel(nn.Module):
    """Tiny model whose only trainable params are 2-D Muon-routed weights."""

    def __init__(self, in_dim: int = 32, hidden: int = 48, out_dim: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, out_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.silu(self.fc1(x)))


def _make_smoke_data(
    *, batch: int, in_dim: int, out_dim: int, seed: int
) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    x = mx.random.normal((batch, in_dim)).astype(mx.bfloat16)
    y = mx.random.normal((batch, out_dim)).astype(mx.float32) * 0.1
    mx.eval(x, y)
    return x, y


def _train_n_steps(
    optimizer: Any,
    *,
    in_dim: int,
    hidden: int,
    out_dim: int,
    batch: int,
    steps: int,
    seed: int,
    dtype: mx.Dtype = mx.bfloat16,
) -> list[float]:
    mx.random.seed(seed)
    model = _MuonOnlyModel(in_dim, hidden, out_dim)
    model.set_dtype(dtype)
    optimizer.init(model.trainable_parameters())
    mx.random.seed(seed + 1)
    x = mx.random.normal((batch, in_dim)).astype(dtype)
    y = mx.random.normal((batch, out_dim)).astype(mx.float32) * 0.1
    mx.eval(x, y)

    def loss_fn(m: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        return mx.mean(mx.square(m(x).astype(mx.float32) - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses: list[float] = []
    for _ in range(steps):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
    return losses


def test_quantize_dequantize_momentum_buffer_round_trip_within_tolerance() -> None:
    mx.random.seed(0)
    momentum = (mx.random.normal((4, 256)) * 0.05).astype(mx.float32)
    mx.eval(momentum)
    payload, absmax = quantize_dynamic_blockwise(
        momentum, MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE
    )
    decoded = dequantize_dynamic_blockwise(payload, absmax, out_dtype=mx.float32)
    mx.eval(payload, absmax, decoded)
    err = float(mx.max(mx.abs(momentum - decoded)).item())
    assert err < 1e-2, f"round-trip max abs error {err} exceeds 1e-2 budget"
    assert payload.dtype == mx.uint8
    assert absmax.dtype == mx.float32
    assert MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE == DEFAULT_BLOCK_SIZE
    assert MUON_QUANTIZED_MOMENTUM_SCHEME == "symmetric_int8_v1"


def test_quantized_muon_state_uint8() -> None:
    mx.random.seed(1)
    model = _MuonOnlyModel(in_dim=16, hidden=24, out_dim=16)
    model.set_dtype(mx.bfloat16)
    opt = QuantizedMuonWithNSCarrier(
        learning_rate=1e-3, momentum=0.95, weight_decay=0.0
    )
    opt.init(model.trainable_parameters())

    flat = dict(tree_flatten(opt.state))
    quant_keys = [k for k in flat if k.endswith("v_quant")]
    absmax_keys = [k for k in flat if k.endswith("v_absmax")]
    assert quant_keys, f"no v_quant entries in state: {sorted(flat)}"
    assert absmax_keys, f"no v_absmax entries in state: {sorted(flat)}"
    for key in quant_keys:
        assert flat[key].dtype == mx.uint8, (key, flat[key].dtype)
    for key in absmax_keys:
        assert flat[key].dtype == mx.float32, (key, flat[key].dtype)
    # No fp32 momentum buffer should exist anywhere.
    fp32_v_keys = [k for k in flat if k.endswith(".v") and flat[k].dtype == mx.float32]
    assert not fp32_v_keys, f"unexpected fp32 momentum: {fp32_v_keys}"


def test_quantized_muon_loss_matches_fp32_within_tolerance() -> None:
    # Use a hidden dim that is a multiple of the 256-block size so each
    # quantization block has full population, mirroring the production
    # 1.797B-param shape (hidden=3584). The smoke runs in fp32 to isolate
    # *quantization* drift from the orthogonal bf16-cast drift; the bf16
    # parameter dtype is exercised separately in
    # ``test_quantized_muon_apply_preserves_param_dtype``.
    in_dim, hidden, out_dim = 64, 256, 64
    batch = 32
    steps = 50
    seed = 7

    fp32_opt = MuonWithNSCarrier(
        learning_rate=1e-3, momentum=0.95, weight_decay=0.0, nesterov=True, ns_steps=5
    )
    quant_opt = QuantizedMuonWithNSCarrier(
        learning_rate=1e-3, momentum=0.95, weight_decay=0.0, nesterov=True, ns_steps=5
    )
    fp32_losses = _train_n_steps(
        fp32_opt,
        in_dim=in_dim,
        hidden=hidden,
        out_dim=out_dim,
        batch=batch,
        steps=steps,
        seed=seed,
        dtype=mx.float32,
    )
    quant_losses = _train_n_steps(
        quant_opt,
        in_dim=in_dim,
        hidden=hidden,
        out_dim=out_dim,
        batch=batch,
        steps=steps,
        seed=seed,
        dtype=mx.float32,
    )

    # Training should make progress under both optimizers.
    assert fp32_losses[-1] < fp32_losses[0]
    assert quant_losses[-1] < quant_losses[0]

    # Quantized loss within 2% of fp32 baseline at the final step (mean over
    # the last 5 to dampen single-step noise). On the production 1.797B
    # shape this margin is much tighter because each layer has thousands of
    # 256-blocks; the 2% gate here is the toy-model envelope.
    fp32_tail = sum(fp32_losses[-5:]) / 5
    quant_tail = sum(quant_losses[-5:]) / 5
    rel_diff = abs(quant_tail - fp32_tail) / max(abs(fp32_tail), 1e-9)
    assert rel_diff < 0.02, (
        f"quantized tail loss {quant_tail:.6f} differs from fp32 {fp32_tail:.6f} "
        f"by {rel_diff*100:.2f}%, exceeding 2% budget"
    )


def test_quantized_muon_memory_smaller() -> None:
    mx.random.seed(2)
    model = _MuonOnlyModel(in_dim=32, hidden=64, out_dim=32)
    model.set_dtype(mx.bfloat16)

    fp32_opt = MuonWithNSCarrier(learning_rate=1e-3, weight_decay=0.0)
    fp32_opt.init(model.trainable_parameters())

    quant_opt = QuantizedMuonWithNSCarrier(learning_rate=1e-3, weight_decay=0.0)
    quant_opt.init(model.trainable_parameters())

    fp32_bytes = _state_bytes(fp32_opt.state)
    quant_bytes = _state_bytes(quant_opt.state)
    # The non-momentum scaffolding (step, learning_rate scalars) is shared, so
    # we measure the v / v_quant + v_absmax leaves directly.
    fp32_v = sum(
        int(arr.size * arr.dtype.size)
        for key, arr in tree_flatten(fp32_opt.state)
        if isinstance(arr, mx.array) and key.endswith(".v")
    )
    quant_v = sum(
        int(arr.size * arr.dtype.size)
        for key, arr in tree_flatten(quant_opt.state)
        if isinstance(arr, mx.array) and (key.endswith("v_quant") or key.endswith("v_absmax"))
    )
    ratio = fp32_v / max(quant_v, 1)
    assert ratio >= 3.5, (
        f"expected ~4x reduction on momentum bytes, got {ratio:.2f}x "
        f"(fp32_v={fp32_v} bytes, quant_v={quant_v} bytes)"
    )
    # Total state is also smaller (scaffolding dominated by momentum here).
    assert quant_bytes < fp32_bytes, (
        f"total state did not shrink: fp32={fp32_bytes}, quant={quant_bytes}"
    )


def test_make_muon_quantize_flag_threads_through_multi_optimizer() -> None:
    opt = make_muon(cppmega_cuda_parity=True, quantize_momentum=True)
    assert isinstance(opt, MuonAdamWMulti)
    assert isinstance(opt.muon, QuantizedMuonWithNSCarrier)
    # AdamW group should remain untouched (parallel agent's Adam8bit work
    # is opt-in via scalar_optimizer="adam8bit", not implied by quantize_momentum).
    assert not isinstance(opt.adamw, QuantizedMuonWithNSCarrier)
    # Default routing: quantize_momentum=False keeps the fp32 momentum.
    opt_default = make_muon()
    assert isinstance(opt_default.muon, MuonWithNSCarrier)
    assert not isinstance(opt_default.muon, QuantizedMuonWithNSCarrier)


@pytest.mark.parametrize("nesterov", [True, False])
def test_quantized_muon_apply_preserves_param_dtype(nesterov: bool) -> None:
    """The bf16 parameter dtype must round-trip through the quantized step."""

    mx.random.seed(3)
    model = _MuonOnlyModel(in_dim=8, hidden=16, out_dim=8)
    model.set_dtype(mx.bfloat16)
    opt = QuantizedMuonWithNSCarrier(
        learning_rate=1e-3,
        momentum=0.95,
        weight_decay=0.0,
        nesterov=nesterov,
        ns_steps=3,
    )
    opt.init(model.trainable_parameters())
    x = mx.random.normal((4, 8)).astype(mx.bfloat16)
    y = mx.zeros((4, 8)).astype(mx.float32)

    def loss_fn(m: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        return mx.mean(mx.square(m(x).astype(mx.float32) - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, x, y)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state, loss)
    for key, arr in tree_flatten(model.trainable_parameters()):
        if isinstance(arr, mx.array):
            assert arr.dtype == mx.bfloat16, (key, arr.dtype)
