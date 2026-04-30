from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from cppmega_mlx.nn.moe import MoEConfig, ReferenceMoE, TopKRouter


def _tiny_config(
    *,
    d_model: int = 8,
    num_experts: int = 4,
    top_k: int = 2,
    shared: bool = True,
) -> MoEConfig:
    return MoEConfig(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        expert_hidden_size=12,
        shared_expert_hidden_size=10 if shared else None,
        activation="swiglu",
    )


def test_moe_config_matches_nam56r_defaults() -> None:
    cfg = MoEConfig(d_model=32)

    assert cfg.num_experts == 16
    assert cfg.top_k == 4
    assert cfg.expert_hidden_size == 896
    assert cfg.shared_expert_hidden_size == 1024
    assert cfg.router_dtype == "fp32"


def test_topk_router_shapes_and_probabilities() -> None:
    router = TopKRouter(d_model=8, num_experts=4, top_k=2)
    x = mx.random.normal((2, 3, 8))

    out = router(x)
    mx.eval(out.logits, out.probabilities, out.top_indices, out.top_weights, out.aux_loss)

    assert out.logits.shape == (2, 3, 4)
    assert out.probabilities.shape == (2, 3, 4)
    assert out.top_indices.shape == (2, 3, 2)
    assert out.top_weights.shape == (2, 3, 2)
    assert out.load.shape == (4,)
    assert out.importance.shape == (4,)

    np.testing.assert_allclose(
        np.array(out.probabilities.sum(axis=-1)),
        np.ones((2, 3), dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(out.top_weights.sum(axis=-1)),
        np.ones((2, 3), dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    assert int(mx.min(out.top_indices).item()) >= 0
    assert int(mx.max(out.top_indices).item()) < 4
    assert math.isfinite(float(out.aux_loss.item()))


def test_topk_router_selects_expected_experts_and_weights_from_probabilities() -> None:
    router = TopKRouter(d_model=2, num_experts=3, top_k=2, bias=True)
    router.gate.weight = mx.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=mx.float32,
    )
    router.gate.bias = mx.zeros((3,), dtype=mx.float32)
    x = mx.array([[[2.0, 1.0], [-1.0, 3.0]]], dtype=mx.float32)

    out = router(x)
    mx.eval(
        out.logits,
        out.probabilities,
        out.top_indices,
        out.top_weights,
        out.load,
        out.importance,
    )

    np.testing.assert_allclose(
        np.array(out.logits),
        np.array([[[2.0, 1.0, -2.0], [-1.0, 3.0, 1.0]]], dtype=np.float32),
        rtol=0,
        atol=0,
    )
    top_indices = np.array(out.top_indices)
    np.testing.assert_array_equal(
        np.sort(top_indices, axis=-1),
        np.array([[[0, 1], [1, 2]]], dtype=top_indices.dtype),
    )

    probabilities = np.array(out.probabilities).reshape(2, 3)
    selected = top_indices.reshape(2, 2)
    gathered = np.take_along_axis(probabilities, selected, axis=-1)
    expected_weights = gathered / gathered.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(
        np.array(out.top_weights).reshape(2, 2),
        expected_weights,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(out.load),
        np.array([0.25, 0.5, 0.25], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(out.importance),
        probabilities.mean(axis=0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_reference_moe_output_shape_and_aux_info() -> None:
    moe = ReferenceMoE(_tiny_config())
    x = mx.random.normal((2, 5, 8))

    out = moe(x)
    mx.eval(
        out.output,
        out.routed_output,
        out.shared_output,
        out.router.top_indices,
        out.router.top_weights,
    )

    assert out.output.shape == x.shape
    assert out.routed_output.shape == x.shape
    assert out.shared_output is not None
    assert out.shared_output.shape == x.shape
    assert out.router.top_indices.shape == (2, 5, 2)
    assert out.router.top_weights.shape == (2, 5, 2)
    assert np.isfinite(np.array(out.output)).all()


def test_reference_moe_without_shared_expert() -> None:
    moe = ReferenceMoE(_tiny_config(shared=False))
    x = mx.random.normal((3, 4, 8))

    out = moe(x)
    mx.eval(out.output, out.routed_output)

    assert out.output.shape == x.shape
    assert out.shared_output is None
    np.testing.assert_allclose(
        np.array(out.output),
        np.array(out.routed_output),
        rtol=0,
        atol=0,
    )


class TinyMoERegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = ReferenceMoE(_tiny_config(d_model=6, num_experts=4, top_k=2))
        self.head = nn.Linear(6, 3, bias=False)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        moe_out = self.moe(x)
        return self.head(moe_out.output), moe_out.router.aux_loss


def _loss_fn(model: TinyMoERegressor, batch: tuple[mx.array, mx.array]) -> mx.array:
    x, target = batch
    pred, aux_loss = model(x)
    return mx.mean(mx.square(pred - target)) + 0.01 * aux_loss


def _flat_tree(tree) -> dict[str, np.ndarray]:
    mx.eval(tree)
    return {name: np.array(value) for name, value in tree_flatten(tree)}


def _assert_finite_nonzero(tree: dict[str, np.ndarray], name: str) -> None:
    assert name in tree
    assert np.isfinite(tree[name]).all(), name
    assert float(np.max(np.abs(tree[name]))) > 0.0, name


def _assert_optimizer_state_for(
    optimizer_state: dict[str, np.ndarray],
    param_name: str,
) -> None:
    _assert_finite_nonzero(optimizer_state, f"{param_name}.m")
    _assert_finite_nonzero(optimizer_state, f"{param_name}.v")


def _force_identity_router(router: TopKRouter) -> None:
    weight = np.zeros((router.num_experts, router.d_model), dtype=np.float32)
    for expert_id in range(router.num_experts):
        weight[expert_id, expert_id] = 1.0
    router.gate.weight = mx.array(weight)


def test_reference_moe_finite_train_step() -> None:
    model = TinyMoERegressor()
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    x = mx.random.normal((2, 4, 6))
    target = mx.random.normal((2, 4, 3))
    loss_and_grad = nn.value_and_grad(model, _loss_fn)

    loss_before, grads = loss_and_grad(model, (x, target))
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss_before)

    loss_after = _loss_fn(model, (x, target))
    mx.eval(loss_after)

    assert math.isfinite(float(loss_before.item()))
    assert math.isfinite(float(loss_after.item()))
    assert float(loss_before.item()) > 0
    assert float(loss_after.item()) > 0


def test_reference_moe_train_step_reaches_all_routed_and_shared_experts() -> None:
    mx.random.seed(401)
    model = TinyMoERegressor()
    _force_identity_router(model.moe.router)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    x = mx.array(
        [
            [
                [2.0, 1.0, 0.0, -1.0, 0.5, -0.5],
                [0.0, -1.0, 2.0, 1.0, -0.25, 0.75],
                [1.0, 0.0, 2.0, -1.0, 0.25, 0.5],
                [-1.0, 2.0, 0.0, 1.0, 0.5, -0.25],
            ],
            [
                [2.0, 1.0, 0.0, -1.0, -0.5, 0.25],
                [0.0, -1.0, 2.0, 1.0, 0.75, -0.25],
                [1.0, 0.0, 2.0, -1.0, -0.75, 0.5],
                [-1.0, 2.0, 0.0, 1.0, 0.25, -0.5],
            ],
        ],
        dtype=mx.float32,
    )
    target = mx.random.normal((2, 4, 3))
    loss_and_grad = nn.value_and_grad(model, _loss_fn)
    before = _flat_tree(model.parameters())
    router_out = model.moe(x).router
    mx.eval(router_out.top_indices)
    used_experts = set(np.array(router_out.top_indices).reshape(-1).tolist())

    loss, grads = loss_and_grad(model, (x, target))
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)
    flat_grads = _flat_tree(grads)
    after = _flat_tree(model.parameters())
    optimizer_state = _flat_tree(optimizer.state)

    assert math.isfinite(float(loss.item()))
    assert used_experts == set(range(model.moe.config.num_experts))
    _assert_finite_nonzero(flat_grads, "moe.router.gate.weight")
    _assert_finite_nonzero(flat_grads, "moe.shared_expert.down_proj.weight")
    _assert_finite_nonzero(flat_grads, "moe.shared_expert.gate_proj.weight")
    _assert_finite_nonzero(flat_grads, "moe.shared_expert.up_proj.weight")
    for expert_id in range(model.moe.config.num_experts):
        _assert_finite_nonzero(flat_grads, f"moe.experts.{expert_id}.gate_proj.weight")
        _assert_finite_nonzero(flat_grads, f"moe.experts.{expert_id}.up_proj.weight")
        _assert_finite_nonzero(flat_grads, f"moe.experts.{expert_id}.down_proj.weight")

    updated_names = [
        "moe.router.gate.weight",
        "moe.shared_expert.gate_proj.weight",
        "moe.shared_expert.up_proj.weight",
        "moe.shared_expert.down_proj.weight",
    ]
    for expert_id in range(model.moe.config.num_experts):
        updated_names.extend(
            [
                f"moe.experts.{expert_id}.gate_proj.weight",
                f"moe.experts.{expert_id}.up_proj.weight",
                f"moe.experts.{expert_id}.down_proj.weight",
            ]
        )
    for name in updated_names:
        delta = after[name] - before[name]
        assert float(np.max(np.abs(delta))) > 0.0, name
        _assert_optimizer_state_for(optimizer_state, name)
