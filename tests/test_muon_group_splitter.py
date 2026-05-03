"""Tests for the Muon + AdamW splitter and ``make_muon`` factory.

These checks pin down the parameter-routing rules that mirror Megatron
emerging_optimizers' ``_is_nonlinear_or_embedding`` predicate (used by
cppmega CUDA), so we can keep CUDA <-> MLX parity traces aligned.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from cppmega_mlx.training.optimizers import (
    EMBEDDING_LIKE_NAME_HINTS,
    MAMBA_SCALAR_LEAVES,
    AdamWFP32Moments,
    MuonAdamWMulti,
    MuonWithNSCarrier,
    _muon_zeropower_newtonschulz5,
    is_muon_compatible,
    make_muon,
    split_param_groups,
)


def _flatten_keys(tree: object) -> set[str]:
    return {key for key, _ in tree_flatten(tree)}


def _flatten_state(tree: object) -> dict[str, mx.array]:
    return {key: value for key, value in tree_flatten(tree) if isinstance(value, mx.array)}


def test_is_muon_compatible_classifies_2d_linear_as_muon() -> None:
    weight = mx.zeros((8, 4))
    assert is_muon_compatible("layers.0.linear.weight", weight) is True


def test_is_muon_compatible_classifies_embedding_as_adamw() -> None:
    weight = mx.zeros((50257, 768))
    for hint in ("token_embedding.weight", "embed.weight", "wte.weight", "wpe.weight"):
        assert is_muon_compatible(hint, weight) is False, hint


def test_is_muon_compatible_classifies_lm_head_as_adamw() -> None:
    weight = mx.zeros((768, 50257))
    assert is_muon_compatible("lm_head.weight", weight) is False


def test_is_muon_compatible_classifies_mamba_scalars_as_adamw() -> None:
    # 1-D Mamba scalars hit the ndim != 2 short-circuit, but the leaf-name
    # check also has to keep them out of the Muon group. We exercise both.
    a_log = mx.zeros((16,))
    assert is_muon_compatible("layers.0.mamba.A_log", a_log) is False

    # If a Mamba scalar were ever 2-D the leaf-name guard would still keep it
    # in the AdamW bucket, so confirm that path explicitly.
    a_log_2d = mx.zeros((4, 4))
    assert is_muon_compatible("layers.0.mamba.A_log", a_log_2d) is False
    assert is_muon_compatible("layers.0.mamba.D", a_log_2d) is False
    assert is_muon_compatible("layers.0.mamba.dt_bias", a_log_2d) is False


def test_is_muon_compatible_classifies_norm_weights_as_adamw() -> None:
    rmsnorm_weight = mx.zeros((128,))
    for name in (
        "layers.0.norm.weight",
        "norm_final.weight",
        "block.attn.norm.weight",
    ):
        assert is_muon_compatible(name, rmsnorm_weight) is False, name


def test_is_muon_compatible_rejects_3d_tensors() -> None:
    # 3-D+ tensors (e.g. fused QKV, conv-style weights) belong to AdamW
    # because Muon's Newton-Schulz orthogonalisation only handles 2-D.
    big = mx.zeros((4, 4, 4))
    assert is_muon_compatible("block.fused_qkv.weight", big) is False


class _SplitterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 8, bias=True)
        self.embedding = nn.Embedding(10, 4)
        self.lm_head = nn.Linear(8, 10, bias=False)
        # 1-D RMSNorm-shaped weight that sits at the model root
        self.norm_weight = mx.ones((8,))

    def __call__(self, ids: mx.array, x: mx.array) -> mx.array:  # pragma: no cover - shape only
        emb = self.embedding(ids)
        return self.lm_head(self.linear(emb + x))


def test_split_param_groups_partitions_pytree_correctly() -> None:
    model = _SplitterModel()
    params = model.trainable_parameters()
    muon_tree, adamw_tree = split_param_groups(params)

    all_keys = _flatten_keys(params)
    muon_keys = _flatten_keys(muon_tree)
    adamw_keys = _flatten_keys(adamw_tree)

    # Each leaf must end up in exactly one group.
    assert muon_keys.isdisjoint(adamw_keys)
    assert muon_keys | adamw_keys == all_keys

    # Spot-check the routing matches Megatron emerging_optimizers' rule.
    assert "linear.weight" in muon_keys  # 2-D nonlinear weight -> Muon
    assert "linear.bias" in adamw_keys  # 1-D bias -> AdamW
    assert "embedding.weight" in adamw_keys  # name hint forces AdamW
    assert "lm_head.weight" in adamw_keys  # name hint forces AdamW
    assert "norm_weight" in adamw_keys  # 1-D RMSNorm scalar -> AdamW


def test_split_param_groups_handles_empty_groups() -> None:
    # AdamW-only model: no 2-D non-embedding weights at all.
    only_scalars = {"norm.weight": mx.ones((4,))}
    muon_tree, adamw_tree = split_param_groups(only_scalars)
    assert muon_tree == {}
    assert _flatten_keys(adamw_tree) == {"norm.weight"}


def test_make_muon_init_runs_and_state_has_both_buckets() -> None:
    model = _SplitterModel()
    optimizer = make_muon()
    optimizer.init(model.trainable_parameters())

    state = optimizer.state
    assert set(state.keys()) == {"muon", "adamw"}
    # Each bucket must carry the standard optimizer scaffolding plus at least
    # one routed parameter, otherwise the audit tooling will treat the bucket
    # as missing and reject the trace.
    assert "step" in state["muon"]
    assert "step" in state["adamw"]
    assert "linear" in state["muon"]  # 2-D weight ended up in Muon bucket
    assert "embedding" in state["adamw"]  # embedding ended up in AdamW bucket
    assert "lm_head" in state["adamw"]


def test_make_muon_update_step_completes_without_error() -> None:
    model = _SplitterModel()
    optimizer = make_muon()
    optimizer.init(model.trainable_parameters())

    before = {
        key: mx.array(value)
        for key, value in tree_flatten(model.trainable_parameters())
        if isinstance(value, mx.array)
    }

    ids = mx.array([[0, 1, 2, 3]])
    x = mx.zeros((1, 4, 4))

    def loss_fn(m: nn.Module, ids: mx.array, x: mx.array) -> mx.array:
        return m(ids, x).sum()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, ids, x)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)

    after = dict(tree_flatten(model.trainable_parameters()))
    assert math.isfinite(float(loss.item()))
    # All parameters that received a gradient should have moved.
    grad_keys = {key for key, _ in tree_flatten(grads)}
    for key in grad_keys:
        before_arr = before[key]
        after_arr = after[key]
        delta = mx.max(mx.abs(after_arr - before_arr)).item()
        assert delta > 0.0, f"parameter {key!r} did not change after update"


class _RecordingOptimizer:
    def __init__(self) -> None:
        self.calls: list[tuple[set[str], set[str]]] = []
        self.state: dict[str, object] = {}
        self.learning_rate = mx.array(0.0)

    def init(self, parameters: object) -> None:
        self.state = {"keys": sorted(_flatten_keys(parameters))}

    def apply_gradients(self, gradients: object, parameters: object) -> object:
        grad_keys = _flatten_keys(gradients)
        param_keys = _flatten_keys(parameters)
        self.calls.append((grad_keys, param_keys))
        assert param_keys == grad_keys
        return parameters


def test_muon_adamw_multi_passes_routed_parameter_subtrees_to_suboptimizers() -> None:
    model = _SplitterModel()
    params = model.trainable_parameters()
    gradients = tree_unflatten(
        [
            (key, mx.ones_like(value))
            for key, value in tree_flatten(params)
            if isinstance(value, mx.array)
        ]
    )
    muon = _RecordingOptimizer()
    adamw = _RecordingOptimizer()
    optimizer = MuonAdamWMulti(muon, adamw)

    updates = optimizer.apply_gradients(gradients, params)

    assert _flatten_keys(updates) == _flatten_keys(params)
    assert muon.calls == [({"linear.weight"}, {"linear.weight"})]
    assert adamw.calls == [
        (
            {"embedding.weight", "linear.bias", "lm_head.weight", "norm_weight"},
            {"embedding.weight", "linear.bias", "lm_head.weight", "norm_weight"},
        )
    ]


def test_make_muon_cppmega_cuda_parity_forces_shared_lr() -> None:
    optimizer = make_muon(cppmega_cuda_parity=True)
    expected = pytest.approx(1e-4, rel=1e-4)
    assert float(optimizer.muon.learning_rate) == expected
    assert float(optimizer.adamw.learning_rate) == expected
    assert isinstance(optimizer.muon, MuonWithNSCarrier)
    assert optimizer.muon.nesterov is False
    assert isinstance(optimizer.adamw, AdamWFP32Moments)
    # Parity mode also pins AdamW's betas to the cppmega CUDA defaults.
    assert list(optimizer.adamw.betas) == [0.9, 0.999]


def test_make_muon_default_lrs_match_keller_recipe() -> None:
    optimizer = make_muon()
    assert float(optimizer.muon.learning_rate) == pytest.approx(2e-3, rel=1e-4)
    assert float(optimizer.adamw.learning_rate) == pytest.approx(1e-4, rel=1e-4)


def test_make_muon_ns_carrier_defaults_to_fp32_state_for_bf16_params() -> None:
    model = _SplitterModel()
    model.set_dtype(mx.bfloat16)
    optimizer = make_muon()
    optimizer.init(model.trainable_parameters())

    state = _flatten_state(optimizer.state["muon"])
    assert state["linear.weight.v"].dtype == mx.float32


def test_make_muon_accepts_bf16_ns_carrier_and_preserves_param_dtype() -> None:
    model = _SplitterModel()
    model.set_dtype(mx.bfloat16)
    optimizer = make_muon(ns_carrier="bf16")
    optimizer.init(model.trainable_parameters())

    ids = mx.array([[0, 1, 2, 3]])
    x = mx.zeros((1, 4, 4), dtype=mx.bfloat16)

    def loss_fn(m: nn.Module, ids: mx.array, x: mx.array) -> mx.array:
        return m(ids, x).sum()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, ids, x)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)

    state = _flatten_state(optimizer.state["muon"])
    params = dict(tree_flatten(model.trainable_parameters()))
    assert state["linear.weight.v"].dtype == mx.float32
    assert params["linear.weight"].dtype == mx.bfloat16
    assert math.isfinite(float(loss.item()))


def test_make_muon_honors_ns_carrier_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CPPMEGA_MUON_NS_CARRIER", "bf16")
    optimizer = make_muon(ns_carrier="fp32")
    assert isinstance(optimizer.muon, MuonWithNSCarrier)
    assert optimizer.muon.ns_carrier == "bf16"


@pytest.mark.parametrize("value", ["fp16", "", "float32"])
def test_make_muon_rejects_invalid_ns_carrier(value: str) -> None:
    with pytest.raises(ValueError, match="ns_carrier must be one of"):
        make_muon(ns_carrier=value)


def test_make_muon_rejects_invalid_ns_carrier_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CPPMEGA_MUON_NS_CARRIER", "fp16")
    with pytest.raises(ValueError, match="ns_carrier must be one of"):
        make_muon()


def test_muon_ns_carrier_does_not_call_mlx_private_ns_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_private_helper(*_args: object, **_kwargs: object) -> mx.array:
        raise AssertionError("MLX private Newton-Schulz helper should not be called")

    monkeypatch.setattr(
        MuonWithNSCarrier,
        "_zeropower_via_newtonschulz5",
        fail_private_helper,
        raising=True,
    )
    optimizer = MuonWithNSCarrier(
        learning_rate=1e-3,
        momentum=0.95,
        nesterov=True,
        ns_steps=2,
        ns_carrier="bf16",
    )
    state: dict[str, mx.array] = {}
    parameter = mx.ones((4, 4), dtype=mx.bfloat16)
    gradient = mx.full((4, 4), 0.125, dtype=mx.bfloat16)
    optimizer.init_single(parameter, state)

    updated = optimizer.apply_single(gradient, parameter, state)
    mx.eval(updated, state)

    assert updated.dtype == parameter.dtype


def test_muon_bf16_ns_carrier_matches_fp32_orthogonalization() -> None:
    mx.random.seed(0)
    update = mx.random.normal((256, 256)).astype(mx.float32) * 0.02
    fp32_orthogonalized = _muon_zeropower_newtonschulz5(
        update,
        steps=5,
        ns_carrier="fp32",
        output_dtype=mx.float32,
    )
    bf16_orthogonalized = _muon_zeropower_newtonschulz5(
        update,
        steps=5,
        ns_carrier="bf16",
        output_dtype=mx.float32,
    )
    mx.eval(fp32_orthogonalized, bf16_orthogonalized)

    max_abs = mx.max(mx.abs(fp32_orthogonalized - bf16_orthogonalized))
    assert float(max_abs.item()) <= 1e-2


def test_make_muon_returns_muon_adamw_multi() -> None:
    optimizer = make_muon()
    assert isinstance(optimizer, MuonAdamWMulti)


def test_embedding_and_mamba_constants_are_immutable() -> None:
    # Sanity: the gates that drive routing are documented constants that the
    # audit tool inspects, so they need to stay frozen tuples/sets.
    assert isinstance(EMBEDDING_LIKE_NAME_HINTS, tuple)
    assert isinstance(MAMBA_SCALAR_LEAVES, frozenset)
    # A few canonical leaf names that cppmega CUDA's predicate also pins down.
    for leaf in ("A_log", "D", "dt_bias", "B_bias", "C_bias", "mimo_x"):
        assert leaf in MAMBA_SCALAR_LEAVES
