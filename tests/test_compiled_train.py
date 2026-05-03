from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
import pytest

from cppmega_mlx.data.batch import synthetic_token_batch
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM
from cppmega_mlx.models.tiny_lm import TinyLM, TinyLMConfig
from cppmega_mlx.training.compiled import (
    CompiledPretrainingStep,
    PretrainingMetrics,
    PretrainingState,
    REGIONAL_COMPILE_TARGETS,
    STABLE_BATCH_KEYS,
    maybe_compile_region,
    normalize_compiled_batch,
    regional_compile,
    should_compile_region,
)


def _tiny_config() -> TinyLMConfig:
    return TinyLMConfig(
        vocab_size=48,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=32,
        max_seq_length=16,
        structure_vocab_size=16,
    )


def _hybrid_config(*, pattern: str = "AEMR", depth: int = 4) -> HybridTinyConfig:
    return HybridTinyConfig(
        vocab_size=48,
        hidden_size=8,
        pattern=pattern,
        depth=depth,
        num_attention_heads=1,
        max_seq_length=8,
        structure_vocab_size=16,
        moe_num_experts=2,
        moe_top_k=1,
        moe_expert_hidden_size=16,
        moe_shared_expert_hidden_size=8,
        mamba_expand=1,
        mamba_head_dim=4,
        mamba_state_dim=4,
        mamba_groups=1,
        mamba_chunk_size=4,
        m2rnn_k_head_dim=2,
        m2rnn_v_head_dim=2,
        m2rnn_num_v_heads=1,
        m2rnn_num_f_heads=1,
        m2rnn_chunk_size=4,
    )


def _flat_params(model: nn.Module) -> dict[str, np.ndarray]:
    mx.eval(model.parameters())
    return {name: np.array(value) for name, value in tree_flatten(model.parameters())}


def _has_parameter_delta(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> bool:
    return any(np.max(np.abs(after[name] - before[name])) > 0 for name in before)


def _has_parameter_delta_under(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    prefix: str,
) -> bool:
    return any(
        name.startswith(prefix) and np.max(np.abs(after[name] - before[name])) > 0
        for name in before
    )


def _assert_params_allclose(
    actual: nn.Module,
    expected: nn.Module,
    *,
    atol: float = 1e-7,
) -> None:
    actual_params = _flat_params(actual)
    expected_params = _flat_params(expected)
    assert actual_params.keys() == expected_params.keys()
    for name, expected_value in expected_params.items():
        np.testing.assert_allclose(actual_params[name], expected_value, rtol=0, atol=atol)


def test_regional_compile_policy_matches_cppmega_receipt() -> None:
    assert REGIONAL_COMPILE_TARGETS == {
        "mamba3_pre": True,
        "data_dep_a": True,
        "rmsnorm": False,
        "rmsnorm_gated": False,
        "moe_router": False,
    }
    assert should_compile_region("data_dep_a") is True
    assert should_compile_region("mamba3_pre") is True
    assert should_compile_region("rmsnorm") is False
    assert should_compile_region("rmsnorm_gated") is False
    assert should_compile_region("moe_router") is False


def test_regional_compile_rejects_unknown_targets_fail_closed() -> None:
    bad_target: Any = "attention"

    with pytest.raises(ValueError, match="unknown regional compile target"):
        should_compile_region(bad_target)
    with pytest.raises(ValueError, match="unknown regional compile target"):
        regional_compile(bad_target, lambda x: x)


def test_regional_compile_allowed_target_calls_mx_compile(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_compile(fn: Any, **kwargs: Any) -> Any:
        calls.append({"fn": fn, "kwargs": kwargs})

        def wrapper(x: mx.array) -> mx.array:
            return fn(x) + 1

        return wrapper

    monkeypatch.setattr("cppmega_mlx.training.compiled.mx.compile", fake_compile)

    @regional_compile("data_dep_a", shapeless=True)
    def add_two(x: mx.array) -> mx.array:
        return x + 2

    result = add_two(mx.array(3))

    assert len(calls) == 1
    assert calls[0]["kwargs"] == {"shapeless": True}
    assert int(result.item()) == 6


def test_regional_compile_denied_target_returns_original_function(
    monkeypatch: Any,
) -> None:
    calls: list[Any] = []

    def fake_compile(fn: Any, **kwargs: Any) -> Any:
        calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr("cppmega_mlx.training.compiled.mx.compile", fake_compile)

    def rmsnorm_like(x: mx.array) -> mx.array:
        return x * 2

    maybe_compiled = maybe_compile_region("rmsnorm", rmsnorm_like, shapeless=True)
    result = maybe_compiled(mx.array(3))

    assert maybe_compiled is rmsnorm_like
    assert calls == []
    assert int(result.item()) == 6


def test_compiled_pretraining_step_updates_tiny_lm_and_state() -> None:
    model = TinyLM(_tiny_config())
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    step = CompiledPretrainingStep(model, optimizer, compile=True)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=model.config.vocab_size,
        seed=17,
        include_structure=True,
    )
    before = _flat_params(model)

    first = step(batch)
    second = step(batch.as_dict())
    after = _flat_params(model)

    assert first.compiled is True
    assert first.updated is True
    assert first.ntokens == 14
    assert first.step == 1
    assert first.trained_tokens == 14
    assert second.step == 2
    assert second.trained_tokens == 28
    assert math.isfinite(first.loss)
    assert math.isfinite(second.loss)
    assert first.loss > 0
    assert second.loss > 0
    assert step.state.to_dict() == {"step": 2, "trained_tokens": 28}
    assert _has_parameter_delta(before, after)


def test_compiled_pretraining_step_accumulates_until_boundary() -> None:
    model = TinyLM(_tiny_config())
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    step = CompiledPretrainingStep(model, optimizer, compile=True, grad_accum_steps=3)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=model.config.vocab_size,
        seed=23,
        include_structure=True,
    )
    before = _flat_params(model)

    first = step(batch)
    after_first = _flat_params(model)
    second = step(batch.as_dict())
    after_second = _flat_params(model)
    third = step(batch)
    after_third = _flat_params(model)

    assert first.updated is False
    assert second.updated is False
    assert third.updated is True
    assert first.ntokens == 14
    assert second.ntokens == 14
    assert third.ntokens == 14
    assert math.isfinite(first.loss)
    assert math.isfinite(second.loss)
    assert math.isfinite(third.loss)
    assert first.loss > 0
    assert second.loss > 0
    assert third.loss > 0
    assert first.step == 1
    assert second.step == 2
    assert third.step == 3
    assert third.trained_tokens == 42
    assert step.state.to_dict() == {"step": 3, "trained_tokens": 42}
    assert not _has_parameter_delta(before, after_first)
    assert not _has_parameter_delta(before, after_second)
    assert _has_parameter_delta(before, after_third)


def test_normalize_compiled_batch_uses_fixed_keys_for_optional_fields() -> None:
    model = TinyLM(_tiny_config())
    plain = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=model.config.vocab_size,
        seed=29,
        include_structure=False,
    )
    structured = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=model.config.vocab_size,
        seed=31,
        include_structure=True,
    )

    plain_dict = normalize_compiled_batch(plain.tokens)
    structured_dict = normalize_compiled_batch(structured.as_dict())

    assert tuple(plain_dict) == STABLE_BATCH_KEYS
    assert tuple(structured_dict) == STABLE_BATCH_KEYS
    plain_tokens = plain_dict["tokens"]
    assert plain_tokens is not None
    assert plain_tokens.shape == (2, 8)
    assert plain_dict["attention_mask"] is None
    assert plain_dict["structure_ids"] is None
    assert structured_dict["attention_mask"] is not None
    assert structured_dict["structure_ids"] is not None


def test_normalize_compiled_batch_does_not_mutate_mapping_inputs() -> None:
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=_tiny_config().vocab_size,
        seed=33,
        include_structure=True,
    ).as_dict()
    original_keys = tuple(batch)

    normalized = normalize_compiled_batch(batch)
    normalized["attention_mask"] = None

    assert tuple(batch) == original_keys
    assert batch["attention_mask"] is not None


def test_compiled_pretraining_step_rejects_shape_churn() -> None:
    model = TinyLM(_tiny_config())
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    step = CompiledPretrainingStep(model, optimizer, compile=True)
    first_batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=model.config.vocab_size,
        seed=35,
        include_structure=False,
    )
    second_batch = synthetic_token_batch(
        batch_size=2,
        seq_length=10,
        vocab_size=model.config.vocab_size,
        seed=36,
        include_structure=False,
    )

    step(first_batch)

    with pytest.raises(ValueError, match="fixed batch shape"):
        step(second_batch)


def test_compiled_pretraining_step_rejects_optional_structure_churn() -> None:
    model = TinyLM(_tiny_config())
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    step = CompiledPretrainingStep(model, optimizer, compile=True)
    plain_batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=model.config.vocab_size,
        seed=35,
        include_structure=False,
    )
    structured_batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=model.config.vocab_size,
        seed=36,
        include_structure=True,
    )

    step(plain_batch)

    with pytest.raises(ValueError, match="fixed batch shape"):
        step(structured_batch)


def test_grad_accum_boundary_is_local_when_resuming_state() -> None:
    model = TinyLM(_tiny_config())
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    state = PretrainingState.from_dict({"step": 5, "trained_tokens": 70})
    step = CompiledPretrainingStep(
        model,
        optimizer,
        state=state,
        compile=True,
        grad_accum_steps=3,
    )
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=model.config.vocab_size,
        seed=37,
        include_structure=False,
    )
    before = _flat_params(model)

    first = step(batch)
    after_first = _flat_params(model)
    second = step(batch)
    after_second = _flat_params(model)
    third = step(batch)
    after_third = _flat_params(model)

    assert first.step == 6
    assert second.step == 7
    assert third.step == 8
    assert first.updated is False
    assert second.updated is False
    assert third.updated is True
    assert not _has_parameter_delta(before, after_first)
    assert not _has_parameter_delta(before, after_second)
    assert _has_parameter_delta(before, after_third)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"step": True}, "step"),
        ({"step": "1"}, "step"),
        ({"step": -1}, "step"),
        ({"trained_tokens": False}, "trained_tokens"),
        ({"trained_tokens": "14"}, "trained_tokens"),
        ({"trained_tokens": -14}, "trained_tokens"),
    ],
)
def test_pretraining_state_from_dict_rejects_coerced_resume_cursors(
    payload: dict[str, Any],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        PretrainingState.from_dict(payload)


def test_compiled_pretraining_step_rejects_non_boolean_compile_flag() -> None:
    model = TinyLM(_tiny_config())
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    bad_compile: Any = "yes"

    with pytest.raises(TypeError, match="compile must be a boolean"):
        CompiledPretrainingStep(model, optimizer, compile=bad_compile)


@pytest.mark.parametrize("grad_accum_steps", [True, "1", 0])
def test_compiled_pretraining_step_rejects_coerced_grad_accum_steps(
    grad_accum_steps: Any,
) -> None:
    model = TinyLM(_tiny_config())
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)

    with pytest.raises(ValueError, match="grad_accum_steps"):
        CompiledPretrainingStep(
            model,
            optimizer,
            compile=False,
            grad_accum_steps=grad_accum_steps,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("grad_accum_steps", "1", "grad_accum_steps"),
        ("pending_microbatches", "0", "pending_microbatches"),
        (
            "gradient_accumulator_present",
            "false",
            "gradient_accumulator_present",
        ),
    ],
)
def test_compiled_pretraining_step_load_state_dict_rejects_coerced_fields(
    field: str,
    value: Any,
    error: str,
) -> None:
    model = TinyLM(_tiny_config())
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    step = CompiledPretrainingStep(model, optimizer, compile=False)
    payload = step.state_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=error):
        step.load_state_dict(payload)


def test_compiled_pretraining_state_roundtrips_at_update_boundary() -> None:
    model = TinyLM(_tiny_config())
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    source = CompiledPretrainingStep(
        model,
        optimizer,
        state={"step": 4, "trained_tokens": 56},
        compile=True,
        grad_accum_steps=2,
    )
    target = CompiledPretrainingStep(
        model,
        optimizer,
        compile=True,
        grad_accum_steps=2,
    )

    target.load_state_dict(source.state_dict())

    assert target.state.to_dict() == {"step": 4, "trained_tokens": 56}
    assert target.state_dict()["pending_microbatches"] == 0
    assert target.state_dict()["gradient_accumulator_present"] is False


def test_eager_pretraining_step_is_resume_state_compatible() -> None:
    model = TinyLM(_tiny_config())
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    state = PretrainingState.from_dict({"step": 5, "trained_tokens": 70})
    step = CompiledPretrainingStep(model, optimizer, state=state, compile=False)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=model.config.vocab_size,
        seed=19,
        include_structure=False,
    )
    before = _flat_params(model)

    metrics = step(batch.tokens)
    after = _flat_params(model)

    assert metrics.compiled is False
    assert metrics.updated is True
    assert metrics.ntokens == 14
    assert metrics.step == 6
    assert metrics.trained_tokens == 84
    assert math.isfinite(metrics.loss)
    assert metrics.loss > 0
    assert state.to_dict() == {"step": 6, "trained_tokens": 84}
    assert _has_parameter_delta(before, after)


def test_hybrid_tiny_mamba3_m2rnn_compiled_and_eager_one_step_parity() -> None:
    config = _hybrid_config(pattern="MR", depth=2)
    batch = synthetic_token_batch(
        batch_size=1,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=41,
        include_structure=True,
    )

    def make_stepper(*, compile: bool) -> CompiledPretrainingStep:
        mx.random.seed(123)
        model = HybridTinyLM(config)
        optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
        mx.eval(model.state, optimizer.state)
        return CompiledPretrainingStep(model, optimizer, compile=compile)

    eager = make_stepper(compile=False)(batch)
    compiled = make_stepper(compile=True)(batch.as_dict())

    assert eager.compiled is False
    assert compiled.compiled is True
    assert eager.updated is True
    assert compiled.updated is True
    assert tuple(config.expanded_pattern().symbols) == ("M", "R")
    assert eager.ntokens == compiled.ntokens == 7
    assert eager.step == compiled.step == 1
    assert eager.trained_tokens == compiled.trained_tokens == 7
    assert math.isfinite(eager.loss)
    assert math.isfinite(compiled.loss)
    assert eager.loss > 0
    assert compiled.loss > 0
    assert math.isclose(compiled.loss, eager.loss, rel_tol=1e-5, abs_tol=1e-6)


def test_hybrid_tiny_mamba3_and_m2rnn_compiled_one_step_parity() -> None:
    for offset, pattern in enumerate(("M", "R")):
        config = _hybrid_config(pattern=pattern, depth=1)
        batch = synthetic_token_batch(
            batch_size=1,
            seq_length=8,
            vocab_size=config.vocab_size,
            seed=53 + offset,
            include_structure=True,
        )

        def run_step(
            *,
            compile: bool,
        ) -> tuple[PretrainingMetrics, dict[str, np.ndarray], dict[str, np.ndarray]]:
            mx.random.seed(211 + offset)
            model = HybridTinyLM(config)
            optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
            mx.eval(model.state, optimizer.state)
            before = _flat_params(model)
            metrics = CompiledPretrainingStep(
                model,
                optimizer,
                compile=compile,
            )(batch.as_dict() if compile else batch)
            after = _flat_params(model)
            return metrics, before, after

        eager, eager_before, eager_after = run_step(compile=False)
        compiled, compiled_before, compiled_after = run_step(compile=True)

        assert eager.compiled is False, pattern
        assert compiled.compiled is True, pattern
        assert eager.updated is True, pattern
        assert compiled.updated is True, pattern
        assert eager.ntokens == compiled.ntokens == 7, pattern
        assert eager.step == compiled.step == 1, pattern
        assert eager.trained_tokens == compiled.trained_tokens == 7, pattern
        assert math.isfinite(eager.loss), pattern
        assert math.isfinite(compiled.loss), pattern
        assert eager.loss > 0, pattern
        assert compiled.loss > 0, pattern
        assert _has_parameter_delta(eager_before, eager_after), pattern
        assert _has_parameter_delta(compiled_before, compiled_after), pattern
        assert math.isclose(compiled.loss, eager.loss, rel_tol=1e-5, abs_tol=1e-6), pattern


def test_hybrid_tiny_mamba3_m2rnn_compiled_step_updates_both_custom_routes() -> None:
    config = _hybrid_config(pattern="MR", depth=2)
    batch = synthetic_token_batch(
        batch_size=1,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=73,
        include_structure=True,
    )
    mx.random.seed(307)
    model = HybridTinyLM(config)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    mx.eval(model.state, optimizer.state)
    before = _flat_params(model)

    metrics = CompiledPretrainingStep(model, optimizer, compile=True)(batch.as_dict())
    after = _flat_params(model)

    assert tuple(model.route_symbols) == ("M", "R")
    assert tuple(model.route_roles) == ("mamba3", "m2rnn")
    assert metrics.compiled is True
    assert metrics.updated is True
    assert metrics.ntokens == 7
    assert metrics.step == 1
    assert metrics.trained_tokens == 7
    assert math.isfinite(metrics.loss)
    assert metrics.loss > 0
    assert _has_parameter_delta_under(
        before,
        after,
        "layers.0.mamba3_block.",
    )
    assert _has_parameter_delta_under(
        before,
        after,
        "layers.1.m2rnn_block.",
    )


def test_hybrid_tiny_mamba3_only_compiled_step_updates_m_route_parameters() -> None:
    config = _hybrid_config(pattern="M", depth=1)
    batch = synthetic_token_batch(
        batch_size=1,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=79,
        include_structure=True,
    )
    mx.random.seed(313)
    model = HybridTinyLM(config)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    mx.eval(model.state, optimizer.state)
    before = _flat_params(model)

    metrics = CompiledPretrainingStep(model, optimizer, compile=True)(batch.as_dict())
    after = _flat_params(model)

    assert tuple(model.route_symbols) == ("M",)
    assert tuple(model.route_roles) == ("mamba3",)
    assert metrics.compiled is True
    assert metrics.updated is True
    assert metrics.ntokens == 7
    assert metrics.step == 1
    assert metrics.trained_tokens == 7
    assert math.isfinite(metrics.loss)
    assert metrics.loss > 0
    for prefix in (
        "layers.0.mamba3_block.in_proj.",
        "layers.0.mamba3_block.out_proj.",
        "layers.0.mamba3_block.conv_",
        "layers.0.mamba3_block.dt_bias",
        "layers.0.mamba3_block.B_norm_weight",
        "layers.0.mamba3_block.C_norm_weight",
        "layers.0.mamba3_block.B_bias",
        "layers.0.mamba3_block.C_bias",
        "layers.0.mamba3_block.D",
    ):
        assert _has_parameter_delta_under(before, after, prefix), prefix
