"""PathCFusedPlusEagerTrainingRuntime contract surface tests.

The mixed-mode runtime is the seam that flips the m04 install gate from
'fused_train_block_training_runtime_incomplete' to 'ok' when the fused
artifact covers only a subset of the model's trainable parameters (the
HybridTinyLM case today). It does this by taking ownership of the
training step and routing residual parameters through the trainer's
eager value_and_grad closure while continuing to back forward/backward
hooks with the fused TileLang artifact.

These tests exercise:
  * Construction validation (callable artifact, forward/backward/vjp,
    value_and_grad{,_contract}).
  * forward/backward/vjp delegation to the wrapped artifact, threading
    the bank_owner.
  * value_and_grad consumption of the trainer-supplied loss closure
    (no eager closure → TypeError; closure runs → its outputs returned).
  * value_and_grad_contract surfaces a closed full-model gradient
    contract (returns_full_model_grads / model_gradient_tree_ready /
    loss_cotangent_bridge_ready / not delegates_to_eager / not
    hidden_packing), and preserves the underlying artifact's coverage
    payload while adding the mixed-mode telemetry.
"""

from __future__ import annotations

from typing import Any, Mapping

import mlx.core as mx
import pytest

from cppmega_mlx.training.compiled import (
    PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_CONTRACT,
    PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT,
    PathCFusedPlusEagerTrainingRuntime,
)


# ---------------------------------------------------------------------------
# Fake fused artifact — minimum contract surface so the runtime can wrap it.
# ---------------------------------------------------------------------------


class _FakeFusedArtifact:
    """In-memory stand-in for PathCFusedTrainBlockCallableArtifact."""

    def __init__(self) -> None:
        self.forward_calls: list[Mapping[str, Any]] = []
        self.backward_calls: list[Mapping[str, Any]] = []
        self.vjp_calls: list[Mapping[str, Any]] = []
        self.inner_contract = {
            "contract": PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT,
            "owner": "CompiledPretrainingStep",
            "uses_fused_train_block_runtime": True,
            "uses_forward_hook": True,
            "uses_backward_or_vjp_hook": True,
            "returns_model_grads": True,
            "returns_full_model_grads": False,
            "gradient_scope": "selected_train_block",
            "covered_parameter_count": 3,
            "trainable_parameter_count": 16,
            "missing_parameter_count": 13,
            "sample_missing_parameter_names": ["layers.0.block.q_proj.weight"],
            "full_model_gradient_tree_ready": False,
            "full_model_gradient_coverage": {
                "full_model_gradient_tree_ready": False,
                "reason": (
                    "selected fused train-block gradients do not cover all "
                    "trainable model parameters"
                ),
                "gradient_scope": "selected_train_block",
                "selected_region": {"name": "fake_region"},
                "covered_parameter_names": [
                    "layers.10.block.B_bias",
                    "layers.10.block.D",
                ],
                "missing_parameter_names": ["layers.0.block.q_proj.weight"],
                "covered_parameter_count": 2,
                "trainable_parameter_count": 16,
                "missing_parameter_count": 14,
            },
            "loss_cotangent_bridge_ready": True,
            "model_gradient_tree_ready": True,
            "delegates_to_eager_loss_and_grad": False,
            "hidden_packing_performed": False,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.forward_calls.append({"args": args, "kwargs": dict(kwargs)})
        return "forward_result"

    def backward(self, *args: Any, **kwargs: Any) -> Any:
        self.backward_calls.append({"args": args, "kwargs": dict(kwargs)})
        return "backward_result"

    def vjp(self, *args: Any, **kwargs: Any) -> Any:
        self.vjp_calls.append({"args": args, "kwargs": dict(kwargs)})
        return "vjp_result"

    def value_and_grad(self, *args: Any, **kwargs: Any) -> Any:
        return ("inner_loss", "inner_ntokens"), {"inner": "grads"}

    def value_and_grad_contract(self) -> Mapping[str, Any]:
        return self.inner_contract


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_rejects_non_callable_artifact() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        PathCFusedPlusEagerTrainingRuntime(
            artifact="not callable",  # type: ignore[arg-type]
            bank_owner=object(),
            owner_name="x",
        )


def test_construction_rejects_artifact_without_forward() -> None:
    class _Bad:
        def __call__(self) -> None:
            pass
        # forward intentionally missing

    with pytest.raises(TypeError, match="must define forward"):
        PathCFusedPlusEagerTrainingRuntime(
            artifact=_Bad(),
            bank_owner=object(),
            owner_name="x",
        )


def test_construction_succeeds_for_minimal_contract() -> None:
    art = _FakeFusedArtifact()
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art,
        bank_owner={"bank": "buffers"},
        owner_name="mixed_mode",
        in_region_parameter_names=("layers.10.block.D",),
    )
    assert runtime.artifact is art
    assert runtime.owner_name == "mixed_mode"
    assert runtime.in_region_parameter_names == frozenset({"layers.10.block.D"})
    assert runtime.training_critical_path is True
    assert runtime.hidden_packing_performed is False
    assert runtime.no_hidden_allocation_policy is True
    assert runtime.uses_fused_train_block_runtime is True
    assert runtime.contract == PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_CONTRACT


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


def test_forward_threads_bank_owner_into_artifact() -> None:
    art = _FakeFusedArtifact()
    bank_owner = object()
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art, bank_owner=bank_owner, owner_name="x",
    )
    assert runtime.forward() == "forward_result"
    assert art.forward_calls[-1]["kwargs"]["bank_owner"] is bank_owner


def test_backward_threads_bank_owner_into_artifact() -> None:
    art = _FakeFusedArtifact()
    bank_owner = object()
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art, bank_owner=bank_owner, owner_name="x",
    )
    assert runtime.backward() == "backward_result"
    assert art.backward_calls[-1]["kwargs"]["bank_owner"] is bank_owner


def test_vjp_threads_bank_owner_into_artifact() -> None:
    art = _FakeFusedArtifact()
    bank_owner = object()
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art, bank_owner=bank_owner, owner_name="x",
    )
    assert runtime.vjp() == "vjp_result"
    assert art.vjp_calls[-1]["kwargs"]["bank_owner"] is bank_owner


# ---------------------------------------------------------------------------
# value_and_grad consumes the trainer-supplied eager closure
# ---------------------------------------------------------------------------


def test_value_and_grad_requires_loss_closure() -> None:
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=_FakeFusedArtifact(), bank_owner={}, owner_name="x",
    )
    with pytest.raises(TypeError, match="loss_and_grad closure"):
        runtime.value_and_grad(model=object(), batch={}, loss_and_grad=None)  # type: ignore[arg-type]


def test_value_and_grad_returns_closure_outputs() -> None:
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=_FakeFusedArtifact(), bank_owner={}, owner_name="x",
    )
    captured: list[Any] = []

    def closure(model: Any, batch: Mapping[str, Any]) -> Any:
        captured.append((model, batch))
        return (mx.array(0.5), mx.array(7, dtype=mx.uint32)), {"layers.0": mx.zeros((1,))}

    (loss, ntokens), grads = runtime.value_and_grad(
        model="m",  # type: ignore[arg-type]
        batch={"tokens": "t"},  # type: ignore[arg-type]
        loss_and_grad=closure,
    )
    assert captured == [("m", {"tokens": "t"})]
    assert isinstance(loss, mx.array)
    assert isinstance(ntokens, mx.array)
    assert "layers.0" in grads


# ---------------------------------------------------------------------------
# Contract surface
# ---------------------------------------------------------------------------


def test_value_and_grad_contract_reports_closed_full_coverage() -> None:
    art = _FakeFusedArtifact()
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art, bank_owner={}, owner_name="x",
        in_region_parameter_names=("a", "b", "c"),
    )
    contract = runtime.value_and_grad_contract()
    assert contract["contract"] == PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT
    assert contract["owner"] == "CompiledPretrainingStep"
    assert contract["uses_fused_train_block_runtime"] is True
    assert contract["uses_forward_hook"] is True
    assert contract["uses_backward_or_vjp_hook"] is True
    assert contract["returns_model_grads"] is True
    assert contract["returns_full_model_grads"] is True
    assert contract["gradient_scope"] == "full_model_via_mixed_mode"
    assert contract["full_model_gradient_tree_ready"] is True
    assert contract["loss_cotangent_bridge_ready"] is True
    assert contract["model_gradient_tree_ready"] is True
    assert contract["delegates_to_eager_loss_and_grad"] is False
    assert contract["hidden_packing_performed"] is False
    assert contract["fused_in_region_parameter_count"] == 3
    assert contract["runtime_class"] == "PathCFusedPlusEagerTrainingRuntime"
    coverage = contract["full_model_gradient_coverage"]
    assert coverage["full_model_gradient_tree_ready"] is True
    assert coverage["gradient_scope"] == "full_model_via_mixed_mode"
    assert coverage["missing_parameter_names"] == []
    assert coverage["missing_parameter_count"] == 0
    assert coverage["fused_in_region_parameter_count"] == 3
    # The covered_parameter_names from the inner artifact's coverage payload
    # are preserved (these are the bricks the artifact actually owns).
    assert coverage["covered_parameter_names"] == [
        "layers.10.block.B_bias",
        "layers.10.block.D",
    ]


def test_bind_unbind_training_graph_round_trip() -> None:
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=_FakeFusedArtifact(), bank_owner={}, owner_name="x",
    )
    runtime.bind_training_graph(
        owner="CompiledPretrainingStep",
        uses_fused_train_block_runtime=True,
    )
    binding = runtime.training_graph_binding()
    assert binding["owner"] == "CompiledPretrainingStep"
    assert binding["uses_fused_train_block_runtime"] is True
    runtime.unbind_training_graph(owner="CompiledPretrainingStep")
    assert runtime.training_graph_binding() == {}


# ---------------------------------------------------------------------------
# Live integration: build_local_gb10_quarter_tiny_smoke_model end-to-end
# ---------------------------------------------------------------------------


def test_live_install_flips_route_to_fused_train_block() -> None:
    """End-to-end check: a real tiny HybridTinyLM auto-installs the runtime
    and the route_payload reports run_path_c_fused_train_block_route."""

    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "m04_train_step", str(Path("scripts/m04_train_step.py").resolve())
    )
    assert spec is not None and spec.loader is not None
    m04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m04)
    from cppmega_mlx.recipes.model_factory import (
        build_local_gb10_quarter_tiny_smoke_model,
    )

    model = build_local_gb10_quarter_tiny_smoke_model()
    args = m04.build_parser().parse_args([
        "--synthetic", "--dtype", "fp8_path_c",
        "--model-profile", "local_gb10_quarter",
        "--dry-run-json", "--output", "/tmp/test_mixed_mode_route.json",
        "--data-path", "/dev/null", "--data-format", "npz",
    ])
    route = m04.fp8_path_c_training_route_payload_for_model(args, model)

    assert route["status"] == "m04_path_c_training_route_available"
    assert route["selected_action"] == "run_path_c_fused_train_block_route"
    assert route["single_fused_train_block_runtime_available"] is True
    assert route["full_end_to_end_training_available"] is True
    assert route["fused_train_block_runtime_available"] is True
    assert route["fused_train_block_blocker_type"] is None

    runtime = model.path_c_fused_train_block_training_runtime
    assert runtime is not None
    assert type(runtime).__name__ == "PathCFusedPlusEagerTrainingRuntime"
    inner_contract = runtime.value_and_grad_contract()
    assert inner_contract["returns_full_model_grads"] is True
    assert inner_contract["gradient_scope"] == "full_model_via_mixed_mode"
    assert inner_contract["delegates_to_eager_loss_and_grad"] is False
