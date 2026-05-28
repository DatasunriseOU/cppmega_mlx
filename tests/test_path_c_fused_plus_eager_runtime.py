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
from mlx.utils import tree_flatten

from cppmega_mlx.training.compiled import (
    PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_CONTRACT,
    PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT,
    PathCFusedPlusEagerTrainingRuntime,
    _path_c_training_runtime_value_and_grad_contract,
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


def test_value_and_grad_contract_reports_closed_full_coverage_with_suffix_bypass() -> None:
    art = _FakeFusedArtifact()

    def _suffix_loss(model: Any, batch: Mapping[str, Any]) -> tuple[mx.array, mx.array]:
        return mx.array(0.0), mx.array(0, dtype=mx.uint32)

    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art,
        bank_owner={},
        owner_name="x",
        in_region_parameter_bank_aliases={
            "a": {"logical_name": "a", "logical_grad_name": "a_grad", "bank": "f32", "dtype": "float32", "offset": 0, "size": 1, "logical_shape": (1,)},
            "b": {"logical_name": "b", "logical_grad_name": "b_grad", "bank": "f32", "dtype": "float32", "offset": 1, "size": 1, "logical_shape": (1,)},
            "c": {"logical_name": "c", "logical_grad_name": "c_grad", "bank": "f32", "dtype": "float32", "offset": 2, "size": 1, "logical_shape": (1,)},
        },
        fused_suffix_loss_fn=_suffix_loss,
    )
    contract = runtime.value_and_grad_contract()
    assert contract["contract"] == PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT
    assert contract["owner"] == "CompiledPretrainingStep"
    assert contract["uses_fused_train_block_runtime"] is True
    assert contract["uses_forward_hook"] is True
    assert contract["uses_backward_or_vjp_hook"] is True
    assert contract["returns_model_grads"] is True
    assert contract["returns_full_model_grads"] is True
    assert contract["gradient_scope"] == "full_model_via_fused_suffix_bypass"
    assert contract["full_model_gradient_tree_ready"] is True
    assert contract["loss_cotangent_bridge_ready"] is True
    assert contract["model_gradient_tree_ready"] is True
    assert contract["delegates_to_eager_loss_and_grad"] is False
    assert contract["hidden_packing_performed"] is False
    assert contract["fused_in_region_parameter_count"] == 3
    assert contract["runtime_class"] == "PathCFusedPlusEagerTrainingRuntime"
    coverage = contract["full_model_gradient_coverage"]
    assert coverage["full_model_gradient_tree_ready"] is True
    assert coverage["gradient_scope"] == "full_model_via_fused_suffix_bypass"
    assert coverage["missing_parameter_names"] == []
    assert coverage["missing_parameter_count"] == 0
    assert coverage["fused_in_region_parameter_count"] == 3
    # The covered_parameter_names from the inner artifact's coverage payload
    # are preserved (these are the bricks the artifact actually owns).
    assert coverage["covered_parameter_names"] == [
        "layers.10.block.B_bias",
        "layers.10.block.D",
    ]


def test_warmup_contract_does_not_pass_training_critical_gate() -> None:
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=_FakeFusedArtifact(),
        bank_owner={},
        owner_name="warmup",
        in_region_parameter_names=("a", "b"),
    )
    contract = runtime.value_and_grad_contract()
    assert contract["suffix_bypass_available"] is False
    assert contract["returns_full_model_grads"] is False
    assert contract["loss_cotangent_bridge_ready"] is False
    assert contract["model_gradient_tree_ready"] is False
    assert contract["delegates_to_eager_loss_and_grad"] is True

    gate_payload = _path_c_training_runtime_value_and_grad_contract(runtime)
    assert gate_payload["status"] == "incomplete"


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
    lowerer_calls: list[dict[str, Any]] = []

    class _NoopKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, ...]] = []

        def __call__(self, *kernel_args: Any) -> None:
            self.calls.append(tuple(kernel_args))

    def fake_lowerer(func_or_mod: Any, *, target: str, **kwargs: Any) -> Any:
        del func_or_mod
        kernel = _NoopKernel()
        lowerer_calls.append({"target": target, "kwargs": dict(kwargs), "kernel": kernel})
        return kernel

    args = m04.build_parser().parse_args([
        "--synthetic", "--dtype", "fp8_path_c",
        "--model-profile", "local_gb10_quarter",
        "--dry-run-json", "--output", "/tmp/test_mixed_mode_route.json",
        "--data-path", "/dev/null", "--data-format", "npz",
        "--use-path-c-fused-train-block-runtime",
    ])
    route = m04.fp8_path_c_training_route_payload_for_model(
        args,
        model,
        fused_train_block_artifact_lowerer=fake_lowerer,
    )

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
    assert (
        inner_contract["gradient_scope"]
        == "full_model_via_fused_replay_cotangent_bridge"
    )
    assert inner_contract["suffix_bypass_available"] is False
    assert inner_contract["replay_cotangent_bridge_available"] is True
    assert inner_contract["delegates_to_eager_loss_and_grad"] is False
    auto_install = route["path_c_fusion"]["fused_train_block_auto_install"]
    assert auto_install["training_runtime_available"] is True
    assert auto_install["training_runtime_contract"]["status"] == "ok"
    assert lowerer_calls[0]["target"] == "metal"


def test_value_and_grad_runs_artifact_warmup_with_telemetry() -> None:
    """value_and_grad must trigger artifact.forward(bank_owner=...) as a
    real warmup pass and record telemetry on last_fused_warmup_payload."""

    class _Owner:
        def __init__(self) -> None:
            self.buffers: dict[str, Any] = {}

    art = _FakeFusedArtifact()
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art, bank_owner=_Owner(), owner_name="x",
    )

    def closure(model: Any, batch: Mapping[str, Any]) -> Any:
        return (mx.array(0.0), mx.array(1, dtype=mx.uint32)), {}

    runtime.value_and_grad(
        model="m",  # type: ignore[arg-type]
        batch={},
        loss_and_grad=closure,
    )
    # The fake artifact recorded a forward call with bank_owner threaded in.
    assert len(art.forward_calls) == 1
    assert art.forward_calls[-1]["kwargs"].get("bank_owner") is runtime.bank_owner
    # And telemetry was recorded.
    telemetry = runtime.last_fused_warmup_payload
    assert telemetry["attempted"] is True
    assert telemetry["completed"] is True
    assert telemetry["elapsed_ns"] >= 0
    assert telemetry["error"] is None


def test_value_and_grad_records_artifact_warmup_failure_without_breaking_step() -> None:
    """If the artifact's forward raises, the runtime still returns eager grads
    and records the failure so receipts can surface it."""

    class _BadOwner:
        @property
        def buffers(self) -> dict[str, Any]:
            raise RuntimeError("bank owner explodes")

    art = _FakeFusedArtifact()
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art, bank_owner=_BadOwner(), owner_name="x",
    )

    def closure(model: Any, batch: Mapping[str, Any]) -> Any:
        return (mx.array(0.0), mx.array(1, dtype=mx.uint32)), {"x": mx.zeros((1,))}

    (loss, ntokens), grads = runtime.value_and_grad(
        model="m",  # type: ignore[arg-type]
        batch={},
        loss_and_grad=closure,
    )
    telemetry = runtime.last_fused_warmup_payload
    # Either the forward raised (BadOwner._kernel_args path) or our mx.eval
    # tripped — either way the runtime continues and returns eager grads.
    assert telemetry["attempted"] is True
    assert telemetry["error"] is not None
    assert isinstance(loss, mx.array)
    assert "x" in grads


def test_live_value_and_grad_replay_bridge_dispatches_artifact() -> None:
    """End-to-end: a real HybridTinyLM mixed runtime must dispatch the compiled
    artifact through forward and replay-backward gates."""

    import importlib.util
    from pathlib import Path

    import mlx.nn as nn

    from cppmega_mlx.recipes.model_factory import (
        build_local_gb10_quarter_tiny_smoke_model,
    )

    spec = importlib.util.spec_from_file_location(
        "m04_train_step", str(Path("scripts/m04_train_step.py").resolve())
    )
    assert spec is not None and spec.loader is not None
    m04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m04)

    model = build_local_gb10_quarter_tiny_smoke_model()
    kernels: list[Any] = []

    class _NoopKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, ...]] = []

        def __call__(self, *kernel_args: Any) -> None:
            self.calls.append(tuple(kernel_args))

    def fake_lowerer(func_or_mod: Any, *, target: str, **kwargs: Any) -> Any:
        del func_or_mod, target, kwargs
        kernel = _NoopKernel()
        kernels.append(kernel)
        return kernel

    args = m04.build_parser().parse_args([
        "--synthetic", "--dtype", "fp8_path_c",
        "--model-profile", "local_gb10_quarter",
        "--dry-run-json", "--output", "/tmp/test_live_warmup.json",
        "--data-path", "/dev/null", "--data-format", "npz",
        "--use-path-c-fused-train-block-runtime",
    ])
    route = m04.fp8_path_c_training_route_payload_for_model(
        args,
        model,
        fused_train_block_artifact_lowerer=fake_lowerer,
    )
    assert route["selected_action"] == "run_path_c_fused_train_block_route"
    runtime = model.path_c_fused_train_block_training_runtime
    assert runtime is not None
    assert type(runtime).__name__ == "PathCFusedPlusEagerTrainingRuntime"

    # Trivial loss closure. Replay bridge ignores this closure in favour of
    # the model-attached fused replay loss; it is still passed to keep the
    # public value_and_grad signature exercised.
    # Match the path_c_training_sequence_length(args)=127 the install path uses
    # so the fused suffix bank slots line up with the prefix output.
    seq_len = 127

    def loss_fn(model: nn.Module, batch: Mapping[str, mx.array]) -> tuple[mx.array, mx.array]:
        logits = model(batch["tokens"])
        loss = logits.sum() * mx.array(0.0)
        ntokens = mx.array(int(batch["tokens"].size), dtype=mx.uint32)
        return loss, ntokens

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    # Provide both tokens and target_tokens with the same length so the
    # batch's effective input length equals seq_len rather than seq_len-1.
    # This keeps the prefix output shape aligned with the fused suffix
    # bank slot when suffix-bypass mode is active.
    tokens = mx.zeros((1, seq_len), dtype=mx.int32)
    target_tokens = mx.zeros((1, seq_len), dtype=mx.int32)

    (loss, ntokens), grads = runtime.value_and_grad(
        model,
        {"tokens": tokens, "target_tokens": target_tokens},
        loss_and_grad,
    )
    mx.eval(loss, ntokens)

    # Replay bridge is now the training critical path: no separate warmup
    # launch, and artifact.forward must be called once with run_backward=0
    # and once with run_backward=1 through the custom VJP.
    warmup_telemetry = runtime.last_fused_warmup_payload
    replay_telemetry = runtime.last_fused_value_and_grad_payload
    assert warmup_telemetry["attempted"] is False
    assert replay_telemetry["attempted"] is True
    assert replay_telemetry["completed"] is True, replay_telemetry
    assert replay_telemetry["elapsed_ns"] > 0
    assert replay_telemetry["error"] is None
    assert replay_telemetry["replay_cotangent_bridge_active"] is True
    assert replay_telemetry["suffix_bypass_active"] is False
    assert len(kernels) == 1
    assert len(kernels[0].calls) == 2

    # Loss/ntokens come from the eager closure; gradient tree must contain
    # parameters that the artifact does NOT cover, proving the eager prefix /
    # suffix graph still provides residual grads while the M/R/A block is
    # crossed through the fused replay custom VJP.
    flat = dict(tree_flatten(grads))
    assert "layers.0.block.q_proj.weight" in flat  # outside fused region



# ---------------------------------------------------------------------------
# Merged-grad mode: bank-residency-aware artifact + sync callable
# ---------------------------------------------------------------------------


class _FakeFusedArtifactWithBankResidentGrads:
    """Bank-aware fake that returns mx.array views as grads for in-region params."""

    def __init__(self) -> None:
        self.value_and_grad_calls: list[Mapping[str, Any]] = []
        self.forward_calls: list[Mapping[str, Any]] = []
        self.bank_grads: dict[str, mx.array] = {}
        self.inner_contract: dict[str, Any] = {
            "contract": PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT,
            "owner": "CompiledPretrainingStep",
            "uses_fused_train_block_runtime": True,
            "uses_forward_hook": True,
            "uses_backward_or_vjp_hook": True,
            "returns_model_grads": True,
            "returns_full_model_grads": False,
            "gradient_scope": "selected_train_block",
            "covered_parameter_count": 2,
            "trainable_parameter_count": 4,
            "missing_parameter_count": 2,
            "sample_missing_parameter_names": [
                "layers.0.block.q_proj.weight",
                "lm_head.weight",
            ],
            "full_model_gradient_tree_ready": False,
            "full_model_gradient_coverage": {
                "full_model_gradient_tree_ready": False,
                "reason": "stub",
                "gradient_scope": "selected_train_block",
                "selected_region": {"name": "fake"},
                "covered_parameter_names": [
                    "layers.10.block.D",
                    "layers.11.block.D",
                ],
                "missing_parameter_names": [],
                "covered_parameter_count": 2,
                "trainable_parameter_count": 4,
                "missing_parameter_count": 0,
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

    def backward(self, *args: Any, **kwargs: Any) -> Any:
        return "backward"

    def vjp(self, *args: Any, **kwargs: Any) -> Any:
        return "vjp"

    def value_and_grad(self, *args: Any, **kwargs: Any) -> Any:
        self.value_and_grad_calls.append(
            {"args": args, "kwargs": dict(kwargs)}
        )
        # Return synthesised bank-resident grads (the runtime should overlay
        # these onto the eager grad tree for the in-region parameters).
        return (
            (mx.array(0.5, dtype=mx.float32), mx.array(8, dtype=mx.uint32)),
            dict(self.bank_grads),
        )

    def value_and_grad_contract(self) -> Mapping[str, Any]:
        return self.inner_contract


class _FakeBankOwner:
    def __init__(self) -> None:
        self.buffers: dict[str, mx.array] = {
            "f32": mx.zeros((32,), dtype=mx.float32),
        }


def test_bank_residency_without_suffix_bypass_fails_closed_to_eager_grads() -> None:
    art = _FakeFusedArtifactWithBankResidentGrads()
    # Two in-region params, each with a unique sentinel grad value the runtime
    # must surface to the trainer (replacing whatever the eager closure
    # produced for the same key).
    fused_grad_10 = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
    fused_grad_11 = mx.array([5.0, 6.0, 7.0, 8.0], dtype=mx.float32)
    art.bank_grads = {
        "layers.10.block.D": fused_grad_10,
        "layers.11.block.D": fused_grad_11,
    }
    bank_owner = _FakeBankOwner()
    sync_calls: list[None] = []

    def sync_callable() -> Mapping[str, Any]:
        sync_calls.append(None)
        return {"status": "ok", "synced": ["x"], "skipped": []}

    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art,
        bank_owner=bank_owner,
        owner_name="merged",
        in_region_parameter_bank_aliases={
            "layers.10.block.D": {
                "logical_name": "brick_10_M_D",
                "logical_grad_name": "brick_10_M_D_grad",
                "bank": "f32",
                "dtype": "float32",
                "offset": 0,
                "size": 4,
                "logical_shape": (4,),
            },
            "layers.11.block.D": {
                "logical_name": "brick_11_R_D",
                "logical_grad_name": "brick_11_R_D_grad",
                "bank": "f32",
                "dtype": "float32",
                "offset": 4,
                "size": 4,
                "logical_shape": (4,),
            },
        },
        model_bank_sync_callable=sync_callable,
    )

    eager_grad_10 = mx.array([99.0, 99.0, 99.0, 99.0], dtype=mx.float32)
    eager_grad_11 = mx.array([99.0, 99.0, 99.0, 99.0], dtype=mx.float32)
    eager_q_proj = mx.zeros((16, 16), dtype=mx.float32)
    eager_lm_head = mx.zeros((256, 16), dtype=mx.float32)

    def closure(model: Any, batch: Mapping[str, Any]) -> Any:
        return (
            (mx.array(0.0), mx.array(1, dtype=mx.uint32)),
            {
                "layers.10.block.D": eager_grad_10,
                "layers.11.block.D": eager_grad_11,
                "layers.0.block.q_proj.weight": eager_q_proj,
                "lm_head.weight": eager_lm_head,
            },
        )

    (loss, ntokens), grads = runtime.value_and_grad(
        model="m",  # type: ignore[arg-type]
        batch={},
        loss_and_grad=closure,
    )
    # Without suffix-bypass / runtime-input writing, the runtime must not sync
    # params into banks, must not call artifact.value_and_grad, and must not
    # overlay fused grad slots. That path would run against stale hidden/target
    # bank inputs and corrupt the eager gradients.
    assert len(sync_calls) == 0
    assert len(art.value_and_grad_calls) == 0
    assert len(art.forward_calls) == 1
    flat_grads = dict(tree_flatten(grads))
    assert flat_grads["layers.10.block.D"].tolist() == [99.0, 99.0, 99.0, 99.0]
    assert flat_grads["layers.11.block.D"].tolist() == [99.0, 99.0, 99.0, 99.0]
    # Out-of-region grads are unchanged (eager wins).
    assert flat_grads["layers.0.block.q_proj.weight"].shape == (16, 16)
    assert flat_grads["lm_head.weight"].shape == (256, 16)
    # Telemetry reflects fail-closed warmup+eager behaviour.
    assert runtime.last_fused_warmup_payload["attempted"] is True
    vg = runtime.last_fused_value_and_grad_payload
    assert vg["attempted"] is False
    assert vg["completed"] is False
    assert vg["merged_parameter_count"] == 0
    assert vg["merged_parameter_names"] == ()
    assert vg["missing_parameter_names"] == ()
    assert vg["bank_sync_status"] is None
    assert vg["suffix_bypass_active"] is False


def test_bank_residency_without_suffix_bypass_never_reads_partial_bank_grads() -> None:
    """Partial fused grad trees are ignored until suffix-bypass supplies
    verified runtime inputs for the kernel."""

    art = _FakeFusedArtifactWithBankResidentGrads()
    # Only one of the two in-region params has a bank-resident grad.
    art.bank_grads = {
        "layers.10.block.D": mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32),
    }
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art,
        bank_owner=_FakeBankOwner(),
        owner_name="merged",
        in_region_parameter_bank_aliases={
            "layers.10.block.D": {"logical_name": "x", "logical_grad_name": "x_grad", "bank": "f32", "dtype": "float32", "offset": 0, "size": 4, "logical_shape": (4,)},
            "layers.11.block.D": {"logical_name": "y", "logical_grad_name": "y_grad", "bank": "f32", "dtype": "float32", "offset": 4, "size": 4, "logical_shape": (4,)},
        },
    )
    eager_grad_11 = mx.array([42.0, 42.0, 42.0, 42.0], dtype=mx.float32)

    def closure(model: Any, batch: Mapping[str, Any]) -> Any:
        return (
            (mx.array(0.0), mx.array(1, dtype=mx.uint32)),
            {
                "layers.10.block.D": mx.zeros((4,)),
                "layers.11.block.D": eager_grad_11,
            },
        )

    (loss, ntokens), grads = runtime.value_and_grad(
        model="m",  # type: ignore[arg-type]
        batch={},
        loss_and_grad=closure,
    )
    flat_grads = dict(tree_flatten(grads))
    # Even the available bank-resident grad is ignored in fail-closed mode;
    # eager remains the source of truth for all parameters.
    assert flat_grads["layers.10.block.D"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert flat_grads["layers.11.block.D"].tolist() == [42.0, 42.0, 42.0, 42.0]
    vg = runtime.last_fused_value_and_grad_payload
    assert vg["merged_parameter_count"] == 0
    assert vg["merged_parameter_names"] == ()
    assert vg["missing_parameter_names"] == ()
    assert vg["suffix_bypass_active"] is False


def test_merged_mode_contract_exposes_bank_residency_signals() -> None:
    art = _FakeFusedArtifactWithBankResidentGrads()
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art,
        bank_owner=_FakeBankOwner(),
        owner_name="merged",
        in_region_parameter_bank_aliases={
            "layers.10.block.D": {"logical_name": "x", "logical_grad_name": "x_grad", "bank": "f32", "dtype": "float32", "offset": 0, "size": 4, "logical_shape": (4,)},
            "layers.11.block.D": {"logical_name": "y", "logical_grad_name": "y_grad", "bank": "f32", "dtype": "float32", "offset": 4, "size": 4, "logical_shape": (4,)},
        },
    )
    contract = runtime.value_and_grad_contract()
    assert contract["parameter_bank_residency_active"] is True
    assert contract["bank_grad_overlay_active"] is False
    assert contract["merged_bank_resident_parameter_count"] == 0
    coverage = contract["full_model_gradient_coverage"]
    assert coverage["parameter_bank_residency_active"] is True
    assert coverage["bank_grad_overlay_active"] is False
    assert coverage["merged_bank_resident_parameter_count"] == 0
    assert coverage["merged_bank_resident_parameter_names"] == ()
    assert "disabled" in coverage["reason"]


def test_warmup_mode_contract_clears_bank_residency_signals() -> None:
    art = _FakeFusedArtifact()
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art,
        bank_owner=_FakeBankOwner(),
        owner_name="warmup",
        in_region_parameter_names=("a", "b"),
    )
    contract = runtime.value_and_grad_contract()
    assert contract["parameter_bank_residency_active"] is False
    assert contract["bank_grad_overlay_active"] is False
    assert contract["merged_bank_resident_parameter_count"] == 0
    coverage = contract["full_model_gradient_coverage"]
    assert coverage["parameter_bank_residency_active"] is False
    assert coverage["bank_grad_overlay_active"] is False
    assert coverage["merged_bank_resident_parameter_count"] == 0
    assert coverage["merged_bank_resident_parameter_names"] == ()


def test_suffix_bypass_contract_field_reflects_attachment() -> None:
    art = _FakeFusedArtifactWithBankResidentGrads()
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art,
        bank_owner=_FakeBankOwner(),
        owner_name="merged_no_bypass",
        in_region_parameter_bank_aliases={
            "a": {"logical_name": "x", "logical_grad_name": "x_grad", "bank": "f32", "dtype": "float32", "offset": 0, "size": 4, "logical_shape": (4,)},
        },
    )
    assert runtime.value_and_grad_contract()["suffix_bypass_available"] is False

    # Attaching a fused_suffix_loss_fn flips the contract signal.
    def _stub_loss(model: Any, batch: Mapping[str, Any]) -> tuple[mx.array, mx.array]:
        return mx.array(0.0), mx.array(0, dtype=mx.uint32)

    runtime_with_bypass = PathCFusedPlusEagerTrainingRuntime(
        artifact=art,
        bank_owner=_FakeBankOwner(),
        owner_name="merged_with_bypass",
        in_region_parameter_bank_aliases={
            "a": {"logical_name": "x", "logical_grad_name": "x_grad", "bank": "f32", "dtype": "float32", "offset": 0, "size": 4, "logical_shape": (4,)},
        },
        fused_suffix_loss_fn=_stub_loss,
    )
    assert (
        runtime_with_bypass.value_and_grad_contract()["suffix_bypass_available"]
        is True
    )
    assert runtime_with_bypass.value_and_grad_contract()["bank_grad_overlay_active"] is True


def test_suffix_bypass_uses_fused_suffix_loss_fn_and_skips_eager_closure() -> None:
    art = _FakeFusedArtifactWithBankResidentGrads()
    captured: dict[str, Any] = {}

    def fused_suffix_loss(model: Any, batch: Mapping[str, Any]) -> tuple[mx.array, mx.array]:
        captured["fused_called"] = True
        # Touch the model parameters so MLX has something to differentiate.
        param = getattr(model, "shared_param")
        return param.sum(), mx.array(7, dtype=mx.uint32)

    import mlx.nn as nn

    class _MiniModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared_param = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)

    model = _MiniModel()
    runtime = PathCFusedPlusEagerTrainingRuntime(
        artifact=art,
        bank_owner=_FakeBankOwner(),
        owner_name="bypass_mini",
        in_region_parameter_bank_aliases={
            "shared_param": {"logical_name": "x", "logical_grad_name": "x_grad", "bank": "f32", "dtype": "float32", "offset": 0, "size": 3, "logical_shape": (3,)},
        },
        fused_suffix_loss_fn=fused_suffix_loss,
    )

    def eager_closure(model: Any, batch: Mapping[str, Any]) -> Any:
        captured["eager_called"] = True
        return (mx.array(99.0), mx.array(0, dtype=mx.uint32)), {"shared_param": mx.zeros((3,))}

    (loss, ntokens), grads = runtime.value_and_grad(
        model, {}, eager_closure
    )
    mx.eval(loss, ntokens, *(g for _, g in tree_flatten(grads)))
    # Bypass ran the fused closure but never the trainer's eager closure.
    assert captured.get("fused_called") is True
    assert "eager_called" not in captured
    # Loss/ntokens come from the fused closure (loss=1+2+3=6, ntokens=7).
    assert float(loss) == 6.0
    assert int(ntokens) == 7
    flat = dict(tree_flatten(grads))
    assert "shared_param" in flat
    # Telemetry confirms suffix-bypass active.
    vg = runtime.last_fused_value_and_grad_payload
    assert vg["suffix_bypass_active"] is True
    assert vg["completed"] is True
    assert vg["error"] is None
