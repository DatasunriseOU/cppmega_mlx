"""G09: cancel/abort Train via abort_token + _ABORT_TOKENS set."""

from __future__ import annotations

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.runner.stages import (
    STAGE_REGISTRY, StageContext, clear_abort, request_abort,
)


def _spec() -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {"nodes": [
            {"id": "attn", "kind": "attention", "params": {}},
            {"id": "mlp", "kind": "mlp",
             "params": {"intermediate_size": 64, "activation": "swiglu"}},
        ], "edges": [{"src": "attn", "dst": "mlp"}]},
        "dim_env": {"B": 1, "S": 8, "H": 32, "nh": 2, "nkv": 1, "head_dim": 16},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    })


def _run(opts: dict) -> dict:
    train = _run_stage(opts)
    return train.extras


def _run_stage(opts: dict):
    # Drive stage_train directly through STAGE_REGISTRY so this test
    # keeps proving G09's contract (stage_train itself observes
    # _ABORT_TOKENS and returns status="cancelled" with extras), even
    # after V7-H10 added a pipeline-level cancel gate that would
    # otherwise short-circuit stages earlier in the chain. The full
    # pipeline overall-status behaviour is covered separately by
    # tests/v4/test_jsonrpc_dispatcher.py and
    # tests/v4/test_pipeline_abort_verify_dry.py.
    spec = _spec()
    ctx = StageContext(spec=spec, options={"train": opts})
    STAGE_REGISTRY["parse"](ctx)
    # Skip verify_build_spec here: V7-H10 added a cancel gate inside
    # that stage which would consume the abort_token (via clear_abort)
    # before stage_train ever sees it. This test specifically proves
    # G09 — that stage_train itself observes the abort_token — so we
    # exercise the build_model + train path in isolation. End-to-end
    # behaviour through run_pipeline is covered separately.
    STAGE_REGISTRY["build_model"](ctx)
    return STAGE_REGISTRY["train"](ctx)


def _run_full_pipeline_overall_status(opts: dict) -> str:
    report = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    return report.overall_status


def test_abort_token_set_before_run_cancels_immediately():
    token = "abort-test-1"
    request_abort(token)
    try:
        stage = _run_stage({"num_steps": 8, "abort_token": token})
        extras = stage.extras
        assert stage.status == "cancelled"
        assert extras.get("aborted") is True
        assert extras["abort_token"] == token
        # Aborted at step 0 → losses empty
        assert extras["num_steps"] == 0
    finally:
        clear_abort(token)


def test_no_abort_token_runs_to_completion():
    extras = _run({"num_steps": 2})
    assert extras.get("aborted") is not True
    assert extras["num_steps"] == 2


def test_abort_token_not_in_set_runs_to_completion():
    extras = _run({"num_steps": 2, "abort_token": "never-set"})
    assert extras.get("aborted") is not True
    assert extras["num_steps"] == 2
