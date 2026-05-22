"""G09: cancel/abort Train via abort_token + _ABORT_TOKENS set."""

from __future__ import annotations

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.runner.stages import request_abort, clear_abort


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
    report = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    return next(s for s in report.stages if s.name == "train")


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
