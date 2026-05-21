"""G10: optimizer state warm-start across sequential train runs."""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec() -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention", "params": {}},
                {"id": "mlp", "kind": "mlp",
                 "params": {"intermediate_size": 64, "activation": "swiglu"}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 32, "nh": 2, "nkv": 1, "head_dim": 16},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    })


def _run(opts: dict) -> dict:
    report = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", f"stage_train failed: {train.error}"
    return train.extras


def test_first_run_no_carry():
    extras = _run({"num_steps": 2, "run_id": "run-A"})
    assert extras["opt_state_carried"] is False
    assert extras["run_id"] == "run-A"


def test_continue_from_loads_carried_state():
    """Two sequential runs: second sets continue_from_run_id=A → carried."""
    a = _run({"num_steps": 2, "run_id": "run-A2"})
    b = _run({"num_steps": 2, "run_id": "run-B2",
              "continue_from_run_id": "run-A2"})
    assert a["opt_state_carried"] is False
    assert b["opt_state_carried"] is True


def test_continue_from_unknown_id_no_carry():
    extras = _run({"num_steps": 2, "run_id": "run-C",
                   "continue_from_run_id": "nonexistent-id"})
    assert extras["opt_state_carried"] is False
