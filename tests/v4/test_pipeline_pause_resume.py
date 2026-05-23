"""V7-H06: stage_train wait_while_paused integration + RPC handlers."""

from __future__ import annotations

import threading
import time

import pytest

from cppmega_v4.jsonrpc.dispatcher import (
    _pipeline_pause, _pipeline_resume,
)
from cppmega_v4.jsonrpc.schema import (
    PipelineAbortParams, VerifyParams,
)
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.runtime import job_control as jc


@pytest.fixture(autouse=True)
def _reset():
    jc.reset()
    yield
    jc.reset()


def _spec() -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 4, "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def test_v7_h06_pipeline_pause_rpc_sets_flag():
    r = _pipeline_pause(PipelineAbortParams(run_id="abc"))
    assert r.run_id == "abc"
    assert jc.is_paused("abc") is True


def test_v7_h06_pipeline_resume_rpc_clears_flag():
    jc.pause("xyz")
    r = _pipeline_resume(PipelineAbortParams(run_id="xyz"))
    assert r.run_id == "xyz"
    assert jc.is_paused("xyz") is False


def test_v7_h06_stage_train_waits_while_paused_then_completes():
    """Pause job-T mid-loop from another thread, resume after 200ms,
    confirm train finishes with the expected step count."""
    jc.pause("job-T")

    def _resumer():
        time.sleep(0.2)
        jc.resume("job-T")

    threading.Thread(target=_resumer, daemon=True).start()
    t0 = time.time()
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {
            "num_steps": 2, "abort_token": "job-T",
        }},
    }))
    elapsed = time.time() - t0
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"paused train failed: {tr.error}"
    assert len(tr.extras["losses"]) == 2
    # Total wall-clock includes the 0.2s pause delay.
    assert elapsed > 0.15
