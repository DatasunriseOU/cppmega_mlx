"""V7-H06b: pipeline.status RPC + run_registry round-trip.

Verifies that pipeline.status reflects backend reality for pause /
resume / abort / running / finished, so the UI can confirm a state
transition before flipping its own indicators.
"""

from __future__ import annotations

import threading
import time

import pytest

from cppmega_v4.jsonrpc.dispatcher import (
    _pipeline_abort, _pipeline_pause, _pipeline_resume, _pipeline_status,
)
from cppmega_v4.jsonrpc.schema import (
    PipelineAbortParams, PipelineStatusParams, VerifyParams,
)
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.runtime import job_control as jc
from cppmega_v4.runtime import run_registry as rr


@pytest.fixture(autouse=True)
def _reset():
    jc.reset()
    rr.reset()
    yield
    jc.reset()
    rr.reset()


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


def test_v7_h06b_status_unknown_for_never_seen_run():
    r = _pipeline_status(PipelineStatusParams(run_id="ghost"))
    assert r.run_id == "ghost"
    assert r.known is False
    assert r.running is False


def test_v7_h06b_status_reports_paused_after_pause_rpc():
    rr.register("run-X")
    _pipeline_pause(PipelineAbortParams(run_id="run-X"))
    r = _pipeline_status(PipelineStatusParams(run_id="run-X"))
    assert r.known is True
    assert r.paused is True
    assert r.running is True


def test_v7_h06b_status_clears_paused_after_resume_rpc():
    rr.register("run-Y")
    _pipeline_pause(PipelineAbortParams(run_id="run-Y"))
    assert _pipeline_status(
        PipelineStatusParams(run_id="run-Y")).paused is True
    _pipeline_resume(PipelineAbortParams(run_id="run-Y"))
    r = _pipeline_status(PipelineStatusParams(run_id="run-Y"))
    assert r.paused is False


def test_v7_h06b_status_marks_aborted_after_abort_rpc():
    rr.register("run-Z")
    _pipeline_abort(PipelineAbortParams(run_id="run-Z"))
    r = _pipeline_status(PipelineStatusParams(run_id="run-Z"))
    assert r.aborted is True


def test_v7_h06b_status_reports_running_during_train_then_finished():
    """Pause the train upfront so the train loop is *guaranteed* to be
    in wait_while_paused when the poller runs. Verify running=True
    + paused=True. Then resume, wait for train to finish, verify
    running=False."""
    # Pause first so step 0 blocks in wait_while_paused.
    jc.pause("run-T")
    captured: dict = {}

    def _resumer():
        time.sleep(0.15)
        r = _pipeline_status(PipelineStatusParams(run_id="run-T"))
        captured["mid"] = {"known": r.known, "running": r.running,
                           "paused": r.paused}
        jc.resume("run-T")

    threading.Thread(target=_resumer, daemon=True).start()
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {
            "num_steps": 2, "run_id": "run-T", "abort_token": "run-T",
        }},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok"

    # Mid-train snapshot taken while paused: running=True + paused=True.
    assert captured["mid"] == {"known": True, "running": True,
                                "paused": True}, captured
    # After train returns: running=False, paused already cleared.
    final = _pipeline_status(PipelineStatusParams(run_id="run-T"))
    assert final.known is True
    assert final.running is False
    assert final.paused is False
    assert final.last_step >= 0


def test_v7_h06b_status_last_step_and_loss_updates_during_train():
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {
            "num_steps": 3, "run_id": "run-S", "abort_token": "run-S",
        }},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok"
    final = _pipeline_status(PipelineStatusParams(run_id="run-S"))
    # 3-step train, last index = 2.
    assert final.last_step == 2
    assert final.last_loss is not None
    assert final.last_loss > 0
