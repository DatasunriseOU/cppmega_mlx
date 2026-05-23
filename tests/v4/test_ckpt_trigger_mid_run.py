"""V7-Q03.4: mid-run checkpoint trigger via request_checkpoint().

Pins: the train loop drains _TRIGGER_CHECKPOINT_QUEUE between steps,
saving weights + opt-state sidecar when an entry matches the active
abort_token. Lets the operator save mid-run without aborting.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.runner.stages import (
    consume_checkpoint_trigger,
    request_checkpoint,
)


def _spec(H: int = 128, S: int = 16) -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 2, "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": S, "H": H,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def test_request_and_consume_roundtrip() -> None:
    """consume returns the path once then clears the queue."""
    request_checkpoint("tok-A", "/tmp/x.safetensors")
    assert consume_checkpoint_trigger("tok-A") == "/tmp/x.safetensors"
    assert consume_checkpoint_trigger("tok-A") is None


def test_consume_unknown_token_returns_none() -> None:
    assert consume_checkpoint_trigger("never-set") is None


def test_mid_run_trigger_saves_during_train() -> None:
    """End-to-end: request_checkpoint mid-run, train loop drains queue,
    weights + opt-state sidecar appear on disk before run completes."""
    with tempfile.TemporaryDirectory() as td:
        mid_path = os.path.join(td, "mid.safetensors")
        sidecar_path = mid_path + ".opt"

        # Pre-stage the trigger BEFORE train starts. The drain at step 0
        # will pick it up and write the file. Use a unique abort_token.
        token = "test-trigger-A"
        request_checkpoint(token, mid_path)

        rep = run_pipeline(_spec(), Pipeline.from_dict({
            "stages": ["parse", "verify_build_spec",
                       "build_model", "train"],
            "stage_options": {"train": {
                "num_steps": 2, "seed": 5,
                "abort_token": token,
            }},
        }))
        tr = next(s for s in rep.stages if s.name == "train")
        assert tr.status == "ok", tr.error

        # Trigger queue must be drained (popped).
        assert consume_checkpoint_trigger(token) is None
        # Files must exist on disk.
        assert os.path.isfile(mid_path), \
            f"weights mid_path missing at {mid_path}"
        assert os.path.isfile(sidecar_path), \
            f"opt-state sidecar missing at {sidecar_path}"
