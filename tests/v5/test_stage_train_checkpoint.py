"""G12: checkpoint save/resume."""

from __future__ import annotations

import pathlib

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


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
    report = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    train = next(s for s in report.stages if s.name == "train")
    return train.extras


def test_checkpoint_save_writes_file(tmp_path):
    save_path = str(tmp_path / "ckpt.safetensors")
    extras = _run({"num_steps": 2, "checkpoint_save_path": save_path})
    assert extras["checkpoint"]["saved_path"] == save_path
    assert extras["checkpoint"]["loaded_path"] is None
    assert pathlib.Path(save_path).exists()


def test_checkpoint_load_reads_file(tmp_path):
    save_path = str(tmp_path / "ckpt.safetensors")
    a = _run({"num_steps": 2, "checkpoint_save_path": save_path})
    assert a["checkpoint"]["saved_path"] == save_path
    b = _run({"num_steps": 2, "checkpoint_load_path": save_path})
    assert b["checkpoint"]["loaded_path"] == save_path


def test_checkpoint_load_missing_file_is_non_fatal():
    extras = _run({"num_steps": 2,
                   "checkpoint_load_path": "/nonexistent/ckpt.safetensors"})
    # train still succeeds; loaded_path is None
    assert extras["checkpoint"]["loaded_path"] is None


def test_no_checkpoint_opts_yields_null_paths():
    extras = _run({"num_steps": 2})
    assert extras["checkpoint"]["saved_path"] is None
    assert extras["checkpoint"]["loaded_path"] is None
