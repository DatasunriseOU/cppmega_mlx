"""V7-H10: abort_token honoured by verify_build_spec + dry_forward.

Previously only stage_train respected _ABORT_TOKENS. UI clicking cancel
during a long verify on a big graph had no effect — the user had to
wait it out. This pins:

  - stage_verify_build_spec returns 'cancelled' at entry if aborted.
  - stage_dry_forward returns 'cancelled' at entry if aborted.
  - run_pipeline driver, between stages, halts and marks remaining
    stages 'skipped' if abort_token is set.
"""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.dispatcher import _pipeline_abort
from cppmega_v4.jsonrpc.schema import PipelineAbortParams, VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.runner.stages import _ABORT_TOKENS, request_abort
from cppmega_v4.runtime import run_registry as rr


@pytest.fixture(autouse=True)
def _reset():
    _ABORT_TOKENS.clear()
    rr.reset()
    yield
    _ABORT_TOKENS.clear()
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


def test_v7_h10_verify_build_spec_stage_function_cancels_at_entry():
    """Direct stage call: the entry-gate inside stage_verify_build_spec
    fires when its abort_token is in _ABORT_TOKENS, independent of the
    pipeline driver."""
    from cppmega_v4.runner.stages import (
        StageContext, stage_verify_build_spec, stage_parse,
    )
    ctx = StageContext(spec=_spec(),
                       options={"verify_build_spec":
                                {"abort_token": "ABRT-V"}})
    parse_result = stage_parse(ctx)
    assert parse_result.status == "ok"

    request_abort("ABRT-V")
    v = stage_verify_build_spec(ctx)
    assert v.status == "cancelled"
    assert v.error == {"type": "Aborted", "abort_token": "ABRT-V"}


def test_v7_h10_dry_forward_stage_function_cancels_at_entry():
    """Direct stage call: stage_dry_forward's entry-gate fires."""
    from cppmega_v4.runner.stages import (
        StageContext, stage_dry_forward, stage_parse, stage_build_model,
        stage_verify_build_spec, stage_resolve_shapes,
        stage_estimate_memory,
    )
    ctx = StageContext(spec=_spec(),
                       options={"dry_forward":
                                {"abort_token": "ABRT-DF"}})
    assert stage_parse(ctx).status == "ok"
    assert stage_verify_build_spec(ctx).status == "ok"
    assert stage_resolve_shapes(ctx).status == "ok"
    assert stage_estimate_memory(ctx).status == "ok"
    assert stage_build_model(ctx).status == "ok"

    request_abort("ABRT-DF")
    df = stage_dry_forward(ctx)
    assert df.status == "cancelled"
    assert df.error == {"type": "Aborted", "abort_token": "ABRT-DF"}


def test_v7_h10_pipeline_driver_between_stages_halts_on_abort_rpc():
    """Run a pipeline whose train stage carries an abort_token; trigger
    pipeline.abort RPC up-front so the very first between-stage gate
    fires before parse even runs."""
    _pipeline_abort(PipelineAbortParams(run_id="ABRT-PD"))
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": 2,
                                    "abort_token": "ABRT-PD"}},
    }))
    parse = next(s for s in rep.stages if s.name == "parse")
    assert parse.status == "cancelled"
    assert rep.overall_status == "cancelled"


def test_v7_h10_no_abort_token_still_runs_to_completion():
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": 2}},
    }))
    assert rep.overall_status == "ok"
    for s in rep.stages:
        assert s.status == "ok", f"{s.name}: {s.status} {s.error}"
