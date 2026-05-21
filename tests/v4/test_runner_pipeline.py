"""F-A.2 runner tests — pipeline orchestrator + 12 built-in stages."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import (
    FULL_STAGES,
    Pipeline,
    PipelineReport,
    SMOKE_STAGES,
    STAGE_REGISTRY,
    StageContext,
    run_pipeline,
)
from cppmega_v4.runner.stages import (
    stage_apply_rewrites,
    stage_build_model,
    stage_check_gotchas,
    stage_dry_forward,
    stage_estimate_memory,
    stage_input_parity_check,
    stage_loss_smoke,
    stage_parse,
    stage_resolve_shapes,
    stage_verify_build_spec,
)


_DIM_ENV = {"B": 1, "S": 4, "H": 64, "nh": 2, "nkv": 1, "head_dim": 32,
            "num_experts": 8, "top_k": 2}


def _spec(**extra) -> VerifyParams:
    payload = {
        "graph": {
            "nodes": [{"id": "a", "kind": "mlp"}, {"id": "b", "kind": "mlp"}],
            "edges": [{"src": "a", "dst": "b"}],
        },
        "dim_env": _DIM_ENV,
        "loss": {"kind": "cross_entropy", "head_outputs": ["b"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 3e-4,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    }
    payload.update(extra)
    return VerifyParams.model_validate(payload)


def _ctx(**extra) -> StageContext:
    return StageContext(spec=_spec(**extra))


# ---------------------------------------------------------------------------
# Registry sanity.
# ---------------------------------------------------------------------------


def test_smoke_stages_are_8():
    assert len(SMOKE_STAGES) == 8


def test_full_stages_include_smoke_plus_three_more():
    assert set(SMOKE_STAGES) <= set(FULL_STAGES)
    assert len(FULL_STAGES) == len(SMOKE_STAGES) + 3


@pytest.mark.parametrize("stage_name", sorted(STAGE_REGISTRY))
def test_every_registry_entry_is_callable(stage_name):
    assert callable(STAGE_REGISTRY[stage_name])


# ---------------------------------------------------------------------------
# Individual stages.
# ---------------------------------------------------------------------------


def test_stage_parse_materialises_graph_and_specs():
    ctx = _ctx()
    r = stage_parse(ctx)
    assert r.status == "ok"
    assert ctx.graph is not None
    assert ctx.build_spec is not None
    assert r.extras["num_nodes"] == 2


def test_stage_verify_build_spec_requires_parse_first():
    r = stage_verify_build_spec(_ctx())
    assert r.status == "fail"


def test_stage_verify_build_spec_passes_after_parse():
    ctx = _ctx()
    stage_parse(ctx)
    r = stage_verify_build_spec(ctx)
    assert r.status == "ok"
    assert r.errors == 0


def test_stage_resolve_shapes_succeeds_on_well_formed_graph():
    ctx = _ctx()
    stage_parse(ctx)
    r = stage_resolve_shapes(ctx)
    assert r.status == "ok"
    assert ctx.resolved is not None


def test_stage_estimate_memory_emits_total_bytes():
    ctx = _ctx()
    stage_parse(ctx)
    r = stage_estimate_memory(ctx)
    assert r.status == "ok"
    assert r.extras["total_bytes"] > 0


def test_stage_check_gotchas_skips_without_sharding():
    ctx = _ctx()
    stage_parse(ctx)
    r = stage_check_gotchas(ctx)
    assert r.status == "skipped"


def test_stage_check_gotchas_fires_with_sharding():
    ctx = _ctx(sharding={
        "topology": {"factory": "h100_8x", "kwargs": {}},
        "axis_assignments": [{"axis_name": "dp", "kind": "fsdp2", "degree": 8}],
        "compile_mode": "regional", "fp8_enabled": False,
    })
    stage_parse(ctx)
    r = stage_check_gotchas(ctx)
    assert r.status == "ok"
    assert isinstance(r.extras["fired"], int)


def test_stage_build_model_succeeds_on_small_graph():
    ctx = _ctx()
    stage_parse(ctx)
    r = stage_build_model(ctx)
    assert r.status == "ok"
    assert ctx.built_model is not None


def test_stage_dry_forward_ok_on_small_graph():
    ctx = _ctx()
    stage_parse(ctx)
    r = stage_dry_forward(ctx)
    assert r.status == "ok"


def test_stage_input_parity_check_skips_without_paths():
    ctx = _ctx()
    stage_parse(ctx)
    r = stage_input_parity_check(ctx)
    assert r.status == "skipped"


def test_stage_loss_smoke_runs_for_cross_entropy():
    ctx = _ctx()
    stage_parse(ctx)
    r = stage_loss_smoke(ctx)
    assert r.status == "ok"


def test_stage_apply_rewrites_is_currently_noop():
    r = stage_apply_rewrites(_ctx())
    assert r.status == "ok"
    assert r.extras["rewrites_applied"] == []


# ---------------------------------------------------------------------------
# Pipeline orchestration.
# ---------------------------------------------------------------------------


def test_pipeline_rejects_unknown_stage():
    with pytest.raises(ValueError, match="unknown stage"):
        Pipeline(stages=("bogus",))


def test_smoke_pipeline_runs_to_completion():
    report = run_pipeline(_spec(), Pipeline.smoke())
    assert isinstance(report, PipelineReport)
    assert report.overall_status == "ok"
    assert len(report.stages) == 8


def test_pipeline_stops_on_first_failure():
    pipe = Pipeline(stages=("parse", "verify_build_spec",
                            "resolve_shapes", "estimate_memory"))
    # Inject an unknown brick → parse fails
    bad = _spec(graph={"nodes": [{"id": "x", "kind": "definitely_not_a_brick"}],
                       "edges": []})
    report = run_pipeline(bad, pipe)
    assert report.overall_status == "fail"
    assert report.stages[0].status == "fail"
    # Remaining stages marked skipped
    for r in report.stages[1:]:
        assert r.status == "skipped"


def test_pipeline_continue_on_failure_does_not_stop():
    pipe = Pipeline(
        stages=("parse", "verify_build_spec"),
        continue_on_failure=True,
    )
    bad = _spec(graph={"nodes": [{"id": "x", "kind": "no_such"}], "edges": []})
    report = run_pipeline(bad, pipe)
    # Both stages ran (parse failed, verify_build_spec also failed because
    # parse didn't populate ctx.build_spec)
    assert all(r.status == "fail" for r in report.stages)


def test_pipeline_from_yaml_round_trip(tmp_path: Path):
    yml = tmp_path / "p.yaml"
    yml.write_text(
        "stages:\n"
        "  - parse\n"
        "  - verify_build_spec\n"
        "continue_on_failure: true\n"
    )
    p = Pipeline.from_yaml(yml)
    assert p.stages == ("parse", "verify_build_spec")
    assert p.continue_on_failure is True


def test_pipeline_report_to_dict_is_json_dumpable():
    report = run_pipeline(_spec(), Pipeline.smoke())
    d = report.to_dict()
    serialised = json.dumps(d)
    assert "stages" in serialised
    assert d["overall_status"] == "ok"


# ---------------------------------------------------------------------------
# input_parity_check end-to-end via parquet fixture.
# ---------------------------------------------------------------------------


def test_input_parity_check_passes_with_real_tokenizer_and_parquet(tmp_path: Path):
    p = tmp_path / "f.parquet"
    pq.write_table(pa.table({
        "input_ids": [list(range(8)) for _ in range(16)],
        "doc_ids": [i // 4 for i in range(16)],
    }), p)
    pipe = Pipeline(
        stages=("parse", "input_parity_check"),
        stage_options={"input_parity_check": {
            "parquet_path": str(p),
            "tokenizer_source": "cppmega_mlx/tokenizer/tokenizer.json",
        }},
    )
    report = run_pipeline(_spec(), pipe)
    assert report.overall_status == "ok"
    assert report.stages[1].status == "ok"
