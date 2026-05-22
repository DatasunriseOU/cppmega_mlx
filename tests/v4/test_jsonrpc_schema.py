"""VBGui F-A schema tests — Pydantic models + envelope discipline.

Locks the wire format described in VisualBuilderPlan.md §5.2. Any
change to a payload shape must be reflected here (and in the bumped
SCHEMA_VERSION).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cppmega_v4.buildspec import DataMaterializationSpec, SideChannelSpec
from cppmega_v4.jsonrpc import (
    EVENT_TAXONOMY,
    METHOD_REGISTRY,
    SCHEMA_VERSION,
    BuildPresetSpecsParams,
    ErrorCode,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    ProbeRunParams,
    SuggestAdaptersParams,
    SuggestShardingParams,
    VerifyParams,
)
from cppmega_v4.jsonrpc.schema import (
    EdgeResolution,
    DataMaterializationSpecPayload,
    FamilySpecPayload,
    GraphSpec,
    InferenceEnrichmentSpecPayload,
    LossSpecPayload,
    OptimSpecPayload,
    PerBrickMemory,
    PipelineAbortParams,
    PipelineAbortResult,
    PipelinePayload,
    PipelineRunParams,
    PipelineRunResult,
    ResolvedGraph,
    ShardingSpecPayload,
    SideChannelSpecPayload,
    StageResult,
    TopologyPayload,
)


# ---------------------------------------------------------------------------
# Version + registries.
# ---------------------------------------------------------------------------


def test_schema_version_is_semver():
    parts = SCHEMA_VERSION.split(".")
    assert len(parts) == 3
    for p in parts:
        int(p)


def test_method_registry_covers_all_documented_methods():
    expected = {
        "verify", "suggest_sharding", "suggest_adapters",
        "build_preset_specs", "probe.run", "pipeline.run", "pipeline.abort",
        "backend.status",
    }
    assert expected <= METHOD_REGISTRY


def test_event_taxonomy_locks_documented_events():
    documented = {
        "node.move", "graph.mutate", "param.edit",
        "loss.update", "optim.update",
        "rewriter.add", "rewriter.remove", "rewriter.reorder",
        "sharding.update", "verify.request", "sharding.request",
        "build.request", "backend.status", "probe.run", "pipeline.run",
        "pipeline.abort",
    }
    assert documented <= EVENT_TAXONOMY


# ---------------------------------------------------------------------------
# JSON-RPC envelope.
# ---------------------------------------------------------------------------


def test_request_envelope_defaults():
    req = JsonRpcRequest(id="x", method="verify")
    assert req.jsonrpc == "2.0"
    assert req.params == {}


def test_request_envelope_rejects_extra_fields():
    with pytest.raises(ValidationError):
        JsonRpcRequest(id="x", method="verify", extra="nope")


def test_response_envelope_round_trip():
    resp = JsonRpcResponse(id=1, result={"foo": "bar"})
    serialised = resp.model_dump(mode="json", exclude_none=True)
    assert serialised == {"jsonrpc": "2.0", "id": 1, "result": {"foo": "bar"}}
    restored = JsonRpcResponse.model_validate(serialised)
    assert restored == resp


def test_error_envelope_serialises_data():
    e = JsonRpcError(code=ErrorCode.INVALID_PARAMS, message="bad",
                     data={"errors": [{"k": "v"}]})
    d = e.model_dump(mode="json", exclude_none=True)
    assert d == {"code": -32602, "message": "bad",
                 "data": {"errors": [{"k": "v"}]}}


def test_error_codes_match_jsonrpc_2_0_reserved():
    assert ErrorCode.PARSE_ERROR == -32700
    assert ErrorCode.INVALID_REQUEST == -32600
    assert ErrorCode.METHOD_NOT_FOUND == -32601
    assert ErrorCode.INVALID_PARAMS == -32602
    assert ErrorCode.INTERNAL_ERROR == -32603


# ---------------------------------------------------------------------------
# Domain payload validation.
# ---------------------------------------------------------------------------


def _verify_params_payload():
    return {
        "graph": {
            "nodes": [
                {"id": "a", "kind": "mlp"},
                {"id": "b", "kind": "mlp"},
            ],
            "edges": [{"src": "a", "dst": "b"}],
        },
        "dim_env": {"B": 1, "S": 4, "H": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["b"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 3e-4,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    }


def test_verify_params_round_trip():
    payload = _verify_params_payload()
    parsed = VerifyParams.model_validate(payload)
    assert parsed.graph.nodes[0].id == "a"
    assert parsed.training is True
    assert parsed.side_channels.mode == "auto"
    assert parsed.data_materialization.packing_policy == "best_fit"
    assert parsed.data_materialization.max_seq_len == 4096
    assert parsed.side_channels.families["platform"].columns == [
        "platform_ids", "source_platform_ids",
    ]
    serial = parsed.model_dump(mode="json")
    assert "graph" in serial
    assert serial["side_channels"]["families"]["structure"]["mode"] == "if_available"
    assert serial["data_materialization"]["required_token_fields"] == [
        "input_ids",
        "target_ids",
        "loss_mask",
        "doc_ids",
        "pack_id",
        "valid_token_count",
        "num_docs",
    ]


def test_verify_params_rejects_unknown_field():
    payload = _verify_params_payload()
    payload["mystery"] = 1
    with pytest.raises(ValidationError):
        VerifyParams.model_validate(payload)


def test_side_channel_payload_validates_policy_values():
    payload = SideChannelSpecPayload(
        families={
            "platform": FamilySpecPayload(
                mode="require",
                columns=["platform_ids"],
                dropout=0.2,
                fallback="error",
            ),
            "syntax": FamilySpecPayload(mode="off"),
        },
        inference=InferenceEnrichmentSpecPayload(
            source="prompt_only",
            timeout_ms=250,
        ),
    )
    assert payload.families["platform"].fallback == "error"

    with pytest.raises(ValidationError):
        FamilySpecPayload(mode="sometimes")
    with pytest.raises(ValidationError):
        FamilySpecPayload(dropout=1.1)
    with pytest.raises(ValidationError):
        InferenceEnrichmentSpecPayload(source="magic")


def test_side_channel_payload_matches_buildspec_defaults():
    payload = SideChannelSpecPayload.model_validate(SideChannelSpec().to_dict())
    assert payload.model_dump(mode="json") == SideChannelSpec().to_dict()

    materialization = DataMaterializationSpecPayload.model_validate(
        DataMaterializationSpec().to_dict()
    )
    assert materialization.model_dump(mode="json") == DataMaterializationSpec().to_dict()


def test_loss_spec_payload_accepts_known_kinds():
    LossSpecPayload(kind="cross_entropy", head_outputs=["x"])
    LossSpecPayload(kind="mtp_weighted", head_outputs=["x", "y"],
                    params={"beta": 0.6})


def test_optim_spec_payload_requires_groups():
    with pytest.raises(ValidationError):
        OptimSpecPayload(kind="adamw")
    payload = OptimSpecPayload(
        kind="adamw",
        groups=[{"matcher": "all", "lr": 1e-4}],
        mixed_precision=False,
    )
    assert payload.mixed_precision is False


def test_sharding_spec_payload_accepts_topology_and_axes():
    s = ShardingSpecPayload(
        topology=TopologyPayload(factory="h100_8x", kwargs={}),
        axis_assignments=[{"axis_name": "dp", "kind": "fsdp2", "degree": 8}],
    )
    assert s.compile_mode == "regional"
    assert s.fp8_enabled is False


def test_graph_spec_default_edges_empty():
    g = GraphSpec(nodes=[{"id": "a", "kind": "mlp"}])
    assert g.edges == []


def test_edge_resolution_enforces_severity_enum():
    EdgeResolution(src="a", dst="b", shape=[1, 4, 64],
                   matched=True, severity="info")
    with pytest.raises(ValidationError):
        EdgeResolution(src="a", dst="b", shape=[1, 4, 64],
                       matched=True, severity="critical")


def test_resolved_graph_defaults_to_no_errors():
    r = ResolvedGraph(edges=[])
    assert r.has_errors is False
    assert r.diagnostics == []


def test_per_brick_memory_defaults_kv_zero():
    m = PerBrickMemory(params_bytes=1, activations_bytes=2)
    assert m.kv_cache_bytes == 0


def test_build_preset_specs_params_validates_hidden():
    p = BuildPresetSpecsParams(preset_name="llama3_8b", hidden_size=64)
    assert p.num_layers is None


def test_suggest_adapters_params_defaults_max_steps():
    p = SuggestAdaptersParams(
        graph={"nodes": [{"id": "a", "kind": "mlp"}]},
        dim_env={"B": 1},
        producer="a", consumer="a",
    )
    assert p.max_steps == 4


def test_suggest_sharding_params_requires_topology():
    with pytest.raises(ValidationError):
        SuggestShardingParams(
            graph={"nodes": [{"id": "a", "kind": "mlp"}]},
            dim_env={"B": 1},
            loss={"kind": "cross_entropy", "head_outputs": ["a"]},
            optim={"kind": "adamw", "groups": [{"matcher": "all", "lr": 1e-4}]},
        )


# ---------------------------------------------------------------------------
# Pipeline payloads.
# ---------------------------------------------------------------------------


def test_pipeline_payload_defaults():
    p = PipelinePayload(stages=["parse", "verify"])
    assert p.stage_options == {}
    assert p.continue_on_failure is False


def test_pipeline_run_params_nests_spec_and_pipeline():
    p = PipelineRunParams(
        spec=VerifyParams.model_validate(_verify_params_payload()),
        pipeline=PipelinePayload(stages=["parse"]),
    )
    assert p.pipeline.stages == ["parse"]


def test_pipeline_abort_params_and_result():
    params = PipelineAbortParams(run_id="train-1")
    assert params.run_id == "train-1"
    result = PipelineAbortResult(run_id=params.run_id)
    assert result.status == "abort_requested"
    assert result.run_id == "train-1"


def test_stage_result_enum():
    StageResult(name="parse", status="ok", elapsed_ms=1.0)
    StageResult(name="parse", status="skipped", elapsed_ms=0.0)
    StageResult(name="parse", status="fail", elapsed_ms=1.0,
                error={"type": "ShapeMismatch", "detail": "X"})
    StageResult(name="train", status="cancelled", elapsed_ms=1.0)
    with pytest.raises(ValidationError):
        StageResult(name="parse", status="weird", elapsed_ms=1.0)


def test_pipeline_run_result_overall_status_enum():
    PipelineRunResult(stages=[], overall_status="ok", total_elapsed_ms=0.0)
    PipelineRunResult(stages=[], overall_status="fail", total_elapsed_ms=0.0)
    PipelineRunResult(
        stages=[StageResult(name="train", status="cancelled", elapsed_ms=0.0)],
        overall_status="cancelled",
        total_elapsed_ms=0.0,
    )
    with pytest.raises(ValidationError):
        PipelineRunResult(stages=[], overall_status="meh", total_elapsed_ms=0.0)


# ---------------------------------------------------------------------------
# Probe envelope.
# ---------------------------------------------------------------------------


def test_probe_run_params_carries_paths():
    p = ProbeRunParams(
        graph={"nodes": [{"id": "a", "kind": "mlp"}]},
        dim_env={"B": 1},
        loss={"kind": "cross_entropy", "head_outputs": ["a"]},
        optim={"kind": "adamw", "groups": [{"matcher": "all", "lr": 1e-4}]},
        tokenizer_source="/tmp/x", parquet_path="/tmp/y",
    )
    assert p.probe_hidden_size == 64
    assert p.run_dry_forward is True
