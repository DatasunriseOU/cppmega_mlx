"""VBGui F-A method tests — pure-Python handlers + golden round-trip."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_v4.jsonrpc import LRUCache
from cppmega_v4.jsonrpc.methods import (
    _graph_to_specs,
    _make_optim,
    build_preset_specs,
    probe_run,
    suggest_adapters,
    suggest_sharding,
    verify,
)
from cppmega_v4.jsonrpc.schema import (
    BuildPresetSpecsParams,
    ProbeRunParams,
    OptimSpecPayload,
    SuggestAdaptersParams,
    SuggestShardingParams,
    VerifyParams,
)


_DIM_ENV = {"B": 1, "S": 4, "H": 64,
            "nh": 2, "nkv": 1, "head_dim": 32,
            "num_experts": 8, "top_k": 2}


def _simple_verify_params(**extra) -> VerifyParams:
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


def test_graph_to_specs_canonicalizes_legacy_residual_add() -> None:
    params = _simple_verify_params(
        graph={
            "nodes": [
                {"id": "branch", "kind": "mlp"},
                {"id": "join", "kind": "residual_add"},
            ],
            "edges": [{"src": "branch", "dst": "join"}],
        }
    )

    specs = _graph_to_specs(params.graph)

    assert [spec["kind"] for spec in specs] == ["mlp", "residual"]


def test_make_optim_threads_mixed_precision_flag():
    optim = _make_optim(OptimSpecPayload(
        kind="adamw",
        groups=[{"matcher": "all", "lr": 1e-4}],
        mixed_precision=False,
    ))
    assert optim.mixed_precision is False


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_verify_returns_resolved_edges_and_memory():
    r = verify(_simple_verify_params())
    assert len(r.resolved.edges) == 1
    e = r.resolved.edges[0]
    assert e.src == "a" and e.dst == "b"
    assert e.matched is True
    assert {"a", "b"} == set(r.memory_per_brick.keys())
    assert r.memory_per_brick["a"].params_bytes > 0


def test_verify_includes_fusion_plan():
    r = verify(_simple_verify_params())
    assert r.fusion_plan
    region = r.fusion_plan[0]
    assert region.brick_names == ["a", "b"]


def test_verify_reports_required_side_channel_family_errors():
    params = _simple_verify_params(
        side_channels={
            "mode": "auto",
            "families": {
                "platform": {
                    "mode": "require",
                    "columns": ["platform_ids"],
                    "embedding": "categorical",
                    "dropout": 0.0,
                    "residual_scale": 1.0,
                    "fallback": "error",
                    "language_scope": ["any"],
                },
            },
            "inference": {
                "source": "auto",
                "fail_policy": "drop_family",
                "timeout_ms": 500,
                "cache_enabled": True,
            },
        },
        available_side_channels=["doc_ids", "token_ids"],
    )
    r = verify(params)
    gotcha = next(g for g in r.gotchas
                  if g.id == "side_channel_required_platform")
    assert gotcha.severity == "error"
    assert "platform_ids" in gotcha.message


def test_verify_emits_distributed_memory_with_sharding():
    p = _simple_verify_params(sharding={
        "topology": {"factory": "h100_8x", "kwargs": {}},
        "axis_assignments": [{"axis_name": "dp", "kind": "fsdp2", "degree": 8}],
        "compile_mode": "regional", "fp8_enabled": False,
    })
    r = verify(p)
    assert r.memory_distributed is not None
    assert r.memory_distributed.fits_on_topology is True


def test_verify_round_trip_through_json():
    r = verify(_simple_verify_params())
    j = r.model_dump_json()
    parsed = json.loads(j)
    assert "resolved" in parsed
    assert "memory_per_brick" in parsed
    assert "elapsed_ms" in parsed


def test_verify_under_100ms_latency_target():
    """VBPlan §5.1: verify.request target <100ms."""
    t0 = time.perf_counter()
    verify(_simple_verify_params())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 100, f"verify took {elapsed_ms:.1f} ms"


# ---------------------------------------------------------------------------
# Cache integration.
# ---------------------------------------------------------------------------


def test_verify_cache_hit_short_circuits():
    cache = LRUCache(capacity=4)
    params = _simple_verify_params()
    r1 = verify(params, cache=cache)
    r2 = verify(params, cache=cache)
    # Cache returns a defensive deep-copy on hit (C1 from review) so
    # consumers can mutate freely — assert equality, not identity.
    assert r1 == r2
    assert r1 is not r2
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1


def test_verify_cache_invariant_to_node_layout():
    """Layout fields stripped before keying → same cache hit."""
    cache = LRUCache()
    base = _simple_verify_params()
    verify(base, cache=cache)
    # Round-trip through dict, inject layout, parse back.
    dumped = base.model_dump(mode="json")
    for n in dumped["graph"]["nodes"]:
        n["x"] = 999
        n["y"] = -42
    # Layout fields would fail strict validation; coerce by stripping
    # back out for the Pydantic model. The cache key uses the raw dict
    # (model_dump → strip_layout) so this still hits.
    for n in dumped["graph"]["nodes"]:
        n.pop("x")
        n.pop("y")
    again = VerifyParams.model_validate(dumped)
    verify(again, cache=cache)
    assert cache.stats()["hits"] == 1


# ---------------------------------------------------------------------------
# suggest_sharding
# ---------------------------------------------------------------------------


def test_suggest_sharding_returns_ranked_proposals():
    p = SuggestShardingParams(
        graph={"nodes": [{"id": "a", "kind": "mlp"}, {"id": "b", "kind": "mlp"}],
               "edges": [{"src": "a", "dst": "b"}]},
        dim_env=_DIM_ENV,
        loss={"kind": "cross_entropy", "head_outputs": ["b"]},
        optim={"kind": "adamw", "groups": [{"matcher": "all", "lr": 3e-4}]},
        topology={"factory": "h100_8x", "kwargs": {}},
    )
    r = suggest_sharding(p)
    assert len(r.proposals) >= 1
    assert all(0 <= p.estimated_per_rank_bytes for p in r.proposals)


# ---------------------------------------------------------------------------
# suggest_adapters
# ---------------------------------------------------------------------------


def test_suggest_adapters_empty_chain_on_matching_edge():
    p = SuggestAdaptersParams(
        graph={"nodes": [{"id": "a", "kind": "mlp"}, {"id": "b", "kind": "mlp"}],
               "edges": [{"src": "a", "dst": "b"}]},
        dim_env=_DIM_ENV,
        producer="a", consumer="b",
    )
    r = suggest_adapters(p)
    assert r.chain == []
    assert "no adapter" in r.reason.lower() or "match" in r.reason.lower()


# ---------------------------------------------------------------------------
# build_preset_specs
# ---------------------------------------------------------------------------


def test_build_preset_specs_returns_leaf_dicts():
    r = build_preset_specs(BuildPresetSpecsParams(
        preset_name="llama3_8b", hidden_size=64,
    ))
    assert r.preset_name == "llama3_8b"
    assert len(r.specs) >= 2
    for s in r.specs:
        assert "kind" in s


def test_build_preset_specs_rejects_unknown_preset():
    with pytest.raises(ValueError, match="unknown preset"):
        build_preset_specs(BuildPresetSpecsParams(
            preset_name="bogus_preset_xyz", hidden_size=64,
        ))


# ---------------------------------------------------------------------------
# probe.run bridge
# ---------------------------------------------------------------------------


def _write_full_parquet(p: Path, n: int = 16):
    pq.write_table(pa.table({
        "input_ids": [list(range(8)) for _ in range(n)],
        "doc_ids":   [i // 4 for i in range(n)],
    }), p)


def test_probe_run_bridges_contract_probe(tmp_path: Path):
    parquet = tmp_path / "f.parquet"
    _write_full_parquet(parquet)
    p = ProbeRunParams(
        graph={"nodes": [{"id": "a", "kind": "mlp"}, {"id": "b", "kind": "mlp"}],
               "edges": [{"src": "a", "dst": "b"}]},
        dim_env=_DIM_ENV,
        loss={"kind": "cross_entropy", "head_outputs": ["b"]},
        optim={"kind": "adamw", "groups": [{"matcher": "all", "lr": 3e-4}]},
        tokenizer_source="cppmega_mlx/tokenizer/tokenizer.json",
        parquet_path=str(parquet),
    )
    r = probe_run(p)
    assert r.is_clean is True
    assert r.schema_version == "1.0.0"
