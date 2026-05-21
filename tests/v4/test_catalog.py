"""E7-10 tests: tooltip catalogue + catalog.explain / catalog.list_options RPC."""

from __future__ import annotations

import pytest

from cppmega_v4.explain import CATALOG, CATEGORIES, get_entry, list_options
from cppmega_v4.explain.catalog import ExplainEntry
from cppmega_v4.jsonrpc import dispatch
from cppmega_v4.jsonrpc.catalog_methods import (
    catalog_explain, catalog_list_options,
)
from cppmega_v4.jsonrpc.schema import (
    METHOD_REGISTRY,
    CatalogExplainParams,
    CatalogListOptionsParams,
)


# ---------------------------------------------------------------------------
# Catalogue coverage
# ---------------------------------------------------------------------------


def test_catalog_has_entries_for_every_category():
    for cat in CATEGORIES:
        opts = list_options(cat)
        assert opts, f"category {cat!r} has no entries"


def test_catalog_covers_all_seven_optimizers():
    for kind in ("adamw", "muon", "muon_adamw_hybrid", "lion",
                 "lion8bit", "adam8bit", "sgd"):
        entry = get_entry("optimizer", kind)
        assert entry is not None, f"missing optimizer entry: {kind}"
        assert entry.summary
        assert entry.when_to_use
        assert entry.when_to_avoid


def test_catalog_covers_six_activations():
    for name in ("gelu", "relu", "relu2", "sqrelu", "silu", "swiglu"):
        entry = get_entry("activation", name)
        assert entry is not None, f"missing activation entry: {name}"


def test_catalog_covers_three_norms():
    for name in ("rmsnorm", "layernorm", "none"):
        entry = get_entry("norm", name)
        assert entry is not None


def test_catalog_covers_six_schedules():
    for name in ("constant", "linear_warmup", "cosine", "wsd",
                 "inv_sqrt", "polynomial"):
        entry = get_entry("schedule", name)
        assert entry is not None


def test_catalog_covers_five_losses():
    for name in ("cross_entropy", "mtp_weighted", "ifim_shaped",
                 "mhc_attn_bias", "custom"):
        entry = get_entry("loss", name)
        assert entry is not None


def test_lion_entry_warns_about_lr_ceiling_in_gotchas():
    entry = get_entry("optimizer", "lion")
    assert entry is not None
    assert any("5e-4" in g or "5e-04" in g for g in entry.gotchas), \
        f"Lion entry should warn about 5e-4 lr ceiling; got: {entry.gotchas}"


def test_lion_recommended_lr_matches_factory_default():
    """Tooltip's recommended lr must match the factory default
    in optim_spec.py so 'Apply recommended' in the UI is a no-op
    against the factory."""
    from cppmega_v4.buildspec.optim_spec import lion
    entry = get_entry("optimizer", "lion")
    factory_spec = lion()
    assert entry is not None
    assert entry.recommended_params["lr"] == factory_spec.groups[0].lr


def test_muon_entry_lists_ns_steps_default():
    entry = get_entry("optimizer", "muon")
    assert entry is not None
    assert "ns_steps" in entry.recommended_params
    assert entry.recommended_params["ns_steps"] == 5


def test_swiglu_entry_marked_as_requiring_gated_brick():
    entry = get_entry("activation", "swiglu")
    assert entry is not None
    text = (entry.summary + entry.when_to_use + entry.when_to_avoid +
            " ".join(entry.gotchas)).lower()
    assert "gated" in text or "gate" in text


# ---------------------------------------------------------------------------
# Direct handler tests
# ---------------------------------------------------------------------------


def test_catalog_explain_returns_known_entry():
    result = catalog_explain(
        CatalogExplainParams(category="optimizer", name="lion"),
    )
    assert result.entry is not None
    assert result.entry.name == "lion"
    assert result.not_found_message is None


def test_catalog_explain_returns_stub_for_unknown():
    result = catalog_explain(
        CatalogExplainParams(category="optimizer", name="sophia"),
    )
    assert result.entry is None
    assert result.not_found_message is not None
    assert "sophia" in result.not_found_message


def test_catalog_explain_payload_is_serialisable():
    """Result must round-trip through Pydantic JSON without losing
    recommended_params or gotchas."""
    result = catalog_explain(
        CatalogExplainParams(category="schedule", name="cosine"),
    )
    serialised = result.model_dump(mode="json")
    assert serialised["entry"]["name"] == "cosine"
    assert "warmup_steps" in serialised["entry"]["recommended_params"]
    assert serialised["entry"]["paper_url"]


def test_catalog_list_options_returns_six_schedules():
    result = catalog_list_options(
        CatalogListOptionsParams(category="schedule"),
    )
    names = [o.name for o in result.options]
    assert set(names) == {
        "constant", "linear_warmup", "cosine", "wsd",
        "inv_sqrt", "polynomial",
    }
    # Each option carries summary and paper_ref (some optional)
    for opt in result.options:
        assert opt.summary


def test_catalog_list_options_unknown_category_empty():
    result = catalog_list_options(
        CatalogListOptionsParams(category="not_a_thing"),
    )
    assert result.options == []


# ---------------------------------------------------------------------------
# RPC dispatcher integration
# ---------------------------------------------------------------------------


def test_method_registry_contains_catalog_methods():
    assert "catalog.explain" in METHOD_REGISTRY
    assert "catalog.list_options" in METHOD_REGISTRY


def test_dispatch_catalog_explain_lion():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "catalog.explain",
        "params": {"category": "optimizer", "name": "lion"},
    })
    assert resp.error is None, resp.error
    assert resp.result["entry"]["name"] == "lion"
    assert "Chen" in str(resp.result["entry"]["paper_ref"])


def test_dispatch_catalog_explain_unknown():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "catalog.explain",
        "params": {"category": "optimizer", "name": "made_up"},
    })
    assert resp.error is None
    assert resp.result["entry"] is None
    assert resp.result["not_found_message"] is not None


def test_dispatch_catalog_list_options_activation():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "catalog.list_options",
        "params": {"category": "activation"},
    })
    assert resp.error is None
    options = resp.result["options"]
    # E7-13 extended the activation registry from 6 → 10 entries.
    assert len(options) == 10
    names = {o["name"] for o in options}
    assert names == {"gelu", "relu", "relu2", "sqrelu", "silu", "mish",
                     "swiglu", "geglu", "reglu", "xielu"}


def test_dispatch_catalog_explain_rejects_extra_params():
    """extra='forbid' on CatalogExplainParams must reject unknown
    fields (defends the schema contract)."""
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "catalog.explain",
        "params": {"category": "optimizer", "name": "lion",
                   "unknown_field": True},
    })
    assert resp.error is not None
    assert resp.error.code == -32602  # INVALID_PARAMS


# ---------------------------------------------------------------------------
# Cache hit-rate (perf gate per Plan §6: catalog.explain < 5 ms cached)
# ---------------------------------------------------------------------------


def test_cache_returns_equal_value_on_repeat():
    """LRUCache deep-copies on get, so identity differs but contents must
    match. Cache hit is proven by `stats().hits` going up."""
    from cppmega_v4.jsonrpc.cache import LRUCache
    cache = LRUCache(capacity=10)
    r1 = catalog_explain(
        CatalogExplainParams(category="optimizer", name="lion"),
        cache=cache,
    )
    stats_before = cache.stats()
    r2 = catalog_explain(
        CatalogExplainParams(category="optimizer", name="lion"),
        cache=cache,
    )
    stats_after = cache.stats()
    assert r1 == r2
    assert stats_after["hits"] == stats_before["hits"] + 1, \
        f"expected cache hit, got stats {stats_before} -> {stats_after}"
