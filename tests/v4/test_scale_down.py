"""V8-R02 unit tests: scale_down + architectures.scale_down RPC.

AC from the ticket: ``scale_down(llama3_8b, 1 GB)`` returns a result
whose estimated_bytes is <= 1 GB and >= 0.5 GB (i.e. the search finds
something meaningful, not just the floor).
"""

from __future__ import annotations

import pytest

from cppmega_v4.architectures.scale_down import (
    ScaleDownResult, scale_down, build_preset_specs_scaled,
)
from cppmega_v4.jsonrpc.dispatcher import dispatch
from cppmega_v4.jsonrpc.schema import JsonRpcRequest


ONE_GB = 1_073_741_824


def test_llama3_8b_scale_down_fits_1gb_target():
    """Canonical AC: 1 GB target lands in (0.5 GB, 1 GB]."""
    r = scale_down("llama3_8b", ONE_GB)
    assert isinstance(r, ScaleDownResult)
    assert r.fits is True
    assert r.estimated_bytes <= ONE_GB, (
        f"over budget: {r.estimated_bytes:,} > {ONE_GB:,}")
    assert r.estimated_bytes >= ONE_GB // 2, (
        f"under-shoots target: {r.estimated_bytes:,} < {ONE_GB // 2:,}")
    assert r.hidden_size >= 64
    assert r.num_layers >= 1
    assert r.scaled_down_from == (4096, 32)
    assert len(r.specs) == 2 * r.num_layers  # attn+mlp per layer


def test_specs_have_unique_names_per_layer():
    """Stamped specs must carry layer-disambiguated names."""
    r = scale_down("llama3_8b", ONE_GB)
    names = [s["name"] for s in r.specs]
    assert len(names) == len(set(names)), f"duplicate names: {names}"
    # All names should embed _L<int>
    assert all("_L" in n for n in names), names


def test_tiny_target_returns_minimum_with_fits_false_or_true():
    """A target so small nothing fits returns the minimum, flagged."""
    r = scale_down("llama3_8b", 1_000)  # 1 KB — impossibly tight
    assert r.hidden_size >= 64
    assert r.num_layers >= 1
    # If the floor fits 1KB it's still 'fits' — both states are valid.
    if not r.fits:
        assert r.estimated_bytes > 1_000


def test_huge_target_picks_canonical_or_close():
    """A 1 TB target should land at or near the canonical size."""
    r = scale_down("llama3_8b", 10**12)
    assert r.fits is True
    canon_h, canon_l = r.scaled_down_from
    # H and L are on the coarse log grid, so equality is exact.
    assert r.hidden_size == canon_h
    assert r.num_layers == canon_l


def test_qwen3_dense_4b_scale_down_fits_small_target():
    r = scale_down("qwen3_dense_4b", 512_000_000)  # 512 MB
    if r.fits:
        assert r.estimated_bytes <= 512_000_000


def test_unknown_preset_falls_back_to_generic():
    """Unknown preset uses the attention+mlp fallback unit."""
    r = scale_down("zzz_unknown_preset", ONE_GB)
    assert isinstance(r, ScaleDownResult)
    assert r.scaled_down_from == (4096, 32)  # generic Llama-3-shape
    assert len(r.specs) == 2 * r.num_layers


def test_invalid_target_rejected():
    with pytest.raises(ValueError):
        scale_down("llama3_8b", 0)
    with pytest.raises(ValueError):
        scale_down("llama3_8b", -1)


def test_build_preset_specs_scaled_unique_names():
    specs = build_preset_specs_scaled("llama3_8b", 512, 4)
    names = [s["name"] for s in specs]
    assert len(names) == len(set(names))
    assert len(specs) == 8  # 2 per layer × 4 layers
    assert all("_L" in n for n in names)


def test_scale_down_rpc_dispatch_end_to_end():
    """Dispatcher route: architectures.scale_down -> result envelope."""
    req = JsonRpcRequest(
        jsonrpc="2.0", id="t-r02-1",
        method="architectures.scale_down",
        params={"preset": "llama3_8b", "target_bytes": ONE_GB},
    )
    resp = dispatch(req)
    assert resp.error is None, resp.error
    assert resp.result is not None
    r = resp.result
    # Dispatcher returns a JSON-serialised dict on the wire.
    assert r["hidden_size"] >= 64
    assert r["num_layers"] >= 1
    assert r["estimated_bytes"] <= ONE_GB
    assert r["fits"] is True
    assert r["scaled_down_from"]["hidden_size"] == 4096
    assert r["scaled_down_from"]["num_layers"] == 32
    assert isinstance(r["specs"], list) and len(r["specs"]) >= 2


def test_rpc_rejects_negative_target_via_pydantic():
    """RPC accepts ints; the value error surfaces from the handler."""
    req = JsonRpcRequest(
        jsonrpc="2.0", id="t-r02-2",
        method="architectures.scale_down",
        params={"preset": "llama3_8b", "target_bytes": 0},
    )
    resp = dispatch(req)
    assert resp.error is not None
    # Detail surfaces the actual ValueError text from the handler.
    assert "target_bytes" in str(resp.error.data.get("detail", "")).lower()
