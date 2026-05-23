"""V8-R04 unit tests: architectures.auto_fit RPC."""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.auto_fit_method import (
    AutoFitHostInfo, AutoFitParams, auto_fit,
)
from cppmega_v4.jsonrpc.dispatcher import dispatch
from cppmega_v4.jsonrpc.schema import JsonRpcRequest


def test_llama3_8b_on_gb10_fits_full_size():
    """The canonical 8B preset fits on a single GB10 (137 GB unified)."""
    r = auto_fit(AutoFitParams(
        preset="llama3_8b",
        host_info=AutoFitHostInfo(topology="gb10_quarter")))
    assert r.topology == "gb10_quarter"
    assert r.headroom == 0.9
    assert r.fits is True
    assert r.scaled.hidden_size == 4096   # canonical (no scale-down needed)
    assert r.scaled.num_layers == 32
    assert r.scaled.estimated_bytes <= int(137e9 * 0.9)
    assert len(r.sharding.proposals) >= 1
    assert "hidden=4096" in r.reason
    assert "GB" in r.reason


def test_huge_preset_on_tiny_topology_still_returns():
    """When nothing fits, auto_fit still returns a result with fits=False
    instead of raising."""
    r = auto_fit(AutoFitParams(
        preset="qwen3_235b_a22b",
        host_info=AutoFitHostInfo(topology="gb10_quarter", headroom=0.5)))
    # The full 235B doesn't fit even on a GB10 at H=8192 — but
    # scale_down will pick the largest (H, L) ≤ target. If no cell on
    # the search grid fits, fits=False.
    assert r.topology == "gb10_quarter"
    # Just check the contract is well-formed.
    assert r.scaled.hidden_size >= 64
    assert r.scaled.num_layers >= 1
    assert isinstance(r.fits, bool)


def test_unknown_topology_rejected():
    with pytest.raises(ValueError, match="unknown topology"):
        auto_fit(AutoFitParams(
            preset="llama3_8b",
            host_info=AutoFitHostInfo(topology="nope")))


def test_host_info_omitted_uses_platform_probe():
    """Without host_info, auto_fit picks a topology from platform.get_info."""
    r = auto_fit(AutoFitParams(preset="llama3_8b"))
    assert r.topology in {
        "m3_ultra_solo", "gb10_quarter",
        "h100_8x", "h200_8x", "a100_8x", "b100_8x",
        "tpu_v6e_8", "tpu_v5p_4"}


def test_dispatch_end_to_end():
    req = JsonRpcRequest(
        jsonrpc="2.0", id="t-r04",
        method="architectures.auto_fit",
        params={"preset": "qwen3_dense_4b",
                "host_info": {"topology": "h100_8x"}},
    )
    resp = dispatch(req)
    assert resp.error is None, resp.error
    r = resp.result
    assert r["topology"] == "h100_8x"
    assert r["fits"] is True
    assert r["scaled"]["hidden_size"] >= 64
    assert isinstance(r["sharding"]["proposals"], list)
    assert "GB" in r["reason"]


def test_headroom_propagates_to_target_bytes():
    """Lower headroom should produce a smaller target budget."""
    r_high = auto_fit(AutoFitParams(
        preset="llama3_8b",
        host_info=AutoFitHostInfo(topology="gb10_quarter", headroom=0.9)))
    r_low = auto_fit(AutoFitParams(
        preset="llama3_8b",
        host_info=AutoFitHostInfo(topology="gb10_quarter", headroom=0.3)))
    # Lower headroom can never pick a *larger* model.
    assert (r_low.scaled.hidden_size * r_low.scaled.num_layers
            <= r_high.scaled.hidden_size * r_high.scaled.num_layers)
