"""V8-R08 pytest: catalog.list_options('feature_injectors').

Asserts the category returns the 5 V8 injection options with the
'rewriter:<Name>' or 'brick:<kind>' paper_ref convention the UI parses.
"""

from __future__ import annotations

from cppmega_v4.jsonrpc.dispatcher import dispatch
from cppmega_v4.jsonrpc.schema import JsonRpcRequest


def test_catalog_returns_feature_injectors():
    req = JsonRpcRequest(
        jsonrpc="2.0", id="t-r08-1",
        method="catalog.list_options",
        params={"category": "feature_injectors"},
    )
    resp = dispatch(req)
    assert resp.error is None, resp.error
    names = [o["name"] for o in resp.result["options"]]
    assert set(names) >= {
        "mtp_weighted", "ifim_shaped", "mhc_attn_bias",
        "engram", "ngram_2_3_4"}
    # paper_ref naming convention
    for opt in resp.result["options"]:
        assert opt["paper_ref"].startswith(("rewriter:", "brick:")), opt


def test_mtp_rewriter_paper_ref():
    req = JsonRpcRequest(
        jsonrpc="2.0", id="t-r08-2",
        method="catalog.list_options",
        params={"category": "feature_injectors"},
    )
    resp = dispatch(req)
    by_name = {o["name"]: o for o in resp.result["options"]}
    assert by_name["mtp_weighted"]["paper_ref"] == "rewriter:MTPRewriter"
    assert by_name["engram"]["paper_ref"] == "brick:engram"
