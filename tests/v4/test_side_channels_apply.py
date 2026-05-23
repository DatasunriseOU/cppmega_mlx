"""V7-H11: side_channels.apply RPC + per-family resolution + gotcha echo."""

from __future__ import annotations

from cppmega_v4.jsonrpc.side_channel_apply_method import (
    SideChannelApplyParams, apply_side_channels,
)
from cppmega_v4.jsonrpc.schema import (
    FamilySpecPayload, SideChannelSpecPayload,
)


def _spec_one(family_name: str, columns: list[str],
              mode: str = "if_available") -> SideChannelSpecPayload:
    return SideChannelSpecPayload(
        mode="if_available",
        families={
            family_name: FamilySpecPayload(
                mode=mode, columns=columns,
                embedding="categorical", dropout=0.0,
            ),
        },
    )


def test_v7_h11_active_when_all_columns_present():
    r = apply_side_channels(SideChannelApplyParams(
        side_channels=_spec_one("platform", ["platform_ids"]),
        available_side_channels=["platform_ids", "token_ids"],
    ))
    assert r.ok is True
    assert r.active_count == 1
    assert r.inactive_count == 0
    assert r.families[0].family == "platform"
    assert r.families[0].active is True
    assert r.families[0].columns_missing == []
    assert r.families[0].reason == "all requested columns present"


def test_v7_h11_inactive_when_columns_absent_and_not_required():
    r = apply_side_channels(SideChannelApplyParams(
        side_channels=_spec_one("syntax",
                                 ["token_ast_depth", "token_sibling_index"]),
        available_side_channels=["doc_ids", "token_ids"],
    ))
    # Not required → gotcha NOT raised, but family is inactive.
    assert r.ok is True
    assert r.active_count == 0
    assert r.inactive_count == 1
    f = r.families[0]
    assert f.active is False
    assert f.columns_missing == ["token_ast_depth", "token_sibling_index"]
    assert "no requested columns present" in f.reason


def test_v7_h11_required_missing_columns_produce_gotcha_and_ok_false():
    r = apply_side_channels(SideChannelApplyParams(
        side_channels=_spec_one("platform", ["platform_ids"],
                                 mode="require"),
        available_side_channels=["doc_ids"],
    ))
    assert r.ok is False
    assert any(g.id == "side_channel_required_platform"
                for g in r.gotchas)
    assert r.families[0].active is False


def test_v7_h11_partial_columns_present_active_with_reason():
    r = apply_side_channels(SideChannelApplyParams(
        side_channels=_spec_one("structure",
                                 ["token_structure_ids", "token_dep_levels",
                                  "token_chunk_starts"]),
        available_side_channels=["token_structure_ids", "doc_ids"],
    ))
    f = r.families[0]
    assert f.active is True
    assert set(f.columns_present) == {"token_structure_ids"}
    assert set(f.columns_missing) == {"token_dep_levels",
                                       "token_chunk_starts"}
    assert "partial: 1/3" in f.reason


def test_v7_h11_mode_off_yields_inactive_without_warning():
    r = apply_side_channels(SideChannelApplyParams(
        side_channels=_spec_one("temporal_diff", ["doc_ids"], mode="off"),
        available_side_channels=["doc_ids"],
    ))
    assert r.ok is True
    assert r.active_count == 0
    assert r.families[0].reason == "family mode=off"
    assert r.gotchas == []


def test_v7_h11_dispatcher_routes_method_name():
    """End-to-end: side_channels.apply method must reach the handler."""
    from cppmega_v4.jsonrpc.dispatcher import dispatch
    response = dispatch({
        "jsonrpc": "2.0", "id": "T1", "method": "side_channels.apply",
        "params": {
            "side_channels": {
                "mode": "if_available", "families": {},
            },
            "available_side_channels": ["doc_ids"],
        },
    })
    assert response.error is None, response.error
    assert response.result is not None
    assert response.result["ok"] is True
    assert response.result["families"] == []


def test_v7_h11_dispatcher_rejects_unknown_extra_field():
    """Schema is strict (extra=forbid)."""
    from cppmega_v4.jsonrpc.dispatcher import dispatch
    response = dispatch({
        "jsonrpc": "2.0", "id": "T2", "method": "side_channels.apply",
        "params": {
            "side_channels": {"mode": "if_available", "families": {}},
            "unknown_extra": 42,
        },
    })
    assert response.error is not None
    assert response.error.code == -32602  # INVALID_PARAMS
