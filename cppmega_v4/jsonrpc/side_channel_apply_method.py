"""V7-H11: side_channels.apply RPC — verify a side-channel config.

The Sidebar SideChannelsTab assembles a SideChannelSpecPayload (per-
family mode/embedding/fallback + inference enrichment policy + a list
of available_side_channels reported by the parquet shard) and asks
the backend "is this config self-consistent, and if I plumb it into
verify+train, what does it actually resolve to?".

Before V7-H11 the Apply button only wrote to local spec; user had no
backend confirmation that, e.g., a required family that asks for
columns the parquet doesn't carry would fail at verify-time. This RPC
runs the same gotcha-checker that verify does, plus echoes the set
of families that would actually be active (mode != off + columns
present in available_side_channels) so the UI can show "applied: 3 of
5 families active".
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.jsonrpc.cache import LRUCache, canonical_sha256
from cppmega_v4.jsonrpc.schema import (
    GotchaPayload, SideChannelSpecPayload,
)


class SideChannelApplyParams(BaseModel):
    """Request: verify a side-channel config against available columns."""

    model_config = ConfigDict(extra="forbid")

    side_channels: SideChannelSpecPayload
    available_side_channels: list[str] = Field(
        default_factory=lambda: ["doc_ids", "token_ids"],
    )


class FamilyApplyStatus(BaseModel):
    """Per-family resolution: did this family actually engage?"""

    model_config = ConfigDict(extra="forbid")

    family: str
    mode: str
    active: bool
    reason: str
    columns_requested: list[str]
    columns_present: list[str]
    columns_missing: list[str]


class SideChannelApplyResult(BaseModel):
    """V7-H11: side-channel apply verdict + per-family resolution."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    families: list[FamilyApplyStatus]
    gotchas: list[GotchaPayload]
    active_count: int
    inactive_count: int
    elapsed_ms: float


def _resolve_family(
    name: str, policy: Any, available: frozenset[str],
    global_mode: str,
) -> FamilyApplyStatus:
    requested = list(policy.columns)
    present = [c for c in requested if c in available]
    missing = [c for c in requested if c not in available]
    mode = policy.mode
    effective_required = (mode == "require") or (global_mode == "require")
    if mode == "off":
        return FamilyApplyStatus(
            family=name, mode=mode, active=False,
            reason="family mode=off",
            columns_requested=requested,
            columns_present=present, columns_missing=missing,
        )
    if not requested:
        return FamilyApplyStatus(
            family=name, mode=mode, active=False,
            reason="no columns declared for family",
            columns_requested=requested,
            columns_present=present, columns_missing=missing,
        )
    if not present:
        reason = (
            "required columns missing"
            if effective_required else
            "no requested columns present in shard"
        )
        return FamilyApplyStatus(
            family=name, mode=mode, active=False, reason=reason,
            columns_requested=requested,
            columns_present=present, columns_missing=missing,
        )
    if missing and effective_required:
        return FamilyApplyStatus(
            family=name, mode=mode, active=False,
            reason=f"required columns missing: {', '.join(missing)}",
            columns_requested=requested,
            columns_present=present, columns_missing=missing,
        )
    reason = (
        "all requested columns present"
        if not missing else
        f"partial: {len(present)}/{len(requested)} columns present"
    )
    return FamilyApplyStatus(
        family=name, mode=mode, active=True, reason=reason,
        columns_requested=requested,
        columns_present=present, columns_missing=missing,
    )


def apply_side_channels(
    params: SideChannelApplyParams,
    *,
    cache: LRUCache | None = None,
) -> SideChannelApplyResult:
    """Verify side-channel config; report per-family resolution + gotchas."""
    cache_key = "side-channel-apply::" + canonical_sha256(
        params.model_dump(mode="json"),
    )
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit

    t0 = time.perf_counter()
    available = frozenset(params.available_side_channels)
    global_mode = params.side_channels.mode
    families: list[FamilyApplyStatus] = []
    for name, policy in sorted(params.side_channels.families.items()):
        families.append(_resolve_family(name, policy, available, global_mode))

    # Reuse the same gotcha helper that verify uses so apply <-> verify
    # agree on what's an error.
    from cppmega_v4.jsonrpc.methods import _side_channel_policy_gotchas
    gotchas = _side_channel_policy_gotchas(params.side_channels, available)

    active = sum(1 for f in families if f.active)
    inactive = sum(1 for f in families if not f.active)
    ok = not any(g.severity == "error" for g in gotchas)

    result = SideChannelApplyResult(
        ok=ok,
        families=families,
        gotchas=gotchas,
        active_count=active,
        inactive_count=inactive,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 3),
    )
    if cache is not None:
        cache.set(cache_key, result)
    return result


__all__ = [
    "SideChannelApplyParams", "SideChannelApplyResult",
    "FamilyApplyStatus", "apply_side_channels",
]
