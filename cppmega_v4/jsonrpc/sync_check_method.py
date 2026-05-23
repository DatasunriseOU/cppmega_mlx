"""V8-R07: ``sync.check`` RPC handler."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.methods import (
    _cache_lookup, _cache_store, _graph_to_specs,
)
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.spec.sync_checker import run_sync_check


__all__ = [
    "SyncCheckParams",
    "SyncCheckResultModel",
    "SyncEntryModel",
    "SyncAdviceModel",
    "sync_check_method",
]


class SyncCheckParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: VerifyParams


class SyncEntryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    after_op: str
    reason: str


class SyncAdviceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: str
    fix: str
    confidence: str


class SyncCheckResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    necessary_syncs: list[SyncEntryModel] = Field(default_factory=list)
    redundant_syncs: list[SyncEntryModel] = Field(default_factory=list)
    advice: list[SyncAdviceModel] = Field(default_factory=list)
    z3_solver_status: str
    z3_elapsed_ms: float


def sync_check_method(
    params: SyncCheckParams, *, cache: LRUCache | None = None,
) -> SyncCheckResultModel:
    key, hit = _cache_lookup(cache, "sync.check", params)
    if hit is not None:
        return hit

    specs = _graph_to_specs(params.spec.graph)
    hidden = params.spec.dim_env.get("H", 64)
    res = run_sync_check(specs, hidden_size=hidden)
    out = SyncCheckResultModel(
        necessary_syncs=[SyncEntryModel(after_op=e.after_op,
                                         reason=e.reason)
                         for e in res.necessary_syncs],
        redundant_syncs=[SyncEntryModel(after_op=e.after_op,
                                         reason=e.reason)
                         for e in res.redundant_syncs],
        advice=[SyncAdviceModel(op=a.op, fix=a.fix,
                                 confidence=a.confidence)
                for a in res.advice],
        z3_solver_status=res.z3_solver_status,
        z3_elapsed_ms=res.z3_elapsed_ms,
    )
    _cache_store(cache, key, out)
    return out
