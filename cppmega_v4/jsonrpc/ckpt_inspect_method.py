"""V7-C03 honest-closure: ckpt.inspect RPC.

Reads safetensors metadata at the given path and returns the parsed
arch / train / opt sub-objects so the UI can render arch_hash,
opt_kind, cppmega_version on Load before the user commits to the
warm-start path. Pairs with stages.write_ckpt_metadata + read_ckpt_metadata
(stages.py:2020-2099) so the round-trip stays in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.jsonrpc.cache import LRUCache


class CkptInspectParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class CkptInspectResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    exists: bool
    has_metadata: bool = False
    cppmega_version: str | None = None
    arch_hash: str | None = None
    opt_kind: str | None = None
    opt_lr: float | None = None
    global_step: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


def ckpt_inspect(
    params: CkptInspectParams, *, cache: LRUCache | None = None,
) -> CkptInspectResult:
    p = Path(params.path)
    if not p.exists():
        return CkptInspectResult(exists=False)

    from cppmega_v4.runner.stages import read_ckpt_metadata

    try:
        meta = read_ckpt_metadata(str(p))
    except Exception as exc:
        return CkptInspectResult(
            exists=True, has_metadata=False, error=str(exc),
        )
    if meta is None:
        return CkptInspectResult(exists=True, has_metadata=False)

    arch = meta.get("arch") if isinstance(meta.get("arch"), dict) else {}
    train = meta.get("train") if isinstance(meta.get("train"), dict) else {}
    opt = meta.get("opt") if isinstance(meta.get("opt"), dict) else {}

    return CkptInspectResult(
        exists=True,
        has_metadata=True,
        cppmega_version=meta.get("cppmega_version"),
        arch_hash=arch.get("config_hash"),
        opt_kind=opt.get("kind"),
        opt_lr=opt.get("lr"),
        global_step=train.get("global_step"),
        raw=meta,
    )


__all__ = ["CkptInspectParams", "CkptInspectResult", "ckpt_inspect"]
