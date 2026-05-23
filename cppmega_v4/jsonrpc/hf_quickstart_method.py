"""V8-R09: ``data.hf_quickstart`` RPC handler.

Wraps :func:`scripts.data.hf_quickstart.hf_quickstart` for the GUI.
The handler runs synchronously — progress arrives on the
``/ws/data/{job_id}`` WebSocket from the underlying data_event_bus.
"""

from __future__ import annotations

import os
from pydantic import BaseModel, ConfigDict

from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.methods import _cache_lookup, _cache_store


__all__ = [
    "HfQuickstartParams",
    "HfQuickstartResult",
    "hf_quickstart_method",
]


class HfQuickstartParams(BaseModel):
    """Input — HF dataset + tokenizer + token budget."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    split: str = "train"
    tokenizer: str = "cppmega_v3"
    n_tokens: int = 100_000
    job_id: str | None = None
    out_dir: str | None = None
    text_field: str = "text"


class HfQuickstartResult(BaseModel):
    """Output — parquet path + token / doc counters + elapsed wall."""

    model_config = ConfigDict(extra="forbid")

    parquet_path: str
    n_tokens_written: int
    n_docs_seen: int
    elapsed_ms: float


def hf_quickstart_method(
    params: HfQuickstartParams, *, cache: LRUCache | None = None,
) -> HfQuickstartResult:
    """Run the HF quickstart and return the resulting parquet shard."""
    # Cache only on full param-key equality; HF jobs aren't typically
    # repeated with identical params so the hit rate is near-zero.
    key, hit = _cache_lookup(cache, "data.hf_quickstart", params)
    if hit is not None:
        return hit

    if os.environ.get("VBGUI_DISABLE_NETWORK") == "1":
        raise RuntimeError(
            "HF Hub network access disabled via VBGUI_DISABLE_NETWORK")

    from scripts.data.hf_quickstart import hf_quickstart as _hf
    r = _hf(
        dataset_id=params.dataset_id,
        split=params.split,
        tokenizer=params.tokenizer,
        n_tokens=params.n_tokens,
        job_id=params.job_id,
        out_dir=params.out_dir,
        text_field=params.text_field,
    )
    out = HfQuickstartResult(
        parquet_path=r.parquet_path,
        n_tokens_written=r.n_tokens_written,
        n_docs_seen=r.n_docs_seen,
        elapsed_ms=r.elapsed_ms,
    )
    _cache_store(cache, key, out)
    return out
