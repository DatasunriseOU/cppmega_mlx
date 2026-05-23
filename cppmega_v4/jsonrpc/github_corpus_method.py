"""V8-R10: ``data.github_corpus`` RPC handler."""

from __future__ import annotations

import os
from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.methods import _cache_lookup, _cache_store


__all__ = [
    "GithubCorpusParams",
    "GithubCorpusResultModel",
    "github_corpus_method",
]


class GithubCorpusParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_url: str
    max_commits: int = 50
    max_tokens: int = 50_000
    tokenizer: str = "cppmega_v3"
    use_clang: bool = False
    use_treesitter: bool = True
    job_id: str | None = None
    out_dir: str | None = None


class GithubCorpusResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parquet_path: str
    n_tokens_written: int
    n_docs_seen: int
    side_channels: list[str] = Field(default_factory=list)
    elapsed_ms: float


def github_corpus_method(
    params: GithubCorpusParams, *, cache: LRUCache | None = None,
) -> GithubCorpusResultModel:
    key, hit = _cache_lookup(cache, "data.github_corpus", params)
    if hit is not None:
        return hit

    if os.environ.get("VBGUI_DISABLE_NETWORK") == "1" and \
            params.repo_url.startswith(("http://", "https://")):
        raise RuntimeError(
            "GitHub clone disabled via VBGUI_DISABLE_NETWORK")

    from scripts.data.github_corpus import github_corpus as _gc
    r = _gc(
        repo_url=params.repo_url,
        max_commits=params.max_commits,
        max_tokens=params.max_tokens,
        tokenizer=params.tokenizer,
        use_clang=params.use_clang,
        use_treesitter=params.use_treesitter,
        job_id=params.job_id,
        out_dir=params.out_dir,
    )
    out = GithubCorpusResultModel(
        parquet_path=r.parquet_path,
        n_tokens_written=r.n_tokens_written,
        n_docs_seen=r.n_docs_seen,
        side_channels=r.side_channels,
        elapsed_ms=r.elapsed_ms,
    )
    _cache_store(cache, key, out)
    return out
