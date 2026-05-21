"""JSON-RPC handlers for the explanation catalogue (E7-10)."""

from __future__ import annotations

from cppmega_v4.explain import CATALOG, get_entry, list_options
from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.schema import (
    CatalogExplainEntryPayload,
    CatalogExplainParams,
    CatalogExplainResult,
    CatalogListOptionsParams,
    CatalogListOptionsResult,
    CatalogOptionSummary,
)


def catalog_explain(
    params: CatalogExplainParams,
    *,
    cache: LRUCache | None = None,
) -> CatalogExplainResult:
    """Return the ExplainEntry for ``(category, name)`` or a stub
    message when nothing is registered. Cacheable (LRU keyed by the
    pair)."""
    cache_key = ("catalog.explain", params.category, params.name)
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit  # type: ignore[return-value]

    entry = get_entry(params.category, params.name)
    if entry is None:
        result = CatalogExplainResult(
            entry=None,
            not_found_message=(
                f"no entry for category={params.category!r}, "
                f"name={params.name!r}; available categories: "
                f"{sorted({c for c, _ in CATALOG})}"
            ),
        )
    else:
        result = CatalogExplainResult(
            entry=CatalogExplainEntryPayload(
                category=entry.category,
                name=entry.name,
                summary=entry.summary,
                when_to_use=entry.when_to_use,
                when_to_avoid=entry.when_to_avoid,
                recommended_params=dict(entry.recommended_params),
                paper_ref=entry.paper_ref,
                paper_url=entry.paper_url,
                gotchas=list(entry.gotchas),
            ),
        )

    if cache is not None:
        cache.set(cache_key, result)
    return result


def catalog_list_options(
    params: CatalogListOptionsParams,
    *,
    cache: LRUCache | None = None,
) -> CatalogListOptionsResult:
    """Return compact summaries for every entry in ``params.category``."""
    cache_key = ("catalog.list_options", params.category)
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit  # type: ignore[return-value]

    entries = list_options(params.category)
    result = CatalogListOptionsResult(options=[
        CatalogOptionSummary(
            name=e.name,
            summary=e.summary,
            paper_ref=e.paper_ref,
        )
        for e in entries
    ])
    if cache is not None:
        cache.set(cache_key, result)
    return result
