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
    """Return compact summaries for every entry in ``params.category``.

    Special category ``compatible_edges`` (V7-E-AUDIT-02): returns one
    option per (src_kind, dst_kind) pair where the shape contracts
    indicate the edge is well-typed. Each option's ``name`` is
    ``"src_kind->dst_kind"``, ``summary`` is ``dst_kind``, and
    ``paper_ref`` is ``src_kind`` so the UI can split with a single
    str.split('->').
    """
    cache_key = ("catalog.list_options", params.category)
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit  # type: ignore[return-value]

    if params.category == "compatible_edges":
        from cppmega_v4.spec.shape_contract import compatible_edges
        pairs = compatible_edges()
        result = CatalogListOptionsResult(options=[
            CatalogOptionSummary(
                name=f"{src}->{dst}", summary=dst, paper_ref=src,
            )
            for src, dst in pairs
        ])
        if cache is not None:
            cache.set(cache_key, result)
        return result

    if params.category == "feature_injectors":
        # V8-R08: enumerate the mid-canvas feature-injection options.
        # Three are rewriters (MTP / IFIM / MHC) — they mutate the
        # build spec. Two are brick kinds (engram / mlp_ngram) — they
        # land as new nodes inserted into the canvas.
        result = CatalogListOptionsResult(options=[
            CatalogOptionSummary(
                name="mtp_weighted",
                summary="Multi-token prediction K=2 head + weighted loss",
                paper_ref="rewriter:MTPRewriter"),
            CatalogOptionSummary(
                name="ifim_shaped",
                summary="Span-aware IFIM loss reshaping",
                paper_ref="rewriter:IFIMRewriter"),
            CatalogOptionSummary(
                name="mhc_attn_bias",
                summary="Multi-head co-occurrence attention bias",
                paper_ref="rewriter:MHCRewriter"),
            CatalogOptionSummary(
                name="engram",
                summary="Standalone local engram (n-gram) branch",
                paper_ref="brick:engram"),
            CatalogOptionSummary(
                name="ngram_2_3_4",
                summary="Engram with default 2,3,4-gram orders",
                paper_ref="brick:engram"),
        ])
        if cache is not None:
            cache.set(cache_key, result)
        return result

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
