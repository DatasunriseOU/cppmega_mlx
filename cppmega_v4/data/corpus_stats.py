"""V7-G04: corpus statistics — token coverage + doc length + vocab.

Computes the three metrics for a parquet shard:
  * token_coverage_pct — fraction of distinct tokens seen / vocab_size.
  * doc_length_p50/p90/p99 + histogram (bin per 64 tokens).
  * vocab_usage_topk + long_tail_count (tokens used <= 1 time).

Pure Python; designed to run as a sidecar emission step after
clang_enriched_to_parquet.py writes a shard.
"""

from __future__ import annotations

import bisect
from collections import Counter
from typing import Iterable


def compute_corpus_stats(
    token_id_lists: Iterable[list[int]],
    *,
    vocab_size: int,
    topk: int = 50,
    hist_bin: int = 64,
) -> dict:
    """Aggregate corpus stats for a shard.

    Args:
        token_id_lists: per-document iterable of token id lists.
        vocab_size: total vocab so coverage % is computable.
        topk: surface the top-k most-used tokens.
        hist_bin: doc-length histogram bin width in tokens.
    """
    seen: set[int] = set()
    doc_lengths: list[int] = []
    tok_counter: Counter[int] = Counter()
    for ids in token_id_lists:
        if not ids:
            continue
        doc_lengths.append(len(ids))
        for t in ids:
            seen.add(t)
            tok_counter[t] += 1
    n_docs = len(doc_lengths)
    if vocab_size <= 0:
        coverage_pct = 0.0
    else:
        coverage_pct = round(len(seen) / vocab_size * 100.0, 4)

    def _pct(arr: list[int], pct: float) -> int:
        if not arr:
            return 0
        s = sorted(arr)
        idx = max(0, min(len(s) - 1, int(round(pct / 100 * len(s)))))
        return int(s[idx])

    # Length histogram
    if doc_lengths:
        max_len = max(doc_lengths)
        n_bins = max(1, (max_len + hist_bin) // hist_bin)
        hist_counts = [0] * n_bins
        for L in doc_lengths:
            b = min(n_bins - 1, L // hist_bin)
            hist_counts[b] += 1
        hist_edges = [i * hist_bin for i in range(n_bins + 1)]
    else:
        hist_counts = []
        hist_edges = []

    # Vocab histogram
    top = tok_counter.most_common(topk)
    long_tail = sum(1 for v in tok_counter.values() if v <= 1)

    return {
        "n_docs": n_docs,
        "vocab_size": int(vocab_size),
        "token_coverage_pct": coverage_pct,
        "unique_tokens_seen": len(seen),
        "doc_length_p50": _pct(doc_lengths, 50),
        "doc_length_p90": _pct(doc_lengths, 90),
        "doc_length_p99": _pct(doc_lengths, 99),
        "doc_length_hist_edges": hist_edges,
        "doc_length_hist_counts": hist_counts,
        "vocab_topk": [{"token_id": int(t), "count": int(c)}
                       for t, c in top],
        "vocab_long_tail_count": long_tail,
    }


__all__ = ["compute_corpus_stats"]
