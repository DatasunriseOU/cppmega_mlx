"""V7-G04: corpus stats — coverage + length percentiles + vocab hist."""

from __future__ import annotations

from cppmega_v4.data.corpus_stats import compute_corpus_stats


def test_v7_g04_coverage_and_percentiles_on_synthetic_corpus():
    docs = [
        list(range(10)),
        list(range(20)),
        list(range(50)),
        [1, 1, 2, 3, 1],
    ]
    stats = compute_corpus_stats(docs, vocab_size=128, topk=5, hist_bin=16)
    assert stats["n_docs"] == 4
    assert stats["vocab_size"] == 128
    # Unique tokens = {0..49} ∪ {1,2,3} = 50 distinct.
    assert stats["unique_tokens_seen"] == 50
    assert 0 < stats["token_coverage_pct"] < 100
    assert stats["doc_length_p50"] in (10, 20)
    assert stats["doc_length_p99"] >= 20
    assert len(stats["vocab_topk"]) <= 5
    # Top-1 should be token id 1 (appears 3 times in last doc).
    assert stats["vocab_topk"][0]["token_id"] == 1


def test_v7_g04_long_tail_count_picks_singletons():
    docs = [[1, 1, 1, 2, 3, 4]]
    stats = compute_corpus_stats(docs, vocab_size=16, topk=4)
    # Tokens 2/3/4 each used once → long tail = 3.
    assert stats["vocab_long_tail_count"] == 3


def test_v7_g04_empty_corpus_returns_zero_shape():
    stats = compute_corpus_stats([], vocab_size=128)
    assert stats["n_docs"] == 0
    assert stats["unique_tokens_seen"] == 0
    assert stats["token_coverage_pct"] == 0.0
    assert stats["doc_length_p50"] == 0


def test_v7_g04_histogram_bins_cover_max_len():
    docs = [list(range(70)), list(range(130))]
    stats = compute_corpus_stats(docs, vocab_size=200, hist_bin=32)
    # Max len 130 → ceil(130/32)=5 bins.
    assert len(stats["doc_length_hist_counts"]) >= 4
    assert sum(stats["doc_length_hist_counts"]) == 2
