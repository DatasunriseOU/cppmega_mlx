from __future__ import annotations

import pytest

from cppmega_mlx.data.domain_schema import DomainEdgeKind, DomainKind
from cppmega_mlx.data.nanochat_pipeline import tokenized_enriched_schema as schema
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched import (
    _remap_char_edge_triples_to_tokens,
    materialize_tokenized_enriched_batch,
)
from scripts.nanochat_data import clang_enriched_to_parquet


_GRAPH_FAMILY_FAILURE_REPOS = (
    "ProcMon-for-Linux",
    "apple-dyld",
)
_POINT_ANCHOR_FAILURE_REPOS = (
    "SPTAG",
    "apple-mlx",
)


class _Encoding:
    def __init__(self, text: str) -> None:
        self.ids = [1000 + (index % 1000) for index in range(len(text))]
        self.offsets = [(index, index + 1) for index in range(len(text))]


class _CharOffsetBackend:
    @staticmethod
    def encode_batch(texts, add_special_tokens=False):
        del add_special_tokens
        return [_Encoding(text) for text in texts]


class _CharOffsetTokenizer:
    _tokenizer = _CharOffsetBackend()

    @staticmethod
    def get_bos_token_id() -> int:
        return 1


@pytest.mark.parametrize("repo", _GRAPH_FAMILY_FAILURE_REPOS)
def test_case5_v4_routes_legacy_cross_domain_mirror_to_its_typed_column(
    repo: str,
) -> None:
    text = "hostcallsembeddedsql"
    cross_domain_edge = {
        "from_char": 0,
        "to_char": text.index("embedded"),
        "kind": int(DomainEdgeKind.EMBEDDED_DOMAIN),
    }
    row = {
        "repo": repo,
        "filepath": "fixture.cc",
        "symbol_identity_schema_version": 3,
        "symbol_identities": [],
        "text": text,
        "domain_edges": [cross_domain_edge],
        "cross_domain_edges": [dict(cross_domain_edge)],
    }

    tokenized = materialize_tokenized_enriched_batch(
        [row],
        _CharOffsetTokenizer(),
        num_threads=1,
    )[0]
    table = clang_enriched_to_parquet.rows_to_table(
        [row],
        tokenized_rows=[tokenized],
    )

    assert table.column("repo").to_pylist() == [repo]
    assert table.column("domain_edges").to_pylist() == [[]]
    assert table.column("cross_domain_edges").to_pylist() == [[cross_domain_edge]]
    assert tokenized[schema.TOKEN_DOMAIN_EDGES_COLUMN] == []
    assert tokenized[schema.TOKEN_CROSS_DOMAIN_EDGES_COLUMN] == [
        {
            "from": 1,
            "to": text.index("embedded") + 1,
            "kind": int(DomainEdgeKind.EMBEDDED_DOMAIN),
        }
    ]
    assert all(value > 0 for value in tokenized[schema.TOKEN_SOURCE_DOC_IDS_COLUMN])
    assert all(
        value > 0 for value in tokenized[schema.TOKEN_SOURCE_IDENTITY_IDS_COLUMN]
    )


@pytest.mark.parametrize("repo", _POINT_ANCHOR_FAILURE_REPOS)
def test_case5_v4_materializes_eof_self_point_anchor_without_clamping(
    repo: str,
) -> None:
    point = 1966
    text = "x" * point
    edge = {
        "from_char": point,
        "to_char": point,
        "kind": int(DomainEdgeKind.CALL),
    }

    tokenized = materialize_tokenized_enriched_batch(
        [
            {
                "repo": repo,
                "filepath": "fixture.cc",
                "text": text,
                "domain_kind": int(DomainKind.CPP),
                "domain_edges": [edge],
            }
        ],
        _CharOffsetTokenizer(),
        num_threads=1,
    )[0]

    assert tokenized[schema.TOKEN_DOMAIN_EDGES_COLUMN] == [
        {
            "from": point + 1,
            "to": point + 1,
            "kind": int(DomainEdgeKind.CALL),
        }
    ]


def test_self_point_anchor_uses_right_token_in_an_uncovered_gap() -> None:
    assert _remap_char_edge_triples_to_tokens(
        [
            {
                "from_char": 2,
                "to_char": 2,
                "kind": int(DomainEdgeKind.CALL),
            }
        ],
        [(0, 0), (0, 2), (4, 6)],
        family="domain",
        source_length=6,
    ) == [{"from": 2, "to": 2, "kind": int(DomainEdgeKind.CALL)}]


def test_self_point_anchor_survives_exact_character_filtering() -> None:
    edge = {
        "from_char": 3,
        "to_char": 3,
        "kind": int(DomainEdgeKind.CALL),
    }
    assert clang_enriched_to_parquet._remap_char_edge_triples(
        [edge],
        [0, 1, 4, 5],
        7,
        family="domain",
        source_length=6,
    ) == [{"from_char": 9, "to_char": 9, "kind": int(DomainEdgeKind.CALL)}]


def test_self_point_anchor_outside_closed_source_range_fails_loudly() -> None:
    with pytest.raises(ValueError, match="outside source point bounds"):
        _remap_char_edge_triples_to_tokens(
            [
                {
                    "from_char": 7,
                    "to_char": 7,
                    "kind": int(DomainEdgeKind.CALL),
                }
            ],
            [(0, 0), (0, 2), (4, 6)],
            family="domain",
            source_length=6,
        )
