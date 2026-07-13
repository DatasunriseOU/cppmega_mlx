from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cppmega_mlx.data.prompt_graph import (
    PromptGraphBuilder,
    PromptGraphContext,
    PromptGraphSegment,
)
from cppmega_mlx.data.prompt_graph_index import ClangPromptProjectIndexProducer


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "case3_prompt_repo"


class _CharacterOffsetTokenizer:
    name_or_path = "case3-producer-character-tokenizer"

    def encode_with_offsets(self, text: str):
        return [ord(char) % 251 + 1 for char in text], [
            (index, index + 1) for index in range(len(text))
        ]


def _case() -> dict[str, object]:
    return json.loads((FIXTURE / "cases.jsonl").read_text(encoding="utf-8"))


def test_real_clang_producer_builds_fixture_index_and_prompt_graph(tmp_path: Path) -> None:
    assert not (FIXTURE / "project_index.json").exists()
    result = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    ).build(FIXTURE, project_id="case3_prompt_repo")
    index = result.index
    case = _case()
    source = index.document_for_path("src/math_prompt.cpp")

    artifact = PromptGraphBuilder(
        _CharacterOffsetTokenizer(), cache_dir=tmp_path / "graph-cache"
    ).build(
        index,
        PromptGraphContext.from_prompt(
            str(case["source_prefix"]),
            document_id=source.id,
            source_path=source.source_path,
            source_start=0,
        ),
    )

    assert result.path.is_file()
    assert result.cache_hit is False
    assert len(index.documents) >= 4
    assert artifact.edge_counts["call"] > 0
    assert artifact.edge_counts["type"] > 0
    assert artifact.edge_counts["def_use"] > 0
    assert artifact.edge_counts["domain"] > 0
    assert artifact.receipt["document_count"] == len(index.documents)
    assert all(symbol.document_id > 0 and symbol.source_path for symbol in index.symbols)
    definitions = [
        symbol
        for symbol in index.symbols
        if symbol.kind in {"function", "type"}
    ]
    assert definitions
    assert all(symbol.usr for symbol in definitions)
    assert all(symbol.canonical_signature for symbol in definitions)
    assert result.receipt["edge_counts"]["call"] > 0


def test_real_producer_preserves_overloads_and_cross_document_calls(tmp_path: Path) -> None:
    index = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    ).build(FIXTURE, project_id="case3_prompt_repo").index
    overloads = [
        symbol
        for symbol in index.symbols
        if symbol.kind == "function"
        and symbol.qname == "case3_repo::repository_helper"
    ]

    assert len(overloads) == 2
    assert len({symbol.usr for symbol in overloads}) == 2
    assert len({symbol.canonical_signature for symbol in overloads}) == 2
    assert len({symbol.semantic_identity for symbol in overloads}) == 2

    helper = index.document_for_path("src/repo_helper.cpp")
    caller = index.document_for_path("src/repo_caller.cpp")
    cross_document_calls = [
        edge
        for edge in index.edges
        if edge.relation == "call"
        and index.symbol_for_identity(edge.source).document_id == caller.id
        and index.symbol_for_identity(edge.target).document_id == helper.id
    ]
    assert len(cross_document_calls) == 1

    artifact = PromptGraphBuilder(_CharacterOffsetTokenizer()).build(
        index,
        PromptGraphContext(
            segments=(
                PromptGraphSegment(
                    helper.source,
                    document_id=helper.id,
                    source_path=helper.source_path,
                    source_start=0,
                ),
                PromptGraphSegment("\n", role="separator"),
                PromptGraphSegment(
                    caller.source,
                    document_id=caller.id,
                    source_path=caller.source_path,
                    source_start=0,
                ),
            )
        ),
    )

    assert artifact.graph_routes["graph_call_edges"]
    assert {value for value in artifact.side_channels["source_doc_ids"] if value} == {
        helper.id,
        caller.id,
    }


def test_index_cache_key_tracks_repository_and_indexer_freshness(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE, repo)
    producer = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    )

    first = producer.build(repo, project_id="case3-cache")
    second = producer.build(repo, project_id="case3-cache")
    assert second.cache_hit is True
    assert second.path == first.path
    assert second.receipt["cache_key"] == first.receipt["cache_key"]

    helper = repo / "src" / "repo_helper.cpp"
    helper.write_text(
        helper.read_text(encoding="utf-8").replace("value + 2", "value + 3"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="repository freshness"):
        first.index.verify_repository(repo)
    changed = producer.build(repo, project_id="case3-cache")
    assert changed.cache_hit is False
    assert changed.path != first.path
    assert changed.receipt["hashes"]["repository_sha256"] != first.receipt["hashes"][
        "repository_sha256"
    ]


def test_corrupt_producer_cache_fails_closed(tmp_path: Path) -> None:
    producer = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    )
    result = producer.build(FIXTURE, project_id="case3-corrupt")
    result.path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cached prompt project index"):
        producer.build(FIXTURE, project_id="case3-corrupt")
