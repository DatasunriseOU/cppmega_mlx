from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cppmega_mlx.data.prompt_graph import PromptProjectIndex
from cppmega_mlx.inference.side_channels import InferenceSideChannelBuilder


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "case3_prompt_repo"


class _CharacterOffsetTokenizer:
    name_or_path = "case3-inference-repository-tokenizer"

    def encode(self, text: str):
        return [ord(char) % 251 + 1 for char in text]

    def encode_with_offsets(self, text: str):
        return self.encode(text), [
            (index, index + 1) for index in range(len(text))
        ]


def _case() -> dict[str, object]:
    return json.loads((FIXTURE / "cases.jsonl").read_text(encoding="utf-8"))


def test_repository_inference_builds_real_graph_and_preserves_overload_ids(
    tmp_path: Path,
) -> None:
    case = _case()
    builder = InferenceSideChannelBuilder(
        _CharacterOffsetTokenizer(),
        fail_policy="error",
    )

    result = builder.build_repository(
        str(case["source_prefix"]),
        repository_root=FIXTURE,
        project_id="tests/case3-prompt-repo",
        source_path="src/math_prompt.cpp",
        source_start=0,
        indexer_root=ROOT,
        index_cache_dir=tmp_path / "index-cache",
        graph_cache_dir=tmp_path / "graph-cache",
    )

    assert result.graph_artifact is not None
    assert result.graph_artifact.graph_routes["graph_call_edges"]
    assert result.provenance["prompt_graph_producer"] == (
        "ClangPromptProjectIndexProducer"
    )
    assert result.provenance["prompt_graph_project_id"] == (
        "tests/case3-prompt-repo"
    )
    assert result.provenance["prompt_graph_identity_schema"] == "v3"

    assert result.project_index is not None
    overloads = [
        symbol
        for symbol in result.project_index.symbols
        if symbol.qname == "case3_repo::repository_helper"
        and symbol.kind == "function"
    ]
    overload_ids = {symbol.symbol_id for symbol in overloads}
    assert len(overload_ids) == 2
    assert max(overload_ids) > 0xFFFFFFFF
    assert len({symbol.usr for symbol in overloads}) == 2
    assert len({symbol.canonical_signature for symbol in overloads}) == 2
    assert all(
        symbol.identity_project == "tests/case3-prompt-repo"
        and symbol.identity_file == "src/repo_helper.cpp"
        and symbol.identity_line > 0
        and symbol.identity_column > 0
        for symbol in overloads
    )

    identities = np.asarray(
        result.side_channels["semantic_graph"]["token_symbol_ids"]
    )
    assert identities.dtype == np.dtype(np.uint64)

    assert result.index_path is not None
    loaded = builder.build_repository(
        str(case["source_prefix"]),
        repository_root=FIXTURE,
        project_id="tests/case3-prompt-repo",
        source_path="src/math_prompt.cpp",
        source_start=0,
        indexer_root=ROOT,
        index_cache_dir=tmp_path / "index-cache",
        graph_cache_dir=tmp_path / "graph-cache-loaded",
        index_path=result.index_path,
    )
    assert loaded.provenance["prompt_graph_index_source"] == "loaded"
    assert loaded.project_index is not None
    assert loaded.project_index.index_sha256 == result.project_index.index_sha256


def test_repository_inference_rejects_synthetic_index_and_does_not_fallback(
    tmp_path: Path,
) -> None:
    case = _case()
    from cppmega_mlx.data.prompt_graph_index import ClangPromptProjectIndexProducer

    real = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "real-index-cache",
        indexer_root=ROOT,
    ).build(FIXTURE, project_id="tests/case3-prompt-repo")
    payload = real.index.to_dict()
    payload["provenance"]["producer"] = "synthetic-test-index"
    synthetic = PromptProjectIndex.from_dict(payload).with_integrity()
    synthetic_path = tmp_path / "synthetic-index.json"
    synthetic_path.write_text(
        json.dumps(synthetic.to_dict()),
        encoding="utf-8",
    )

    builder = InferenceSideChannelBuilder(
        _CharacterOffsetTokenizer(),
        fail_policy="error",
    )
    with pytest.raises(ValueError, match="production repository index"):
        builder.build_repository(
            str(case["source_prefix"]),
            repository_root=FIXTURE,
            project_id="tests/case3-prompt-repo",
            source_path="src/math_prompt.cpp",
            source_start=0,
            indexer_root=ROOT,
            index_cache_dir=tmp_path / "index-cache",
            graph_cache_dir=tmp_path / "graph-cache",
            index_path=synthetic_path,
        )
