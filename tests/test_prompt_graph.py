from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from cppmega_mlx.data.prompt_graph import (
    CppPromptTokenizerAdapter,
    PromptGraphArtifact,
    PromptGraphBuilder,
    PromptGraphContext,
    PromptProjectIndex,
)
from cppmega_mlx.data.prompt_graph_index import ClangPromptProjectIndexProducer
from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "case3_prompt_repo"


class _CharacterOffsetTokenizer:
    name_or_path = "case3-character-offset-tokenizer"

    def encode_with_offsets(self, text: str):
        return [ord(char) % 251 + 1 for char in text], [
            (index, index + 1) for index in range(len(text))
        ]


def _case() -> dict:
    return json.loads((FIXTURE / "cases.jsonl").read_text(encoding="utf-8"))


def _index(tmp_path: Path) -> PromptProjectIndex:
    return ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    ).build(FIXTURE, project_id="tests/case3-prompt-repo").index


def _context(index: PromptProjectIndex) -> PromptGraphContext:
    document = index.document_for_path("src/math_prompt.cpp")
    return PromptGraphContext.from_repository_prompt(
        index,
        _case()["source_prefix"],
        document_id=document.id,
        source_path=document.source_path,
        source_start=0,
        language="cpp",
    )


def test_real_repo_fixture_round_trips_through_compile_prompt(tmp_path: Path) -> None:
    case = _case()
    completion = json.loads(
        (FIXTURE / "gold_completions.jsonl").read_text(encoding="utf-8")
    )["completion"]
    source = (FIXTURE / "src" / "math_prompt.cpp").read_text(encoding="utf-8")
    index = _index(tmp_path)
    document = index.document_for_path("src/math_prompt.cpp")

    assert case["source_prefix"] + completion + case["source_suffix"] == source
    assert document.source == source
    assert case["prompt_graph_repo"] == "."
    assert case["sidecar_contract"]["prompt_graph_required"] is True


def test_builder_maps_symbol_call_type_def_use_chunk_and_domain_routes(tmp_path: Path) -> None:
    prompt = _case()["source_prefix"]
    index = _index(tmp_path)
    artifact = PromptGraphBuilder(_CharacterOffsetTokenizer()).build(
        index, _context(index)
    )

    assert artifact.token_count == len(_context(index).text)
    assert artifact.token_count > len(prompt)
    assert artifact.edge_counts["call"] > 0
    assert artifact.edge_counts["type"] > 0
    assert artifact.edge_counts["def_use"] > 0
    assert artifact.edge_counts["domain"] > 0
    assert artifact.graph_routes["graph_call_edges"]
    assert artifact.graph_routes["graph_type_edges"]
    assert artifact.graph_routes["graph_domain_edges"]
    assert any(artifact.side_channels["symbol_ids"])
    assert any(artifact.side_channels["call_targets"])
    assert any(artifact.side_channels["type_refs"])
    assert any(artifact.side_channels["def_use"])
    assert artifact.receipt["symbol_count"] > 0
    assert artifact.receipt["chunk_count"] > 0


def test_cppmega_tokenizer_offsets_are_exact_and_match_normal_encode(tmp_path: Path) -> None:
    tokenizer = load_cppmega_tokenizer(
        ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json"
    )
    prompt = _case()["source_prefix"]
    index = _index(tmp_path)
    context = _context(index)
    token_ids, offsets = tokenizer.encode_with_offsets(context.text)
    remote_ids, remote_offsets = CppPromptTokenizerAdapter(
        tokenizer._tokenizer,
        tokenizer_path=tokenizer.path,
    ).encode_with_offsets(context.text)
    artifact = PromptGraphBuilder(tokenizer).build(index, context)

    assert token_ids == tokenizer.encode(context.text)
    assert remote_ids == token_ids
    assert remote_offsets == offsets
    assert list(artifact.token_ids) == token_ids
    tokenizer_payload = json.loads(tokenizer.path.read_text(encoding="utf-8"))
    canonical_tokenizer = json.dumps(
        tokenizer_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert artifact.receipt["hashes"]["tokenizer_sha256"] == sha256(
        canonical_tokenizer.encode("utf-8")
    ).hexdigest()
    assert len(offsets) == len(token_ids)
    assert all(0 <= start < end <= len(context.text) for start, end in offsets)
    target_start = sum(len(segment.text) for segment in context.segments[:-1])
    type_char = target_start + prompt.index("Accumulator acc")
    type_token = next(
        token_index
        for token_index, (start, end) in enumerate(artifact.token_spans)
        if start <= type_char < end
    )
    start, end = artifact.token_spans[type_token]
    assert start <= type_char < end
    assert artifact.side_channels["type_refs"][type_token] > 0


def test_cache_key_is_hash_addressed_and_corruption_fails_closed(
    tmp_path: Path,
) -> None:
    builder = PromptGraphBuilder(
        _CharacterOffsetTokenizer(), cache_dir=tmp_path
    )
    index = _index(tmp_path)
    context = _context(index)
    first = builder.build(index, context)
    cache_key = builder.cache_key(index, context)
    cache_path = tmp_path / f"{cache_key}.json"

    assert cache_path.is_file()
    assert first.receipt["cache_key"] == cache_key
    assert set(first.receipt["hashes"]) == {
        "artifact_sha256",
        "context_sha256",
        "index_sha256",
        "prompt_sha256",
        "source_sha256",
        "tokenizer_sha256",
    }
    assert builder.build(index, context).to_json() == first.to_json()
    assert cache_path in list(tmp_path.glob("*.json"))

    cache_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cached prompt graph artifact"):
        builder.build(index, context)


def test_artifact_checksum_rejects_tampering(tmp_path: Path) -> None:
    index = _index(tmp_path)
    artifact = PromptGraphBuilder(_CharacterOffsetTokenizer()).build(
        index, _context(index)
    )
    payload = artifact.to_dict()
    payload["side_channels"]["symbol_ids"][0] += 1

    with pytest.raises(ValueError, match="artifact_sha256"):
        PromptGraphArtifact.from_dict(payload)

    payload = artifact.to_dict()
    payload["receipt"]["edge_counts"]["call"] = 99
    with pytest.raises(ValueError, match="artifact_sha256"):
        PromptGraphArtifact.from_dict(payload)


def test_window_remaps_routes_and_extends_generated_graph_state(tmp_path: Path) -> None:
    index = _index(tmp_path)
    artifact = PromptGraphBuilder(_CharacterOffsetTokenizer()).build(
        index, _context(index)
    )
    generated = 3
    window_start = 10
    model_inputs = artifact.model_inputs(
        total_token_count=artifact.token_count + generated,
        window_start=window_start,
        window_end=artifact.token_count + generated,
    )

    assert model_inputs.token_count == artifact.token_count + generated - window_start
    assert model_inputs.graph_routes["graph_call_edges"]
    assert model_inputs.graph_routes["graph_type_edges"]
    assert model_inputs.graph_routes["graph_domain_edges"]
    assert model_inputs.graph_routes["graph_generated_query_edges"]
    assert all(value > 0 for value in model_inputs.side_channels["structure_ids"][-generated:])
    assert model_inputs.side_channels["confidence_ids"][-generated:] == [1] * generated
    relation_bias = model_inputs.dense_relation_attention_bias()
    edge_kind_bias = model_inputs.dense_edge_kind_attention_bias()
    bias = model_inputs.dense_attention_bias()
    assert len(bias) == model_inputs.token_count
    assert all(len(row) == model_inputs.token_count for row in bias)
    assert sum(sum(row) for row in relation_bias) > 0.0
    assert sum(sum(row) for row in edge_kind_bias) > 0.0
    assert sum(sum(row) for row in bias) > 0.0
    assert bias == [
        [relation + kind for relation, kind in zip(relation_row, kind_row)]
        for relation_row, kind_row in zip(relation_bias, edge_kind_bias)
    ]
    with pytest.raises(ValueError, match="nonzero.*graph-off ablation"):
        model_inputs.dense_edge_kind_attention_bias(default_weight=0.0)


def test_index_and_builder_fail_closed_on_invalid_or_empty_graph(tmp_path: Path) -> None:
    index = _index(tmp_path)
    payload = index.to_dict()
    chunk = payload["chunks"][0]
    document = next(
        row for row in payload["documents"] if row["id"] == chunk["document_id"]
    )
    chunk["end"] = len(document["source"]) + 1
    with pytest.raises(ValueError, match="chunk.*invalid span"):
        PromptProjectIndex.from_dict(payload)

    payload = index.to_dict()
    payload["edges"] = []
    with pytest.raises(ValueError, match="no visible graph relations"):
        empty_index = PromptProjectIndex.from_dict(payload)
        PromptGraphBuilder(_CharacterOffsetTokenizer()).build(
            empty_index, _context(empty_index)
        )
