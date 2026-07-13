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
    TOKEN_SIDECAR_DEFAULTS,
)
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


def _index() -> PromptProjectIndex:
    return PromptProjectIndex.from_json_path(FIXTURE / "project_index.json")


def _context() -> PromptGraphContext:
    return PromptGraphContext.from_prompt(
        _case()["source_prefix"], source_start=0, language="cpp"
    )


def test_real_repo_fixture_round_trips_through_compile_prompt() -> None:
    case = _case()
    completion = json.loads(
        (FIXTURE / "completions.jsonl").read_text(encoding="utf-8")
    )["completion"]
    source = (FIXTURE / "src" / "math_prompt.cpp").read_text(encoding="utf-8")
    index = _index()

    assert case["source_prefix"] + completion + case["source_suffix"] == source
    assert index.source == source
    assert case["prompt_graph_index"] == "project_index.json"
    assert case["sidecar_contract"]["prompt_graph_required"] is True


def test_builder_maps_symbol_call_type_def_use_chunk_and_domain_routes() -> None:
    prompt = _case()["source_prefix"]
    artifact = PromptGraphBuilder(_CharacterOffsetTokenizer()).build(
        _index(), _context()
    )

    assert artifact.token_count == len(prompt)
    assert artifact.edge_counts == {
        "build": 0,
        "call": 1,
        "cross_domain": 0,
        "def_use": 1,
        "diagnostic": 0,
        "domain": 1,
        "shell": 0,
        "type": 1,
    }
    assert artifact.graph_routes["graph_call_edges"] == [[2, 1]]
    assert artifact.graph_routes["graph_type_edges"] == [[3, 0]]
    assert artifact.graph_routes["graph_domain_edges"] == [
        [
            artifact.first_token_for_identity("call:warmup->clamp_to_zero"),
            artifact.first_token_for_identity("tiny::clamp_to_zero"),
            2,
        ]
    ]
    assert artifact.side_channels["symbol_ids"][prompt.index("warmup")] == 30
    assert artifact.side_channels["call_targets"][
        prompt.index("clamp_to_zero(x)")
    ] == 20
    assert artifact.side_channels["type_refs"][
        prompt.index("Accumulator acc")
    ] == 10
    assert artifact.side_channels["def_use"][prompt.index("{x}") + 1] == 42
    assert artifact.receipt["symbol_count"] == 8
    assert artifact.receipt["chunk_count"] == 4


def test_cppmega_tokenizer_offsets_are_exact_and_match_normal_encode() -> None:
    tokenizer = load_cppmega_tokenizer(
        ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json"
    )
    prompt = _case()["source_prefix"]

    token_ids, offsets = tokenizer.encode_with_offsets(prompt)
    remote_ids, remote_offsets = CppPromptTokenizerAdapter(
        tokenizer._tokenizer,
        tokenizer_path=tokenizer.path,
    ).encode_with_offsets(prompt)
    artifact = PromptGraphBuilder(tokenizer).build(_index(), _context())

    assert token_ids == tokenizer.encode(prompt)
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
    assert all(0 <= start < end <= len(prompt) for start, end in offsets)
    type_token = artifact.first_token_for_identity(
        "type-use:add_one_checked:Accumulator"
    )
    start, end = artifact.token_spans[type_token]
    assert start <= prompt.index("Accumulator acc") < end


def test_cache_key_is_hash_addressed_and_corruption_fails_closed(
    tmp_path: Path,
) -> None:
    builder = PromptGraphBuilder(
        _CharacterOffsetTokenizer(), cache_dir=tmp_path
    )
    first = builder.build(_index(), _context())
    cache_key = builder.cache_key(_index(), _context())
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
    assert builder.build(_index(), _context()).to_json() == first.to_json()
    assert list(tmp_path.glob("*.json")) == [cache_path]

    cache_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cached prompt graph artifact"):
        builder.build(_index(), _context())


def test_artifact_checksum_rejects_tampering() -> None:
    artifact = PromptGraphBuilder(_CharacterOffsetTokenizer()).build(
        _index(), _context()
    )
    payload = artifact.to_dict()
    payload["side_channels"]["symbol_ids"][0] += 1

    with pytest.raises(ValueError, match="artifact_sha256"):
        PromptGraphArtifact.from_dict(payload)

    payload = artifact.to_dict()
    payload["receipt"]["edge_counts"]["call"] = 99
    with pytest.raises(ValueError, match="artifact_sha256"):
        PromptGraphArtifact.from_dict(payload)


def test_window_remaps_routes_and_zero_extends_generated_tokens() -> None:
    artifact = PromptGraphBuilder(_CharacterOffsetTokenizer()).build(
        _index(), _context()
    )
    generated = 3
    window_start = 10
    model_inputs = artifact.model_inputs(
        total_token_count=artifact.token_count + generated,
        window_start=window_start,
        window_end=artifact.token_count + generated,
    )

    assert model_inputs.token_count == artifact.token_count + generated - window_start
    assert model_inputs.graph_routes["graph_call_edges"] == [[2, 1]]
    assert model_inputs.graph_routes["graph_type_edges"] == [[3, 0]]
    assert model_inputs.graph_routes["graph_domain_edges"][0][:2] == [
        artifact.first_token_for_identity("call:warmup->clamp_to_zero")
        - window_start,
        artifact.first_token_for_identity("tiny::clamp_to_zero") - window_start,
    ]
    assert all(
        values[-generated:] == [TOKEN_SIDECAR_DEFAULTS[name]] * generated
        for name, values in model_inputs.side_channels.items()
    )
    bias = model_inputs.dense_attention_bias()
    assert len(bias) == model_inputs.token_count
    assert all(len(row) == model_inputs.token_count for row in bias)
    assert sum(sum(row) for row in bias) > 0.0


def test_index_and_builder_fail_closed_on_invalid_or_empty_graph() -> None:
    payload = json.loads((FIXTURE / "project_index.json").read_text(encoding="utf-8"))
    payload["chunks"][0]["end"] = len(payload["source"]) + 1
    with pytest.raises(ValueError, match="chunk.*invalid span"):
        PromptProjectIndex.from_dict(payload)

    payload = json.loads((FIXTURE / "project_index.json").read_text(encoding="utf-8"))
    payload["edges"] = []
    with pytest.raises(ValueError, match="no visible graph relations"):
        PromptGraphBuilder(_CharacterOffsetTokenizer()).build(
            PromptProjectIndex.from_dict(payload), _context()
        )
