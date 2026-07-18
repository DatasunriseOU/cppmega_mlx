from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.graph_packet import EdgeIndex, GraphBatch, GraphPacket
from cppmega_mlx.data.prompt_graph import PromptProjectIndex
from cppmega_mlx.inference import generate_tokens
from cppmega_mlx.inference.generation import _model_kwargs_for_slice
from cppmega_mlx.inference.side_channels import InferenceSideChannelBuilder
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig


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


def test_repository_inference_routes_reach_real_dense_cpp_lm(
    tmp_path: Path,
) -> None:
    case = _case()
    result = InferenceSideChannelBuilder(
        _CharacterOffsetTokenizer(),
        fail_policy="error",
    ).build_repository(
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
    graph_batch = result.model_kwargs["graph_batch"]
    assert isinstance(graph_batch, GraphBatch)
    call_edge = graph_batch.graphs[0].edge("call")
    assert call_edge is not None and call_edge.num_edges > 0
    assert graph_batch.edge_kinds[0]["call"].shape == (call_edge.num_edges,)
    assert result.provenance["prompt_graph_model_route_inputs"] == (
        "graph_batch"
    )
    assert set(result.model_kwargs["document_ids"].tolist()[0]) == {1}

    mx.random.seed(37)
    model = DenseCppLM(
        DenseCppLMConfig(
            vocab_size=512,
            hidden_size=32,
            depth=1,
            ffn_hidden_size=64,
            max_seq_length=1024,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=8,
            ngram_hash_enabled=False,
            graph_routes_enabled=True,
            require_graph_routes=True,
            graph_attention_bias_beta=10.0,
        )
    )
    routed_logits = model.logits(result.prompt_ids, **result.model_kwargs)

    missing_graph_batch = dict(result.model_kwargs)
    del missing_graph_batch["graph_batch"]
    with pytest.raises(RuntimeError, match="typed GraphBatch"):
        model.logits(result.prompt_ids, **missing_graph_batch)

    graphless_kwargs = dict(result.model_kwargs)
    graphless_kwargs["graph_batch"] = _empty_graph_batch(graph_batch)
    graphless_logits = model.logits(result.prompt_ids, **graphless_kwargs)
    mx.eval(routed_logits, graphless_logits)

    assert float(mx.sum(mx.abs(routed_logits - graphless_logits)).item()) > 1e-4

    generated = generate_tokens(
        model,
        result.prompt_ids,
        max_new_tokens=2,
        temperature=0.0,
        model_kwargs=result.model_kwargs,
    )
    mx.eval(generated)
    assert generated.shape == (1, result.prompt_ids.shape[1] + 2)

    slice_start = int(graph_batch.chunk_starts[0][0].item())
    slice_end = int(graph_batch.chunk_ends[0][1].item())
    sliced = _model_kwargs_for_slice(
        result.model_kwargs,
        start=slice_start,
        tokens=result.prompt_ids[:, slice_start:slice_end],
    )
    sliced_graph = sliced["graph_batch"]
    assert isinstance(sliced_graph, GraphBatch)
    assert sliced["structure_ids"].shape == (1, slice_end - slice_start)
    assert int(mx.max(sliced_graph.chunk_ends[0]).item()) <= (
        slice_end - slice_start
    )


def _empty_graph_batch(graph_batch: GraphBatch) -> GraphBatch:
    graphs = []
    for graph in graph_batch.graphs:
        empty_edges = {
            relation: EdgeIndex.from_pairs(
                [],
                relation=relation,
                num_nodes=edge.num_nodes,
            )
            for relation, edge in graph.edges.items()
        }
        graphs.append(GraphPacket(edges=empty_edges, num_nodes=graph.num_nodes))
    return GraphBatch(
        graphs=tuple(graphs),
        chunk_starts=graph_batch.chunk_starts,
        chunk_ends=graph_batch.chunk_ends,
        chunk_kinds=graph_batch.chunk_kinds,
        chunk_dep_levels=graph_batch.chunk_dep_levels,
    )


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
