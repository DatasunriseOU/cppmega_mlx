from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from cppmega_mlx.data import prompt_graph_index as prompt_graph_index_module
from cppmega_mlx.data.prompt_graph import (
    PromptGraphBuilder,
    PromptGraphContext,
    PromptGraphSegment,
    PromptProjectIndex,
)
from cppmega_mlx.data.prompt_graph_index import (
    ClangPromptProjectIndexProducer,
)
from cppmega_mlx.data.symbol_identity import compute_symbol_id


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
    ).build(FIXTURE, project_id="tests/case3-prompt-repo")
    index = result.index
    case = _case()
    source = index.document_for_path("src/math_prompt.cpp")

    artifact = PromptGraphBuilder(
        _CharacterOffsetTokenizer(), cache_dir=tmp_path / "graph-cache"
    ).build(
        index,
        PromptGraphContext.from_repository_prompt(
            index,
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
    assert result.receipt["hashes"]["compile_args_sha256"]
    assert result.receipt["hashes"]["dependency_closure_sha256"]
    assert result.receipt["hashes"]["libclang_version_sha256"]
    assert result.receipt["toolchain"]["libclang_version"]
    dependency_paths = {
        segment.source_path
        for segment in PromptGraphContext.from_repository_prompt(
            index,
            str(case["source_prefix"]),
            document_id=source.id,
            source_path=source.source_path,
            source_start=0,
        ).segments
        if segment.role == "dependency"
    }
    assert "include/repo_api.hpp" in dependency_paths
    assert "src/repo_helper.cpp" in dependency_paths


def test_real_producer_preserves_overloads_and_cross_document_calls(tmp_path: Path) -> None:
    index = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    ).build(FIXTURE, project_id="tests/case3-prompt-repo").index
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
    assert all(
        index.document_for_id(symbol.document_id).source[
            symbol.start : symbol.end
        ]
        == "repository_helper"
        for symbol in overloads
    )

    constructors = [
        symbol
        for symbol in index.symbols
        if symbol.kind == "function"
        and symbol.qname == "case3_repo::Accumulator::Accumulator"
    ]
    assert len(constructors) == 1
    constructor = constructors[0]
    constructor_source = index.document_for_id(constructor.document_id).source
    assert constructor_source[constructor.start : constructor.end] == "Accumulator"
    qualifier_start = constructor_source.index("Accumulator")
    assert constructor.start == constructor_source.index(
        "Accumulator", qualifier_start + 1
    )

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


def test_real_producer_emits_integrity_bound_v3_production_index(
    tmp_path: Path,
) -> None:
    project_id = "tests/case3-production"
    result = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    ).build(FIXTURE, project_id=project_id)

    result.index.validate_production_repository_index(
        expected_project_id=project_id,
        expected_indexer_root=ROOT,
    )
    assert result.receipt["strict_diagnostics"] is True
    assert result.receipt["index_payload_sha256"] == result.index.payload_sha256
    assert result.receipt["symbol_identity_schema_version"] == 3
    assert result.receipt["identity_adapters"] == [
        "case4_symbol_reference_for_cursor_v3"
    ]
    assert max(symbol.symbol_id for symbol in result.index.symbols) > 0xFFFFFFFF


def test_production_index_rejects_raw_identity_adapter_provenance(
    tmp_path: Path,
) -> None:
    project_id = "tests/case3-native-identity"
    result = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    ).build(FIXTURE, project_id=project_id)
    payload = result.index.to_dict()
    payload["provenance"]["identity_adapters"] = [
        "raw_clang_usr_signature_v3_adapter"
    ]
    tampered = PromptProjectIndex.from_dict(payload).with_integrity()

    with pytest.raises(ValueError, match="untrusted identity adapter"):
        tampered.validate_production_repository_index(
            expected_project_id=project_id,
            expected_indexer_root=ROOT,
        )


def test_prompt_producer_requires_native_case4_v3_identity_helper(
    tmp_path: Path,
) -> None:
    class _CursorKind:
        name = "FUNCTION_DECL"

    cursor = SimpleNamespace(
        kind=_CursorKind(),
        spelling="route",
        location=SimpleNamespace(
            file=SimpleNamespace(name=str(tmp_path / "repo" / "route.cpp")),
            line=4,
            column=5,
        ),
    )
    indexer = SimpleNamespace(get_qualified_name=lambda _cursor: "api::route")

    with pytest.raises(RuntimeError, match="native CASE 4 v3"):
        prompt_graph_index_module._identity_for_cursor(
            indexer,
            cursor,
            repo_root=tmp_path / "repo",
            project_id="tests/native-identity",
            source_path="route.cpp",
        )


def test_prompt_producer_rejects_identity_without_provenance(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    source_file = repo_root / "route.cpp"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("int route() { return 1; }\n", encoding="utf-8")
    symbol_key = (
        "usr:schema=v3\x1fproject=tests/native-identity\x1f"
        "usr=c:@F@route#"
    )

    class _CursorKind:
        name = "FUNCTION_DECL"

    cursor = SimpleNamespace(
        kind=_CursorKind(),
        spelling="route",
        location=SimpleNamespace(
            file=SimpleNamespace(name=str(source_file)),
            line=1,
            column=5,
        ),
    )

    class _Indexer:
        @staticmethod
        def symbol_reference_for_cursor(_cursor, **_kwargs):
            return {
                "symbol_identity_schema_version": 3,
                "symbol_key": symbol_key,
                "symbol_id": compute_symbol_id(symbol_key),
                "qname": "api::route",
                "usr": "c:@F@route#",
                "canonical_signature": "display=route()|type=int ()",
                "symbol_kind": "FUNCTION_DECL",
            }

    with pytest.raises(ValueError, match="provenance"):
        prompt_graph_index_module._identity_for_cursor(
            _Indexer,
            cursor,
            repo_root=repo_root,
            project_id="tests/native-identity",
            source_path="route.cpp",
        )


def test_cached_index_payload_checksum_rejects_tampering(tmp_path: Path) -> None:
    producer = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    )
    result = producer.build(FIXTURE, project_id="tests/case3-payload-integrity")
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    payload["symbols"][0]["qname"] += "_tampered"
    result.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="payload integrity mismatch"):
        producer.build(FIXTURE, project_id="tests/case3-payload-integrity")


def test_production_index_rejects_non_checkout_indexer_provenance(
    tmp_path: Path,
) -> None:
    project_id = "tests/case3-indexer-provenance"
    result = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    ).build(FIXTURE, project_id=project_id)
    payload = result.index.to_dict()
    wrong_indexer = ROOT / "cppmega_mlx" / "data" / "prompt_graph.py"
    payload["provenance"]["indexer_path"] = str(wrong_indexer)
    payload["provenance"]["hashes"]["indexer_sha256"] = sha256(
        wrong_indexer.read_bytes()
    ).hexdigest()
    tampered = PromptProjectIndex.from_dict(payload).with_integrity()

    with pytest.raises(ValueError, match="same-checkout indexer"):
        tampered.validate_production_repository_index(
            expected_project_id=project_id,
            expected_indexer_root=ROOT,
        )


def test_production_index_rejects_usr_signature_contract_tampering(
    tmp_path: Path,
) -> None:
    project_id = "tests/case3-semantic-contract"
    result = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    ).build(FIXTURE, project_id=project_id)
    payload = result.index.to_dict()
    definition = next(
        row
        for row in payload["symbols"]
        if row["kind"] == "function"
        and row["qname"] == "case3_repo::repository_helper"
        and "args=(int)" in row["canonical_signature"]
    )
    occurrence = next(
        row
        for row in payload["symbols"]
        if row["identity"] != definition["identity"]
        and row["semantic_identity"] == definition["semantic_identity"]
    )
    occurrence["canonical_signature"] += "|tampered=true"
    tampered = PromptProjectIndex.from_dict(payload).with_integrity()

    with pytest.raises(ValueError, match="semantic identity contract"):
        tampered.validate_production_repository_index(
            expected_project_id=project_id,
            expected_indexer_root=ROOT,
        )


def test_tampered_cached_overload_edge_fails_closed(tmp_path: Path) -> None:
    project_id = "tests/case3-cache-integrity"
    producer = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    )
    result = producer.build(FIXTURE, project_id=project_id)
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    helper_definitions = [
        row
        for row in payload["symbols"]
        if row["kind"] == "function"
        and row["qname"] == "case3_repo::repository_helper"
    ]
    int_definition = next(
        row for row in helper_definitions if "args=(int)" in row["canonical_signature"]
    )
    double_definition = next(
        row
        for row in helper_definitions
        if "args=(double)" in row["canonical_signature"]
    )
    edge = next(
        row
        for row in payload["edges"]
        if row["relation"] == "call" and row["target"] == int_definition["identity"]
    )
    edge["target"] = double_definition["identity"]
    tampered_index = PromptProjectIndex.from_dict(payload).with_integrity()
    result.path.write_text(
        json.dumps(tampered_index.to_dict()) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="edge changes semantic identity"):
        producer.build(FIXTURE, project_id=project_id)


def test_cached_graph_edge_target_must_be_definition_owned(tmp_path: Path) -> None:
    project_id = "tests/case3-definition-target"
    producer = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    )
    result = producer.build(FIXTURE, project_id=project_id)
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    definition = next(
        row
        for row in payload["symbols"]
        if row["kind"] == "function"
        and row["qname"] == "case3_repo::repository_helper"
        and "args=(int)" in row["canonical_signature"]
    )
    occurrence = next(
        row
        for row in payload["symbols"]
        if row["semantic_identity"] == definition["semantic_identity"]
        and row["kind"] in {"callsite", "use"}
    )
    edge = next(
        row
        for row in payload["edges"]
        if row["relation"] == "call" and row["target"] == definition["identity"]
    )
    edge["target"] = occurrence["identity"]
    tampered_index = PromptProjectIndex.from_dict(payload).with_integrity()
    result.path.write_text(
        json.dumps(tampered_index.to_dict()) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="definition-owned target"):
        producer.build(FIXTURE, project_id=project_id)


def test_index_cache_key_tracks_repository_and_indexer_freshness(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE, repo)
    producer = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    )

    first = producer.build(repo, project_id="tests/case3-cache")
    second = producer.build(repo, project_id="tests/case3-cache")
    assert second.cache_hit is True
    assert second.path == first.path
    assert second.receipt["cache_key"] == first.receipt["cache_key"]

    helper = repo / "src" / "repo_helper.cpp"
    helper.write_text(
        helper.read_text(encoding="utf-8").replace(
            "value < 0 ? 0 : value",
            "value < 1 ? 0 : value",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="repository freshness"):
        first.index.verify_repository(repo)
    changed = producer.build(repo, project_id="tests/case3-cache")
    assert changed.cache_hit is False
    assert changed.path != first.path
    assert changed.receipt["hashes"]["repository_sha256"] != first.receipt["hashes"][
        "repository_sha256"
    ]
    assert changed.receipt["hashes"]["dependency_closure_sha256"] != first.receipt[
        "hashes"
    ]["dependency_closure_sha256"]


def test_index_cache_key_tracks_resolved_compile_arguments(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE, repo)
    compile_commands = repo / "compile_commands.json"

    def write_commands(define: str) -> None:
        rows = []
        for relative in (
            "src/math_prompt.cpp",
            "src/repo_helper.cpp",
            "src/repo_caller.cpp",
        ):
            source = repo / relative
            rows.append(
                {
                    "directory": str(repo),
                    "file": str(source),
                    "arguments": [
                        "clang++",
                        "-std=c++20",
                        f"-I{repo / 'include'}",
                        define,
                        "-c",
                        str(source),
                    ],
                }
            )
        compile_commands.write_text(json.dumps(rows), encoding="utf-8")

    producer = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    )
    write_commands("-DCASE3_CACHE_VERSION=1")
    first = producer.build(repo, project_id="tests/case3-compile-args")
    write_commands("-DCASE3_CACHE_VERSION=2")
    second = producer.build(repo, project_id="tests/case3-compile-args")

    assert first.path != second.path
    assert first.receipt["hashes"]["compile_args_sha256"] != second.receipt[
        "hashes"
    ]["compile_args_sha256"]
    assert any(
        "-DCASE3_CACHE_VERSION=2" in args
        for args in second.receipt["toolchain"]["compile_args_by_file"].values()
    )


def test_producer_dependency_is_explicit() -> None:
    with pytest.raises(ValueError, match="requires explicit indexer_root"):
        ClangPromptProjectIndexProducer(cache_dir="unused")


def test_strict_producer_rejects_clang_parse_errors(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE, repo)
    broken = repo / "src" / "broken.cpp"
    broken.write_text("int broken( {\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="strict clang.*rejected.*broken.cpp"):
        ClangPromptProjectIndexProducer(
            cache_dir=tmp_path / "index-cache",
            indexer_root=ROOT,
            strict_diagnostics=True,
        ).build(repo, project_id="tests/case3-strict-error")


def test_corrupt_producer_cache_fails_closed(tmp_path: Path) -> None:
    producer = ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    )
    result = producer.build(FIXTURE, project_id="tests/case3-corrupt")
    result.path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cached prompt project index"):
        producer.build(FIXTURE, project_id="tests/case3-corrupt")
