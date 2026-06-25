from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow as pa  # type: ignore[import-not-found]
import pyarrow.parquet as pq  # type: ignore[import-not-found]
import pytest

from cppmega_mlx.data.parquet_dataset import (
    MultiShardTokenParquetDataset,
    TokenParquetDataset,
)
from cppmega_mlx.data.packing import document_boundary_mask
from cppmega_mlx.data.nanochat_pipeline import platform_vocab
from cppmega_mlx.data.nanochat_pipeline import tokenized_enriched_schema as schema
from cppmega_mlx.data.nanochat_pipeline.build_context import (
    detect_build_context,
    find_compile_commands_file,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched import (
    materialize_tokenized_enriched_batch,
)
from cppmega_v4.data.doc_id_assignment import (
    assign_sharded_doc_ids,
    write_doc_id_manifest,
)
from scripts.nanochat_data import clang_enriched_to_parquet
from scripts.nanochat_data.pack_enriched_rows import (
    DOC_IDS_COLUMN,
    INPUT_IDS_COLUMN,
    read_tokenized_documents,
    pack_documents,
)
from scripts.nanochat_data.token_budget import chunk_enriched_document, load_tokenizer
from scripts.nanochat_data.token_budget import tokenizer_fingerprint
from tools.clang_indexer import index_project
from tools.clang_indexer.index_project import FunctionDef, PartInfo
from tools.clang_indexer.process_commits import (
    BuildContextResolver,
    FileAnalysis,
    _build_enriched_from_parts,
    analyze_file_clang,
)


class _CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]


def _write_token_parquet(
    path: Path,
    rows: list[list[int]],
    *,
    tokenizer_fingerprint: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"token_ids": rows}
    if tokenizer_fingerprint is not None:
        data["tokenizer_fingerprint"] = [tokenizer_fingerprint] * len(rows)
    pq.write_table(pa.table(data), path)
    return path


def _dataset_token_rows(dataset: TokenParquetDataset | MultiShardTokenParquetDataset):
    return [np.array(batch.tokens).tolist() for batch in dataset.iter_batches(loop=False)]


def _load_nanochat_module(relative_path: str) -> ModuleType:
    source = Path("/Users/dave/sources/nanochat") / relative_path
    if not source.exists():
        pytest.skip(f"nanochat source file is not available: {source}")
    spec = importlib.util.spec_from_file_location(
        f"_nanochat_{source.stem}",
        source,
    )
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot import nanochat source file: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_platform_vocab_matches_nanochat_source_of_truth() -> None:
    nanochat_vocab = _load_nanochat_module("nanochat/platform_vocab.py")

    assert platform_vocab.PLATFORM_VOCAB == nanochat_vocab.PLATFORM_VOCAB
    assert platform_vocab.PLATFORM_VOCAB_SIZE == nanochat_vocab.PLATFORM_VOCAB_SIZE
    assert platform_vocab.MAX_PLATFORM_IDS == nanochat_vocab.MAX_PLATFORM_IDS


def test_local_tokenized_schema_keeps_full_nanochat_column_contract() -> None:
    nanochat_schema = _load_nanochat_module("nanochat/tokenized_enriched_schema.py")

    assert schema.TOKENIZED_ENRICHED_COLUMNS == nanochat_schema.TOKENIZED_ENRICHED_COLUMNS


def test_clang_enriched_parquet_schema_preserves_token_semantic_columns() -> None:
    required = {
        "ast_depth",
        "sibling_index",
        "ast_node_type",
        "symbol_ids",
        "call_targets",
        "type_refs",
        "def_use",
        schema.TOKEN_SYMBOL_IDS_COLUMN,
        schema.TOKEN_CALL_TARGETS_COLUMN,
        schema.TOKEN_TYPE_REFS_COLUMN,
        schema.TOKEN_DEF_USE_COLUMN,
    }

    assert required <= set(clang_enriched_to_parquet._SCHEMA.names)


def test_clang_enriched_docs_to_table_carries_token_semantic_columns() -> None:
    rows = [
        {
            "text": "int main() { return f(); }",
            "source_doc_id": "demo.cc@main",
            "actual_token_count": 4,
            "structure_ids": [3, 3, 3, 3],
            "chunk_boundaries": [{"start": 0, "end": 24, "kind": 3, "dep_level": 0}],
            "call_edges": [],
            "type_edges": [],
            "ast_depth": [0, 1, 2, 1],
            "sibling_index": [0, 0, 1, 2],
            "ast_node_type": [1, 2, 3, 4],
            "symbol_ids": [0, 11, 11, 0],
            "call_targets": [0, 22, 0, 0],
            "type_refs": [0, 0, 33, 0],
            "def_use": [0, 1, 2, 0],
        }
    ]
    tokenized_rows = [
        {
            schema.TOKEN_IDS_COLUMN: [1, 2, 3, 4],
            schema.TOKEN_SYMBOL_IDS_COLUMN: [0, 11, 11, 0],
            schema.TOKEN_CALL_TARGETS_COLUMN: [0, 22, 0, 0],
            schema.TOKEN_TYPE_REFS_COLUMN: [0, 0, 33, 0],
            schema.TOKEN_DEF_USE_COLUMN: [0, 1, 2, 0],
        }
    ]

    table = clang_enriched_to_parquet.rows_to_table(
        rows,
        tokenized_rows=tokenized_rows,
    )

    assert table.column("source_doc_id").to_pylist() == ["demo.cc@main"]
    for column in (
        "ast_depth",
        "sibling_index",
        "ast_node_type",
        "symbol_ids",
        "call_targets",
        "type_refs",
        "def_use",
        schema.TOKEN_SYMBOL_IDS_COLUMN,
        schema.TOKEN_CALL_TARGETS_COLUMN,
        schema.TOKEN_TYPE_REFS_COLUMN,
        schema.TOKEN_DEF_USE_COLUMN,
    ):
        expected = (
            tokenized_rows[0][column] if column in tokenized_rows[0] else rows[0][column]
        )
        assert table.column(column).to_pylist() == [expected]


def test_converter_header_alignment_preserves_char_metadata_coordinates() -> None:
    record = {
        "text": "abc",
        "structure_ids": [1, 2, 3],
        "chunk_boundaries": [{"start": 0, "end": 3, "kind": 3, "dep_level": 0}],
        "call_edges": [],
        "type_edges": [],
        "ast_depth": [4, 5, 6],
        "sibling_index": [7, 8, 9],
        "ast_node_type": [10, 11, 12],
        "symbol_ids": [13, 14, 15],
        "call_targets": [16, 17, 18],
        "type_refs": [19, 20, 21],
        "def_use": [1, 2, 1],
    }

    docs = clang_enriched_to_parquet.process_record_with_policy(
        record,
        _CharTokenizer(),
        max_tokens=4096,
        overflow_policy="drop",
    )

    assert len(docs) == 1
    doc = docs[0]
    header_len = len(doc["text"]) - len(record["text"])
    assert header_len > 0
    assert doc["structure_ids"] == [0] * header_len + record["structure_ids"]
    assert doc["platform_info"] == clang_enriched_to_parquet._DEFAULT_PLATFORM_INFO
    for key in (
        "ast_depth",
        "sibling_index",
        "ast_node_type",
        "symbol_ids",
        "call_targets",
        "type_refs",
        "def_use",
    ):
        assert doc[key] == [0] * header_len + record[key]


def test_local_parquet_conversion_streams_row_groups(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "out.parquet"
    records = [
        {
            "text": "int one() { return 1; }",
            "structure_ids": [3] * len("int one() { return 1; }"),
            "chunk_boundaries": [
                {"start": 0, "end": len("int one() { return 1; }"), "kind": 3, "dep_level": 0}
            ],
            "call_edges": [],
            "type_edges": [],
        },
        {
            "text": "int two() { return 2; }",
            "structure_ids": [3] * len("int two() { return 2; }"),
            "chunk_boundaries": [
                {"start": 0, "end": len("int two() { return 2; }"), "kind": 3, "dep_level": 0}
            ],
            "call_edges": [],
            "type_edges": [],
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    summary = clang_enriched_to_parquet.convert_local_jsonl_to_parquet(
        input_path,
        output_path,
        tokenizer=load_tokenizer(),
        max_tokens=4096,
        overflow_policy="drop",
        materialize_tokenized_enriched=True,
        local_batch_size=1,
    )

    parquet_file = pq.ParquetFile(output_path)
    assert summary == {"docs_in": 2, "docs_out": 2}
    assert parquet_file.metadata.num_rows == 2
    assert parquet_file.metadata.num_row_groups == 2
    # V7-G02: when the input row doesn't carry an explicit
    # source_doc_id, convert_local_jsonl_to_parquet now prefers the
    # stable doc signature (text_sha256:...) over the legacy
    # "<name>:<index>" form. Two distinct documents must still produce
    # two distinct, deterministic ids of the new shape.
    source_doc_ids = parquet_file.read(
        columns=["source_doc_id"]
    ).column("source_doc_id").to_pylist()
    assert len(source_doc_ids) == 2
    assert all(
        isinstance(v, str) and v.startswith("text_sha256:") and len(v) > len("text_sha256:")
        for v in source_doc_ids
    ), source_doc_ids
    assert len(set(source_doc_ids)) == 2
    fingerprints = parquet_file.read(
        columns=["tokenizer_fingerprint"]
    ).column("tokenizer_fingerprint").to_pylist()
    assert len(set(fingerprints)) == 1
    assert fingerprints[0] == tokenizer_fingerprint(load_tokenizer())


def test_tokenizer_fingerprint_and_ids_stable_across_independent_shards(
    tmp_path: Path,
) -> None:
    texts = [f"int fn_{idx}() {{ return {idx}; }}" for idx in range(32)]

    shard_token_rows = []
    shard_fingerprints = []
    for shard_idx in range(3):
        tokenizer = load_tokenizer()
        input_path = tmp_path / f"input_{shard_idx}.jsonl"
        output_path = tmp_path / f"train_{shard_idx:05d}.parquet"
        input_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "text": text,
                        "structure_ids": [3] * len(text),
                        "chunk_boundaries": [
                            {
                                "start": 0,
                                "end": len(text),
                                "kind": 3,
                                "dep_level": 0,
                            }
                        ],
                        "call_edges": [],
                        "type_edges": [],
                    }
                )
                for text in texts
            )
            + "\n",
            encoding="utf-8",
        )
        clang_enriched_to_parquet.convert_local_jsonl_to_parquet(
            input_path,
            output_path,
            tokenizer=tokenizer,
            max_tokens=4096,
            overflow_policy="drop",
            materialize_tokenized_enriched=True,
            local_batch_size=8,
        )
        table = pq.read_table(
            output_path,
            columns=["token_ids", "tokenizer_fingerprint"],
        )
        shard_token_rows.append(table.column("token_ids").to_pylist())
        shard_fingerprints.append(
            set(table.column("tokenizer_fingerprint").to_pylist())
        )

    assert shard_token_rows[0] == shard_token_rows[1] == shard_token_rows[2]
    assert len({next(iter(fingerprints)) for fingerprints in shard_fingerprints}) == 1


def test_multishard_parquet_rejects_tokenizer_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    shard0 = tmp_path / "train_00000.parquet"
    shard1 = tmp_path / "train_00001.parquet"
    pq.write_table(
        pa.table(
            {
                "token_ids": [[1, 2, 3, 4]],
                "tokenizer_fingerprint": ["a" * 64],
            }
        ),
        shard0,
    )
    pq.write_table(
        pa.table(
            {
                "token_ids": [[5, 6, 7, 8]],
                "tokenizer_fingerprint": ["b" * 64],
            }
        ),
        shard1,
    )

    with pytest.raises(ValueError, match="tokenizer_fingerprint mismatch"):
        MultiShardTokenParquetDataset(
            [shard0, shard1],
            seq_len=4,
            batch_size=1,
            token_key="token_ids",
        )


def test_multi_shard_parquet_stream_matches_one_by_one_shard_order(
    tmp_path: Path,
) -> None:
    shard0 = _write_token_parquet(
        tmp_path / "corpus" / "val_00000.parquet",
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        tokenizer_fingerprint="c" * 64,
    )
    shard1 = _write_token_parquet(
        tmp_path / "corpus" / "val_00001.parquet",
        [[9, 10, 11, 12]],
        tokenizer_fingerprint="c" * 64,
    )
    kwargs = {"seq_len": 4, "batch_size": 1, "token_key": "token_ids"}
    streamed = MultiShardTokenParquetDataset([shard0, shard1], **kwargs)

    expected = (
        _dataset_token_rows(TokenParquetDataset(shard0, **kwargs))
        + _dataset_token_rows(TokenParquetDataset(shard1, **kwargs))
    )

    assert not hasattr(streamed, "_datasets")
    assert streamed.parquet_receipt["stream"]["tokenizer_fingerprint"] == "c" * 64
    assert _dataset_token_rows(streamed) == expected
    assert [
        batch.metadata["parquet_stream"]["shard_index"]
        for batch in streamed.iter_batches(loop=False)
    ] == [0, 0, 1]


def test_cross_shard_doc_id_manifest_reconstructs_boundary_spanning_doc(
    tmp_path: Path,
) -> None:
    shards = [
        [{"source_doc_id": "alpha", "token_ids": [1, 2], "text": "ab"}],
        [
            {"source_doc_id": "alpha", "token_ids": [3, 4], "text": "cd"},
            {"source_doc_id": "beta", "token_ids": [9], "text": "x"},
        ],
        [{"source_doc_id": "alpha", "token_ids": [5, 6], "text": "ef"}],
    ]

    assignment = assign_sharded_doc_ids(shards)
    alpha_id = assignment.doc_ids_by_shard[0][0]

    assert assignment.doc_ids_by_shard[1][0] == alpha_id
    assert assignment.doc_ids_by_shard[2][0] == alpha_id
    assert assignment.doc_ids_by_shard[1][1] != alpha_id
    assert assignment.manifest[str(alpha_id)] == [
        {"shard_index": 0, "start_row": 0, "end_row": 1},
        {"shard_index": 1, "start_row": 0, "end_row": 1},
        {"shard_index": 2, "start_row": 0, "end_row": 1},
    ]
    manifest_path = tmp_path / "doc_id_manifest.json"
    write_doc_id_manifest(manifest_path, assignment)
    sidecar = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert sidecar["schema"] == "cppmega_doc_id_manifest_v1"
    assert sidecar["manifest"][str(alpha_id)] == assignment.manifest[str(alpha_id)]

    reconstructed: list[int] = []
    for span in assignment.manifest[str(alpha_id)]:
        for row in shards[span["shard_index"]][span["start_row"]:span["end_row"]]:
            reconstructed.extend(row["token_ids"])
    assert reconstructed == [1, 2, 3, 4, 5, 6]

    alpha_mask = document_boundary_mask([[alpha_id] * len(reconstructed)], causal=True)
    assert bool(alpha_mask[0, 2, 1])
    assert bool(alpha_mask[0, 4, 3])


def test_pack_enriched_rows_preserves_source_doc_id_across_input_shards(
    tmp_path: Path,
) -> None:
    shard0 = tmp_path / "train_00000.parquet"
    shard1 = tmp_path / "train_00001.parquet"
    pq.write_table(
        pa.table({"token_ids": [[1, 2]], "source_doc_id": ["alpha"]}),
        shard0,
    )
    pq.write_table(
        pa.table(
            {
                "token_ids": [[3, 4], [9, 10]],
                "source_doc_id": ["alpha", "beta"],
            }
        ),
        shard1,
    )

    docs = read_tokenized_documents(tmp_path)
    rows, overflow = pack_documents(
        docs,
        target_length=6,
        pad_token_id=0,
        strategy="sequential",
    )

    assert overflow == []
    assert rows[0][INPUT_IDS_COLUMN] == [1, 2, 3, 4, 9, 10]
    assert rows[0][DOC_IDS_COLUMN][0] == rows[0][DOC_IDS_COLUMN][2]
    assert rows[0][DOC_IDS_COLUMN][3] != rows[0][DOC_IDS_COLUMN][4]


def test_token_budget_slices_semantic_char_metadata() -> None:
    doc = {
        "text": "abcdef",
        "structure_ids": [1, 2, 3, 4, 5, 6],
        "chunk_boundaries": [],
        "call_edges": [],
        "type_edges": [],
        "ast_depth": [10, 11, 12, 13, 14, 15],
        "sibling_index": [20, 21, 22, 23, 24, 25],
        "ast_node_type": [30, 31, 32, 33, 34, 35],
        "symbol_ids": [40, 41, 42, 43, 44, 45],
        "call_targets": [50, 51, 52, 53, 54, 55],
        "type_refs": [60, 61, 62, 63, 64, 65],
        "def_use": [0, 1, 2, 0, 1, 2],
    }

    pieces = chunk_enriched_document(doc, max_tokens=3, tokenizer=_CharTokenizer())

    assert [piece["text"] for piece in pieces] == ["abc", "def"]
    assert pieces[0]["symbol_ids"] == [40, 41, 42]
    assert pieces[1]["symbol_ids"] == [43, 44, 45]
    assert pieces[0]["def_use"] == [0, 1, 2]
    assert pieces[1]["def_use"] == [0, 1, 2]


def test_clang_indexer_ast_metadata_comes_from_clang(tmp_path: Path) -> None:
    source = tmp_path / "demo.cc"
    source.write_text(
        "int helper() { return 1; }\n"
        "int main() { return helper(); }\n",
        encoding="utf-8",
    )

    index_project._configure_libclang(None)
    clang_index = index_project.Index.create()
    funcs, _typedefs = index_project.parse_translation_unit(
        str(source),
        clang_index,
        ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
        str(tmp_path),
    )

    main = next(func for func in funcs if func.name == "main")
    assert len(main.ast_depth) == len(main.text)
    assert len(main.sibling_index) == len(main.text)
    assert len(main.ast_node_type) == len(main.text)
    assert any(main.ast_depth)
    assert 20 in main.ast_node_type  # clang CALL_EXPR bucket


def test_build_context_discovers_nested_compile_commands_and_absolutizes_includes(
    tmp_path: Path,
) -> None:
    (tmp_path / "build").mkdir()
    (tmp_path / "include").mkdir()
    (tmp_path / "src").mkdir()
    compile_commands = [
        {
            "directory": str(tmp_path),
            "command": "clang++ -Iinclude -std=c++20 -c src/main.cc -o main.o",
            "file": "src/main.cc",
        }
    ]
    (tmp_path / "build" / "compile_commands.json").write_text(
        json.dumps(compile_commands),
        encoding="utf-8",
    )

    discovered = find_compile_commands_file(str(tmp_path))
    context, _default_args, compile_index = detect_build_context(str(tmp_path))

    assert discovered == str(tmp_path / "build" / "compile_commands.json")
    assert context["build_system"] == "compile_commands"
    assert compile_index is not None
    file_args, build_info = compile_index.lookup(str(tmp_path / "src" / "main.cc"))
    assert file_args is not None
    assert f"-I{tmp_path / 'include'}" in file_args
    assert "src/main.cc" not in file_args
    assert build_info == {
        "build_system": "compile_commands",
        "source": "compile_commands",
        "compiler": "clang++",
    }


def test_commit_clang_analysis_uses_repo_build_context_for_virtual_source(
    tmp_path: Path,
) -> None:
    (tmp_path / "build").mkdir()
    (tmp_path / "include").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "include" / "dep.hpp").write_text(
        "inline int dep() { return 7; }\n",
        encoding="utf-8",
    )
    (tmp_path / "build" / "compile_commands.json").write_text(
        json.dumps(
            [
                {
                    "directory": str(tmp_path),
                    "command": "clang++ -Iinclude -std=c++17 -c src/main.cc -o main.o",
                    "file": "src/main.cc",
                }
            ]
        ),
        encoding="utf-8",
    )
    source = '#include "dep.hpp"\nint main() { return dep(); }\n'

    index_project._configure_libclang(None)
    clang_index = index_project.Index.create()
    fallback = analyze_file_clang(
        source,
        "src/main.cc",
        clang_index,
        str(tmp_path / "fallback"),
    )
    repo_root, compile_args, build_info = BuildContextResolver(
        repo_root=str(tmp_path)
    ).resolve({"filepath": "src/main.cc"})
    build_aware = analyze_file_clang(
        source,
        "src/main.cc",
        clang_index,
        str(tmp_path / "build-aware"),
        compile_args=compile_args,
        repo_root=repo_root,
        build_info=build_info,
    )

    assert next(func for func in fallback.functions if func.name == "main").callees == []
    main = next(func for func in build_aware.functions if func.name == "main")
    assert main.callees == ["dep"]
    assert build_aware.build_info["build_system"] == "compile_commands"


def test_commit_enriched_builder_emits_semantic_columns_without_libclang_runtime() -> None:
    helper_text = "int helper() { return 1; }"
    main_text = "int main() { return helper(); }"
    helper = FunctionDef(
        "helper",
        "helper",
        "src/demo.cc",
        1,
        helper_text,
        [],
        ast_depth=[1] * len(helper_text),
        sibling_index=[0] * len(helper_text),
        ast_node_type=[1] * len(helper_text),
    )
    main = FunctionDef(
        "main",
        "main",
        "src/demo.cc",
        3,
        main_text,
        ["helper"],
        ast_depth=[2] * len(main_text),
        sibling_index=[1] * len(main_text),
        ast_node_type=[20] * len(main_text),
    )
    analysis = FileAnalysis("", functions=[helper, main])
    parts: list[PartInfo] = [
        (helper_text, 3, 0, "helper", "helper"),
        (main_text, 3, 1, "main", "main"),
    ]

    doc = _build_enriched_from_parts(
        parts,
        analysis,
        None,
        {"filepath": "src/demo.cc"},
    )

    text_len = len(doc["text"])
    for key in ("symbol_ids", "call_targets", "type_refs", "def_use"):
        assert len(doc[key]) == text_len
    assert len(doc["ast_depth"]) == text_len
    assert 20 in doc["ast_node_type"]
    assert any(doc["symbol_ids"])
    assert any(doc["call_targets"])
    assert any(value == 1 for value in doc["def_use"])


def test_commit_enriched_builder_emits_temporal_char_annotations() -> None:
    old_text = "int main() { return 1; }"
    new_text = "int main() { return 2; }"
    old_main = FunctionDef("main", "main", "src/demo.cc", 1, old_text, [])
    new_main = FunctionDef("main", "main", "src/demo.cc", 1, new_text, [])
    diff = "\n".join(
        [
            "diff --git a/src/demo.cc b/src/demo.cc",
            "--- a/src/demo.cc",
            "+++ b/src/demo.cc",
            "@@ -1 +1 @@",
            "-int main() { return 1; }",
            "+int main() { return 2; }",
        ]
    )
    parts: list[PartInfo] = [
        ("// === PRE-COMMIT ===", 0, 0, "", None),
        (old_text, 3, 0, "main", "main"),
        ("// === POST-COMMIT ===", 0, 0, "", None),
        (new_text, 3, 0, "main", "main"),
    ]

    doc = _build_enriched_from_parts(
        parts,
        FileAnalysis("", functions=[old_main]),
        FileAnalysis("", functions=[new_main]),
        {
            "filepath": "src/demo.cc",
            "old_content": old_text,
            "new_content": new_text,
            "diff": diff,
        },
        section_kinds=["c", "o", "c", "n"],
    )

    assert len(doc["change_mask_pre"]) == len(doc["text"])
    assert len(doc["change_mask_post"]) == len(doc["text"])
    assert any(doc["change_mask_pre"])
    assert any(doc["change_mask_post"])
    assert any(value == 2 for value in doc["edit_op_per_char"])
    assert any(doc["hunk_id_per_char"])


def test_tokenized_materializer_maps_temporal_char_annotations_to_tokens() -> None:
    text = "int main() { return 2; }"
    change_mask = [0] * len(text)
    edit_ops = [3] * len(text)
    start = text.index("2")
    change_mask[start] = 1
    edit_ops[start] = 2
    docs = [
        {
            "text": text,
            "structure_ids": [3] * len(text),
            "chunk_boundaries": [{"start": 0, "end": len(text), "kind": 3, "dep_level": 0}],
            "call_edges": [],
            "type_edges": [],
            "change_mask_post": change_mask,
            "hunk_id_per_char": change_mask,
            "edit_op_per_char": edit_ops,
        }
    ]

    row = materialize_tokenized_enriched_batch(docs, load_tokenizer())[0]

    assert any(row[schema.TOKEN_CHANGE_MASK_POST_COLUMN])
    assert any(value == 2 for value in row[schema.EDIT_OP_PER_TOKEN_COLUMN])
    assert row[schema.CHANGED_CHUNK_IDS_COLUMN] == [0]
    assert row[schema.CHANGED_CHUNK_SPANS_COLUMN]
