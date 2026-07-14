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
from cppmega_mlx.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    delimiter_token_ids,
)
from cppmega_mlx.data.nanochat_pipeline import platform_vocab
from cppmega_mlx.data.nanochat_pipeline import tokenized_enriched_schema as schema
from cppmega_mlx.data.nanochat_pipeline.build_context import (
    detect_build_context,
    find_compile_commands_file,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched import (
    _inherit_zero_width_source_tokens,
    materialize_tokenized_enriched_batch,
)
from cppmega_v4.data.doc_id_assignment import (
    assign_sharded_doc_ids,
    write_doc_id_manifest,
)
from scripts.nanochat_data import clang_enriched_to_parquet, token_budget
from scripts.nanochat_data.pack_enriched_rows import (
    DOC_IDS_COLUMN,
    INPUT_IDS_COLUMN,
    LOSS_MASK_COLUMN,
    NUM_DOCS_COLUMN,
    read_tokenized_documents,
    pack_documents,
)
from scripts.nanochat_data.token_budget import chunk_enriched_document, load_tokenizer
from scripts.nanochat_data.token_budget import tokenizer_fingerprint
from tools.clang_indexer import index_project
from tools.clang_indexer.index_project import FunctionDef, MacroDef, PartInfo, ProjectIndex
from tools.clang_indexer.process_commits import (
    AnalysisCache,
    BuildContextResolver,
    FileAnalysis,
    _macro_dependency_parts_for_commit_targets,
    _build_enriched_from_parts,
    analyze_file_clang,
    process_record,
)


class _CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def get_vocab(self) -> dict[str, int]:
        return {}


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


def test_synthetic_special_token_inherits_nearest_exact_source() -> None:
    assert _inherit_zero_width_source_tokens(
        [0, 11, 22, 0],
        [(0, 0), (0, 1), (1, 2), (2, 2)],
        field="token_source_identity_ids",
    ) == [11, 11, 22, 22]


def test_nonempty_token_without_source_identity_fails_closed() -> None:
    with pytest.raises(ValueError, match="nonempty token span"):
        _inherit_zero_width_source_tokens(
            [11, 0, 22],
            [(0, 1), (1, 2), (2, 3)],
            field="token_source_identity_ids",
        )


def test_local_tokenized_schema_keeps_full_nanochat_column_contract() -> None:
    nanochat_schema = _load_nanochat_module("nanochat/tokenized_enriched_schema.py")

    local_columns = schema.TOKENIZED_ENRICHED_COLUMNS
    local_positions = {column: index for index, column in enumerate(local_columns)}
    upstream_columns = nanochat_schema.TOKENIZED_ENRICHED_COLUMNS

    assert set(upstream_columns) <= set(local_columns)
    assert [
        column for column in local_columns
        if column in upstream_columns
    ] == list(upstream_columns)
    assert {
        schema.TOKEN_DOMAIN_IDS_COLUMN,
        schema.TOKEN_BUILD_EDGES_COLUMN,
        schema.TOKEN_SHELL_EDGES_COLUMN,
        schema.TOKEN_DIAGNOSTIC_EDGES_COLUMN,
        schema.TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
    } <= set(local_columns)
    assert local_positions[schema.TOKEN_TYPE_EDGES_COLUMN] < local_positions[
        schema.TOKEN_DOMAIN_IDS_COLUMN
    ]


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
    assert clang_enriched_to_parquet.REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION == 3
    assert "symbol_identities" in clang_enriched_to_parquet._SCHEMA.names
    for column in (
        "symbol_ids",
        "call_targets",
        "type_refs",
        schema.TOKEN_SYMBOL_IDS_COLUMN,
        schema.TOKEN_CALL_TARGETS_COLUMN,
        schema.TOKEN_TYPE_REFS_COLUMN,
    ):
        assert clang_enriched_to_parquet._SCHEMA.field(column).type.value_type == pa.uint64()
    boundary_type = clang_enriched_to_parquet._SCHEMA.field(
        "chunk_boundaries"
    ).type.value_type
    assert boundary_type.field("symbol_id").type == pa.uint64()
    metadata = clang_enriched_to_parquet._SCHEMA.metadata or {}
    assert metadata[
        clang_enriched_to_parquet.DOMAIN_SCHEMA_SHA256_METADATA_KEY.encode("utf-8")
    ] == clang_enriched_to_parquet.DOMAIN_SCHEMA_SHA256.encode("ascii")
    assert metadata[
        clang_enriched_to_parquet.TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode(
            "utf-8"
        )
    ] == clang_enriched_to_parquet.TOKENIZER_CONTRACT_SHA256.encode("ascii")


def test_clang_enriched_docs_to_table_carries_token_semantic_columns() -> None:
    keys = [f"usr:schema=v3\x1fproject=test\x1fusr=c:@F@symbol{i}#" for i in range(3)]
    symbol_id, call_id, type_id = [index_project._compute_symbol_id(key) for key in keys]
    identities = [
        {"symbol_id": value, "symbol_key": key}
        for key, value in zip(keys, (symbol_id, call_id, type_id), strict=True)
    ]
    rows = [
        {
            "symbol_identity_schema_version": 3,
            "symbol_identities": identities,
            "text": "int main() { return f(); }",
            "source_doc_id": "demo.cc@main",
            schema.DOC_TYPE_COLUMN: "code_header",
            schema.HEADER_FRAGMENT_KIND_COLUMN: "function_template",
            "actual_token_count": 4,
            "structure_ids": [3, 3, 3, 3],
            "chunk_boundaries": [{"start": 0, "end": 24, "kind": 3, "dep_level": 0}],
            "call_edges": [],
            "type_edges": [],
            "ast_depth": [0, 1, 2, 1],
            "sibling_index": [0, 0, 1, 2],
            "ast_node_type": [1, 2, 3, 4],
            "symbol_ids": [0, symbol_id, symbol_id, 0],
            "call_targets": [0, call_id, 0, 0],
            "type_refs": [0, 0, type_id, 0],
            "def_use": [0, 1, 2, 0],
        }
    ]
    tokenized_rows = [
        {
            schema.TOKEN_IDS_COLUMN: [1, 2, 3, 4],
            schema.TOKEN_SYMBOL_IDS_COLUMN: [0, symbol_id, symbol_id, 0],
            schema.TOKEN_CALL_TARGETS_COLUMN: [0, call_id, 0, 0],
            schema.TOKEN_TYPE_REFS_COLUMN: [0, 0, type_id, 0],
            schema.TOKEN_DEF_USE_COLUMN: [0, 1, 2, 0],
        }
    ]

    table = clang_enriched_to_parquet.rows_to_table(
        rows,
        tokenized_rows=tokenized_rows,
    )

    assert table.column("source_doc_id").to_pylist() == ["demo.cc@main"]
    assert table.column(schema.DOC_TYPE_COLUMN).to_pylist() == ["code_header"]
    assert table.column(schema.HEADER_FRAGMENT_KIND_COLUMN).to_pylist() == [
        "function_template"
    ]
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


def test_local_convert_backfills_static_code_repo_provenance(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "out.parquet"
    input_path.write_text(
        json.dumps(
            {
                "symbol_identity_schema_version": 3,
                "symbol_identities": [],
                "text": "int add(int a, int b) { return a + b; }",
                "filepath": "include/math.hpp",
                "structure_ids": [3] * 40,
                "chunk_boundaries": [
                    {"start": 0, "end": 40, "kind": 3, "dep_level": 0}
                ],
                "call_edges": [],
                "type_edges": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    clang_enriched_to_parquet.convert_local_jsonl_to_parquet(
        input_path,
        output_path,
        tokenizer=_CharTokenizer(),
        max_tokens=4096,
        overflow_policy="drop",
        default_repo="tests/demo-lib",
    )

    table = pq.read_table(
        output_path,
        columns=[
            schema.REPO_COLUMN,
            schema.FILEPATH_COLUMN,
            schema.REPO_STABLE_ID_COLUMN,
            schema.FILEPATH_STABLE_ID_COLUMN,
        ],
    )
    assert table.column(schema.REPO_COLUMN).to_pylist() == ["tests/demo-lib"]
    assert table.column(schema.FILEPATH_COLUMN).to_pylist() == ["include/math.hpp"]
    assert table.column(schema.REPO_STABLE_ID_COLUMN).to_pylist() == [
        clang_enriched_to_parquet.stable_repo_id("tests/demo-lib")
    ]
    assert table.column(schema.FILEPATH_STABLE_ID_COLUMN).to_pylist() == [
        clang_enriched_to_parquet.stable_filepath_id(
            "tests/demo-lib",
            "include/math.hpp",
        )
    ]


def test_local_convert_fails_on_static_code_without_repo_context(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "out.parquet"
    input_path.write_text(
        json.dumps(
            {
                "text": "int f() { return 1; }",
                "doc_type": "code",
                "filepath": "src/f.cc",
                "structure_ids": [3] * 21,
                "chunk_boundaries": [
                    {"start": 0, "end": 21, "kind": 3, "dep_level": 0}
                ],
                "call_edges": [],
                "type_edges": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing static code repo provenance"):
        clang_enriched_to_parquet.convert_local_jsonl_to_parquet(
            input_path,
            output_path,
            tokenizer=_CharTokenizer(),
            max_tokens=4096,
            overflow_policy="drop",
        )


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


def test_embedded_domain_spans_follow_filtering_and_header_insertion() -> None:
    dead = "#ifdef __SYMBIAN32__\nconst char* dead = \"SELECT dead\";\n#endif\n"
    live = 'const char* live = R"SQL(SELECT live;)SQL";\n'
    text = dead + live
    live_start = text.index("SELECT live")
    live_end = live_start + len("SELECT live;")
    record = {
        "text": text,
        "structure_ids": [1] * len(text),
        "embedded_domain_spans": [
            {
                "start": live_start,
                "end": live_end,
                "domain_kind": int(DomainKind.SQL),
            }
        ],
    }

    docs = clang_enriched_to_parquet.process_record_with_policy(
        record,
        _CharTokenizer(),
        max_tokens=10_000,
    )

    assert len(docs) == 1
    doc = docs[0]
    filtered, _ = clang_enriched_to_parquet.filter_dead_platforms_with_mapping(text)
    header_len = len(doc["text"]) - len(filtered)
    expected_start = header_len + filtered.index("SELECT live")
    assert doc["embedded_domain_spans"] == [
        {
            "start": expected_start,
            "end": expected_start + len("SELECT live;"),
            "domain_kind": int(DomainKind.SQL),
        }
    ]


def test_embedded_domain_spans_discard_deleted_regions_and_fail_on_partial_filter() -> None:
    text = (
        "before\n"
        "#ifdef __SYMBIAN32__\n"
        'const char* dead = R"SQL(SELECT dead;)SQL";\n'
        "#endif\n"
        "after\n"
    )
    dead_start = text.index("SELECT dead")
    dead_end = dead_start + len("SELECT dead;")
    deleted = clang_enriched_to_parquet.process_record_with_policy(
        {
            "text": text,
            "structure_ids": [1] * len(text),
            "embedded_domain_spans": [
                {
                    "start": dead_start,
                    "end": dead_end,
                    "domain_kind": int(DomainKind.SQL),
                }
            ],
        },
        _CharTokenizer(),
        max_tokens=10_000,
    )[0]
    assert deleted["embedded_domain_spans"] == []

    with pytest.raises(ValueError, match="cannot exactly remap embedded domain span"):
        clang_enriched_to_parquet.process_record_with_policy(
            {
                "text": text,
                "structure_ids": [1] * len(text),
                "embedded_domain_spans": [
                    {
                        "start": 0,
                        "end": len(text),
                        "domain_kind": int(DomainKind.SQL),
                    }
                ],
            },
            _CharTokenizer(),
            max_tokens=10_000,
        )


def test_dead_platform_filter_remaps_live_sidecars_and_drops_removed_edges() -> None:
    live_a = "int live_a() { return 1; }\n"
    dead = "#ifdef __SYMBIAN32__\nint dead() { return 0; }\n#endif\n"
    live_b = "int live_b() { return live_a(); }\n"
    text = live_a + dead + live_b
    live_b_start = text.index("int live_b")
    live_b_name = text.index("live_b")
    live_a_call = text.rindex("live_a")
    dead_name = text.index("dead")

    record = {
        "text": text,
        "structure_ids": [3] * len(text),
        "chunk_boundaries": [
            {"start": 0, "end": len(live_a), "kind": 3, "dep_level": 0, "name": "live_a"},
            {
                "start": len(live_a),
                "end": len(live_a) + len(dead),
                "kind": 3,
                "dep_level": 0,
                "name": "dead",
            },
            {
                "start": live_b_start,
                "end": len(text),
                "kind": 3,
                "dep_level": 0,
                "name": "live_b",
            },
        ],
        "call_edges": [
            {"from": 2, "to": 0},
            {"from": 2, "to": 1},
        ],
        "type_edges": [{"from": 0, "to": 2}],
        "symbol_ids": list(range(1000, 1000 + len(text))),
        "domain_edges": [
            {
                "from_char": live_b_name,
                "to_char": live_a_call,
                "kind": int(DomainEdgeKind.CALL),
            },
            {
                "from_char": live_b_name,
                "to_char": dead_name,
                "kind": int(DomainEdgeKind.CALL),
            },
        ],
    }

    docs = clang_enriched_to_parquet.process_record_with_policy(
        record,
        _CharTokenizer(),
        max_tokens=4096,
        overflow_policy="drop",
    )

    assert len(docs) == 1
    doc = docs[0]
    assert "__SYMBIAN32__" not in doc["text"]
    assert "int dead()" not in doc["text"]
    live_b_out = doc["text"].index("int live_b")
    live_b_name_out = doc["text"].index("live_b")
    live_a_call_out = doc["text"].rindex("live_a")

    assert doc["structure_ids"][live_b_out] == 3
    assert doc["symbol_ids"][live_b_out] == record["symbol_ids"][live_b_start]
    assert [chunk["name"] for chunk in doc["chunk_boundaries"]] == ["live_a", "live_b"]
    assert doc["call_edges"] == [{"from": 1, "to": 0}]
    assert doc["type_edges"] == [{"from": 0, "to": 1}]
    assert doc["domain_edges"] == [
        {
            "from_char": live_b_name_out,
            "to_char": live_a_call_out,
            "kind": int(DomainEdgeKind.CALL),
        }
    ]


def test_local_parquet_conversion_streams_row_groups(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "out.parquet"
    records = [
        {
            "symbol_identity_schema_version": 3,
            "symbol_identities": [],
            "text": "int one() { return 1; }",
            "structure_ids": [3] * len("int one() { return 1; }"),
            "chunk_boundaries": [
                {"start": 0, "end": len("int one() { return 1; }"), "kind": 3, "dep_level": 0}
            ],
            "call_edges": [],
            "type_edges": [],
        },
        {
            "symbol_identity_schema_version": 3,
            "symbol_identities": [],
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
    token_source_doc_ids = parquet_file.read(
        columns=["token_source_doc_ids"]
    ).column("token_source_doc_ids").to_pylist()
    assert all(
        source_id > 0
        for row in token_source_doc_ids
        for source_id in row
    )
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
                        "symbol_identity_schema_version": 3,
                        "symbol_identities": [],
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
    assert rows[0][DOC_IDS_COLUMN] == [1, 1, 2, 2, 3, 3]
    assert rows[0][LOSS_MASK_COLUMN] == [1, 0, 1, 0, 1, 0]
    assert rows[0][NUM_DOCS_COLUMN] == 3
    assert rows[0][schema.TOKEN_SOURCE_DOC_IDS_COLUMN] == [1] * 6
    stable_sources = rows[0][schema.TOKEN_SOURCE_IDENTITY_IDS_COLUMN]
    assert stable_sources[0] == stable_sources[2]
    assert stable_sources[3] != stable_sources[4]
    assert all(value > 0 for value in stable_sources)
    assert docs[0].stable_doc_id == docs[1].stable_doc_id
    assert docs[1].stable_doc_id != docs[2].stable_doc_id


def test_pack_enriched_rows_does_not_merge_file_provenance_without_logical_id(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "train_00000.parquet"
    pq.write_table(
        pa.table(
            {
                "token_ids": [[1, 2], [3, 4]],
                "repo_stable_id": ["repo", "repo"],
                "filepath_stable_id": ["include/shared.hpp", "include/shared.hpp"],
            }
        ),
        shard,
    )

    docs = read_tokenized_documents(shard)
    assert [doc.source_doc_index for doc in docs] == [0, 1]
    rows, overflow = pack_documents(
        docs,
        target_length=4,
        pad_token_id=0,
        strategy="sequential",
    )

    assert overflow == []
    assert rows[0][DOC_IDS_COLUMN] == [1, 1, 2, 2]
    assert rows[0][LOSS_MASK_COLUMN] == [1, 0, 1, 0]
    assert rows[0][NUM_DOCS_COLUMN] == 2


def test_pack_enriched_rows_does_not_collide_anonymous_and_typed_source_ids(
    tmp_path: Path,
) -> None:
    pq.write_table(
        pa.table(
            {
                "token_ids": [[1, 2], [3, 4]],
                "source_doc_id": [None, "alpha"],
            }
        ),
        tmp_path / "train_00000.parquet",
    )

    docs = read_tokenized_documents(tmp_path)
    rows, overflow = pack_documents(
        docs,
        target_length=4,
        pad_token_id=0,
        strategy="sequential",
    )

    assert overflow == []
    assert rows[0][schema.TOKEN_SOURCE_DOC_IDS_COLUMN] == [1] * 4
    source_ids = rows[0][schema.TOKEN_SOURCE_IDENTITY_IDS_COLUMN]
    assert source_ids[:2] == [source_ids[0], source_ids[0]]
    assert source_ids[2:] == [source_ids[2], source_ids[2]]
    assert source_ids[0] > 0 and source_ids[2] > 0
    assert source_ids[0] != source_ids[2]


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


def test_token_budget_clips_embedded_domain_spans_at_every_split_boundary() -> None:
    text = "aaSELECT123bb"
    span_start = text.index("SELECT")
    span_end = span_start + len("SELECT123")
    doc = {
        "text": text,
        "structure_ids": [1] * len(text),
        "embedded_domain_spans": [
            {
                "start": span_start,
                "end": span_end,
                "domain_kind": int(DomainKind.SQL),
            }
        ],
    }

    pieces = token_budget.chunk_enriched_document(doc, 5, _CharTokenizer())

    source_offset = 0
    for piece in pieces:
        piece_end = source_offset + len(piece["text"])
        overlap_start = max(span_start, source_offset)
        overlap_end = min(span_end, piece_end)
        expected = []
        if overlap_start < overlap_end:
            expected = [
                {
                    "start": overlap_start - source_offset,
                    "end": overlap_end - source_offset,
                    "domain_kind": int(DomainKind.SQL),
                }
            ]
        assert piece["embedded_domain_spans"] == expected
        source_offset = piece_end
    assert source_offset == len(text)


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
        project_id="tests/nanochat-fixture",
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
        project_id="tests/nanochat-fixture",
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
        project_id="tests/nanochat-fixture",
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


def test_commit_enriched_builder_emits_cpp_domain_macro_sidecars() -> None:
    macro_text = "#define CPPMEGA_COMMIT_WRAP(x) ((x) + 1)\n"
    main_text = "int main(int x) { return CPPMEGA_COMMIT_WRAP(x); }"
    main = FunctionDef(
        "main",
        "main",
        "src/demo.cc",
        3,
        main_text,
        [],
    )
    analysis = FileAnalysis(macro_text, functions=[main])
    parts: list[PartInfo] = [
        (macro_text, 1, 0, "", None),
        (main_text, 3, 0, "main", "main"),
    ]

    doc = _build_enriched_from_parts(
        parts,
        analysis,
        None,
        {"filepath": "src/demo.cc"},
    )

    text_len = len(doc["text"])
    assert doc["domain_kind"] == int(DomainKind.CPP)
    for key in (
        "domain_ids",
        "domain_role_ids",
        "domain_entity_ids",
        "domain_scope_ids",
        "domain_source_doc_ids",
        "domain_confidence_ids",
    ):
        assert len(doc[key]) == text_len
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_PARAM_USE)
        for edge in doc["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INVOCATION)
        for edge in doc["domain_edges"]
    )


def test_commit_macro_dependency_parts_emit_precise_expansion_routes() -> None:
    macro_index = ProjectIndex()
    base = MacroDef(
        "CPPMEGA_COMMIT_BASE",
        "include/macros.h",
        1,
        "#define CPPMEGA_COMMIT_BASE(x) ((x) + 1)\n",
        params=["x"],
        project_id="tests/nanochat-fixture",
        visible_in_file="src/demo.cc",
        visible_line=1,
        sequence=0,
    )
    wrap = MacroDef(
        "CPPMEGA_COMMIT_WRAP",
        "include/macros.h",
        2,
        "#define CPPMEGA_COMMIT_WRAP(x) CPPMEGA_COMMIT_BASE(x)\n",
        params=["x"],
        project_id="tests/nanochat-fixture",
        visible_in_file="src/demo.cc",
        visible_line=1,
        sequence=1,
    )
    macro_index.add_macro(base)
    macro_index.add_macro(wrap)
    main_text = "int main(int x) { return CPPMEGA_COMMIT_WRAP(x); }"
    main = FunctionDef("main", "main", "src/demo.cc", 3, main_text, [])
    macro_parts = _macro_dependency_parts_for_commit_targets(
        macro_index,
        [(main_text, "src/demo.cc", 3)],
    )

    assert [part[3] for part in macro_parts] == [
        "CPPMEGA_COMMIT_BASE",
        "CPPMEGA_COMMIT_WRAP",
    ]

    doc = _build_enriched_from_parts(
        [*macro_parts, (main_text, 3, 0, "main", "main")],
        FileAnalysis("", functions=[main]),
        None,
        {"filepath": "src/demo.cc"},
    )

    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INVOCATION)
        for edge in doc["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_EXPANSION_INCLUDE_ORDER)
        for edge in doc["domain_edges"]
    )


def test_process_record_pulls_header_macros_into_commit_docs_without_libclang(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "macros.h").write_text(
        "#define CPPMEGA_COMMIT_BASE(x) ((x) + 1)\n"
        "#define CPPMEGA_COMMIT_WRAP(x) CPPMEGA_COMMIT_BASE(x)\n",
        encoding="utf-8",
    )
    old_content = (
        '#include "../include/macros.h"\n'
        "int main(int x) {\n"
        "  return CPPMEGA_COMMIT_WRAP(x);\n"
        "}\n"
        "// enough trailing text for the minimum file-size guard\n"
    )
    new_content = old_content.replace(
        "return CPPMEGA_COMMIT_WRAP(x);",
        "return CPPMEGA_COMMIT_WRAP(x) + 2;",
    )
    (tmp_path / "src" / "demo.cc").write_text(new_content, encoding="utf-8")

    def fake_analyzer(
        content: str,
        filepath: str,
        clang_index,
        tmpdir: str,
        **_kwargs,
    ) -> FileAnalysis:
        body_line = (
            "  return CPPMEGA_COMMIT_WRAP(x) + 2;"
            if "+ 2" in content
            else "  return CPPMEGA_COMMIT_WRAP(x);"
        )
        text = f"int main(int x) {{\n{body_line}\n}}"
        return FileAnalysis(
            '#include "../include/macros.h"\n',
            functions=[
                FunctionDef(
                    "main",
                    "main",
                    filepath,
                    2,
                    text,
                    [],
                    end_line=4,
                )
            ],
        )

    docs = process_record(
        {
            "repo": "tests/demo",
            "filepath": "src/demo.cc",
            "old_content": old_content,
            "new_content": new_content,
            "diff": "\n".join(
                [
                    "diff --git a/src/demo.cc b/src/demo.cc",
                    "--- a/src/demo.cc",
                    "+++ b/src/demo.cc",
                    "@@ -3 +3 @@",
                    "-  return CPPMEGA_COMMIT_WRAP(x);",
                    "+  return CPPMEGA_COMMIT_WRAP(x) + 2;",
                ]
            ),
            "subject": "Use wrapped macro",
        },
        clang_index=object(),
        tmpdir=str(tmp_path / "tmp"),
        max_tokens=4096,
        max_file_bytes=100000,
        doc_format="chain",
        max_dep_depth=1,
        build_context=BuildContextResolver(repo_root=str(tmp_path)),
        analyzer=fake_analyzer,
    )

    assert docs
    doc = docs[0]
    assert "#define CPPMEGA_COMMIT_BASE" in doc["text"]
    assert "#define CPPMEGA_COMMIT_WRAP" in doc["text"]
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_INVOCATION)
        for edge in doc["domain_edges"]
    )
    assert any(
        edge["kind"] == int(DomainEdgeKind.MACRO_EXPANSION_INCLUDE_ORDER)
        for edge in doc["domain_edges"]
    )


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


def test_process_record_reuses_identical_file_analysis_with_cache(tmp_path) -> None:
    source = "\n".join(
        [
            "int helper() { return 1; }",
            "int main() { return helper(); }",
            "// enough text to pass the process_record minimum length guard",
        ]
    )
    main = FunctionDef(
        "main",
        "main",
        "src/demo.cc",
        2,
        "int main() { return helper(); }",
        [],
        end_line=2,
    )
    record = {
        "repo": "tests/demo",
        "old_content": source,
        "new_content": source,
        "diff": "\n".join(
            [
                "diff --git a/src/demo.cc b/src/demo.cc",
                "--- a/src/demo.cc",
                "+++ b/src/demo.cc",
                "@@ -2 +2 @@",
                "-int main() { return helper(); }",
                "+int main() { return helper(); }",
            ]
        ),
        "filepath": "src/demo.cc",
        "subject": "cache analysis",
    }
    calls = []

    def analyzer(content, filepath, clang_index, tmpdir, **_kwargs):
        calls.append((content, filepath, tmpdir))
        return FileAnalysis("", functions=[main])

    cache = AnalysisCache(max_entries=8)
    for _ in range(2):
        docs = process_record(
            record,
            clang_index=object(),
            tmpdir=str(tmp_path),
            max_tokens=8192,
            max_file_bytes=10000,
            doc_format="diff",
            max_dep_depth=1,
            analysis_cache=cache,
            analyzer=analyzer,
        )
        assert docs

    assert len(calls) == 1
    assert cache.hits == 3
    assert cache.misses == 1


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


def test_tokenized_materializer_inserts_domain_delimiters_and_routes() -> None:
    text = "add_executable(app main.cpp)\n"
    target_start = text.index("app")
    source_start = text.index("main.cpp")
    role_ids = [0] * len(text)
    role_ids[target_start : target_start + 3] = [int(DomainRoleKind.TARGET)] * 3
    role_ids[source_start : source_start + len("main.cpp")] = [
        int(DomainRoleKind.SOURCE)
    ] * len("main.cpp")
    docs = [
        {
            "text": text,
            "domain_kind": int(DomainKind.CMAKE),
            "domain_ids": [int(DomainKind.CMAKE)] * len(text),
            "domain_role_ids": role_ids,
            "domain_source_doc_ids": [0] * len(text),
            "repo_stable_id": "repo-17",
            "filepath_stable_id": "cmake-file-23",
            "domain_edges": [],
            "build_edges": [
                {
                    "from_char": target_start,
                    "to_char": source_start,
                    "kind": int(DomainEdgeKind.BUILD_TARGET_SOURCE),
                }
            ],
        }
    ]

    row = materialize_tokenized_enriched_batch(docs, load_tokenizer())[0]
    start_id, end_id = delimiter_token_ids(DomainKind.CMAKE)

    assert row[schema.TOKEN_IDS_COLUMN][1] == start_id
    assert row[schema.TOKEN_IDS_COLUMN][-1] == end_id
    assert row[schema.TOKEN_DOMAIN_IDS_COLUMN][1] == int(DomainKind.CMAKE)
    assert row[schema.TOKEN_ROLE_IDS_COLUMN][1] == int(DomainRoleKind.DELIMITER)
    assert row[schema.TOKEN_ROLE_IDS_COLUMN][-1] == int(DomainRoleKind.DELIMITER)
    assert row[schema.TOKEN_BUILD_EDGES_COLUMN]
    assert min(row[schema.TOKEN_SOURCE_DOC_IDS_COLUMN]) > 0
    edge = row[schema.TOKEN_BUILD_EDGES_COLUMN][0]
    assert edge["kind"] == int(DomainEdgeKind.BUILD_TARGET_SOURCE)
    assert row[schema.TOKEN_ROLE_IDS_COLUMN][edge["from"]] == int(DomainRoleKind.TARGET)
    assert row[schema.TOKEN_ROLE_IDS_COLUMN][edge["to"]] == int(DomainRoleKind.SOURCE)


def test_tokenized_materializer_inserts_distinct_meson_delimiters() -> None:
    text = "project('demo', 'cpp', default_options: ['cpp_std=c++23'])\n"
    docs = [
        {
            "text": text,
            "domain_kind": int(DomainKind.MESON),
            "domain_ids": [int(DomainKind.MESON)] * len(text),
            "domain_role_ids": [0] * len(text),
        }
    ]

    row = materialize_tokenized_enriched_batch(docs, load_tokenizer())[0]
    start_id, end_id = delimiter_token_ids(DomainKind.MESON)

    assert row[schema.TOKEN_IDS_COLUMN][1] == start_id
    assert row[schema.TOKEN_IDS_COLUMN][-1] == end_id
    assert row[schema.TOKEN_DOMAIN_IDS_COLUMN][1] == int(DomainKind.MESON)
    assert row[schema.TOKEN_DOMAIN_IDS_COLUMN][-1] == int(DomainKind.MESON)
    assert row[schema.TOKEN_ROLE_IDS_COLUMN][1] == int(DomainRoleKind.DELIMITER)
    assert row[schema.TOKEN_ROLE_IDS_COLUMN][-1] == int(DomainRoleKind.DELIMITER)


def test_tokenized_materializer_inserts_compile_commands_delimiters() -> None:
    text = '[{"file": "src/main.cc", "command": "c++ -std=c++23 src/main.cc"}]\n'
    docs = [
        {
            "text": text,
            "domain_kind": int(DomainKind.COMPILE_COMMANDS),
            "domain_ids": [int(DomainKind.COMPILE_COMMANDS)] * len(text),
            "domain_role_ids": [0] * len(text),
        }
    ]

    row = materialize_tokenized_enriched_batch(docs, load_tokenizer())[0]
    start_id, end_id = delimiter_token_ids(DomainKind.COMPILE_COMMANDS)

    assert row[schema.TOKEN_IDS_COLUMN][1] == start_id
    assert row[schema.TOKEN_IDS_COLUMN][-1] == end_id
    assert row[schema.TOKEN_DOMAIN_IDS_COLUMN][1] == int(DomainKind.COMPILE_COMMANDS)
    assert row[schema.TOKEN_DOMAIN_IDS_COLUMN][-1] == int(DomainKind.COMPILE_COMMANDS)
    assert row[schema.TOKEN_ROLE_IDS_COLUMN][1] == int(DomainRoleKind.DELIMITER)
    assert row[schema.TOKEN_ROLE_IDS_COLUMN][-1] == int(DomainRoleKind.DELIMITER)
