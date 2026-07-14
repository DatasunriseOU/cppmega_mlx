from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_mlx.data.source_identity import source_identity
from cppmega_mlx.data.symbol_identity import compute_symbol_id
from cppmega_mlx.training.megatron_objectives import materialize_megatron_document
from cppmega_mlx.training.objective_mixer import (
    EligibilityAwareTaskMixer,
    RealizedObjective,
)
from cppmega_mlx.training.objectives import build_fim
from cppmega_mlx.training.task_mixer import STAGE1_DEFAULT_RATES, TaskKind
from scripts import materialize_megatron_objectives as materializer
from scripts.nanochat_data.clang_enriched_to_parquet import _SCHEMA


TOKEN_COUNT = 160
CONSTITUENT_TOKEN_COUNT = TOKEN_COUNT // 4
DEPENDENCY_SOURCE_DOC_IDS = [
    constituent for constituent in (1, 2, 3, 4) for _ in range(CONSTITUENT_TOKEN_COUNT)
]
CASE3_FIXTURE = Path(__file__).parent / "fixtures" / "case3_prompt_repo"


def _physical_source(filepath: str):
    return source_identity({"repo": "example/dependency-project", "filepath": filepath})


def _symbol_record(kind: str) -> dict[str, int | str]:
    key = f"example/dependency-project::{kind}"
    return {"symbol_id": compute_symbol_id(key), "symbol_key": key}


def _base_row(*, filepath: str) -> dict[str, object]:
    symbol = _symbol_record("symbol")
    callee = _symbol_record("callee")
    type_ref = _symbol_record("type")
    return {
        "repo": "example/dependency-project",
        "filepath": filepath,
        "commit_hash": "0123456789abcdef",
        "token_ids": list(range(100, 100 + TOKEN_COUNT)),
        "platform_ids": [2, 62],
        "token_structure_ids": [1] * TOKEN_COUNT,
        "token_dep_levels": [
            level for level in range(4) for _ in range(CONSTITUENT_TOKEN_COUNT)
        ],
        "token_ast_depth": [0, 1, 1, 0] * (TOKEN_COUNT // 4),
        "token_sibling_index": [0, 0, 1, 0] * (TOKEN_COUNT // 4),
        "token_ast_node_type": [1, 2, 2, 1] * (TOKEN_COUNT // 4),
        "token_symbol_ids": [int(symbol["symbol_id"])] * 2 + [0] * (TOKEN_COUNT - 2),
        "token_call_targets": [0] * 6
        + [int(callee["symbol_id"])] * 2
        + [0] * (TOKEN_COUNT - 8),
        "token_type_refs": [0] * 10
        + [int(type_ref["symbol_id"])] * 2
        + [0] * (TOKEN_COUNT - 12),
        "token_def_use": [0] * TOKEN_COUNT,
        "token_domain_ids": [1] * TOKEN_COUNT,
        "token_role_ids": [1] * TOKEN_COUNT,
        "token_entity_ids": [0] * TOKEN_COUNT,
        "token_scope_ids": [0] * TOKEN_COUNT,
        "token_confidence_ids": [1] * TOKEN_COUNT,
        "token_change_mask_pre": [],
        "token_change_mask_post": [],
        "token_chunk_starts": [offset * CONSTITUENT_TOKEN_COUNT for offset in range(4)],
        "token_chunk_ends": [
            (offset + 1) * CONSTITUENT_TOKEN_COUNT for offset in range(4)
        ],
        "token_chunk_kinds": [3, 3, 3, 3],
        "token_chunk_dep_levels": [0, 1, 2, 3],
        "token_call_edges": [{"from": 3, "to": 0}],
        "token_type_edges": [],
        "token_domain_edges": [],
        "token_build_edges": [],
        "token_shell_edges": [],
        "token_diagnostic_edges": [],
        "token_cross_domain_edges": [],
        "symbol_identities": [symbol, callee, type_ref],
    }


def _dependency_row() -> tuple[dict[str, object], int]:
    physical = _physical_source("src/dependency.cpp")
    row = _base_row(filepath="src/dependency.cpp")
    row.update(
        {
            "ifim_instruction_token_ids": [700, 701],
            "token_source_doc_ids": DEPENDENCY_SOURCE_DOC_IDS,
            "token_source_identity_ids": [physical.source_identity_id] * TOKEN_COUNT,
            "source_identity_registry": [physical.as_dict()],
        }
    )
    return row, physical.source_identity_id


def _commit_row() -> tuple[dict[str, object], int]:
    physical = _physical_source("src/commit.cpp")
    row = _base_row(filepath="src/commit.cpp")
    row.update(
        {
            "doc_ids": [
                value for value in range(1, TOKEN_COUNT // 2 + 1) for _ in range(2)
            ],
            "token_symbol_ids": [0] * TOKEN_COUNT,
            "token_call_targets": [0] * TOKEN_COUNT,
            "token_type_refs": [0] * TOKEN_COUNT,
            "token_call_edges": [],
            "commit_msg_token_ids": [800, 801],
            "pre_token_ids": [810, 811, 812],
            "post_token_ids": [820, 821, 822],
            "diff_token_ids": [830, 831, 832],
            "token_source_doc_ids": [1] * TOKEN_COUNT,
            "token_source_identity_ids": [physical.source_identity_id] * TOKEN_COUNT,
            "source_identity_registry": [physical.as_dict()],
        }
    )
    return row, physical.source_identity_id


def _fully_eligible_commit_row(*, filepath: str) -> dict[str, object]:
    physical = _physical_source(filepath)
    row = _base_row(filepath=filepath)
    row.update(
        {
            "doc_ids": [1] * TOKEN_COUNT,
            "ifim_instruction_token_ids": [700, 701],
            "commit_msg_token_ids": [800, 801],
            "pre_token_ids": [810, 811, 812],
            "post_token_ids": [820, 821, 822],
            "diff_token_ids": [830, 831, 832],
            "token_source_doc_ids": [1] * TOKEN_COUNT,
            "token_source_identity_ids": [physical.source_identity_id] * TOKEN_COUNT,
            "source_identity_registry": [physical.as_dict()],
        }
    )
    return row


def _write_source_rows(path: Path, rows: list[dict[str, object]]) -> None:
    schema = pa.schema(
        [*_SCHEMA, pa.field("doc_ids", pa.list_(pa.uint32()))],
        metadata=_SCHEMA.metadata,
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _write_actual_case3_code_parquet(path: Path, tmp_path: Path) -> pa.Table:
    from scripts.nanochat_data import clang_enriched_to_parquet as converter
    from tools.clang_indexer import index_project

    docs: list[dict[str, object]] = []
    index_project.process_project(
        str(CASE3_FIXTURE),
        max_tokens=16384,
        parse_workers=1,
        enriched=True,
        emit_doc=docs.append,
        project_id="tests/case3-prompt-repo",
    )
    raw_dir = tmp_path / "actual-case3"
    raw_dir.mkdir()
    jsonl_path = raw_dir / "case3.jsonl"
    raw_parquet_path = raw_dir / "case3.parquet"
    jsonl_path.write_text(
        "".join(json.dumps(doc) + "\n" for doc in docs),
        encoding="utf-8",
    )
    converter.convert_local_jsonl_to_parquet(
        jsonl_path,
        raw_parquet_path,
        tokenizer=converter.load_tokenizer(
            str(
                Path(__file__).parents[1]
                / "cppmega_mlx"
                / "tokenizer"
                / "tokenizer.json"
            )
        ),
        max_tokens=16384,
        overflow_policy="drop",
        materialize_tokenized_enriched=True,
    )
    table = pq.read_table(raw_parquet_path)
    rows = table.to_pylist()
    multi_index = next(
        index
        for index, row in enumerate(rows)
        if row["doc_type"] == "code" and len(set(row["token_source_identity_ids"])) == 4
    )
    header_index = next(
        index for index, row in enumerate(rows) if row["doc_type"] == "code_header"
    )
    build_index = next(
        index for index, row in enumerate(rows) if row["doc_type"] == "build"
    )
    selected = table.take(pa.array([multi_index, header_index, build_index]))
    pq.write_table(selected, path)
    return selected


def _valid_values(table: pa.Table, column: str, row_index: int) -> list[int]:
    valid = int(table["valid_token_count"][row_index].as_py())
    return [int(value) for value in table[column][row_index].as_py()[:valid]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dependency_source_preserves_attention_and_constituent_provenance(
    tmp_path: Path,
) -> None:
    row, physical_id = _dependency_row()
    source_path = tmp_path / "dependency.parquet"
    _write_source_rows(source_path, [row])

    source = next(materializer._iter_sources([str(source_path)], seed=17))
    assert source.code_packet is not None
    packet = source.code_packet

    assert np.asarray(packet.document_ids).tolist() == [1] * TOKEN_COUNT
    assert np.asarray(packet.document_ids).dtype == np.dtype("uint32")
    assert np.asarray(packet.source_doc_ids).tolist() == DEPENDENCY_SOURCE_DOC_IDS
    assert np.asarray(packet.source_doc_ids).dtype == np.dtype("uint32")
    assert (
        np.asarray(packet.source_identity_ids).tolist() == [physical_id] * TOKEN_COUNT
    )
    assert np.asarray(packet.source_identity_ids).dtype == np.dtype("uint64")

    realized = EligibilityAwareTaskMixer(
        {TaskKind.CAUSAL_LM: 1.0}, seed=17
    ).materialize(source, step_index=0)
    document = materialize_megatron_document(
        realized,
        source,
        require_production_sidecars=True,
    )

    assert document.row["doc_ids"] == [1] * TOKEN_COUNT
    assert document.row["token_source_doc_ids"] == DEPENDENCY_SOURCE_DOC_IDS
    assert document.row["token_source_identity_ids"] == [physical_id] * TOKEN_COUNT
    assert document.row["token_change_mask_pre"] == [0] * TOKEN_COUNT
    assert document.row["token_change_mask_post"] == [0] * TOKEN_COUNT


def test_dependency_source_rejects_nonempty_misaligned_change_mask(
    tmp_path: Path,
) -> None:
    row, _physical_id = _dependency_row()
    row["token_change_mask_pre"] = [1, 0]
    source_path = tmp_path / "misaligned-change-mask.parquet"
    _write_source_rows(source_path, [row])
    source = next(materializer._iter_sources([str(source_path)], seed=19))
    realized = EligibilityAwareTaskMixer(
        {TaskKind.CAUSAL_LM: 1.0}, seed=19
    ).materialize(source, step_index=0)

    with pytest.raises(
        ValueError,
        match="typed metadata token_change_mask_pre length 2 .* token length 160",
    ):
        materialize_megatron_document(
            realized,
            source,
            require_production_sidecars=True,
        )


@pytest.mark.parametrize(
    "task",
    (TaskKind.FIM, TaskKind.AST_FIM, TaskKind.IFIM),
)
def test_transformed_dependency_objective_derives_only_uint32_document_identity(
    tmp_path: Path,
    task: TaskKind,
) -> None:
    row, physical_id = _dependency_row()
    source_path = tmp_path / f"{task.value}.parquet"
    _write_source_rows(source_path, [row])
    source = next(materializer._iter_sources([str(source_path)], seed=23))
    realized = EligibilityAwareTaskMixer({task: 1.0}, seed=23).materialize(
        source, step_index=0
    )

    document = materialize_megatron_document(
        realized,
        source,
        require_production_sidecars=True,
    )

    derived_ids = set(document.row["token_source_doc_ids"])
    assert len(derived_ids) == 1
    assert derived_ids.isdisjoint({1, 2, 3, 4})
    assert 0 < next(iter(derived_ids)) < (1 << 32)
    assert set(document.row["token_source_identity_ids"]) == {physical_id}
    assert document.row["source_identity_registry"] == row["source_identity_registry"]


def test_transformed_dependency_objective_rejects_multiple_physical_sources(
    tmp_path: Path,
) -> None:
    row, _physical_id = _dependency_row()
    second = _physical_source("include/dependency.hpp")
    first_registry = list(row["source_identity_registry"])
    row["token_source_identity_ids"] = [
        int(first_registry[0]["source_identity_id"])
    ] * (TOKEN_COUNT // 2) + [second.source_identity_id] * (TOKEN_COUNT // 2)
    row["source_identity_registry"] = [*first_registry, second.as_dict()]
    source_path = tmp_path / "multi-physical.parquet"
    _write_source_rows(source_path, [row])
    source = next(materializer._iter_sources([str(source_path)], seed=29))

    for task in (TaskKind.FIM, TaskKind.AST_FIM, TaskKind.IFIM):
        with pytest.raises(
            ValueError,
            match="no eligible objective.*spans 2 physical source identities",
        ):
            EligibilityAwareTaskMixer({task: 1.0}, seed=29).materialize(
                source, step_index=0
            )
    for task in (
        TaskKind.SYMBOL_RECOVERY,
        TaskKind.TYPE_RECOVERY,
        TaskKind.CALLEE_RECOVERY,
    ):
        with pytest.raises(
            ValueError,
            match="no eligible objective.*whole code packet spans 2 physical",
        ):
            EligibilityAwareTaskMixer({task: 1.0}, seed=29).materialize(
                source, step_index=0
            )

    assert source.code_packet is not None
    realized = RealizedObjective(
        task=TaskKind.FIM,
        example=build_fim(source.code_packet, seed=29),
        ineligible={},
        source_index=0,
    )

    with pytest.raises(ValueError, match="exactly one positive physical source"):
        materialize_megatron_document(
            realized,
            source,
            require_production_sidecars=True,
        )


def test_full_quota_materialization_accepts_mixed_dependency_and_commit_rows(
    tmp_path: Path,
) -> None:
    dependency, dependency_physical_id = _dependency_row()
    commit, _commit_physical_id = _commit_row()
    source_path = tmp_path / "mixed.parquet"
    output_dir = tmp_path / "objectives"
    _write_source_rows(source_path, [dependency, commit])
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.materialize_megatron_objectives",
            "--data-glob",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--samples",
            "60",
            "--seq-len",
            "256",
            "--quota-window-samples",
            "60",
            "--shard-rows",
            "17",
            "--seed",
            "17",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    cli_receipt = json.loads(completed.stdout)
    assert cli_receipt["documents"] == 60

    shard_paths = sorted(output_dir.glob("objectives_*.parquet"))
    assert len(shard_paths) == 4
    tables = [pq.read_table(path) for path in shard_paths]
    table = pa.concat_tables(tables)
    assert table.num_rows == 60
    assert table.schema.field("doc_ids").type == pa.list_(pa.uint32())
    assert table.schema.field("token_source_doc_ids").type == pa.list_(pa.uint32())
    assert table.schema.field("token_source_identity_ids").type == pa.list_(pa.uint64())

    expected_quotas = EligibilityAwareTaskMixer(STAGE1_DEFAULT_RATES, seed=17).quotas(
        60
    )
    assert Counter(table["objective_kind"].to_pylist()) == Counter(
        {task.value: count for task, count in expected_quotas.items()}
    )

    dependency_causal_rows = [
        row_index
        for row_index, kind in enumerate(table["objective_kind"].to_pylist())
        if kind == TaskKind.CAUSAL_LM.value
        and set(_valid_values(table, "token_source_identity_ids", row_index))
        == {dependency_physical_id}
    ]
    assert dependency_causal_rows
    for row_index in dependency_causal_rows:
        assert _valid_values(table, "doc_ids", row_index) == [1] * TOKEN_COUNT
        assert (
            _valid_values(table, "token_source_doc_ids", row_index)
            == DEPENDENCY_SOURCE_DOC_IDS
        )
        assert (
            _valid_values(table, "token_change_mask_pre", row_index)
            == [0] * TOKEN_COUNT
        )
        assert (
            _valid_values(table, "token_change_mask_post", row_index)
            == [0] * TOKEN_COUNT
        )

    for task in (TaskKind.FIM, TaskKind.AST_FIM, TaskKind.IFIM):
        rows = [
            row_index
            for row_index, kind in enumerate(table["objective_kind"].to_pylist())
            if kind == task.value
        ]
        assert rows
        for row_index in rows:
            assert set(
                _valid_values(table, "token_source_identity_ids", row_index)
            ) == {dependency_physical_id}
            derived = set(_valid_values(table, "token_source_doc_ids", row_index))
            assert len(derived) == 1
            assert derived.isdisjoint({1, 2, 3, 4})

    artifact_path = output_dir / "objective_materialization.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    contract_path = output_dir / artifact["objective_contract"]["path"]
    assert artifact["objective_contract"]["file_sha256"] == _sha256(contract_path)
    for shard in artifact["parquet_shards"]:
        assert shard["sha256"] == _sha256(output_dir / shard["path"])
    artifact_payload = dict(artifact)
    artifact_set_sha256 = artifact_payload.pop("artifact_set_sha256")
    canonical = json.dumps(
        artifact_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    assert artifact_set_sha256 == hashlib.sha256(canonical).hexdigest()


def test_actual_case3_multi_physical_mix_materializes_full_quota_via_cli(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "mixed-input"
    input_dir.mkdir()
    code_path = input_dir / "code.parquet"
    commits_path = input_dir / "commits.parquet"
    code = _write_actual_case3_code_parquet(code_path, tmp_path)
    _write_source_rows(
        commits_path,
        [
            _fully_eligible_commit_row(filepath="src/commit-one.cpp"),
            _fully_eligible_commit_row(filepath="src/commit-two.cpp"),
        ],
    )
    output_dir = tmp_path / "objectives"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.materialize_megatron_objectives",
            "--data-glob",
            str(input_dir / "*.parquet"),
            "--output-dir",
            str(output_dir),
            "--samples",
            "60",
            "--seq-len",
            "512",
            "--quota-window-samples",
            "60",
            "--shard-rows",
            "17",
            "--seed",
            "17",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["documents"] == 60
    code_rows = code.to_pylist()
    multi_physical_ids = set(code_rows[0]["token_source_identity_ids"])
    assert len(multi_physical_ids) == 4
    assert code_rows[0].get("doc_ids") is None
    assert len(code_rows[0]["token_change_mask_pre"]) == 0
    assert len(code_rows[0]["token_change_mask_post"]) == 0
    assert all(len(set(row["token_source_identity_ids"])) == 1 for row in code_rows[1:])

    shard_paths = sorted(output_dir.glob("objectives_*.parquet"))
    table = pa.concat_tables([pq.read_table(path) for path in shard_paths])
    kinds = table["objective_kind"].to_pylist()
    expected_quotas = EligibilityAwareTaskMixer(STAGE1_DEFAULT_RATES, seed=17).quotas(
        60
    )
    assert Counter(kinds) == Counter(
        {task.value: count for task, count in expected_quotas.items()}
    )

    multi_causal_rows = []
    for row_index, kind in enumerate(kinds):
        identities = set(_valid_values(table, "token_source_identity_ids", row_index))
        if identities == multi_physical_ids:
            assert kind == TaskKind.CAUSAL_LM.value
            multi_causal_rows.append(row_index)
    assert len(multi_causal_rows) == 12
    for row_index in multi_causal_rows:
        assert (
            _valid_values(table, "token_source_doc_ids", row_index)
            == code_rows[0]["token_source_doc_ids"]
        )
        assert not any(_valid_values(table, "token_change_mask_pre", row_index))
        assert not any(_valid_values(table, "token_change_mask_post", row_index))

    artifact = json.loads(
        (output_dir / "objective_materialization.json").read_text(encoding="utf-8")
    )
    contract_path = output_dir / artifact["objective_contract"]["path"]
    assert artifact["objective_contract"]["file_sha256"] == _sha256(contract_path)
    assert all(
        shard["sha256"] == _sha256(output_dir / shard["path"])
        for shard in artifact["parquet_shards"]
    )
    artifact_payload = dict(artifact)
    artifact_set_sha256 = artifact_payload.pop("artifact_set_sha256")
    canonical = json.dumps(
        artifact_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    assert artifact_set_sha256 == hashlib.sha256(canonical).hexdigest()
