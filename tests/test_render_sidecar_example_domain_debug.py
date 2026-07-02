from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cppmega_mlx.data.domain_schema import DomainKind, DomainRoleKind, ParseConfidence
from cppmega_mlx.data.tokenizer_contract import DOMAIN_DELIMITER_TOKEN_IDS
from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
from scripts import render_sidecar_example as renderer
from scripts.render_sidecar_example import build_sidecar, render_row, to_markdown


class _TinyTokenizer:
    def token_for_id(self, token_id: int) -> str:
        if 191 <= token_id <= 222:
            return f"<RESERVED_{token_id}>"
        return f"T{token_id}"


def test_render_sidecar_debug_maps_domain_delimiters_to_logical_names() -> None:
    ids = [
        DOMAIN_DELIMITER_TOKEN_IDS["CPP_CODE_START"],
        10,
        DOMAIN_DELIMITER_TOKEN_IDS["CPP_CODE_END"],
        DOMAIN_DELIMITER_TOKEN_IDS["CMAKE_START"],
        20,
        DOMAIN_DELIMITER_TOKEN_IDS["CMAKE_END"],
        DOMAIN_DELIMITER_TOKEN_IDS["COMPILER_ERROR_START"],
        30,
        DOMAIN_DELIMITER_TOKEN_IDS["COMPILER_ERROR_END"],
    ]
    row = {
        "token_domain_ids": [
            int(DomainKind.CPP),
            int(DomainKind.CPP),
            int(DomainKind.CPP),
            int(DomainKind.CMAKE),
            int(DomainKind.CMAKE),
            int(DomainKind.CMAKE),
            int(DomainKind.COMPILER_ERROR),
            int(DomainKind.COMPILER_ERROR),
            int(DomainKind.COMPILER_ERROR),
        ],
        "token_role_ids": [
            int(DomainRoleKind.DELIMITER),
            int(DomainRoleKind.IDENTIFIER),
            int(DomainRoleKind.DELIMITER),
            int(DomainRoleKind.DELIMITER),
            int(DomainRoleKind.COMMAND),
            int(DomainRoleKind.DELIMITER),
            int(DomainRoleKind.DELIMITER),
            int(DomainRoleKind.MESSAGE),
            int(DomainRoleKind.DELIMITER),
        ],
        "token_entity_ids": [0] * len(ids),
        "token_scope_ids": [0] * len(ids),
        "token_source_doc_ids": [0] * len(ids),
        "token_confidence_ids": [int(ParseConfidence.EXACT)] * len(ids),
        "token_domain_edges": [{"from": 3, "to": 6, "kind": 90}],
        "token_build_edges": [{"from": 4, "to": 1, "kind": 26}],
        "token_diagnostic_edges": [{"from": 7, "to": 1, "kind": 60}],
    }

    sidecar = build_sidecar(row, ids, _TinyTokenizer(), window=16)
    rendered_tokens = [entry["tok"] for entry in sidecar["per_token"]]

    assert rendered_tokens[0].startswith("<CPP_CODE_START>")
    assert rendered_tokens[3].startswith("<CMAKE_START>")
    assert rendered_tokens[6].startswith("<COMPILER_ERROR_START>")
    assert sidecar["_legend"]["domain_delimiters_by_id"]["191"] == "<CPP_CODE_START>"
    assert sidecar["per_token"][6]["E_domain"] == int(DomainKind.COMPILER_ERROR)
    assert sidecar["per_token"][7]["E_role"] == int(DomainRoleKind.MESSAGE)
    assert sidecar["E_domain_routes"]["token_build_edges_total"] == 1
    assert sidecar["E_domain_routes"]["token_diagnostic_edges"] == [
        {"from": 7, "to": 1, "kind": 60}
    ]


def test_render_row_markdown_shows_mixed_domain_logical_delimiters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tok = load_cppmega_tokenizer(Path("cppmega_mlx/tokenizer"))
    cpp_ids = tok.encode("int add(int a, int b) { return a + b; }\n")
    cmake_ids = tok.encode("add_executable(app main.cpp)\n")
    error_ids = tok.encode("main.cpp:1:1: error: expected ';'\n")
    assert isinstance(cpp_ids, list)
    assert isinstance(cmake_ids, list)
    assert isinstance(error_ids, list)
    ids = (
        [DOMAIN_DELIMITER_TOKEN_IDS["CPP_CODE_START"]]
        + cpp_ids
        + [DOMAIN_DELIMITER_TOKEN_IDS["CPP_CODE_END"]]
        + [DOMAIN_DELIMITER_TOKEN_IDS["CMAKE_START"]]
        + cmake_ids
        + [DOMAIN_DELIMITER_TOKEN_IDS["CMAKE_END"]]
        + [DOMAIN_DELIMITER_TOKEN_IDS["COMPILER_ERROR_START"]]
        + error_ids
        + [DOMAIN_DELIMITER_TOKEN_IDS["COMPILER_ERROR_END"]]
    )
    n = len(ids)
    row = {
        "input_ids": ids,
        "target_ids": ids[1:] + [0],
        "loss_mask": [1] * n,
        "token_domain_ids": [int(DomainKind.CPP)] * n,
        "token_role_ids": [0] * n,
        "token_entity_ids": [0] * n,
        "token_scope_ids": [0] * n,
        "token_source_doc_ids": [0] * n,
        "token_confidence_ids": [int(ParseConfidence.EXACT)] * n,
        "token_diagnostic_edges": [{"from": n - 2, "to": 2, "kind": 60}],
    }
    for pos, token_id in enumerate(ids):
        if token_id in DOMAIN_DELIMITER_TOKEN_IDS.values():
            row["token_role_ids"][pos] = int(DomainRoleKind.DELIMITER)
    cmake_start = ids.index(DOMAIN_DELIMITER_TOKEN_IDS["CMAKE_START"])
    cmake_end = ids.index(DOMAIN_DELIMITER_TOKEN_IDS["CMAKE_END"])
    diag_start = ids.index(DOMAIN_DELIMITER_TOKEN_IDS["COMPILER_ERROR_START"])
    diag_end = ids.index(DOMAIN_DELIMITER_TOKEN_IDS["COMPILER_ERROR_END"])
    row["token_domain_ids"][cmake_start:cmake_end + 1] = [int(DomainKind.CMAKE)] * (
        cmake_end + 1 - cmake_start
    )
    row["token_domain_ids"][diag_start:diag_end + 1] = [
        int(DomainKind.COMPILER_ERROR)
    ] * (diag_end + 1 - diag_start)

    path = tmp_path / "mixed.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    monkeypatch.setattr(renderer, "clang_format", lambda code: (code, True))

    result = render_row(str(path), 0, window=n)
    md = to_markdown(result)

    assert "<CPP_CODE_START>" in md
    assert "<CMAKE_START>" in md
    assert "<COMPILER_ERROR_START>" in md
    assert "add_executable" not in result.formatted_code
    assert "expected" not in result.formatted_code
    assert "int add" in result.formatted_code
