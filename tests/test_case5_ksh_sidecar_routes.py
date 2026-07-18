from __future__ import annotations

from cppmega_mlx.data.domain_ingestion import parse_domain_document
from cppmega_mlx.data.domain_schema import DomainEdgeKind, DomainKind, DomainRoleKind, ParseConfidence
from cppmega_mlx.data.nanochat_pipeline import tokenized_enriched_schema as schema
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched import (
    materialize_tokenized_enriched_batch,
)
from cppmega_mlx.data.tokenizer_contract import DOMAIN_DELIMITER_TOKEN_IDS
from scripts.nanochat_data.token_budget import _slice_doc_char_range


class _Encoding:
    def __init__(self, text: str) -> None:
        # The backend deliberately exposes offsets for normalized text. The
        # materializer must translate those offsets back to the source document.
        self.ids = [1000 + (index % 1000) for index in range(len(text))]
        self.offsets = [(index, index + 1) for index in range(len(text))]


class _OffsetBackend:
    @staticmethod
    def encode_batch(texts, add_special_tokens=False):
        del add_special_tokens
        return [_Encoding(text) for text in texts]


class _OffsetTokenizer:
    _tokenizer = _OffsetBackend()

    @staticmethod
    def get_bos_token_id() -> int:
        return 1


_ROUTE_CASES = (
    {
        "name": "cpp",
        "path": "src/main.cpp",
        "text": 'int run(){ sqlite3_exec(db, "SELECT id FROM jobs", 0, 0, 0); }\n',
        "domain": DomainKind.CPP,
        "char_edges": {
            "domain_edges": [
                {"from_char": 4, "to_char": 11, "kind": int(DomainEdgeKind.CALL)},
                {"from_char": 11, "to_char": 29, "kind": int(DomainEdgeKind.EMBEDDED_DOMAIN)},
            ],
            "cross_domain_edges": [
                {"from_char": 11, "to_char": 29, "kind": int(DomainEdgeKind.EMBEDDED_DOMAIN)}
            ],
        },
        "token_edges": {
            "token_domain_edges": [{"from": 12, "to": 25, "kind": int(DomainEdgeKind.CALL)}],
            "token_cross_domain_edges": [
                {"from": 25, "to": 50, "kind": int(DomainEdgeKind.EMBEDDED_DOMAIN)}
            ],
        },
    },
    {
        "name": "cmake",
        "path": "CMakeLists.txt",
        "text": "add_executable(app main.cpp)\n",
        "domain": DomainKind.CMAKE,
        "char_edges": {
            "build_edges": [
                {"from_char": 0, "to_char": 15, "kind": int(DomainEdgeKind.BUILD_COMMAND_TARGET)},
                {"from_char": 15, "to_char": 19, "kind": int(DomainEdgeKind.BUILD_TARGET_SOURCE)},
            ]
        },
        "token_edges": {
            "token_build_edges": [
                {"from": 2, "to": 17, "kind": int(DomainEdgeKind.BUILD_COMMAND_TARGET)},
                {"from": 17, "to": 27, "kind": int(DomainEdgeKind.BUILD_TARGET_SOURCE)},
            ]
        },
    },
    {
        "name": "make",
        "path": "Makefile",
        "text": "app: main.o\n\t$(CXX) main.o -o app\n",
        "domain": DomainKind.MAKE,
        "char_edges": {
            "build_edges": [
                {"from_char": 0, "to_char": 5, "kind": int(DomainEdgeKind.BUILD_TARGET_DEP)},
                {"from_char": 0, "to_char": 13, "kind": int(DomainEdgeKind.BUILD_RULE_COMMAND)},
            ]
        },
        "token_edges": {
            "token_build_edges": [
                {"from": 2, "to": 13, "kind": int(DomainEdgeKind.BUILD_TARGET_DEP)},
                {"from": 2, "to": 30, "kind": int(DomainEdgeKind.BUILD_RULE_COMMAND)},
            ]
        },
    },
    {
        "name": "ksh",
        "path": "scripts/run.ksh",
        "text": "print input.txt | sed s/x/y/ > out.txt\n",
        "domain": DomainKind.KSH,
        "char_edges": {
            "shell_edges": [
                {"from_char": 0, "to_char": 6, "kind": int(DomainEdgeKind.SHELL_COMMAND_FILE)},
                {"from_char": 0, "to_char": 18, "kind": int(DomainEdgeKind.SHELL_PIPE)},
                {"from_char": 18, "to_char": 22, "kind": int(DomainEdgeKind.SHELL_COMMAND_FILE)},
                {"from_char": 18, "to_char": 31, "kind": int(DomainEdgeKind.SHELL_REDIR_OUT)},
            ]
        },
        "token_edges": {
            "token_shell_edges": [
                {"from": 2, "to": 14, "kind": int(DomainEdgeKind.SHELL_COMMAND_FILE)},
                {"from": 2, "to": 38, "kind": int(DomainEdgeKind.SHELL_PIPE)},
                {"from": 38, "to": 48, "kind": int(DomainEdgeKind.SHELL_COMMAND_FILE)},
                {"from": 38, "to": 69, "kind": int(DomainEdgeKind.SHELL_REDIR_OUT)},
            ]
        },
    },
    {
        "name": "compiler_error",
        "path": "compile.log",
        "text": "src/main.cpp:3:2: error: no member named 'x'\n",
        "domain": DomainKind.COMPILER_ERROR,
        "char_edges": {
            "diagnostic_edges": [{"from_char": 18, "to_char": 0, "kind": int(DomainEdgeKind.DIAG_PRIMARY_LOCATION)}]
        },
        "token_edges": {
            "token_diagnostic_edges": [{"from": 26, "to": 2, "kind": int(DomainEdgeKind.DIAG_PRIMARY_LOCATION)}]
        },
    },
    {
        "name": "linker_error",
        "path": "link.log",
        "text": "ld: error: undefined reference to `missing()'\n",
        "domain": DomainKind.LINKER_ERROR,
        "char_edges": {
            "diagnostic_edges": [{"from_char": 0, "to_char": 35, "kind": int(DomainEdgeKind.LINK_UNDEFINED_SYMBOL)}]
        },
        "token_edges": {
            "token_diagnostic_edges": [{"from": 2, "to": 67, "kind": int(DomainEdgeKind.LINK_UNDEFINED_SYMBOL)}]
        },
    },
    {
        "name": "build_error",
        "path": "build.log",
        "text": "ninja: build stopped: subcommand failed.\n",
        "domain": DomainKind.BUILD_ERROR,
        "char_edges": {
            "diagnostic_edges": [
                {"from_char": 0, "to_char": 5, "kind": int(DomainEdgeKind.DIAG_BUILD_TARGET)},
                {"from_char": 0, "to_char": 20, "kind": int(DomainEdgeKind.DIAG_BUILD_TARGET)},
            ]
        },
        "token_edges": {
            "token_diagnostic_edges": [
                {"from": 2, "to": 7, "kind": int(DomainEdgeKind.DIAG_BUILD_TARGET)},
                {"from": 2, "to": 34, "kind": int(DomainEdgeKind.DIAG_BUILD_TARGET)},
            ]
        },
    },
)


def test_case5_parse_to_token_sidecars_freezes_ksh_and_all_graph_families() -> None:
    enriched_docs = []
    for source_doc_id, case in enumerate(_ROUTE_CASES, start=11):
        parsed = parse_domain_document(
            case["path"],
            case["text"],
            source_doc_id=source_doc_id,
            provenance={"repo": "fixture/case5-routes", "filepath": case["path"]},
        )
        if case["name"] == "cpp":
            # The lexical C++ parser supplies the host/embedded route. Add one
            # ordinary domain edge so the family-pure domain column is exercised
            # in the same parse-to-materialize receipt.
            parsed.add_edge(1, 5, DomainEdgeKind.CALL)
        enriched = parsed.to_enriched_document()
        for field, expected in case["char_edges"].items():
            assert enriched[field] == expected
        enriched_docs.append(enriched)

    rows = materialize_tokenized_enriched_batch(
        enriched_docs,
        _OffsetTokenizer(),
        num_threads=1,
    )

    token_columns = schema.TOKENIZED_ENRICHED_DOMAIN_TOKEN_COLUMNS
    graph_columns = schema.TOKENIZED_ENRICHED_DOMAIN_GRAPH_COLUMNS
    for case, row in zip(_ROUTE_CASES, rows, strict=True):
        token_ids = row[schema.TOKEN_IDS_COLUMN]
        assert token_ids[0] == 1
        assert token_ids[1] == DOMAIN_DELIMITER_TOKEN_IDS[
            f"{case['domain'].name if case['domain'] != DomainKind.CPP else 'CPP_CODE'}_START"
        ]
        assert token_ids[-1] == DOMAIN_DELIMITER_TOKEN_IDS[
            f"{case['domain'].name if case['domain'] != DomainKind.CPP else 'CPP_CODE'}_END"
        ]
        assert row[schema.TOKEN_ROLE_IDS_COLUMN][1] == int(DomainRoleKind.DELIMITER)
        assert row[schema.TOKEN_ROLE_IDS_COLUMN][-1] == int(DomainRoleKind.DELIMITER)
        assert row[schema.TOKEN_CONFIDENCE_IDS_COLUMN][1] == int(ParseConfidence.EXACT)
        assert row[schema.TOKEN_CONFIDENCE_IDS_COLUMN][-1] == int(ParseConfidence.EXACT)

        for column in token_columns:
            assert len(row[column]) == len(token_ids), column
        assert all(value > 0 for value in row[schema.TOKEN_SOURCE_DOC_IDS_COLUMN])
        assert all(value > 0 for value in row[schema.TOKEN_SOURCE_IDENTITY_IDS_COLUMN])

        expected_edges = case["token_edges"]
        for column in graph_columns:
            assert row[column] == expected_edges.get(column, []), column
        assert all(
            len(edge) == 3
            and 0 <= int(edge["from"]) < len(token_ids)
            and 0 <= int(edge["to"]) < len(token_ids)
            for column in graph_columns
            for edge in row[column]
        )

    cpp_row = rows[0]
    sql_start = DOMAIN_DELIMITER_TOKEN_IDS["SQL_START"]
    sql_end = DOMAIN_DELIMITER_TOKEN_IDS["SQL_END"]
    assert sql_start in cpp_row[schema.TOKEN_IDS_COLUMN]
    assert sql_end in cpp_row[schema.TOKEN_IDS_COLUMN]
    assert int(DomainKind.SQL) in cpp_row[schema.TOKEN_DOMAIN_IDS_COLUMN]


def test_case5_ksh_discovery_and_materialization_keep_frozen_delimiter_ids(tmp_path) -> None:
    from cppmega_mlx.data.domain_ingestion import discover_project_domain_files

    path = tmp_path / "scripts" / "release.ksh"
    path.parent.mkdir(parents=True)
    path.write_text("#!/usr/bin/env ksh\nprint ok\n")

    discovered = discover_project_domain_files(tmp_path)
    assert [(item.path.name, item.domain, item.adapter) for item in discovered] == [
        ("release.ksh", DomainKind.KSH, "ksh")
    ]


def test_case5_ksh_token_budget_slice_keeps_edges_family_pure() -> None:
    text = "print input.txt | tee out.txt\n"
    enriched = parse_domain_document(
        "scripts/run.ksh",
        text,
        source_doc_id=17,
        provenance={"repo": "fixture/case5-routes", "filepath": "scripts/run.ksh"},
    ).to_enriched_document()

    assert enriched["domain_edges"] == enriched["shell_edges"]

    sliced = _slice_doc_char_range(enriched, 0, len(text))

    assert sliced["domain_edges"] == []
    assert sliced["shell_edges"] == [
        {"from_char": 0, "to_char": 6, "kind": int(DomainEdgeKind.SHELL_COMMAND_FILE)},
        {"from_char": 0, "to_char": 18, "kind": int(DomainEdgeKind.SHELL_PIPE)},
        {"from_char": 18, "to_char": 22, "kind": int(DomainEdgeKind.SHELL_COMMAND_FILE)},
    ]
