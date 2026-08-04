from __future__ import annotations

from pathlib import Path

import pytest

from cppmega_mlx.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched import (
    materialize_tokenized_enriched_batch,
)


def _role_tokens(parsed, role: DomainRoleKind) -> set[str]:
    return {
        token.text
        for token, role_id in zip(parsed.tokens, parsed.role_ids, strict=True)
        if int(role_id) == int(role)
    }


def test_ksh_parser_keeps_domain_and_shell_graph_edges() -> None:
    from cppmega_mlx.data.shell_parsers import parse_ksh

    parsed = parse_ksh("typeset name=value\nprint $name | sed 's/x/y/' > out.txt\n")

    assert parsed.domain == DomainKind.KSH
    assert parsed.metadata["parser_adapter"] == "ksh"
    assert "typeset" in _role_tokens(parsed, DomainRoleKind.KEYWORD)
    assert "print" in _role_tokens(parsed, DomainRoleKind.COMMAND)
    assert "sed" in _role_tokens(parsed, DomainRoleKind.COMMAND)
    assert "out.txt" in _role_tokens(parsed, DomainRoleKind.PATH)
    assert any(
        edge[2] == int(DomainEdgeKind.SHELL_PIPE) for edge in parsed.edges
    )
    assert any(
        edge[2] == int(DomainEdgeKind.SHELL_REDIR_OUT) for edge in parsed.edges
    )

    from cppmega_mlx.data.agent_trajectory import parse_shell_action_domain

    assert parse_shell_action_domain("print ok", shell_kind="ksh").domain == DomainKind.KSH


def test_python_parser_uses_stdlib_ast_and_tokenize_for_typed_spans_and_edges() -> None:
    from cppmega_mlx.data.python_parsers import parse_python

    source = (
        '"""module docs"""\n'
        "# module comment\n"
        "def greet(name: str) -> str:\n"
        "    message = f\"Hello {name}\"\n"
        "    print(message)\n"
        "    return message\n"
    )
    parsed = parse_python(source)

    assert parsed.domain == DomainKind.PYTHON
    assert parsed.metadata["parser_adapter"] == "python-ast-tokenize"
    assert "def" in _role_tokens(parsed, DomainRoleKind.KEYWORD)
    assert "greet" in _role_tokens(parsed, DomainRoleKind.IDENTIFIER)
    assert "# module comment" in _role_tokens(parsed, DomainRoleKind.COMMENT)
    assert any(
        parsed.text[token.start : token.end] == '"""module docs"""'
        and role_id == int(DomainRoleKind.DOCSTRING)
        for token, role_id in zip(parsed.tokens, parsed.role_ids, strict=True)
    )
    assert set(parsed.confidence_ids) == {int(ParseConfidence.EXACT)}
    assert any(
        edge[2] == int(DomainEdgeKind.AST_PARENT) for edge in parsed.edges
    )
    assert any(edge[2] == int(DomainEdgeKind.CALL) for edge in parsed.edges)
    assert any(edge[2] == int(DomainEdgeKind.DEF_USE) for edge in parsed.edges)

    enriched = parsed.to_enriched_document()
    assert enriched["domain_kind"] == int(DomainKind.PYTHON)
    assert len(enriched["domain_ids"]) == len(source)
    assert enriched["domain_edges"]


def test_python_parser_marks_syntax_errors_raw_without_losing_domain_identity() -> None:
    from cppmega_mlx.data.python_parsers import parse_python

    parsed = parse_python("def broken(:\n    pass\n")

    assert parsed.domain == DomainKind.PYTHON
    assert set(parsed.confidence_ids) == {int(ParseConfidence.RAW)}
    assert parsed.metadata["unsupported_syntax"] == "malformed_python_syntax"


def test_python_parser_maps_utf8_ast_columns_to_character_spans() -> None:
    from cppmega_mlx.data.python_parsers import parse_python

    source = "def café(value: str):\n    return value\n"
    parsed = parse_python(source)
    cafe_start = source.index("café")

    cafe_tokens = [
        token
        for token in parsed.tokens
        if token.text == "café"
    ]
    assert len(cafe_tokens) == 1
    assert cafe_tokens[0].start == cafe_start
    assert cafe_tokens[0].end == cafe_start + len("café")
    assert parsed.confidence_ids
    assert set(parsed.confidence_ids) == {int(ParseConfidence.EXACT)}


def test_domain_dispatch_and_discovery_cover_ksh_and_python(tmp_path: Path) -> None:
    from cppmega_mlx.data.domain_ingestion import (
        discover_project_domain_files,
        parse_domain_document,
        resolve_domain_parser,
    )

    ksh_text = "#!/bin/ksh\nprint ok\n"
    python_text = "def answer():\n    return 42\n"
    ksh_path = tmp_path / "script.ksh"
    python_path = tmp_path / "module.py"
    shebang_path = tmp_path / "run"
    ksh_path.write_text(ksh_text)
    python_path.write_text(python_text)
    shebang_path.write_text("#!/usr/bin/env python3\nprint(42)\n")

    assert resolve_domain_parser(ksh_path, ksh_text).domain == DomainKind.KSH
    assert resolve_domain_parser(python_path, python_text).domain == DomainKind.PYTHON
    assert resolve_domain_parser(shebang_path, shebang_path.read_text()).domain == DomainKind.PYTHON
    assert parse_domain_document(python_path, python_text).domain == DomainKind.PYTHON

    discovered = {
        item.path.relative_to(tmp_path).as_posix(): item.domain
        for item in discover_project_domain_files(tmp_path)
    }
    assert discovered == {
        "module.py": DomainKind.PYTHON,
        "run": DomainKind.PYTHON,
        "script.ksh": DomainKind.KSH,
    }


def test_postgres_mule_internal_fixture_round_trips_byte_exactly(
    tmp_path: Path,
) -> None:
    from cppmega_mlx.data.domain_ingestion import (
        decode_domain_prefix,
        discover_project_domain_files,
        iter_domain_file_chunks,
    )

    encoded = (
        b"-- MULE \x92 internal encoding fixture\n"
        b"SELECT '\x81' AS byte_value;\n"
    )
    path = tmp_path / "src/test/mb/sql/mule_internal.sql"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    discovered = discover_project_domain_files(tmp_path)
    decoded_prefix = decode_domain_prefix(encoded, path=path)
    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=17))

    assert [item.path for item in discovered] == [path]
    assert decoded_prefix.encode("latin-1") == encoded
    assert {chunk.source_encoding for chunk in chunks} == {"mule-internal"}
    assert b"".join(chunk.text.encode("latin-1") for chunk in chunks) == encoded
    assert all(chunk.byte_end - chunk.byte_start <= 17 for chunk in chunks)


@pytest.mark.parametrize(
    ("relative_path", "payload", "error"),
    [
        (
            "src/test/mb/sql/not_mule_internal.sql",
            b"-- near miss \x92\nSELECT '\x81';\n",
            "invalid UTF-8 or Windows-1252",
        ),
        (
            "src/test/mb/sql/mule_internal.sql",
            b"-- MULE \x92\nSELECT '\x81\x00';\n",
            "NUL byte",
        ),
    ],
)
def test_postgres_mule_internal_contract_fails_closed(
    tmp_path: Path,
    relative_path: str,
    payload: bytes,
    error: str,
) -> None:
    from cppmega_mlx.data.domain_ingestion import iter_domain_file_chunks

    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=error):
        list(iter_domain_file_chunks(path))


class _Encoding:
    def __init__(self, text: str) -> None:
        self.ids = [1000 + (index % 1000) for index in range(len(text))]
        self.offsets = [(index, index + 1) for index in range(len(text))]


class _CharOffsetBackend:
    @staticmethod
    def encode_batch(texts, add_special_tokens=False):
        del add_special_tokens
        return [_Encoding(text) for text in texts]


class _CharOffsetTokenizer:
    _tokenizer = _CharOffsetBackend()

    @staticmethod
    def get_bos_token_id() -> int:
        return 1


def _materialize_domain_row(domain: DomainKind) -> dict:
    text = "print(42)\n"
    return {
        "repo": "fixture/domain-contract",
        "filepath": "script.py",
        "text": text,
        "domain_kind": int(domain),
        "domain_ids": [int(domain)] * len(text),
        "domain_role_ids": [0] * len(text),
        "domain_entity_ids": [0] * len(text),
        "domain_scope_ids": [0] * len(text),
        "domain_source_doc_ids": [1] * len(text),
        "domain_confidence_ids": [int(ParseConfidence.EXACT)] * len(text),
    }


def test_materialization_inserts_new_delimiters_and_preserves_old_row_slots() -> None:
    new_row = materialize_tokenized_enriched_batch(
        [_materialize_domain_row(DomainKind.PYTHON)],
        _CharOffsetTokenizer(),
        num_threads=1,
    )[0]
    old_row = materialize_tokenized_enriched_batch(
        [_materialize_domain_row(DomainKind.TCSH)],
        _CharOffsetTokenizer(),
        num_threads=1,
    )[0]

    assert new_row["token_ids"][1] == 247
    assert new_row["token_ids"][-1] == 248
    assert old_row["token_ids"][1] == 207
    assert old_row["token_ids"][-1] == 208
    assert old_row["token_domain_ids"][1] == int(DomainKind.TCSH)


def test_index_project_materializes_ksh_and_python_as_typed_domain_docs(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    ksh_path = tmp_path / "run.ksh"
    ksh_path.write_text("#!/bin/ksh\nprint ok\n")

    assert index_project.find_shell_files(str(tmp_path)) == [
        (str(ksh_path), "ksh")
    ]

    doc = index_project.build_build_doc(
        "module.py",
        "def answer():\n    return 42\n",
        "python",
        project_id="fixture/domain-contract",
    )
    assert doc["domain_kind"] == int(DomainKind.PYTHON)
    assert doc["doc_type"] == "code"
    assert doc["domain_parse_info"]["parser"] == "python-ast-tokenize"
    assert doc["language_info"]["primary_language"] == "python"
