from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cppmega_mlx.data import domain_schema
from cppmega_mlx.data.build_parsers import (
    parse_autoconf,
    parse_automake,
    parse_configure,
)
from cppmega_mlx.data.diagnostic_parsers import (
    parse_clang_diagnostic,
    parse_sanitizer_output,
    parse_test_output,
)
from cppmega_mlx.data.domain_schema import (
    DOMAIN_DELIMITER_ROLES,
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
    delimiter_token_ids,
    normalize_domain_edge_record,
    validate_domain_delimiter_contract,
)
from cppmega_mlx.data.nanochat_pipeline import tokenized_enriched
from cppmega_mlx.data.shell_parsers import (
    parse_bash,
    parse_ksh,
    parse_sh,
    parse_tcsh,
    parse_zsh,
)
from cppmega_mlx.data.tokenizer_contract import DOMAIN_DELIMITER_TOKEN_IDS


def _role_tokens(parsed, role: DomainRoleKind) -> set[str]:
    return {
        token.text
        for token, role_id in zip(parsed.tokens, parsed.role_ids, strict=True)
        if int(role_id) == int(role)
    }


def _assert_parser_vectors_are_token_aligned(parsed) -> None:
    expected = len(parsed.tokens)
    assert len(parsed.domain_ids) == expected
    assert len(parsed.role_ids) == expected
    assert len(parsed.entity_ids) == expected
    assert len(parsed.scope_ids) == expected
    assert len(parsed.source_doc_ids) == expected
    assert len(parsed.source_identity_ids) == expected
    assert len(parsed.confidence_ids) == expected


def test_case5_reserved_delimiter_contract_covers_all_ingested_domains() -> None:
    validate_domain_delimiter_contract()

    required = (
        DomainKind.CPP,
        DomainKind.CMAKE,
        DomainKind.MAKE,
        DomainKind.NINJA,
        DomainKind.BAZEL,
        DomainKind.CONFIGURE,
        DomainKind.AUTOCONF,
        DomainKind.AUTOMAKE,
        DomainKind.BASH,
        DomainKind.SH,
        DomainKind.ZSH,
        DomainKind.TCSH,
        DomainKind.KSH,
        DomainKind.BUILD_DIAGNOSTIC,
        DomainKind.LINKER_DIAGNOSTIC,
        DomainKind.TEST_OUTPUT,
        DomainKind.SANITIZER_OUTPUT,
        DomainKind.SQL,
        DomainKind.PYTHON,
    )
    for domain in required:
        assert domain in DOMAIN_DELIMITER_ROLES
        start_id, end_id = delimiter_token_ids(domain)
        assert start_id != end_id
        assert start_id > 0 and end_id > 0

    delimiter_ids = [
        token_id
        for domain in DOMAIN_DELIMITER_ROLES
        for token_id in delimiter_token_ids(domain)
    ]
    assert len(delimiter_ids) == len(set(delimiter_ids))
    assert set(delimiter_ids) == set(DOMAIN_DELIMITER_TOKEN_IDS.values())


@pytest.mark.parametrize(
    ("path", "text", "domain", "adapter"),
    [
        ("src/main.cpp", "int main() { return 0; }\n", DomainKind.CPP, "cpp-lexical"),
        ("CMakeLists.txt", "add_executable(app main.cpp)\n", DomainKind.CMAKE, "cmake"),
        ("Makefile", "app: main.o\n", DomainKind.MAKE, "make"),
        ("build.ninja", "build app: link main.o\n", DomainKind.NINJA, "ninja"),
        ("BUILD.bazel", "cc_binary(name = \"app\")\n", DomainKind.BAZEL, "bazel-starlark"),
        ("configure", "#!/bin/sh\nexec ./config.status \"$@\"\n", DomainKind.CONFIGURE, "configure-shell"),
        ("configure.ac", "AC_INIT([demo], [1.0])\n", DomainKind.AUTOCONF, "autoconf"),
        ("Makefile.am", "bin_PROGRAMS = demo\n", DomainKind.AUTOMAKE, "automake"),
        ("scripts/run.bash", "declare -A seen=([app]=1)\n", DomainKind.BASH, "bash"),
        ("scripts/run.sh", "name=value; export name\n", DomainKind.SH, "posix-sh"),
        ("scripts/run.zsh", "autoload -Uz compinit\n", DomainKind.ZSH, "zsh"),
        ("scripts/run.tcsh", "setenv CXX clang++\n", DomainKind.TCSH, "tcsh"),
        ("scripts/run.ksh", "typeset name=value\n", DomainKind.KSH, "ksh"),
        (
            "compile.log",
            "src/main.cpp:3:2: warning: unused variable 'x'\n",
            DomainKind.COMPILER_DIAGNOSTIC,
            "clang-diagnostic",
        ),
        ("link.log", "ld: warning: ignoring duplicate libraries: -lm\n", DomainKind.LINKER_DIAGNOSTIC, "linker-diagnostic"),
        ("test.log", "FAILED tests/test_app.py::test_cli - AssertionError\n", DomainKind.TEST_OUTPUT, "test-output"),
        (
            "asan.log",
            "==1==ERROR: AddressSanitizer: heap-use-after-free\n",
            DomainKind.SANITIZER_OUTPUT,
            "sanitizer-output",
        ),
        ("schema.sql", "CREATE TABLE users(id INTEGER);\n", DomainKind.SQL, "sql-lexical"),
        ("module.py", "def answer():\n    return 42\n", DomainKind.PYTHON, "python-ast-tokenize"),
    ],
)
def test_typed_parser_adapter_registry_covers_case5_domains(
    path: str,
    text: str,
    domain: DomainKind,
    adapter: str,
) -> None:
    from cppmega_mlx.data.domain_ingestion import (
        DomainParserAdapter,
        parse_domain_document,
        resolve_domain_parser,
    )

    resolved = resolve_domain_parser(path, text)
    assert isinstance(resolved, DomainParserAdapter)
    assert resolved.domain == domain

    parsed = parse_domain_document(path, text)
    assert parsed.domain == domain
    assert parsed.metadata["parser_adapter"] == adapter
    _assert_parser_vectors_are_token_aligned(parsed)


def test_unrecognized_domain_path_is_explicit_raw_output() -> None:
    from cppmega_mlx.data.domain_ingestion import parse_domain_document

    parsed = parse_domain_document("notes.opaque", "tool-specific opaque payload\n")

    assert parsed.domain == DomainKind.TOOL_OUTPUT
    assert set(parsed.confidence_ids) == {int(ParseConfidence.RAW)}
    assert parsed.metadata["unsupported_syntax"] == "unrecognized_domain_path"


def test_autotools_configure_autoconf_automake_are_distinct_and_raw_on_malformed() -> None:
    from cppmega_mlx.data.domain_ingestion import parse_domain_document

    configure = parse_domain_document("configure", "#!/bin/sh\n./config.status --recheck\n")
    autoconf = parse_domain_document("configure.ac", "AC_INIT([demo], [1.0])\nAC_PROG_CXX\n")
    automake = parse_automake("bin_PROGRAMS = demo\ndemo_SOURCES = main.cpp util.cpp\n")

    assert configure.domain == DomainKind.CONFIGURE
    assert configure.metadata["build_kind"] == "configure"
    assert autoconf.domain == DomainKind.AUTOCONF
    assert automake.domain == DomainKind.AUTOMAKE
    assert "AC_INIT" in _role_tokens(autoconf, DomainRoleKind.COMMAND)
    assert "demo_SOURCES" in _role_tokens(automake, DomainRoleKind.VARIABLE)

    malformed = parse_autoconf("AC_INIT([demo], [1.0]\nAC_OUTPUT(foo\n")
    assert malformed.domain == DomainKind.AUTOCONF
    assert set(np.asarray(malformed.confidence_ids).tolist()) == {int(ParseConfidence.RAW)}
    assert malformed.metadata["unsupported_syntax"] == "malformed_autoconf_macro"

    malformed_automake = parse_automake('bin_PROGRAMS = "demo\n')
    assert malformed_automake.domain == DomainKind.AUTOMAKE
    assert set(malformed_automake.confidence_ids) == {int(ParseConfidence.RAW)}
    assert malformed_automake.metadata["unsupported_syntax"] == "malformed_automake_syntax"

    malformed_configure = parse_configure("#!/bin/sh\nif true; then\necho ok\n")
    assert malformed_configure.domain == DomainKind.CONFIGURE
    assert set(malformed_configure.confidence_ids) == {int(ParseConfidence.RAW)}
    assert malformed_configure.metadata["unsupported_syntax"] == "malformed_configure_shell"


def test_project_domain_discovery_finds_build_shell_and_configure_files(tmp_path) -> None:
    from cppmega_mlx.data.domain_ingestion import discover_project_domain_files

    files = {
        "CMakeLists.txt": "add_executable(app main.cpp)\n",
        "Makefile": "all:\n\t$(MAKE) -C src\n",
        "build.ninja": "rule cc\n  command = clang++ -c $in -o $out\n",
        "BUILD.bazel": "cc_library(name = \"core\", srcs = [\"core.cc\"])\n",
        "configure": "#!/bin/sh\nexec ./config.status \"$@\"\n",
        "configure.ac": "AC_INIT([demo], [1.0])\n",
        "Makefile.am": "bin_PROGRAMS = demo\n",
        "scripts/bootstrap.sh": "#!/bin/sh\necho boot\n",
        "scripts/env.zsh": "#!/bin/zsh\nprint -r -- $path\n",
        "scripts/setup.tcsh": "#!/bin/tcsh\nsetenv CXX clang++\n",
        "scripts/setup.ksh": "#!/bin/ksh\ntypeset CXX=clang++\n",
        "module.py": "def answer():\n    return 42\n",
        "scripts/release": "#!/usr/bin/env bash\nset -euo pipefail\n",
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    discovered = discover_project_domain_files(tmp_path)
    by_path = {item.path.relative_to(tmp_path).as_posix(): item.domain for item in discovered}

    assert by_path["CMakeLists.txt"] == DomainKind.CMAKE
    assert by_path["Makefile"] == DomainKind.MAKE
    assert by_path["build.ninja"] == DomainKind.NINJA
    assert by_path["BUILD.bazel"] == DomainKind.BAZEL
    assert by_path["configure"] == DomainKind.CONFIGURE
    assert by_path["configure.ac"] == DomainKind.AUTOCONF
    assert by_path["Makefile.am"] == DomainKind.AUTOMAKE
    assert by_path["scripts/bootstrap.sh"] == DomainKind.SH
    assert by_path["scripts/env.zsh"] == DomainKind.ZSH
    assert by_path["scripts/setup.tcsh"] == DomainKind.TCSH
    assert by_path["scripts/setup.ksh"] == DomainKind.KSH
    assert by_path["module.py"] == DomainKind.PYTHON
    assert by_path["scripts/release"] == DomainKind.BASH


def test_index_project_discovers_shells_and_keeps_autotools_kinds_distinct(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    files = {
        "configure": "#!/bin/sh\nexec ./config.status \"$@\"\n",
        "configure.ac": "AC_INIT([demo], [1.0])\n",
        "Makefile.am": "bin_PROGRAMS = demo\n",
        "scripts/bootstrap.sh": "#!/bin/sh\necho boot\n",
        "scripts/release": "#!/usr/bin/env bash\nset -euo pipefail\n",
        "scripts/env.zsh": "#!/bin/zsh\nprint -r -- $path\n",
        "scripts/setup.tcsh": "#!/bin/tcsh\nsetenv CXX clang++\n",
        "scripts/setup.ksh": "#!/bin/ksh\ntypeset CXX=clang++\n",
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    assert index_project.classify_build_file("configure") == "configure"
    assert index_project.classify_build_file("configure.ac") == "autoconf"
    assert index_project.classify_build_file("Makefile.am") == "automake"

    found = {
        Path(path).relative_to(tmp_path).as_posix(): kind
        for path, kind in index_project.find_shell_files(str(tmp_path))
    }
    assert found == {
        "scripts/bootstrap.sh": "sh",
        "scripts/env.zsh": "zsh",
        "scripts/release": "bash",
        "scripts/setup.tcsh": "tcsh",
        "scripts/setup.ksh": "ksh",
    }


def test_domain_discovery_chunks_large_and_fails_loud_on_unreadable_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cppmega_mlx.data.domain_ingestion import discover_project_domain_files
    from tools.clang_indexer import index_project

    build_file = tmp_path / "CMakeLists.txt"
    build_file.write_text("x" * 500_001)
    assert [item.path for item in discover_project_domain_files(tmp_path)] == [
        build_file
    ]
    assert index_project.find_build_files(str(tmp_path)) == [
        (str(build_file), "cmake")
    ]

    build_file.write_text("add_library(app app.cpp)\n")
    original_open = Path.open

    def _open(path: Path, *args, **kwargs):
        if path == build_file:
            raise OSError("simulated read failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _open)
    with pytest.raises(OSError, match="failed to read domain input"):
        discover_project_domain_files(tmp_path)


def test_compile_commands_is_compile_db_input_and_build_domain_text(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    cmake_file = tmp_path / "CMakeLists.txt"
    cmake_file.write_text("add_library(app src/app.cpp)\n")
    entries = [
        {
            "directory": str(tmp_path),
            "file": f"src/unit_{index:05d}.cpp",
            "arguments": [
                "clang++",
                "-Iinclude",
                "-std=c++20",
                "-c",
                f"src/unit_{index:05d}.cpp",
            ],
        }
        for index in range(5_000)
    ]
    compile_commands = tmp_path / "compile_commands.json"
    compile_commands.write_text(json.dumps(entries))
    assert compile_commands.stat().st_size > index_project.BUILD_FILE_SIZE_CAP

    assert set(index_project.find_build_files(str(tmp_path))) == {
        (str(cmake_file), "cmake"),
        (str(compile_commands), "compile_commands"),
    }

    compile_db = index_project.load_compile_commands(str(tmp_path))
    assert compile_db is not None
    assert len(compile_db) == len(entries)
    first_source = str(tmp_path / "src" / "unit_00000.cpp")
    assert compile_db[first_source]["build_info"]["source"] == "compile_commands"
    assert "-std=c++20" in compile_db[first_source]["compile_args"]


def test_non_utf8_domain_input_fails_closed_without_lossy_decode(
    tmp_path: Path,
) -> None:
    from cppmega_mlx.data.domain_ingestion import discover_project_domain_files
    from tools.clang_indexer import index_project

    script = tmp_path / "invalid.sh"
    script.write_bytes(b"#!/bin/sh\nprintf '\x81'\n")

    with pytest.raises(ValueError, match="invalid UTF-8 or Windows-1252"):
        discover_project_domain_files(tmp_path)

    with pytest.raises(ValueError, match="invalid UTF-8 or Windows-1252"):
        index_project.emit_build_documents(
            [(str(script), "sh")],
            default_build_info=None,
        )


def test_shell_shebang_overrides_generic_sh_suffix_in_both_discovery_paths(
    tmp_path: Path,
) -> None:
    from cppmega_mlx.data.domain_ingestion import resolve_domain_parser
    from tools.clang_indexer import index_project

    text = "#!/usr/bin/env bash\ndeclare -A seen=([app]=1)\n"
    script = tmp_path / "scripts" / "release.sh"
    script.parent.mkdir(parents=True)
    script.write_text(text)

    assert resolve_domain_parser(script, text).domain == DomainKind.BASH
    assert index_project.find_shell_files(str(tmp_path)) == [(str(script), "bash")]


def test_shell_discovery_accepts_a_single_trailing_nul_terminator(
    tmp_path: Path,
) -> None:
    from cppmega_mlx.data.domain_ingestion import discover_project_domain_files
    from cppmega_mlx.data.domain_schema import DomainKind
    from tools.clang_indexer import index_project

    script = tmp_path / "single.ksh"
    script.write_bytes(b"#!/bin/ksh\nprint ok\0")

    assert index_project.find_shell_files(str(tmp_path)) == [(str(script), "ksh")]
    discovered = discover_project_domain_files(tmp_path)
    assert [(item.path, item.domain) for item in discovered] == [
        (script, DomainKind.KSH)
    ]


def test_process_project_emits_every_discovered_domain_once_with_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.clang_indexer import index_project

    files = {
        "CMakeLists.txt": "add_executable(app main.cpp)\n",
        "configure": "#!/bin/sh\nexec ./config.status \"$@\"\n",
        "scripts/release.sh": "#!/usr/bin/env bash\necho release\n",
        "schema.sql": "CREATE TABLE jobs(id INTEGER PRIMARY KEY);\n",
        "compiler.log": "src/main.cpp:4:2: warning: unused variable 'x'\n",
        "link.log": "ld: undefined reference to `missing_symbol'\n",
        "sanitizer.log": "ERROR: AddressSanitizer: heap-use-after-free\n",
        "test-results.log": "12 passed, 0 failed, 0 errors\n",
        "module.py": "def answer():\n    return 42\n",
    }
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def _libclang_must_not_be_touched(*_args, **_kwargs):
        raise AssertionError("domain-only repositories must not initialize libclang")

    monkeypatch.setattr(index_project, "_configure_libclang", _libclang_must_not_be_touched)
    monkeypatch.setattr(index_project, "load_compile_commands", lambda *_args: None)
    monkeypatch.setattr(
        index_project,
        "detect_build_context",
        lambda *_args: ({}, [], None),
    )
    monkeypatch.setattr(index_project, "get_default_compile_args", lambda *_args: [])
    monkeypatch.setattr(index_project, "check_memory_limit", lambda *_args, **_kwargs: None)

    docs = index_project.process_project(
        str(tmp_path),
        enriched=True,
        project_id="fixtures/case5-domain-ingestion",
    )
    docs_by_path = {Path(str(doc["filepath"])).as_posix(): doc for doc in docs}

    assert set(docs_by_path) == set(files)
    assert len(docs) == len(files)
    expected_domains = {
        "CMakeLists.txt": DomainKind.CMAKE,
        "configure": DomainKind.CONFIGURE,
        "scripts/release.sh": DomainKind.BASH,
        "schema.sql": DomainKind.SQL,
        "compiler.log": DomainKind.COMPILER_DIAGNOSTIC,
        "link.log": DomainKind.LINKER_ERROR,
        "sanitizer.log": DomainKind.SANITIZER_OUTPUT,
        "test-results.log": DomainKind.TEST_OUTPUT,
        "module.py": DomainKind.PYTHON,
    }
    for relative, domain in expected_domains.items():
        doc = docs_by_path[relative]
        assert doc["domain_kind"] == int(domain)
        assert doc["domain_source_doc_ids"]
        assert min(doc["domain_source_doc_ids"]) > 0
        assert doc["domain_source_identity_ids"]
        assert min(doc["domain_source_identity_ids"]) > 0
        assert doc["source_identity_registry"]
        assert len(doc["source_identity_registry"][0]["canonical_sha256"]) == 64


def test_shell_dialects_keep_distinct_adapters_roles_and_metadata() -> None:
    parsed = {
        "bash": parse_bash("declare -A seen=([app]=1)\nprintf '%s\\n' \"${!seen[@]}\"\n"),
        "sh": parse_sh("name=value; export name\n"),
        "zsh": parse_zsh("autoload -Uz compinit\nprint -r -- $path\n"),
        "tcsh": parse_tcsh("setenv CXX clang++\necho $CXX\n"),
        "ksh": parse_ksh("typeset CXX=clang++\nprint $CXX\n"),
    }

    assert parsed["bash"].domain == DomainKind.BASH
    assert parsed["sh"].domain == DomainKind.SH
    assert parsed["zsh"].domain == DomainKind.ZSH
    assert parsed["tcsh"].domain == DomainKind.TCSH
    assert parsed["ksh"].domain == DomainKind.KSH
    assert {doc.metadata["parser_adapter"] for doc in parsed.values()} == {
        "bash",
        "posix-sh",
        "zsh",
        "tcsh",
        "ksh",
    }
    assert "autoload" in _role_tokens(parsed["zsh"], DomainRoleKind.KEYWORD)
    assert "setenv" in _role_tokens(parsed["tcsh"], DomainRoleKind.KEYWORD)
    assert "CXX" in _role_tokens(parsed["tcsh"], DomainRoleKind.ENVIRONMENT)
    assert "typeset" in _role_tokens(parsed["ksh"], DomainRoleKind.KEYWORD)
    assert "declare" in _role_tokens(parsed["bash"], DomainRoleKind.KEYWORD)
    assert "export" in _role_tokens(parsed["sh"], DomainRoleKind.KEYWORD)


def test_shell_syntax_balance_ignores_comments_and_quotes_and_rejects_extra_closers() -> None:
    quoted = parse_sh('printf "%s\\n" "if"\n# for value in ignored\necho ok\n')
    extra_done = parse_sh("echo ok\ndone\n")
    nested = parse_bash(
        "if true; then\n"
        "  for value in one; do echo \"$value\"; done\n"
        "fi\n"
    )
    tcsh_if = parse_tcsh("if (1) then\n  echo ok\nendif\n")

    assert set(quoted.confidence_ids) == {int(ParseConfidence.HEURISTIC)}
    assert "unsupported_syntax" not in quoted.metadata
    assert set(nested.confidence_ids) == {int(ParseConfidence.HEURISTIC)}
    assert set(tcsh_if.confidence_ids) == {int(ParseConfidence.HEURISTIC)}
    assert set(extra_done.confidence_ids) == {int(ParseConfidence.RAW)}
    assert extra_done.metadata["unsupported_syntax"] == "malformed_sh_shell"


def test_warning_build_linker_test_and_sanitizer_diagnostics_keep_severity_metadata() -> None:
    from cppmega_mlx.data.domain_ingestion import parse_domain_document

    warning = parse_clang_diagnostic(
        "src/main.cpp:12:7: warning: unused variable 'x'\n",
        tool="clang",
    )
    assert warning.domain == DomainKind.COMPILER_DIAGNOSTIC
    assert warning.metadata["severity"] == "warning"
    assert warning.metadata["tool"] == "clang"
    assert warning.metadata["stage"] == "compile"

    build = parse_domain_document("build.log", "ninja: build stopped: subcommand failed\n")
    linker = parse_domain_document("link.log", "ld: warning: ignoring duplicate libraries: -lm\n")
    test = parse_domain_document("test.log", "FAILED tests/test_app.py::test_cli - AssertionError\n")
    sanitizer = parse_domain_document(
        "asan.log",
        "==123==ERROR: AddressSanitizer: heap-use-after-free on address 0x1\n",
    )

    assert build.domain == DomainKind.BUILD_ERROR
    assert build.metadata["stage"] == "build"
    assert linker.domain == DomainKind.LINKER_DIAGNOSTIC
    assert linker.metadata["severity"] == "warning"
    assert test.domain == DomainKind.TEST_OUTPUT
    assert test.metadata["severity"] == "failure"
    assert sanitizer.domain == DomainKind.SANITIZER_OUTPUT
    assert sanitizer.metadata["tool"] == "AddressSanitizer"


def test_test_runtime_status_counts_do_not_turn_zero_failures_into_failure() -> None:
    clean = parse_test_output("12 passed, 0 failed, 0 errors in 1.2s\n")
    empty = parse_test_output("0 passed, 0 failed, 0 errors\n")
    status_first = parse_test_output("Tests run: 12, failures: 0, errors: 0\nOK\n")
    failed = parse_test_output("11 passed, 1 failed, 0 errors\n")

    assert clean.metadata["severity"] == "pass"
    assert empty.metadata["severity"] == "pass"
    assert status_first.metadata["severity"] == "pass"
    assert failed.metadata["severity"] == "failure"


def test_gcc_no_column_test_and_sanitizer_locations_keep_diagnostic_links() -> None:
    from cppmega_mlx.data.diagnostic_parsers import parse_gcc_diagnostic

    gcc = parse_gcc_diagnostic("src/main.cpp:12: warning: unused variable 'x'\n")
    test = parse_test_output(
        "FAILED tests/test_app.py::test_cli\n"
        "tests/test_app.py:27: AssertionError: expected 2\n"
    )
    sanitizer = parse_sanitizer_output(
        "==123==ERROR: AddressSanitizer: heap-use-after-free\n"
        "    #0 0x1 in run src/main.cpp:42:9\n"
    )

    assert gcc.domain == DomainKind.COMPILER_DIAGNOSTIC
    assert gcc.metadata["severity"] == "warning"
    assert "src/main.cpp" in _role_tokens(gcc, DomainRoleKind.FILE)
    assert any(edge[2] == int(DomainEdgeKind.DIAG_PRIMARY_LOCATION) for edge in gcc.edges)

    assert test.domain == DomainKind.TEST_OUTPUT
    assert "test_cli" in _role_tokens(test, DomainRoleKind.TEST_NAME)
    assert any(edge[2] == int(DomainEdgeKind.TEST_FAILURE_LOCATION) for edge in test.edges)

    assert sanitizer.domain == DomainKind.SANITIZER_OUTPUT
    assert sanitizer.metadata["tool"] == "AddressSanitizer"
    assert "src/main.cpp" in _role_tokens(sanitizer, DomainRoleKind.FILE)
    assert any(edge[2] == int(DomainEdgeKind.DIAG_PRIMARY_LOCATION) for edge in sanitizer.edges)


def test_embedded_sql_blocks_are_extracted_with_cross_domain_edges() -> None:
    from cppmega_mlx.data.domain_ingestion import (
        extract_embedded_domain_blocks,
        parse_domain_document,
    )

    cpp = (
        "void migrate(sqlite3* db) {\n"
        "  sqlite3_exec(db, R\"SQL(CREATE TABLE users(id INTEGER PRIMARY KEY);)SQL\", nullptr, nullptr, nullptr);\n"
        "}\n"
    )
    blocks = extract_embedded_domain_blocks(cpp, host_domain=DomainKind.CPP)

    assert len(blocks) == 1
    block = blocks[0]
    assert block.domain == DomainKind.SQL
    assert "CREATE" in _role_tokens(block.parsed, DomainRoleKind.KEYWORD)
    assert "users" in _role_tokens(block.parsed, DomainRoleKind.TARGET)
    assert block.cross_domain_edge[2] == int(DomainEdgeKind.EMBEDDED_DOMAIN)
    assert block.parsed.metadata["embedded_in"] == "CPP"

    host = parse_domain_document("src/migrate.cpp", cpp, source_doc_id=17)
    enriched = host.to_enriched_document()
    sql_start = cpp.index("CREATE TABLE")
    assert enriched["domain_ids"][sql_start] == int(DomainKind.SQL)
    assert enriched["domain_source_doc_ids"][sql_start] == 17
    assert enriched["embedded_domain_spans"] == [
        {
            "start": block.start,
            "end": block.end,
            "domain_kind": int(DomainKind.SQL),
        }
    ]
    assert enriched["cross_domain_edges"]
    assert enriched["cross_domain_edges"][0]["kind"] == int(DomainEdgeKind.EMBEDDED_DOMAIN)


def test_embedded_sql_finds_ordinary_database_strings_and_exec_sql() -> None:
    from cppmega_mlx.data.domain_ingestion import extract_embedded_domain_blocks

    cpp = (
        'sqlite3_exec(db, "CREATE TABLE jobs(id INTEGER);", nullptr, nullptr, nullptr);\n'
        "EXEC SQL DELETE FROM jobs WHERE id = 7;\n"
    )
    blocks = extract_embedded_domain_blocks(cpp, host_domain=DomainKind.CPP)

    assert [cpp[block.start : block.end] for block in blocks] == [
        "CREATE TABLE jobs(id INTEGER);",
        "EXEC SQL DELETE FROM jobs WHERE id = 7;",
    ]
    assert all(block.domain == DomainKind.SQL for block in blocks)
    assert all(
        block.cross_domain_edge[2] == int(DomainEdgeKind.EMBEDDED_DOMAIN)
        for block in blocks
    )


def test_embedded_sql_routes_every_adjacent_cpp_string_literal() -> None:
    from cppmega_mlx.data.domain_ingestion import extract_embedded_domain_blocks

    cpp = (
        'sqlite3_exec(db, "CREATE TABLE " /* keep */ "jobs("\n'
        '                    "id INTEGER);", nullptr, nullptr, nullptr);\n'
    )
    blocks = extract_embedded_domain_blocks(cpp, host_domain=DomainKind.CPP)
    assert [cpp[block.start : block.end] for block in blocks] == [
        "CREATE TABLE ",
        "jobs(",
        "id INTEGER);",
    ]
    assert {block.cross_domain_edge[0] for block in blocks} == {
        cpp.index("sqlite3_exec")
    }


def test_raw_sql_literal_only_links_to_the_call_argument_that_contains_it() -> None:
    from cppmega_mlx.data.domain_ingestion import extract_embedded_domain_blocks

    standalone = (
        'sqlite3_exec(db, "SELECT 1;", nullptr, nullptr, nullptr);\n'
        'auto schema = R"SQL(CREATE TABLE jobs(id INTEGER);)SQL";\n'
    )
    blocks = extract_embedded_domain_blocks(standalone, host_domain=DomainKind.CPP)
    raw_block = next(
        block
        for block in blocks
        if standalone[block.start : block.end].startswith("CREATE TABLE")
    )
    assert raw_block.cross_domain_edge[:2] == (raw_block.start, raw_block.start)

    inside_call = (
        'sqlite3_exec(db, R"SQL(CREATE TABLE "jobs"(id INTEGER);)SQL", '
        "nullptr, nullptr, nullptr);\n"
    )
    inside = extract_embedded_domain_blocks(inside_call, host_domain=DomainKind.CPP)
    assert len(inside) == 1
    assert inside[0].cross_domain_edge[0] == inside_call.index("sqlite3_exec")


def test_cpp_lexical_roles_and_embedded_sql_reach_tokenized_sidecars() -> None:
    from tools.clang_indexer import index_project

    class _Encoding:
        def __init__(self, text: str) -> None:
            self.ids = [1000 + (ord(char) % 1000) for char in text]
            self.offsets = [(idx, idx + 1) for idx in range(len(text))]

    class _OffsetBackend:
        def encode_batch(self, texts, add_special_tokens=False):
            del add_special_tokens
            return [_Encoding(text) for text in texts]

    class _OffsetTokenizer:
        _tokenizer = _OffsetBackend()

        @staticmethod
        def get_bos_token_id() -> int:
            return 1

    cpp = (
        "#include <sqlite3.h>\n"
        "/// Apply the schema.\n"
        "void migrate(sqlite3* db) {\n"
        "  sqlite3_exec(db, R\"SQL(CREATE TABLE users(id INTEGER);)SQL\", nullptr, nullptr, nullptr);\n"
        "}\n"
    )
    sidecars = index_project._cpp_domain_sidecars(cpp)
    sql_start = cpp.index("CREATE TABLE")
    assert sidecars["domain_ids"][sql_start] == int(DomainKind.SQL)
    assert int(DomainRoleKind.PREPROCESSOR) in sidecars["domain_role_ids"]
    assert int(DomainRoleKind.DOCSTRING) in sidecars["domain_role_ids"]
    assert sidecars["cross_domain_edges"]
    assert sidecars["cross_domain_edges"][0] in sidecars["domain_edges"]

    row = tokenized_enriched.materialize_tokenized_enriched_batch(
        [{"text": cpp, **sidecars}],
        _OffsetTokenizer(),
        num_threads=1,
    )[0]
    sql_open, sql_close = delimiter_token_ids(DomainKind.SQL)
    assert sql_open in row["token_ids"]
    assert sql_close in row["token_ids"]
    assert any(
        edge["kind"] == int(DomainEdgeKind.EMBEDDED_DOMAIN)
        for edge in row["token_cross_domain_edges"]
    )


def test_parser_output_preserves_provenance_and_token_alignment() -> None:
    from cppmega_mlx.data.domain_ingestion import parse_domain_document

    parsed = parse_domain_document(
        "CMakeLists.txt",
        "add_executable(app main.cpp)\n",
        source_doc_id=73,
        provenance={"repo": "owner/demo", "filepath": "CMakeLists.txt", "commit": "abc"},
        diagnostic_links=[{"diagnostic_id": "diag-1", "line": 9}],
    )
    _assert_parser_vectors_are_token_aligned(parsed)
    assert set(parsed.domain_ids) == {int(DomainKind.CMAKE)}
    assert set(parsed.source_doc_ids) == {73}
    assert parsed.metadata["provenance"] == {
        "repo": "owner/demo",
        "filepath": "CMakeLists.txt",
        "commit": "abc",
    }
    assert parsed.metadata["diagnostic_links"] == [
        {"diagnostic_id": "diag-1", "line": 9}
    ]

    packet = parsed.to_packet(token_ids=list(range(100, 100 + len(parsed.tokens))))
    assert np.asarray(packet.source_doc_ids).tolist() == [73] * len(parsed.tokens)
    assert np.asarray(packet.domain_ids).tolist() == [int(DomainKind.CMAKE)] * len(parsed.tokens)

    enriched = parsed.to_enriched_document()
    for key in (
        "domain_ids",
        "domain_role_ids",
        "domain_entity_ids",
        "domain_scope_ids",
        "domain_source_doc_ids",
        "domain_source_identity_ids",
        "domain_confidence_ids",
    ):
        assert len(enriched[key]) == len(parsed.text), key
    assert enriched["domain_parse_info"]["diagnostic_links"] == [
        {"diagnostic_id": "diag-1", "line": 9}
    ]
    assert enriched["domain_edges"] == enriched["build_edges"]
    assert enriched["build_edges"]


def test_source_identity_is_uint64_with_full_sha256_registry_witness() -> None:
    from cppmega_mlx.data.source_identity import (
        source_identity,
        validate_source_identity_registry,
    )

    long_path = "src/" + "nested/" * 80 + "unit.cpp"
    identity = source_identity({"repo": "owner/repo", "source_path": long_path})
    assert 0 < identity.source_identity_id < (1 << 64)
    assert len(identity.canonical_sha256) == 64
    assert long_path in identity.source
    registry = validate_source_identity_registry(
        [identity.as_dict()],
        referenced_ids=[identity.source_identity_id],
    )
    assert registry[identity.source_identity_id] == identity

    corrupted = identity.as_dict()
    corrupted["canonical_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match canonical source"):
        validate_source_identity_registry([corrupted])

    uppercase = identity.as_dict()
    uppercase["canonical_sha256"] = identity.canonical_sha256.upper()
    with pytest.raises(ValueError, match="does not match canonical source"):
        validate_source_identity_registry([uppercase])


def test_assembled_multifile_doc_preserves_fragment_source_identity() -> None:
    from tools.clang_indexer.index_project import ProjectIndex, build_enriched_doc

    first = "int first();"
    second = "int second();"
    doc = build_enriched_doc(
        [
            (first, 2, 0, "first", None, None, None, "src/first.cpp"),
            (second, 2, 0, "second", None, None, None, "lib/second.cpp"),
        ],
        ProjectIndex(),
        filepath="src/root.cpp",
    )
    second_start = len(first) + 2
    source_doc_ids = doc["domain_source_doc_ids"]
    identity_ids = doc["domain_source_identity_ids"]
    assert source_doc_ids[:second_start] == [1] * second_start
    assert source_doc_ids[second_start:] == [2] * len(second)
    assert len(set(identity_ids[:second_start])) == 1
    assert len(set(identity_ids[second_start:])) == 1
    assert identity_ids[0] != identity_ids[second_start]
    registry_sources = {
        entry["source"] for entry in doc["source_identity_registry"]
    }
    assert any("src/first.cpp" in source for source in registry_sources)
    assert any("lib/second.cpp" in source for source in registry_sources)


def test_domain_edge_routes_are_family_pure_after_legacy_aggregate_ingestion() -> None:
    assert domain_schema.DOMAIN_EDGE_FIELD_FAMILIES["domain_edges"] == "domain"
    assert normalize_domain_edge_record(
        {"from": 0, "to": 1, "kind": int(DomainEdgeKind.BUILD_TARGET_DEP)},
        family="aggregate",
    ) == (0, 1, int(DomainEdgeKind.BUILD_TARGET_DEP))
    build_edge = {
        "from_char": 0,
        "to_char": 1,
        "kind": int(DomainEdgeKind.BUILD_TARGET_DEP),
    }
    canonical = domain_schema.canonicalize_domain_edge_fields(
        {
            "domain_edges": [build_edge, dict(build_edge)],
            "build_edges": [dict(build_edge), dict(build_edge)],
        }
    )
    assert canonical["domain_edges"] == []
    assert canonical["build_edges"] == [build_edge, build_edge]


def test_malformed_domain_edges_and_unknown_domain_kind_fail_closed() -> None:
    with pytest.raises(ValueError, match="missing src/dst/kind"):
        tokenized_enriched._remap_char_edge_triples_to_tokens(
            [{"from": 0, "kind": int(DomainEdgeKind.BUILD_TARGET_SOURCE)}],
            [(0, 1), (1, 2)],
        )

    with pytest.raises(ValueError, match="unknown domain_kind"):
        tokenized_enriched._domain_kind_from_doc({"domain_kind": "not-a-domain"})

    with pytest.raises(ValueError, match="unknown domain edge kind 9999"):
        tokenized_enriched._remap_char_edge_triples_to_tokens(
            [{"from": 0, "to": 1, "kind": 9999}],
            [(0, 1), (1, 2)],
        )

    with pytest.raises(ValueError, match="could not be mapped to token spans"):
        tokenized_enriched._remap_char_edge_triples_to_tokens(
            [{"from": 0, "to": 1, "kind": int(DomainEdgeKind.BUILD_TARGET_DEP)}],
            [],
        )

    with pytest.raises(ValueError, match="outside source text bounds"):
        tokenized_enriched._remap_char_edge_triples_to_tokens(
            [{"from": 0, "to": 3, "kind": int(DomainEdgeKind.BUILD_TARGET_DEP)}],
            [(0, 1), (1, 2), (2, 3)],
            family="build",
            source_length=3,
        )

    with pytest.raises(ValueError, match="not contained in a nonempty token span"):
        tokenized_enriched._remap_char_edge_triples_to_tokens(
            [{"from": 0, "to": 1, "kind": int(DomainEdgeKind.BUILD_TARGET_DEP)}],
            [(0, 1), (2, 3)],
            family="build",
            source_length=3,
        )


def test_delimiter_insertion_keeps_exclusive_chunk_ends_before_closing_marker() -> None:
    from cppmega_mlx.data.nanochat_pipeline import tokenized_enriched_schema as schema

    row = {
        schema.TOKEN_IDS_COLUMN: [10, 11, 12, 13],
        schema.TOKEN_CHUNK_STARTS_COLUMN: [1, 3],
        schema.TOKEN_CHUNK_ENDS_COLUMN: [3, 4],
    }

    tokenized_enriched._insert_domain_delimiter_pair(
        row,
        domain=DomainKind.SQL,
        start_at=1,
        end_at=3,
    )

    sql_start, sql_end = delimiter_token_ids(DomainKind.SQL)
    assert row[schema.TOKEN_IDS_COLUMN] == [10, sql_start, 11, 12, sql_end, 13]
    assert row[schema.TOKEN_DOMAIN_IDS_COLUMN] == [
        0,
        int(DomainKind.SQL),
        int(DomainKind.SQL),
        int(DomainKind.SQL),
        int(DomainKind.SQL),
        0,
    ]
    assert row[schema.TOKEN_ROLE_IDS_COLUMN] == [
        0,
        int(DomainRoleKind.DELIMITER),
        0,
        0,
        int(DomainRoleKind.DELIMITER),
        0,
    ]
    assert row[schema.TOKEN_CHUNK_STARTS_COLUMN] == [2, 5]
    assert row[schema.TOKEN_CHUNK_ENDS_COLUMN] == [4, 6]


def test_packer_rejects_malformed_edges_and_assigns_source_provenance() -> None:
    pytest.importorskip("pyarrow")
    from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
        SOURCE_IDENTITY_REGISTRY_COLUMN,
        TOKEN_BUILD_EDGES_COLUMN,
        TOKEN_IDS_COLUMN,
        TOKEN_SOURCE_DOC_IDS_COLUMN,
        TOKEN_SOURCE_IDENTITY_IDS_COLUMN,
    )
    from scripts.nanochat_data.pack_enriched_rows import (
        normalize_document_record,
        pack_documents,
    )
    from cppmega_mlx.data.source_identity import source_identity

    with pytest.raises(ValueError, match="missing src/dst/kind"):
        normalize_document_record(
            {
                TOKEN_IDS_COLUMN: [10, 11],
                TOKEN_BUILD_EDGES_COLUMN: [{"from": 0, "kind": int(DomainEdgeKind.BUILD_TARGET_DEP)}],
            },
            source_doc_index=0,
        )

    first_identity = source_identity({"source_path": "first.cc"})
    second_identity = source_identity({"source_path": "second.cc"})
    with pytest.raises(ValueError, match="missing exact token source identities"):
        normalize_document_record(
            {
                TOKEN_IDS_COLUMN: [10, 11],
                TOKEN_SOURCE_IDENTITY_IDS_COLUMN: [
                    first_identity.source_identity_id,
                    0,
                ],
                SOURCE_IDENTITY_REGISTRY_COLUMN: [
                    first_identity.as_dict(),
                    second_identity.as_dict(),
                ],
            },
            source_doc_index=0,
        )

    docs = [
        normalize_document_record({TOKEN_IDS_COLUMN: [10, 11]}, source_doc_index=0),
        normalize_document_record({TOKEN_IDS_COLUMN: [20, 21]}, source_doc_index=1),
    ]
    rows, overflow = pack_documents(
        docs,
        target_length=5,
        pad_token_id=0,
        strategy="sequential",
    )
    assert overflow == []
    source_ids = rows[0][TOKEN_SOURCE_DOC_IDS_COLUMN]
    assert source_ids[0] == source_ids[1] > 0
    assert source_ids[2] == source_ids[3] > 0
    assert source_ids[0] == source_ids[2] == 1
    identity_ids = rows[0][TOKEN_SOURCE_IDENTITY_IDS_COLUMN]
    assert identity_ids[0] == identity_ids[1] > 0
    assert identity_ids[2] == identity_ids[3] > 0
    assert identity_ids[0] != identity_ids[2]
    assert source_ids[4] == 0
