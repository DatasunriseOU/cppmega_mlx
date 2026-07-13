"""Typed ingestion registry for build, shell, diagnostic, and embedded domains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Callable, Mapping, Sequence

from cppmega_mlx.data.build_parsers import (
    parse_autoconf,
    parse_automake,
    parse_bazel,
    parse_cmake,
    parse_configure,
    parse_make,
    parse_ninja,
)
from cppmega_mlx.data.build_parsers.base import ParsedDomainDocument, strip_quotes
from cppmega_mlx.data.diagnostic_parsers import (
    parse_build_error,
    parse_clang_diagnostic,
    parse_linker_error,
    parse_sanitizer_output,
    parse_test_output,
)
from cppmega_mlx.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)
from cppmega_mlx.data.shell_parsers import parse_bash, parse_sh, parse_tcsh, parse_zsh
from cppmega_mlx.data.source_identity import (
    MAX_SOURCE_ID,
    stable_source_identity_id,
)


Parser = Callable[[str], ParsedDomainDocument]


@dataclass(frozen=True)
class DomainParserAdapter:
    name: str
    domain: DomainKind
    parser: Parser


@dataclass(frozen=True)
class DiscoveredDomainFile:
    path: Path
    domain: DomainKind
    adapter: str


@dataclass(frozen=True)
class EmbeddedDomainBlock:
    start: int
    end: int
    domain: DomainKind
    parsed: ParsedDomainDocument
    cross_domain_edge: tuple[int, int, int]


_CPP_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cxx", ".c++", ".cp", ".h", ".hh", ".hpp",
    ".hxx", ".h++", ".inl", ".inc", ".ipp", ".tpp", ".txx", ".cu", ".cuh",
}
_CPP_KEYWORDS = {
    "alignas", "alignof", "asm", "auto", "bool", "break", "case", "catch",
    "char", "class", "concept", "const", "consteval", "constexpr", "constinit",
    "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype",
    "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
    "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
    "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept",
    "nullptr", "operator", "override", "private", "protected", "public", "register",
    "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static",
    "static_assert", "static_cast", "struct", "switch", "template", "this",
    "thread_local", "throw", "true", "try", "typedef", "typeid", "typename",
    "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while",
}
_SQL_KEYWORDS = {
    "ADD", "ALTER", "AS", "BEGIN", "BY", "COMMIT", "CREATE", "DELETE", "DROP",
    "EXEC", "EXISTS", "FROM", "GROUP", "HAVING", "INDEX", "INSERT", "INTO", "JOIN", "KEY",
    "LIMIT", "NOT", "NULL", "ON", "OR", "ORDER", "PRIMARY", "REFERENCES", "ROLLBACK",
    "PRAGMA", "SELECT", "SET", "SQL", "TABLE", "TRIGGER", "UNION", "UNIQUE", "UPDATE", "VALUES", "VIEW",
    "WHERE", "WITH",
}
_SQL_TARGET_PREDECESSORS = {"TABLE", "INTO", "UPDATE", "FROM", "JOIN", "VIEW", "INDEX"}


def _tokens_overlapping(
    parsed: ParsedDomainDocument,
    start: int,
    end: int,
) -> list[int]:
    return [
        idx
        for idx, token in enumerate(parsed.tokens)
        if token.start < end and token.end > start
    ]


def parse_cpp_lexical(text: str) -> ParsedDomainDocument:
    doc = ParsedDomainDocument.new(
        domain=DomainKind.CPP,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"parser_adapter": "cpp-lexical", "language": "c-cpp"},
    )
    for idx, token in enumerate(doc.tokens):
        value = strip_quotes(token.text)
        if value in _CPP_KEYWORDS:
            doc.set_role(idx, DomainRoleKind.KEYWORD)
        elif token.text.startswith(('"', "'")):
            doc.set_role(idx, DomainRoleKind.STRING)
        elif value.isidentifier():
            doc.set_role(idx, DomainRoleKind.IDENTIFIER)

    comment_spans: list[tuple[int, int, DomainRoleKind]] = []
    for pattern, flags, role in (
        (r"(?m)//(?!/)[^\n]*$", 0, DomainRoleKind.COMMENT),
        (r"/\*(?!\*).*?\*/", re.DOTALL, DomainRoleKind.COMMENT),
        (r"(?m)///[^\n]*$", 0, DomainRoleKind.DOCSTRING),
        (r"/\*\*.*?\*/", re.DOTALL, DomainRoleKind.DOCSTRING),
        (r"(?m)^[ \t]*#.*$", 0, DomainRoleKind.PREPROCESSOR),
    ):
        for match in re.finditer(pattern, text, flags):
            comment_spans.append((match.start(), match.end(), role))
    for start, end, role in comment_spans:
        for idx in _tokens_overlapping(doc, start, end):
            doc.set_role(idx, role)
    return doc


def parse_sql_lexical(text: str) -> ParsedDomainDocument:
    doc = ParsedDomainDocument.new(
        domain=DomainKind.SQL,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"parser_adapter": "sql-lexical"},
    )
    previous_keyword: str | None = None
    for idx, token in enumerate(doc.tokens):
        value = strip_quotes(token.text)
        upper = value.upper()
        if upper in _SQL_KEYWORDS:
            doc.set_role(idx, DomainRoleKind.KEYWORD)
            previous_keyword = upper
        elif token.text.startswith(('"', "'")):
            doc.set_role(idx, DomainRoleKind.STRING)
        elif value.isidentifier():
            if previous_keyword in _SQL_TARGET_PREDECESSORS:
                doc.set_role(idx, DomainRoleKind.TARGET, entity=idx + 1)
            else:
                doc.set_role(idx, DomainRoleKind.IDENTIFIER)
            previous_keyword = None
        elif value not in {"(", ")", ",", ";", "."}:
            previous_keyword = None
    for quote in ('"', "'"):
        if len(re.findall(rf"(?<!\\){re.escape(quote)}", text)) % 2:
            return doc.mark_raw("malformed_sql_literal")
    return doc


def _parse_raw_output(text: str) -> ParsedDomainDocument:
    return ParsedDomainDocument.new(
        domain=DomainKind.TOOL_OUTPUT,
        text=text,
        confidence=ParseConfidence.RAW,
        metadata={
            "parser_adapter": "raw-output",
            "unsupported_syntax": "unrecognized_domain_path",
            "raw_reason": "unrecognized_domain_path",
        },
    )


_ADAPTERS = {
    "cpp": DomainParserAdapter("cpp-lexical", DomainKind.CPP, parse_cpp_lexical),
    "cmake": DomainParserAdapter("cmake", DomainKind.CMAKE, parse_cmake),
    "make": DomainParserAdapter("make", DomainKind.MAKE, parse_make),
    "ninja": DomainParserAdapter("ninja", DomainKind.NINJA, parse_ninja),
    "bazel": DomainParserAdapter("bazel-starlark", DomainKind.BAZEL, parse_bazel),
    "configure": DomainParserAdapter("configure-shell", DomainKind.CONFIGURE, parse_configure),
    "autoconf": DomainParserAdapter("autoconf", DomainKind.AUTOCONF, parse_autoconf),
    "automake": DomainParserAdapter("automake", DomainKind.AUTOMAKE, parse_automake),
    "bash": DomainParserAdapter("bash", DomainKind.BASH, parse_bash),
    "sh": DomainParserAdapter("posix-sh", DomainKind.SH, parse_sh),
    "zsh": DomainParserAdapter("zsh", DomainKind.ZSH, parse_zsh),
    "tcsh": DomainParserAdapter("tcsh", DomainKind.TCSH, parse_tcsh),
    "compiler": DomainParserAdapter(
        "clang-diagnostic", DomainKind.COMPILER_DIAGNOSTIC, parse_clang_diagnostic
    ),
    "linker": DomainParserAdapter(
        "linker-diagnostic", DomainKind.LINKER_DIAGNOSTIC, parse_linker_error
    ),
    "build-diagnostic": DomainParserAdapter(
        "build-diagnostic", DomainKind.BUILD_DIAGNOSTIC, parse_build_error
    ),
    "test": DomainParserAdapter("test-output", DomainKind.TEST_OUTPUT, parse_test_output),
    "sanitizer": DomainParserAdapter(
        "sanitizer-output", DomainKind.SANITIZER_OUTPUT, parse_sanitizer_output
    ),
    "sql": DomainParserAdapter("sql-lexical", DomainKind.SQL, parse_sql_lexical),
    "raw": DomainParserAdapter("raw-output", DomainKind.TOOL_OUTPUT, _parse_raw_output),
}


def _shell_kind_from_shebang(text: str) -> str | None:
    first_line = text.lstrip().splitlines()[0] if text.strip() else ""
    if not first_line.startswith("#!"):
        return None
    words = re.findall(r"[A-Za-z0-9_+.-]+", first_line.lower())
    for kind in ("tcsh", "csh", "zsh", "bash", "sh"):
        if kind in words:
            return "tcsh" if kind == "csh" else kind
    return None


def resolve_domain_parser(path: str | Path, text: str = "") -> DomainParserAdapter:
    """Resolve one parser using exact path contracts, then content signatures."""

    path_obj = Path(path)
    name = path_obj.name
    lower_name = name.lower()
    suffix = path_obj.suffix.lower()

    if name == "CMakeLists.txt":
        return _ADAPTERS["cmake"]
    if name in {"Makefile", "GNUmakefile", "makefile"}:
        return _ADAPTERS["make"]
    if name == "build.ninja":
        return _ADAPTERS["ninja"]
    if name in {"BUILD", "BUILD.bazel", "WORKSPACE", "WORKSPACE.bazel", "MODULE.bazel"}:
        return _ADAPTERS["bazel"]
    if name == "configure":
        return _ADAPTERS["configure"]
    if name in {"configure.ac", "configure.in"} or suffix == ".m4":
        return _ADAPTERS["autoconf"]
    if name in {"Makefile.am", "Makefile.in"}:
        return _ADAPTERS["automake"]

    shebang_kind = _shell_kind_from_shebang(text)
    if shebang_kind is not None:
        return _ADAPTERS[shebang_kind]

    if suffix in _CPP_EXTENSIONS:
        return _ADAPTERS["cpp"]
    if suffix == ".cmake":
        return _ADAPTERS["cmake"]
    if suffix == ".mk":
        return _ADAPTERS["make"]
    if suffix == ".ninja":
        return _ADAPTERS["ninja"]
    if suffix == ".bzl":
        return _ADAPTERS["bazel"]
    if suffix in {".bash"}:
        return _ADAPTERS["bash"]
    if suffix == ".sh":
        return _ADAPTERS["sh"]
    if suffix == ".zsh":
        return _ADAPTERS["zsh"]
    if suffix in {".tcsh", ".csh"}:
        return _ADAPTERS["tcsh"]
    if suffix == ".sql":
        return _ADAPTERS["sql"]

    lower = text.lower()
    if any(tool.lower() in lower for tool in (
        "AddressSanitizer", "LeakSanitizer", "MemorySanitizer",
        "ThreadSanitizer", "UndefinedBehaviorSanitizer",
    )):
        return _ADAPTERS["sanitizer"]
    if lower_name.startswith(("test", "pytest")) or re.search(r"(?m)^(FAILED|PASSED)\s+\S+::", text):
        return _ADAPTERS["test"]
    if lower_name.startswith(("link", "ld.")) or re.search(
        r"(?im)^(?:ld|lld|link(?:\.exe)?):|undefined reference|unresolved external symbol",
        text,
    ):
        return _ADAPTERS["linker"]
    if re.search(
        r"(?m)^[^:\n]+:\d+:(?:\d+:)?\s*(?:fatal error|error|warning|note):",
        text,
    ):
        return _ADAPTERS["compiler"]
    if lower_name.startswith("build") or any(
        marker in lower for marker in ("ninja:", "cmake error", "build stopped")
    ):
        return _ADAPTERS["build-diagnostic"]
    return _ADAPTERS["raw"]


_CPP_RAW_SQL_RE = re.compile(
    r'R"(?P<tag>[A-Za-z0-9_]{0,16})\((?P<body>.*?)\)(?P=tag)"',
    re.DOTALL,
)
_SQL_PREFIX_RE = re.compile(
    r"^\s*(?:CREATE|ALTER|INSERT|UPDATE|DELETE|SELECT|DROP|WITH|PRAGMA|BEGIN)\b",
    re.IGNORECASE,
)
_SQL_CALL_RE = re.compile(
    r"\b(?P<call>sqlite3_exec|sqlite3_prepare(?:_v2)?|PQexec|PQprepare|mysql_query)\s*\(",
)
_SQL_CALL_QUERY_ARGUMENT = {
    "sqlite3_exec": 1,
    "sqlite3_prepare": 1,
    "sqlite3_prepare_v2": 1,
    "PQexec": 1,
    "PQprepare": 2,
    "mysql_query": 1,
}
_EXEC_SQL_RE = re.compile(r"(?im)\bEXEC[ \t]+SQL\b[^;]*;")


def _call_argument_spans(text: str, open_paren: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    argument_start = open_paren + 1
    depth = 0
    quote: str | None = None
    escaped = False
    idx = argument_start
    while idx < len(text):
        char = text[idx]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\" and quote != "'":
                escaped = True
            elif char == quote:
                quote = None
        elif text.startswith('R"', idx):
            raw_literal = _CPP_RAW_SQL_RE.match(text, idx)
            if raw_literal is None:
                return []
            idx = raw_literal.end()
            continue
        elif text.startswith("//", idx):
            newline = text.find("\n", idx + 2)
            idx = len(text) if newline < 0 else newline
            continue
        elif text.startswith("/*", idx):
            comment_end = text.find("*/", idx + 2)
            if comment_end < 0:
                return []
            idx = comment_end + 2
            continue
        elif char in {'"', "'"}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            if char == ")" and depth == 0:
                spans.append((argument_start, idx))
                return spans
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            spans.append((argument_start, idx))
            argument_start = idx + 1
        idx += 1
    return []


def _sql_call_query_spans(text: str) -> list[tuple[int, int, int]]:
    query_spans: list[tuple[int, int, int]] = []
    for call_match in _SQL_CALL_RE.finditer(text):
        argument_spans = _call_argument_spans(text, call_match.end() - 1)
        query_index = _SQL_CALL_QUERY_ARGUMENT[call_match.group("call")]
        if query_index >= len(argument_spans):
            continue
        argument_start, argument_end = argument_spans[query_index]
        query_spans.append((argument_start, argument_end, call_match.start("call")))
    return query_spans


def extract_embedded_domain_blocks(
    text: str,
    *,
    host_domain: DomainKind,
    source_doc_id: int | None = None,
) -> list[EmbeddedDomainBlock]:
    """Extract exact host-character spans for supported embedded languages."""

    if DomainKind(host_domain) != DomainKind.CPP:
        return []
    resolved_source_id = (
        stable_source_identity_id({"text": text})
        if source_doc_id in (None, 0)
        else int(source_doc_id)
    )
    if not 0 < resolved_source_id <= MAX_SOURCE_ID:
        raise ValueError(f"source_doc_id must be positive uint32, got {resolved_source_id}")
    candidates: list[tuple[int, int, int]] = []
    query_call_spans = _sql_call_query_spans(text)
    for match in _CPP_RAW_SQL_RE.finditer(text):
        body = match.group("body")
        tag = match.group("tag").upper()
        if "SQL" not in tag and not _SQL_PREFIX_RE.match(body):
            continue
        start, end = match.span("body")
        owner = next(
            (
                call_start
                for argument_start, argument_end, call_start in query_call_spans
                if argument_start <= match.start() and match.end() <= argument_end
            ),
            None,
        )
        candidates.append((start, end, start if owner is None else owner))

    for argument_start, argument_end, call_anchor in query_call_spans:
        argument = text[argument_start:argument_end]
        for literal in re.finditer(r'"(?P<body>(?:\\.|[^"\\])*)"', argument):
            body = literal.group("body")
            if not _SQL_PREFIX_RE.match(body):
                continue
            start = argument_start + literal.start("body")
            end = argument_start + literal.end("body")
            candidates.append((start, end, call_anchor))

    for match in _EXEC_SQL_RE.finditer(text):
        candidates.append((match.start(), match.end(), match.start()))

    blocks: list[EmbeddedDomainBlock] = []
    previous_end = -1
    for start, end, call_anchor in sorted(set(candidates)):
        if start < previous_end:
            continue
        body = text[start:end]
        parsed = parse_sql_lexical(body)
        parsed.set_source_doc_id(resolved_source_id)
        parsed.metadata["embedded_in"] = host_domain.name
        blocks.append(
            EmbeddedDomainBlock(
                start=start,
                end=end,
                domain=DomainKind.SQL,
                parsed=parsed,
                cross_domain_edge=(
                    call_anchor,
                    start,
                    int(DomainEdgeKind.EMBEDDED_DOMAIN),
                ),
            )
        )
        previous_end = end
    return blocks


def parse_domain_document(
    path: str | Path,
    text: str,
    *,
    source_doc_id: int | None = None,
    provenance: Mapping[str, object] | None = None,
    diagnostic_links: Sequence[Mapping[str, object]] | None = None,
) -> ParsedDomainDocument:
    """Parse one file and enforce token-aligned domain/provenance vectors."""

    adapter = resolve_domain_parser(path, text)
    resolved_source_id = (
        stable_source_identity_id(
            {
                **dict(provenance or {}),
                "source_path": str(path),
                "text": text,
            }
        )
        if source_doc_id in (None, 0)
        else int(source_doc_id)
    )
    if not 0 < resolved_source_id <= MAX_SOURCE_ID:
        raise ValueError(f"source_doc_id must be positive uint32, got {resolved_source_id}")
    parsed = adapter.parser(text)
    if parsed.domain != adapter.domain:
        allowed_dynamic = {
            "clang-diagnostic": {DomainKind.COMPILER_DIAGNOSTIC, DomainKind.COMPILER_ERROR},
            "linker-diagnostic": {DomainKind.LINKER_DIAGNOSTIC, DomainKind.LINKER_ERROR},
            "build-diagnostic": {DomainKind.BUILD_DIAGNOSTIC, DomainKind.BUILD_ERROR},
        }
        if parsed.domain not in allowed_dynamic.get(adapter.name, set()):
            raise ValueError(
                f"parser adapter {adapter.name} returned {parsed.domain.name}, "
                f"expected {adapter.domain.name}"
            )
    parsed.metadata["parser_adapter"] = adapter.name
    parsed.metadata["source_path"] = str(path)
    parsed.set_source_doc_id(resolved_source_id)
    if provenance is not None:
        parsed.metadata["provenance"] = dict(provenance)
    if diagnostic_links is not None:
        parsed.metadata["diagnostic_links"] = [dict(link) for link in diagnostic_links]
    if parsed.domain == DomainKind.CPP:
        parsed.embedded_blocks = extract_embedded_domain_blocks(
            text,
            host_domain=DomainKind.CPP,
            source_doc_id=resolved_source_id,
        )
    parsed.validate()
    return parsed


def discover_project_domain_files(
    root: str | Path,
    *,
    extra_exclude_dirs: set[str] | None = None,
    include_cpp: bool = True,
) -> list[DiscoveredDomainFile]:
    """Discover supported project-domain files without treating every file as raw."""

    root_path = Path(root)
    skip_dirs = {".git"} | (extra_exclude_dirs or set())
    discovered: list[DiscoveredDomainFile] = []
    candidate_paths: list[Path] = []
    for directory, dirs, filenames in os.walk(root_path):
        dirs[:] = sorted(name for name in dirs if name not in skip_dirs)
        candidate_paths.extend(Path(directory) / name for name in sorted(filenames))
    for path in candidate_paths:
        try:
            if path.stat().st_size > 500_000:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        adapter = resolve_domain_parser(path, text)
        if adapter.name == "raw-output" or (
            not include_cpp and adapter.domain == DomainKind.CPP
        ):
            continue
        discovered.append(
            DiscoveredDomainFile(path=path, domain=adapter.domain, adapter=adapter.name)
        )
    return discovered


__all__ = [
    "DiscoveredDomainFile",
    "DomainParserAdapter",
    "EmbeddedDomainBlock",
    "discover_project_domain_files",
    "extract_embedded_domain_blocks",
    "parse_cpp_lexical",
    "parse_domain_document",
    "parse_sql_lexical",
    "resolve_domain_parser",
]
