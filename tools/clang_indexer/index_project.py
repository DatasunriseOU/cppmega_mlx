#!/usr/bin/env python3
# ruff: noqa: E402
"""
Clang-based cross-file dependency indexer for C++ training data preparation.

Uses libclang to parse C++ translation units with full semantic analysis,
building a cross-file call graph and generating bottom-up training documents.

Architecture:
  1. Walk project directory, find all .cpp/.cc/.cxx/.c files
  2. Parse each with libclang (optionally using compile_commands.json)
  3. Extract functions, classes, and cross-file call references
  4. Build global call graph
  5. Topological sort: HAL/system → drivers → subsystems → API
  6. Generate 16K-token training documents with bottom-up dependency ordering

Usage:
  # With compile_commands.json (best quality):
  python index_project.py --project-dir /path/to/project --output chunks.jsonl

  # Without build system (fallback mode):
  python index_project.py --project-dir /path/to/project --output chunks.jsonl --no-compile-db

  # Process multiple projects in parallel:
  python index_project.py --projects-list projects.txt --output chunks.jsonl --workers 48
"""

import argparse
import ctypes.util
import glob
import importlib
import json
import os
import sys
import hashlib

# Increase recursion limit for deeply nested ASTs (gcc-mirror, llvm-project, boost)
sys.setrecursionlimit(50000)
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Protocol, TypeAlias, cast

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega_mlx.data.nanochat_pipeline.language_info import detect_language_info
from cppmega_mlx.data.nanochat_pipeline.build_context import (
    detect_build_context,
    find_compile_commands_file,
    load_compile_commands_file,
)
from scripts.nanochat_data.memory_guard import check_memory_limit, start_memory_guard

if TYPE_CHECKING:
    import clang.cindex as clang_cindex  # pyright: ignore[reportMissingImports]
    from clang.cindex import Config as ClangConfig  # pyright: ignore[reportMissingImports]
    from clang.cindex import Cursor, CursorKind, Index, TranslationUnit  # pyright: ignore[reportMissingImports]
else:
    clang_cindex = None
    ClangConfig = object
    Cursor = object
    CursorKind = object
    Index = object
    TranslationUnit = object


class _MissingCursorKind:
    def __getattr__(self, name: str) -> str:
        return f"<missing-clang-cursorkind:{name}>"


class _MissingIndex:
    @staticmethod
    def create():
        raise ImportError(
            "libclang Python bindings not found. Install with: pip install libclang"
        )


class _ClangCIndexModule(Protocol):
    __file__: str


class _ClangConfig(Protocol):
    @staticmethod
    def set_library_file(filename: str) -> None: ...


class _IndexFactory(Protocol):
    @staticmethod
    def create() -> "Index": ...


class PlatformDetectFn(Protocol):
    def __call__(self, text: str) -> dict[str, object] | None: ...


DetectLanguageInfoFn = Callable[
    [str, str | None, dict[str, object] | None, list[str] | None, dict | None],
    dict[str, object] | None,
]


clang_cindex_module: _ClangCIndexModule | None
clang_config_cls: _ClangConfig
index_cls: _IndexFactory
_CLANG_IMPORT_ERROR: ImportError | None = None

try:
    clang_cindex_runtime = importlib.import_module("clang.cindex")
    clang_cindex_module = cast(_ClangCIndexModule, clang_cindex_runtime)
    clang_config_cls = cast(_ClangConfig, getattr(clang_cindex_runtime, "Config"))
    index_cls = cast(_IndexFactory, getattr(clang_cindex_runtime, "Index"))
    if not TYPE_CHECKING:
        clang_cindex = clang_cindex_runtime  # type: ignore[assignment]
        ClangConfig = getattr(clang_cindex_runtime, "Config")  # type: ignore[assignment]
        Cursor = getattr(clang_cindex_runtime, "Cursor")  # type: ignore[assignment]
        CursorKind = getattr(clang_cindex_runtime, "CursorKind")  # type: ignore[assignment]
        Index = getattr(clang_cindex_runtime, "Index")  # type: ignore[assignment]
        TranslationUnit = getattr(clang_cindex_runtime, "TranslationUnit")  # type: ignore[assignment]
except ImportError as _clang_import_err:
    _CLANG_IMPORT_ERROR = _clang_import_err
    clang_cindex_module = None
    clang_config_cls = cast(_ClangConfig, object)
    index_cls = cast(_IndexFactory, _MissingIndex)
    if not TYPE_CHECKING:
        CursorKind = _MissingCursorKind()  # type: ignore[assignment]
        Index = _MissingIndex  # type: ignore[assignment]


_LIBCLANG_CONFIGURED = False
_COMMON_LIBCLANG_CANDIDATES = [
    "/usr/lib/llvm-21/lib/libclang-21.so.1",
    "/usr/lib/llvm-21/lib/libclang.so.1",
    "/usr/lib/llvm-20/lib/libclang-20.so.1",
    "/usr/lib/llvm-20/lib/libclang.so.1",
    "/usr/lib/llvm-19/lib/libclang-19.so.1",
    "/usr/lib/llvm-19/lib/libclang.so.1",
    "/usr/lib/llvm-18/lib/libclang-18.so.1",
    "/usr/lib/llvm-18/lib/libclang.so.1",
    "/lib/x86_64-linux-gnu/libclang-18.so.18",
    "/lib/x86_64-linux-gnu/libclang-18.so",
    "/lib/x86_64-linux-gnu/libclang.so.1",
    "/lib/x86_64-linux-gnu/libclang.so",
    "/usr/lib/libclang.so.1",
    "/usr/lib/libclang.so",
]


def _iter_bundled_libclang_candidates() -> Iterator[str]:
    if clang_cindex_module is None:
        return
    module_path = getattr(clang_cindex_module, "__file__", "")
    if not module_path:
        return
    native_dir = Path(module_path).resolve().parent / "native"
    if not native_dir.is_dir():
        return
    for pattern in ("libclang.so*", "libclang.dylib*", "libclang.dll*"):
        for candidate in sorted(native_dir.glob(pattern)):
            if candidate.is_file():
                yield str(candidate)


def _iter_libclang_candidates(explicit_path: str | None = None) -> Iterator[str]:
    seen: set[str] = set()

    def add(value: str | None) -> str | None:
        if not value:
            return None
        candidate = value.strip()
        if not candidate or candidate in seen:
            return None
        seen.add(candidate)
        return candidate

    candidate = add(explicit_path)
    if candidate:
        yield candidate
    for env_name in (
        "NANOCHAT_LIBCLANG_PATH",
        "LIBCLANG_PATH",
        "LIBCLANG_FILE",
        "CLANG_LIBRARY_FILE",
    ):
        candidate = add(os.environ.get(env_name))
        if candidate:
            yield candidate
    for candidate in _iter_bundled_libclang_candidates():
        normalized = add(candidate)
        if normalized:
            yield normalized
    for candidate in _COMMON_LIBCLANG_CANDIDATES:
        if os.path.exists(candidate):
            normalized = add(candidate)
            if normalized:
                yield normalized
    detected = add(ctypes.util.find_library("clang"))
    if detected:
        yield detected
    for pattern in (
        "/usr/lib*/**/libclang*.so*",
        "/usr/local/lib*/**/libclang*.so*",
        "/lib*/**/libclang*.so*",
    ):
        for candidate in glob.iglob(pattern, recursive=True):
            if os.path.isfile(candidate):
                normalized = add(candidate)
                if normalized:
                    yield normalized


def _configure_libclang(libclang_path: str | None = None) -> str | None:
    global _LIBCLANG_CONFIGURED
    if _LIBCLANG_CONFIGURED:
        return os.environ.get("NANOCHAT_LIBCLANG_PATH") or libclang_path
    if _CLANG_IMPORT_ERROR is not None:
        raise ImportError(
            "libclang Python bindings not found. Install with: pip install libclang"
        ) from _CLANG_IMPORT_ERROR

    last_error: Exception | None = None
    for candidate in _iter_libclang_candidates(libclang_path):
        try:
            clang_config_cls.set_library_file(candidate)
            index_cls.create()
        except Exception as exc:  # pragma: no cover - runtime-specific
            last_error = exc
            continue
        os.environ["NANOCHAT_LIBCLANG_PATH"] = candidate
        _LIBCLANG_CONFIGURED = True
        return candidate

    try:
        index_cls.create()
    except Exception as exc:
        last_error = exc
    else:
        _LIBCLANG_CONFIGURED = True
        return None

    message = (
        "Unable to load libclang. "
        "Set --libclang-path or NANOCHAT_LIBCLANG_PATH to an absolute libclang.so path."
    )
    if last_error is not None:
        raise RuntimeError(f"{message} Last error: {last_error}") from last_error
    raise RuntimeError(message)

PartInfo: TypeAlias = tuple[str, int, int, str, str | None]


# C++ source file extensions
CPP_EXTENSIONS = {'.cpp', '.cc', '.cxx', '.c', '.c++', '.cp'}
HEADER_EXTENSIONS = {'.h', '.hpp', '.hxx', '.hh', '.h++', '.inl', '.inc'}
# Files we feed to libclang for indexing. Headers are included so that struct/
# class/enum/typedef DEFINITIONS (which live in headers) enter the index as
# type-def chunks, enabling function->type-def ``type_edges``. Headers are parsed
# as standalone TUs; libclang resolves the definitions they contain.
INDEX_EXTENSIONS = CPP_EXTENSIONS | HEADER_EXTENSIONS

# --------------------------------------------------------------------------- #
# Build / compilation file discovery (ADDITIVE; C/C++ path unchanged).
#
# Build files are ingested as their OWN 'build' doc type (distinct from code):
# full tokenized text + a 'build' structure kind + a build-system language tag +
# a platform sidecar where derivable. They carry NO call/type/symbol graph
# (build files are not C++), exactly like commit docs on the code channels.
#
# BUILD_NAME_KINDS maps an EXACT basename -> build-system tag (the language tag).
# BUILD_EXT_KINDS maps a lowercase extension -> build-system tag. A file is a
# build file if its basename hits BUILD_NAME_KINDS OR its extension hits
# BUILD_EXT_KINDS (basename wins on conflict).
# --------------------------------------------------------------------------- #
BUILD_NAME_KINDS: dict[str, str] = {
    # CMake
    "CMakeLists.txt": "cmake",
    # Make
    "Makefile": "make",
    "GNUmakefile": "make",
    "makefile": "make",
    # Autotools
    "configure": "autotools",
    "configure.ac": "autotools",
    "configure.in": "autotools",
    "Makefile.am": "autotools",
    "Makefile.in": "autotools",
    # Bazel
    "BUILD": "bazel",
    "BUILD.bazel": "bazel",
    "WORKSPACE": "bazel",
    "WORKSPACE.bazel": "bazel",
    "MODULE.bazel": "bazel",
    # Meson
    "meson.build": "meson",
    "meson_options.txt": "meson",
    # Ninja
    "build.ninja": "ninja",
    # compile_commands
    "compile_commands.json": "compile_commands",
    # Conan / vcpkg
    "conanfile.txt": "conan",
    "conanfile.py": "conan",
    "vcpkg.json": "vcpkg",
    # Docker (build env)
    "Dockerfile": "dockerfile",
}
BUILD_EXT_KINDS: dict[str, str] = {
    ".cmake": "cmake",
    ".mk": "make",
    ".m4": "autotools",
    ".bzl": "bazel",
    ".ninja": "ninja",
    ".vcxproj": "msvc",
    ".sln": "msvc",
}
# Cap the build slice to a sane share of corpus tokens so a giant build tree can
# never dominate the LM corpus. Applied at discovery (see find_build_files).
BUILD_FILE_SIZE_CAP = 500_000  # bytes; mirror find_cpp_files' per-file cap


def classify_build_file(fname: str) -> str | None:
    """Return the build-system tag for ``fname`` or None if it is not a build file.

    Exact-basename match (CMakeLists.txt, Makefile, BUILD, ...) takes priority
    over extension match (.cmake, .mk, .bzl, ...). conanfile.* is matched by the
    explicit names above; no silent wildcard guessing (RULE #1: ONE clear path).
    """
    if fname in BUILD_NAME_KINDS:
        return BUILD_NAME_KINDS[fname]
    ext = os.path.splitext(fname)[1].lower()
    if ext in BUILD_EXT_KINDS:
        return BUILD_EXT_KINDS[ext]
    return None


# System/stdlib function prefixes (skip for dependency tracking)
SYSTEM_PREFIXES = (
    'std::', 'boost::', '__builtin', '__', 'operator', 'printf', 'fprintf',
    'sprintf', 'snprintf', 'scanf', 'malloc', 'calloc', 'realloc', 'free',
    'memcpy', 'memmove', 'memset', 'memcmp', 'strlen', 'strcpy', 'strcat',
    'strcmp', 'fopen', 'fclose', 'fread', 'fwrite', 'exit', 'abort',
    'assert', 'pthread_', 'EXPECT_', 'ASSERT_', 'TEST',
)


class FunctionDef:
    """A function definition with its source location and call references."""
    __slots__ = ['name', 'qualified_name', 'file', 'line', 'end_line', 'text',
                 'callees', 'referenced_types', 'dep_level', 'is_definition',
                 'ast_depth', 'sibling_index', 'ast_node_type']

    def __init__(self, name: str, qualified_name: str, file: str, line: int,
                 text: str, callees: list, is_definition: bool = True,
                 end_line: int = 0, ast_depth: list[int] | None = None,
                 sibling_index: list[int] | None = None,
                 ast_node_type: list[int] | None = None,
                 referenced_types: list | None = None):
        self.name = name
        self.qualified_name = qualified_name
        self.file = file
        self.line = line
        self.end_line = end_line or (line + text.count('\n'))
        self.text = text
        self.callees = callees  # list of qualified names called
        # qualified names of record/enum/typedef types referenced by this
        # function (params, return, locals, member access) -- captured during the
        # SAME libclang parse as callees so it round-trips IPC and feeds the
        # offline type_refs/type_edges builders. Mirrors `callees`.
        self.referenced_types = list(referenced_types or [])
        self.dep_level = 0
        self.is_definition = is_definition
        self.ast_depth = list(ast_depth or [])
        self.sibling_index = list(sibling_index or [])
        self.ast_node_type = list(ast_node_type or [])

    def to_dict(self) -> dict:
        """Serialize for multiprocessing IPC."""
        return {
            'name': self.name, 'qualified_name': self.qualified_name,
            'file': self.file, 'line': self.line, 'text': self.text,
            'callees': self.callees, 'is_definition': self.is_definition,
            'end_line': self.end_line,
            'ast_depth': self.ast_depth,
            'sibling_index': self.sibling_index,
            'ast_node_type': self.ast_node_type,
            'referenced_types': self.referenced_types,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'FunctionDef':
        return cls(**d)


class TypeDef:
    """A type DEFINITION (struct/class/enum/union/typedef/using).

    Captured during the same libclang parse as functions so the offline doc
    builder can emit a type-def CHUNK whose qname == the type's qualified name.
    That chunk is the *target* of a function->type ``type_edge`` (the source is
    the function chunk's ``referenced_types``). Definitions usually live in
    headers, which is why headers are now part of the index file set.
    """
    __slots__ = ['name', 'qualified_name', 'file', 'line', 'end_line', 'text',
                 'kind']

    def __init__(self, name: str, qualified_name: str, file: str, line: int,
                 text: str, kind: int, end_line: int = 0):
        self.name = name
        self.qualified_name = qualified_name
        self.file = file
        self.line = line
        self.end_line = end_line or (line + text.count('\n'))
        self.text = text
        # node-bucket kind for parts_info (4=class/struct, 7=typedef/using)
        self.kind = kind

    def to_dict(self) -> dict:
        return {
            'name': self.name, 'qualified_name': self.qualified_name,
            'file': self.file, 'line': self.line, 'text': self.text,
            'kind': self.kind, 'end_line': self.end_line,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TypeDef':
        return cls(**d)


class ProjectIndex:
    """Cross-file function index for a single project."""

    def __init__(self):
        # qualified_name -> FunctionDef (definitions only)
        self.functions: dict[str, FunctionDef] = {}
        # file -> list of function qualified_names defined there
        self.file_functions: dict[str, list[str]] = defaultdict(list)
        # file -> preamble text (includes, typedefs, forward decls)
        self.file_preambles: dict[str, str] = {}
        # qualified_name -> list of qualified_names that call it
        self.callers: dict[str, list[str]] = defaultdict(list)
        # qualified_name -> TypeDef (struct/class/enum/union/typedef/using).
        # Separate from `functions` so the call graph is untouched; consulted
        # when building docs to pull a referenced type's DEFINITION in as a
        # type-edge target chunk.
        self.typedefs: dict[str, TypeDef] = {}

    def add_typedef(self, td: TypeDef):
        """Register a type definition. First definition with text wins; never
        overwrite a real definition with a shorter/forward one."""
        key = td.qualified_name
        if not key:
            return
        existing = self.typedefs.get(key)
        if existing is not None and len(existing.text) >= len(td.text):
            return
        self.typedefs[key] = td

    def add_function(self, func: FunctionDef):
        """Add a function definition to the index."""
        key = func.qualified_name
        if key in self.functions and self.functions[key].is_definition:
            return  # don't overwrite definitions with declarations
        self.functions[key] = func
        if func.is_definition:
            self.file_functions[func.file].append(key)

    def build_reverse_edges(self):
        """Build caller -> callee reverse edges for dep level computation."""
        self.callers.clear()
        for qname, func in self.functions.items():
            for callee in func.callees:
                if callee in self.functions:
                    self.callers[callee].append(qname)

    def compute_dep_levels(self):
        """Compute dependency levels via BFS from leaves."""
        # Find leaves: functions with no callees in the index
        in_degree = {}
        for qname, func in self.functions.items():
            local_callees = [c for c in func.callees if c in self.functions and c != qname]
            in_degree[qname] = len(local_callees)

        queue = deque()
        for qname, deg in in_degree.items():
            if deg == 0:
                self.functions[qname].dep_level = 0
                queue.append(qname)

        self.build_reverse_edges()

        while queue:
            qname = queue.popleft()
            level = self.functions[qname].dep_level
            for caller_name in self.callers.get(qname, []):
                new_level = level + 1
                if new_level > self.functions[caller_name].dep_level:
                    self.functions[caller_name].dep_level = new_level
                in_degree[caller_name] -= 1
                if in_degree[caller_name] == 0:
                    queue.append(caller_name)

        # Handle cycles
        max_level = max((f.dep_level for f in self.functions.values()), default=0)
        for qname, deg in in_degree.items():
            if deg > 0:
                self.functions[qname].dep_level = max_level + 1

    def stats(self) -> dict:
        """Return index statistics."""
        return {
            'total_functions': len(self.functions),
            'total_files': len(self.file_functions),
            'definitions': sum(1 for f in self.functions.values() if f.is_definition),
            'max_dep_level': max((f.dep_level for f in self.functions.values()), default=0),
        }


def is_system_function(name: str | None) -> bool:
    """Check if a function name looks like a system/stdlib function."""
    if not isinstance(name, str) or not name:
        return False
    return any(name.startswith(p) for p in SYSTEM_PREFIXES)


_DROP_CLANG_ARG_PREFIXES = (
    "-fsanitize=",
    "-fno-sanitize=",
    "-fsanitize-recover=",
    "-fno-sanitize-recover=",
    "-fsanitize-trap=",
    "-fsanitize-ignorelist=",
    "-fsanitize-blacklist=",
    "-fsanitize-coverage=",
)

_DROP_CLANG_ARGS = {
    "-fsanitize-address-use-after-scope",
    "-fno-sanitize-address-use-after-scope",
    "-shared-libasan",
    "-static-libasan",
    "-shared-libsan",
    "-static-libsan",
}


def _is_runtime_only_sanitizer_flag(text: str) -> bool:
    if text in _DROP_CLANG_ARGS:
        return True
    if text.startswith(("-shared-lib", "-static-lib")) and "san" in text:
        return True
    return False


def _sanitize_compile_args_for_clang(args: list[str] | None) -> list[str]:
    """Drop nullish and runtime-only compile args that break libclang parsing."""
    if not args:
        return []
    sanitized: list[str] = []
    for arg in args:
        if arg is None:
            continue
        text = str(arg).strip()
        if not text:
            continue
        if _is_runtime_only_sanitizer_flag(text):
            continue
        if any(text.startswith(prefix) for prefix in _DROP_CLANG_ARG_PREFIXES):
            continue
        sanitized.append(text)
    return sanitized


def get_function_text(cursor: Cursor, tu: TranslationUnit) -> str:
    """Extract the source text for a cursor's extent."""
    extent = cursor.extent
    start = extent.start
    end = extent.end

    try:
        filename = start.file.name if start.file else None
        if not filename or not os.path.exists(filename):
            return ""

        with open(filename, 'r', errors='replace') as f:
            content = f.read()

        # Convert offsets
        start_offset = start.offset
        end_offset = end.offset
        if start_offset < len(content) and end_offset <= len(content):
            return content[start_offset:end_offset]
    except Exception:
        pass
    return ""


def _byte_to_char_mapper(source: str) -> Callable[[int], int]:
    source_bytes = source.encode("utf-8", errors="replace")
    byte_len = len(source_bytes)
    if byte_len == len(source):
        return lambda byte_offset: max(0, min(byte_offset, len(source)))

    byte_to_char = [0] * (byte_len + 1)
    byte_offset = 0
    for char_idx, ch in enumerate(source):
        ch_len = len(ch.encode("utf-8", errors="replace"))
        for inner in range(ch_len):
            if byte_offset + inner < byte_len:
                byte_to_char[byte_offset + inner] = char_idx
        byte_offset += ch_len
    byte_to_char[byte_len] = len(source)

    def _map(byte_offset: int) -> int:
        if byte_offset <= 0:
            return 0
        if byte_offset >= byte_len:
            return len(source)
        return byte_to_char[byte_offset]

    return _map


def _cursor_extent_offsets(
    cursor: Cursor,
    filename: str,
    byte_to_char: Callable[[int], int],
) -> tuple[int, int] | None:
    extent = cursor.extent
    start = extent.start
    end = extent.end
    start_file = start.file.name if start.file else None
    end_file = end.file.name if end.file else None
    normalized_filename = os.path.normcase(os.path.normpath(filename))
    if (
        start_file is None
        or end_file is None
        or os.path.normcase(os.path.normpath(start_file)) != normalized_filename
        or os.path.normcase(os.path.normpath(end_file)) != normalized_filename
    ):
        return None
    char_start = byte_to_char(int(start.offset))
    char_end = byte_to_char(int(end.offset))
    if char_start >= char_end:
        return None
    return char_start, char_end


_CLANG_CURSOR_BUCKETS = {
    # 1-9: declarations
    "FUNCTION_DECL": 1,
    "CXX_METHOD": 1,
    "CONSTRUCTOR": 1,
    "DESTRUCTOR": 1,
    "CONVERSION_FUNCTION": 1,
    "FUNCTION_TEMPLATE": 1,
    "CLASS_DECL": 2,
    "CLASS_TEMPLATE": 2,
    "CLASS_TEMPLATE_PARTIAL_SPECIALIZATION": 2,
    "STRUCT_DECL": 3,
    "UNION_DECL": 3,
    "ENUM_DECL": 4,
    "NAMESPACE": 5,
    "VAR_DECL": 6,
    "PARM_DECL": 6,
    "FIELD_DECL": 7,
    "TYPEDEF_DECL": 9,
    "TYPE_ALIAS_DECL": 9,
    "USING_DECLARATION": 9,
    "USING_DIRECTIVE": 9,
    # 10-19: statements
    "COMPOUND_STMT": 10,
    "IF_STMT": 11,
    "FOR_STMT": 12,
    "WHILE_STMT": 13,
    "DO_STMT": 13,
    "RETURN_STMT": 14,
    "SWITCH_STMT": 15,
    "CASE_STMT": 15,
    "DEFAULT_STMT": 15,
    "TRY_STMT": 17,
    "CXX_CATCH_STMT": 17,
    "CXX_THROW_EXPR": 17,
    "BREAK_STMT": 18,
    "CONTINUE_STMT": 18,
    "GOTO_STMT": 18,
    "LABEL_STMT": 19,
    # 20-29: expressions
    "CALL_EXPR": 20,
    "BINARY_OPERATOR": 21,
    "COMPOUND_ASSIGNMENT_OPERATOR": 23,
    "UNARY_OPERATOR": 22,
    "CONDITIONAL_OPERATOR": 24,
    "MEMBER_REF_EXPR": 25,
    "ARRAY_SUBSCRIPT_EXPR": 26,
    "CSTYLE_CAST_EXPR": 27,
    "CXX_STATIC_CAST_EXPR": 27,
    "CXX_DYNAMIC_CAST_EXPR": 27,
    "CXX_REINTERPRET_CAST_EXPR": 27,
    "CXX_CONST_CAST_EXPR": 27,
    "CXX_NEW_EXPR": 28,
    "CXX_DELETE_EXPR": 28,
    "LAMBDA_EXPR": 29,
    "PAREN_EXPR": 29,
    # 30-39: type and reference nodes
    "TYPE_REF": 30,
    "TEMPLATE_REF": 32,
    "NAMESPACE_REF": 33,
    "DECL_REF_EXPR": 83,
    "OVERLOADED_DECL_REF": 83,
    "MEMBER_REF": 83,
    # 40-49: literals
    "INTEGER_LITERAL": 40,
    "FLOATING_LITERAL": 40,
    "IMAGINARY_LITERAL": 40,
    "STRING_LITERAL": 41,
    "CHARACTER_LITERAL": 42,
    "CXX_BOOL_LITERAL_EXPR": 43,
    "CXX_NULL_PTR_LITERAL_EXPR": 44,
    # 80-89: misc/preprocessor
    "INCLUSION_DIRECTIVE": 81,
    "MACRO_DEFINITION": 81,
    "MACRO_INSTANTIATION": 81,
    "TRANSLATION_UNIT": 82,
}


def _bucket_clang_cursor_kind(kind) -> int:
    name = getattr(kind, "name", str(kind))
    if "." in name:
        name = name.rsplit(".", 1)[-1]
    return _CLANG_CURSOR_BUCKETS.get(name, 255)


def extract_clang_ast_metadata(
    source: str,
    tu: TranslationUnit,
    filename: str,
) -> tuple[list[int], list[int], list[int]]:
    """Extract per-character AST metadata from clang cursor extents."""
    text_len = len(source)
    ast_depth = [0] * text_len
    sibling_index = [0] * text_len
    ast_node_type = [0] * text_len
    if text_len == 0 or tu is None:
        return ast_depth, sibling_index, ast_node_type

    byte_to_char = _byte_to_char_mapper(source)
    stack: list[tuple[Cursor, int, int]] = [(tu.cursor, 0, 0)]
    while stack:
        cursor, depth, sib_idx = stack.pop()
        offsets = _cursor_extent_offsets(cursor, filename, byte_to_char)
        if offsets is not None:
            start, end = offsets
            end = min(end, text_len)
            bucket = _bucket_clang_cursor_kind(cursor.kind)
            clamped_depth = min(depth, 65535)
            clamped_sib = min(sib_idx, 65535)
            for char_idx in range(start, end):
                ast_depth[char_idx] = clamped_depth
                sibling_index[char_idx] = clamped_sib
                ast_node_type[char_idx] = bucket

        children = list(cursor.get_children())
        child_depth = depth + 1
        for child_sib in range(len(children) - 1, -1, -1):
            stack.append((children[child_sib], child_depth, child_sib))

    return ast_depth, sibling_index, ast_node_type


def _slice_metadata(values: list[int], start: int, end: int) -> list[int]:
    if not values:
        return []
    return list(values[start:end])


def _cursor_text_and_metadata(
    cursor: Cursor,
    source: str,
    filename: str,
    byte_to_char: Callable[[int], int],
    ast_depth: list[int],
    sibling_index: list[int],
    ast_node_type: list[int],
) -> tuple[str, list[int], list[int], list[int], tuple[int, int] | None]:
    offsets = _cursor_extent_offsets(cursor, filename, byte_to_char)
    if offsets is None:
        return "", [], [], [], None
    start, end = offsets
    if start < 0 or end > len(source) or start >= end:
        return "", [], [], [], None
    return (
        source[start:end],
        _slice_metadata(ast_depth, start, end),
        _slice_metadata(sibling_index, start, end),
        _slice_metadata(ast_node_type, start, end),
        offsets,
    )


def extract_callees(cursor: Cursor) -> list[str]:
    """Extract all function call references from a cursor's children."""
    callees = set()

    def walk(node: Cursor):
        if node.kind == CursorKind.CALL_EXPR:
            ref = node.referenced
            if ref and ref.spelling:
                # Get fully qualified name
                qname = get_qualified_name(ref)
                if qname and not is_system_function(qname):
                    callees.add(qname)
        for child in node.get_children():
            walk(child)

    walk(cursor)
    return list(callees)


_REFERENCED_TYPE_DECL_KINDS = frozenset({
    CursorKind.STRUCT_DECL,
    CursorKind.CLASS_DECL,
    CursorKind.CLASS_TEMPLATE,
    CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
    CursorKind.ENUM_DECL,
    CursorKind.UNION_DECL,
    CursorKind.TYPEDEF_DECL,
    CursorKind.TYPE_ALIAS_DECL,
    CursorKind.TYPE_ALIAS_TEMPLATE_DECL,
})


def extract_referenced_types(cursor: Cursor) -> list[str]:
    """Extract qualified names of record/enum/typedef types referenced by a
    function (params, return, locals, member access, casts, template args).

    Mirrors :func:`extract_callees` but for TYPE relationships — captured during
    the SAME libclang parse pass and persisted on ``FunctionDef.referenced_types``
    so the offline doc-build path can emit ``type_refs``/``type_edges``. TYPE_REF
    cursors are clang's explicit type-usage markers; we resolve each to its
    declaring record/enum/typedef qname and drop system types.
    """
    types: set[str] = set()

    def walk(node: Cursor):
        if node.kind == CursorKind.TYPE_REF:
            ref = node.referenced
            if ref is not None and ref.spelling and ref.kind in _REFERENCED_TYPE_DECL_KINDS:
                qname = get_qualified_name(ref)
                if qname and not is_system_function(qname):
                    types.add(qname)
        for child in node.get_children():
            walk(child)

    walk(cursor)
    return list(types)


def get_qualified_name(cursor: Cursor) -> str:
    """Get the fully qualified name of a cursor (namespace::class::func)."""
    parts = []
    c = cursor
    while c and c.kind != CursorKind.TRANSLATION_UNIT:
        if c.spelling:
            parts.append(c.spelling)
        c = c.semantic_parent
    parts.reverse()
    return '::'.join(parts)


def extract_preamble(tu: TranslationUnit, filename: str) -> str:
    """Extract #include directives and forward declarations from a file."""
    preamble_parts = []
    for cursor in tu.cursor.get_children():
        if cursor.location.file and cursor.location.file.name != filename:
            continue
        if cursor.kind in (CursorKind.INCLUSION_DIRECTIVE,
                           CursorKind.USING_DIRECTIVE,
                           CursorKind.USING_DECLARATION,
                           CursorKind.TYPEDEF_DECL,
                           CursorKind.TYPE_ALIAS_DECL,
                           CursorKind.NAMESPACE_ALIAS):
            text = get_function_text(cursor, tu)
            if text:
                preamble_parts.append(text)
    return '\n'.join(preamble_parts)


FUNCTION_KINDS = {
    CursorKind.FUNCTION_DECL,
    CursorKind.CXX_METHOD,
    CursorKind.FUNCTION_TEMPLATE,
    CursorKind.CONSTRUCTOR,
    CursorKind.DESTRUCTOR,
    CursorKind.CONVERSION_FUNCTION,
}

CONTAINER_KINDS = {
    CursorKind.NAMESPACE,
    CursorKind.CLASS_DECL,
    CursorKind.STRUCT_DECL,
    CursorKind.CLASS_TEMPLATE,
    CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
}

# Cursor kinds whose DEFINITIONS we register as TypeDef chunks (type-edge
# targets). class/struct/union map to node-bucket kind 4; typedef/using map to
# kind 7 (see _CLANG_PART_NODE_BUCKETS). Enums get kind 4 as well (record-like).
_TYPE_DEF_KIND_BUCKET = {
    CursorKind.STRUCT_DECL: 4,
    CursorKind.CLASS_DECL: 4,
    CursorKind.CLASS_TEMPLATE: 4,
    CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION: 4,
    CursorKind.UNION_DECL: 4,
    CursorKind.ENUM_DECL: 4,
    CursorKind.TYPEDEF_DECL: 7,
    CursorKind.TYPE_ALIAS_DECL: 7,
    CursorKind.TYPE_ALIAS_TEMPLATE_DECL: 7,
}


def parse_translation_unit(
    filepath: str,
    index: Index,
    compile_args: list[str],
    project_dir: str,
) -> tuple[list[FunctionDef], list[TypeDef]]:
    """Parse a single C/C++ file (source OR header) and extract function
    definitions (with callees + referenced types) AND type definitions
    (struct/class/enum/union/typedef/using).

    Type definitions are returned alongside functions so the index can register
    them as type-edge target chunks. Headers are parsed as standalone TUs.
    """
    functions: list[FunctionDef] = []
    typedefs: list[TypeDef] = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as source_file:
            source = source_file.read()
    except OSError:
        return functions, typedefs

    try:
        tu = index.parse(
            filepath,
            args=compile_args,
            options=(
                TranslationUnit.PARSE_INCOMPLETE |
                TranslationUnit.PARSE_PRECOMPILED_PREAMBLE
            ),
        )
    except Exception as e:
        print(f"  WARN: Failed to parse {filepath}: {e}", file=sys.stderr)
        return functions, typedefs

    rel_path = os.path.relpath(filepath, project_dir)
    ast_depth, sibling_index, ast_node_type = extract_clang_ast_metadata(
        source,
        tu,
        filepath,
    )
    byte_to_char = _byte_to_char_mapper(source)
    project_dir_abs = os.path.abspath(project_dir)

    def _in_project(cursor) -> bool:
        """True if the cursor's location file lives under project_dir.

        Type DEFINITIONS usually live in project HEADERS that get #included into
        the .cpp TU, so we cannot restrict to ``== filepath`` for them or we'd
        capture nothing. We DO exclude system/third-party headers (outside
        project_dir) to avoid indexing std:: types.
        """
        loc = cursor.location.file
        if loc is None:
            return False
        return os.path.abspath(loc.name).startswith(project_dir_abs + os.sep)

    def visit(cursor):
        """Recursively visit cursors, descending into namespaces and classes.

        Functions are captured only from the primary file (``== filepath``) to
        avoid duplicating inline defs across every TU that includes the header.
        Type DEFINITIONS are captured from any project-local file (incl. headers
        pulled in via #include); duplicates across TUs are deduped by qname in
        ``ProjectIndex.add_typedef``.
        """
        loc = cursor.location.file
        if loc is None:
            return
        in_primary_file = loc.name == filepath
        in_project = in_primary_file or _in_project(cursor)
        if not in_project:
            return  # skip system/third-party headers entirely

        if in_primary_file and cursor.kind in FUNCTION_KINDS and cursor.is_definition():
            text, func_ast_depth, func_sibling_index, func_ast_node_type, _offsets = (
                _cursor_text_and_metadata(
                    cursor,
                    source,
                    filepath,
                    byte_to_char,
                    ast_depth,
                    sibling_index,
                    ast_node_type,
                )
            )
            if text and len(text) >= 20:
                callees = extract_callees(cursor)
                referenced_types = extract_referenced_types(cursor)
                qname = get_qualified_name(cursor)
                functions.append(FunctionDef(
                    name=cursor.spelling,
                    qualified_name=qname,
                    file=rel_path,
                    line=cursor.location.line,
                    text=text,
                    callees=callees,
                    is_definition=True,
                    end_line=cursor.extent.end.line,
                    ast_depth=func_ast_depth,
                    sibling_index=func_sibling_index,
                    ast_node_type=func_ast_node_type,
                    referenced_types=referenced_types,
                ))
            return

        # Register type DEFINITIONS (struct/class/enum/union/typedef/using) as
        # type-edge target chunks. We require an actual definition so a forward
        # declaration never shadows the real body.
        type_bucket = _TYPE_DEF_KIND_BUCKET.get(cursor.kind)
        if type_bucket is not None and cursor.is_definition():
            qname = get_qualified_name(cursor)
            if qname and not is_system_function(qname):
                td_text = get_function_text(cursor, tu)
                if td_text and len(td_text) >= 8:
                    typedefs.append(TypeDef(
                        name=cursor.spelling,
                        qualified_name=qname,
                        file=rel_path,
                        line=cursor.location.line,
                        text=td_text,
                        kind=type_bucket,
                        end_line=cursor.extent.end.line,
                    ))
            # Records (class/struct) can contain nested types/methods — recurse.

        if cursor.kind in CONTAINER_KINDS:
            # Recurse into namespaces, classes, structs for nested defs.
            for child in cursor.get_children():
                visit(child)

    for cursor in tu.cursor.get_children():
        visit(cursor)

    return functions, typedefs


_DEFAULT_SKIP_DIRS = frozenset({
    '.git', 'build', 'cmake-build', '__pycache__', 'node_modules',
    '.vs', '.vscode', 'third_party', 'external', 'deps', 'vendor',
    'test', 'tests', 'unittests', 'benchmarks',
        # Keep generated/noisy corpus paths out of clang indexing.
    'testing', 'examples', 'example', 'samples', 'sample', 'docs', 'doc',
    # Additional noise dirs for large repos
    'fuzzers', 'fuzzing', 'regression', 'fixtures',
})

C_EXTENSIONS = {'.c'}


def find_cpp_files(
    project_dir: str,
    extra_exclude_dirs: set[str] | None = None,
) -> list[str]:
    """Find all C/C++ source files in a directory."""
    skip_dirs = _DEFAULT_SKIP_DIRS | (extra_exclude_dirs or set())
    files = []
    for root, dirs, filenames in os.walk(project_dir):
        # Prune recursion in-place — os.walk won't descend into skipped dirs
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in INDEX_EXTENSIONS:
                filepath = os.path.join(root, fname)
                # Skip very large files
                try:
                    if os.path.getsize(filepath) > 500_000:
                        continue
                except OSError:
                    continue
                files.append(filepath)
    return files


def find_build_files(
    project_dir: str,
    extra_exclude_dirs: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Find build/compilation files. Sibling of ``find_cpp_files``.

    Returns a list of ``(abs_path, build_kind)`` tuples so the build-system tag
    (cmake/make/bazel/ninja/meson/...) is known at discovery time. Same
    ``os.walk``/skip-dir/size-cap logic as ``find_cpp_files``. Note: build files
    legitimately live under dirs that code discovery prunes (e.g. ``build/`` for
    ``compile_commands.json``, ``third_party/`` for vendored CMake), so we do NOT
    apply ``_DEFAULT_SKIP_DIRS`` here -- only ``.git`` and the caller's extra
    excludes -- otherwise the richest platform signal (compile_commands.json in
    ``build/``) would be silently dropped.
    """
    skip_dirs = {'.git'} | (extra_exclude_dirs or set())
    files: list[tuple[str, str]] = []
    for root, dirs, filenames in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in filenames:
            build_kind = classify_build_file(fname)
            if build_kind is None:
                continue
            filepath = os.path.join(root, fname)
            try:
                if os.path.getsize(filepath) > BUILD_FILE_SIZE_CAP:
                    continue
            except OSError:
                continue
            files.append((filepath, build_kind))
    return files


def load_compile_commands(project_dir: str) -> Optional[dict]:
    """Load compile_commands.json if available."""
    cc_path = find_compile_commands_file(project_dir)
    if cc_path is None:
        return None
    compile_index = load_compile_commands_file(cc_path)
    if compile_index is None:
        return None
    file_args = {}
    for entry in compile_index.entries:
        file_args[entry.filepath] = {
            "compile_args": list(entry.compile_args),
            "build_info": dict(entry.build_info),
        }
    return file_args
    return None


def _discover_include_dirs(project_dir: str) -> list[str]:
    """Recursively find directories named 'include' for -I flags."""
    include_dirs = set()
    for candidate in ['include', 'src', 'lib', 'source', '.']:
        d = os.path.join(project_dir, candidate)
        if os.path.isdir(d):
            include_dirs.add(d)
    # Walk up to 3 levels deep for nested include/ dirs
    for root, dirs, _ in os.walk(project_dir):
        depth = root[len(project_dir):].count(os.sep)
        if depth > 3:
            dirs.clear()
            continue
        dirs[:] = [d for d in dirs if d not in {'.git', 'build', 'third_party', 'external', 'test', 'tests'}]
        if os.path.basename(root) == 'include':
            include_dirs.add(root)
    return sorted(include_dirs)


def _is_sane_compile_args(args: list[str]) -> bool:
    """Check if build_context returned usable args (not shell script garbage)."""
    for arg in args:
        if arg is None:
            return False
        # Garbage indicators: semicolons, unresolved $vars, paths after -x
        if ';' in arg or '$' in arg:
            return False
        # Empty -I flags
        if arg == '-I':
            return False
        # Compound flags (multiple args jammed into one string)
        # e.g., "-std=c89 -Werror -O0" — legitimate args never have spaces
        # except -x <lang> which is handled as separate args
        if ' ' in arg and not arg.startswith('-D'):
            return False
    return True


_SIMPLE_FALLBACK_ARGS = ["-fsyntax-only", "-Wno-everything"]


def get_default_compile_args(project_dir: str) -> list[str]:
    """Generate default compile args for projects without compile_commands.json."""
    _platform_info, args, _compile_index = detect_build_context(project_dir)

    if _is_sane_compile_args(args):
        result = list(args)
    else:
        # build_context returned garbage (e.g., shell fragments from configure)
        result = list(_SIMPLE_FALLBACK_ARGS)

    if not any(arg.startswith('-I') for arg in result):
        result.append(f'-I{project_dir}')
        for d in _discover_include_dirs(project_dir):
            result.append(f'-I{d}')
    return result


def _adapt_args_for_file(args: list[str], filepath: str) -> list[str]:
    """Adapt compile args based on file extension — .c files need C mode, not C++,
    and headers must be parsed as ``-x c++-header`` (otherwise libclang fails to
    parse a standalone .h/.hpp as a translation unit)."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in HEADER_EXTENSIONS:
        # Force C++ header mode so struct/class/enum/typedef DEFINITIONS in
        # header-only files parse and become type-def chunks. Strip any existing
        # -x <lang> pair / joined -x form first so it doesn't conflict.
        adapted = []
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg == '-x':
                skip_next = True
                continue
            if arg.startswith('-x') and arg != '-x':
                continue
            adapted.append(arg)
        return ['-x', 'c++-header'] + adapted
    if ext in C_EXTENSIONS:
        adapted = []
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg == '-x':
                # Skip -x <lang> pair
                skip_next = True
                continue
            if arg.startswith('-std=c++'):
                adapted.append('-std=c11')
                continue
            if arg.startswith('-xc') or arg.startswith('-x '):
                # -xc++ joined form — replace with -xc
                continue
            adapted.append(arg)
        # Ensure C mode is set
        adapted = ['-x', 'c', '-std=c11'] + adapted
        return adapted
    return args


def estimate_tokens(text: str) -> int:
    """Estimate token count (~4 bytes per token for code)."""
    return max(1, len(text) // 4)


def collect_transitive_deps(
    root_qname: str,
    index: ProjectIndex,
    max_depth: int = 5,
) -> list[str]:
    """BFS to collect transitive dependencies of a function."""
    visited = {root_qname}
    queue = deque([(root_qname, 0)])
    deps = []

    while queue:
        qname, depth = queue.popleft()
        if depth >= max_depth:
            continue
        func = index.functions.get(qname)
        if not func:
            continue
        for callee in func.callees:
            if callee not in visited and callee in index.functions:
                visited.add(callee)
                deps.append(callee)
                queue.append((callee, depth + 1))

    return deps


_CLANG_PART_NODE_BUCKETS = {
    0: 0,   # other
    1: 81,  # preamble / preprocessor context
    2: 35,  # function signature/declarator
    3: 1,   # function body/definition
    4: 2,   # class or struct declaration
    5: 7,   # class member
    6: 80,  # comment/docstring
    7: 9,   # typedef/using
    8: 5,   # namespace
}


def extract_ast_metadata_from_parts(
    full_text: str,
    parts_info: list[PartInfo],
    index: ProjectIndex | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Emit clang AST metadata for assembled documents.

    The clang indexer builds documents from clang-extracted parts rather than
    reparsing the assembled training text. For function parts this copies the
    exact clang cursor metadata captured while parsing the original translation
    unit. Non-code scaffolding such as section comments and separators stays
    zero because it is not part of the clang AST.
    """
    text_len = len(full_text)
    ast_depth = [0] * text_len
    sibling_index = [0] * text_len
    ast_node_type = [0] * text_len
    offset = 0

    for part_index, (part_text, kind, dep_level, _name, _qname) in enumerate(parts_info):
        part_len = len(part_text)
        if part_len <= 0:
            continue
        end = min(offset + part_len, text_len)
        if offset >= text_len or offset >= end:
            break
        func = index.functions.get(_qname) if index is not None and _qname else None
        if (
            func is not None
            and len(func.ast_depth) == part_len
            and len(func.sibling_index) == part_len
            and len(func.ast_node_type) == part_len
        ):
            ast_depth[offset:end] = func.ast_depth[: end - offset]
            sibling_index[offset:end] = func.sibling_index[: end - offset]
            ast_node_type[offset:end] = func.ast_node_type[: end - offset]
        offset += part_len
        if part_index < len(parts_info) - 1:
            offset += 2

    return ast_depth, sibling_index, ast_node_type


# -------------------------------------------------------------------------
# Semantic metadata extraction (symbol_ids, call_targets, type_refs, def_use)
# -------------------------------------------------------------------------

# Def-use constants
DEF_USE_NONE = 0
DEF_USE_DEF = 1
DEF_USE_USE = 2

# CursorKind sets for classification
_DEFINITION_KINDS = {
    CursorKind.FUNCTION_DECL,
    CursorKind.CXX_METHOD,
    CursorKind.CONSTRUCTOR,
    CursorKind.DESTRUCTOR,
    CursorKind.CONVERSION_FUNCTION,
    CursorKind.FUNCTION_TEMPLATE,
    CursorKind.VAR_DECL,
    CursorKind.FIELD_DECL,
    CursorKind.PARM_DECL,
    CursorKind.CLASS_DECL,
    CursorKind.STRUCT_DECL,
    CursorKind.UNION_DECL,
    CursorKind.ENUM_DECL,
    CursorKind.ENUM_CONSTANT_DECL,
    CursorKind.TYPEDEF_DECL,
    CursorKind.TYPE_ALIAS_DECL,
    CursorKind.NAMESPACE,
    CursorKind.CLASS_TEMPLATE,
    CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
}

_CALL_KINDS = {
    CursorKind.CALL_EXPR,
}

_TYPE_REF_KINDS = {
    CursorKind.TYPE_REF,
    CursorKind.TEMPLATE_REF,
    CursorKind.NAMESPACE_REF,
}

_REFERENCE_KINDS = {
    CursorKind.DECL_REF_EXPR,
    CursorKind.MEMBER_REF_EXPR,
    CursorKind.MEMBER_REF,
    CursorKind.TYPE_REF,
    CursorKind.TEMPLATE_REF,
    CursorKind.NAMESPACE_REF,
    CursorKind.OVERLOADED_DECL_REF,
    CursorKind.VARIABLE_REF,
}


def _compute_symbol_id(qname: str) -> int:
    """Deterministic 32-bit symbol ID from a qualified name.

    Uses the lower 31 bits of md5, leaving 0 as the sentinel for 'no symbol'.
    """
    if not qname:
        return 0
    h = int(hashlib.md5(qname.encode("utf-8", errors="replace")).hexdigest(), 16)
    # Mask to 31 bits (positive int32), reserve 0 for no-symbol
    result = (h & 0x7FFFFFFF) or 1
    return result


def extract_semantic_metadata(
    source: str,
    tu: 'TranslationUnit',
    filename: str,
) -> dict:
    """Extract per-character semantic metadata from a translation unit.

    Walks the clang AST and produces four char-level arrays:

    - symbol_ids:   per-char deterministic hash of the symbol's qualified name
                    (0 = no symbol)
    - call_targets: per-char symbol ID of the call target for CALL_EXPR tokens
                    (0 = not a call)
    - type_refs:    per-char symbol ID of the referenced type for TYPE_REF tokens
                    (0 = not a type reference)
    - def_use:      per-char def/use marker (0=none, 1=def, 2=use)

    Returns:
        dict with keys 'symbol_ids', 'call_targets', 'type_refs', 'def_use',
        each a list[int] of length len(source).
    """
    text_len = len(source)
    symbol_ids = [0] * text_len
    call_targets = [0] * text_len
    type_refs = [0] * text_len
    def_use = [0] * text_len

    if text_len == 0 or tu is None:
        return {
            "symbol_ids": symbol_ids,
            "call_targets": call_targets,
            "type_refs": type_refs,
            "def_use": def_use,
        }

    source_bytes = source.encode("utf-8")
    byte_len = len(source_bytes)

    # Build byte-to-char mapping for non-ASCII sources
    if byte_len != text_len:
        byte_to_char = [0] * byte_len
        byte_offset = 0
        for char_idx, ch in enumerate(source):
            ch_bytes = len(ch.encode("utf-8"))
            for b in range(ch_bytes):
                if byte_offset + b < byte_len:
                    byte_to_char[byte_offset + b] = char_idx
            byte_offset += ch_bytes
    else:
        byte_to_char = None  # identity mapping

    def _byte_to_char_offset(byte_off: int) -> int:
        if byte_to_char is None:
            return byte_off
        if byte_off >= byte_len:
            return text_len
        return byte_to_char[byte_off]

    def _visit(cursor):
        """Walk the AST and annotate char ranges."""
        loc = cursor.location
        if not loc or not loc.file:
            return
        if loc.file.name != filename:
            return

        extent = cursor.extent
        if not extent:
            return

        start_offset = extent.start.offset
        end_offset = extent.end.offset
        if start_offset >= end_offset:
            return

        # Convert to char offsets
        char_start = _byte_to_char_offset(start_offset)
        char_end = _byte_to_char_offset(end_offset)
        if char_start >= text_len or char_start >= char_end:
            return

        char_end = min(char_end, text_len)
        kind = cursor.kind

        # Symbol identification: get qualified name for definitions and refs
        qname = ""
        if kind in _DEFINITION_KINDS and cursor.is_definition():
            qname = get_qualified_name(cursor)
            if qname:
                sym_id = _compute_symbol_id(qname)
                for ci in range(char_start, char_end):
                    symbol_ids[ci] = sym_id
                    def_use[ci] = DEF_USE_DEF

        elif kind in _REFERENCE_KINDS:
            ref = cursor.referenced
            if ref and ref.spelling:
                qname = get_qualified_name(ref)
                if qname:
                    sym_id = _compute_symbol_id(qname)
                    for ci in range(char_start, char_end):
                        symbol_ids[ci] = sym_id
                        def_use[ci] = DEF_USE_USE

        # Call target annotation
        if kind in _CALL_KINDS:
            ref = cursor.referenced
            if ref and ref.spelling:
                target_qname = get_qualified_name(ref)
                if target_qname:
                    target_id = _compute_symbol_id(target_qname)
                    for ci in range(char_start, char_end):
                        call_targets[ci] = target_id

        # Type reference annotation
        if kind in _TYPE_REF_KINDS:
            ref = cursor.referenced
            if ref and ref.spelling:
                ref_qname = get_qualified_name(ref)
                if ref_qname:
                    ref_id = _compute_symbol_id(ref_qname)
                    for ci in range(char_start, char_end):
                        type_refs[ci] = ref_id

        # Recurse into children
        for child in cursor.get_children():
            _visit(child)

    try:
        for child in tu.cursor.get_children():
            _visit(child)
    except (RecursionError, Exception):
        # Graceful degradation: return whatever we have so far
        pass

    return {
        "symbol_ids": symbol_ids,
        "call_targets": call_targets,
        "type_refs": type_refs,
        "def_use": def_use,
    }


def extract_semantic_metadata_from_parts(
    full_text: str,
    parts_info: list[PartInfo],
    index: ProjectIndex,
) -> dict:
    """Produce semantic metadata from pre-extracted parts without running libclang.

    This is the offline path used when the enriched doc is built from
    already-parsed FunctionDef objects. It assigns symbol_ids and def_use
    based on the parts_info metadata (qname, kind) and the call graph in
    the ProjectIndex.

    Returns:
        dict with keys 'symbol_ids', 'call_targets', 'type_refs', 'def_use',
        each a list[int] of length len(full_text).
    """
    text_len = len(full_text)
    symbol_ids = [0] * text_len
    call_targets = [0] * text_len
    type_refs = [0] * text_len
    def_use_arr = [0] * text_len

    offset = 0
    for i, part in enumerate(parts_info):
        part_text = part[0]
        kind = part[1]
        part_len = len(part_text)
        if offset + part_len > text_len:
            break

        qname = part[4] if len(part) > 4 else None

        if qname:
            sym_id = _compute_symbol_id(qname)
            # Function parts are definition sites
            for ci in range(offset, offset + part_len):
                symbol_ids[ci] = sym_id
                def_use_arr[ci] = DEF_USE_DEF

            # Annotate call_targets + type_refs by marking each referenced
            # name's occurrences within this part (approximate call-site / type-
            # use spans; exact char offsets would need a re-parse). This replaces
            # the old 1-char-per-callee stub and the never-written type_refs.
            func_def = index.functions.get(qname)
            if func_def:
                def _mark(arr, name, value):
                    if not name or not value:
                        return
                    s = 0
                    plen = len(name)
                    while True:
                        idx = part_text.find(name, s)
                        if idx < 0:
                            break
                        for ci in range(offset + idx, min(offset + idx + plen, offset + part_len)):
                            arr[ci] = value
                        s = idx + plen
                for callee_qname in func_def.callees:
                    _mark(call_targets, callee_qname.split('::')[-1], _compute_symbol_id(callee_qname))
                for type_qname in getattr(func_def, 'referenced_types', []):
                    _mark(type_refs, type_qname.split('::')[-1], _compute_symbol_id(type_qname))

        elif kind == 1:  # PREAMBLE
            # Preamble: mark as use sites (references to included entities)
            for ci in range(offset, offset + part_len):
                def_use_arr[ci] = DEF_USE_USE

        offset += part_len
        if i < len(parts_info) - 1:
            offset += 2  # "\n\n" separator

    return {
        "symbol_ids": symbol_ids,
        "call_targets": call_targets,
        "type_refs": type_refs,
        "def_use": def_use_arr,
    }


def build_enriched_doc(
    parts_info: list[PartInfo],
    index: ProjectIndex,
    filepath: str | None = None,
    compile_args: list[str] | None = None,
    build_info: dict | None = None,
) -> dict:
    """Build enriched document from parts with metadata.

    Args:
        parts_info: list of (text, kind, dep_level, name, qname_or_none) tuples
            kind: 0=other, 1=preamble, 2=func_signature, 3=func_body,
                  4=class_decl, 5=class_member, 6=comment, 7=typedef, 8=namespace
        index: ProjectIndex for computing edges

    Returns:
        dict with text, structure_ids, chunk_boundaries, call_edges, type_edges,
        ast_depth, sibling_index, ast_node_type
    """
    texts = [p[0] for p in parts_info]
    full_text = '\n\n'.join(texts)
    text_len = len(full_text)

    structure_ids = [0] * text_len
    chunk_boundaries = []
    offset = 0

    # Map chunk_idx -> qname for edge computation
    chunk_qnames = {}        # function chunks only (call-edge sources/targets)
    chunk_callees = {}
    chunk_all_qnames = {}    # ANY named chunk incl. type defs (type-edge targets)
    chunk_types = {}         # referenced_types per function chunk (type-edge sources)

    for i, (part_text, kind, dep_level, name, qname) in enumerate(parts_info):
        part_len = len(part_text)
        if offset + part_len > text_len:
            break

        # Fill structure_ids
        for j in range(offset, offset + part_len):
            structure_ids[j] = kind

        chunk_boundaries.append({
            'start': offset,
            'end': offset + part_len,
            'kind': kind,
            'dep_level': dep_level,
            'name': name,
        })

        # Track qnames for edge computation.
        if qname:
            chunk_all_qnames[i] = qname
        if qname and qname in index.functions:
            chunk_qnames[i] = qname
            chunk_callees[i] = index.functions[qname].callees
            chunk_types[i] = getattr(index.functions[qname], 'referenced_types', [])

        offset += part_len
        if i < len(parts_info) - 1:
            offset += 2  # "\n\n"

    # Compute call_edges
    call_edges = []
    for ci, caller_qname in chunk_qnames.items():
        callees = chunk_callees.get(ci, [])
        for callee_qname in callees:
            for cj, target_qname in chunk_qnames.items():
                if ci != cj and target_qname == callee_qname:
                    call_edges.append({'from': ci, 'to': cj})

    # Compute type_edges: a function chunk referencing type T -> the chunk that
    # defines T (mirror of the call_edges loop, over referenced_types).
    type_edges: list[dict[str, object]] = []
    for ci, ref_types in chunk_types.items():
        for t in ref_types:
            for cj, q in chunk_all_qnames.items():
                if ci != cj and q == t:
                    type_edges.append({'from': ci, 'to': cj})

    ast_depth, sibling_index, ast_node_type = extract_ast_metadata_from_parts(
        full_text,
        parts_info,
        index,
    )

    # v4 enrichment: per-file platform detection
    import sys as _sys
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _v5_dir = os.path.join(_parent, 'v5_gke_orchestrator')
    if _v5_dir not in _sys.path:
        _sys.path.insert(0, _v5_dir)
    platform_info: dict[str, object] | None
    try:
        _platform_detect = importlib.import_module("platform_detect")
        _detect_plat = cast(PlatformDetectFn, getattr(_platform_detect, "detect_platforms"))
        platform_info = _detect_plat(full_text)
    except ImportError:
        platform_info = None

    # Extract semantic metadata (symbol IDs, call targets, type refs, def-use)
    semantic_meta = extract_semantic_metadata_from_parts(full_text, parts_info, index)

    result = {
        'text': full_text,
        'structure_ids': structure_ids,
        'chunk_boundaries': chunk_boundaries,
        'call_edges': call_edges,
        'type_edges': type_edges,
        'ast_depth': ast_depth,
        'sibling_index': sibling_index,
        'ast_node_type': ast_node_type,
        'symbol_ids': semantic_meta['symbol_ids'],
        'call_targets': semantic_meta['call_targets'],
        'type_refs': semantic_meta['type_refs'],
        'def_use': semantic_meta['def_use'],
    }
    language_info = detect_language_info(
        full_text,
        filepath,
        platform_info,
        compile_args=compile_args,
        build_info=build_info,
    )

    if platform_info:
        result['platform_info'] = platform_info
    if build_info:
        result['build_info'] = build_info
    if language_info:
        result['language_info'] = language_info
    return result


# --------------------------------------------------------------------------- #
# Build-file doc emission (ADDITIVE).
#
# A build file is emitted as a 'build' enriched doc with the SAME dict shape as
# build_enriched_doc, but: full raw text; structure_ids ALL set to BUILD_KIND=9
# (extends the 0-8 code-kind vocab); EMPTY call/type/symbol graph (correct --
# build files are not C++, like commit docs on the code channels); a doc_type of
# 'build'; and a language_info whose primary_language is the build-system tag
# (cmake/make/bazel/ninja/meson/...). platform_info / build_info carry the
# derived A-platform signal where available. NO libclang involved.
# --------------------------------------------------------------------------- #
BUILD_KIND = 9  # structure_ids kind for build files (extends 0-8 code vocab)


def build_build_doc(
    filepath: str,
    text: str,
    build_kind: str,
    *,
    platform_info: dict | None = None,
    build_info: dict | None = None,
) -> dict:
    """Build a single 'build' enriched doc from a build/compilation file.

    FAIL LOUD (RULE #1): callers pass already-read text; an empty/whitespace-only
    build file is a real signal failure and the caller raises rather than skip.
    """
    text_len = len(text)
    structure_ids = [BUILD_KIND] * text_len
    chunk_boundaries = [{
        'start': 0,
        'end': text_len,
        'kind': BUILD_KIND,
        'dep_level': 0,
        'name': os.path.basename(filepath),
    }]

    # Per-file platform detection from the build text itself (additive to the
    # repo-level build_info threaded in by process_project).
    detected_platform: dict[str, object] | None = platform_info
    if detected_platform is None:
        import sys as _sys
        _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _v5_dir = os.path.join(_parent, 'v5_gke_orchestrator')
        if _v5_dir not in _sys.path:
            _sys.path.insert(0, _v5_dir)
        try:
            _platform_detect = importlib.import_module("platform_detect")
            _detect_plat = cast(PlatformDetectFn, getattr(_platform_detect, "detect_platforms"))
            detected_platform = _detect_plat(text)
        except ImportError:
            detected_platform = None

    # Language tag: the build-system family IS the primary_language so the model
    # knows which build system this is. detect_language_info has no build-system
    # output key, so we set primary_language directly (additive, rendered for
    # free by language_info_to_prefix as "// language: primary=<build_kind> ...").
    language_info = {
        "primary_language": build_kind,
        "primary_standard": (build_info or {}).get("standard"),
        "primary_dialect": None,
        "embedded_languages": [],
        "signals": [f"build_file:{build_kind}"],
        "detector_sources": ["build_file"],
        "confidence": "high",
    }

    result: dict[str, object] = {
        'text': text,
        'doc_type': 'build',
        'build_kind': build_kind,
        'structure_ids': structure_ids,
        'chunk_boundaries': chunk_boundaries,
        # Build files carry NO call/type/symbol graph (not C++) -- empty/0, like
        # commit docs on the code channels. This is correct, not a fallback.
        'call_edges': [],
        'type_edges': [],
        'ast_depth': [],
        'sibling_index': [],
        'ast_node_type': [],
        'symbol_ids': [],
        'call_targets': [],
        'type_refs': [],
        'def_use': [],
        'language_info': language_info,
    }
    if detected_platform:
        result['platform_info'] = detected_platform
    if build_info:
        result['build_info'] = build_info
    return result


# --------------------------------------------------------------------------- #
# Function-level tokenized-hash dedup (CORRECTED design).
#
# Per the user-specified ordering, dedup happens at the FUNCTION level, on the
# function AFTER OUR TOKENIZER (the whitespace sentinels canonicalize format),
# and BEFORE the dependency grouping. The dedup decides which functions get
# emitted as their OWN ROOT document: a function whose tokenized hash was already
# seen (in this repo OR globally via the shared db, including as a dependency of
# an earlier kept root) does NOT get a duplicate standalone root doc. CRUCIALLY,
# we do NOT delete deduped functions from index.functions -- collect_transitive_
# deps must still resolve them so each distinct kept root embeds its full
# dependency chain (a shared helper legitimately appears as context inside two
# different roots' docs; it just is not emitted twice as a standalone root).
# --------------------------------------------------------------------------- #
_CPPMEGA_TOKENIZER = None


def _load_cppmega_tokenizer(tokenizer_path: str):
    """Load (and memoize) OUR CppMegaTokenizer. FAIL LOUD on failure (RULE #1)."""
    global _CPPMEGA_TOKENIZER
    if _CPPMEGA_TOKENIZER is not None:
        return _CPPMEGA_TOKENIZER
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
    _CPPMEGA_TOKENIZER = load_cppmega_tokenizer(tokenizer_path)
    return _CPPMEGA_TOKENIZER


def dedup_root_functions(
    index: ProjectIndex,
    *,
    tokenizer_path: str,
    dedup_db: str | None,
    near: bool = True,
) -> tuple[set[str], dict[str, int]]:
    """Decide which root functions are duplicates (drop their standalone doc).

    For each function definition we compute ``token_ids = tokenizer.encode(text)``
    and test against:
      * exact: ``sha1(token_ids)`` (identical-after-tokenizer), and
      * near: MinHash-LSH @0.7 over token-id 5-gram shingles.

    A function whose hash was already seen is marked as a DROPPED ROOT: the
    grouping loop will not emit a standalone document for it. Functions stay in
    ``index.functions`` untouched so ``collect_transitive_deps`` still resolves
    them as dependencies of OTHER kept roots (so the shared canonical copy is
    embedded in each dependent root's doc -- no duplicate STANDALONE docs, but
    dependency chains stay intact).

    When ``dedup_db`` is given the dedup is GLOBAL + resumable + cross-stream via
    the shared SQLite ``DedupStore`` (fail-loud if it cannot open / datasketch is
    missing). When absent, a per-repo in-RAM exact set is used (no near).

    Returns (dropped_root_qnames, {dropped_exact, dropped_near, kept_roots}).
    """
    tok = _load_cppmega_tokenizer(tokenizer_path)

    store = None
    seen_local: set[bytes] = set()
    if dedup_db:
        # FAIL LOUD: open failure / missing datasketch raises inside DedupStore.
        from dedup_store import DedupStore  # noqa: F401
        store = DedupStore(dedup_db, near=near, commit_every=2000)

    dropped_roots: set[str] = set()
    dropped_exact = 0
    dropped_near = 0
    kept_roots = 0

    # Deterministic order: by (file, line) so the canonical kept copy is stable
    # across runs (important for resumability and reproducible packs).
    items = sorted(
        index.functions.items(),
        key=lambda kv: (kv[1].file or "", kv[1].line or 0, kv[0]),
    )
    for qname, func in items:
        if not (func.is_definition and func.text):
            continue
        token_ids = tok.encode(func.text)
        if store is not None:
            if store.seen_exact_tokens(token_ids):
                dropped_exact += 1
                dropped_roots.add(qname)
                continue
            if near and store.seen_near_tokens(token_ids):
                dropped_near += 1
                dropped_roots.add(qname)
                continue
        else:
            from dedup_store import _sha1_tokens
            h = _sha1_tokens(token_ids)
            if h in seen_local:
                dropped_exact += 1
                dropped_roots.add(qname)
                continue
            seen_local.add(h)
        kept_roots += 1

    if store is not None:
        store.commit()
        store.close()

    return dropped_roots, {
        "kept_roots": kept_roots,
        "dropped_exact": dropped_exact,
        "dropped_near": dropped_near,
    }


def emit_build_documents(
    build_files: list[tuple[str, str]],
    *,
    default_build_info: dict | None,
    compile_index: object | None = None,
    tokenizer_path: str | None = None,
    dedup_db: str | None = None,
    dedup_near: bool = True,
) -> list[dict]:
    """Emit one 'build' doc per build file, with WHOLE-DOC tokenized-hash dedup.

    Build-file dedup is at the WHOLE-DOC level (not function-level): a build file
    whose tokenized text hash was already seen (this repo OR globally via the
    shared db) is dropped, mirroring the commit-doc whole-doc dedup. Uses the
    SAME shared DedupStore tables (token-id exact + near) so build docs dedup
    against each other globally and resumably.

    FAIL LOUD (RULE #1): a discovered build file that cannot be read/decoded or
    tokenizes empty RAISES -- we never silently skip a build file.
    """
    docs: list[dict] = []
    if not build_files:
        return docs

    tok = None
    store = None
    seen_local: set[bytes] = set()
    if tokenizer_path:
        tok = _load_cppmega_tokenizer(tokenizer_path)
        if dedup_db:
            from dedup_store import DedupStore
            store = DedupStore(dedup_db, near=dedup_near, commit_every=2000)

    dropped = 0
    for filepath, build_kind in sorted(build_files):
        # FAIL LOUD on unreadable build files -- do not paper over a broken file.
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        if not text or not text.strip():
            raise RuntimeError(
                f"build file {filepath} ({build_kind}) is empty/whitespace-only; "
                f"refusing to emit an empty build doc (RULE #1: fail loud)"
            )

        per_file_build_info = dict(default_build_info) if default_build_info else {}
        per_file_build_info["build_system"] = build_kind

        if tok is not None:
            token_ids = tok.encode(text)
            if not token_ids:
                raise RuntimeError(
                    f"build file {filepath} ({build_kind}) tokenized to ZERO ids; "
                    f"tokenizer/build-doc bug (RULE #1: fail loud)"
                )
            if store is not None:
                if store.seen_exact_tokens(token_ids):
                    dropped += 1
                    continue
                if dedup_near and store.seen_near_tokens(token_ids):
                    dropped += 1
                    continue
            else:
                from dedup_store import _sha1_tokens
                h = _sha1_tokens(token_ids)
                if h in seen_local:
                    dropped += 1
                    continue
                seen_local.add(h)

        docs.append(
            build_build_doc(
                filepath,
                text,
                build_kind,
                build_info=per_file_build_info or None,
            )
        )

    if store is not None:
        store.commit()
        store.close()

    print(
        f"  Build docs: emitted={len(docs)} dropped_dup={dropped} "
        f"(whole-doc tokenized-hash dedup)",
        file=sys.stderr,
    )
    return docs


def build_training_documents(
    index: ProjectIndex,
    max_tokens: int = 16384,
    max_dep_depth: int = 5,
    enriched: bool = False,
    *,
    project_dir: str | None = None,
    compile_db: dict | None = None,
    default_args: list[str] | None = None,
    default_build_info: dict | None = None,
    tokenizer_path: str | None = None,
    dedup_db: str | None = None,
    dedup_near: bool = True,
) -> list:
    """Build training documents with bottom-up dependency ordering.

    Returns list[str] when enriched=False, list[dict] when enriched=True.

    When ``tokenizer_path`` is given, FUNCTION-LEVEL tokenized-hash dedup runs
    BEFORE grouping (corrected design): a function whose tokenized hash was
    already seen does not get its OWN standalone root document, but it stays in
    the index so it can still be embedded as a DEPENDENCY of other kept roots.
    """
    documents: list[str | dict[str, object]] = []

    index.compute_dep_levels()

    # Function-level dedup BEFORE grouping (corrected design). Requires OUR
    # tokenizer; when no tokenizer_path is supplied we skip it (legacy callers /
    # unit tests) and rely on the caller's higher-level dedup.
    dropped_roots: set[str] = set()
    if tokenizer_path:
        dropped_roots, stats = dedup_root_functions(
            index,
            tokenizer_path=tokenizer_path,
            dedup_db=dedup_db,
            near=dedup_near,
        )
        print(
            f"  Function-level dedup: kept_roots={stats['kept_roots']} "
            f"dropped_exact={stats['dropped_exact']} "
            f"dropped_near={stats['dropped_near']}",
            file=sys.stderr,
        )

    for qname, func in index.functions.items():
        if not func.is_definition:
            continue
        # Skip emitting a STANDALONE root doc for a deduped function (its hash was
        # already seen). It remains resolvable as a dependency of other roots.
        if qname in dropped_roots:
            continue

        # Collect transitive deps
        dep_qnames = collect_transitive_deps(qname, index, max_dep_depth)

        # Sort by dep_level (leaves/most foundational first)
        dep_funcs: list[FunctionDef] = []
        for dq in dep_qnames:
            df = index.functions.get(dq)
            if df and df.is_definition and df.text:
                dep_funcs.append(df)
        dep_funcs.sort(key=lambda f: f.dep_level)

        # Build document
        preamble = index.file_preambles.get(func.file, '')

        def _assemble(dep_funcs_list: list[FunctionDef]) -> str:
            parts: list[str] = []
            if preamble:
                parts.append(preamble)
            for df in dep_funcs_list:
                parts.append(df.text)
            parts.append(func.text)
            return '\n\n'.join(parts)

        doc = _assemble(dep_funcs)
        tokens = estimate_tokens(doc)

        # Token budget management
        if tokens > max_tokens * 2 and dep_funcs:
            # Too big: trim deps from highest dep_level first
            while tokens > max_tokens * 2 and dep_funcs:
                dep_funcs.pop()  # remove highest-level dep
                doc = _assemble(dep_funcs)
                tokens = estimate_tokens(doc)

        if tokens < 20:
            continue

        # NOTE: doc-level md5 dedup removed (corrected design). Dedup now happens
        # at the FUNCTION level on the tokenized hash, BEFORE grouping, via
        # dedup_root_functions above -- duplicate functions are not emitted as
        # standalone roots, so no second-pass doc dedup is needed here.

        if enriched:
            # Build parts_info for enriched output. Thread repo-level build
            # context through fallback lanes so build-file-derived args keep
            # their authoritative provenance in language_info/build_info.
            parts_info: list[PartInfo] = []
            if preamble:
                parts_info.append((preamble, 1, 0, '', None))  # kind=1 PREAMBLE
            for df in dep_funcs:
                parts_info.append((df.text, 2, df.dep_level, df.name,
                                   df.qualified_name))  # kind=2 FUNC
            parts_info.append((func.text, 2, func.dep_level, func.name,
                               func.qualified_name))  # kind=2 FUNC (root)

            # Pull in DEFINITIONS of types referenced by the root func + its
            # dep funcs as type-edge target chunks. build_enriched_doc matches
            # a function chunk's referenced_types against any chunk whose qname
            # equals the type qname (chunk_all_qnames) to emit type_edges; the
            # type def must therefore be present as a chunk. Definitions live in
            # headers (now indexed), registered in index.typedefs.
            func_qnames_in_doc = {func.qualified_name}
            func_qnames_in_doc.update(df.qualified_name for df in dep_funcs)
            referenced: list[str] = list(func.referenced_types)
            for df in dep_funcs:
                referenced.extend(df.referenced_types)
            added_type_qnames: set[str] = set()
            for tq in referenced:
                if tq in added_type_qnames or tq in func_qnames_in_doc:
                    continue
                td = index.typedefs.get(tq)
                if td is None or not td.text:
                    continue
                added_type_qnames.add(tq)
                # kind from TypeDef (4=class/struct/enum/union, 7=typedef/using)
                parts_info.append((td.text, td.kind, 0, td.name,
                                   td.qualified_name))

            compile_args = list(default_args or [])
            build_info = dict(default_build_info) if default_build_info else None
            if project_dir is not None:
                abs_func_path = os.path.normpath(os.path.join(project_dir, func.file))
                file_build = compile_db.get(abs_func_path) if compile_db else None
                if file_build:
                    compile_args = file_build.get("compile_args", compile_args)
                    build_info = file_build.get("build_info") or build_info
            documents.append(
                build_enriched_doc(
                    parts_info,
                    index,
                    filepath=func.file,
                    compile_args=compile_args,
                    build_info=build_info,
                )
            )
        else:
            documents.append(doc)

    return documents


def _resolve_file_args(filepath, compile_db, default_args):
    """Get per-file compile args, adapting C vs C++ flags."""
    if compile_db and filepath in compile_db:
        build_entry = compile_db[filepath]
        args = _sanitize_compile_args_for_clang(build_entry.get("compile_args", default_args))
    else:
        args = _sanitize_compile_args_for_clang(default_args)
    return _adapt_args_for_file(args, filepath)


def _parse_file_batch(args_tuple):
    """Worker function for parallel parsing. Each worker creates its own Index."""
    filepaths, compile_db, default_args, project_dir = args_tuple
    sys.setrecursionlimit(50000)  # Set in each worker process too
    _configure_libclang()
    clang_index = Index.create()
    func_results: list[dict] = []
    type_results: list[dict] = []
    errors = 0
    for filepath in filepaths:
        args = _resolve_file_args(filepath, compile_db, default_args)
        try:
            functions, typedefs = parse_translation_unit(filepath, clang_index, args, project_dir)
            func_results.extend(f.to_dict() for f in functions)
            type_results.extend(t.to_dict() for t in typedefs)
        except (Exception, RecursionError):
            errors += 1
    return {"functions": func_results, "typedefs": type_results}, len(filepaths), errors


def _parse_single_file_worker(args_tuple):
    """Parse one file in a fresh subprocess so a segfault is file-local."""
    filepath, compile_db, default_args, project_dir = args_tuple
    sys.setrecursionlimit(50000)
    _configure_libclang()
    clang_index = Index.create()
    args = _resolve_file_args(filepath, compile_db, default_args)
    functions, typedefs = parse_translation_unit(filepath, clang_index, args, project_dir)
    return {
        "functions": [f.to_dict() for f in functions],
        "typedefs": [t.to_dict() for t in typedefs],
    }


def _parse_files_isolated(
    cpp_files: list[str],
    compile_db: dict | None,
    default_args: list[str],
    project_dir: str,
    *,
    max_workers: int,
) -> tuple[ProjectIndex, int, int]:
    """Parse files via one-file subprocess tasks so crashes don't kill the whole repo."""
    index_obj = ProjectIndex()
    parsed = 0
    errors = 0
    worker_count = max(1, max_workers)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _parse_single_file_worker,
                (filepath, compile_db, default_args, project_dir),
            ): filepath
            for filepath in cpp_files
        }
        for future in as_completed(futures):
            filepath = futures[future]
            try:
                payload = future.result(timeout=120)  # 2 min per file
                for d in payload["functions"]:
                    index_obj.add_function(FunctionDef.from_dict(d))
                for td in payload["typedefs"]:
                    index_obj.add_typedef(TypeDef.from_dict(td))
                parsed += 1
            except TimeoutError:
                errors += 1
                if errors <= 5:
                    print(f"  TIMEOUT parsing {filepath} (>120s)", file=sys.stderr)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR parsing {filepath}: {e}", file=sys.stderr)
            if parsed % 500 == 0 and parsed > 0:
                print(
                    f"  Parsed {parsed}/{len(cpp_files)} files, {len(index_obj.functions)} functions",
                    file=sys.stderr,
                )
    return index_obj, parsed, errors


def process_project(
    project_dir: str,
    max_tokens: int = 16384,
    max_dep_depth: int = 5,
    parse_workers: int = 1,
    enriched: bool = False,
    extra_exclude_dirs: set[str] | None = None,
    memory_limit_gb: float = 10.0,
    tokenizer_path: str | None = None,
    dedup_db: str | None = None,
    dedup_near: bool = True,
) -> list:
    """Process a single project: parse all files, build index, generate docs."""
    project_dir = os.path.abspath(project_dir)
    project_name = os.path.basename(project_dir)
    _configure_libclang()

    print(f"\n--- Processing project: {project_name} ---", file=sys.stderr)

    # Find source files
    cpp_files = find_cpp_files(project_dir, extra_exclude_dirs=extra_exclude_dirs)
    print(f"  Found {len(cpp_files)} C/C++ source files", file=sys.stderr)

    # Discover build/compilation files (ADDITIVE; emitted as 'build' docs).
    build_files = find_build_files(project_dir, extra_exclude_dirs=extra_exclude_dirs)
    print(f"  Found {len(build_files)} build/compilation files", file=sys.stderr)

    # Load or derive build context. Done unconditionally so build-only repos
    # (no C/C++ at all) still get A-platform enrichment for their build docs.
    compile_db = load_compile_commands(project_dir)
    _platform_info, _raw_args, _compile_index = detect_build_context(project_dir)
    default_args = _sanitize_compile_args_for_clang(
        get_default_compile_args(project_dir)
    )
    default_build_info = {
        key: value
        for key, value in _platform_info.items()
        if key in {"build_system", "source", "compiler", "standard"} and value is not None
    }

    # Parse all files and build index
    index_obj = ProjectIndex()

    # Use parallel parsing for large projects
    effective_workers = min(parse_workers, max(1, len(cpp_files) // 100))
    if not cpp_files:
        # Build-only repo (no C/C++): skip libclang parsing entirely. Build docs
        # are still emitted below so we do NOT silently drop the repo.
        print("  No C/C++ sources -- emitting build docs only", file=sys.stderr)
    elif effective_workers > 1 and len(cpp_files) > 200:
        print(f"  Using {effective_workers} parse workers", file=sys.stderr)
        chunk_size = max(50, len(cpp_files) // effective_workers)
        batches = []
        for i in range(0, len(cpp_files), chunk_size):
            batch = cpp_files[i:i + chunk_size]
            batches.append((batch, compile_db, default_args, project_dir))

        total_parsed = 0
        total_errors = 0
        try:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                for payload, parsed_count, error_count in executor.map(_parse_file_batch, batches):
                    for d in payload["functions"]:
                        index_obj.add_function(FunctionDef.from_dict(d))
                    for td in payload["typedefs"]:
                        index_obj.add_typedef(TypeDef.from_dict(td))
                    total_parsed += parsed_count
                    total_errors += error_count
                    check_memory_limit(memory_limit_gb, label="index_project")
                    print(f"  Parsed {total_parsed}/{len(cpp_files)} files, "
                          f"{len(index_obj.functions)} functions", file=sys.stderr)

            print(f"  Parsed {total_parsed} files ({total_errors} errors), "
                  f"{len(index_obj.functions)} functions indexed", file=sys.stderr)
        except Exception as exc:
            print(
                f"  WARN: parallel parse failed ({exc}); retrying isolated per-file workers",
                file=sys.stderr,
            )
            index_obj, parsed, errors = _parse_files_isolated(
                cpp_files,
                compile_db,
                default_args,
                project_dir,
                max_workers=min(4, effective_workers),
            )
            print(f"  Parsed {parsed} files ({errors} errors), "
                  f"{len(index_obj.functions)} functions indexed", file=sys.stderr)
    else:
        # Sequential for small projects
        clang_index = Index.create()
        parsed = 0
        errors = 0
        for filepath in cpp_files:
            args = _resolve_file_args(filepath, compile_db, default_args)
            try:
                functions, typedefs = parse_translation_unit(filepath, clang_index, args, project_dir)
                for func in functions:
                    index_obj.add_function(func)
                for td in typedefs:
                    index_obj.add_typedef(td)
                parsed += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR parsing {filepath}: {e}", file=sys.stderr)
            if parsed % 500 == 0 and parsed > 0:
                check_memory_limit(memory_limit_gb, label="index_project")
                print(f"  Parsed {parsed}/{len(cpp_files)} files, "
                      f"{len(index_obj.functions)} functions", file=sys.stderr)
        print(f"  Parsed {parsed} files ({errors} errors), "
              f"{len(index_obj.functions)} functions indexed", file=sys.stderr)

    # Build training documents (C/C++ code path -- unchanged).
    documents: list = []
    if cpp_files:
        documents = build_training_documents(
            index_obj,
            max_tokens,
            max_dep_depth,
            enriched=enriched,
            project_dir=project_dir,
            compile_db=compile_db,
            default_args=default_args,
            default_build_info=default_build_info,
            tokenizer_path=tokenizer_path,
            dedup_db=dedup_db,
            dedup_near=dedup_near,
        )
    check_memory_limit(memory_limit_gb, label="index_project")
    print(f"  Generated {len(documents)} code training documents", file=sys.stderr)

    # Emit build/compilation files as their own 'build' docs (ADDITIVE). Only in
    # enriched mode: build docs are dict-shaped (structure_ids/language_info/
    # platform_ids); plain-text mode would lose the build-system tag + sidecars.
    if enriched and build_files:
        build_docs = emit_build_documents(
            build_files,
            default_build_info=default_build_info,
            compile_index=_compile_index,
            tokenizer_path=tokenizer_path,
            dedup_db=dedup_db,
            dedup_near=dedup_near,
        )
        documents.extend(build_docs)
        check_memory_limit(memory_limit_gb, label="index_project")
    elif build_files and not enriched:
        print(
            "  WARN: build files found but --enriched not set; build docs are "
            "enriched-only and were NOT emitted",
            file=sys.stderr,
        )

    print(f"  Generated {len(documents)} total training documents", file=sys.stderr)

    stats = index_obj.stats()
    print(f"  Index stats: {stats}", file=sys.stderr)

    return documents


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Clang-based cross-file C++ dependency indexer')
    parser.add_argument('--project-dir', type=str,
                        help='Single project directory to process')
    parser.add_argument('--projects-list', type=str,
                        help='File listing project directories (one per line)')
    parser.add_argument('--projects-dir', type=str,
                        help='Directory containing multiple project subdirectories')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL file path')
    parser.add_argument('--max-tokens', type=int, default=16384,
                        help='Max tokens per training document (default: 16384)')
    parser.add_argument('--max-dep-depth', type=int, default=5,
                        help='Max dependency resolution depth (default: 5)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers for multi-project mode')
    parser.add_argument('--parse-workers', type=int, default=8,
                        help='Number of parallel parse workers within each project (default: 8)')
    parser.add_argument('--libclang-path', type=str, default=None,
                        help='Path to libclang.so (auto-detected if not set)')
    parser.add_argument('--append', action='store_true',
                        help='Append to output file instead of overwriting')
    parser.add_argument('--enriched', action='store_true',
                        help='Emit enriched output with structure_ids, chunk_boundaries, '
                             'call_edges, type_edges alongside text')
    parser.add_argument('--exclude-dirs', type=str, default=None,
                        help='Comma-separated extra directory names to exclude from file discovery')
    parser.add_argument('--memory-limit-gb', type=float, default=10.0,
                        help='Abort if this Python wrapper exceeds this max RSS in GiB (default: 10)')
    parser.add_argument('--dedup-db', type=str, default=None,
                        help='Path to the shared global dedup SQLite store. When set, '
                             'FUNCTION-LEVEL exact+near duplicates (on the tokenized '
                             'hash, BEFORE grouping) are dropped GLOBALLY across all '
                             'repos/streams (no in-RAM fallback). When absent, a '
                             'per-repo in-RAM exact set is used.')
    parser.add_argument('--tokenizer-path', type=str, default=None,
                        help='Path to OUR tokenizer.json. REQUIRED to enable the '
                             'function-level tokenized-hash dedup (corrected design). '
                             'Without it, no function-level dedup runs.')
    parser.add_argument('--no-near-dedup', action='store_true',
                        help='Disable MinHash-LSH near dedup (exact-only).')

    args = parser.parse_args()
    start_memory_guard(args.memory_limit_gb, label="index_project")

    try:
        _configure_libclang(args.libclang_path)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Collect project directories
    project_dirs = []
    if args.project_dir:
        project_dirs.append(args.project_dir)
    elif args.projects_list:
        with open(args.projects_list) as f:
            project_dirs = [line.strip() for line in f if line.strip()]
    elif args.projects_dir:
        for entry in sorted(os.listdir(args.projects_dir)):
            full = os.path.join(args.projects_dir, entry)
            if os.path.isdir(full):
                project_dirs.append(full)
    else:
        parser.error("Provide --project-dir, --projects-list, or --projects-dir")

    print(f"Processing {len(project_dirs)} project(s)", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"Max tokens: {args.max_tokens}", file=sys.stderr)
    print(f"Memory limit: {args.memory_limit_gb} GiB", file=sys.stderr)

    extra_exclude = set(args.exclude_dirs.split(',')) if args.exclude_dirs else None

    total_docs = 0
    # CORRECTED design: dedup is FUNCTION-LEVEL, on the tokenized hash, BEFORE
    # grouping -- it happens inside process_project -> build_training_documents ->
    # dedup_root_functions, backed by the shared global SQLite DedupStore at
    # --dedup-db (cross-repo + cross-stream, fail-loud). There is NO second-pass
    # doc-level dedup here. The per-process DedupStore opens against --dedup-db so
    # the multi-project ProcessPoolExecutor mode dedups globally via SQLite WAL.
    dedup_near = not args.no_near_dedup
    if args.dedup_db:
        if not args.tokenizer_path:
            print(
                "ERROR: --dedup-db requires --tokenizer-path (function-level "
                "dedup hashes the tokenized function ids).",
                file=sys.stderr,
            )
            return 1
        # FAIL LOUD up front: open once here so a bad db / missing datasketch
        # crashes before any heavy parsing (RULE #1). Closed immediately; each
        # process_project reopens against the same WAL db.
        from dedup_store import DedupStore
        DedupStore(args.dedup_db, near=dedup_near, commit_every=1000).close()
        print(
            f"Dedup: GLOBAL function-level store at {args.dedup_db} "
            f"(exact{'+near' if dedup_near else ''}, tokenized hash)",
            file=sys.stderr,
        )
    elif args.tokenizer_path:
        print("Dedup: per-repo in-RAM function-level exact set (no --dedup-db)",
              file=sys.stderr)
    else:
        print("Dedup: DISABLED (no --tokenizer-path given)", file=sys.stderr)

    append_mode = getattr(args, 'append', False)
    enriched = args.enriched
    with open(args.output, 'a' if append_mode else 'w') as out:
        if args.workers > 1 and len(project_dirs) > 1:
            # Multi-project parallel mode
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        process_project, pd, args.max_tokens, args.max_dep_depth,
                        args.parse_workers, enriched, extra_exclude,
                        args.memory_limit_gb, args.tokenizer_path, args.dedup_db,
                        dedup_near,
                    ): pd
                    for pd in project_dirs
                }
                for future in as_completed(futures):
                    pd = futures[future]
                    try:
                        docs = future.result()
                        for doc in docs:
                            if enriched:
                                json.dump(doc, out)
                            else:
                                json.dump({'text': doc}, out)
                            out.write('\n')
                            total_docs += 1
                    except Exception as e:
                        print(f"ERROR processing {pd}: {e}", file=sys.stderr)
        else:
            # Sequential mode
            for pd in project_dirs:
                try:
                    docs = process_project(pd, args.max_tokens, args.max_dep_depth,
                                           args.parse_workers, enriched, extra_exclude,
                                           args.memory_limit_gb, args.tokenizer_path,
                                           args.dedup_db, dedup_near)
                    for doc in docs:
                        if enriched:
                            json.dump(doc, out)
                        else:
                            json.dump({'text': doc}, out)
                        out.write('\n')
                        total_docs += 1
                    out.flush()
                except Exception as e:
                    print(f"ERROR processing {pd}: {e}", file=sys.stderr)
                    if len(project_dirs) == 1:
                        return 1
                    print("  Skipping project, continuing...", file=sys.stderr)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Total documents: {total_docs}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
