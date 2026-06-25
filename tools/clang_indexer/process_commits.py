#!/usr/bin/env python3
# ruff: noqa: E402
"""Process git commit diffs into enriched training documents using libclang.

Uses libclang for syntax tree metadata, qualified names, resolved call
references, and file-local dependency ordering.

Pipeline:
  1. Read JSONL records: {old_content, new_content, diff, subject, body, filepath, repo}
  2. Parse old/new content with libclang (via temp files)
  3. Extract functions/classes, build call graph, compute dep levels
  4. Parse diff hunks → find changed functions → BFS transitive deps → build chains
  5. Format PRE-COMMIT/POST-COMMIT documents with enriched metadata
  6. Tokenize, enforce max-tokens budget, output enriched JSONL

Usage:
    python3 tools/clang_indexer/process_commits.py \
        --inputs raw_commits.jsonl \
        --output enriched_commits.jsonl \
        --max-tokens 4096 --format both

    # Then convert to parquet:
    python3 scripts/data/clang_enriched_to_4k_parquet.py \
        --input-file enriched_commits.jsonl \
        --output-file enriched_commits.parquet \
        --overflow-policy drop
"""

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Protocol, cast

# Increase recursion limit for deeply nested ASTs
sys.setrecursionlimit(50000)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse infrastructure from index_project.py
from tools.clang_indexer.index_project import (
    _configure_libclang,
    _adapt_args_for_file,
    _sanitize_compile_args_for_clang,
    FunctionDef,
    PartInfo,
    ProjectIndex,
    extract_callees,
    get_qualified_name,
    FUNCTION_KINDS,
    CONTAINER_KINDS,
    _byte_to_char_mapper,
    _cursor_text_and_metadata,
    extract_clang_ast_metadata,
    extract_semantic_metadata_from_parts,
)
from cppmega_mlx.data.nanochat_pipeline.build_context import detect_build_context
from scripts.nanochat_data.memory_guard import check_memory_limit, start_memory_guard

if TYPE_CHECKING:
    from clang.cindex import CursorKind, Index, TranslationUnit  # pyright: ignore[reportMissingImports]
else:
    CursorKind = object
    Index = object
    TranslationUnit = object


class _ClangRuntimeSurface(Protocol):
    Index: type[object]
    TranslationUnit: type[object]
    CursorKind: object


clang_runtime: _ClangRuntimeSurface | None
try:
    clang_runtime = cast(_ClangRuntimeSurface, importlib.import_module("clang.cindex"))
except ImportError:
    clang_runtime = None
else:
    if not TYPE_CHECKING:
        Index = cast(type[object], clang_runtime.Index)  # type: ignore[assignment]
        TranslationUnit = cast(type[object], clang_runtime.TranslationUnit)  # type: ignore[assignment]
        CursorKind = clang_runtime.CursorKind  # type: ignore[assignment]

detect_language_info: Callable[..., dict[str, object] | None] | None
try:
    from cppmega_mlx.data.nanochat_pipeline.language_info import detect_language_info
except ImportError:
    detect_language_info = None

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClassDef:
    """A class/struct definition with source span."""
    name: str
    qualified_name: str
    text: str
    start_line: int
    end_line: int
    ast_depth: list[int] = field(default_factory=list)
    sibling_index: list[int] = field(default_factory=list)
    ast_node_type: list[int] = field(default_factory=list)


@dataclass
class FileAnalysis:
    """Single-file analysis result from libclang."""
    preamble: str
    functions: list[FunctionDef] = field(default_factory=list)
    classes: list[ClassDef] = field(default_factory=list)
    preamble_ast_depth: list[int] = field(default_factory=list)
    preamble_sibling_index: list[int] = field(default_factory=list)
    preamble_ast_node_type: list[int] = field(default_factory=list)
    compile_args: list[str] = field(default_factory=list)
    build_info: dict[str, object] = field(default_factory=dict)

    def build_local_index(self) -> ProjectIndex:
        """Build a ProjectIndex from this file's functions for dep computation."""
        idx = ProjectIndex()
        for func in self.functions:
            idx.add_function(func)
        idx.compute_dep_levels()
        return idx


@dataclass
class HunkRange:
    """Parsed unified diff hunk range."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')

EDIT_OP_UNCHANGED = 0
EDIT_OP_INSERTED = 1
EDIT_OP_MODIFIED = 2
EDIT_OP_CONTEXT = 3


def parse_hunk_ranges(diff: str) -> list[HunkRange]:
    """Parse unified diff @@ lines into hunk ranges."""
    return [
        HunkRange(
            old_start=int(m.group(1)),
            old_count=int(m.group(2) or '1'),
            new_start=int(m.group(3)),
            new_count=int(m.group(4) or '1'),
        )
        for m in _HUNK_RE.finditer(diff)
    ]


def changed_lines(hunks: list[HunkRange], use_old: bool) -> set[int]:
    """Get set of line numbers touched by hunks (1-based)."""
    lines: set[int] = set()
    for h in hunks:
        start = h.old_start if use_old else h.new_start
        count = max(h.old_count if use_old else h.new_count, 1)
        lines.update(range(start, start + count))
    return lines


def compute_new_line_edit_ops(diff: str) -> tuple[dict[int, int], dict[int, int]]:
    """Map 1-based new-file lines to (edit-op id, 0-based hunk index).

    Returns ``(ops, line_hunk)`` where ``ops[new_line] -> EDIT_OP_*`` and
    ``line_hunk[new_line] -> hunk_index`` (0 for the first ``@@`` hunk, 1 for the
    second, ...). The hunk index lets the per-token ``hunk_id`` carry which edit
    region a token belongs to, not just a binary changed flag.
    """
    ops: dict[int, int] = {}
    line_hunk: dict[int, int] = {}
    current_new_line: int | None = None
    pending_removes = 0
    hunk_idx = -1
    for line in diff.splitlines():
        match = _HUNK_RE.match(line)
        if match:
            current_new_line = int(match.group(3))
            pending_removes = 0
            hunk_idx += 1
            continue
        if current_new_line is None:
            continue
        if line.startswith('-') and not line.startswith('---'):
            pending_removes += 1
            continue
        if line.startswith('+') and not line.startswith('+++'):
            if pending_removes > 0:
                ops[current_new_line] = EDIT_OP_MODIFIED
                pending_removes -= 1
            else:
                ops[current_new_line] = EDIT_OP_INSERTED
            line_hunk[current_new_line] = hunk_idx
            current_new_line += 1
            continue
        pending_removes = 0
        current_new_line += 1
    return ops, line_hunk


def compute_old_line_hunk(hunks: list[HunkRange]) -> dict[int, int]:
    """Map 1-based OLD-file lines to the 0-based hunk index that touches them."""
    line_hunk: dict[int, int] = {}
    for hunk_idx, h in enumerate(hunks):
        for line_no in range(h.old_start, h.old_start + max(h.old_count, 1)):
            line_hunk.setdefault(line_no, hunk_idx)
    return line_hunk


def _line_ranges_by_number(text: str) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    offset = 0
    for line_no, line in enumerate(text.splitlines(keepends=True), start=1):
        end = offset + len(line)
        ranges[line_no] = (offset, end)
        offset = end
    if text and (not text.endswith(('\n', '\r'))):
        ranges.setdefault(len(ranges) + 1, (offset, len(text)))
    return ranges


def _line_ranges_for_changed_functions(
    analysis: FileAnalysis,
    changed_line_numbers: set[int],
) -> list[tuple[int, int]]:
    if not analysis.functions and not analysis.classes:
        return []
    ranges: list[tuple[int, int]] = []
    for func in analysis.functions:
        if any(line in changed_line_numbers for line in range(func.line, func.end_line + 1)):
            ranges.append((func.line, func.end_line))
    for cls in analysis.classes:
        if any(line in changed_line_numbers for line in range(cls.start_line, cls.end_line + 1)):
            ranges.append((cls.start_line, cls.end_line))
    return ranges


def _build_commit_temporal_metadata(
    full_text: str,
    part_texts: list[str],
    section_kinds: list[str],
    *,
    record: dict,
    old_analysis: FileAnalysis,
    new_analysis: Optional[FileAnalysis],
) -> dict[str, list[int]]:
    text_len = len(full_text)
    change_mask_pre = [0] * text_len
    change_mask_post = [0] * text_len
    hunk_id_per_char = [-1] * text_len  # -1 = unchanged/context (no hunk)
    edit_op_per_char = [EDIT_OP_CONTEXT] * text_len

    diff = str(record.get('diff', '') or '')
    hunks = parse_hunk_ranges(diff)
    if not hunks:
        return {
            'change_mask_pre': [],
            'change_mask_post': [],
            'hunk_id_per_char': [],
            'edit_op_per_char': [],
        }

    old_changed_lines = changed_lines(hunks, use_old=True)
    new_line_ops, new_line_hunk = compute_new_line_edit_ops(diff)
    old_line_hunk = compute_old_line_hunk(hunks)
    old_changed_ranges = _line_ranges_for_changed_functions(old_analysis, old_changed_lines)
    old_content = str(record.get('old_content', '') or '')
    new_content = str(record.get('new_content', '') or '')
    old_line_ranges = _line_ranges_by_number(str(record.get('old_content', '') or ''))
    new_line_ranges = _line_ranges_by_number(str(record.get('new_content', '') or ''))
    # Build text->op and the CONSISTENT text->hunk map in one pass so any text
    # marked changed also carries its hunk index (a changed line always has a
    # hunk via compute_new_line_edit_ops).
    new_line_ops_by_text: dict[str, int] = {}
    new_line_hunk_by_text: dict[str, int] = {}
    for line_no, (start, end) in new_line_ranges.items():
        stripped = new_content[start:end].strip()
        if not stripped:
            continue
        op = new_line_ops.get(line_no, EDIT_OP_UNCHANGED)
        previous = new_line_ops_by_text.get(stripped)
        if previous is None or (previous == EDIT_OP_UNCHANGED and op != EDIT_OP_UNCHANGED):
            new_line_ops_by_text[stripped] = op
            new_line_hunk_by_text[stripped] = new_line_hunk.get(line_no, -1)

    def old_part_hunk(part: str) -> int:
        """Hunk index of the first hunk overlapping a changed OLD part, else -1."""
        if not part.strip() or not old_changed_ranges:
            return -1
        for start_line, end_line in old_changed_ranges:
            start_end = old_line_ranges.get(start_line)
            end_end = old_line_ranges.get(end_line)
            if not start_end or not end_end:
                continue
            changed_text = old_content[start_end[0]:end_end[1]].strip()
            if changed_text and (part.strip() in changed_text or changed_text in part.strip()):
                for ln in range(start_line, end_line + 1):
                    if ln in old_line_hunk:
                        return old_line_hunk[ln]
                return -1
        return -1

    def new_line_op(part_line: str) -> int:
        stripped = part_line.strip()
        if not stripped:
            return EDIT_OP_CONTEXT
        return new_line_ops_by_text.get(stripped, EDIT_OP_UNCHANGED)

    def new_line_hunk_for(part_line: str) -> int:
        stripped = part_line.strip()
        if not stripped:
            return -1
        return new_line_hunk_by_text.get(stripped, -1)

    offset = 0
    for idx, part in enumerate(part_texts):
        source_kind = section_kinds[idx] if idx < len(section_kinds) else 'c'
        part_len = len(part)
        if part_len <= 0:
            continue
        part_end = min(offset + part_len, text_len)
        if source_kind == 'n':
            pos = offset
            for line in part.splitlines(keepends=True):
                line_len = len(line)
                op = new_line_op(line)
                changed = int(op in (EDIT_OP_INSERTED, EDIT_OP_MODIFIED))
                h_idx = new_line_hunk_for(line) if changed else -1
                # RULE #1: a changed new line that maps to no hunk is a real
                # inconsistency (op and hunk maps are built together), not a
                # degraded value -> fail loud rather than emit a bogus id.
                if changed and h_idx < 0:
                    raise ValueError(
                        f"process_commits: changed new line maps to no hunk in "
                        f"{record.get('commit_hash')}:{record.get('filepath')}")
                end = min(pos + line_len, text_len)
                for char_idx in range(pos, end):
                    change_mask_post[char_idx] = changed
                    hunk_id_per_char[char_idx] = h_idx
                    edit_op_per_char[char_idx] = op
                pos = end
        elif source_kind == 'o':
            h_idx = old_part_hunk(part)
            changed = int(h_idx >= 0)
            for char_idx in range(offset, part_end):
                change_mask_pre[char_idx] = changed
                hunk_id_per_char[char_idx] = h_idx
                edit_op_per_char[char_idx] = EDIT_OP_CONTEXT
        else:
            for char_idx in range(offset, part_end):
                edit_op_per_char[char_idx] = EDIT_OP_CONTEXT
        offset += part_len
        if idx < len(part_texts) - 1:
            offset += 2

    if not any(change_mask_pre) and not any(change_mask_post):
        return {
            'change_mask_pre': [],
            'change_mask_post': [],
            'hunk_id_per_char': [],
            'edit_op_per_char': [],
        }
    return {
        'change_mask_pre': change_mask_pre,
        'change_mask_post': change_mask_post,
        'hunk_id_per_char': hunk_id_per_char,
        'edit_op_per_char': edit_op_per_char,
    }


# ---------------------------------------------------------------------------
# Clang single-file analysis
# ---------------------------------------------------------------------------

_C_EXTENSIONS = {'.c'}
_SIMPLE_FALLBACK_ARGS_CPP = ['-fsyntax-only', '-Wno-everything', '-std=c++17']
_SIMPLE_FALLBACK_ARGS_C = ['-x', 'c', '-fsyntax-only', '-Wno-everything', '-std=c11']


def _same_clang_path(left: str | None, right: str) -> bool:
    if not left:
        return False
    return os.path.normcase(os.path.normpath(left)) == os.path.normcase(os.path.normpath(right))


def _add_unique_include(args: list[str], include_dir: str) -> None:
    if not include_dir:
        return
    include_arg = f"-I{os.path.normpath(include_dir)}"
    if include_arg not in args:
        args.append(include_arg)


def _analysis_compile_args(
    filepath: str,
    compile_args: list[str] | None,
    repo_root: str | None,
) -> list[str]:
    ext = Path(filepath).suffix.lower() or '.cpp'
    if compile_args:
        args = _sanitize_compile_args_for_clang(list(compile_args))
    elif ext in _C_EXTENSIONS:
        args = list(_SIMPLE_FALLBACK_ARGS_C)
    else:
        args = list(_SIMPLE_FALLBACK_ARGS_CPP)

    if repo_root:
        _add_unique_include(args, repo_root)
        _add_unique_include(args, os.path.dirname(os.path.join(repo_root, filepath)))
    return _adapt_args_for_file(args, filepath)


def _extract_preamble_from_source(
    tu: TranslationUnit,
    filename: str,
    source: str,
    byte_to_char: Callable[[int], int],
    ast_depth: list[int],
    sibling_index: list[int],
    ast_node_type: list[int],
) -> tuple[str, list[int], list[int], list[int]]:
    texts: list[str] = []
    depth_parts: list[int] = []
    sibling_parts: list[int] = []
    node_type_parts: list[int] = []
    for cursor in tu.cursor.get_children():
        cursor_file = cursor.location.file.name if cursor.location.file else None
        if not _same_clang_path(cursor_file, filename):
            continue
        if cursor.kind in (
            CursorKind.INCLUSION_DIRECTIVE,
            CursorKind.USING_DIRECTIVE,
            CursorKind.USING_DECLARATION,
            CursorKind.TYPEDEF_DECL,
            CursorKind.TYPE_ALIAS_DECL,
            CursorKind.NAMESPACE_ALIAS,
        ):
            text, depth, sibling, node_type, _offsets = _cursor_text_and_metadata(
                cursor,
                source,
                filename,
                byte_to_char,
                ast_depth,
                sibling_index,
                ast_node_type,
            )
            if not text:
                continue
            if texts:
                depth_parts.append(0)
                sibling_parts.append(0)
                node_type_parts.append(0)
            texts.append(text)
            depth_parts.extend(depth)
            sibling_parts.extend(sibling)
            node_type_parts.extend(node_type)
    return "\n".join(texts), depth_parts, sibling_parts, node_type_parts


class BuildContextResolver:
    """Resolve per-record compile context without reloading build files per row."""

    def __init__(self, repo_root: str | None = None, repo_dir: str | None = None) -> None:
        self.repo_root = os.path.abspath(repo_root) if repo_root else None
        self.repo_dir = os.path.abspath(repo_dir) if repo_dir else None
        self._cache: dict[str, tuple[dict, list[str], object | None]] = {}

    def _record_repo_root(self, record: dict) -> str | None:
        explicit = record.get("repo_path")
        if isinstance(explicit, str) and explicit:
            return os.path.abspath(explicit)
        if self.repo_root:
            return self.repo_root
        repo_name = record.get("repo")
        if self.repo_dir and isinstance(repo_name, str) and repo_name:
            candidate = os.path.join(self.repo_dir, repo_name)
            if os.path.isdir(candidate):
                return os.path.abspath(candidate)
        return None

    def _load(self, repo_root: str) -> tuple[dict, list[str], object | None]:
        cached = self._cache.get(repo_root)
        if cached is not None:
            return cached
        context = detect_build_context(repo_root)
        self._cache[repo_root] = context
        return context

    def resolve(self, record: dict) -> tuple[str | None, list[str] | None, dict[str, object]]:
        compile_args = record.get("compile_args")
        build_info = record.get("build_info")
        args = [str(arg) for arg in compile_args] if isinstance(compile_args, list) else None
        info = dict(build_info) if isinstance(build_info, dict) else {}

        repo_root = self._record_repo_root(record)
        if repo_root is None or not os.path.isdir(repo_root):
            return None, args, info

        platform_info, default_args, compile_index = self._load(repo_root)
        if not info:
            info = {
                key: value
                for key, value in platform_info.items()
                if key in {"build_system", "source", "compiler", "standard"} and value is not None
            }
        filepath = str(record.get("filepath") or "")
        if compile_index is not None and filepath:
            lookup_path = filepath if os.path.isabs(filepath) else os.path.join(repo_root, filepath)
            lookup = getattr(compile_index, "lookup", None)
            if callable(lookup):
                file_args, file_info = lookup(lookup_path)
                if file_args:
                    args = list(file_args)
                if file_info:
                    info = {**info, **dict(file_info)}
        if args is None:
            args = list(default_args)
        return repo_root, args, info


def analyze_file_clang(
    content: str,
    filepath: str,
    clang_index: Index,
    tmpdir: str,
    *,
    compile_args: list[str] | None = None,
    repo_root: str | None = None,
    build_info: dict[str, object] | None = None,
) -> FileAnalysis:
    """Parse a single file's content with libclang and extract functions/classes.

    Prefer the original repository path plus libclang unsaved_files so quoted
    includes and compile_commands flags describe the real translation unit.
    Fall back to a temp source only when no repository root is available.
    """
    if not content or len(content) < 20:
        return FileAnalysis(preamble='')

    ext = Path(filepath).suffix.lower() or '.cpp'
    os.makedirs(tmpdir, exist_ok=True)
    if repo_root:
        source_path = filepath if os.path.isabs(filepath) else os.path.join(repo_root, filepath)
        source_path = os.path.normpath(source_path)
        unsaved_files = [(source_path, content)]
    else:
        source_path = os.path.join(tmpdir, f"source{ext}")
        unsaved_files = None
        with open(source_path, 'w', errors='replace') as f:
            f.write(content)

    args = _analysis_compile_args(source_path, compile_args, repo_root)

    try:
        tu = clang_index.parse(
            source_path,
            args=args,
            unsaved_files=unsaved_files,
            options=(
                TranslationUnit.PARSE_INCOMPLETE
                | TranslationUnit.PARSE_PRECOMPILED_PREAMBLE
            ),
        )
    except Exception:
        return FileAnalysis(preamble='')

    ast_depth, sibling_index, ast_node_type = extract_clang_ast_metadata(
        content,
        tu,
        source_path,
    )
    byte_to_char = _byte_to_char_mapper(content)

    # Extract preamble
    (
        preamble,
        preamble_ast_depth,
        preamble_sibling_index,
        preamble_ast_node_type,
    ) = _extract_preamble_from_source(
        tu,
        source_path,
        content,
        byte_to_char,
        ast_depth,
        sibling_index,
        ast_node_type,
    )

    # Extract functions and classes
    functions: list[FunctionDef] = []
    classes: list[ClassDef] = []

    def visit(cursor):
        if not cursor.location.file:
            return
        if not _same_clang_path(cursor.location.file.name, source_path):
            return

        if cursor.kind in FUNCTION_KINDS and cursor.is_definition():
            text, func_ast_depth, func_sibling_index, func_ast_node_type, _offsets = (
                _cursor_text_and_metadata(
                    cursor,
                    content,
                    source_path,
                    byte_to_char,
                    ast_depth,
                    sibling_index,
                    ast_node_type,
                )
            )
            if text and len(text) >= 20:
                callees = extract_callees(cursor)
                qname = get_qualified_name(cursor)
                start_line = cursor.extent.start.line
                end_line = cursor.extent.end.line
                functions.append(FunctionDef(
                    name=cursor.spelling,
                    qualified_name=qname,
                    file=filepath,
                    line=start_line,
                    text=text,
                    callees=callees,
                    is_definition=True,
                    end_line=end_line,
                    ast_depth=func_ast_depth,
                    sibling_index=func_sibling_index,
                    ast_node_type=func_ast_node_type,
                ))

        elif cursor.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL,
                              CursorKind.CLASS_TEMPLATE):
            if cursor.is_definition():
                text, cls_ast_depth, cls_sibling_index, cls_ast_node_type, _offsets = (
                    _cursor_text_and_metadata(
                        cursor,
                        content,
                        source_path,
                        byte_to_char,
                        ast_depth,
                        sibling_index,
                        ast_node_type,
                    )
                )
                if text and len(text) >= 20:
                    qname = get_qualified_name(cursor)
                    classes.append(ClassDef(
                        name=cursor.spelling,
                        qualified_name=qname,
                        text=text,
                        start_line=cursor.extent.start.line,
                        end_line=cursor.extent.end.line,
                        ast_depth=cls_ast_depth,
                        sibling_index=cls_sibling_index,
                        ast_node_type=cls_ast_node_type,
                    ))
                # Recurse into class children to extract inline methods,
                # constructors, and destructors defined inside the class body.
                for child in cursor.get_children():
                    visit(child)

        elif cursor.kind in CONTAINER_KINDS:
            for child in cursor.get_children():
                visit(child)

    for cursor in tu.cursor.get_children():
        visit(cursor)

    return FileAnalysis(
        preamble=preamble,
        functions=functions,
        classes=classes,
        preamble_ast_depth=preamble_ast_depth,
        preamble_sibling_index=preamble_sibling_index,
        preamble_ast_node_type=preamble_ast_node_type,
        compile_args=args,
        build_info=dict(build_info or {}),
    )


# ---------------------------------------------------------------------------
# Changed function/class detection
# ---------------------------------------------------------------------------

def find_changed_functions(
    analysis: FileAnalysis,
    changed_line_set: set[int],
) -> list[int]:
    """Return indices of functions whose line spans overlap changed lines."""
    result = []
    for i, func in enumerate(analysis.functions):
        end_line = func.end_line
        func_lines = set(range(func.line, end_line + 1))
        if func_lines & changed_line_set:
            result.append(i)
    return result


def find_changed_classes(
    analysis: FileAnalysis,
    changed_line_set: set[int],
) -> list[int]:
    """Return indices of classes whose line spans overlap changed lines."""
    result = []
    for i, cls in enumerate(analysis.classes):
        cls_lines = set(range(cls.start_line, cls.end_line + 1))
        if cls_lines & changed_line_set:
            result.append(i)
    return result


# ---------------------------------------------------------------------------
# Symbol-level change tracking
# ---------------------------------------------------------------------------

@dataclass
class SymbolEntry:
    """A symbol (function or class) with a stable integer ID.

    ID scheme: functions are numbered 0..N-1, classes are numbered N..N+M-1
    where N = len(analysis.functions), M = len(analysis.classes).
    """
    id: int
    name: str
    qualified_name: str
    kind: str  # 'function' or 'class'
    start_line: int
    end_line: int


def build_symbol_table(analysis: FileAnalysis) -> list[SymbolEntry]:
    """Build a flat symbol table from functions and classes.

    Functions get IDs 0..N-1, classes get IDs N..N+M-1.
    """
    symbols: list[SymbolEntry] = []
    for i, func in enumerate(analysis.functions):
        symbols.append(SymbolEntry(
            id=i,
            name=func.name,
            qualified_name=func.qualified_name,
            kind='function',
            start_line=func.line,
            end_line=func.end_line,
        ))
    n_funcs = len(analysis.functions)
    for j, cls in enumerate(analysis.classes):
        symbols.append(SymbolEntry(
            id=n_funcs + j,
            name=cls.name,
            qualified_name=cls.qualified_name,
            kind='class',
            start_line=cls.start_line,
            end_line=cls.end_line,
        ))
    return symbols


def compute_changed_symbol_ids(
    analysis: FileAnalysis,
    changed_line_set: set[int],
) -> list[int]:
    """Return stable IDs of symbols whose line spans overlap changed lines.

    Uses the same ID scheme as build_symbol_table(): functions first, then
    classes.
    """
    ids: list[int] = []
    for i, func in enumerate(analysis.functions):
        func_lines = set(range(func.line, func.end_line + 1))
        if func_lines & changed_line_set:
            ids.append(i)
    n_funcs = len(analysis.functions)
    for j, cls in enumerate(analysis.classes):
        cls_lines = set(range(cls.start_line, cls.end_line + 1))
        if cls_lines & changed_line_set:
            ids.append(n_funcs + j)
    return ids


def compute_ripple_candidates(
    analysis: FileAnalysis,
    changed_symbol_ids: list[int],
    max_depth: int = 3,
) -> list[dict]:
    """For each changed symbol, find callers and type dependents that might
    need updating.

    Returns a list of dicts:
        {
            'changed_symbol_id': int,
            'changed_symbol_name': str,
            'candidates': [
                {
                    'symbol_id': int,
                    'symbol_name': str,
                    'relation': str,  # 'caller' | 'type_dependent' | 'co_member'
                    'depth': int,     # BFS depth from the changed symbol
                }
            ]
        }

    Ripple relations:
    - 'caller': a function that calls the changed function (reverse edge)
    - 'type_dependent': a function that uses a changed class as parameter/return type
    - 'co_member': another method in the same class as a changed method
    """
    if not changed_symbol_ids:
        return []

    symbols = build_symbol_table(analysis)
    id_to_sym = {s.id: s for s in symbols}
    n_funcs = len(analysis.functions)

    # Build reverse call graph: callee_qname -> list of caller symbol IDs
    callee_to_callers: dict[str, list[int]] = defaultdict(list)
    for i, func in enumerate(analysis.functions):
        for callee_qname in func.callees:
            callee_to_callers[callee_qname].append(i)

    # Build class membership: class_qname -> list of function symbol IDs
    # (functions whose qualified_name starts with "ClassName::")
    class_members: dict[str, list[int]] = defaultdict(list)
    for j, cls in enumerate(analysis.classes):
        prefix = cls.qualified_name + '::'
        for i, func in enumerate(analysis.functions):
            if func.qualified_name.startswith(prefix):
                class_members[cls.qualified_name].append(i)

    # Build type usage: for each function, which classes appear in its line range
    # (heuristic: class name appears in function text)
    func_uses_class: dict[int, list[int]] = defaultdict(list)
    for i, func in enumerate(analysis.functions):
        for j, cls in enumerate(analysis.classes):
            if cls.name in func.text:
                func_uses_class[i].append(n_funcs + j)

    result = []
    for changed_id in changed_symbol_ids:
        sym = id_to_sym.get(changed_id)
        if sym is None:
            continue

        candidates: list[dict] = []
        seen: set[int] = {changed_id}

        if sym.kind == 'function':
            # BFS through reverse call graph to find callers
            queue = deque([(sym.qualified_name, 1)])
            visited_qnames: set[str] = {sym.qualified_name}
            while queue:
                qname, depth = queue.popleft()
                if depth > max_depth:
                    continue
                for caller_id in callee_to_callers.get(qname, []):
                    if caller_id not in seen:
                        seen.add(caller_id)
                        caller_func = analysis.functions[caller_id]
                        candidates.append({
                            'symbol_id': caller_id,
                            'symbol_name': caller_func.qualified_name,
                            'relation': 'caller',
                            'depth': depth,
                        })
                        if depth < max_depth and caller_func.qualified_name not in visited_qnames:
                            visited_qnames.add(caller_func.qualified_name)
                            queue.append((caller_func.qualified_name, depth + 1))

            # co_member: other methods in the same class
            for cls_qname, member_ids in class_members.items():
                if changed_id in member_ids:
                    for mid in member_ids:
                        if mid not in seen:
                            seen.add(mid)
                            candidates.append({
                                'symbol_id': mid,
                                'symbol_name': analysis.functions[mid].qualified_name,
                                'relation': 'co_member',
                                'depth': 1,
                            })

        elif sym.kind == 'class':
            cls_idx = changed_id - n_funcs
            cls = analysis.classes[cls_idx]
            # type_dependent: functions that mention this class
            for i, func in enumerate(analysis.functions):
                if i not in seen and cls.name in func.text:
                    seen.add(i)
                    candidates.append({
                        'symbol_id': i,
                        'symbol_name': func.qualified_name,
                        'relation': 'type_dependent',
                        'depth': 1,
                    })
            # co_member: methods of this class
            for mid in class_members.get(cls.qualified_name, []):
                if mid not in seen:
                    seen.add(mid)
                    candidates.append({
                        'symbol_id': mid,
                        'symbol_name': analysis.functions[mid].qualified_name,
                        'relation': 'co_member',
                        'depth': 1,
                    })

        if candidates:
            result.append({
                'changed_symbol_id': changed_id,
                'changed_symbol_name': sym.qualified_name,
                'candidates': candidates,
            })

    return result


# ---------------------------------------------------------------------------
# Transitive dependency chain extraction
# ---------------------------------------------------------------------------

def extract_function_chain(
    analysis: FileAnalysis,
    changed_indices: list[int],
    max_depth: int = 5,
) -> list[FunctionDef]:
    """Extract changed functions + transitive deps, sorted by dep_level (leaves first)."""
    if not changed_indices:
        return []

    # Build local index for dep computation
    idx = analysis.build_local_index()

    # Collect all transitive deps for each changed function
    all_qnames = set()
    for i in changed_indices:
        func = analysis.functions[i]
        all_qnames.add(func.qualified_name)
        # BFS through local call graph
        visited = {func.qualified_name}
        queue = deque([(func.qualified_name, 0)])
        while queue:
            qname, depth = queue.popleft()
            if depth >= max_depth:
                continue
            f = idx.functions.get(qname)
            if not f:
                continue
            for callee in f.callees:
                if callee not in visited and callee in idx.functions:
                    visited.add(callee)
                    all_qnames.add(callee)
                    queue.append((callee, depth + 1))

    # Collect FunctionDefs, sorted by dep_level (leaves first)
    result = []
    for func in analysis.functions:
        if func.qualified_name in all_qnames:
            result.append(func)
    result.sort(key=lambda f: f.dep_level)
    return result


# ---------------------------------------------------------------------------
# Document formatting
# ---------------------------------------------------------------------------

def build_docstring(record: dict) -> str:
    """Build C++ docstring comment from commit metadata."""
    parts = ['/**']
    subject = record.get('subject', '').strip()
    if subject:
        parts.append(f' * @brief {subject}')
        parts.append(' *')

    repo = record.get('repo', '')
    filepath = record.get('filepath', '')
    if repo:
        parts.append(f' * Repository: {repo}')
    if filepath:
        parts.append(f' * File: {filepath}')

    body = record.get('body', '').strip()
    if body:
        parts.append(' *')
        for line in body.splitlines()[:8]:
            parts.append(f' * {line.rstrip()}')

    parts.append(' */')
    return '\n'.join(parts)


def format_chain_document(
    record: dict,
    old_analysis: FileAnalysis,
    new_analysis: FileAnalysis,
    hunks: list[HunkRange],
    max_dep_depth: int = 5,
) -> Optional[dict]:
    """Format A: PRE-COMMIT chain → POST-COMMIT chain with enriched metadata."""
    old_lines = changed_lines(hunks, use_old=True)
    new_lines = changed_lines(hunks, use_old=False)

    old_changed_funcs = find_changed_functions(old_analysis, old_lines)
    new_changed_funcs = find_changed_functions(new_analysis, new_lines)
    old_changed_classes = find_changed_classes(old_analysis, old_lines)
    new_changed_classes = find_changed_classes(new_analysis, new_lines)

    if (not old_changed_funcs and not new_changed_funcs
            and not old_changed_classes and not new_changed_classes):
        return None

    old_chain = extract_function_chain(old_analysis, old_changed_funcs, max_dep_depth)
    new_chain = extract_function_chain(new_analysis, new_changed_funcs, max_dep_depth)

    # Build parts_info: (text, kind, dep_level, name, qname)
    parts_info: list[PartInfo] = []
    section_kinds: list[str] = []

    # Docstring (kind=6: comment)
    docstring = build_docstring(record)
    parts_info.append((docstring, 6, 0, '', None))
    section_kinds.append('c')

    # PRE-COMMIT section
    if old_chain or old_changed_classes:
        parts_info.append(('// === PRE-COMMIT ===', 0, 0, '', None))
        section_kinds.append('c')
        short_preamble = '\n'.join(old_analysis.preamble.splitlines()[:20])
        if short_preamble:
            parts_info.append((short_preamble, 1, 0, '', None))
            section_kinds.append('o')
        for ci in old_changed_classes:
            cls = old_analysis.classes[ci]
            parts_info.append((cls.text, 4, 0, cls.name, cls.qualified_name))
            section_kinds.append('o')
        for func in old_chain:
            parts_info.append((func.text, 3, func.dep_level, func.name, func.qualified_name))
            section_kinds.append('o')

    # POST-COMMIT section
    subject = record.get('subject', '').strip()
    if new_chain or new_changed_classes:
        parts_info.append((f'// === POST-COMMIT: {subject} ===', 0, 0, '', None))
        section_kinds.append('c')
        short_preamble = '\n'.join(new_analysis.preamble.splitlines()[:20])
        if short_preamble:
            parts_info.append((short_preamble, 1, 0, '', None))
            section_kinds.append('n')
        for ci in new_changed_classes:
            cls = new_analysis.classes[ci]
            parts_info.append((cls.text, 4, 0, cls.name, cls.qualified_name))
            section_kinds.append('n')
        for func in new_chain:
            parts_info.append((func.text, 3, func.dep_level, func.name, func.qualified_name))
            section_kinds.append('n')

    if len(parts_info) <= 1:
        return None

    # Compute changed_symbol_ids and ripple_candidates from the NEW analysis
    # (post-commit state is what the model learns to predict).  Fall back to
    # old analysis when new is empty.
    primary = new_analysis if new_analysis.functions or new_analysis.classes else old_analysis
    csids = compute_changed_symbol_ids(primary, new_lines if (new_analysis.functions or new_analysis.classes) else old_lines)
    ripple = compute_ripple_candidates(primary, csids)

    return _build_enriched_from_parts(
        parts_info, old_analysis, new_analysis, record,
        changed_symbol_ids=csids, ripple_candidates=ripple,
        section_kinds=section_kinds,
    )


def format_diff_document(
    record: dict,
    old_analysis: FileAnalysis,
    hunks: list[HunkRange],
    max_dep_depth: int = 5,
) -> Optional[dict]:
    """Format B: Context code from old version + raw unified diff."""
    old_lines = changed_lines(hunks, use_old=True)
    old_changed_funcs = find_changed_functions(old_analysis, old_lines)
    old_changed_classes = find_changed_classes(old_analysis, old_lines)

    if not old_changed_funcs and not old_changed_classes:
        return None

    old_chain = extract_function_chain(old_analysis, old_changed_funcs, max_dep_depth)

    parts_info: list[PartInfo] = []
    section_kinds: list[str] = []

    # Docstring
    docstring = build_docstring(record)
    parts_info.append((docstring, 6, 0, '', None))
    section_kinds.append('c')

    # Context: old code chain
    parts_info.append(('// === CONTEXT ===', 0, 0, '', None))
    section_kinds.append('c')
    short_preamble = '\n'.join(old_analysis.preamble.splitlines()[:20])
    if short_preamble:
        parts_info.append((short_preamble, 1, 0, '', None))
        section_kinds.append('o')
    for ci in old_changed_classes:
        cls = old_analysis.classes[ci]
        parts_info.append((cls.text, 4, 0, cls.name, cls.qualified_name))
        section_kinds.append('o')
    for func in old_chain:
        parts_info.append((func.text, 3, func.dep_level, func.name, func.qualified_name))
        section_kinds.append('o')

    # Raw diff
    diff_text = record.get('diff', '')
    if diff_text:
        parts_info.append(('// === DIFF ===', 0, 0, '', None))
        section_kinds.append('c')
        parts_info.append((diff_text, 0, 0, '', None))
        section_kinds.append('c')

    if len(parts_info) <= 2:
        return None

    # Compute changed_symbol_ids and ripple_candidates from old analysis
    csids = compute_changed_symbol_ids(old_analysis, old_lines)
    ripple = compute_ripple_candidates(old_analysis, csids)

    return _build_enriched_from_parts(
        parts_info, old_analysis, None, record,
        changed_symbol_ids=csids, ripple_candidates=ripple,
        section_kinds=section_kinds,
    )


def _analysis_ast_maps(
    analysis: FileAnalysis | None,
) -> dict[str, tuple[list[int], list[int], list[int]]]:
    if analysis is None:
        return {}
    result: dict[str, tuple[list[int], list[int], list[int]]] = {}
    for func in analysis.functions:
        result[func.qualified_name] = (
            func.ast_depth,
            func.sibling_index,
            func.ast_node_type,
        )
    for cls in analysis.classes:
        result[cls.qualified_name] = (
            cls.ast_depth,
            cls.sibling_index,
            cls.ast_node_type,
        )
    return result


def _copy_clang_ast_part(
    ast_depth: list[int],
    sibling_index: list[int],
    ast_node_type: list[int],
    *,
    offset: int,
    part_len: int,
    source: tuple[list[int], list[int], list[int]] | None,
) -> None:
    if source is None:
        return
    src_depth, src_sibling, src_node_type = source
    if (
        len(src_depth) != part_len
        or len(src_sibling) != part_len
        or len(src_node_type) != part_len
    ):
        return
    end = offset + part_len
    ast_depth[offset:end] = src_depth
    sibling_index[offset:end] = src_sibling
    ast_node_type[offset:end] = src_node_type


def _build_enriched_from_parts(
    parts_info: list[PartInfo],
    old_analysis: FileAnalysis,
    new_analysis: Optional[FileAnalysis],
    record: dict,
    changed_symbol_ids: Optional[list[int]] = None,
    ripple_candidates: Optional[list[dict]] = None,
    section_kinds: Optional[list[str]] = None,
) -> dict:
    """Build enriched document dict from parts_info list.

    Produces the same schema as build_enriched_doc() in index_project.py:
    text, structure_ids, chunk_boundaries, call_edges, type_edges,
    ast_depth, sibling_index, ast_node_type, symbol_ids, call_targets,
    type_refs, def_use.

    Additionally emits:
    - changed_symbol_ids: list of int — IDs of symbols modified in this commit
    - ripple_candidates: list of dicts — symbols that may need updating due to changes
    """
    texts = [p[0] for p in parts_info]
    full_text = '\n\n'.join(texts)
    text_len = len(full_text)

    structure_ids = [0] * text_len
    ast_depth = [0] * text_len
    sibling_index = [0] * text_len
    ast_node_type = [0] * text_len
    chunk_boundaries = []
    offset = 0

    # Maps for edge computation: chunk_idx -> (qname, callees_list)
    chunk_qnames: dict[int, str] = {}
    chunk_callees: dict[int, list[str]] = {}

    # Merge functions from both analyses for callee lookup
    all_funcs: dict[str, FunctionDef] = {}
    for func in old_analysis.functions:
        all_funcs[func.qualified_name] = func
    if new_analysis:
        for func in new_analysis.functions:
            all_funcs[func.qualified_name] = func
    old_ast = _analysis_ast_maps(old_analysis)
    new_ast = _analysis_ast_maps(new_analysis)

    for i, (part_text, kind, dep_level, name, qname) in enumerate(parts_info):
        part_len = len(part_text)
        if offset + part_len > text_len:
            break

        for j in range(offset, offset + part_len):
            structure_ids[j] = kind
        source_kind = section_kinds[i] if section_kinds and i < len(section_kinds) else 'c'
        part_ast: tuple[list[int], list[int], list[int]] | None = None
        if qname:
            if source_kind == 'n':
                part_ast = new_ast.get(qname) or old_ast.get(qname)
            else:
                part_ast = old_ast.get(qname) or new_ast.get(qname)
        elif kind == 1:
            analysis = new_analysis if source_kind == 'n' and new_analysis else old_analysis
            if analysis and len(analysis.preamble_ast_depth) >= part_len:
                part_ast = (
                    analysis.preamble_ast_depth[:part_len],
                    analysis.preamble_sibling_index[:part_len],
                    analysis.preamble_ast_node_type[:part_len],
                )
        _copy_clang_ast_part(
            ast_depth,
            sibling_index,
            ast_node_type,
            offset=offset,
            part_len=part_len,
            source=part_ast,
        )

        chunk_boundaries.append({
            'start': offset,
            'end': offset + part_len,
            'kind': kind,
            'dep_level': dep_level,
            'name': name,
        })

        if qname and qname in all_funcs:
            chunk_qnames[i] = qname
            chunk_callees[i] = all_funcs[qname].callees

        offset += part_len
        if i < len(parts_info) - 1:
            offset += 2  # "\n\n"

    # Compute call_edges between chunks
    call_edges = []
    for ci, caller_qname in chunk_qnames.items():
        callees = chunk_callees.get(ci, [])
        for callee_qname in callees:
            for cj, target_qname in chunk_qnames.items():
                if ci != cj and target_qname == callee_qname:
                    call_edges.append({'from': ci, 'to': cj})

    type_edges: list[dict[str, object]] = []

    semantic_index = ProjectIndex()
    for func in all_funcs.values():
        semantic_index.add_function(func)
    semantic_meta = extract_semantic_metadata_from_parts(
        full_text,
        parts_info,
        semantic_index,
    )

    result: dict = {
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
        'changed_symbol_ids': changed_symbol_ids or [],
        'ripple_candidates': ripple_candidates or [],
    }
    temporal_meta = _build_commit_temporal_metadata(
        full_text,
        texts,
        section_kinds or ['c'] * len(texts),
        record=record,
        old_analysis=old_analysis,
        new_analysis=new_analysis,
    )
    result.update(temporal_meta)

    # Platform info detection mirrors the source-indexer path.
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _v5_dir = os.path.join(_parent, 'v5_gke_orchestrator')
    if _v5_dir not in sys.path:
        sys.path.insert(0, _v5_dir)
    platform_info: dict[str, object] | None
    try:
        _platform_detect = importlib.import_module("platform_detect")
        _detect_plat = cast(
            Callable[[str], dict[str, object] | None],
            getattr(_platform_detect, "detect_platforms"),
        )
        platform_info = _detect_plat(full_text)
    except ImportError:
        platform_info = None
    if platform_info:
        result['platform_info'] = platform_info
    compile_args = record.get("compile_args")
    build_info = record.get("build_info")
    if isinstance(build_info, dict) and build_info:
        result["build_info"] = build_info

    # Language info detection
    filepath = record.get('filepath', '')
    if detect_language_info is not None:
        try:
            lang_info = detect_language_info(
                full_text,
                filepath,
                platform_info,
                compile_args=compile_args if isinstance(compile_args, list) else None,
                build_info=build_info if isinstance(build_info, dict) else None,
            )
        except TypeError:
            lang_info = detect_language_info(full_text, filepath)
        if lang_info:
            result['language_info'] = lang_info

    return result


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_tokenizer = None


def _load_tokenizer(path: Optional[str]):
    global _tokenizer
    if _tokenizer is not None:
        return
    if path and os.path.exists(path):
        try:
            from tokenizers import Tokenizer  # type: ignore[reportMissingImports]
            _tokenizer = Tokenizer.from_file(path)
            return
        except Exception:
            pass
    # Fallback: estimate ~4 bytes per token
    _tokenizer = None


def count_tokens(text: str) -> int:
    if _tokenizer is not None:
        return len(_tokenizer.encode(text).ids)
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_record(
    record: dict,
    clang_index: Index,
    tmpdir: str,
    max_tokens: int,
    max_file_bytes: int,
    doc_format: str,
    max_dep_depth: int,
    build_context: BuildContextResolver | None = None,
) -> list[dict]:
    """Process a single commit record into enriched documents."""
    old_content = record.get('old_content', '')
    new_content = record.get('new_content', '')
    diff = record.get('diff', '')

    if len(old_content) > max_file_bytes or len(new_content) > max_file_bytes:
        return []
    if len(old_content) < 50 or len(new_content) < 50:
        return []

    hunks = parse_hunk_ranges(diff)
    if not hunks:
        return []

    filepath = record.get('filepath', 'source.cpp')
    repo_root: str | None = None
    compile_args: list[str] | None = None
    build_info: dict[str, object] = {}
    if build_context is not None:
        repo_root, compile_args, build_info = build_context.resolve(record)
        if compile_args:
            record.setdefault("compile_args", compile_args)
        if build_info:
            record.setdefault("build_info", build_info)

    # Parse old and new with clang (use separate temp subdirs to avoid conflicts)
    old_dir = os.path.join(tmpdir, 'old')
    new_dir = os.path.join(tmpdir, 'new')
    os.makedirs(old_dir, exist_ok=True)
    os.makedirs(new_dir, exist_ok=True)

    old_analysis = analyze_file_clang(
        old_content,
        filepath,
        clang_index,
        old_dir,
        compile_args=compile_args,
        repo_root=repo_root,
        build_info=build_info,
    )
    new_analysis = analyze_file_clang(
        new_content,
        filepath,
        clang_index,
        new_dir,
        compile_args=compile_args,
        repo_root=repo_root,
        build_info=build_info,
    )

    documents: list[dict[str, object]] = []

    if doc_format in ('chain', 'both'):
        doc = format_chain_document(record, old_analysis, new_analysis, hunks, max_dep_depth)
        if doc:
            tokens = count_tokens(doc['text'])
            if tokens <= max_tokens and len(doc['text']) >= 100:
                doc['actual_token_count'] = tokens
                documents.append(doc)

    if doc_format in ('diff', 'both'):
        doc = format_diff_document(record, old_analysis, hunks, max_dep_depth)
        if doc:
            tokens = count_tokens(doc['text'])
            if tokens <= max_tokens and len(doc['text']) >= 100:
                doc['actual_token_count'] = tokens
                documents.append(doc)

    return documents


def process_jsonl_file(
    input_path: str,
    output_file,
    clang_index: Index,
    tmpdir: str,
    max_tokens: int,
    max_file_bytes: int,
    doc_format: str,
    max_dep_depth: int,
    memory_limit_gb: float,
    build_context: BuildContextResolver | None = None,
) -> dict:
    """Process a JSONL input file, writing enriched docs to output."""
    stats = {
        'records_read': 0,
        'documents_written': 0,
        'records_skipped': 0,
        'records_empty': 0,
        'parse_errors': 0,
    }
    seen_hashes: set[str] = set()

    with open(input_path, 'r', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats['parse_errors'] += 1
                continue

            stats['records_read'] += 1

            try:
                docs = process_record(
                    record, clang_index, tmpdir,
                    max_tokens, max_file_bytes, doc_format, max_dep_depth,
                    build_context=build_context,
                )
            except Exception as e:
                stats['parse_errors'] += 1
                if stats['parse_errors'] <= 10:
                    print(f"  WARN: Record {line_num}: {e}", file=sys.stderr)
                continue

            if not docs:
                stats['records_empty'] += 1
                continue

            for doc in docs:
                doc_hash = hashlib.md5(doc['text'].encode()).hexdigest()
                if doc_hash in seen_hashes:
                    stats['records_skipped'] += 1
                    continue
                seen_hashes.add(doc_hash)
                output_file.write(json.dumps(doc, ensure_ascii=False) + '\n')
                stats['documents_written'] += 1

            if stats['records_read'] % 1000 == 0:
                check_memory_limit(memory_limit_gb, label="process_commits")
                print(
                    f"  [{input_path}] {stats['records_read']} records, "
                    f"{stats['documents_written']} docs written",
                    file=sys.stderr,
                )

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Process git commit diffs into enriched training documents using libclang',
    )
    parser.add_argument(
        '--inputs', required=True, nargs='+',
        help='Input JSONL files with commit records (old_content, new_content, diff, ...)',
    )
    parser.add_argument(
        '--output', required=True,
        help='Output enriched JSONL file',
    )
    parser.add_argument(
        '--max-tokens', type=int, default=4096,
        help='Maximum tokens per document (default: 4096)',
    )
    parser.add_argument(
        '--max-file-bytes', type=int, default=500000,
        help='Skip files larger than this (default: 500000)',
    )
    parser.add_argument(
        '--max-dep-depth', type=int, default=5,
        help='Maximum BFS depth for transitive deps (default: 5)',
    )
    parser.add_argument(
        '--format', choices=['chain', 'diff', 'both'], default='both',
        dest='doc_format',
        help='Document format: chain (PRE/POST), diff (context+patch), or both',
    )
    parser.add_argument(
        '--tokenizer-path', default=None,
        help='Path to tokenizer.json for exact token counting (fallback: estimate)',
    )
    parser.add_argument(
        '--libclang-path', default=None,
        help='Explicit path to libclang.so',
    )
    parser.add_argument(
        '--repo-root', default=None,
        help='Single source repo root to use for compile_commands/include resolution',
    )
    parser.add_argument(
        '--repo-dir', default=None,
        help='Parent directory containing repos named by each record["repo"]',
    )
    parser.add_argument(
        '--memory-limit-gb', type=float, default=10.0,
        help='Abort if this Python wrapper exceeds this max RSS in GiB (default: 10).',
    )
    args = parser.parse_args()
    start_memory_guard(args.memory_limit_gb, label="process_commits")

    print("Clang commit processor starting", file=sys.stderr)
    print(f"  inputs: {args.inputs}", file=sys.stderr)
    print(f"  output: {args.output}", file=sys.stderr)
    print(f"  max_tokens: {args.max_tokens}", file=sys.stderr)
    print(f"  format: {args.doc_format}", file=sys.stderr)
    print(f"  memory_limit_gb: {args.memory_limit_gb}", file=sys.stderr)
    if args.repo_root:
        print(f"  repo_root: {args.repo_root}", file=sys.stderr)
    if args.repo_dir:
        print(f"  repo_dir: {args.repo_dir}", file=sys.stderr)

    # Configure libclang
    try:
        libclang_path = _configure_libclang(args.libclang_path)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"  libclang: {libclang_path or 'system default'}", file=sys.stderr)

    # Load tokenizer
    _load_tokenizer(args.tokenizer_path)
    if _tokenizer:
        print(f"  tokenizer: {args.tokenizer_path}", file=sys.stderr)
    else:
        print("  tokenizer: estimate (~4 bytes/token)", file=sys.stderr)

    clang_index = Index.create()
    build_context = BuildContextResolver(repo_root=args.repo_root, repo_dir=args.repo_dir)

    t0 = time.time()
    total_stats = {
        'records_read': 0,
        'documents_written': 0,
        'records_skipped': 0,
        'records_empty': 0,
        'parse_errors': 0,
    }

    with tempfile.TemporaryDirectory(prefix='clang_commits_') as tmpdir:
        # Use /dev/shm if available for faster temp file I/O
        shm_tmpdir = None
        if os.path.isdir('/dev/shm'):
            shm_tmpdir = tempfile.mkdtemp(prefix='clang_commits_', dir='/dev/shm')
            actual_tmpdir = shm_tmpdir
        else:
            actual_tmpdir = tmpdir

        try:
            with open(args.output, 'w') as out_f:
                for input_path in args.inputs:
                    if not os.path.exists(input_path):
                        print(f"  WARN: {input_path} not found, skipping", file=sys.stderr)
                        continue

                    print(f"\n  Processing {input_path}...", file=sys.stderr)
                    stats = process_jsonl_file(
                        input_path, out_f, clang_index, actual_tmpdir,
                        args.max_tokens, args.max_file_bytes,
                        args.doc_format, args.max_dep_depth,
                        args.memory_limit_gb,
                        build_context=build_context,
                    )

                    for k in total_stats:
                        total_stats[k] += stats[k]

                    print(f"  Done: {stats}", file=sys.stderr)
        finally:
            if shm_tmpdir:
                import shutil
                shutil.rmtree(shm_tmpdir, ignore_errors=True)

    elapsed = time.time() - t0
    print(f"\nTotal: {total_stats}", file=sys.stderr)
    print(f"Time: {elapsed:.1f}s", file=sys.stderr)
    if total_stats['records_read'] > 0:
        rate = total_stats['records_read'] / elapsed
        print(f"Rate: {rate:.1f} records/sec", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
