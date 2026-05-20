#!/usr/bin/env python3
"""Materialize token-coordinate metadata columns from local GB10 parquet rows.

The current local GB10 parquet samples keep token_ids plus source-coordinate
metadata such as structure_ids and chunk_boundaries. This script retokenizes
each row with the cppmega tokenizer, verifies the generated ids match the
stored token_ids exactly, then writes token-coordinate columns next to the
original columns:

  token_structure_ids
  token_dep_levels
  token_chunk_starts / token_chunk_ends / token_chunk_kinds / token_chunk_dep_levels
  token_call_edges / token_type_edges

It never rewrites the input file in place.
"""

from __future__ import annotations

import argparse
import bisect
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer  # noqa: E402


TOKENIZER_DEFAULT = ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json"
SOURCE_DEFAULT = (
    ROOT
    / "data"
    / "parquet_samples"
    / "gb10"
    / "clang_semantic_4k_v10"
    / "val_00000.parquet"
)
DEST_DEFAULT = (
    ROOT
    / "data"
    / "parquet_samples"
    / "gb10"
    / "clang_semantic_4k_v10_tokencoords"
    / "val_00000.parquet"
)

TOKEN_STRUCTURE_IDS_COLUMN = "token_structure_ids"
TOKEN_DEP_LEVELS_COLUMN = "token_dep_levels"
TOKEN_CHUNK_STARTS_COLUMN = "token_chunk_starts"
TOKEN_CHUNK_ENDS_COLUMN = "token_chunk_ends"
TOKEN_CHUNK_KINDS_COLUMN = "token_chunk_kinds"
TOKEN_CHUNK_DEP_LEVELS_COLUMN = "token_chunk_dep_levels"
TOKEN_CALL_EDGES_COLUMN = "token_call_edges"
TOKEN_TYPE_EDGES_COLUMN = "token_type_edges"
TOKEN_COORDINATE_COLUMNS = (
    TOKEN_STRUCTURE_IDS_COLUMN,
    TOKEN_DEP_LEVELS_COLUMN,
    TOKEN_CHUNK_STARTS_COLUMN,
    TOKEN_CHUNK_ENDS_COLUMN,
    TOKEN_CHUNK_KINDS_COLUMN,
    TOKEN_CHUNK_DEP_LEVELS_COLUMN,
    TOKEN_CALL_EDGES_COLUMN,
    TOKEN_TYPE_EDGES_COLUMN,
)
TOKEN_COORDINATE_SCHEMA_VERSION = "cppmega.token_coordinates.v1"

_KIND_STR_TO_INT = {
    "other": 0,
    "preamble": 1,
    "func_signature": 2,
    "func_body": 3,
    "class_decl": 4,
    "class_member": 5,
    "comment": 6,
    "typedef": 7,
    "namespace": 8,
}


@dataclass
class ConversionStats:
    rows: int = 0
    token_count_mismatches: int = 0
    retokenize_mismatches: int = 0
    structure_length_mismatches: int = 0
    rows_with_chunks: int = 0
    chunks: int = 0
    dropped_chunks: int = 0
    call_edges: int = 0
    token_call_edges: int = 0
    type_edges: int = 0
    token_type_edges: int = 0

    def add(self, other: "ConversionStats") -> None:
        for key, value in asdict(other).items():
            setattr(self, key, int(getattr(self, key)) + int(value))


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": message,
                    "schema_version": TOKEN_COORDINATE_SCHEMA_VERSION,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output", type=Path, default=DEST_DEFAULT)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--row-group-size", type=int, default=1024)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Convert only the first N rows; 0 converts the full input.",
    )
    parser.add_argument(
        "--allow-mismatches",
        action="store_true",
        help=(
            "Write rows even if retokenized ids mismatch token_ids. The default "
            "fails closed because token-coordinate metadata would be ambiguous."
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Optional JSON receipt path; defaults to OUTPUT.receipt.json.",
    )
    return parser


def convert_parquet_file(
    input_path: Path,
    output_path: Path,
    *,
    tokenizer_path: Path = TOKENIZER_DEFAULT,
    batch_size: int = 1024,
    row_group_size: int = 1024,
    max_rows: int = 0,
    allow_mismatches: bool = False,
) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ImportError("pyarrow is required to materialize token-coordinate parquet") from exc

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")
    if max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    if not input_path.is_file():
        raise FileNotFoundError(f"input parquet not found: {input_path}")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("refusing to overwrite input parquet in place")

    tokenizer = load_cppmega_tokenizer(tokenizer_path)
    hf_tokenizer = getattr(tokenizer, "_tokenizer", None)
    if hf_tokenizer is None:
        raise TypeError("cppmega tokenizer does not expose the backing HF tokenizer")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(input_path)
    writer = None
    stats = ConversionStats()
    written_rows = 0

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            if max_rows and written_rows >= max_rows:
                break
            table = pa.Table.from_batches([batch])
            if max_rows:
                remaining = max_rows - written_rows
                if table.num_rows > remaining:
                    table = table.slice(0, remaining)

            converted_columns, batch_stats = materialize_token_coordinate_columns(
                table.to_pydict(),
                hf_tokenizer=hf_tokenizer,
                allow_mismatches=allow_mismatches,
            )
            stats.add(batch_stats)
            converted_table = append_token_coordinate_columns(
                table,
                converted_columns,
                pa=pa,
            )
            if writer is None:
                metadata = dict(converted_table.schema.metadata or {})
                metadata.update(
                    {
                        b"cppmega.token_coordinate_schema": TOKEN_COORDINATE_SCHEMA_VERSION.encode(),
                        b"cppmega.token_coordinate_source": str(input_path).encode(),
                        b"cppmega.token_coordinate_tokenizer": str(tokenizer_path).encode(),
                    }
                )
                converted_table = converted_table.replace_schema_metadata(metadata)
                writer = pq.ParquetWriter(
                    output_path,
                    converted_table.schema,
                    compression="snappy",
                    use_dictionary=True,
                )
            writer.write_table(converted_table, row_group_size=row_group_size)
            written_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    receipt = {
        "allow_mismatches": bool(allow_mismatches),
        "input": str(input_path),
        "output": str(output_path),
        "schema_version": TOKEN_COORDINATE_SCHEMA_VERSION,
        "status": "ok",
        "stats": asdict(stats),
        "token_coordinate_columns": list(TOKEN_COORDINATE_COLUMNS),
        "tokenizer": str(tokenizer_path),
    }
    if stats.retokenize_mismatches and not allow_mismatches:
        # This should be unreachable because the first mismatch raises.
        receipt["status"] = "error"
    return receipt


def append_token_coordinate_columns(
    table: Any,
    columns: Mapping[str, Sequence[Any]],
    *,
    pa: Any,
) -> Any:
    arrays = {
        TOKEN_STRUCTURE_IDS_COLUMN: pa.array(
            columns[TOKEN_STRUCTURE_IDS_COLUMN],
            type=pa.large_list(pa.int32()),
        ),
        TOKEN_DEP_LEVELS_COLUMN: pa.array(
            columns[TOKEN_DEP_LEVELS_COLUMN],
            type=pa.large_list(pa.int32()),
        ),
        TOKEN_CHUNK_STARTS_COLUMN: pa.array(
            columns[TOKEN_CHUNK_STARTS_COLUMN],
            type=pa.large_list(pa.int32()),
        ),
        TOKEN_CHUNK_ENDS_COLUMN: pa.array(
            columns[TOKEN_CHUNK_ENDS_COLUMN],
            type=pa.large_list(pa.int32()),
        ),
        TOKEN_CHUNK_KINDS_COLUMN: pa.array(
            columns[TOKEN_CHUNK_KINDS_COLUMN],
            type=pa.large_list(pa.int32()),
        ),
        TOKEN_CHUNK_DEP_LEVELS_COLUMN: pa.array(
            columns[TOKEN_CHUNK_DEP_LEVELS_COLUMN],
            type=pa.large_list(pa.int32()),
        ),
        TOKEN_CALL_EDGES_COLUMN: pa.array(
            columns[TOKEN_CALL_EDGES_COLUMN],
            type=_edge_list_type(pa),
        ),
        TOKEN_TYPE_EDGES_COLUMN: pa.array(
            columns[TOKEN_TYPE_EDGES_COLUMN],
            type=_edge_list_type(pa),
        ),
    }
    result = table
    for name in TOKEN_COORDINATE_COLUMNS:
        if name in result.column_names:
            result = result.drop([name])
        result = result.append_column(name, arrays[name])
    return result


def materialize_token_coordinate_columns(
    rows: Mapping[str, Sequence[Any]],
    *,
    hf_tokenizer: Any,
    allow_mismatches: bool = False,
) -> tuple[dict[str, list[Any]], ConversionStats]:
    texts = rows.get("text")
    token_ids_rows = rows.get("token_ids")
    if texts is None or token_ids_rows is None:
        raise ValueError("input parquet must contain text and token_ids columns")

    count = len(texts)
    out: dict[str, list[Any]] = {name: [] for name in TOKEN_COORDINATE_COLUMNS}
    stats = ConversionStats()
    for row_index in range(count):
        text = str(texts[row_index])
        token_ids = [int(value) for value in token_ids_rows[row_index]]
        actual_token_count = _optional_int(_row_value(rows, "actual_token_count", row_index))
        if actual_token_count is not None and actual_token_count != len(token_ids):
            stats.token_count_mismatches += 1

        normalized, normalized_to_original = _normalize_whitespace_with_offsets(text)
        encoding = hf_tokenizer.encode(normalized, add_special_tokens=False)
        encoded_ids = [int(value) for value in encoding.ids]
        if encoded_ids != token_ids:
            stats.retokenize_mismatches += 1
            if not allow_mismatches:
                raise ValueError(
                    "retokenized ids do not match token_ids at batch row "
                    f"{row_index}: encoded_len={len(encoded_ids)} "
                    f"stored_len={len(token_ids)}"
                )

        token_spans = [
            _normalized_span_to_original_span(
                normalized_to_original,
                int(start),
                int(end),
            )
            for start, end in encoding.offsets
        ]
        if len(token_spans) != len(token_ids):
            stats.retokenize_mismatches += 1
            if not allow_mismatches:
                raise ValueError(
                    "retokenized offsets do not match token_ids length at batch row "
                    f"{row_index}: offsets={len(token_spans)} tokens={len(token_ids)}"
                )

        structure_ids = [
            int(value)
            for value in (_row_value(rows, "structure_ids", row_index) or [])
        ]
        if structure_ids and len(structure_ids) != len(text):
            stats.structure_length_mismatches += 1
        token_structure_ids = _chars_to_token_ids_by_start(structure_ids, token_spans)
        chunks = _normalize_chunk_rows(_row_value(rows, "chunk_boundaries", row_index))
        token_chunks = _chunks_to_token_spans(chunks, token_spans, len(token_ids))
        token_dep_levels = _token_dep_levels(token_chunks, len(token_ids))
        layout = _chunk_layout(
            token_chunks,
            token_count=len(token_ids),
            call_edges=_normalize_edge_pairs(_row_value(rows, "call_edges", row_index)),
            type_edges=_normalize_edge_pairs(_row_value(rows, "type_edges", row_index)),
        )

        stats.rows += 1
        stats.rows_with_chunks += int(bool(chunks))
        stats.chunks += len(chunks)
        stats.dropped_chunks += max(0, len(chunks) - len(token_chunks))
        stats.call_edges += len(_normalize_edge_pairs(_row_value(rows, "call_edges", row_index)))
        stats.token_call_edges += len(layout[TOKEN_CALL_EDGES_COLUMN])
        stats.type_edges += len(_normalize_edge_pairs(_row_value(rows, "type_edges", row_index)))
        stats.token_type_edges += len(layout[TOKEN_TYPE_EDGES_COLUMN])

        out[TOKEN_STRUCTURE_IDS_COLUMN].append(token_structure_ids)
        out[TOKEN_DEP_LEVELS_COLUMN].append(token_dep_levels)
        for key in (
            TOKEN_CHUNK_STARTS_COLUMN,
            TOKEN_CHUNK_ENDS_COLUMN,
            TOKEN_CHUNK_KINDS_COLUMN,
            TOKEN_CHUNK_DEP_LEVELS_COLUMN,
            TOKEN_CALL_EDGES_COLUMN,
            TOKEN_TYPE_EDGES_COLUMN,
        ):
            out[key].append(layout[key])

    return out, stats


def _normalize_whitespace_with_offsets(text: str) -> tuple[str, list[tuple[int, int]]]:
    chars: list[str] = []
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "\r\n":
            j = i + 1
            while j < len(text) and text[j] in "\r\n":
                j += 1
            _append_replacement(chars, spans, "<NL>", i, j)
            i = j
        elif ch in " \t":
            j = i + 1
            while j < len(text) and text[j] in " \t":
                j += 1
            _append_replacement(chars, spans, "<SPACE>", i, j)
            i = j
        else:
            chars.append(ch)
            spans.append((i, i + 1))
            i += 1
    return "".join(chars), spans


def _append_replacement(
    chars: list[str],
    spans: list[tuple[int, int]],
    replacement: str,
    start: int,
    end: int,
) -> None:
    for ch in replacement:
        chars.append(ch)
        spans.append((start, end))


def _normalized_span_to_original_span(
    normalized_to_original: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int]:
    if not normalized_to_original:
        return (0, 0)
    start = min(max(start, 0), len(normalized_to_original))
    end = min(max(end, start), len(normalized_to_original))
    if end <= start:
        return (0, 0)
    spans = normalized_to_original[start:end]
    return min(item[0] for item in spans), max(item[1] for item in spans)


def _chars_to_token_ids_by_start(
    char_ids: Sequence[int],
    token_spans: Sequence[tuple[int, int]],
) -> list[int]:
    if not char_ids:
        return [0] * len(token_spans)
    out: list[int] = []
    for start, end in token_spans:
        if end > start and 0 <= start < len(char_ids):
            out.append(int(char_ids[start]))
        else:
            out.append(0)
    return out


def _chunks_to_token_spans(
    chunks: Sequence[Mapping[str, Any]],
    token_spans: Sequence[tuple[int, int]],
    token_count: int,
) -> list[dict[str, Any]]:
    if not chunks or not token_spans:
        return []
    token_starts = [int(start) for start, _ in token_spans]
    token_chunks: list[dict[str, Any]] = []
    for original_index, chunk in enumerate(chunks):
        char_start = _optional_int(chunk.get("start"))
        char_end = _optional_int(chunk.get("end"))
        if char_start is None:
            continue
        if char_end is None:
            char_end = char_start
        start = _token_index_for_char_start(token_starts, char_start)
        end = _token_exclusive_end_for_char_end(token_starts, char_end, token_count)
        if end <= start:
            end = min(start + 1, token_count)
        if start < 0 or start >= token_count or end <= start:
            continue
        token_chunks.append(
            {
                "original_index": int(original_index),
                "start": int(start),
                "end": int(end),
                "kind": _kind_to_int(chunk.get("kind", 0)),
                "dep_level": int(chunk.get("dep_level", 0) or 0),
            }
        )
    token_chunks.sort(
        key=lambda item: (
            int(item["start"]),
            int(item["end"]),
            int(item["original_index"]),
        )
    )
    return token_chunks


def _token_index_for_char_start(token_starts: Sequence[int], char_start: int) -> int:
    token_index = bisect.bisect_right(token_starts, int(char_start)) - 1
    if token_index < 0:
        return 0
    if token_index >= len(token_starts):
        return len(token_starts) - 1
    return token_index


def _token_exclusive_end_for_char_end(
    token_starts: Sequence[int],
    char_end: int,
    token_count: int,
) -> int:
    if not token_starts:
        return 0
    end = bisect.bisect_left(token_starts, int(char_end))
    if end < 0:
        return 0
    if end > token_count:
        return token_count
    return end


def _token_dep_levels(
    token_chunks: Sequence[Mapping[str, Any]],
    token_count: int,
) -> list[int]:
    dep_levels = [0] * token_count
    for chunk in token_chunks:
        start = int(chunk["start"])
        end = int(chunk["end"])
        dep_level = int(chunk.get("dep_level", 0) or 0)
        for token_index in range(max(start, 0), min(end, token_count)):
            dep_levels[token_index] = dep_level
    return dep_levels


def _chunk_layout(
    token_chunks: Sequence[Mapping[str, Any]],
    *,
    token_count: int,
    call_edges: Sequence[tuple[int, int]],
    type_edges: Sequence[tuple[int, int]],
) -> dict[str, list[Any]]:
    del token_count
    index_map = {
        int(chunk["original_index"]): new_index
        for new_index, chunk in enumerate(token_chunks)
    }
    return {
        TOKEN_CHUNK_STARTS_COLUMN: [int(chunk["start"]) for chunk in token_chunks],
        TOKEN_CHUNK_ENDS_COLUMN: [int(chunk["end"]) for chunk in token_chunks],
        TOKEN_CHUNK_KINDS_COLUMN: [int(chunk["kind"]) for chunk in token_chunks],
        TOKEN_CHUNK_DEP_LEVELS_COLUMN: [
            int(chunk.get("dep_level", 0) or 0) for chunk in token_chunks
        ],
        TOKEN_CALL_EDGES_COLUMN: _remap_edges(call_edges, index_map),
        TOKEN_TYPE_EDGES_COLUMN: _remap_edges(type_edges, index_map),
    }


def _remap_edges(
    edges: Sequence[tuple[int, int]],
    index_map: Mapping[int, int],
) -> list[dict[str, int]]:
    remapped: list[dict[str, int]] = []
    for src, dst in edges:
        if int(src) in index_map and int(dst) in index_map:
            remapped.append(
                {
                    "from": int(index_map[int(src)]),
                    "to": int(index_map[int(dst)]),
                }
            )
    return remapped


def _normalize_chunk_rows(value: Any) -> list[dict[str, Any]]:
    value = _decode_json_like(value)
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    chunks: list[dict[str, Any]] = []
    for item in value:
        item = _decode_json_like(item)
        if isinstance(item, Mapping):
            chunks.append(dict(item))
    return chunks


def _normalize_edge_pairs(value: Any) -> list[tuple[int, int]]:
    value = _decode_json_like(value)
    if value is None:
        return []
    pairs: list[tuple[int, int]] = []
    for item in value if isinstance(value, list) else []:
        item = _decode_json_like(item)
        if isinstance(item, Mapping) and "from" in item and "to" in item:
            pairs.append((int(item["from"]), int(item["to"])))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((int(item[0]), int(item[1])))
    return pairs


def _kind_to_int(kind: Any) -> int:
    if isinstance(kind, int) and not isinstance(kind, bool):
        return int(kind)
    text = str(kind)
    if text.isdigit():
        return int(text)
    return _KIND_STR_TO_INT.get(text.lower(), 0)


def _decode_json_like(value: Any) -> Any:
    while isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if decoded == value:
            return value
        value = decoded
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _row_value(rows: Mapping[str, Sequence[Any]], key: str, index: int) -> Any:
    values = rows.get(key)
    if values is None:
        return None
    return values[index]


def _edge_list_type(pa: Any) -> Any:
    return pa.large_list(
        pa.struct(
            [
                pa.field("from", pa.int32()),
                pa.field("to", pa.int32()),
            ]
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    receipt_path = args.receipt or args.output.with_suffix(args.output.suffix + ".receipt.json")
    try:
        receipt = convert_parquet_file(
            args.input,
            args.output,
            tokenizer_path=args.tokenizer,
            batch_size=args.batch_size,
            row_group_size=args.row_group_size,
            max_rows=args.max_rows,
            allow_mismatches=args.allow_mismatches,
        )
    except Exception as exc:
        error = {
            "error": str(exc),
            "input": str(args.input),
            "output": str(args.output),
            "schema_version": TOKEN_COORDINATE_SCHEMA_VERSION,
            "status": "error",
        }
        print(json.dumps(error, indent=2, sort_keys=True))
        raise SystemExit(1) from exc
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
