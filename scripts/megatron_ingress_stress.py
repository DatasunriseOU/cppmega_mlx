#!/usr/bin/env python3
"""Generate and smoke Megatron indexed ingress at bounded memory."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import platform
import resource
import shutil
import struct
import sys
import tempfile
import time
from typing import Any, NoReturn

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from cppmega_mlx.data.batch import LMTokenBatch  # noqa: E402
from cppmega_mlx.data.token_dataset import TokenBatchDataset, open_token_dataset  # noqa: E402

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_INDEX_VERSION = 1
_DTYPE_CODES = {
    np.dtype(np.uint16): 8,
    np.dtype(np.int32): 4,
}
_DEFAULT_PEAK_LIMIT_BYTES = 10 * 1024**3
_STRUCTURE_SIDE_CHANNELS = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        print(
            json.dumps(
                _base_receipt(status="error", error=message),
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2)


class StressError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(
        description=(
            "Generate a local Megatron .bin/.idx fixture in bounded chunks, "
            "read batches through cppmega_mlx.data, and emit a JSON receipt."
        )
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--token-count", type=int, default=100_000_000)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--dtype", choices=("uint16", "int32"), default="uint16")
    parser.add_argument("--vocab-size", type=int, default=65_536)
    parser.add_argument("--chunk-tokens", type=int, default=1_000_000)
    parser.add_argument("--include-document-ids", action="store_true")
    parser.add_argument("--include-structure-ids", action="store_true")
    parser.add_argument(
        "--max-peak-bytes",
        type=int,
        default=_DEFAULT_PEAK_LIMIT_BYTES,
        help="Fail if process peak RSS exceeds this ceiling.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_stress(args)
    except Exception as exc:
        print(
            json.dumps(
                _base_receipt(status="error", error=str(exc)),
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 2


def run_stress(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    owned_tempdir: tempfile.TemporaryDirectory[str] | None = None
    if args.output_dir is None:
        owned_tempdir = tempfile.TemporaryDirectory(prefix="cppmega-megatron-stress-")
        output_dir = Path(owned_tempdir.name)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(output_dir.iterdir()) and not args.overwrite:
            raise StressError(
                f"output directory is not empty; pass --overwrite: {output_dir}"
            )
    try:
        if args.overwrite:
            _clear_generated_files(output_dir)
        generate_started = time.perf_counter()
        prefixes = _generate_fixture(
            output_dir,
            token_count=int(args.token_count),
            shards=int(args.shards),
            dtype=np.dtype(args.dtype),
            vocab_size=int(args.vocab_size),
            chunk_tokens=int(args.chunk_tokens),
            include_document_ids=bool(args.include_document_ids),
            include_structure_ids=bool(args.include_structure_ids),
        )
        generation_seconds = time.perf_counter() - generate_started
        dataset_path = output_dir if len(prefixes) > 1 else prefixes[0]
        read_started = time.perf_counter()
        dataset = open_token_dataset(
            dataset_path,
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            format="megatron",
        )
        batches = _read_batches(dataset, max_batches=int(args.batches))
        read_seconds = time.perf_counter() - read_started
        first_batch = batches[0]
        tokens_read = len(batches) * int(args.batch_size) * int(args.seq_len)
        peak_bytes = _peak_rss_bytes()
        payload = {
            **_base_receipt(status="ok"),
            "batch_size": int(args.batch_size),
            "batches_read": len(batches),
            "chunk_tokens": int(args.chunk_tokens),
            "dataset": _dataset_receipt(dataset),
            "dataset_path": str(dataset_path),
            "dtype": str(np.dtype(args.dtype).name),
            "generated_bytes": _generated_bytes(prefixes),
            "generation_seconds": generation_seconds,
            "keep_data": bool(args.keep_data or args.output_dir is not None),
            "memory_peak_bytes": peak_bytes,
            "memory_peak_gib": peak_bytes / 1024**3,
            "memory_peak_limit_bytes": int(args.max_peak_bytes),
            "memory_peak_within_limit": peak_bytes <= int(args.max_peak_bytes),
            "read_seconds": read_seconds,
            "read_tokens_per_second": tokens_read / read_seconds if read_seconds else None,
            "seq_len": int(args.seq_len),
            "shards": int(args.shards),
            "side_channel_presence": _side_channel_presence(first_batch),
            "token_count": int(args.token_count),
            "tokens_read": tokens_read,
            "vocab_size": int(args.vocab_size),
        }
        if not payload["memory_peak_within_limit"]:
            payload["status"] = "error"
            payload["error"] = (
                "peak memory exceeded configured ceiling: "
                f"{peak_bytes} > {int(args.max_peak_bytes)}"
            )
        return payload
    finally:
        if owned_tempdir is not None and not args.keep_data:
            owned_tempdir.cleanup()


def _validate_args(args: argparse.Namespace) -> None:
    if args.token_count < 1:
        raise StressError("token-count must be positive")
    if args.shards < 1:
        raise StressError("shards must be positive")
    if args.seq_len < 2:
        raise StressError("seq-len must be at least 2")
    if args.batch_size < 1:
        raise StressError("batch-size must be positive")
    if args.batches < 1:
        raise StressError("batches must be positive")
    if args.chunk_tokens < 1:
        raise StressError("chunk-tokens must be positive")
    if args.vocab_size < 2:
        raise StressError("vocab-size must be at least 2")
    if args.max_peak_bytes < 1:
        raise StressError("max-peak-bytes must be positive")


def _clear_generated_files(output_dir: Path) -> None:
    for candidate in output_dir.iterdir():
        if candidate.is_dir():
            shutil.rmtree(candidate)
        else:
            candidate.unlink()


def _generate_fixture(
    output_dir: Path,
    *,
    token_count: int,
    shards: int,
    dtype: np.dtype,
    vocab_size: int,
    chunk_tokens: int,
    include_document_ids: bool,
    include_structure_ids: bool,
) -> list[Path]:
    counts = _split_counts(token_count, shards)
    prefixes: list[Path] = []
    global_offset = 0
    for shard_index, shard_tokens in enumerate(counts):
        prefix = output_dir / f"stress_{shard_index:05d}"
        prefixes.append(prefix)
        _write_token_file(
            prefix.with_suffix(".bin"),
            token_count=shard_tokens,
            dtype=dtype,
            vocab_size=vocab_size,
            global_offset=global_offset,
            chunk_tokens=chunk_tokens,
        )
        _write_index(prefix.with_suffix(".idx"), token_count=shard_tokens, dtype=dtype)
        sidecar: dict[str, Any] = {
            "source_format": "megatron-stress",
            "tokenizer_contract": "custom",
            "vocab_size": vocab_size,
        }
        if include_structure_ids:
            structure_path = prefix.with_name(f"{prefix.name}_structure_ids.bin")
            _write_mod_file(
                structure_path,
                token_count=shard_tokens,
                modulus=7,
                dtype=np.dtype(np.int16),
                global_offset=global_offset,
                chunk_tokens=chunk_tokens,
            )
            sidecar["side_channel_paths"] = {
                "structure_ids": {
                    "path": structure_path.name,
                    "dtype": "int16",
                },
            }
        if include_document_ids:
            doc_path = prefix.with_name(f"{prefix.name}_doc_ids.bin")
            _write_constant_file(
                doc_path,
                token_count=shard_tokens,
                value=shard_index,
                dtype=np.dtype(np.int32),
                chunk_tokens=chunk_tokens,
            )
            sidecar["doc_ids"] = {
                "path": doc_path.name,
                "dtype": "int32",
            }
        prefix.with_suffix(".idx.json").write_text(
            json.dumps(sidecar, sort_keys=True),
            encoding="utf-8",
        )
        global_offset += shard_tokens
    return prefixes


def _split_counts(token_count: int, shards: int) -> list[int]:
    base, remainder = divmod(token_count, shards)
    counts = [base + (1 if shard < remainder else 0) for shard in range(shards)]
    if any(count <= 0 for count in counts):
        raise StressError("token-count must be at least the number of shards")
    if any(count > np.iinfo(np.int32).max for count in counts):
        raise StressError("each shard token count must fit the MMIDIDX int32 length")
    return counts


def _write_token_file(
    path: Path,
    *,
    token_count: int,
    dtype: np.dtype,
    vocab_size: int,
    global_offset: int,
    chunk_tokens: int,
) -> None:
    mmap = np.memmap(path, mode="w+", dtype=dtype, shape=(token_count,))
    for start in range(0, token_count, chunk_tokens):
        stop = min(start + chunk_tokens, token_count)
        values = (
            np.arange(global_offset + start, global_offset + stop, dtype=np.int64)
            % vocab_size
        )
        mmap[start:stop] = values.astype(dtype, copy=False)
    mmap.flush()
    del mmap


def _write_mod_file(
    path: Path,
    *,
    token_count: int,
    modulus: int,
    dtype: np.dtype,
    global_offset: int,
    chunk_tokens: int,
) -> None:
    mmap = np.memmap(path, mode="w+", dtype=dtype, shape=(token_count,))
    for start in range(0, token_count, chunk_tokens):
        stop = min(start + chunk_tokens, token_count)
        values = (
            np.arange(global_offset + start, global_offset + stop, dtype=np.int64)
            % modulus
        )
        mmap[start:stop] = values.astype(dtype, copy=False)
    mmap.flush()
    del mmap


def _write_constant_file(
    path: Path,
    *,
    token_count: int,
    value: int,
    dtype: np.dtype,
    chunk_tokens: int,
) -> None:
    mmap = np.memmap(path, mode="w+", dtype=dtype, shape=(token_count,))
    for start in range(0, token_count, chunk_tokens):
        stop = min(start + chunk_tokens, token_count)
        mmap[start:stop] = np.full(stop - start, value, dtype=dtype)
    mmap.flush()
    del mmap


def _write_index(path: Path, *, token_count: int, dtype: np.dtype) -> None:
    dtype_code = _DTYPE_CODES.get(dtype)
    if dtype_code is None:
        raise StressError(f"unsupported stress dtype {dtype.name!r}")
    with path.open("wb") as fh:
        fh.write(_INDEX_HEADER)
        fh.write(struct.pack("<Q", _INDEX_VERSION))
        fh.write(struct.pack("<B", dtype_code))
        fh.write(struct.pack("<Q", 1))
        fh.write(struct.pack("<Q", 2))
        np.asarray([token_count], dtype=np.int32).tofile(fh)
        np.asarray([0], dtype=np.int64).tofile(fh)
        np.asarray([0, 1], dtype=np.int64).tofile(fh)


def _read_batches(dataset: TokenBatchDataset, *, max_batches: int) -> list[LMTokenBatch]:
    batches: list[LMTokenBatch] = []
    for batch in dataset.iter_batches(loop=False):
        mx.eval(batch.as_dict())
        batches.append(batch)
        if len(batches) >= max_batches:
            break
    if len(batches) < max_batches:
        raise StressError(
            f"dataset yielded {len(batches)} full batches; requested {max_batches}"
        )
    return batches


def _side_channel_presence(batch: LMTokenBatch) -> dict[str, bool]:
    return {
        "attention_mask": batch.attention_mask is not None,
        "document_ids": batch.document_ids is not None,
        **{key: value is not None for key, value in batch.structure_fields().items()},
    }


def _dataset_receipt(dataset: TokenBatchDataset) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "batch_size": int(dataset.batch_size),
        "dropped_samples": int(dataset.dropped_samples),
        "metadata": _jsonable(dataset.metadata),
        "num_batches": int(dataset.num_batches),
        "num_samples": int(dataset.num_samples),
        "path": str(dataset.path),
        "seq_len": int(dataset.seq_len),
        "token_id_range": list(dataset.token_id_range()),
    }
    index_metadata = getattr(dataset, "index_metadata", None)
    if index_metadata is not None:
        receipt["index_metadata"] = _jsonable(index_metadata)
    return receipt


def _generated_bytes(prefixes: list[Path]) -> int:
    total = 0
    for prefix in prefixes:
        for candidate in prefix.parent.glob(f"{prefix.name}*"):
            if candidate.is_file():
                total += candidate.stat().st_size
    return total


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return raw
    return raw * 1024


def _base_receipt(*, status: str, error: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "distributed_megatron_parity_claim": False,
        "gb10_parity_claim": False,
        "local_only": True,
        "m4_vs_gb10_parity_claim": False,
        "receipt_scope": "local_megatron_indexed_ingress_stress",
        "status": status,
        "training_wired": False,
    }
    if error is not None:
        payload["error"] = error
    return payload


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
