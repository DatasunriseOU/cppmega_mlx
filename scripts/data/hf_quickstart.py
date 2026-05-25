"""V8-R09: stream a HF Hub dataset → tokenize → emit a parquet shard.

Karpathy-nanochat style: connect to a streaming HF dataset, encode each
document with the cppmega tokenizer, append token IDs until we have at
least ``n_tokens`` tokens, write a single parquet shard with columns
``["token_ids", "doc_ids", "byte_offsets", "byte_lengths"]`` (the
canonical cppmega 4k-aligned schema).

The function is callable directly (used by tests) AND exposed through
the ``data.hf_quickstart`` RPC. Progress events are published to the
``data_event_bus`` keyed by ``job_id`` — the UI subscribes on
``/ws/data/{job_id}`` to render a live progress bar.

Dependency choice: ``datasets.load_dataset`` with ``streaming=True``
keeps memory bounded — we never hold more than one document in
memory. Tokenization is per-document so the loop yields control to
the event bus on every iteration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HFQuickstartResult:
    parquet_path: str
    n_tokens_written: int
    n_docs_seen: int
    elapsed_ms: float


def hf_quickstart(
    dataset_id: str,
    *,
    split: str = "train",
    tokenizer: str = "cppmega_v3",
    n_tokens: int = 100_000,
    job_id: str | None = None,
    out_dir: str | None = None,
    text_field: str = "text",
    progress_every: int = 50,
) -> HFQuickstartResult:
    """Stream ``dataset_id``, tokenize with ``tokenizer``, emit parquet.

    Args:
      dataset_id: HF Hub dataset path, e.g. "HuggingFaceFW/fineweb-edu".
      split: dataset split.
      tokenizer: path to a tokenizer.json OR a name we look up in the
        cppmega tokenizer presets ("cppmega_v3" → bundled tokenizer).
      n_tokens: keep streaming until at least this many tokens are
        accumulated (always rounded up to the next document boundary).
      job_id: opaque ID; progress events publish on this key.
      out_dir: target dir; defaults to /tmp/vbgui.
      text_field: HF dataset column to tokenize. fineweb-edu uses
        "text"; other datasets may use "content" / "document".

    Returns:
      HFQuickstartResult with parquet_path and counters.
    """
    import os
    import json
    
    # 1. Resolve cache directory and construct a unique persistent filename
    cache_base = Path(out_dir or os.environ.get("VBGUI_CACHE_DIR") or "/Users/dave/sources/cppmega.mlx/data/cache/datasets")
    cache_base.mkdir(parents=True, exist_ok=True)
    
    dataset_clean = dataset_id.replace("/", "--").replace(" ", "_")
    tokenizer_clean = tokenizer.replace("/", "--").replace(" ", "_").replace(".json", "")
    
    cache_filename = f"{dataset_clean}_{tokenizer_clean}_{n_tokens}_{split}_{text_field}.parquet"
    cache_meta_filename = f"{dataset_clean}_{tokenizer_clean}_{n_tokens}_{split}_{text_field}.meta.json"
    
    out_path = cache_base / cache_filename
    cache_meta_path = cache_base / cache_meta_filename
    
    # 2. Check if persistent Cache HIT is available
    if out_path.exists() and cache_meta_path.exists():
        try:
            with open(cache_meta_path, "r") as f:
                meta = json.load(f)
            print(f"[hf_quickstart] Cache HIT! Loading persistent cached dataset: {out_path}", flush=True)
            from cppmega_v4.runtime import data_event_bus as _db
            if job_id is not None:
                _db.publish(job_id, {"phase": "start", "dataset_id": dataset_id, "n_tokens_target": n_tokens})
                _db.publish(job_id, {"phase": "progress", "n_docs": meta["n_docs"], "n_tokens": meta["n_tokens"]})
                _db.publish(job_id, {
                    "phase": "done",
                    "parquet_path": str(out_path),
                    "n_tokens": meta["n_tokens"],
                    "n_docs": meta["n_docs"],
                    "elapsed_ms": meta["elapsed_ms"]
                })
                _db.publish(job_id, None)
            return HFQuickstartResult(
                parquet_path=str(out_path),
                n_tokens_written=meta["n_tokens"],
                n_docs_seen=meta["n_docs"],
                elapsed_ms=meta["elapsed_ms"]
            )
        except Exception as e:
            print(f"[hf_quickstart] Failed to read cache metadata sidecar: {e}. Falling back to download.", flush=True)

    # 3. Resolve the tokenizer (bundled presets first, then local files, then HF Hub)
    from tokenizers import Tokenizer
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
    
    if tokenizer in ("cppmega_v3", "cppmega_native_65k"):
        tok_path = (
            Path(__file__).parent.parent.parent
            / "cppmega_mlx" / "tokenizer" / "tokenizer.json")
        tok = load_cppmega_tokenizer(tok_path)
    else:
        tok_path = Path(tokenizer)
        if tok_path.exists():
            tok = load_cppmega_tokenizer(tok_path)
        else:
            try:
                # Try loading directly from HuggingFace Hub (supports gated tokenizers via HF_TOKEN)
                token = os.environ.get("HF_TOKEN")
                tok = Tokenizer.from_pretrained(tokenizer, token=token)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load tokenizer '{tokenizer}' from path or HuggingFace Hub: {e}"
                )

    # 4. Stream the HF dataset (authenticated with HF_TOKEN for SFT/math collections)
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(dataset_id, split=split, streaming=True, token=token)

    from cppmega_v4.runtime import data_event_bus as _db
    if job_id is not None:
        _db.publish(job_id, {"phase": "start",
                              "dataset_id": dataset_id,
                              "n_tokens_target": n_tokens})

    token_ids_col: list[list[int]] = []
    doc_ids_col: list[int] = []
    byte_off_col: list[int] = []
    byte_len_col: list[int] = []
    total_tokens = 0
    doc_idx = 0
    t0 = time.perf_counter()
    for row in ds:
        text = row.get(text_field)
        if not isinstance(text, str) or not text:
            continue
        ids = tok.encode(text)
        if not isinstance(ids, list):
            continue
        token_ids_col.append(ids)
        doc_ids_col.append(doc_idx)
        encoded = text.encode("utf-8", errors="replace")
        byte_off_col.append(0)
        byte_len_col.append(len(encoded))
        total_tokens += len(ids)
        doc_idx += 1
        if job_id is not None and doc_idx % progress_every == 0:
            _db.publish(job_id, {"phase": "progress",
                                  "n_docs": doc_idx,
                                  "n_tokens": total_tokens})
        if total_tokens >= n_tokens:
            break

    # Write the parquet shard.
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "token_ids":    pa.array(token_ids_col,
                                  type=pa.list_(pa.int64())),
        "doc_ids":      pa.array(doc_ids_col,    type=pa.int64()),
        "byte_offsets": pa.array(byte_off_col,   type=pa.int64()),
        "byte_lengths": pa.array(byte_len_col,   type=pa.int64()),
    })
    pq.write_table(table, out_path)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Write cache metadata sidecar
    try:
        with open(cache_meta_path, "w") as f:
            json.dump({
                "dataset_id": dataset_id,
                "tokenizer": tokenizer,
                "n_tokens": total_tokens,
                "n_docs": doc_idx,
                "split": split,
                "text_field": text_field,
                "elapsed_ms": elapsed_ms
            }, f, indent=2)
    except Exception as e:
        print(f"[hf_quickstart] Failed to write cache metadata sidecar: {e}", flush=True)

    if job_id is not None:
        _db.publish(job_id, {"phase": "done",
                              "parquet_path": str(out_path),
                              "n_tokens": total_tokens,
                              "n_docs": doc_idx,
                              "elapsed_ms": elapsed_ms})
        _db.publish(job_id, None)   # close
    return HFQuickstartResult(
        parquet_path=str(out_path),
        n_tokens_written=total_tokens,
        n_docs_seen=doc_idx,
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Pure-Python in-memory iterable fixture used by tests so we don't
# need HF Hub network access in CI. Same record shape as
# ``datasets.load_dataset(..., streaming=True)``.
# ---------------------------------------------------------------------------


def hf_quickstart_from_iterable(
    rows: list[dict[str, Any]],
    *,
    tokenizer: str = "cppmega_v3",
    n_tokens: int = 100_000,
    job_id: str | None = None,
    out_dir: str | None = None,
    text_field: str = "text",
) -> HFQuickstartResult:
    """Same as :func:`hf_quickstart` but pulls documents from an
    in-memory iterable instead of HF Hub. Used by tests."""
    out_dir_p = Path(out_dir or "/tmp/vbgui")
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_p / f"{job_id or 'hf-quickstart'}.parquet"

    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
    if tokenizer == "cppmega_v3":
        tok_path = (
            Path(__file__).parent.parent.parent
            / "cppmega_mlx" / "tokenizer" / "tokenizer.json")
    else:
        tok_path = Path(tokenizer)
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer not found: {tok_path}")
    tok = load_cppmega_tokenizer(tok_path)

    token_ids_col: list[list[int]] = []
    doc_ids_col: list[int] = []
    byte_off_col: list[int] = []
    byte_len_col: list[int] = []
    total_tokens = 0
    doc_idx = 0
    t0 = time.perf_counter()
    for row in rows:
        text = row.get(text_field)
        if not isinstance(text, str) or not text:
            continue
        ids = tok.encode(text)
        if not isinstance(ids, list):
            continue
        token_ids_col.append(ids)
        doc_ids_col.append(doc_idx)
        encoded = text.encode("utf-8", errors="replace")
        byte_off_col.append(0)
        byte_len_col.append(len(encoded))
        total_tokens += len(ids)
        doc_idx += 1
        if total_tokens >= n_tokens:
            break

    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "token_ids":    pa.array(token_ids_col,
                                  type=pa.list_(pa.int64())),
        "doc_ids":      pa.array(doc_ids_col,    type=pa.int64()),
        "byte_offsets": pa.array(byte_off_col,   type=pa.int64()),
        "byte_lengths": pa.array(byte_len_col,   type=pa.int64()),
    })
    pq.write_table(table, out_path)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return HFQuickstartResult(
        parquet_path=str(out_path),
        n_tokens_written=total_tokens,
        n_docs_seen=doc_idx,
        elapsed_ms=elapsed_ms,
    )
