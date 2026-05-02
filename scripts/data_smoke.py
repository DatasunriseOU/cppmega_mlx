#!/usr/bin/env python3
"""Smoke local token dataset ingress without wiring training."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any, Literal, NoReturn, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from cppmega_mlx.data.batch import LMTokenBatch  # noqa: E402
from cppmega_mlx.data.packing import pack_documents_with_eos  # noqa: E402
from cppmega_mlx.data.token_dataset import (  # noqa: E402
    TokenBatchDataset,
    TokenDatasetFormat,
    open_token_dataset,
)


SmokeDatasetFormat = Literal["npz", "megatron"]
SUPPORTED_FORMATS: tuple[SmokeDatasetFormat, ...] = ("npz", "megatron")
STRUCTURE_SIDE_CHANNELS = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)


class JsonArgumentParser(argparse.ArgumentParser):
    """Argparse variant that keeps CLI failures machine-readable."""

    def error(self, message: str) -> NoReturn:
        print(
            json.dumps(
                _base_receipt(status="error", error=message),
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2)


class SmokeError(RuntimeError):
    """Expected fail-closed smoke validation error."""


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(
        description=(
            "Open one or a few local token dataset batches and emit a JSON "
            "ingress receipt. This does not train or benchmark."
        )
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to a .npz shard or suffixless/.bin/.idx Megatron indexed shard.",
    )
    parser.add_argument(
        "--dataset-format",
        default=None,
        help=(
            "Dataset format override. Smoke support is fail-closed to npz and "
            "megatron only."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--token-key", default="tokens")
    parser.add_argument(
        "--batches",
        type=int,
        default=1,
        help="Maximum number of full batches to open.",
    )
    parser.add_argument(
        "--require-structure-side-channels",
        action="store_true",
        help="Fail closed when no token-aligned structure side channels are present.",
    )
    parser.add_argument(
        "--pack-documents",
        action="store_true",
        help=(
            "Run deterministic concat-with-EOS sequence packing on the first "
            "opened batch via cppmega_mlx.data.packing."
        ),
    )
    parser.add_argument(
        "--eos-token-id",
        type=int,
        default=None,
        help="EOS token ID required by --pack-documents.",
    )
    parser.add_argument(
        "--pad-token-id",
        type=int,
        default=0,
        help="Pad token ID used by --pack-documents.",
    )
    parser.add_argument(
        "--megatron-dtype",
        default=None,
        help="Optional dtype for raw .bin Megatron handoffs without .idx metadata.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_smoke(args)
    except SmokeError as exc:
        payload = _base_receipt(
            status="error",
            error=str(exc),
            dataset_format=args.dataset_format,
            dataset_path=args.dataset_path,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    except Exception as exc:
        payload = _base_receipt(
            status="error",
            error=str(exc),
            dataset_format=args.dataset_format,
            dataset_path=args.dataset_path,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset_path)
    dataset_format = _resolve_dataset_format(dataset_path, args.dataset_format)
    if args.batch_size < 1:
        raise SmokeError("batch-size must be positive")
    if args.seq_len < 2:
        raise SmokeError("seq-len must be at least 2")
    if args.batches < 1:
        raise SmokeError("batches must be positive")

    dataset = _open_dataset(args, dataset_format)
    batches = _read_batches(dataset, max_batches=int(args.batches))
    first_batch = batches[0]
    side_channels = _side_channel_presence(first_batch)
    structure_channels = [
        key for key in STRUCTURE_SIDE_CHANNELS if side_channels.get(key, False)
    ]
    if args.require_structure_side_channels and not structure_channels:
        raise SmokeError(
            "required structure side channels are missing from the first full batch"
        )

    batch_shape = [int(dim) for dim in first_batch.tokens.shape]
    payload: dict[str, Any] = {
        **_base_receipt(
            status="ok",
            dataset_format=dataset_format,
            dataset_path=str(dataset_path),
        ),
        "batch_shape": batch_shape,
        "batch_size": batch_shape[0],
        "batches_read": len(batches),
        "dataset": _dataset_receipt(dataset),
        "requested_batches": int(args.batches),
        "seq_len": batch_shape[1],
        "side_channel_presence": side_channels,
        "side_channels": [key for key, present in side_channels.items() if present],
        "structure_side_channel_presence": {
            key: side_channels[key] for key in STRUCTURE_SIDE_CHANNELS
        },
        "structure_side_channels": structure_channels,
        "structure_side_channels_present": bool(structure_channels),
        "token_key": args.token_key,
    }
    if args.pack_documents:
        payload["packing"] = _packing_receipt(first_batch, args)
    else:
        payload["packing"] = {"enabled": False}
    return payload


def _open_dataset(
    args: argparse.Namespace,
    dataset_format: SmokeDatasetFormat,
) -> TokenBatchDataset:
    kwargs: dict[str, Any] = {"token_key": args.token_key, "loop": False}
    if dataset_format == "megatron" and args.megatron_dtype is not None:
        kwargs["dtype"] = args.megatron_dtype
    return open_token_dataset(
        args.dataset_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        format=cast(TokenDatasetFormat, dataset_format),
        **kwargs,
    )


def _read_batches(dataset: TokenBatchDataset, *, max_batches: int) -> list[LMTokenBatch]:
    batches: list[LMTokenBatch] = []
    for batch in dataset.iter_batches(loop=False):
        mx.eval(batch.as_dict())
        batches.append(batch)
        if len(batches) >= max_batches:
            break
    if not batches:
        raise SmokeError("dataset did not yield a full batch for the requested shape")
    return batches


def _resolve_dataset_format(
    dataset_path: Path,
    raw_format: str | None,
) -> SmokeDatasetFormat:
    if raw_format is not None:
        if raw_format not in SUPPORTED_FORMATS:
            supported = ", ".join(SUPPORTED_FORMATS)
            raise SmokeError(
                f"unsupported dataset format {raw_format!r}; supported: {supported}"
            )
        return cast(SmokeDatasetFormat, raw_format)

    suffix = dataset_path.suffix.lower()
    if suffix == ".npz":
        return "npz"
    if suffix in {".bin", ".idx", ".json"}:
        return "megatron"
    if dataset_path.with_suffix(".bin").exists() or dataset_path.with_suffix(
        ".idx"
    ).exists():
        return "megatron"
    raise SmokeError(
        "could not infer dataset format; pass --dataset-format npz or megatron"
    )


def _side_channel_presence(batch: LMTokenBatch) -> dict[str, bool]:
    presence = {
        "attention_mask": batch.attention_mask is not None,
        **{key: value is not None for key, value in batch.structure_fields().items()},
    }
    return dict(sorted(presence.items()))


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


def _packing_receipt(batch: LMTokenBatch, args: argparse.Namespace) -> dict[str, Any]:
    if args.eos_token_id is None:
        raise SmokeError("--pack-documents requires --eos-token-id")
    tokens = np.array(batch.tokens, copy=True)
    if tokens.ndim != 2:
        raise SmokeError(f"cannot pack non-2D tokens with shape {tokens.shape}")
    documents = [row[:-1] for row in tokens.tolist()]
    packed = pack_documents_with_eos(
        documents,
        seq_len=int(args.seq_len),
        eos_token_id=int(args.eos_token_id),
        pad_token_id=int(args.pad_token_id),
    )
    return {
        "boundary_mask_shape": [int(dim) for dim in packed.boundary_mask.shape],
        "doc_ids_shape": [int(dim) for dim in packed.doc_ids.shape],
        "document_source": "first_batch_rows_without_final_token",
        "enabled": True,
        "packed_shape": [int(dim) for dim in packed.tokens.shape],
        "token_mask_true": int(packed.token_mask.sum()),
    }


def _base_receipt(
    *,
    status: str,
    error: str | None = None,
    dataset_format: str | None = None,
    dataset_path: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "distributed_megatron_parity_claim": False,
        "gb10_parity_claim": False,
        "local_only": True,
        "m4_vs_gb10_parity_claim": False,
        "receipt_scope": "local_token_dataset_ingress_smoke",
        "status": status,
        "trainable_metal_kernel_adoption_claim": False,
        "training_wired": False,
    }
    if dataset_format is not None:
        payload["dataset_format"] = dataset_format
    if dataset_path is not None:
        payload["dataset_path"] = dataset_path
    if error is not None:
        payload["error"] = error
    return payload


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(cast(Any, value)))
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
