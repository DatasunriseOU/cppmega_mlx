"""Megatron indexed token dataset reader for local MLX training."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
import numpy as np

from cppmega_mlx.config.model import (
    LOCAL_PROFILE_VOCAB_SIZE,
    MEGACPP_TOKENIZER_VOCAB_SIZE,
)
from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.data.graph_packet import EdgeIndex, GraphBatch, GraphPacket
from cppmega_mlx.data.token_dataset import BatchCursor, TokenDatasetMetadata
from cppmega_mlx.data.symbol_identity import SYMBOL_IDENTITY_SCHEMA_VERSION

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_INDEX_VERSION = 1
_ATTENTION_SIDE_CHANNEL_KEY = "attention_mask"
_LOSS_MASK_SIDE_CHANNEL_KEY = "loss_mask"
_STRUCTURE_SIDE_CHANNEL_KEYS = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)
_PLATFORM_SIDE_CHANNEL_KEYS = ("platform_ids",)
_SEMANTIC_SIDE_CHANNEL_KEYS = (
    "symbol_ids",
    "call_targets",
    "type_refs",
    "def_use",
)
_SYMBOL_ID_SIDE_CHANNEL_KEYS = ("symbol_ids", "call_targets", "type_refs")
_TEMPORAL_SIDE_CHANNEL_KEYS = (
    "change_mask_pre",
    "change_mask_post",
    "hunk_ids",
    "edit_ops",
)
_DOMAIN_SIDE_CHANNEL_KEYS = (
    "domain_ids",
    "role_ids",
    "entity_ids",
    "scope_ids",
    "source_doc_ids",
    "confidence_ids",
)
_SIDE_CHANNEL_KEYS = (
    _ATTENTION_SIDE_CHANNEL_KEY,
    _LOSS_MASK_SIDE_CHANNEL_KEY,
    *_STRUCTURE_SIDE_CHANNEL_KEYS,
    *_PLATFORM_SIDE_CHANNEL_KEYS,
    *_SEMANTIC_SIDE_CHANNEL_KEYS,
    *_TEMPORAL_SIDE_CHANNEL_KEYS,
    *_DOMAIN_SIDE_CHANNEL_KEYS,
)
_LM_BATCH_DIRECT_SIDE_CHANNEL_KEYS = (
    _ATTENTION_SIDE_CHANNEL_KEY,
    _LOSS_MASK_SIDE_CHANNEL_KEY,
    *_STRUCTURE_SIDE_CHANNEL_KEYS,
    *_PLATFORM_SIDE_CHANNEL_KEYS,
)
_FAMILY_SIDE_CHANNEL_KEYS: dict[str, tuple[str, ...]] = {
    "semantic_graph": _SEMANTIC_SIDE_CHANNEL_KEYS,
    "temporal_diff": _TEMPORAL_SIDE_CHANNEL_KEYS,
    "domain_routes": _DOMAIN_SIDE_CHANNEL_KEYS,
}
_SIDE_CHANNEL_SCHEMA_VERSION = 1
_SIDE_CHANNEL_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    _ATTENTION_SIDE_CHANNEL_KEY: ("token_attention_mask",),
    "structure_ids": ("token_structure_ids",),
    "dep_levels": ("token_dep_levels",),
    "ast_depth_ids": ("token_ast_depth",),
    "sibling_index_ids": ("token_sibling_index",),
    "node_type_ids": ("token_ast_node_type",),
    "platform_ids": ("token_platform_ids",),
    "symbol_ids": ("token_symbol_ids",),
    "call_targets": ("token_call_targets",),
    "type_refs": ("token_type_refs",),
    "def_use": ("token_def_use",),
    "change_mask_pre": ("token_change_mask_pre",),
    "change_mask_post": ("token_change_mask_post",),
    "hunk_ids": ("hunk_id_per_token",),
    "edit_ops": ("edit_op_per_token",),
    "domain_ids": ("token_domain_ids",),
    "role_ids": ("token_role_ids",),
    "entity_ids": ("token_entity_ids",),
    "scope_ids": ("token_scope_ids",),
    "source_doc_ids": ("token_source_doc_ids",),
    "confidence_ids": ("token_confidence_ids",),
}
_DOCUMENT_ID_KEYS = ("document_ids", "doc_ids", "packing_document_ids")
_SIDE_CHANNEL_ALIAS_TO_KEY = {
    alias: key
    for key, aliases in _SIDE_CHANNEL_KEY_ALIASES.items()
    for alias in aliases
}
_SUPPORTED_SIDE_CHANNEL_ENTRY_KEYS = (
    *_SIDE_CHANNEL_KEYS,
    *_SIDE_CHANNEL_ALIAS_TO_KEY,
)
_AMBIGUOUS_SIDECAR_KEYS = (
    "side_channels",
)
_UNSUPPORTED_NGRAM_SIDECAR_KEYS = (
    "ngram_ids",
    "ngram_hash",
    "ngram_hash_ids",
    "ngram_sidecar",
    "ngrams",
)
_GRAPH_SIDECAR_SCHEMAS = ("cppmega_graph_routes_v1", "cppmega_graph_routes_v2")
_GRAPH_PAIR_EDGE_KEYS = ("token_call_edges", "token_type_edges")
_GRAPH_TRIPLE_EDGE_KEYS = (
    "token_domain_edges",
    "token_build_edges",
    "token_shell_edges",
    "token_diagnostic_edges",
    "token_cross_domain_edges",
)
_GRAPH_EDGE_KEYS = (*_GRAPH_PAIR_EDGE_KEYS, *_GRAPH_TRIPLE_EDGE_KEYS)
_GRAPH_CHUNK_KEYS = (
    "token_chunk_starts",
    "token_chunk_ends",
    "token_chunk_kinds",
    "token_chunk_dep_levels",
)
_GRAPH_SIDECAR_KEYS = (*_GRAPH_EDGE_KEYS, *_GRAPH_CHUNK_KEYS)
_GRAPH_RELATION_BY_KEY = {
    "token_call_edges": "call",
    "token_type_edges": "type",
    "token_domain_edges": "domain",
    "token_build_edges": "build",
    "token_shell_edges": "shell",
    "token_diagnostic_edges": "diagnostic",
    "token_cross_domain_edges": "cross_domain",
}

_INDEX_DTYPES: dict[int, np.dtype] = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.int8),
    3: np.dtype(np.int16),
    4: np.dtype(np.int32),
    5: np.dtype(np.int64),
    8: np.dtype(np.uint16),
}

_NAMED_DTYPES: dict[str, np.dtype] = {
    "uint8": np.dtype(np.uint8),
    "int8": np.dtype(np.int8),
    "int16": np.dtype(np.int16),
    "uint16": np.dtype(np.uint16),
    "int32": np.dtype(np.int32),
    "uint32": np.dtype(np.uint32),
    "int64": np.dtype(np.int64),
    "uint64": np.dtype(np.uint64),
}

_SIDE_CHANNEL_NAMED_DTYPES: dict[str, np.dtype] = {
    **_NAMED_DTYPES,
    "float32": np.dtype(np.float32),
}


@dataclass(frozen=True)
class MegatronIndexedMetadata:
    """Parsed on-disk layout metadata for a Megatron indexed token shard."""

    bin_path: Path
    idx_path: Path | None
    metadata_path: Path | None
    dtype: str
    sequence_count: int
    document_count: int
    token_count: int
    source_format: str = "megatron"


@dataclass(frozen=True)
class MegatronIndexedMultiShardMetadata:
    """Aggregated metadata for a directory of Megatron indexed token shards."""

    root_path: Path
    shard_count: int
    dtype: str
    sequence_count: int
    document_count: int
    token_count: int
    shards: tuple[MegatronIndexedMetadata, ...]
    source_format: str = "megatron-multishard"


@dataclass(frozen=True)
class MegatronGraphRoutePacket:
    """Document-aligned graph route sidecar for one sequence/window."""

    graph: GraphPacket
    chunk_starts: np.ndarray
    chunk_ends: np.ndarray
    chunk_kinds: np.ndarray
    chunk_dep_levels: np.ndarray
    edge_kinds: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def num_chunks(self) -> int:
        return int(self.chunk_starts.shape[0])


class MegatronIndexedDataset:
    """Read fixed token windows from Megatron .bin/.idx shards.

    The reader intentionally implements only the stable MMIDIDX layout and
    explicit raw-binary handoffs.  Unknown headers, dtype codes, and ambiguous
    raw binaries fail closed instead of pulling in the original training stack.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        seq_len: int,
        batch_size: int,
        dtype: str | np.dtype | None = None,
        metadata_path: str | Path | None = None,
        token_key: str = "tokens",
        shuffle: bool = False,
        seed: int = 0,
        loop: bool = False,
        resume_batch: int = 0,
        metadata: TokenDatasetMetadata | None = None,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if resume_batch < 0:
            raise ValueError("resume_batch must be non-negative")
        if token_key != "tokens":
            raise ValueError("Megatron indexed datasets use fixed token_key='tokens'")

        resolved = _resolve_paths(path, metadata_path=metadata_path)
        sidecar = _load_sidecar(resolved.metadata_path)

        self.path = resolved.prefix
        self.bin_path = resolved.bin_path
        self.idx_path = resolved.idx_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.token_key = token_key
        self.shuffle = shuffle
        self.seed = seed
        self.loop = loop
        self.resume_batch = resume_batch

        if self.idx_path is not None and self.idx_path.exists():
            index = _parse_mmididx(self.idx_path)
            token_dtype = index.dtype
            sequence_offsets = index.sequence_offsets
            sequence_lengths = index.sequence_lengths
            document_count = int(index.document_indices.shape[0] - 1)
            source_format = str(sidecar.get("source_format", "megatron"))
        else:
            raw_dtype = _coerce_dtype(dtype or sidecar.get("dtype"))
            if raw_dtype is None:
                raise ValueError(
                    "raw .bin datasets require an explicit dtype or JSON sidecar dtype"
                )
            token_count = _raw_token_count(
                self.bin_path, dtype=raw_dtype, sidecar=sidecar
            )
            token_dtype = raw_dtype
            sequence_offsets = np.array([0], dtype=np.int64)
            sequence_lengths = np.array([token_count], dtype=np.int64)
            document_count = int(sidecar.get("document_count", 1))
            source_format = str(sidecar.get("source_format", "megatron-raw"))

        _validate_bin_references(
            self.bin_path,
            dtype=token_dtype,
            offsets=sequence_offsets,
            lengths=sequence_lengths,
        )
        self._restores_compact_fixed_rows = _is_compact_fixed_row_shard(
            sidecar,
            sequence_lengths=sequence_lengths,
            seq_len=self.seq_len,
        )
        if self._restores_compact_fixed_rows:
            windows = sequence_offsets.astype(np.int64, copy=True)
            window_lengths = sequence_lengths.astype(np.int64, copy=True)
        else:
            windows = _build_windows(
                sequence_offsets,
                sequence_lengths,
                self.seq_len,
                itemsize=token_dtype.itemsize,
            )
            window_lengths = np.full(windows.shape, self.seq_len, dtype=np.int64)
        if not len(windows):
            raise ValueError("Megatron token data does not contain a full fixed-shape sample")

        self._dtype = token_dtype
        self._sequence_offsets = sequence_offsets.astype(np.int64, copy=True)
        self._sequence_lengths = sequence_lengths.astype(np.int64, copy=True)
        self._bin_mmap = np.memmap(self.bin_path, mode="r", dtype=np.uint8)
        self._windows = windows
        self._window_lengths = window_lengths
        self._side_channels = _load_side_channels(
            sidecar,
            prefix=resolved.prefix,
            metadata_path=resolved.metadata_path,
            token_dtype=token_dtype,
            token_count=int(sequence_lengths.sum()),
            token_windows=windows,
        )
        self._document_ids = _load_document_ids(
            sidecar,
            prefix=resolved.prefix,
            metadata_path=resolved.metadata_path,
            token_dtype=token_dtype,
            token_count=int(sequence_lengths.sum()),
            token_windows=windows,
        )
        self._graph_sidecars = _load_graph_sidecars(
            sidecar,
            prefix=resolved.prefix,
            metadata_path=resolved.metadata_path,
            sequence_count=int(sequence_lengths.shape[0]),
        )
        self.index_metadata = MegatronIndexedMetadata(
            bin_path=self.bin_path,
            idx_path=self.idx_path if self.idx_path and self.idx_path.exists() else None,
            metadata_path=resolved.metadata_path,
            dtype=self._dtype.name,
            sequence_count=int(sequence_lengths.shape[0]),
            document_count=document_count,
            token_count=int(sequence_lengths.sum()),
            source_format=source_format,
        )
        self.metadata = metadata if metadata is not None else _token_metadata(
            sidecar, source_format=source_format
        )

    def __len__(self) -> int:
        return self.num_batches

    @property
    def num_samples(self) -> int:
        return int(self._windows.shape[0])

    @property
    def num_batches(self) -> int:
        return self.num_samples // self.batch_size

    @property
    def dropped_samples(self) -> int:
        return self.num_samples - self.num_batches * self.batch_size

    def sample_order(self, *, epoch: int = 0) -> np.ndarray:
        """Return deterministic sample order for an epoch."""

        order = np.arange(self.num_samples, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + epoch)
            rng.shuffle(order)
        return order

    def iter_batches(
        self,
        *,
        resume_batch: int | None = None,
        epoch: int = 0,
        loop: bool | None = None,
    ) -> Iterator[LMTokenBatch]:
        """Yield fixed-shape LMTokenBatch objects."""

        effective_loop = self.loop if loop is None else loop
        start_batch = self.resume_batch if resume_batch is None else resume_batch
        if start_batch < 0:
            raise ValueError("resume_batch must be non-negative")

        current_epoch = epoch
        batch_offset = start_batch
        while True:
            if self.num_batches == 0:
                return
            if batch_offset >= self.num_batches:
                if not effective_loop:
                    return
                current_epoch += batch_offset // self.num_batches
                batch_offset = batch_offset % self.num_batches

            order = self.sample_order(epoch=current_epoch)
            for local_batch in range(batch_offset, self.num_batches):
                sample_idx = order[
                    local_batch * self.batch_size : (local_batch + 1) * self.batch_size
                ]
                yield self._make_batch(sample_idx)

            if not effective_loop:
                return
            current_epoch += 1
            batch_offset = 0

    def cursor_after(self, consumed_batches: int, *, epoch: int = 0) -> BatchCursor:
        """Return the deterministic cursor after consumed_batches."""

        if consumed_batches < 0:
            raise ValueError("consumed_batches must be non-negative")
        total_batches = self.resume_batch + consumed_batches
        if self.num_batches == 0:
            return BatchCursor(
                epoch=epoch,
                batch_offset=0,
                global_batch_offset=total_batches,
            )
        return BatchCursor(
            epoch=epoch + total_batches // self.num_batches,
            batch_offset=total_batches % self.num_batches,
            global_batch_offset=total_batches,
        )

    def token_id_range(self) -> tuple[int, int]:
        """Return the min/max token IDs in the indexed token storage."""

        token_count = self.index_metadata.token_count
        if token_count <= 0:
            raise ValueError("Megatron token data does not contain any tokens")
        token_view = np.memmap(self.bin_path, mode="r", dtype=self._dtype)[:token_count]
        token_min = int(token_view.min())
        token_max = int(token_view.max())
        if token_min < 0:
            raise ValueError("token IDs must be non-negative")
        if token_max > np.iinfo(np.int32).max:
            raise ValueError("token IDs exceed int32 range")
        return token_min, token_max

    def graph_route_packet_for_sample(self, sample_index: int) -> MegatronGraphRoutePacket:
        """Return document-aligned graph routes for a fixed-shape sample.

        Graph route sidecars are stored per MMIDIDX sequence/document.  They are
        valid only when the sample window exactly equals one sequence.  If a
        caller opens a larger sequence as multiple token windows, slicing the
        chunk graph would require coordinate remapping, so this fails closed.
        """

        if not self._graph_sidecars:
            raise ValueError("Megatron indexed dataset has no graph route sidecars")
        if sample_index < 0 or sample_index >= self.num_samples:
            raise IndexError(f"sample_index {sample_index} out of range for {self.num_samples} samples")
        sequence_index = self._sequence_index_for_window(int(sample_index))
        starts = _read_graph_vector(self._graph_sidecars["token_chunk_starts"], sequence_index)
        ends = _read_graph_vector(self._graph_sidecars["token_chunk_ends"], sequence_index)
        kinds = _read_graph_vector(self._graph_sidecars["token_chunk_kinds"], sequence_index)
        dep_levels = _read_graph_vector(self._graph_sidecars["token_chunk_dep_levels"], sequence_index)
        lengths = {
            "token_chunk_starts": int(starts.shape[0]),
            "token_chunk_ends": int(ends.shape[0]),
            "token_chunk_kinds": int(kinds.shape[0]),
            "token_chunk_dep_levels": int(dep_levels.shape[0]),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"graph chunk sidecar length mismatch: {lengths}")
        num_nodes = int(starts.shape[0])
        if (
            np.any(starts < 0)
            or np.any(ends <= starts)
            or np.any(ends > self.seq_len)
        ):
            raise ValueError(
                "graph chunk spans must satisfy 0 <= start < end <= sequence length"
            )
        edges: dict[str, EdgeIndex] = {}
        edge_kinds: dict[str, np.ndarray] = {}
        for key, relation in _GRAPH_RELATION_BY_KEY.items():
            if key not in self._graph_sidecars:
                continue
            if key in _GRAPH_PAIR_EDGE_KEYS:
                pairs = _read_graph_pairs(self._graph_sidecars[key], sequence_index)
                relation_num_nodes = num_nodes
            else:
                triples = _read_graph_triples(self._graph_sidecars[key], sequence_index)
                pairs = triples[:, :2]
                edge_kinds[relation] = triples[:, 2].astype(np.int32, copy=False)
                relation_num_nodes = self.seq_len
            if pairs.size and (np.any(pairs < 0) or np.any(pairs >= relation_num_nodes)):
                raise ValueError(
                    f"{key} endpoints exceed {relation_num_nodes} nodes for sample {sample_index}"
                )
            edges[relation] = EdgeIndex.from_pairs(
                pairs,
                relation=relation,
                num_nodes=relation_num_nodes,
            )
        return MegatronGraphRoutePacket(
            graph=GraphPacket(edges=edges, num_nodes=num_nodes),
            chunk_starts=starts.astype(np.int32, copy=False),
            chunk_ends=ends.astype(np.int32, copy=False),
            chunk_kinds=kinds.astype(np.int32, copy=False),
            chunk_dep_levels=dep_levels.astype(np.int32, copy=False),
            edge_kinds=edge_kinds,
        )

    def _sequence_index_for_window(self, window_index: int) -> int:
        byte_offset = int(self._windows[window_index])
        sequence_index = int(np.searchsorted(self._sequence_offsets, byte_offset, side="right") - 1)
        if sequence_index < 0 or sequence_index >= int(self._sequence_offsets.shape[0]):
            raise ValueError(f"window {window_index} does not map to a sequence")
        if int(self._sequence_offsets[sequence_index]) != byte_offset:
            raise NotImplementedError(
                "graph route sidecars are document-aligned; token-window slicing "
                "requires a sequence-aligned sample start"
            )
        sequence_length = int(self._sequence_lengths[sequence_index])
        if self._restores_compact_fixed_rows:
            if sequence_length > self.seq_len:
                raise ValueError(
                    "compact fixed-row sequence length exceeds the configured seq_len"
                )
        elif sequence_length != self.seq_len:
            raise NotImplementedError(
                "graph route sidecars require sequence length to equal seq_len; "
                "regenerate packed graph sidecars at the training block size"
            )
        return sequence_index

    def _make_batch(self, sample_idx: np.ndarray) -> LMTokenBatch:
        return _lm_batch_from_numpy(
            self._make_numpy_batch(sample_idx),
            graph_batch=self._make_graph_batch(sample_idx),
        )

    def _make_graph_batch(self, sample_idx: np.ndarray) -> GraphBatch | None:
        if not self._graph_sidecars:
            return None
        packets = [self.graph_route_packet_for_sample(int(index)) for index in sample_idx]
        return _graph_batch_from_route_packets(packets)

    def _make_numpy_batch(self, sample_idx: np.ndarray) -> dict[str, np.ndarray]:
        tokens = np.zeros((sample_idx.shape[0], self.seq_len), dtype=np.int32)
        side_channels = {
            key: np.zeros(
                (sample_idx.shape[0], self.seq_len),
                dtype=_target_side_channel_dtype(key),
            )
            for key in self._side_channels
        }
        if "hunk_ids" in side_channels:
            side_channels["hunk_ids"].fill(-1)
        document_ids = (
            np.zeros((sample_idx.shape[0], self.seq_len), dtype=np.int32)
            if self._document_ids is not None
            else None
        )
        for row, window_index in enumerate(sample_idx):
            resolved_window = int(window_index)
            byte_offset = int(self._windows[resolved_window])
            window_length = int(self._window_lengths[resolved_window])
            token_view = np.frombuffer(
                self._bin_mmap,
                dtype=self._dtype,
                count=window_length,
                offset=byte_offset,
            )
            tokens[row, :window_length] = _to_int32_tokens(token_view)
            for key, storage in self._side_channels.items():
                side_view = np.frombuffer(
                    storage.mmap,
                    dtype=storage.dtype,
                    count=window_length,
                    offset=int(storage.windows[resolved_window]),
                )
                side_channels[key][row, :window_length] = _to_side_channel_values(
                    key, side_view
                )
            if self._document_ids is not None and document_ids is not None:
                doc_view = np.frombuffer(
                    self._document_ids.mmap,
                    dtype=self._document_ids.dtype,
                    count=window_length,
                    offset=int(self._document_ids.windows[resolved_window]),
                )
                document_ids[row, :window_length] = _to_document_id_values(doc_view)

        batch: dict[str, np.ndarray] = {"tokens": tokens}
        for key in _SIDE_CHANNEL_KEYS:
            if key in side_channels:
                batch[key] = side_channels[key]
        if document_ids is not None:
            batch["document_ids"] = document_ids
        return batch


class MegatronIndexedMultiShardDataset:
    """Read a directory of fixed-shape Megatron indexed shards as one dataset."""

    def __init__(
        self,
        path: str | Path,
        *,
        seq_len: int,
        batch_size: int,
        dtype: str | np.dtype | None = None,
        metadata_path: str | Path | None = None,
        token_key: str = "tokens",
        shuffle: bool = False,
        seed: int = 0,
        loop: bool = False,
        resume_batch: int = 0,
        metadata: TokenDatasetMetadata | None = None,
    ) -> None:
        if metadata_path is not None:
            raise ValueError(
                "multi-shard Megatron directories use per-shard sidecars; "
                "metadata_path is not supported"
            )
        if resume_batch < 0:
            raise ValueError("resume_batch must be non-negative")

        self.path = Path(path)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.token_key = token_key
        self.shuffle = shuffle
        self.seed = seed
        self.loop = loop
        self.resume_batch = resume_batch

        prefixes = _find_multishard_prefixes(self.path)
        self._shards = tuple(
            MegatronIndexedDataset(
                prefix,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                token_key=token_key,
            )
            for prefix in prefixes
        )
        schema = _validate_multishard_schema(self._shards)
        self._side_channel_keys = schema.side_channel_keys
        self._document_ids_present = schema.document_ids_present
        self._batch_keys = (
            "tokens",
            *self._side_channel_keys,
            *(("document_ids",) if self._document_ids_present else ()),
        )
        self._side_channels = {key: None for key in self._side_channel_keys}
        sample_counts = np.asarray(
            [shard.num_samples for shard in self._shards], dtype=np.int64
        )
        self._sample_offsets = np.concatenate(
            [np.array([0], dtype=np.int64), np.cumsum(sample_counts, dtype=np.int64)]
        )
        self.index_metadata = MegatronIndexedMultiShardMetadata(
            root_path=self.path,
            shard_count=len(self._shards),
            dtype=schema.token_dtype,
            sequence_count=sum(
                shard.index_metadata.sequence_count for shard in self._shards
            ),
            document_count=sum(
                shard.index_metadata.document_count for shard in self._shards
            ),
            token_count=sum(shard.index_metadata.token_count for shard in self._shards),
            shards=tuple(shard.index_metadata for shard in self._shards),
        )
        self.metadata = (
            metadata
            if metadata is not None
            else _multi_shard_token_metadata(self._shards)
        )

    def __len__(self) -> int:
        return self.num_batches

    @property
    def num_samples(self) -> int:
        return int(self._sample_offsets[-1])

    @property
    def num_batches(self) -> int:
        return self.num_samples // self.batch_size

    @property
    def dropped_samples(self) -> int:
        return self.num_samples - self.num_batches * self.batch_size

    def sample_order(self, *, epoch: int = 0) -> np.ndarray:
        order = np.arange(self.num_samples, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + epoch)
            rng.shuffle(order)
        return order

    def iter_batches(
        self,
        *,
        resume_batch: int | None = None,
        epoch: int = 0,
        loop: bool | None = None,
    ) -> Iterator[LMTokenBatch]:
        effective_loop = self.loop if loop is None else loop
        start_batch = self.resume_batch if resume_batch is None else resume_batch
        if start_batch < 0:
            raise ValueError("resume_batch must be non-negative")

        current_epoch = epoch
        batch_offset = start_batch
        while True:
            if self.num_batches == 0:
                return
            if batch_offset >= self.num_batches:
                if not effective_loop:
                    return
                current_epoch += batch_offset // self.num_batches
                batch_offset = batch_offset % self.num_batches

            order = self.sample_order(epoch=current_epoch)
            for local_batch in range(batch_offset, self.num_batches):
                sample_idx = order[
                    local_batch * self.batch_size : (local_batch + 1) * self.batch_size
                ]
                yield self._make_batch(sample_idx)

            if not effective_loop:
                return
            current_epoch += 1
            batch_offset = 0

    def cursor_after(self, consumed_batches: int, *, epoch: int = 0) -> BatchCursor:
        if consumed_batches < 0:
            raise ValueError("consumed_batches must be non-negative")
        total_batches = self.resume_batch + consumed_batches
        if self.num_batches == 0:
            return BatchCursor(
                epoch=epoch,
                batch_offset=0,
                global_batch_offset=total_batches,
            )
        return BatchCursor(
            epoch=epoch + total_batches // self.num_batches,
            batch_offset=total_batches % self.num_batches,
            global_batch_offset=total_batches,
        )

    def token_id_range(self) -> tuple[int, int]:
        ranges = [shard.token_id_range() for shard in self._shards]
        return min(low for low, _ in ranges), max(high for _, high in ranges)

    def graph_route_packet_for_sample(self, sample_index: int) -> MegatronGraphRoutePacket:
        if sample_index < 0 or sample_index >= self.num_samples:
            raise IndexError(f"sample_index {sample_index} out of range for {self.num_samples} samples")
        shard_index = int(
            np.searchsorted(
                self._sample_offsets[1:],
                np.asarray([sample_index], dtype=np.int64),
                side="right",
            )[0]
        )
        local_sample_index = int(sample_index - self._sample_offsets[shard_index])
        return self._shards[shard_index].graph_route_packet_for_sample(local_sample_index)

    def _make_batch(self, sample_idx: np.ndarray) -> LMTokenBatch:
        return _lm_batch_from_numpy(
            self._make_numpy_batch(sample_idx),
            graph_batch=self._make_graph_batch(sample_idx),
        )

    def _make_graph_batch(self, sample_idx: np.ndarray) -> GraphBatch | None:
        if not any(shard._graph_sidecars for shard in self._shards):
            return None
        packets = [self.graph_route_packet_for_sample(int(index)) for index in sample_idx]
        return _graph_batch_from_route_packets(packets)

    def _make_numpy_batch(self, sample_idx: np.ndarray) -> dict[str, np.ndarray]:
        shard_indices = np.searchsorted(
            self._sample_offsets[1:],
            sample_idx,
            side="right",
        )
        arrays: dict[str, np.ndarray] | None = None
        for shard_index in np.unique(shard_indices):
            row_positions = np.flatnonzero(shard_indices == shard_index)
            local_sample_idx = (
                sample_idx[row_positions] - self._sample_offsets[int(shard_index)]
            )
            shard_arrays = self._shards[int(shard_index)]._make_numpy_batch(
                local_sample_idx.astype(np.int64, copy=False)
            )
            if tuple(shard_arrays) != self._batch_keys:
                raise ValueError("Megatron multi-shard batch schema changed during read")
            if arrays is None:
                arrays = {
                    key: np.empty(
                        (sample_idx.shape[0], self.seq_len),
                        dtype=value.dtype,
                    )
                    for key, value in shard_arrays.items()
                }
            for key, value in shard_arrays.items():
                arrays[key][row_positions] = value
        if arrays is None:
            raise ValueError("cannot build an empty Megatron multi-shard batch")
        return arrays


def open_megatron_indexed_dataset(
    path: str | Path,
    *,
    seq_len: int,
    batch_size: int,
    dtype: str | np.dtype | None = None,
    metadata_path: str | Path | None = None,
    token_key: str = "tokens",
    shuffle: bool = False,
    seed: int = 0,
    loop: bool = False,
    resume_batch: int = 0,
    metadata: TokenDatasetMetadata | None = None,
) -> MegatronIndexedDataset | MegatronIndexedMultiShardDataset:
    """Open standalone local Megatron-indexed shards for CLI/training code.

    This is the explicit fail-closed ingress for macOS/MLX paths that already
    have Megatron .bin/.idx token shards.  It intentionally depends only on
    the local reader, NumPy, and MLX; it does not import Megatron or Torch
    runtime modules.
    """

    path = Path(path)
    if path.is_dir():
        return MegatronIndexedMultiShardDataset(
            path,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
            metadata_path=metadata_path,
            token_key=token_key,
            shuffle=shuffle,
            seed=seed,
            loop=loop,
            resume_batch=resume_batch,
            metadata=metadata,
        )
    return MegatronIndexedDataset(
        path,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
        metadata_path=metadata_path,
        token_key=token_key,
        shuffle=shuffle,
        seed=seed,
        loop=loop,
        resume_batch=resume_batch,
        metadata=metadata,
    )


def megatron_indexed_side_channel_schema() -> dict[str, dict[str, object]]:
    """Return the documented token-aligned side-channel schema.

    Canonical keys are the names delivered on :class:LMTokenBatch.  Aliases
    match cppmega Parquet token-level column names and are normalized at the
    JSON sidecar boundary.
    """

    return {
        key: {
            "aliases": list(_SIDE_CHANNEL_KEY_ALIASES.get(key, ())),
            "default_dtype": _default_side_channel_dtype(key).name,
            "target_dtype": _target_side_channel_dtype(key).name,
            "allowed_dtypes": _allowed_side_channel_dtype_names(key),
            "model_kwarg": key in (*_STRUCTURE_SIDE_CHANNEL_KEYS, *_PLATFORM_SIDE_CHANNEL_KEYS),
            "family": _side_channel_family(key),
        }
        for key in _SIDE_CHANNEL_KEYS
    }


def megatron_indexed_graph_sidecar_schema() -> dict[str, dict[str, object]]:
    """Return the documented document-aligned graph route sidecar schema."""

    return {
        key: {
            "kind": "edge_pairs"
            if key in _GRAPH_PAIR_EDGE_KEYS
            else "edge_triples"
            if key in _GRAPH_TRIPLE_EDGE_KEYS
            else "ragged_1d",
            "dtype": "int32"
            if key in _GRAPH_EDGE_KEYS
            else "uint32"
            if key in ("token_chunk_starts", "token_chunk_ends")
            else "uint16",
            "shape_tail": [2]
            if key in _GRAPH_PAIR_EDGE_KEYS
            else [3]
            if key in _GRAPH_TRIPLE_EDGE_KEYS
            else [],
            "schema": "cppmega_graph_routes_v2",
        }
        for key in _GRAPH_SIDECAR_KEYS
    }


def _graph_batch_from_route_packets(
    packets: list[MegatronGraphRoutePacket],
) -> GraphBatch | None:
    if not packets:
        return None
    return GraphBatch(
        graphs=tuple(packet.graph for packet in packets),
        chunk_starts=tuple(mx.array(packet.chunk_starts) for packet in packets),
        chunk_ends=tuple(mx.array(packet.chunk_ends) for packet in packets),
        chunk_kinds=tuple(mx.array(packet.chunk_kinds) for packet in packets),
        chunk_dep_levels=tuple(mx.array(packet.chunk_dep_levels) for packet in packets),
        edge_kinds=tuple(
            {
                relation: mx.array(values, dtype=mx.int32)
                for relation, values in packet.edge_kinds.items()
            }
            for packet in packets
        ),
    )


def _lm_batch_from_numpy(
    arrays: dict[str, np.ndarray],
    *,
    graph_batch: GraphBatch | None = None,
) -> LMTokenBatch:
    kwargs = {
        key: mx.array(arrays[key])
        for key in _LM_BATCH_DIRECT_SIDE_CHANNEL_KEYS
        if key in arrays
    }
    side_channels = {
        family: {
            key: mx.array(arrays[key])
            for key in keys
            if key in arrays
        }
        for family, keys in _FAMILY_SIDE_CHANNEL_KEYS.items()
    }
    side_channels = {
        family: columns
        for family, columns in side_channels.items()
        if columns
    }
    document_ids = arrays.get("document_ids")
    return LMTokenBatch(
        tokens=mx.array(arrays["tokens"]),
        document_ids=None if document_ids is None else mx.array(document_ids),
        graph_batch=graph_batch,
        side_channels=side_channels or None,
        **kwargs,
    )


@dataclass(frozen=True)
class _ResolvedPaths:
    prefix: Path
    bin_path: Path
    idx_path: Path | None
    metadata_path: Path | None


@dataclass(frozen=True)
class _ParsedIndex:
    dtype: np.dtype
    sequence_lengths: np.ndarray
    sequence_offsets: np.ndarray
    document_indices: np.ndarray


@dataclass(frozen=True)
class _SideChannelStorage:
    path: Path
    dtype: np.dtype
    windows: np.ndarray
    mmap: np.memmap


@dataclass(frozen=True)
class _GraphSidecarStorage:
    key: str
    kind: str
    dtype: np.dtype
    offsets: np.ndarray
    data: np.ndarray
    item_count: int
    shape_tail: tuple[int, ...]
    coordinate_space: str | None


@dataclass(frozen=True)
class _MultiShardSchema:
    token_dtype: str
    side_channel_keys: tuple[str, ...]
    side_channel_dtypes: tuple[tuple[str, str], ...]
    document_ids_present: bool
    document_ids_dtype: str | None


def _find_multishard_prefixes(path: Path) -> tuple[Path, ...]:
    if not path.is_dir():
        raise ValueError(f"Megatron multi-shard path must be a directory: {path}")
    idx_paths = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix == ".idx"
    )
    if not idx_paths:
        raise ValueError(
            f"Megatron multi-shard directory must contain .idx shards: {path}"
        )
    return tuple(idx_path.with_suffix("") for idx_path in idx_paths)


def _validate_multishard_schema(
    shards: tuple[MegatronIndexedDataset, ...],
) -> _MultiShardSchema:
    if not shards:
        raise ValueError("Megatron multi-shard dataset requires at least one shard")
    reference = _single_shard_schema(shards[0])
    for shard in shards[1:]:
        schema = _single_shard_schema(shard)
        if schema != reference:
            raise ValueError(
                "Megatron multi-shard schema mismatch: token dtype, side-channel "
                "keys/dtypes, and document-id sidecars must match across shards"
            )
    return reference


def _single_shard_schema(shard: MegatronIndexedDataset) -> _MultiShardSchema:
    side_channel_keys = tuple(
        key for key in _SIDE_CHANNEL_KEYS if key in shard._side_channels
    )
    side_channel_dtypes = tuple(
        (key, shard._side_channels[key].dtype.name) for key in side_channel_keys
    )
    return _MultiShardSchema(
        token_dtype=shard.index_metadata.dtype,
        side_channel_keys=side_channel_keys,
        side_channel_dtypes=side_channel_dtypes,
        document_ids_present=shard._document_ids is not None,
        document_ids_dtype=(
            None if shard._document_ids is None else shard._document_ids.dtype.name
        ),
    )


def _multi_shard_token_metadata(
    shards: tuple[MegatronIndexedDataset, ...],
) -> TokenDatasetMetadata:
    first = shards[0].metadata
    for shard in shards[1:]:
        current = shard.metadata
        if (
            current.vocab_size != first.vocab_size
            or current.tokenizer_contract != first.tokenizer_contract
            or current.local_profile_vocab_size != first.local_profile_vocab_size
            or current.megacpp_tokenizer_vocab_size
            != first.megacpp_tokenizer_vocab_size
        ):
            raise ValueError(
                "Megatron multi-shard tokenizer metadata mismatch: vocab and "
                "tokenizer contract must match across shards"
            )
    return TokenDatasetMetadata(
        vocab_size=first.vocab_size,
        tokenizer_contract=first.tokenizer_contract,
        local_profile_vocab_size=first.local_profile_vocab_size,
        megacpp_tokenizer_vocab_size=first.megacpp_tokenizer_vocab_size,
        source_format="megatron-multishard",
    )


def _resolve_paths(
    path: str | Path, *, metadata_path: str | Path | None
) -> _ResolvedPaths:
    raw_path = Path(path)
    explicit_metadata = Path(metadata_path) if metadata_path is not None else None

    if raw_path.suffix == ".idx":
        prefix = raw_path.with_suffix("")
        idx_path: Path | None = raw_path
        bin_path = prefix.with_suffix(".bin")
    elif raw_path.suffix == ".bin":
        prefix = raw_path.with_suffix("")
        bin_path = raw_path
        candidate_idx = prefix.with_suffix(".idx")
        idx_path = candidate_idx if candidate_idx.exists() else None
    elif raw_path.suffix == ".json":
        explicit_metadata = raw_path
        if raw_path.name.endswith(".idx.json"):
            prefix = Path(str(raw_path)[: -len(".idx.json")])
        else:
            prefix = raw_path.with_suffix("")
        bin_path = prefix.with_suffix(".bin")
        candidate_idx = prefix.with_suffix(".idx")
        idx_path = candidate_idx if candidate_idx.exists() else None
    else:
        prefix = raw_path
        bin_path = prefix.with_suffix(".bin")
        candidate_idx = prefix.with_suffix(".idx")
        idx_path = candidate_idx if candidate_idx.exists() else None

    sidecar = explicit_metadata or _find_sidecar(prefix)
    return _ResolvedPaths(
        prefix=prefix,
        bin_path=bin_path,
        idx_path=idx_path,
        metadata_path=sidecar,
    )


def _find_sidecar(prefix: Path) -> Path | None:
    for candidate in (
        Path(str(prefix) + ".idx.json"),
        prefix.with_suffix(".json"),
        Path(str(prefix) + ".bin.json"),
    ):
        if candidate.exists():
            return candidate
    return None


def _load_sidecar(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"metadata sidecar must be a JSON object: {path}")
    return payload


def _load_side_channels(
    sidecar: dict[str, Any],
    *,
    prefix: Path,
    metadata_path: Path | None,
    token_dtype: np.dtype,
    token_count: int,
    token_windows: np.ndarray,
) -> dict[str, _SideChannelStorage]:
    _reject_ambiguous_side_channel_metadata(sidecar)
    entries = _side_channel_entries(sidecar)
    if any(key in entries for key in _SYMBOL_ID_SIDE_CHANNEL_KEYS):
        version = sidecar.get("symbol_identity_schema_version")
        if version != SYMBOL_IDENTITY_SCHEMA_VERSION:
            raise ValueError(
                "semantic symbol sidecars require symbol_identity_schema_version="
                f"{SYMBOL_IDENTITY_SCHEMA_VERSION}, got {version!r}"
            )
    base_dir = metadata_path.parent if metadata_path is not None else prefix.parent
    storages: dict[str, _SideChannelStorage] = {}
    for key, entry in entries.items():
        path, dtype = _parse_side_channel_entry(key, entry, base_dir=base_dir)
        capacity, remainder = divmod(path.stat().st_size, dtype.itemsize)
        if remainder:
            raise ValueError(
                f"{key} side-channel file size is not divisible by dtype itemsize {dtype.itemsize}"
            )
        if int(capacity) != token_count:
            raise ValueError(
                f"{key} side-channel token count {int(capacity)} does not match "
                f"token shard count {token_count}"
            )
        token_item_offset = token_windows // token_dtype.itemsize
        windows = token_item_offset * dtype.itemsize
        mmap = np.memmap(path, mode="r", dtype=np.uint8)
        storages[key] = _SideChannelStorage(
            path=path,
            dtype=dtype,
            windows=windows.astype(np.int64, copy=False),
            mmap=mmap,
        )
    return storages


def _load_document_ids(
    sidecar: dict[str, Any],
    *,
    prefix: Path,
    metadata_path: Path | None,
    token_dtype: np.dtype,
    token_count: int,
    token_windows: np.ndarray,
) -> _SideChannelStorage | None:
    entry = _document_id_entry(sidecar)
    if entry is None:
        return None
    base_dir = metadata_path.parent if metadata_path is not None else prefix.parent
    path, dtype = _parse_side_channel_entry("document_ids", entry, base_dir=base_dir)
    capacity, remainder = divmod(path.stat().st_size, dtype.itemsize)
    if remainder:
        raise ValueError(
            "document_ids file size is not divisible by dtype itemsize "
            f"{dtype.itemsize}"
        )
    if int(capacity) != token_count:
        raise ValueError(
            f"document_ids token count {int(capacity)} does not match "
            f"token shard count {token_count}"
        )
    token_item_offset = token_windows // token_dtype.itemsize
    windows = token_item_offset * dtype.itemsize
    return _SideChannelStorage(
        path=path,
        dtype=dtype,
        windows=windows.astype(np.int64, copy=False),
        mmap=np.memmap(path, mode="r", dtype=np.uint8),
    )


def _load_graph_sidecars(
    sidecar: dict[str, Any],
    *,
    prefix: Path,
    metadata_path: Path | None,
    sequence_count: int,
) -> dict[str, _GraphSidecarStorage]:
    entries = sidecar.get("graph_sidecar_paths")
    if entries is None:
        return {}
    if not isinstance(entries, dict):
        raise ValueError("graph_sidecar_paths must be a mapping of key to CSR metadata")
    schema = sidecar.get("graph_sidecar_schema")
    if schema not in _GRAPH_SIDECAR_SCHEMAS:
        raise ValueError(
            f"unsupported graph_sidecar_schema {schema!r}; expected one of "
            f"{_GRAPH_SIDECAR_SCHEMAS!r}"
        )
    base_dir = metadata_path.parent if metadata_path is not None else prefix.parent
    storages: dict[str, _GraphSidecarStorage] = {}
    for raw_key, entry in entries.items():
        key = str(raw_key)
        if key not in _GRAPH_SIDECAR_KEYS:
            raise NotImplementedError(f"unsupported graph sidecar key {key!r}")
        if schema == "cppmega_graph_routes_v1" and key in _GRAPH_TRIPLE_EDGE_KEYS:
            raise ValueError(f"{key} requires cppmega_graph_routes_v2")
        if key in storages:
            raise ValueError(f"{key} graph sidecar declared more than once")
        storages[key] = _parse_graph_sidecar_entry(
            key,
            entry,
            base_dir=base_dir,
            sequence_count=sequence_count,
            schema=str(schema),
        )
    missing = [key for key in _GRAPH_CHUNK_KEYS if key not in storages]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"graph route sidecars require all chunk metadata; missing {joined}")
    return storages


def _parse_graph_sidecar_entry(
    key: str,
    entry: Any,
    *,
    base_dir: Path,
    sequence_count: int,
    schema: str,
) -> _GraphSidecarStorage:
    if not isinstance(entry, dict):
        raise ValueError(f"{key} graph sidecar entry must be an object")
    try:
        kind = str(entry["kind"])
        offsets_path = Path(str(entry["offsets_path"]))
        data_path = Path(str(entry["data_path"]))
        offset_dtype = str(entry.get("offset_dtype", "int64"))
        dtype = _coerce_graph_sidecar_dtype(key, entry.get("dtype"))
    except KeyError as error:
        raise ValueError(f"{key} graph sidecar entry missing required field {error.args[0]!r}") from error
    if key in _GRAPH_PAIR_EDGE_KEYS:
        if kind != "edge_pairs":
            raise ValueError(f"{key} graph sidecar kind must be edge_pairs, got {kind!r}")
        shape_tail = tuple(int(x) for x in entry.get("shape_tail", [2]))
        if shape_tail != (2,):
            raise ValueError(f"{key} edge sidecar shape_tail must be [2], got {shape_tail}")
    elif key in _GRAPH_TRIPLE_EDGE_KEYS:
        if kind != "edge_triples":
            raise ValueError(f"{key} graph sidecar kind must be edge_triples, got {kind!r}")
        shape_tail = tuple(int(x) for x in entry.get("shape_tail", [3]))
        if shape_tail != (3,):
            raise ValueError(f"{key} edge sidecar shape_tail must be [3], got {shape_tail}")
    elif key in _GRAPH_CHUNK_KEYS:
        if kind != "ragged_1d":
            raise ValueError(f"{key} graph sidecar kind must be ragged_1d, got {kind!r}")
        shape_tail = ()
    else:
        raise NotImplementedError(f"unsupported graph sidecar key {key!r}")
    coordinate_space = entry.get("coordinate_space")
    if schema == "cppmega_graph_routes_v2":
        expected_coordinate = (
            "chunk_index"
            if key in _GRAPH_PAIR_EDGE_KEYS
            or key in {"token_chunk_kinds", "token_chunk_dep_levels"}
            else "token_index"
        )
        if coordinate_space != expected_coordinate:
            raise ValueError(
                f"{key} coordinate_space must be {expected_coordinate!r}, "
                f"got {coordinate_space!r}"
            )
    if offset_dtype != "int64":
        raise ValueError(f"{key} graph sidecar offsets must be int64, got {offset_dtype!r}")
    if not offsets_path.is_absolute():
        offsets_path = base_dir / offsets_path
    if not data_path.is_absolute():
        data_path = base_dir / data_path
    if not offsets_path.exists():
        raise FileNotFoundError(offsets_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    offsets_capacity, offsets_remainder = divmod(
        offsets_path.stat().st_size,
        np.dtype(np.int64).itemsize,
    )
    if offsets_remainder:
        raise ValueError(f"{key} graph offsets file size is not divisible by int64")
    expected_offsets = sequence_count + 1
    if int(offsets_capacity) != expected_offsets:
        raise ValueError(
            f"{key} graph offsets count {int(offsets_capacity)} does not match "
            f"sequence_count+1 {expected_offsets}"
        )
    offsets = np.memmap(offsets_path, mode="r", dtype=np.int64)
    if offsets.shape[0] == 0 or int(offsets[0]) != 0:
        raise ValueError(f"{key} graph offsets must start at 0")
    if np.any(np.diff(offsets) < 0):
        raise ValueError(f"{key} graph offsets must be monotonic")
    item_count = int(offsets[-1])
    declared_count = entry.get("item_count")
    if declared_count is not None and int(declared_count) != item_count:
        raise ValueError(
            f"{key} graph item_count {declared_count} does not match offsets[-1] {item_count}"
        )
    shape_multiplier = int(np.prod(shape_tail, dtype=np.int64)) if shape_tail else 1
    data_capacity, data_remainder = divmod(data_path.stat().st_size, dtype.itemsize)
    if data_remainder:
        raise ValueError(f"{key} graph data file size is not divisible by dtype itemsize")
    expected_values = item_count * shape_multiplier
    if int(data_capacity) != expected_values:
        raise ValueError(
            f"{key} graph data value count {int(data_capacity)} does not match "
            f"item_count*shape_tail {expected_values}"
        )
    data = (
        np.zeros((0,), dtype=dtype)
        if expected_values == 0
        else np.memmap(data_path, mode="r", dtype=dtype)
    )
    return _GraphSidecarStorage(
        key=key,
        kind=kind,
        dtype=dtype,
        offsets=np.asarray(offsets, dtype=np.int64),
        data=data,
        item_count=item_count,
        shape_tail=shape_tail,
        coordinate_space=None if coordinate_space is None else str(coordinate_space),
    )


def _coerce_graph_sidecar_dtype(key: str, value: Any | None) -> np.dtype:
    if value is None:
        raise ValueError(f"{key} graph sidecar entry must include dtype")
    if not isinstance(value, str):
        raise ValueError(f"{key} graph sidecar dtype must be a string")
    dtype = _NAMED_DTYPES.get(value)
    if dtype is None:
        raise ValueError(f"unsupported {key} graph sidecar dtype {value!r}")
    if dtype.kind not in {"i", "u"}:
        raise ValueError(f"{key} graph sidecar dtype must be integer")
    return dtype


def _graph_range(storage: _GraphSidecarStorage, sequence_index: int) -> tuple[int, int]:
    start = int(storage.offsets[sequence_index])
    end = int(storage.offsets[sequence_index + 1])
    if end < start:
        raise ValueError(f"{storage.key} graph offsets are not monotonic at {sequence_index}")
    return start, end


def _read_graph_pairs(storage: _GraphSidecarStorage, sequence_index: int) -> np.ndarray:
    if storage.kind != "edge_pairs" or storage.shape_tail != (2,):
        raise ValueError(f"{storage.key} graph storage is not edge_pairs")
    start, end = _graph_range(storage, sequence_index)
    flat_start = start * 2
    flat_end = end * 2
    values = np.asarray(storage.data[flat_start:flat_end], dtype=np.int64)
    if values.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    pairs = values.reshape(-1, 2)
    if np.any(pairs < 0) or np.any(pairs > np.iinfo(np.int32).max):
        raise ValueError(f"{storage.key} graph edge values must fit int32")
    return pairs.astype(np.int32, copy=False)


def _read_graph_triples(storage: _GraphSidecarStorage, sequence_index: int) -> np.ndarray:
    if storage.kind != "edge_triples" or storage.shape_tail != (3,):
        raise ValueError(f"{storage.key} graph storage is not edge_triples")
    start, end = _graph_range(storage, sequence_index)
    values = np.asarray(storage.data[start * 3 : end * 3], dtype=np.int64)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    triples = values.reshape(-1, 3)
    if np.any(triples < 0) or np.any(triples > np.iinfo(np.int32).max):
        raise ValueError(f"{storage.key} graph edge values must fit non-negative int32")
    return triples.astype(np.int32, copy=False)


def _read_graph_vector(storage: _GraphSidecarStorage, sequence_index: int) -> np.ndarray:
    if storage.kind != "ragged_1d" or storage.shape_tail:
        raise ValueError(f"{storage.key} graph storage is not ragged_1d")
    start, end = _graph_range(storage, sequence_index)
    values = np.asarray(storage.data[start:end], dtype=np.int64)
    if np.any(values < 0) or np.any(values > np.iinfo(np.int32).max):
        raise ValueError(f"{storage.key} graph vector values must fit int32")
    return values.astype(np.int32, copy=False)


def _document_id_entry(sidecar: dict[str, Any]) -> Any | None:
    entries: list[Any] = []
    mapping = sidecar.get("side_channel_paths")
    if mapping is not None:
        if not isinstance(mapping, dict):
            raise ValueError("side_channel_paths must be a mapping of key to path metadata")
        for raw_key, entry in mapping.items():
            if str(raw_key) in _DOCUMENT_ID_KEYS and _declares_side_channel(entry):
                entries.append(entry)
    for key in _DOCUMENT_ID_KEYS:
        if key in sidecar and _declares_side_channel(sidecar[key]):
            entries.append(sidecar[key])
    if len(entries) > 1:
        raise ValueError("document_ids sidecar declared more than once")
    return entries[0] if entries else None


def _reject_ambiguous_side_channel_metadata(sidecar: dict[str, Any]) -> None:
    ambiguous = [
        key
        for key in _AMBIGUOUS_SIDECAR_KEYS
        if key in sidecar and _declares_side_channel(sidecar[key])
    ]
    if ambiguous:
        keys = ", ".join(sorted(ambiguous))
        raise NotImplementedError(
            "Megatron indexed side-channel metadata must use explicit "
            f"'side_channel_paths' or top-level path entries; ambiguous keys: {keys}"
        )

    ngram = [
        key
        for key in _UNSUPPORTED_NGRAM_SIDECAR_KEYS
        if key in sidecar and _declares_side_channel(sidecar[key])
    ]
    if ngram:
        keys = ", ".join(sorted(ngram))
        raise NotImplementedError(
            "Megatron indexed ngram sidecars are not supported; ngram hashes are "
            f"derived from input_ids in the model path. Unsupported keys: {keys}"
        )


def _side_channel_entries(sidecar: dict[str, Any]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    mapping = sidecar.get("side_channel_paths")
    if mapping is not None:
        if not isinstance(mapping, dict):
            raise ValueError("side_channel_paths must be a mapping of key to path metadata")
        for raw_key, entry in mapping.items():
            if str(raw_key) in _DOCUMENT_ID_KEYS:
                continue
            key = _canonical_side_channel_key(str(raw_key))
            if key is None:
                raise NotImplementedError(f"unsupported side-channel key {raw_key!r}")
            if key in entries:
                raise ValueError(f"{key} side-channel declared more than once")
            entries[key] = entry
    for raw_key in _SUPPORTED_SIDE_CHANNEL_ENTRY_KEYS:
        if raw_key in sidecar and _declares_side_channel(sidecar[raw_key]):
            key = _canonical_side_channel_key(raw_key)
            if key is None:
                raise NotImplementedError(f"unsupported side-channel key {raw_key!r}")
            if key in entries:
                raise ValueError(f"{key} side-channel declared more than once")
            entries[key] = sidecar[raw_key]
    return entries


def _canonical_side_channel_key(key: str) -> str | None:
    if key in _SIDE_CHANNEL_KEYS:
        return key
    return _SIDE_CHANNEL_ALIAS_TO_KEY.get(key)


def _parse_side_channel_entry(
    key: str, entry: Any, *, base_dir: Path
) -> tuple[Path, np.dtype]:
    if isinstance(entry, str):
        path = Path(entry)
        dtype_value = None
    elif isinstance(entry, dict):
        if "path" not in entry:
            raise ValueError(f"{key} side-channel entry must include a path")
        path = Path(str(entry["path"]))
        dtype_value = entry.get("dtype")
    else:
        raise ValueError(
            f"{key} side-channel entry must be a path string or object with path/dtype"
        )
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        raise FileNotFoundError(path)
    dtype = _coerce_side_channel_dtype(key, dtype_value)
    return path, dtype


def _coerce_side_channel_dtype(key: str, value: Any | None) -> np.dtype:
    if value is None:
        return _default_side_channel_dtype(key)
    if not isinstance(value, str):
        try:
            dtype = np.dtype(value)
        except TypeError as error:
            raise ValueError(f"unsupported {key} side-channel dtype {value!r}") from error
    else:
        dtype = _SIDE_CHANNEL_NAMED_DTYPES.get(value)
        if dtype is None:
            raise ValueError(f"unsupported {key} side-channel dtype {value!r}")
    if key == _ATTENTION_SIDE_CHANNEL_KEY:
        if dtype != np.dtype(np.float32):
            raise ValueError("attention_mask side-channel dtype must be float32")
        return dtype
    if key in _SYMBOL_ID_SIDE_CHANNEL_KEYS:
        if dtype != np.dtype(np.uint64):
            raise ValueError(
                f"{key} side-channel dtype must be uint64 for v"
                f"{SYMBOL_IDENTITY_SCHEMA_VERSION} symbol identities, got {dtype.name}"
            )
        return dtype
    if dtype.kind not in {"i", "u"}:
        raise ValueError(f"{key} side-channel dtype must be an integer dtype")
    if dtype.itemsize > np.dtype(np.int64).itemsize:
        raise ValueError(f"{key} side-channel dtype {dtype.name!r} is too wide")
    return dtype


def _default_side_channel_dtype(key: str) -> np.dtype:
    if key == _ATTENTION_SIDE_CHANNEL_KEY:
        return np.dtype(np.float32)
    if key in _SYMBOL_ID_SIDE_CHANNEL_KEYS:
        return np.dtype(np.uint64)
    return np.dtype(np.int32)


def _target_side_channel_dtype(key: str) -> np.dtype:
    if key == _ATTENTION_SIDE_CHANNEL_KEY:
        return np.dtype(np.float32)
    if key in _SYMBOL_ID_SIDE_CHANNEL_KEYS:
        return np.dtype(np.uint64)
    return np.dtype(np.int32)


def _allowed_side_channel_dtype_names(key: str) -> list[str]:
    if key == _ATTENTION_SIDE_CHANNEL_KEY:
        return ["float32"]
    if key in _SYMBOL_ID_SIDE_CHANNEL_KEYS:
        return ["uint64"]
    return [
        name
        for name, dtype in _SIDE_CHANNEL_NAMED_DTYPES.items()
        if dtype.kind in {"i", "u"}
    ]


def _side_channel_family(key: str) -> str:
    if key in {_ATTENTION_SIDE_CHANNEL_KEY, _LOSS_MASK_SIDE_CHANNEL_KEY}:
        return "attention"
    if key in _STRUCTURE_SIDE_CHANNEL_KEYS:
        return "structure"
    if key in _PLATFORM_SIDE_CHANNEL_KEYS:
        return "platform"
    if key in _SEMANTIC_SIDE_CHANNEL_KEYS:
        return "semantic_graph"
    if key in _TEMPORAL_SIDE_CHANNEL_KEYS:
        return "temporal_diff"
    if key in _DOMAIN_SIDE_CHANNEL_KEYS:
        return "domain_routes"
    raise ValueError(f"unknown side-channel key {key!r}")


def _to_side_channel_values(key: str, values: np.ndarray) -> np.ndarray:
    if key == _ATTENTION_SIDE_CHANNEL_KEY:
        return values.astype(np.float32, copy=False)
    if key in _SYMBOL_ID_SIDE_CHANNEL_KEYS:
        if values.dtype.kind not in {"i", "u"}:
            raise ValueError(f"{key} side-channel IDs must use an integer dtype")
        if values.dtype.kind == "i" and np.any(values < 0):
            raise ValueError(f"{key} side-channel IDs must be non-negative")
        return values.astype(np.uint64, copy=False)
    if key != "hunk_ids" and np.any(values < 0):
        raise ValueError(f"{key} side-channel IDs must be non-negative")
    if key == "hunk_ids" and np.any(values < np.iinfo(np.int32).min):
        raise ValueError("hunk_ids side-channel values are below int32 range")
    if np.any(values > np.iinfo(np.int32).max):
        raise ValueError(f"{key} side-channel IDs exceed int32 range")
    return values.astype(np.int32, copy=False)


def _to_document_id_values(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind not in {"i", "u"}:
        raise ValueError("document_ids must use an integer dtype")
    if np.any(values < 0):
        raise ValueError("document_ids must be non-negative")
    if np.any(values > np.iinfo(np.int32).max):
        raise ValueError("document_ids exceed int32 range")
    return values.astype(np.int32, copy=False)


def _declares_side_channel(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, dict | list | tuple | set):
        return bool(value)
    return bool(value)


def _parse_mmididx(path: Path) -> _ParsedIndex:
    data = np.memmap(path, mode="r", dtype=np.uint8)
    view = memoryview(data)
    offset = 0

    header = bytes(view[offset : offset + len(_INDEX_HEADER)])
    offset += len(_INDEX_HEADER)
    if header != _INDEX_HEADER:
        raise NotImplementedError(f"unsupported Megatron .idx header in {path}")

    version = _unpack("<Q", view, offset)
    offset += 8
    if version != _INDEX_VERSION:
        raise NotImplementedError(
            f"unsupported Megatron .idx version {version}; expected {_INDEX_VERSION}"
        )

    dtype_code = _unpack("<B", view, offset)
    offset += 1
    dtype = _INDEX_DTYPES.get(dtype_code)
    if dtype is None:
        raise NotImplementedError(
            f"unsupported Megatron token dtype code {dtype_code}"
        )

    sequence_count = _unpack("<Q", view, offset)
    offset += 8
    document_count = _unpack("<Q", view, offset)
    offset += 8

    if sequence_count > np.iinfo(np.int64).max or document_count > np.iinfo(np.int64).max:
        raise ValueError("Megatron index counts exceed supported range")

    lengths = np.frombuffer(
        view, dtype=np.int32, count=int(sequence_count), offset=offset
    ).astype(np.int64, copy=True)
    offset += int(sequence_count) * np.dtype(np.int32).itemsize

    pointers = np.frombuffer(
        view, dtype=np.int64, count=int(sequence_count), offset=offset
    ).astype(np.int64, copy=True)
    offset += int(sequence_count) * np.dtype(np.int64).itemsize

    documents = np.frombuffer(
        view, dtype=np.int64, count=int(document_count), offset=offset
    ).astype(np.int64, copy=True)
    offset += int(document_count) * np.dtype(np.int64).itemsize

    remaining = int(data.size) - offset
    if remaining not in {0, int(sequence_count)}:
        raise NotImplementedError(
            f"unsupported Megatron .idx trailer length {remaining} bytes"
        )
    if remaining == int(sequence_count):
        np.frombuffer(view, dtype=np.int8, count=int(sequence_count), offset=offset)

    _validate_index_arrays(lengths, pointers, documents, itemsize=dtype.itemsize)
    return _ParsedIndex(
        dtype=dtype,
        sequence_lengths=lengths,
        sequence_offsets=pointers,
        document_indices=documents,
    )


def _unpack(fmt: str, data: memoryview, offset: int) -> int:
    size = struct.calcsize(fmt)
    if offset + size > len(data):
        raise ValueError("truncated Megatron .idx header")
    return int(struct.unpack_from(fmt, data, offset)[0])


def _validate_index_arrays(
    lengths: np.ndarray, pointers: np.ndarray, documents: np.ndarray, *, itemsize: int
) -> None:
    if pointers.shape != lengths.shape:
        raise ValueError("Megatron .idx sequence pointer count must match sequence lengths")
    if np.any(lengths < 0):
        raise ValueError("Megatron .idx contains negative sequence lengths")
    if np.any(pointers < 0):
        raise ValueError("Megatron .idx contains negative sequence pointers")
    if lengths.shape[0]:
        expected = np.concatenate(
            [np.array([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)]
        )
        expected *= itemsize
        if not np.array_equal(pointers, expected):
            raise ValueError("Megatron .idx sequence pointers do not match token dtype")
    if documents.shape[0] == 0:
        raise ValueError("Megatron .idx document_indices must include a sentinel")
    if documents[0] != 0 or documents[-1] != lengths.shape[0]:
        raise ValueError("Megatron .idx document_indices must span all sequences")
    if np.any(np.diff(documents) < 0):
        raise ValueError("Megatron .idx document_indices must be monotonic")


def _coerce_dtype(value: str | np.dtype | type[np.generic] | None) -> np.dtype | None:
    if value is None:
        return None
    if isinstance(value, np.dtype):
        dtype = value
    elif isinstance(value, str):
        dtype = _NAMED_DTYPES.get(value)
        if dtype is None:
            raise ValueError(f"unsupported token dtype {value!r}")
    else:
        dtype = np.dtype(value)
    if dtype.name not in _NAMED_DTYPES:
        raise ValueError(f"unsupported token dtype {dtype.name!r}")
    return dtype


def _raw_token_count(
    bin_path: Path, *, dtype: np.dtype, sidecar: dict[str, Any]
) -> int:
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)
    capacity, remainder = divmod(bin_path.stat().st_size, dtype.itemsize)
    if remainder:
        raise ValueError(
            f"{bin_path} size is not divisible by dtype itemsize {dtype.itemsize}"
        )
    token_count = int(sidecar.get("token_count", capacity))
    if token_count < 0:
        raise ValueError("metadata token_count must be non-negative")
    if token_count > capacity:
        raise ValueError("metadata token_count exceeds raw .bin capacity")
    return token_count


def _validate_bin_references(
    bin_path: Path, *, dtype: np.dtype, offsets: np.ndarray, lengths: np.ndarray
) -> None:
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)
    file_size = bin_path.stat().st_size
    itemsize = dtype.itemsize
    ends = offsets + lengths * itemsize
    if np.any(ends > file_size):
        raise ValueError(f"Megatron .idx references bytes past {bin_path}")
    if np.any(offsets % itemsize != 0):
        raise ValueError("Megatron .idx sequence pointers must align to dtype size")


def _is_compact_fixed_row_shard(
    sidecar: dict[str, Any],
    *,
    sequence_lengths: np.ndarray,
    seq_len: int,
) -> bool:
    """Recognize converter output that omits only fixed-row padding."""

    declared_capacity = sidecar.get("source_capacity_token_count")
    if declared_capacity is None:
        return False

    sequence_count = int(sequence_lengths.shape[0])
    expected_capacity = sequence_count * seq_len
    if int(declared_capacity) != expected_capacity:
        raise ValueError(
            "source_capacity_token_count does not match sequence_count * seq_len"
        )
    if int(sidecar.get("document_count", -1)) != sequence_count:
        raise ValueError(
            "compact fixed-row document_count does not match MMIDIDX sequence count"
        )
    token_count = int(sequence_lengths.sum(dtype=np.int64))
    if int(sidecar.get("token_count", -1)) != token_count:
        raise ValueError(
            "compact fixed-row token_count does not match MMIDIDX sequence lengths"
        )
    if np.any(sequence_lengths <= 0) or np.any(sequence_lengths > seq_len):
        raise ValueError(
            "compact fixed-row sequence lengths must satisfy 0 < length <= seq_len"
        )
    return True


def _build_windows(
    sequence_offsets: np.ndarray,
    sequence_lengths: np.ndarray,
    seq_len: int,
    *,
    itemsize: int,
) -> np.ndarray:
    windows: list[int] = []
    for byte_offset, length in zip(sequence_offsets, sequence_lengths, strict=True):
        full = int(length) // seq_len
        if full == 0:
            continue
        start = int(byte_offset)
        for sample in range(full):
            windows.append(start + sample * seq_len * itemsize)
    return np.asarray(windows, dtype=np.int64)


def _to_int32_tokens(tokens: np.ndarray) -> np.ndarray:
    if np.any(tokens < 0):
        raise ValueError("token IDs must be non-negative")
    if np.any(tokens > np.iinfo(np.int32).max):
        raise ValueError("token IDs exceed int32 range")
    return tokens.astype(np.int32, copy=False)


def _token_metadata(
    sidecar: dict[str, Any], *, source_format: str
) -> TokenDatasetMetadata:
    return TokenDatasetMetadata(
        vocab_size=int(sidecar.get("vocab_size", MEGACPP_TOKENIZER_VOCAB_SIZE)),
        tokenizer_contract=str(sidecar.get("tokenizer_contract", "megacpp")),  # type: ignore[arg-type]
        local_profile_vocab_size=int(
            sidecar.get("local_profile_vocab_size", LOCAL_PROFILE_VOCAB_SIZE)
        ),
        megacpp_tokenizer_vocab_size=int(
            sidecar.get("megacpp_tokenizer_vocab_size", MEGACPP_TOKENIZER_VOCAB_SIZE)
        ),
        source_format=source_format,
    )


__all__ = [
    "MegatronIndexedDataset",
    "MegatronGraphRoutePacket",
    "MegatronIndexedMetadata",
    "MegatronIndexedMultiShardDataset",
    "MegatronIndexedMultiShardMetadata",
    "megatron_indexed_graph_sidecar_schema",
    "megatron_indexed_side_channel_schema",
    "open_megatron_indexed_dataset",
]
