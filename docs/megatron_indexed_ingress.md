# Megatron Indexed Ingress

`MegatronIndexedDataset` is the local MLX ingress for existing Megatron
`.bin/.idx` token shards. It is intentionally standalone: the reader imports
MLX and NumPy only, accepts the stable `MMIDIDX` index layout or explicit raw
`.bin` handoffs, and fails closed instead of importing Megatron, Torch, or CUDA
runtime code.

## Training Opener

Training code can open indexed shards through the generic dataset opener:

```python
from cppmega_mlx.data.token_dataset import open_token_dataset

dataset = open_token_dataset(
    "/path/to/clang_semantic_4k_v10_train",
    format="megatron",
    seq_len=4096,
    batch_size=1,
)
```

or through the standalone reader ingress:

```python
from cppmega_mlx.data.megatron_indexed import open_megatron_indexed_dataset

dataset = open_megatron_indexed_dataset(
    "/path/to/clang_semantic_4k_v10_train",
    seq_len=4096,
    batch_size=1,
)
```

The `path` may be a suffixless prefix, a `.bin`, a `.idx`, or a metadata JSON
sidecar path. Suffixless prefixes are resolved to `<prefix>.bin` plus optional
`<prefix>.idx`; metadata is discovered in this order:

```text
<prefix>.idx.json
<prefix>.json
<prefix>.bin.json
```

Raw `.bin` inputs without `.idx` must provide `dtype` either as an opener
argument or in JSON metadata. This keeps ambiguous byte streams fail-closed.

## Sidecar Schema

Side channels are token-aligned binary files layered beside a token shard. They
are not part of the source `../cppmega/scripts/data_prep_parquet_to_megatron.py`
converter today; that converter writes token-only `.bin/.idx` outputs. For MLX
local training, write an `.idx.json` sidecar next to the token shard:

```json
{
  "vocab_size": 131072,
  "tokenizer_contract": "megacpp",
  "source_format": "megatron-indexed-sidecar",
  "side_channel_paths": {
    "structure_ids": {"path": "structure_ids.bin", "dtype": "int16"},
    "dep_levels": {"path": "dep_levels.bin", "dtype": "uint8"},
    "ast_depth_ids": {"path": "ast_depth_ids.bin", "dtype": "uint8"},
    "sibling_index_ids": {"path": "sibling_index_ids.bin", "dtype": "uint16"},
    "node_type_ids": {"path": "node_type_ids.bin", "dtype": "int32"},
    "attention_mask": {"path": "attention_mask.bin", "dtype": "float32"}
  }
}
```

Top-level path entries are also accepted for small handoffs:

```json
{
  "node_type_ids": {"path": "node_type_ids.bin", "dtype": "uint16"}
}
```

Supported canonical keys:

| Key | Default dtype | MLX batch dtype | Model kwarg |
| --- | --- | --- | --- |
| `attention_mask` | `float32` | `float32` | no, loss mask only |
| `structure_ids` | `int32` | `int32` | yes |
| `dep_levels` | `int32` | `int32` | yes |
| `ast_depth_ids` | `int32` | `int32` | yes |
| `sibling_index_ids` | `int32` | `int32` | yes |
| `node_type_ids` | `int32` | `int32` | yes |

Cppmega Parquet token-level aliases are normalized at the Megatron sidecar
boundary:

| Alias | Canonical key |
| --- | --- |
| `token_attention_mask` | `attention_mask` |
| `token_structure_ids` | `structure_ids` |
| `token_dep_levels` | `dep_levels` |
| `token_ast_depth` | `ast_depth_ids` |
| `token_sibling_index` | `sibling_index_ids` |
| `token_ast_node_type` | `node_type_ids` |

Do not declare both a canonical key and its alias in the same sidecar. The
reader rejects duplicate declarations instead of guessing which file wins.

## Fail-Closed Rules

- Unknown `.idx` headers, unsupported index versions, unknown dtype codes, and
  invalid pointer layouts raise errors.
- Raw `.bin` datasets without dtype metadata raise an error.
- Side-channel file lengths must exactly match the indexed token count.
- Structure side channels must be integer typed, non-negative, and fit `int32`.
- `attention_mask` must be `float32`.
- Ambiguous `side_channels` metadata is rejected.
- Ngram sidecars are rejected because ngram hashes are derived from `input_ids`
  in the model path.

This ingress proves local MLX consumption of Megatron-indexed token shards and
token-aligned sidecars only. It does not claim source converter preservation of
side channels, distributed Megatron parity, or M4-vs-GB10 throughput parity.
