"""Fast Stage-1 DenseCppLM training smoke.

Loads packed rows from the golden-mini code fixtures (and optionally a small
slice of the real ``clang_semantic_4k_v10`` corpus) as :class:`CodePacket`
objects, builds a tiny :class:`DenseCppLM` (smoke dims, same code path as the
real ~500M profile), and runs a short AdamW loop on masked next-token
cross-entropy.

It prints the loss curve and ASSERTS the final loss is below the initial loss,
i.e. the model — including the structure side channels — is learning. Designed
to run fast on CPU/MLX.

RULE #1 (fail fast / fail loud): every missing column / shape mismatch RAISES.
There is no synthetic-data fallback; if the fixtures are absent the script
crashes with WHERE + WHAT.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig

# Fixture / corpus locations (absolute, per repo layout).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_MINI_CODE = os.path.join(REPO_ROOT, "tests", "fixtures", "golden_mini", "code")
CLANG_SEMANTIC_4K = "/Users/dave/sources/parquet/clang_semantic_4k_v10"

# Parquet column -> CodePacket field for the packed code schema.
_TOKEN_COL = "input_ids"
_TARGET_COL = "target_ids"
_MASK_COL = "loss_mask"
_PLATFORM_COL = "platform_ids"
_STRUCTURE_COLS = {
    "token_structure_ids": "structure_ids",
    "token_dep_levels": "dep_levels",
    "token_ast_depth": "ast_depth",
    "token_sibling_index": "sibling_index",
    "token_ast_node_type": "ast_node_type",
}


def _load_packets(
    parquet_paths: Sequence[str],
    *,
    seq_len: int,
    max_rows: int,
) -> list[CodePacket]:
    """Read packed rows into CodePackets, truncating each window to ``seq_len``."""

    import numpy as np
    import pyarrow.parquet as pq

    packets: list[CodePacket] = []
    for path in parquet_paths:
        table = pq.read_table(path)
        cols = table.column_names
        if _TOKEN_COL not in cols:
            raise ValueError(f"{path}: missing required column {_TOKEN_COL!r}")
        n = table.num_rows
        for row in range(n):
            if len(packets) >= max_rows:
                return packets

            def vec(name: str) -> mx.array | None:
                if name not in cols:
                    return None
                raw = table[name][row].as_py()
                arr = np.asarray(raw, dtype=np.int32)[:seq_len]
                if arr.shape[0] != seq_len:
                    raise ValueError(
                        f"{path}[row={row}] column {name!r} has length "
                        f"{arr.shape[0]} < seq_len {seq_len}"
                    )
                return mx.array(arr)

            token_ids = vec(_TOKEN_COL)
            target_ids = vec(_TARGET_COL)
            loss_mask = vec(_MASK_COL)
            if target_ids is None:
                raise ValueError(f"{path}[row={row}] missing {_TARGET_COL!r}")

            structure = {field: vec(col) for col, field in _STRUCTURE_COLS.items()}

            # Per-document platform ids are a short (K,) list in this schema.
            # Stored per-packet as a 1-D (K,) vector; _stack_batch stacks them
            # into the (B, K) tensor the platform embedding consumes.
            platform_ids = None
            if _PLATFORM_COL in cols:
                praw = table[_PLATFORM_COL][row].as_py()
                if praw:
                    platform_ids = mx.array(np.asarray(praw, dtype=np.int32))  # (K,)

            packets.append(
                CodePacket(
                    token_ids=token_ids,
                    target_ids=target_ids,
                    loss_mask=loss_mask,
                    structure_ids=structure["structure_ids"],
                    dep_levels=structure["dep_levels"],
                    ast_depth=structure["ast_depth"],
                    sibling_index=structure["sibling_index"],
                    ast_node_type=structure["ast_node_type"],
                    metadata={"platform_vec": platform_ids, "source": path},
                )
            )
    return packets


def _stack_batch(packets: Sequence[CodePacket]) -> dict[str, mx.array | None]:
    """Stack single-window packets into a (B, S) batch dict for the model."""

    def stack(attr: str) -> mx.array | None:
        values = [getattr(p, attr) for p in packets]
        if any(v is None for v in values):
            return None
        return mx.stack([v for v in values], axis=0)

    platform_vecs = [p.metadata.get("platform_vec") for p in packets]
    platform = (
        mx.stack(platform_vecs, axis=0)  # (B, K)
        if all(v is not None for v in platform_vecs)
        else None
    )
    return {
        "input_ids": stack("token_ids"),
        "targets": stack("target_ids"),
        "loss_mask": stack("loss_mask"),
        "structure_ids": stack("structure_ids"),
        "dep_levels": stack("dep_levels"),
        "ast_depth_ids": stack("ast_depth"),
        "sibling_index_ids": stack("sibling_index"),
        "node_type_ids": stack("ast_node_type"),
        "platform_ids": platform,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-clang-slice",
        action="store_true",
        help="Also pull a few rows from clang_semantic_4k_v10 if present.",
    )
    args = parser.parse_args()

    mx.random.seed(args.seed)

    paths = sorted(glob.glob(os.path.join(GOLDEN_MINI_CODE, "*.parquet")))
    if not paths:
        raise FileNotFoundError(
            f"no golden-mini code parquet found under {GOLDEN_MINI_CODE}"
        )
    if args.use_clang_slice and os.path.isdir(CLANG_SEMANTIC_4K):
        paths += sorted(glob.glob(os.path.join(CLANG_SEMANTIC_4K, "*.parquet")))[:1]

    packets = _load_packets(paths, seq_len=args.seq_len, max_rows=args.max_rows)
    if not packets:
        raise RuntimeError("loaded zero packets; cannot run smoke")
    print(f"loaded {len(packets)} packets from {len(paths)} parquet file(s)")
    batch = _stack_batch(packets)
    side_present = [
        k for k in ("structure_ids", "dep_levels", "ast_depth_ids",
                    "sibling_index_ids", "node_type_ids", "platform_ids")
        if batch[k] is not None
    ]
    print(f"side channels present: {side_present}")

    vocab = int(mx.max(batch["input_ids"]).item()) + 1
    vocab = max(vocab, int(mx.max(batch["targets"]).item()) + 1)
    cfg = DenseCppLMConfig(
        vocab_size=1 << (max(vocab - 1, 1)).bit_length(),  # next pow2 >= vocab
        hidden_size=args.hidden,
        depth=args.depth,
        ffn_hidden_size=args.ffn,
        max_seq_length=args.seq_len,
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=max(32, args.hidden // 8),
        ngram_hash_table_size=8192,
        ngram_hash_embed_dim=16,
    )
    model = DenseCppLM(cfg)
    print(
        f"smoke model: vocab={cfg.vocab_size} d={cfg.hidden_size} depth={cfg.depth} "
        f"ffn={cfg.ffn_hidden_size} params={model.num_parameters()/1e6:.2f}M "
        f"GQA q={cfg.num_query_heads}/kv={cfg.num_kv_heads}"
    )

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(m: DenseCppLM) -> mx.array:
        _, loss = m(
            batch["input_ids"],
            targets=batch["targets"],
            loss_mask=batch["loss_mask"],
            structure_ids=batch["structure_ids"],
            dep_levels=batch["dep_levels"],
            ast_depth_ids=batch["ast_depth_ids"],
            sibling_index_ids=batch["sibling_index_ids"],
            node_type_ids=batch["node_type_ids"],
            platform_ids=batch["platform_ids"],
        )
        return loss

    value_and_grad = nn.value_and_grad(model, loss_fn)

    curve: list[float] = []
    for step in range(args.steps):
        loss, grads = value_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        curve.append(float(loss.item()))
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss {curve[-1]:.4f}")

    initial, final = curve[0], curve[-1]
    print(f"loss curve: initial={initial:.4f} -> final={final:.4f} "
          f"(min={min(curve):.4f})")
    if not final < initial:
        raise AssertionError(
            f"final loss {final:.4f} not below initial {initial:.4f}; "
            "structure did not learn"
        )
    print("SMOKE PASS: structure learns (final loss < initial loss)")


if __name__ == "__main__":
    main()
