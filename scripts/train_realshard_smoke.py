"""Stage-1 DenseCppLM smoke on the REAL per-sample 4k shards (token_ids format).

Unlike train_smoke_dense500m.py (packed golden-mini), this reads the existing
clang_semantic_4k_v10 shards directly: token_ids + the 5 token-aligned structure
side-channels, derives next-token targets, and runs a short AdamW loop. Proves the
dense GQA model trains on real corpus data with the structure side-channels on.
RULE #1: missing column / short row RAISES; no fabrication.
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
import pyarrow.parquet as pq
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig

CLANG = "/Users/dave/sources/parquet/clang_semantic_4k_v10"
STRUCT = {  # parquet col -> model kwarg
    "token_structure_ids": "structure_ids", "token_dep_levels": "dep_levels",
    "token_ast_depth": "ast_depth_ids", "token_sibling_index": "sibling_index_ids",
    "token_ast_node_type": "node_type_ids",
}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    mx.random.seed(a.seed)
    S = a.seq_len

    path = sorted(glob.glob(os.path.join(CLANG, "shard_*.parquet")))[0]
    cols = ["token_ids", *STRUCT.keys()]
    t = pq.read_table(path, columns=cols)
    need = S + 1
    toks, chans = [], {k: [] for k in STRUCT}
    for i in range(t.num_rows):
        tok = t["token_ids"][i].as_py()
        if len(tok) < need:
            continue
        toks.append(np.asarray(tok[:need], dtype=np.int32))
        for c in STRUCT:
            v = t[c][i].as_py()
            if len(v) < S:
                raise ValueError(f"{path}[{i}] {c} len {len(v)} < {S}")
            chans[c].append(np.asarray(v[:S], dtype=np.int32))
        if len(toks) >= a.rows:
            break
    if len(toks) < a.rows:
        raise RuntimeError(f"only {len(toks)} rows with >= {need} tokens in {path}")

    tk = np.stack(toks)  # (B, S+1)
    inp = mx.array(tk[:, :S]); tgt = mx.array(tk[:, 1:S + 1])
    side = {STRUCT[c]: mx.array(np.stack(chans[c])) for c in STRUCT}
    loss_mask = mx.ones((a.rows, S), dtype=mx.int32)
    vocab = 1 << (max(int(tk.max()), 1)).bit_length()
    print(f"real shard: {os.path.basename(path)}  batch={a.rows}x{S}  vocab={vocab}")
    print(f"side channels: {sorted(side.keys())}")

    cfg = DenseCppLMConfig(vocab_size=vocab, hidden_size=a.hidden, depth=a.depth,
                           ffn_hidden_size=a.hidden * 2, max_seq_length=S,
                           num_query_heads=8, num_kv_heads=2,
                           head_dim=max(32, a.hidden // 8),
                           ngram_hash_table_size=8192, ngram_hash_embed_dim=16)
    model = DenseCppLM(cfg)
    print(f"model: d={cfg.hidden_size} depth={cfg.depth} params={model.num_parameters()/1e6:.1f}M")
    opt = optim.AdamW(learning_rate=a.lr, weight_decay=0.01)

    def loss_fn(m):
        _, loss = m(inp, targets=tgt, loss_mask=loss_mask, platform_ids=None, **side)
        return loss
    vag = nn.value_and_grad(model, loss_fn)
    curve = []
    for step in range(a.steps):
        loss, grads = vag(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        curve.append(float(loss.item()))
        if step % max(1, a.steps // 12) == 0 or step == a.steps - 1:
            print(f"  step {step:4d}  loss {curve[-1]:.4f}")
    print(f"loss: initial={curve[0]:.4f} -> final={curve[-1]:.4f} min={min(curve):.4f}")
    if not curve[-1] < curve[0]:
        raise AssertionError(f"final {curve[-1]:.4f} !< initial {curve[0]:.4f}")
    print("REAL-SHARD SMOKE PASS: dense GQA model trains on real data with structure channels")

if __name__ == "__main__":
    main()
