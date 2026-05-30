# Vendored mlx-lm Provenance — cppmega_v4 Path E + mHC/MTP

**Compiled:** May 2026
**Source of truth:** `cppmega_v4/nn/_external/VENDORED_MANIFEST.json`
**Drift check:** `cppmega_v4/nn/_external/_vendored_provenance.py`
(also enforced by `scripts/path_sanity_guard.py` and
`tests/v4/test_vendored_provenance.py`)
**Upstream repo:** https://github.com/ml-explore/mlx-lm — License: MIT (© Apple Inc.)

---

## Why this exists

The `*_vendored.py` files under `cppmega_v4/nn/_external/` are point-in-time
copies (or close derivatives) of upstream mlx-lm code. Nothing in Python keeps
a "verbatim" snapshot verbatim, so a snapshot could silently diverge from the
PR it claims to mirror (the **DRIFT** risk). The manifest pins each file to its
upstream PR and a `sha256`; the drift check recomputes the hash and fails if a
file changed without the manifest being updated.

Run the check directly:

```bash
python -m cppmega_v4.nn._external._vendored_provenance
# or as part of the v4 guardrails:
python scripts/path_sanity_guard.py --contracts-only
```

If you intentionally re-vendor a file, recompute its hash
(`shasum -a 256 <file>`) and update both the `sha256` and provenance fields in
`VENDORED_MANIFEST.json`.

---

## Pinned snapshots

| File | PR | Upstream path | Kind | Used by |
|---|---|---|---|---|
| `_mlx_lm_gated_delta_vendored.py` | [#1217](https://github.com/ml-explore/mlx-lm/pull/1217) | `mlx_lm/models/gated_delta.py` | verbatim | **GDN/KDA Path E** kernel + ops |
| `_mlx_lm_fp8_dequant_vendored.py` | [#1224](https://github.com/ml-explore/mlx-lm/pull/1224) | `mlx_lm/models/qwen3_5_moe.py` (`Model.sanitize` FP8 block) | derived | Lightning Indexer FP8 (pure-MLX, **not** Path E) + MoE loaders |
| `_mlx_lm_gated_delta_vjp_metal_vendored.py` | #1217 (kernel reused) | `mlx_lm/models/gated_delta.py` | derived (cppmega-authored VJP) | **GDN Path E training backward** (fused Metal VJP) |
| `_mlx_lm_gated_delta_vjp_vendored.py` | #1217 (ops reused) | `mlx_lm/models/gated_delta.py` | derived (cppmega-authored VJP) | GDN Path E backward (Python checkpointed ref) |
| `_mlx_lm_hyper_connection_vendored.py` | [#1189](https://github.com/ml-explore/mlx-lm/pull/1189) | `mlx_lm/models/hyper_connection.py` | verbatim | V4 mHC residual |
| `_mlx_lm_sinkhorn_vendored.py` | #1189 | `mlx_lm/models/sinkhorn.py` | verbatim | V4 mHC Sinkhorn projection |
| `_mlx_lm_mtp_module_vendored.py` | [#990](https://github.com/ml-explore/mlx-lm/pull/990) | `mlx_lm/models/qwen3_5.py` (MTPModule excerpt) | derived | V4 MTP speculative-decode module |

The exact `sha256` for each file lives in `VENDORED_MANIFEST.json` (the
machine-checked source of truth); this table is the human-readable mirror.

---

## Benchmark: GDN backward — Metal E-VJP vs Path B fused bwd

Measured on this Mac (Metal, MLX 0.32.0.dev) via `mx.grad(sum(y))`, float32,
gate `g<=0` (Path E eligible), 8 timed iters after 3 warmups:

| Shape (B,T,H,K,V) | Path B fused bwd | **E-VJP (Metal)** | Faster |
|---|---|---|---|
| 1, 512, 4, 32, 32 | 5.01 ms | **1.21 ms** | E-VJP 4.1x |
| 1, 1024, 4, 32, 32 | 10.37 ms | **2.08 ms** | E-VJP 5.0x |
| 2, 2048, 2, 32, 32 | 21.41 ms | **3.77 ms** | E-VJP 5.7x |

**Recommendation:** when the Path E forward is eligible (gate `g<=0`,
`Dk%32==0 & Dv%4==0`), the fused Metal E-VJP
(`_mlx_lm_gated_delta_vjp_metal_vendored.gated_delta_update_vjp_metal`) should
be the GDN training backward default — it is 4-5.7x faster than Path B's fused
backward and the gap widens with sequence length. Path B remains the fallback
for amplifying gates / ineligible shapes (where E fails closed) and for
`max(K,V) > 256` (where E-VJP and Path B's bwd both have limits).

### Notes on the Path E snapshot (PR #1217)

The vendored `_mlx_lm_gated_delta_vendored.py` snapshot already incorporates
the PR #1066 Kahan-compensated `kv_mem` accumulation + 4-way time-loop unroll.
It therefore does **not** match an older mlx-lm checkout that predates PR #1066
— the manifest pins the *snapshot we ship*, not whatever happens to be
`pip install`-ed locally.
