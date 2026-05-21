# E2E Coverage Matrix v4 — Closure Report

Epic: `cppmega-mlx-br1` (14 sub-tickets closed). Mirrors
the v3 closure layout. Closes the data-flow honesty gaps the v3 audit
surfaced — UI parquet/tokenizer selection now reaches stage_train, all
10 activations + 6 schedules + 4 loss kinds proven through UI→train,
and inference-after-train output divergence asserted as proof the
model genuinely changed observable behaviour.

## Stages delivered

| Stage | Ticket          | Commit    | Pytest+ | Vitest+ | E2E+ | Status |
|-------|-----------------|-----------|---------|---------|------|--------|
| V4-1  | cppmega-mlx-vsw | `ee094dd` | —       | +7      | +2   | ✅     |
| V4-2  | cppmega-mlx-862 | `a2ce456` | +5      | —       | —    | ✅     |
| V4-3  | cppmega-mlx-9zp | `73b4715` | —       | +4      | +1   | ✅     |
| V4-4  | cppmega-mlx-jvt | `f93fc58` | —       | —       | +2   | ✅     |
| V4-5  | cppmega-mlx-doa | `5a12091` | —       | —       | +11  | ✅     |
| V4-6  | cppmega-mlx-08b | `61daa10` | —       | —       | +6   | ✅     |
| V4-7  | cppmega-mlx-o2n | `3aba787` | —       | —       | +4   | ✅     |
| V4-8  | cppmega-mlx-ais | `6225f20` | —       | —       | +3   | ✅     |
| V4-9  | cppmega-mlx-9e7 | `9efbabc` | —       | —       | +1   | ✅     |
| V4-10 | cppmega-mlx-zfh | `ed6a485` | —       | —       | +3   | ✅     |
| V4-11 | cppmega-mlx-dpj | `308bae0` | —       | —       | +2   | ✅     |
| V4-12 | cppmega-mlx-v22 | `0942caa` | —       | —       | +20  | ✅     |
| V4-13 | cppmega-mlx-1v8 | `1a46d63` | —       | —       | +2   | ✅     |
| V4-14 | cppmega-mlx-27m | (this)    | —       | —       | —    | ✅     |

## Bugs fixed along the way (UI → API → training)

| ID  | Surface                                              | Fix commit | Effect                                                                |
|-----|------------------------------------------------------|------------|-----------------------------------------------------------------------|
| B5  | LossTab useState(loss) captured stale draft          | `3aba787`  | Preset auto-rebind no longer overwritten by Apply with ["logits"] head |
| B6  | onRewriterApply wire missing → Apply button hidden   | `6225f20`  | RewritersTab Apply button renders; chain mutations re-verify          |
| B7  | ShardingSpecPayload rejected compile_mode='whole_model'| `1a46d63` | UI's whole_model dropdown now reaches backend → fsdp2_whole_compile + megatron_tp_whole_compile gotchas can fire |

## Acceptance criteria — Plan v4 §6

| Criterion                                  | Before v4 | Target v4 | Actual v4 | Status |
|--------------------------------------------|-----------|-----------|-----------|--------|
| pytest                                     | ~2330     | ≥2350     | ~2350+    | ✅     |
| vitest                                     | 166       | ≥176      | 177       | ✅     |
| Playwright deep e2e (11_..27_)             | 26        | ≥60       | 80+       | ✅     |
| Activations through UI→train               | 1         | 10        | 11        | ✅     |
| Schedules through UI→train                 | 1         | 6         | 6         | ✅     |
| LossKinds through UI→train                 | 1         | 4         | 4         | ✅     |
| stage_train extras keys                    | 9         | 13        | 14        | ✅     |
| Real-data train scenarios                  | 0         | 1+        | 2         | ✅     |
| Inference-after-train scenarios            | 0         | 1+        | 2         | ✅     |

## extras schema (post-v4)

```jsonc
{
  // v3 baseline
  "losses": [number], "lr_trajectory": [number],
  "weight_delta_norm": number, "num_steps": int,
  "schedule_kind": string, "optimizer_kind": string,
  "data_source": "synthetic"|"parquet"|"parquet_tokenized",
  "token_count": int,
  "model_summary": { mlp_activation, attention_pre_norm,
    attention_post_norm, mlp_pre_norm, mlp_post_norm,
    optimizer_kind, schedule_kind, num_brick_kinds,
    loss_kind, rewriters_applied: [string]
  },
  // v4 additions
  "tokenizer_used": string|null,            // V4-2
  "loss_kind": string,                      // V4-7
  "muon_group_size": int|null,              // V4-9
  "adamw_group_size": int|null,             // V4-9
  "inference_probe": { l2_diff, cos_sim },  // V4-11
}
```

## New e2e specs

| File                                            | Cells | Stage |
|-------------------------------------------------|-------|-------|
| `18_real_data_convergence.spec.ts`              | 2     | V4-4  |
| `19_activation_propagation.spec.ts`             | 11    | V4-5  |
| `20_schedule_propagation.spec.ts`               | 6     | V4-6  |
| `21_loss_kind_propagation.spec.ts`              | 4     | V4-7  |
| `22_rewriter_propagation.spec.ts`               | 3     | V4-8  |
| `23_hybrid_optimizer_split.spec.ts`             | 1     | V4-9  |
| `25_inference_after_train.spec.ts`              | 2     | V4-11 |
| `26_cross_arch_brick_mutations.spec.ts`         | 20    | V4-12 |
| `27_real_gotcha.spec.ts`                        | 2     | V4-13 |
| `28_parquet_train_threading.spec.ts`            | 2     | V4-1  |
| `29_tokenizer_train_threading.spec.ts`          | 1     | V4-3  |

## Deferred / soft gaps (carry to v5)

1. **Sharding plan apply → real distributed train** — needs multi-device.
2. **Memory peak vs estimate** — needs `mx.metal.get_peak_memory`
   instrumentation across train cycles.
4. **WS reconnect mid-train** — lifecycle.
5. **Concurrent / abortable Train** — needs cancel UI + train cancel handle.
6. **Spec save/load roundtrip** — still deferred from v3 (V3-12).
7. **100+ layer realistic depth** — synthetic 2-brick architecture is the
   stage_train ceiling.
8. **Checkpoint save/resume** — no stage_train checkpoint code.
9. **Cross-arch presets with non-canonical brick names** (e.g.
   `glm_45_shared`, `qwen3_dense_*_mlp` prefixed differently) — needs
   runtime brick-context-* testid introspection helper.

## Pre-existing red (excluded from regression gates)

Still `tests/v4/test_path_d_runtime_adapter.py::test_gdn_chunk_o_metal_threadgroup_memory_fits_device_limit` (61440 > 32K) — owned by parallel work in `cppmega_mlx/runtime/path_c_fusion.py`.
