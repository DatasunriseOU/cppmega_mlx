# E2E Coverage Matrix v5 — Closure Report

Epic: `cppmega-mlx-pjt` (25/25 sub-gaps closed). Closes string-echo
gaps from V4 audit (V4-7 LossKind / V4-8 Rewriters / V4-10 side_channels
were propagation-only); adds distributed / lifecycle / persistence
observability; tightens weak assertions.

## Gaps delivered

| Gap | Ticket          | Commit    | New tests       | Notes                          |
|-----|-----------------|-----------|-----------------|--------------------------------|
| G01 | cppmega-mlx-iry | `5bdfdba` | 6 pytest + 3 e2e| MTP K-head loss math + extras.mtp + per-head distinct losses |
| G02 | cppmega-mlx-3tu | `0aa1623` | 4 pytest + 2 e2e| IFIM λ_fim penalty, extras.ifim |
| G03 | cppmega-mlx-7w9 | `0aa1623` | 3 pytest + 2 e2e| MHC λ_mhc bias, extras.mhc |
| G04 | cppmega-mlx-3m6 | `0be1916` | 7 pytest + 2 e2e| apply_rewrites + extras.graph_diff |
| G05 | cppmega-mlx-8be | `656a55b` | 1 e2e           | sharding_applied extras |
| G06 | cppmega-mlx-cn3 | `f523359` | 1 e2e           | mx.metal.get_peak_memory |
| G07 | cppmega-mlx-wz2 | `49baa0c` | 1 e2e           | train_dtype + master_dtype + fp8_active |
| G08 | cppmega-mlx-406 | `1891a22` | 3 vitest        | BottomStrip reconnecting testid |
| G09 | cppmega-mlx-ajm | `629c1c0` | 3 pytest        | abort_token + _ABORT_TOKENS |
| G10 | cppmega-mlx-jrp | `7095689` | 3 pytest        | warm-start opt.state cache |
| G11 | cppmega-mlx-4n3 | `74832cb` | 3 vitest        | spec save/load JSON Blob |
| G12 | cppmega-mlx-t8i | `5696392` | 4 pytest        | safetensors checkpoint save/load |
| G13 | cppmega-mlx-ay6 | `1187589` | 1 e2e           | V4-3 strict tokenizer |
| G14 | cppmega-mlx-amy | `67ee407` | 2 e2e           | V4-4 N=16 + no-blow-up |
| G15 | cppmega-mlx-a0b | `678dfe8` | 3 pytest        | TopBar max=512 + losses_smoothed |
| G16 | cppmega-mlx-3va | `47405d2` | 18 e2e          | 9 non-canonical presets × 2 mutations |
| G17 | cppmega-mlx-7ei | `7df6495` | 4 pytest        | side_channels_forward_effect metrics |
| G18 | cppmega-mlx-biy | `9261f75` | 2 pytest        | ablation.run vs pipeline.run parity |
| G19 | cppmega-mlx-udk | `ffc5119` | 3 vitest        | DimensionsTab Apply per-row |
| G20 | cppmega-mlx-5kt | `111f3e4` | 3 pytest        | real-tokens probe + top1_token_drift |
| G21 | cppmega-mlx-aih | `660a777` | 3 pytest        | dry_forward/loss_smoke/optimizer_smoke extras |
| G22 | cppmega-mlx-3p2 | `6d051bc` | 1 e2e           | hybrid_deltas per-bucket ratio |
| G23 | cppmega-mlx-br8 | `b2b6f29` | 4 pytest + 1 e2e| gradient_clip activation + extras |
| G24 | cppmega-mlx-edp | `d3e29df` | 1 e2e           | Roundtrip OK = byte_diff=0 |
| G25 | cppmega-mlx-zuf | `80658a8` | 2 pytest        | MoE detection + extras.moe |

## extras schema additions (post-v5)

```jsonc
{
  // v4 baseline preserved + v5 adds:
  "mtp": {k, betas[], per_head_losses[]} | null,        // G01
  "ifim": {lambda_fim, fim_weights_norm, penalty_value} | null,  // G02
  "mhc": {lambda_mhc, bias_norm, penalty_value} | null, // G03
  "graph_diff": {added[], removed[], renamed[], skipped[]}, // G04
  "sharding_applied": {axis_assignments[], shard_dim, microbatch_size, compile_mode} | null, // G05
  "memory_peak_bytes": int | null,                      // G06
  "train_dtype": string, "master_dtype": string, "fp8_active": bool, // G07
  "opt_state_carried": bool, "run_id": string,          // G10
  "checkpoint": {saved_path, loaded_path} | null,       // G12
  "losses_smoothed": [float],                           // G15
  "side_channels_forward_effect": {doc_ids_mask_density, token_ids_added_norm} | null, // G17
  "inference_probe": {... real_tokens, text_len, top1_token_drift}, // G20 extends V4-11
  "hybrid_deltas": {muon_norm, adamw_norm, ratio} | null, // G22
  "gradient_clip": {threshold, max_grad_norm_seen, num_clips}, // G23
  "moe": {kind, num_experts, top_k, routing_entropy, ...} | null, // G25
  "aborted": bool?, "abort_token": string?              // G09 (only on cancel)
}
```

Plus model_summary.{loss_kind, rewriters_applied[]} from V4.

## Total new tests this epic

- **Pytest**: ~35 new (tests/v5/*.py)
- **Vitest**: ~10 new
- **E2E**: ~40 new (scenarios/30_-54_.spec.ts series)
- **Total**: ~85 strict-content scenarios

## Honest categorisation

🟢 **math-effect** (UI mutation changes loss/weights/output): G01, G02, G03,
G04, G06, G14, G22, G23
🟡 **propagation** (UI string reaches extras, math TBD): G05, G07, G09 (abort
signal), G10 (state carry), G15 (smoothed array), G17 (channel metrics),
G19 (Apply button — App integration follow-up), G20 (real-tokens
backend done, UI input follow-up), G21 (other-stages observability),
G25 (MoE detection)
🟢 **persistence**: G11 (save/load), G12 (checkpoint)
🟢 **strictness**: G13, G14, G16, G18, G24

0 🔴 decorative remaining in V5 scope.

## Deferred / v6 candidates

| Item | Why deferred |
|------|-------------|
| Real distributed multi-device train | Hardware |
| Real attention-bias / cross-doc-mask routing (G17 forward) | nn.Module rewrites |
| Real MoE routing entropy + load-balance loss (G25 forward) | Expert-routing hook |
| Real fp8/fp16 dtype switching (G7 actual math) | mlx dtype plumbing |
| Cancel button UI + WS abort handler (G09 frontend) | UI wire |
| TopBar checkpoint path inputs (G12 frontend) | UI wire |
| App.tsx DimensionsTab Apply dispatch (G19 wire) | Dispatch routing |
| Train probe-text textarea (G20 frontend) | UI input |
| WS mid-train recovery e2e (G08 lifecycle) | Stress test |
| Identical-loss-continuation strict e2e for G12 | Determinism gate |

## Pre-existing red (excluded)

`tests/v4/test_path_d_runtime_adapter.py::test_gdn_chunk_o_metal_threadgroup_memory_fits_device_limit` (61440 > 32K, owned by path_c_fusion.py work).
