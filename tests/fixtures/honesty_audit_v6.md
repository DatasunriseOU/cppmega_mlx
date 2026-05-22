# V6 honesty audit

Classifier from `scripts/audit_v3_v4_v5.py` (heuristic).

| Category | Count |
|---|---|
| 🟢 math-effect | 294 |
| 🟡 propagation | 35 |
| 🔴 decorative | 1119 |

Total: 1448

## By file

### tests/v4/test_ablation_run.py
🟢=1 🟡=0 🔴=7 / total 8

### tests/v4/test_architectures_list_presets.py
🟢=0 🟡=0 🔴=4 / total 4

### tests/v4/test_benchmark_matrix.py
🟢=2 🟡=0 🔴=9 / total 11

### tests/v4/test_benchmark_receipt.py
🟢=0 🟡=0 🔴=5 / total 5

### tests/v4/test_block_dispatch_env_override.py
🟢=5 🟡=0 🔴=2 / total 7

### tests/v4/test_catalog.py
🟢=1 🟡=0 🔴=20 / total 21

### tests/v4/test_csa_hca_indexer_adapter.py
🟢=2 🟡=0 🔴=5 / total 7

### tests/v4/test_csa_hca_streaming_decode.py
🟢=1 🟡=0 🔴=5 / total 6

### tests/v4/test_csa_hca_v4.py
🟢=2 🟡=0 🔴=8 / total 10

### tests/v4/test_data_inspector.py
🟢=1 🟡=0 🔴=10 / total 11

### tests/v4/test_data_roundtrip.py
🟢=1 🟡=0 🔴=7 / total 8

### tests/v4/test_e2e_fixture_matrix.py
🟢=2 🟡=0 🔴=13 / total 15

### tests/v4/test_e2e_matrix_report.py
🟢=0 🟡=0 🔴=11 / total 11

### tests/v4/test_engram_v4.py
🟢=0 🟡=0 🔴=7 / total 7

### tests/v4/test_extended_activations.py
🟢=0 🟡=0 🔴=9 / total 9

### tests/v4/test_fp8_dequant.py
🟢=1 🟡=0 🔴=4 / total 5

### tests/v4/test_fused_fp8_gemm.py
🟢=1 🟡=0 🔴=4 / total 5

### tests/v4/test_fusion_roadmap_gaps.py
🟢=0 🟡=0 🔴=6 / total 6

### tests/v4/test_fusion_stage_a.py
🟢=1 🟡=0 🔴=23 / total 24

### tests/v4/test_fusion_stage_b.py
🟢=0 🟡=0 🔴=10 / total 10

### tests/v4/test_fusion_stage_c.py
🟢=1 🟡=0 🔴=22 / total 23

### tests/v4/test_fusion_stage_d.py
🟢=0 🟡=0 🔴=22 / total 22

### tests/v4/test_fusion_stage_e.py
🟢=0 🟡=0 🔴=19 / total 19

### tests/v4/test_galcov_stage_a.py
🟢=0 🟡=0 🔴=12 / total 12

### tests/v4/test_galcov_stage_b.py
🟢=0 🟡=0 🔴=9 / total 9

### tests/v4/test_galcov_stage_c.py
🟢=0 🟡=0 🔴=10 / total 10

### tests/v4/test_galcov_stage_d.py
🟢=0 🟡=0 🔴=8 / total 8

### tests/v4/test_gated_attention.py
🟢=1 🟡=0 🔴=7 / total 8

### tests/v4/test_inference_log.py
🟢=1 🟡=0 🔴=9 / total 10

### tests/v4/test_jsonrpc_cache.py
🟢=0 🟡=0 🔴=15 / total 15

### tests/v4/test_jsonrpc_dispatcher.py
🟢=2 🟡=0 🔴=9 / total 11

### tests/v4/test_jsonrpc_methods.py
🟢=2 🟡=0 🔴=12 / total 14

### tests/v4/test_jsonrpc_review_fixes.py
🟢=1 🟡=0 🔴=12 / total 13

### tests/v4/test_jsonrpc_schema.py
🟢=4 🟡=0 🔴=24 / total 28

### tests/v4/test_jsonrpc_server.py
🟢=3 🟡=0 🔴=6 / total 9

### tests/v4/test_jupyterlite_scaffold.py
🟢=0 🟡=0 🔴=5 / total 5

### tests/v4/test_kda_path_b_bwd.py
🟢=4 🟡=0 🔴=1 / total 5

### tests/v4/test_kda_paths.py
🟢=4 🟡=0 🔴=21 / total 25

### tests/v4/test_kimi_delta_attention_path_a.py
🟢=1 🟡=0 🔴=13 / total 14

### tests/v4/test_lightning_indexer.py
🟢=0 🟡=0 🔴=7 / total 7

### tests/v4/test_lightning_indexer_fp8.py
🟢=0 🟡=0 🔴=6 / total 6

### tests/v4/test_linear_attention_path_a.py
🟢=1 🟡=0 🔴=13 / total 14

### tests/v4/test_linear_attention_path_b.py
🟢=0 🟡=0 🔴=7 / total 7

### tests/v4/test_linear_attention_path_c.py
🟢=1 🟡=0 🔴=5 / total 6

### tests/v4/test_linear_attention_path_d.py
🟢=1 🟡=0 🔴=7 / total 8

### tests/v4/test_linear_attention_path_e.py
🟢=2 🟡=0 🔴=3 / total 5

### tests/v4/test_mbspec_stage_a.py
🟢=17 🟡=0 🔴=25 / total 42

### tests/v4/test_mbspec_stage_b.py
🟢=22 🟡=0 🔴=0 / total 22

### tests/v4/test_mbspec_stage_c.py
🟢=20 🟡=0 🔴=5 / total 25

### tests/v4/test_mbspec_stage_d.py
🟢=17 🟡=0 🔴=8 / total 25

### tests/v4/test_mbspec_stage_e.py
🟢=13 🟡=0 🔴=9 / total 22

### tests/v4/test_memory_parity.py
🟢=1 🟡=1 🔴=0 / total 2

### tests/v4/test_mhc_metal_v4.py
🟢=0 🟡=0 🔴=6 / total 6

### tests/v4/test_mhc_v4.py
🟢=0 🟡=0 🔴=12 / total 12

### tests/v4/test_mla_absorb.py
🟢=1 🟡=0 🔴=4 / total 5

### tests/v4/test_mla_block.py
🟢=2 🟡=0 🔴=5 / total 7

### tests/v4/test_mlp_activation_switch.py
🟢=0 🟡=0 🔴=5 / total 5

### tests/v4/test_mlx_lm_bricks.py
🟢=2 🟡=0 🔴=14 / total 16

### tests/v4/test_moe_fp8.py
🟢=3 🟡=0 🔴=5 / total 8

### tests/v4/test_moe_v4.py
🟢=5 🟡=0 🔴=13 / total 18

### tests/v4/test_mtp_speculative_adapter.py
🟢=2 🟡=0 🔴=4 / total 6

### tests/v4/test_mtp_v4.py
🟢=5 🟡=0 🔴=11 / total 16

### tests/v4/test_norm_builder_params.py
🟢=1 🟡=0 🔴=8 / total 9

### tests/v4/test_norm_validation.py
🟢=0 🟡=0 🔴=10 / total 10

### tests/v4/test_nsa_v4.py
🟢=4 🟡=0 🔴=7 / total 11

### tests/v4/test_optim_spec_lion.py
🟢=1 🟡=0 🔴=23 / total 24

### tests/v4/test_path_b_bwd.py
🟢=7 🟡=0 🔴=1 / total 8

### tests/v4/test_path_d_runtime_adapter.py
🟢=0 🟡=0 🔴=19 / total 19

### tests/v4/test_path_dispatch.py
🟢=2 🟡=0 🔴=16 / total 18

### tests/v4/test_path_e_training.py
🟢=1 🟡=0 🔴=2 / total 3

### tests/v4/test_probe_stage_a.py
🟢=1 🟡=0 🔴=16 / total 17

### tests/v4/test_probe_stage_b.py
🟢=8 🟡=0 🔴=10 / total 18

### tests/v4/test_probe_stage_c.py
🟢=2 🟡=0 🔴=10 / total 12

### tests/v4/test_pspec_stage_a.py
🟢=0 🟡=0 🔴=50 / total 50

### tests/v4/test_pspec_stage_b.py
🟢=5 🟡=0 🔴=20 / total 25

### tests/v4/test_pspec_stage_c.py
🟢=1 🟡=0 🔴=24 / total 25

### tests/v4/test_pspec_stage_d.py
🟢=1 🟡=0 🔴=17 / total 18

### tests/v4/test_pspec_stage_e.py
🟢=2 🟡=0 🔴=11 / total 13

### tests/v4/test_run_template.py
🟢=0 🟡=0 🔴=21 / total 21

### tests/v4/test_runner_cli.py
🟢=0 🟡=0 🔴=12 / total 12

### tests/v4/test_runner_pipeline.py
🟢=0 🟡=10 🔴=12 / total 22

### tests/v4/test_schedules.py
🟢=8 🟡=0 🔴=20 / total 28

### tests/v4/test_side_channel_spec.py
🟢=0 🟡=0 🔴=5 / total 5

### tests/v4/test_stage_train_data.py
🟢=0 🟡=0 🔴=5 / total 5

### tests/v4/test_stage_train_dtype_real.py
🟢=4 🟡=0 🔴=0 / total 4

### tests/v4/test_stage_train_fake_ranks.py
🟢=2 🟡=0 🔴=2 / total 4

### tests/v4/test_stage_train_fp16.py
🟢=3 🟡=0 🔴=0 / total 3

### tests/v4/test_stage_train_moe_real.py
🟢=3 🟡=0 🔴=0 / total 3

### tests/v4/test_stage_train_optimizers.py
🟢=3 🟡=0 🔴=6 / total 9

### tests/v4/test_stage_train_resume_drift.py
🟢=2 🟡=0 🔴=0 / total 2

### tests/v4/test_stage_train_sharding_real.py
🟢=3 🟡=0 🔴=0 / total 3

### tests/v4/test_stage_train_strict_continuation.py
🟢=2 🟡=1 🔴=0 / total 3

### tests/v4/test_stage_train_tokenize.py
🟢=1 🟡=0 🔴=4 / total 5

### tests/v4/test_streaming_kv_cache.py
🟢=0 🟡=0 🔴=7 / total 7

### tests/v4/test_suggest_optim_groups.py
🟢=0 🟡=0 🔴=9 / total 9

### tests/v4/test_tokenizer_playground.py
🟢=0 🟡=0 🔴=11 / total 11

### tests/v4/test_unified_superblock_real_factories.py
🟢=7 🟡=0 🔴=3 / total 10

### tests/v4/test_unified_superblock_v4.py
🟢=3 🟡=0 🔴=7 / total 10

### tests/v4/test_vbspec_stage_a.py
🟢=0 🟡=0 🔴=24 / total 24

### tests/v4/test_vbspec_stage_b.py
🟢=0 🟡=0 🔴=13 / total 13

### tests/v4/test_vbspec_stage_c.py
🟢=0 🟡=0 🔴=21 / total 21

### tests/v4/test_vbspec_stage_d.py
🟢=0 🟡=0 🔴=18 / total 18

### tests/v4/test_vbspec_stage_e.py
🟢=0 🟡=0 🔴=16 / total 16

### tests/v4/test_widget.py
🟢=0 🟡=0 🔴=8 / total 8

### tests/v5/test_ablation_parity.py
🟢=1 🟡=0 🔴=1 / total 2

### tests/v5/test_stage_extras_other.py
🟢=1 🟡=0 🔴=2 / total 3

### tests/v5/test_stage_train_cancel.py
🟢=0 🟡=0 🔴=3 / total 3

### tests/v5/test_stage_train_checkpoint.py
🟢=0 🟡=4 🔴=0 / total 4

### tests/v5/test_stage_train_clip.py
🟢=0 🟡=0 🔴=4 / total 4

### tests/v5/test_stage_train_ifim_mhc.py
🟢=3 🟡=0 🔴=4 / total 7

### tests/v5/test_stage_train_long_run.py
🟢=3 🟡=0 🔴=0 / total 3

### tests/v5/test_stage_train_moe.py
🟢=0 🟡=0 🔴=2 / total 2

### tests/v5/test_stage_train_mtp.py
🟢=2 🟡=0 🔴=4 / total 6

### tests/v5/test_stage_train_real_probe.py
🟢=0 🟡=0 🔴=3 / total 3

### tests/v5/test_stage_train_rewriters.py
🟢=1 🟡=0 🔴=6 / total 7

### tests/v5/test_stage_train_side_channels.py
🟢=4 🟡=0 🔴=5 / total 9

### tests/v5/test_stage_train_warm_start.py
🟢=0 🟡=0 🔴=3 / total 3

### vbgui/e2e/scenarios/01_canvas_smoke.spec.ts
🟢=0 🟡=2 🔴=0 / total 2

### vbgui/e2e/scenarios/04_tokenizer_playground.spec.ts
🟢=0 🟡=0 🔴=2 / total 2

### vbgui/e2e/scenarios/05_data_inspector.spec.ts
🟢=3 🟡=0 🔴=0 / total 3

### vbgui/e2e/scenarios/06_sharding_proposals.spec.ts
🟢=0 🟡=0 🔴=3 / total 3

### vbgui/e2e/scenarios/07_gotchas.spec.ts
🟢=0 🟡=0 🔴=2 / total 2

### vbgui/e2e/scenarios/08_manual_drag_drop.spec.ts
🟢=0 🟡=0 🔴=3 / total 3

### vbgui/e2e/scenarios/09_e2e_manual.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/10_new_ui.spec.ts
🟢=0 🟡=0 🔴=12 / total 12

### vbgui/e2e/scenarios/11_ui_to_train.spec.ts
🟢=5 🟡=0 🔴=0 / total 5

### vbgui/e2e/scenarios/12_train_convergence.spec.ts
🟢=2 🟡=0 🔴=0 / total 2

### vbgui/e2e/scenarios/13_ablation_math.spec.ts
🟢=2 🟡=0 🔴=0 / total 2

### vbgui/e2e/scenarios/14_cross_arch_deep.spec.ts
🟢=1 🟡=1 🔴=0 / total 2

### vbgui/e2e/scenarios/15_gating.spec.ts
🟢=1 🟡=1 🔴=1 / total 3

### vbgui/e2e/scenarios/17_roundtrip_warning.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/18_real_data_convergence.spec.ts
🟢=1 🟡=0 🔴=0 / total 1

### vbgui/e2e/scenarios/23_hybrid_optimizer_split.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/24_side_channels.spec.ts
🟢=0 🟡=1 🔴=2 / total 3

### vbgui/e2e/scenarios/25_inference_after_train.spec.ts
🟢=2 🟡=0 🔴=0 / total 2

### vbgui/e2e/scenarios/26_cross_arch_brick_mutations.spec.ts
🟢=2 🟡=0 🔴=0 / total 2

### vbgui/e2e/scenarios/27_real_gotcha.spec.ts
🟢=1 🟡=0 🔴=1 / total 2

### vbgui/e2e/scenarios/28_parquet_train_threading.spec.ts
🟢=1 🟡=1 🔴=0 / total 2

### vbgui/e2e/scenarios/29_tokenizer_train_threading.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/30_mtp_math.spec.ts
🟢=3 🟡=0 🔴=0 / total 3

### vbgui/e2e/scenarios/31_ifim_mhc_math.spec.ts
🟢=4 🟡=0 🔴=0 / total 4

### vbgui/e2e/scenarios/33_rewriter_graph_diff.spec.ts
🟢=1 🟡=0 🔴=1 / total 2

### vbgui/e2e/scenarios/34_sharding_apply.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/35_memory_peak_parity.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/36_precision_toggles.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/45_cross_arch_non_canonical.spec.ts
🟢=2 🟡=0 🔴=0 / total 2

### vbgui/e2e/scenarios/51_hybrid_delta.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/52_clip_norm.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/53_roundtrip_exact.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/55_sharding_apply.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/57_train_cancel.spec.ts
🟢=1 🟡=0 🔴=0 / total 1

### vbgui/e2e/scenarios/58_warm_start.spec.ts
🟢=1 🟡=0 🔴=0 / total 1

### vbgui/e2e/scenarios/59_checkpoint_save_load.spec.ts
🟢=0 🟡=2 🔴=0 / total 2

### vbgui/e2e/scenarios/60_long_run_ui.spec.ts
🟢=1 🟡=0 🔴=0 / total 1

### vbgui/e2e/scenarios/61_dimensions_feedback.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/62_probe_text.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/63_save_load_roundtrip.spec.ts
🟢=1 🟡=0 🔴=0 / total 1

### vbgui/e2e/scenarios/65_memory_parity.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/66_ws_reconnect.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/68_ablation_expand.spec.ts
🟢=1 🟡=0 🔴=0 / total 1

### vbgui/e2e/scenarios/69_per_rank_shard.spec.ts
🟢=1 🟡=0 🔴=0 / total 1

### vbgui/e2e/scenarios/70_dtype_actual.spec.ts
🟢=2 🟡=0 🔴=0 / total 2

### vbgui/e2e/scenarios/71_moe_routing.spec.ts
🟢=1 🟡=0 🔴=0 / total 1

### vbgui/e2e/scenarios/72_strict_continuation.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/73_fake_ranks.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/74_resume_drift.spec.ts
🟢=1 🟡=1 🔴=0 / total 2

### vbgui/e2e/scenarios/75_concurrent_train.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/77_long_checkpoint_walk.spec.ts
🟢=1 🟡=0 🔴=0 / total 1


## V7 follow-ups

Decorative-only tests (1119) — candidates for tightening:

- `tests/v4/test_kda_path_b_bwd.py::test_kda_forward_matches_path_a`
- `tests/v4/test_lightning_indexer_fp8.py::test_fp8_indexer_constructs_with_fp8_storage`
- `tests/v4/test_lightning_indexer_fp8.py::test_fp8_indexer_constructs_with_bf16_fallback`
- `tests/v4/test_lightning_indexer_fp8.py::test_fp8_indexer_forward_shape`
- `tests/v4/test_lightning_indexer_fp8.py::test_quantize_indexer_weights_round_trip_close_to_fp32`
- `tests/v4/test_lightning_indexer_fp8.py::test_fp8_indexer_load_rejects_non_uint8`
- `tests/v4/test_lightning_indexer_fp8.py::test_fp8_indexer_topk_is_stop_gradient`
- `tests/v4/test_streaming_kv_cache.py::test_cache_construct_validation`
- `tests/v4/test_streaming_kv_cache.py::test_cache_append_validation`
- `tests/v4/test_streaming_kv_cache.py::test_cache_matches_oneshot_mean_pool`
- `tests/v4/test_streaming_kv_cache.py::test_cache_partial_block_finalized`
- `tests/v4/test_streaming_kv_cache.py::test_cache_counters_tracked`
- `tests/v4/test_streaming_kv_cache.py::test_cache_reset_clears_state`
- `tests/v4/test_streaming_kv_cache.py::test_cache_handles_single_token_appends`
- `tests/v4/test_linear_attention_path_a.py::test_config_rejects_invalid`
- `tests/v4/test_linear_attention_path_a.py::test_config_derived_dims`
- `tests/v4/test_linear_attention_path_a.py::test_num_v_heads_divisibility_validated`
- `tests/v4/test_linear_attention_path_a.py::test_naive_recurrent_shape`
- `tests/v4/test_linear_attention_path_a.py::test_naive_recurrent_output_final_state`
- `tests/v4/test_linear_attention_path_a.py::test_naive_recurrent_initial_state_carries`
- `tests/v4/test_linear_attention_path_a.py::test_parity_with_fla_torch_naive`
- `tests/v4/test_linear_attention_path_a.py::test_block_forward_shape`
- `tests/v4/test_linear_attention_path_a.py::test_block_rejects_wrong_rank_or_dim`
- `tests/v4/test_linear_attention_path_a.py::test_block_is_identity_at_init`
- `tests/v4/test_linear_attention_path_a.py::test_block_short_conv_runs`
- `tests/v4/test_linear_attention_path_a.py::test_block_doc_ids_changes_output`
- `tests/v4/test_linear_attention_path_a.py::test_block_doc_ids_shape_validation`
- `tests/v4/test_benchmark_matrix.py::test_promote_when_candidate_beats_incumbent_by_margin`
- `tests/v4/test_benchmark_matrix.py::test_no_promote_within_margin`
- `tests/v4/test_benchmark_matrix.py::test_skip_unavailable_paths`
- `tests/v4/test_benchmark_matrix.py::test_keep_incumbent_when_unchanged`
- `tests/v4/test_benchmark_matrix.py::test_run_matrix_gdn_covers_all_5_paths`
- `tests/v4/test_benchmark_matrix.py::test_run_matrix_kda_covers_all_5_paths`
- `tests/v4/test_benchmark_matrix.py::test_run_matrix_winner_beats_or_equals_path_a`
- `tests/v4/test_benchmark_matrix.py::test_write_matrix_receipt_produces_valid_json`
- `tests/v4/test_benchmark_matrix.py::test_promotion_decision_serializable`
- `tests/v4/test_jupyterlite_scaffold.py::test_jupyterlite_config_exists_and_parses`
- `tests/v4/test_jupyterlite_scaffold.py::test_jupyterlite_demo_notebook_is_valid`
- `tests/v4/test_jupyterlite_scaffold.py::test_jupyterlite_demo_imports_widget_class`
- `tests/v4/test_jupyterlite_scaffold.py::test_pages_workflow_exists_and_has_required_jobs`
- `tests/v4/test_jupyterlite_scaffold.py::test_jupyterlite_subdirs_present`
- `tests/v4/test_path_dispatch.py::test_path_status_truthy`
- `tests/v4/test_path_dispatch.py::test_parse_path_override_none_when_unset_or_auto`
- `tests/v4/test_path_dispatch.py::test_parse_path_override_returns_path`
- `tests/v4/test_path_dispatch.py::test_parse_path_override_rejects_unknown`
- `tests/v4/test_path_dispatch.py::test_auto_pick_prefers_first_available`
- `tests/v4/test_path_dispatch.py::test_auto_pick_falls_back_to_path_a`
- `tests/v4/test_path_dispatch.py::test_gdn_statuses_keys`
- `tests/v4/test_path_dispatch.py::test_gdn_path_a_always_available`
- `tests/v4/test_path_dispatch.py::test_gdn_path_d_and_c_reasons_are_coherent`
- … 1069 more
