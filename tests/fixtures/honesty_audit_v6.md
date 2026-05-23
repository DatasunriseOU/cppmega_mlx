# V6 honesty audit

Classifier from `scripts/audit_v3_v4_v5.py` (heuristic).

| Category | Count |
|---|---|
| 🟢 math-effect | 354 |
| 🟡 propagation | 49 |
| 🔴 decorative | 1290 |

Total: 1693

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

### tests/v4/test_block_transplant.py
🟢=2 🟡=0 🔴=0 / total 2

### tests/v4/test_cache_stats.py
🟢=0 🟡=0 🔴=5 / total 5

### tests/v4/test_catalog.py
🟢=1 🟡=0 🔴=20 / total 21

### tests/v4/test_checkpoint_arch_mismatch.py
🟢=2 🟡=0 🔴=2 / total 4

### tests/v4/test_checkpoint_metadata.py
🟢=0 🟡=1 🔴=4 / total 5

### tests/v4/test_checkpoint_quantize.py
🟢=1 🟡=0 🔴=4 / total 5

### tests/v4/test_checkpoint_shard.py
🟢=0 🟡=0 🔴=4 / total 4

### tests/v4/test_checkpoint_streaming.py
🟢=0 🟡=0 🔴=3 / total 3

### tests/v4/test_checkpoint_topology.py
🟢=0 🟡=0 🔴=2 / total 2

### tests/v4/test_ckpt_inspect_method.py
🟢=1 🟡=0 🔴=2 / total 3

### tests/v4/test_collective_proxy.py
🟢=0 🟡=0 🔴=5 / total 5

### tests/v4/test_corpus_stats.py
🟢=0 🟡=0 🔴=4 / total 4

### tests/v4/test_corpus_stats_sidecar.py
🟢=0 🟡=0 🔴=2 / total 2

### tests/v4/test_csa_hca_indexer_adapter.py
🟢=2 🟡=0 🔴=5 / total 7

### tests/v4/test_csa_hca_streaming_decode.py
🟢=1 🟡=0 🔴=5 / total 6

### tests/v4/test_csa_hca_v4.py
🟢=2 🟡=0 🔴=8 / total 10

### tests/v4/test_data_inspector.py
🟢=1 🟡=0 🔴=13 / total 14

### tests/v4/test_data_roundtrip.py
🟢=1 🟡=0 🔴=7 / total 8

### tests/v4/test_dim_scaling_sweep.py
🟢=1 🟡=0 🔴=1 / total 2

### tests/v4/test_doc_id_assignment.py
🟢=0 🟡=0 🔴=4 / total 4

### tests/v4/test_dtype_cost_method.py
🟢=0 🟡=0 🔴=2 / total 2

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

### tests/v4/test_fp8_probe.py
🟢=0 🟡=0 🔴=2 / total 2

### tests/v4/test_fsdp_per_device_memory.py
🟢=3 🟡=0 🔴=0 / total 3

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

### tests/v4/test_gen_method.py
🟢=0 🟡=0 🔴=5 / total 5

### tests/v4/test_gen_run_method.py
🟢=0 🟡=0 🔴=6 / total 6

### tests/v4/test_generate_eos.py
🟢=0 🟡=0 🔴=4 / total 4

### tests/v4/test_generate_stream.py
🟢=0 🟡=0 🔴=4 / total 4

### tests/v4/test_grad_clip_realistic.py
🟢=0 🟡=0 🔴=2 / total 2

### tests/v4/test_hybrid_lm.py
🟢=1 🟡=0 🔴=5 / total 6

### tests/v4/test_hybrid_muon_adamw_scale.py
🟢=1 🟡=0 🔴=1 / total 2

### tests/v4/test_hybrid_precision.py
🟢=4 🟡=0 🔴=0 / total 4

### tests/v4/test_inference_log.py
🟢=1 🟡=0 🔴=9 / total 10

### tests/v4/test_inspect_histogram.py
🟢=0 🟡=0 🔴=3 / total 3

### tests/v4/test_job_control.py
🟢=0 🟡=0 🔴=6 / total 6

### tests/v4/test_jsonrpc_cache.py
🟢=0 🟡=0 🔴=15 / total 15

### tests/v4/test_jsonrpc_dispatcher.py
🟢=2 🟡=0 🔴=9 / total 11

### tests/v4/test_jsonrpc_methods.py
🟢=2 🟡=0 🔴=12 / total 14

### tests/v4/test_jsonrpc_review_fixes.py
🟢=1 🟡=0 🔴=12 / total 13

### tests/v4/test_jsonrpc_schema.py
🟢=5 🟡=0 🔴=23 / total 28

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

### tests/v4/test_kv_cache.py
🟢=0 🟡=0 🔴=8 / total 8

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

### tests/v4/test_llama3_8b_scaled_e2e.py
🟢=1 🟡=0 🔴=1 / total 2

### tests/v4/test_loss_scaler.py
🟢=8 🟡=0 🔴=0 / total 8

### tests/v4/test_loss_surface.py
🟢=2 🟡=0 🔴=0 / total 2

### tests/v4/test_lr_schedule_long_horizon.py
🟢=1 🟡=0 🔴=2 / total 3

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
🟢=2 🟡=1 🔴=2 / total 5

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

### tests/v4/test_moe_capacity.py
🟢=1 🟡=0 🔴=4 / total 5

### tests/v4/test_moe_fp8.py
🟢=3 🟡=0 🔴=5 / total 8

### tests/v4/test_moe_realistic_scale.py
🟢=2 🟡=0 🔴=0 / total 2

### tests/v4/test_moe_v4.py
🟢=5 🟡=0 🔴=13 / total 18

### tests/v4/test_moe_v4_capacity_overflow.py
🟢=0 🟡=0 🔴=2 / total 2

### tests/v4/test_moe_v4_inference.py
🟢=0 🟡=0 🔴=4 / total 4

### tests/v4/test_moe_v4_shared_vs_routed.py
🟢=1 🟡=0 🔴=0 / total 1

### tests/v4/test_moe_v4_specialisation.py
🟢=1 🟡=0 🔴=0 / total 1

### tests/v4/test_moe_v4_trajectory.py
🟢=0 🟡=0 🔴=3 / total 3

### tests/v4/test_mtp_speculative_adapter.py
🟢=2 🟡=0 🔴=4 / total 6

### tests/v4/test_mtp_v4.py
🟢=5 🟡=0 🔴=11 / total 16

### tests/v4/test_multi_node_topology.py
🟢=0 🟡=0 🔴=5 / total 5

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

### tests/v4/test_per_brick_grads_extras.py
🟢=0 🟡=0 🔴=1 / total 1

### tests/v4/test_per_brick_probes.py
🟢=1 🟡=0 🔴=1 / total 2

### tests/v4/test_pipeline_pause_resume.py
🟢=1 🟡=0 🔴=2 / total 3

### tests/v4/test_pipeline_pause_resume_rpc.py
🟢=0 🟡=0 🔴=3 / total 3

### tests/v4/test_pp_proxy.py
🟢=0 🟡=0 🔴=5 / total 5

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

### tests/v4/test_real_corpus_epoch_convergence.py
🟢=1 🟡=0 🔴=0 / total 1

### tests/v4/test_run_template.py
🟢=0 🟡=0 🔴=21 / total 21

### tests/v4/test_runner_cli.py
🟢=0 🟡=0 🔴=12 / total 12

### tests/v4/test_runner_pipeline.py
🟢=0 🟡=10 🔴=12 / total 22

### tests/v4/test_samplers.py
🟢=0 🟡=0 🔴=8 / total 8

### tests/v4/test_schedules.py
🟢=8 🟡=0 🔴=20 / total 28

### tests/v4/test_side_channel_spec.py
🟢=0 🟡=0 🔴=5 / total 5

### tests/v4/test_stage_train_8bit_optim.py
🟢=2 🟡=0 🔴=0 / total 2

### tests/v4/test_stage_train_data.py
🟢=0 🟡=0 🔴=8 / total 8

### tests/v4/test_stage_train_dtype_real.py
🟢=4 🟡=0 🔴=0 / total 4

### tests/v4/test_stage_train_fake_ranks.py
🟢=2 🟡=0 🔴=2 / total 4

### tests/v4/test_stage_train_fim.py
🟢=0 🟡=0 🔴=3 / total 3

### tests/v4/test_stage_train_fp16.py
🟢=3 🟡=0 🔴=0 / total 3

### tests/v4/test_stage_train_moe_real.py
🟢=3 🟡=0 🔴=0 / total 3

### tests/v4/test_stage_train_multishard.py
🟢=0 🟡=0 🔴=3 / total 3

### tests/v4/test_stage_train_optimizers.py
🟢=3 🟡=0 🔴=6 / total 9

### tests/v4/test_stage_train_perplexity.py
🟢=2 🟡=0 🔴=0 / total 2

### tests/v4/test_stage_train_resume_drift.py
🟢=2 🟡=0 🔴=0 / total 2

### tests/v4/test_stage_train_rng_roundtrip.py
🟢=1 🟡=0 🔴=2 / total 3

### tests/v4/test_stage_train_sharding_real.py
🟢=3 🟡=0 🔴=0 / total 3

### tests/v4/test_stage_train_strict_continuation.py
🟢=2 🟡=1 🔴=0 / total 3

### tests/v4/test_stage_train_tokenize.py
🟢=1 🟡=0 🔴=4 / total 5

### tests/v4/test_stage_train_val_loss.py
🟢=0 🟡=0 🔴=3 / total 3

### tests/v4/test_streaming_kv_cache.py
🟢=0 🟡=0 🔴=7 / total 7

### tests/v4/test_suggest_optim_groups.py
🟢=0 🟡=0 🔴=9 / total 9

### tests/v4/test_symbolic_dim_validation.py
🟢=0 🟡=5 🔴=0 / total 5

### tests/v4/test_tokenizer_cross_shard_stability.py
🟢=0 🟡=0 🔴=3 / total 3

### tests/v4/test_tokenizer_playground.py
🟢=0 🟡=0 🔴=11 / total 11

### tests/v4/test_tokenizer_preset_matrix.py
🟢=1 🟡=0 🔴=0 / total 1

### tests/v4/test_tp_proxy.py
🟢=0 🟡=0 🔴=4 / total 4

### tests/v4/test_train_event_bus.py
🟢=1 🟡=0 🔴=5 / total 6

### tests/v4/test_train_events.py
🟢=3 🟡=0 🔴=1 / total 4

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

### tests/v4/test_weight_delta_long_horizon.py
🟢=3 🟡=0 🔴=0 / total 3

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

### tests/v5/test_stage_train_loss_scaler.py
🟢=4 🟡=0 🔴=0 / total 4

### tests/v5/test_stage_train_moe.py
🟢=0 🟡=0 🔴=2 / total 2

### tests/v5/test_stage_train_moe_capacity.py
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

### vbgui/e2e/scenarios/51_drag_mlstm_into_edge.spec.ts
🟢=1 🟡=0 🔴=0 / total 1

### vbgui/e2e/scenarios/51_hybrid_delta.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/52_block_swap.spec.ts
🟢=1 🟡=0 🔴=0 / total 1

### vbgui/e2e/scenarios/52_clip_norm.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/53_roundtrip_exact.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/53_scaling_sweep.spec.ts
🟢=1 🟡=0 🔴=0 / total 1

### vbgui/e2e/scenarios/54_cross_preset_transplant.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/55_sharding_apply.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/55_tokenizer_matrix.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/56_symbolic_dim_warn.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/57_parallel_block_composition.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/57_train_cancel.spec.ts
🟢=1 🟡=0 🔴=0 / total 1

### vbgui/e2e/scenarios/58_gallery_sortable.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

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

### vbgui/e2e/scenarios/78_undo_redo.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/79_concurrent_train_race.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/80_fim_toggle.spec.ts
🟢=0 🟡=2 🔴=0 / total 2

### vbgui/e2e/scenarios/81_ckpt_inspect.spec.ts
🟢=0 🟡=1 🔴=0 / total 1

### vbgui/e2e/scenarios/82_loss_scaler.spec.ts
🟢=2 🟡=0 🔴=0 / total 2

### vbgui/e2e/scenarios/83_moe_capacity_factor.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/84_dtype_cost_display.spec.ts
🟢=0 🟡=0 🔴=1 / total 1

### vbgui/e2e/scenarios/85_pause_resume.spec.ts
🟢=0 🟡=0 🔴=2 / total 2

### vbgui/e2e/scenarios/86_corpus_stats_inspector.spec.ts
🟢=0 🟡=1 🔴=0 / total 1


## V7 follow-ups

Decorative-only tests (1290) — candidates for tightening:

- `tests/v4/test_kda_path_b_bwd.py::test_kda_forward_matches_path_a`
- `tests/v4/test_gen_method.py::test_v7_f01_greedy_strategy_runs_to_length`
- `tests/v4/test_gen_method.py::test_v7_f01_top_k_strategy_returns_valid_tokens`
- `tests/v4/test_gen_method.py::test_v7_f01_top_p_strategy_returns_valid_tokens`
- `tests/v4/test_gen_method.py::test_v7_f01_events_carry_step_token_finish`
- `tests/v4/test_gen_method.py::test_v7_f01_eos_halt_when_prompt_already_at_eos_minus_one`
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
- `tests/v4/test_per_brick_grads_extras.py::test_v7_h07_extras_carries_per_brick_grad_norms`
- `tests/v4/test_train_event_bus.py::test_publish_with_no_subscribers_is_noop`
- `tests/v4/test_train_event_bus.py::test_publish_to_other_run_id_is_isolated`
- `tests/v4/test_train_event_bus.py::test_unsubscribe_removes_queue_from_distribution`
- `tests/v4/test_train_event_bus.py::test_multiple_subscribers_all_receive`
- `tests/v4/test_train_event_bus.py::test_cross_thread_publish_safe`
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
- `tests/v4/test_moe_v4_inference.py::test_v7_e06_inference_router_selects_distinct_experts`
- `tests/v4/test_moe_v4_inference.py::test_v7_e06_inference_aux_loss_is_zero`
- `tests/v4/test_moe_v4_inference.py::test_v7_e06_inference_router_bias_does_not_change`
- `tests/v4/test_moe_v4_inference.py::test_v7_e06_inference_deterministic_outputs_same_seed`
- `tests/v4/test_ckpt_inspect_method.py::test_ckpt_inspect_missing_file`
- `tests/v4/test_ckpt_inspect_method.py::test_ckpt_inspect_no_metadata`
- `tests/v4/test_benchmark_matrix.py::test_promote_when_candidate_beats_incumbent_by_margin`
- `tests/v4/test_benchmark_matrix.py::test_no_promote_within_margin`
- `tests/v4/test_benchmark_matrix.py::test_skip_unavailable_paths`
- `tests/v4/test_benchmark_matrix.py::test_keep_incumbent_when_unchanged`
- `tests/v4/test_benchmark_matrix.py::test_run_matrix_gdn_covers_all_5_paths`
- `tests/v4/test_benchmark_matrix.py::test_run_matrix_kda_covers_all_5_paths`
- … 1240 more
