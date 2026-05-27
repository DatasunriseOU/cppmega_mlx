# cppmega 1B Path Matrix

- Command: `scripts/bench_1b_training_matrix.py --steps 20 --batch-size 1 --block-size 2048 --dtypes bf16 --optimizers adamw --paths path_b --fresh-process --m04-memory-cap-gib 60 --out reports/cppmega_1b_path_b_vs_c_speed_probe.md --csv reports/cppmega_1b_path_b_vs_c_speed_probe.csv --json reports/cppmega_1b_path_b_vs_c_speed_probe.json --work-dir reports/raw/20260527_path_b_vs_c_speed_probe_cells --tilelang-cache-dir reports/raw/20260527_path_b_vs_c_speed_probe_tilelang_cache`
- cppmega SHA: `7756076`
- TileLang SHA: `69d70a22`
- MLX SHA: `d168ca5ca`
- MLX version: `0.32.0.dev20260514+d168ca5ca`

| dtype | optimizer | path | status | tok/s | step/s | compile s | peak GB | cache hit | profile trace | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| bf16 | adamw | path_b | ok | 217.738 | 0.107107 | 11.7891 | 29.2372 |  |  | ok |

## Cell Commands

- `bf16_adamw_path_b`: `/Volumes/external/sources/nanochat/.venv/bin/python3 scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 2048 --dtype bfloat16 --optimizer adamw --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output reports/raw/20260527_path_b_vs_c_speed_probe_cells/bf16_adamw_path_b.json --json --memory-limit-total-bytes 64424509440 --memory-limit-wired-ratio 0.99 --memory-limit-metal-ratio 0.99 --apply-memory-limit-plan`
