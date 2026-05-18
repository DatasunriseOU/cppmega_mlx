# cppmega 1B Path Matrix

- Command: `scripts/bench_1b_training_matrix.py --batch-size 1 --block-size 2048 --steps 20 --dtypes bf16 --optimizers lion --paths path_b,path_c_warm --mamba3-bwd path_b --fresh-process --work-dir reports/raw/cppmega_1b_path_matrix_recheck_20260518_lion --tilelang-cache-dir /tmp/cppmega_1b_path_matrix_tilelang_cache_20260518_lion_recheck --out reports/v4/cppmega_1b_path_matrix_recheck_lion_20260518.md --csv reports/v4/cppmega_1b_path_matrix_recheck_lion_20260518.csv --json reports/v4/cppmega_1b_path_matrix_recheck_lion_20260518.json`
- cppmega SHA: `430983d`
- TileLang SHA: `2d23cd19`
- MLX SHA: `d168ca5ca`
- MLX version: `0.32.0.dev20260514+d168ca5ca`

| dtype | optimizer | path | status | tok/s | step/s | compile s | peak GB | cache hit | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| bf16 | lion | path_b | ok | 800.366 | 0.396588 | 3.67782 | 33.5137 |  | ok |
| bf16 | lion | path_c_warm | ok | 541.751 | 0.276135 | 22.0888 | 33.5134 | False | ok |

## Cell Commands

- `bf16_lion_path_b`: `/Volumes/external/sources/nanochat/.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 2048 --dtype bfloat16 --optimizer lion --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output reports/raw/cppmega_1b_path_matrix_recheck_20260518_lion/bf16_lion_path_b.json --json`
- `bf16_lion_path_c_warm`: `/Volumes/external/sources/nanochat/.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 2048 --dtype bfloat16 --optimizer lion --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output reports/raw/cppmega_1b_path_matrix_recheck_20260518_lion/bf16_lion_path_c_warm.json --json`
