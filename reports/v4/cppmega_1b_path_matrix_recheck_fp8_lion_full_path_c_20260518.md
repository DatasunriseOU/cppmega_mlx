# cppmega 1B Path Matrix

- Command: `scripts/bench_1b_training_matrix.py --batch-size 1 --block-size 2048 --steps 20 --dtypes fp8 --optimizers lion --paths path_b,path_c_warm --mamba3-bwd path_c --fresh-process --work-dir reports/raw/cppmega_1b_path_matrix_recheck_20260518_fp8_lion_full_path_c --tilelang-cache-dir /tmp/cppmega_1b_path_matrix_tilelang_cache_20260518_recheck_fp8_lion_full_path_c --out reports/v4/cppmega_1b_path_matrix_recheck_fp8_lion_full_path_c_20260518.md --csv reports/v4/cppmega_1b_path_matrix_recheck_fp8_lion_full_path_c_20260518.csv --json reports/v4/cppmega_1b_path_matrix_recheck_fp8_lion_full_path_c_20260518.json`
- cppmega SHA: `f49a7e4`
- TileLang SHA: `2d23cd19`
- MLX SHA: `d168ca5ca`
- MLX version: `0.32.0.dev20260514+d168ca5ca`

| dtype | optimizer | path | status | tok/s | step/s | compile s | peak GB | cache hit | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| fp8 | lion | path_b | ok | 222.926 | 0.103231 | 4.77335 | 34.4309 |  | ok |
| fp8 | lion | path_c_warm | ok | 520.08 | 0.262242 | 38.0628 | 56.1679 | False | ok |

## Cell Commands

- `fp8_lion_path_b`: `/Volumes/external/sources/nanochat/.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 2048 --dtype fp8_path_b --optimizer lion --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output reports/raw/cppmega_1b_path_matrix_recheck_20260518_fp8_lion_full_path_c/fp8_lion_path_b.json --json`
- `fp8_lion_path_c_warm`: `/Volumes/external/sources/nanochat/.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 2048 --dtype fp8_path_c --optimizer lion --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output reports/raw/cppmega_1b_path_matrix_recheck_20260518_fp8_lion_full_path_c/fp8_lion_path_c_warm.json --json`
