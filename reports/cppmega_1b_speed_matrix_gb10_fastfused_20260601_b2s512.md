# cppmega 1B Speed Matrix — gb10 (NVIDIA GB10, CUDA) — FAST FUSED Path-C

- Date: 20260601
- Host: gb10 (NVIDIA GB10, CUDA sm_121)
- Model profile: local_gb10_quarter
- Data: data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet (parquet, token_ids)
- Settings: seq-len 512, batch 2, --steps 10 warm, --grad-checkpoint, --optimizer-quant-scheme dynamic_int8_v1 (tokens/step = 1024)
- Paths: `path_b` (reference); `path_c` = Path-C flag-OFF (serial mamba3, prior baseline); `path_c_chunked` = Path-C **flag-ON** (CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN=1, FAST chunked fused mamba3).
- Path-C route: SPLIT/WARM (Path C fwd + Path B mamba3 bwd, mamba3 bwd=path_b).
- loss check = all losses finite AND final<initial (fail-loud per RULE #1).
- nvfp4: accepted route but full training step fails-LOUD (no nvfp4 training kernels yet) — honest blocked cell.
- CUDA note: the CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN flag is target-agnostic (same on Metal+CUDA); per-cell results report whatever CUDA measures.
- Per-cell bound: 1800s OS timeout (fail-loud)
- cppmega SHA: `a319599` · TileLang SHA: `unknown`

| dtype | optimizer | bits | path | status | tok/s | step/s | compile s | peak GB | loss check | first/last loss | flag |
| ----- | --------- | ---: | ---- | ------ | ----: | -----: | --------: | ------: | ---------- | --------------- | ---- |
| bf16 | muon | 16 | path_b | ok | 159 | 0.165 | 13.5 | 27 | PASS | 11.3 → 5.79 |  |
| bf16 | muon | 16 | path_c | blocked |  |  |  | 17.2 | FAIL |  | 0 |
| bf16 | muon | 16 | path_c_chunked | blocked |  |  |  | 17.2 | FAIL |  | 1 |
| bf16 | adamw | 16 | path_b | ok | 259 | 0.27 | 9.31 | 32 | PASS | 11.3 → 5.79 |  |
| bf16 | adamw | 16 | path_c | blocked |  |  |  | 22.3 | FAIL |  | 0 |
| bf16 | adamw | 16 | path_c_chunked | blocked |  |  |  | 22.3 | FAIL |  | 1 |

## Path-C flag-OFF → flag-ON (chunked fused mamba3) speedup

- step/s speedup = chunked step/s ÷ serial step/s (steady-state, >1 = chunked faster).
- compile speedup = serial first-step s ÷ chunked first-step s (>1 = chunked compiles/first-steps faster).

| dtype | optimizer | bits | serial step/s | chunked step/s | step/s speedup | serial compile s | chunked compile s | compile speedup |
| ----- | --------- | ---: | ------------: | -------------: | -------------: | ---------------: | ----------------: | --------------: |
| bf16 | muon | 16 |  |  |  |  |  |  |
| bf16 | adamw | 16 |  |  |  |  |  |  |

## Per-Cell Commands

- `bf16_muon16_path_b` (flag=): `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 10 --batch-size 2 --seq-len 512 --dtype bfloat16 --optimizer muon --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output /tmp/cppmega_1b_speed_matrix_gb10_fastfused_20260601_b2s512_cells/bf16_muon16_path_b.json --json`
- `bf16_muon16_path_c` (flag=0): `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 10 --batch-size 2 --seq-len 512 --dtype bfloat16 --optimizer muon --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output /tmp/cppmega_1b_speed_matrix_gb10_fastfused_20260601_b2s512_cells/bf16_muon16_path_c.json --json`
- `bf16_muon16_path_c_chunked` (flag=1): `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 10 --batch-size 2 --seq-len 512 --dtype bfloat16 --optimizer muon --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output /tmp/cppmega_1b_speed_matrix_gb10_fastfused_20260601_b2s512_cells/bf16_muon16_path_c_chunked.json --json`
- `bf16_adamw16_path_b` (flag=): `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 10 --batch-size 2 --seq-len 512 --dtype bfloat16 --optimizer adamw --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output /tmp/cppmega_1b_speed_matrix_gb10_fastfused_20260601_b2s512_cells/bf16_adamw16_path_b.json --json`
- `bf16_adamw16_path_c` (flag=0): `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 10 --batch-size 2 --seq-len 512 --dtype bfloat16 --optimizer adamw --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output /tmp/cppmega_1b_speed_matrix_gb10_fastfused_20260601_b2s512_cells/bf16_adamw16_path_c.json --json`
- `bf16_adamw16_path_c_chunked` (flag=1): `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 10 --batch-size 2 --seq-len 512 --dtype bfloat16 --optimizer adamw --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output /tmp/cppmega_1b_speed_matrix_gb10_fastfused_20260601_b2s512_cells/bf16_adamw16_path_c_chunked.json --json`
