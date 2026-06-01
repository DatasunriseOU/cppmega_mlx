# cppmega v4 Speed Matrix (op-level GDN/KDA) — Local Metal (Apple M4 Max)

- Date: 20260531
- Host: macOS-26.5-arm64-arm-64bit-Mach-O
- cppmega SHA: `ffb3069`
- Python: 3.13.12
- Op measured: v4 linear-attention FORWARD (GDN gated-delta + KDA recurrent), NOT full m04 1B training (the v4 benchmark harness is op-level).
- Shape: B=1 T=512 H=8 Dk=64 Dv=64 (Path-E-eligible: Dk%32==0, Dv%4==0)
- Sweep: block {gdn,kda} x path {a,b,c,d,e} x dtype {f32,bf16,fp16,fp16d}. warmup=2, iters=5, median of 5. allow_fallback=DISABLED (fail-loud).
- dtype axis: f32=harness reference; bf16=REAL cast of q/k/v/beta/g. fp8/nvfp4 are NOT representable as MLX op-input array dtypes (no float8 array dtype; only to_fp8/from_fp8 packers) so they do not apply at the op level and are omitted (the v3 fp8 axis was an m04-training GEMM storage scheme, a different layer).
- Per-cell bound: 1800s OS timeout, fresh subprocess (GPU cleared on exit).

| block | dtype | path | status | measured_path | median fwd ms | throughput Melem/s | reason |
| ----- | ----- | ---- | ------ | ------------- | ------------: | -----------------: | ------ |
| gdn | f32 | a | ok | a | 24.292 | 10.79 | pure-MLX reference |
| gdn | f32 | b | ok | b | 2.156 | 121.60 | hand-MSL GDN forward via mx.fast.metal_kernel; the autograd-aware wrapper gdn_apply_path_b in linear_attention_path_b_bwd.py also provides a real Metal backward |
| gdn | f32 | c | ok | c | 2.738 | 95.73 | GDN Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| gdn | f32 | d | failed | d | — | — | RuntimeError: GDN path_d unavailable and fallback disabled: GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| gdn | f32 | e | ok | e | 1.097 | 239.06 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| gdn | bf16 | a | ok | a | 21.740 | 12.06 | pure-MLX reference |
| gdn | bf16 | b | ok | b | 2.176 | 120.48 | hand-MSL GDN forward via mx.fast.metal_kernel; the autograd-aware wrapper gdn_apply_path_b in linear_attention_path_b_bwd.py also provides a real Metal backward |
| gdn | bf16 | c | ok | c | 4.328 | 60.58 | GDN Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| gdn | bf16 | d | failed | d | — | — | RuntimeError: GDN path_d unavailable and fallback disabled: GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| gdn | bf16 | e | ok | e | 1.148 | 228.41 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| gdn | fp16 | a | ok | a | 22.130 | 11.85 | pure-MLX reference |
| gdn | fp16 | b | ok | b | 2.225 | 117.83 | hand-MSL GDN forward via mx.fast.metal_kernel; the autograd-aware wrapper gdn_apply_path_b in linear_attention_path_b_bwd.py also provides a real Metal backward |
| gdn | fp16 | c | ok | c | 1.883 | 139.24 | GDN Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| gdn | fp16 | d | failed | d | — | — | RuntimeError: GDN path_d unavailable and fallback disabled: GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| gdn | fp16 | e | ok | e | 1.160 | 226.08 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| gdn | fp16d | a | ok | a | 22.109 | 11.86 | pure-MLX reference |
| gdn | fp16d | b | ok | b | 2.437 | 107.56 | hand-MSL GDN forward via mx.fast.metal_kernel; the autograd-aware wrapper gdn_apply_path_b in linear_attention_path_b_bwd.py also provides a real Metal backward |
| gdn | fp16d | c | ok | c | 2.009 | 130.49 | GDN Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| gdn | fp16d | d | failed | d | — | — | RuntimeError: GDN path_d unavailable and fallback disabled: GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| gdn | fp16d | e | ok | e | 1.163 | 225.43 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| kda | f32 | a | ok | a | 21.870 | 11.99 | pure-MLX KDA reference |
| kda | f32 | b | ok | b | 3.075 | 85.26 | hand-MSL KDA forward via mx.fast.metal_kernel; the autograd-aware wrapper kda_apply_path_b in kda_path_b_bwd.py also provides a real Metal backward (V <= 256) |
| kda | f32 | c | ok | c | 3.776 | 69.42 | KDA Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| kda | f32 | d | failed | d | — | — | RuntimeError: KDA path_d unavailable and fallback disabled: KDA Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| kda | f32 | e | ok | e | 1.119 | 234.22 | vendored mlx-lm gated_delta vectorised-gate Metal kernel (fast kernel for Dk%32==0 & Dv%4==0; fails closed for smaller dims so the dispatcher falls back to Path |
| kda | bf16 | a | ok | a | 22.927 | 11.43 | pure-MLX KDA reference |
| kda | bf16 | b | ok | b | 2.950 | 88.86 | hand-MSL KDA forward via mx.fast.metal_kernel; the autograd-aware wrapper kda_apply_path_b in kda_path_b_bwd.py also provides a real Metal backward (V <= 256) |
| kda | bf16 | c | ok | c | 5.652 | 46.38 | KDA Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| kda | bf16 | d | failed | d | — | — | RuntimeError: KDA path_d unavailable and fallback disabled: KDA Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| kda | bf16 | e | ok | e | 0.957 | 274.01 | vendored mlx-lm gated_delta vectorised-gate Metal kernel (fast kernel for Dk%32==0 & Dv%4==0; fails closed for smaller dims so the dispatcher falls back to Path |
| kda | fp16 | a | ok | a | 20.496 | 12.79 | pure-MLX KDA reference |
| kda | fp16 | b | ok | b | 2.594 | 101.04 | hand-MSL KDA forward via mx.fast.metal_kernel; the autograd-aware wrapper kda_apply_path_b in kda_path_b_bwd.py also provides a real Metal backward (V <= 256) |
| kda | fp16 | c | ok | c | 2.496 | 105.04 | KDA Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| kda | fp16 | d | failed | d | — | — | RuntimeError: KDA path_d unavailable and fallback disabled: KDA Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| kda | fp16 | e | ok | e | 0.914 | 286.93 | vendored mlx-lm gated_delta vectorised-gate Metal kernel (fast kernel for Dk%32==0 & Dv%4==0; fails closed for smaller dims so the dispatcher falls back to Path |
| kda | fp16d | a | ok | a | 20.897 | 12.54 | pure-MLX KDA reference |
| kda | fp16d | b | ok | b | 3.259 | 80.43 | hand-MSL KDA forward via mx.fast.metal_kernel; the autograd-aware wrapper kda_apply_path_b in kda_path_b_bwd.py also provides a real Metal backward (V <= 256) |
| kda | fp16d | c | ok | c | 3.825 | 68.54 | KDA Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal', execution_backend='tvm_ffi'). tilelang + host TileLang→MSL infra reachable |
| kda | fp16d | d | failed | d | — | — | RuntimeError: KDA path_d unavailable and fallback disabled: KDA Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| kda | fp16d | e | ok | e | 0.921 | 284.68 | vendored mlx-lm gated_delta vectorised-gate Metal kernel (fast kernel for Dk%32==0 & Dv%4==0; fails closed for smaller dims so the dispatcher falls back to Path |

## Per-Cell Commands

- `gdn_f32_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_a f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_b f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_c f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_d f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_e f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_a bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_b bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_c bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_d bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_e bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_a fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_b fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_c fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_d fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_e fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_a fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_b fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_c fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_d fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' gdn path_e fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_a f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_b f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_c f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_d f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_e f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_a bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_b bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_c bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_d bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_e bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_a fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_b fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_c fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_d fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_e fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_a`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_a fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_b`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_b fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_c`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_c fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_d`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_d fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_e`: `/opt/homebrew/bin/timeout 1800 /Volumes/external/sources/nanochat/.venv/bin/python -c '<cell-script>' kda path_e fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
