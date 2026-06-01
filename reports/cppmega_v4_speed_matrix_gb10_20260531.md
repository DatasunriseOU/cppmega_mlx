# cppmega v4 Speed Matrix (op-level GDN/KDA) — gb10 (CUDA sm_121, GB10)

- Date: 20260531
- Host: Linux-6.17.0-1021-nvidia-aarch64-with-glibc2.39
- cppmega SHA: `66932dc`
- Python: 3.13.11
- Op measured: v4 linear-attention FORWARD (GDN gated-delta + KDA recurrent), NOT full m04 1B training (the v4 benchmark harness is op-level).
- Shape: B=1 T=512 H=8 Dk=64 Dv=64 (Path-E-eligible: Dk%32==0, Dv%4==0)
- Sweep: block {gdn,kda} x path {a,b,c,d,e} x dtype {f32,bf16,fp16,fp16d}. warmup=2, iters=5, median of 5. allow_fallback=DISABLED (fail-loud).
- dtype axis: f32=harness reference; bf16=REAL cast of q/k/v/beta/g. fp8/nvfp4 are NOT representable as MLX op-input array dtypes (no float8 array dtype; only to_fp8/from_fp8 packers) so they do not apply at the op level and are omitted (the v3 fp8 axis was an m04-training GEMM storage scheme, a different layer).
- Per-cell bound: 1800s OS timeout, fresh subprocess (GPU cleared on exit).

| block | dtype | path | status | measured_path | median fwd ms | throughput Melem/s | reason |
| ----- | ----- | ---- | ------ | ------------- | ------------: | -----------------: | ------ |
| gdn | f32 | a | ok | a | 39.837 | 6.58 | pure-MLX reference |
| gdn | f32 | b | failed | b | — | — | RuntimeError: [metal_kernel] No Metal back-end. |
| gdn | f32 | c | failed | c | — | — | RuntimeError: GDN dispatch: selected path_c crashed at runtime (DLPackDeviceError: array is on DLDeviceType(13):0, but this path requires kDLMetal:0). Refusing  |
| gdn | f32 | d | failed | d | — | — | RuntimeError: GDN path_d unavailable and fallback disabled: GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| gdn | f32 | e | ok | e | 25.921 | 10.11 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| gdn | bf16 | a | ok | a | 32.323 | 8.11 | pure-MLX reference |
| gdn | bf16 | b | failed | b | — | — | RuntimeError: [metal_kernel] No Metal back-end. |
| gdn | bf16 | c | failed | c | — | — | RuntimeError: GDN dispatch: selected path_c crashed at runtime (DLPackDeviceError: array is on DLDeviceType(13):0, but this path requires kDLMetal:0). Refusing  |
| gdn | bf16 | d | failed | d | — | — | RuntimeError: GDN path_d unavailable and fallback disabled: GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| gdn | bf16 | e | ok | e | 59.365 | 4.42 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| gdn | fp16 | a | ok | a | 37.773 | 6.94 | pure-MLX reference |
| gdn | fp16 | b | failed | b | — | — | RuntimeError: [metal_kernel] No Metal back-end. |
| gdn | fp16 | c | failed | c | — | — | RuntimeError: GDN dispatch: selected path_c crashed at runtime (DLPackDeviceError: array is on DLDeviceType(13):0, but this path requires kDLMetal:0). Refusing  |
| gdn | fp16 | d | failed | d | — | — | RuntimeError: GDN path_d unavailable and fallback disabled: GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| gdn | fp16 | e | ok | e | 60.128 | 4.36 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| gdn | fp16d | a | ok | a | 40.831 | 6.42 | pure-MLX reference |
| gdn | fp16d | b | failed | b | — | — | RuntimeError: [metal_kernel] No Metal back-end. |
| gdn | fp16d | c | failed | c | — | — | RuntimeError: GDN dispatch: selected path_c crashed at runtime (DLPackDeviceError: array is on DLDeviceType(13):0, but this path requires kDLMetal:0). Refusing  |
| gdn | fp16d | d | failed | d | — | — | RuntimeError: GDN path_d unavailable and fallback disabled: GDN Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| gdn | fp16d | e | ok | e | 19.371 | 13.53 | vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; forward runs for ANY Dk via the in-MSL remainder-mask and any Dv; gate must be g<=0 for GDN, oth |
| kda | f32 | a | ok | a | 24.244 | 10.81 | pure-MLX KDA reference |
| kda | f32 | b | failed | b | — | — | RuntimeError: [metal_kernel] No Metal back-end. |
| kda | f32 | c | failed | c | — | — | RuntimeError: KDA dispatch: selected path_c crashed at runtime (DLPackDeviceError: array is on DLDeviceType(13):0, but this path requires kDLMetal:0). Refusing  |
| kda | f32 | d | failed | d | — | — | RuntimeError: KDA path_d unavailable and fallback disabled: KDA Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| kda | f32 | e | failed | e | — | — | RuntimeError: KDA path_e unavailable and fallback disabled: KDA Path E requires Metal for the vendored gated_delta kernel; Metal is unavailable on this host |
| kda | bf16 | a | ok | a | 35.946 | 7.29 | pure-MLX KDA reference |
| kda | bf16 | b | failed | b | — | — | RuntimeError: [metal_kernel] No Metal back-end. |
| kda | bf16 | c | failed | c | — | — | RuntimeError: KDA dispatch: selected path_c crashed at runtime (DLPackDeviceError: array is on DLDeviceType(13):0, but this path requires kDLMetal:0). Refusing  |
| kda | bf16 | d | failed | d | — | — | RuntimeError: KDA path_d unavailable and fallback disabled: KDA Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| kda | bf16 | e | failed | e | — | — | RuntimeError: KDA path_e unavailable and fallback disabled: KDA Path E requires Metal for the vendored gated_delta kernel; Metal is unavailable on this host |
| kda | fp16 | a | ok | a | 33.943 | 7.72 | pure-MLX KDA reference |
| kda | fp16 | b | failed | b | — | — | RuntimeError: [metal_kernel] No Metal back-end. |
| kda | fp16 | c | failed | c | — | — | RuntimeError: KDA dispatch: selected path_c crashed at runtime (DLPackDeviceError: array is on DLDeviceType(13):0, but this path requires kDLMetal:0). Refusing  |
| kda | fp16 | d | failed | d | — | — | RuntimeError: KDA path_d unavailable and fallback disabled: KDA Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| kda | fp16 | e | failed | e | — | — | RuntimeError: KDA path_e unavailable and fallback disabled: KDA Path E requires Metal for the vendored gated_delta kernel; Metal is unavailable on this host |
| kda | fp16d | a | ok | a | 33.433 | 7.84 | pure-MLX KDA reference |
| kda | fp16d | b | failed | b | — | — | RuntimeError: [metal_kernel] No Metal back-end. |
| kda | fp16d | c | failed | c | — | — | RuntimeError: KDA dispatch: selected path_c crashed at runtime (DLPackDeviceError: array is on DLDeviceType(13):0, but this path requires kDLMetal:0). Refusing  |
| kda | fp16d | d | failed | d | — | — | RuntimeError: KDA path_d unavailable and fallback disabled: KDA Path D: Triton kernel -> poc.triton_frontend.from_triton_kernel → tilelang.compile. unsafe trito |
| kda | fp16d | e | failed | e | — | — | RuntimeError: KDA path_e unavailable and fallback disabled: KDA Path E requires Metal for the vendored gated_delta kernel; Metal is unavailable on this host |

## Per-Cell Commands

- `gdn_f32_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_a f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_b f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_c f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_d f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_f32_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_e f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_a bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_b bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_c bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_d bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_bf16_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_e bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_a fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_b fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_c fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_d fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_e fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_a fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_b fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_c fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_d fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `gdn_fp16d_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' gdn path_e fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_a f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_b f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_c f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_d f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_f32_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_e f32` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_a bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_b bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_c bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_d bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_bf16_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_e bf16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_a fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_b fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_c fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_d fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_e fp16` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_a`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_a fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_b`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_b fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_c`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_c fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_d`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_d fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
- `kda_fp16d_path_e`: `/usr/bin/timeout 1800 /home/dave/cppmega-venv/bin/python -c '<cell-script>' kda path_e fp16d` (driver: scripts/run_v4_speed_matrix_20260531.py)
