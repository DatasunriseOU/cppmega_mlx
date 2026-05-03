# TileLang Metal Mamba3 Path C Profile

Scope: local M4 Max validation of Mamba3 Path C, the TileLang
`@T.prim_func` form lowered to Metal and dispatched through MLX
`mx.fast.metal_kernel`, against Path B, the hand-written MSL kernel.

Environment:
- Host: `Davids-Mac-Studio.local`
- Device: Apple M4 Max, `applegpu_g16s`
- Python: 3.13.12
- MLX: 0.31.1
- TileLang: 0.1.9+git7f4a5cb8

## Commands

Correctness:

```bash
.venv/bin/python -m pytest tests/test_tilelang_mamba3_path_c.py -q
```

Benchmarks:

```bash
.venv/bin/python scripts/bench_tilelang_mamba3_path_c.py \
  --seq 128 --warmup 10 --iters 100 \
  --output docs/upstream/tilelang_metal_mamba3/mamba3_path_c_seq128_run.json \
  --msl-dump docs/upstream/tilelang_metal_mamba3/mamba3_path_c_lowered_seq128.metal \
  --diff-output docs/upstream/tilelang_metal_mamba3/mamba3_path_b_vs_c_seq128.diff

.venv/bin/python scripts/bench_tilelang_mamba3_path_c.py \
  --warmup 10 --iters 100 \
  --output docs/upstream/tilelang_metal_mamba3/mamba3_path_c_spec_run.json \
  --msl-dump docs/upstream/tilelang_metal_mamba3/mamba3_path_c_lowered_spec.metal \
  --diff-output docs/upstream/tilelang_metal_mamba3/mamba3_path_b_vs_c_spec.diff

.venv/bin/python scripts/bench_tilelang_mamba3_path_c.py \
  --seq 1024 --warmup 10 --iters 50 \
  --output docs/upstream/tilelang_metal_mamba3/mamba3_path_c_seq1024_run.json \
  --msl-dump docs/upstream/tilelang_metal_mamba3/mamba3_path_c_lowered_seq1024.metal \
  --diff-output docs/upstream/tilelang_metal_mamba3/mamba3_path_b_vs_c_seq1024.diff
```

Profiler captures require Metal capture to be enabled:

```bash
MTL_CAPTURE_ENABLED=1 .venv/bin/python <capture script>
```

Markers are the separate capture filenames:
- `captures/mamba3_spec_path_b_fwd.gputrace`
- `captures/mamba3_spec_path_c_fwd.gputrace`
- `captures/mamba3_spec_path_b_bwd.gputrace`
- `captures/mamba3_spec_path_c_bwd.gputrace`

## Correctness

`tests/test_tilelang_mamba3_path_c.py` passed: 11 passed, 2 TileLang
builder deprecation warnings.

All benchmark receipts reported bit-identical forward parity against Path B:

| Shape | `y_max_abs` | `h_max_abs` |
| --- | ---: | ---: |
| B=2 T=128 H=4 P=32 N=64 fp32 | 0.000e+00 | 0.000e+00 |
| B=2 T=512 H=4 P=32 N=64 fp32 | 0.000e+00 | 0.000e+00 |
| B=2 T=1024 H=4 P=32 N=64 fp32 | 0.000e+00 | 0.000e+00 |

## Timings

Median timings from same-process Path B/Path C matched runs:

| Shape | Metric | Path B | Path C | C/B |
| --- | --- | ---: | ---: | ---: |
| T=128 | fwd | 0.723 ms | 0.683 ms | 0.946 |
| T=128 | bwd | 1.333 ms | 1.323 ms | 0.992 |
| T=128 | fwd+bwd | 2.056 ms | 2.006 ms | 0.976 |
| T=512 | fwd | 1.046 ms | 1.079 ms | 1.031 |
| T=512 | bwd | 6.933 ms | 6.940 ms | 1.001 |
| T=512 | fwd+bwd | 7.979 ms | 8.019 ms | 1.005 |
| T=1024 | fwd | 2.141 ms | 2.163 ms | 1.011 |
| T=1024 | bwd | 12.307 ms | 13.100 ms | 1.064 |
| T=1024 | fwd+bwd | 14.448 ms | 15.263 ms | 1.056 |

Peak memory matched Path B at every measured shape:

| Shape | fwd peak | fwd+bwd peak |
| --- | ---: | ---: |
| T=128 | 1.38 MB | 26.83 MB |
| T=512 | 4.78 MB | 106.38 MB |
| T=1024 | 9.31 MB | 212.44 MB |

Interpretation: the original table claim holds for the checked shapes. Path C
is stable and near Path B, but the current checked-in 256-thread Path C is not
clearly better than Path B.

## Profiler Captures

Programmatic MLX capture only emitted files when `MTL_CAPTURE_ENABLED=1` was set.
The initial attempt without that environment variable produced no trace files.

Captured traces:

| Marker filename | Size | Scope |
| --- | ---: | --- |
| `captures/mamba3_spec_path_b_fwd.gputrace` | 136184234 bytes | spec-shape Path B forward |
| `captures/mamba3_spec_path_c_fwd.gputrace` | 136183768 bytes | spec-shape Path C forward |
| `captures/mamba3_spec_path_b_bwd.gputrace` | 352162953 bytes | spec-shape Path B fwd+bwd |
| `captures/mamba3_spec_path_c_bwd.gputrace` | 352161968 bytes | spec-shape Path C fwd+bwd |

## Bottlenecks

Path C is not a separate runtime path. It lowers TileLang DSL to MSL and then
uses the same MLX `mx.fast.metal_kernel` dispatcher as Path B.

The lowered MSL is structurally the same per-lane scan:
- One thread owns each `(batch, head, headdim)` lane.
- Time is serial inside the thread.
- State dimension is serial inside the thread with `thread float h_state[64]`.
- Forward uses per-timestep `exp` for decay and another `exp` for the sigmoid.
- Backward rematerializes `h_steps` into global scratch, then performs a serial
  reverse scan and emits per-lane partial gradients.
- Host-side MLX reductions over the `P` axis are still needed for `dB`, `dC`,
  `dA`, `ddt`, and `dD`.

The dominant cost is therefore the same in Path B and Path C: serial recurrent
work per lane plus large backward scratch traffic. TileLang lowering currently
does not introduce a different schedule that would make Path C inherently
faster.

## Threadgroup Probe

No source files were edited for this probe. A fresh Python process monkeypatched
`_threads_for` in memory and rebuilt the Path C shape-specialized kernels for
the spec shape.

| Path C threads | fwd median | fwd+bwd median | Notes |
| ---: | ---: | ---: | --- |
| 64 | 1.012 ms | 6.452 ms | Best fwd+bwd in this probe |
| 128 | 0.980 ms | 6.681 ms | Best fwd-only in this probe |
| 256 | 1.065 ms | 7.828 ms | Current checked-in policy |

This is the only observed path to make Path C clearly better than the current
table numbers without changing the algorithm. It needs a guarded production
change and a rerun across the same shapes before changing the hot path.

## Verdict

Path C is correct and stable enough to keep as the upstream TileLang Metal
repro artifact. It does not beat Path B as currently checked in, but the
threadgroup probe suggests that reducing Path C from 256 threads to 64 or 128
threads could make it clearly faster at the spec shape. That should be a small,
isolated follow-up change with regression tests and the same three-shape
benchmark matrix.
