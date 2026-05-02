# Mamba3 and M2RNN Local Route Benchmark Baseline

This is a local Apple M4 Max regression baseline for the MLX tiny route matrix.
It is not GB10 parity evidence. Use it only to catch route/backend/device
metadata regressions and large local performance changes in the Mamba3 and
M2RNN routes.

The NAM56R source placement contract is stricter than these tiny receipts:
source M layers map to Mamba3 positions and source R layers map to M2RNN
positions in the AEMEAEMEAEMR depth-52 layout, but the local rows below do not
exercise nam56r_full_spec.py, nam56r_te_spec.py, nam56r_noconv_spec.py,
Transformer Engine, Triton scans, TP mixer behavior, native MLA/MTP/DSA, or
H200/GB10 train launchers.

/Users/dave/.codex/prompts/executor.md was checked for this executor-style
subtask.

## Environment

- Date: 2026-04-30
- Host label: M4 Max
- Device: Apple M4 Max
- Default device: Device(gpu, 0)
- MLX: 0.31.1
- MLX-LM: 0.31.2
- MLX Metal: 0.31.1
- Python: 3.13.12
- Platform: macOS-26.4.1-arm64-arm-64bit-Mach-O
- Data contract: synthetic tokens

## Benchmark Command

bash
./.venv/bin/python scripts/bench_matrix.py \
  --json \
  --hardware-label "M4 Max" \
  --batch-sizes 1 \
  --seq-lens 8 \
  --profiles hybrid-smoke \
  --routes mamba3,m2rnn,hybrid-aemr \
  --compile-modes eager \
  --dtype float32 \
  --warmup-steps 1 \
  --steps 3


## Results

| Route       | Resolved model route | Route symbols | Backend           | Backend summary                    | Tokens/s | Mean step ms | Peak bytes |
| ----------- | -------------------- | ------------- | ----------------- | ---------------------------------- | -------: | -----------: | ---------: |
| mamba3      | hybrid-m             | M             | mlx               | mamba3:1                           |  2557.32 |       3.1283 |     116544 |
| m2rnn       | hybrid-r             | R             | mlx               | m2rnn:1                            |  3265.90 |       2.4496 |     200680 |
| hybrid-aemr | hybrid               | AEMR          | mlx+mlx.fast.sdpa | attention:1,m2rnn:1,mamba3:1,moe:1 |  1339.83 |       5.9709 |     247478 |

The fresh local receipt above ran the named alias rows for Mamba3, M2RNN, and
the full AEMR hybrid route. mamba3 is the machine-readable alias for resolved
MLX model route hybrid-m; m2rnn is the alias for hybrid-r; hybrid-aemr
is the alias for the full resolved route hybrid. All measured rows reported
device name Apple M4 Max.

These numbers are short-run local smoke measurements. Do not compare them to
GB10 rows unless both sides were collected with identical
comparison_key.workload and comparison_key.software values. That guard must
cover the route workload and the software stack, including framework/backend,
Python, platform, MLX, MLX-LM, and MLX-Metal metadata.
Matrix receipts are explicit about that limitation: summary rows, per-case
rows, matched_run, and bench_receipt carry receipt_scope: local_only,
local_only: true, and gb10_parity_claim: false until a matched GB10 row is
present.

The comparison parser also accepts receipt-only exports where the workload,
software, and timing fields live under bench_receipt. Those rows still require
identical nested workload and software keys before a ratio is emitted. In
particular, parquet data labels such as parquet_clang_v10_code must stay in
the workload key and must not be compared to synthetic_tokens smoke rows.
If a copied row contains multiple modern key sources, such as top-level
comparison_key plus bench_receipt.comparison_key, the sources must agree
inside the row. Conflicting row-local key sources are refusal evidence, not a
tie-breaker for producing ratios.
When forwarding Mamba3/M2RNN rows for GB10 comparison, run
scripts/compare_bench_rows.py --package-dir ... and archive the package.
matched_comparisons.jsonl is the only ratio-bearing artifact; refused pairs are
kept separately as mismatch evidence.

## Training Smoke Receipts

The tiny local trainer was checked against the MLX-LM train-step pattern of
nn.value_and_grad, optimizer update, mx.eval, and a compiled state capture
containing model.state, optimizer.state, and mx.random.state. The local
CompiledPretrainingStep keeps the same state capture and normalizes optional
side channels to a stable batch key set before mx.compile.

Common tiny Mamba3 flags:

bash
--batch-size 1 --seq-len 4 --steps 1 --hidden-size 8 \
  --num-attention-heads 1 --pattern M --depth 1 \
  --mamba-expand 1 --mamba-head-dim 4 --mamba-state-dim 4 \
  --mamba-groups 1 --mamba-chunk-size 4


Common tiny M2RNN flags:

bash
--batch-size 1 --seq-len 4 --steps 1 --hidden-size 8 \
  --num-attention-heads 1 --pattern R --depth 1 \
  --m2rnn-k-head-dim 2 --m2rnn-v-head-dim 2 \
  --m2rnn-num-v-heads 1 --m2rnn-num-f-heads 1 \
  --m2rnn-chunk-size 4


Commands run on 2026-04-30:

bash
./.venv/bin/python scripts/train_hybrid_tiny.py --json \
  --batch-size 1 --seq-len 4 --steps 1 --hidden-size 8 \
  --num-attention-heads 1 --pattern M --depth 1 \
  --mamba-expand 1 --mamba-head-dim 4 --mamba-state-dim 4 \
  --mamba-groups 1 --mamba-chunk-size 4


Result: PASS, route M, backend mamba3, eager, finite loss 3.313008,
trained tokens 3.

bash
./.venv/bin/python scripts/train_hybrid_tiny.py --json \
  --batch-size 1 --seq-len 4 --steps 1 --hidden-size 8 \
  --num-attention-heads 1 --pattern M --depth 1 \
  --mamba-expand 1 --mamba-head-dim 4 --mamba-state-dim 4 \
  --mamba-groups 1 --mamba-chunk-size 4 --compile


Result: PASS, route M, backend mamba3, compiled, finite loss 3.313008,
trained tokens 3.

bash
./.venv/bin/python scripts/train_hybrid_tiny.py --json \
  --batch-size 1 --seq-len 4 --steps 1 --hidden-size 8 \
  --num-attention-heads 1 --pattern R --depth 1 \
  --m2rnn-k-head-dim 2 --m2rnn-v-head-dim 2 \
  --m2rnn-num-v-heads 1 --m2rnn-num-f-heads 1 \
  --m2rnn-chunk-size 4


Result: PASS, route R, backend m2rnn, eager, finite loss 4.126661,
trained tokens 3.

bash
./.venv/bin/python scripts/train_hybrid_tiny.py --json \
  --batch-size 1 --seq-len 4 --steps 1 --hidden-size 8 \
  --num-attention-heads 1 --pattern R --depth 1 \
  --m2rnn-k-head-dim 2 --m2rnn-v-head-dim 2 \
  --m2rnn-num-v-heads 1 --m2rnn-num-f-heads 1 \
  --m2rnn-chunk-size 4 --compile


Result: PASS, route R, backend m2rnn, compiled, finite loss 4.126661,
trained tokens 3.

Fresh local train-smoke receipts were collected on 2026-05-02 after adding the
trainer memory receipt. These are two-step synthetic-token runs on the local
Apple M4 Max with batch size 1 and sequence length 4. `peak_memory_reset=true`
means the MLX allocator peak counter was reset immediately before the measured
training call; memory columns are allocator bytes in active/cache/peak order.

| Route | Mode     | Result | Tokens/s | Mean step ms | Final loss | Memory before a/c/p | Memory after a/c/p | Peak bytes |
| ----- | -------- | ------ | -------: | -----------: | ---------: | ------------------- | ------------------ | ---------: |
| M     | eager    | ok     |   758.14 |       5.8177 |   3.992802 | 212/8/0             | 41272/74904/105544 |     105544 |
| M     | compiled | ok     |   210.13 |      17.9608 |   3.992802 | 212/8/0             | 41480/63892/94952  |      94952 |
| R     | eager    | ok     |   876.49 |       4.1838 |   3.309039 | 228/8/0             | 37824/81100/112716 |     112716 |
| R     | compiled | ok     |   273.31 |      13.0414 |   3.309039 | 228/8/0             | 37752/59172/84436  |      84436 |

The 2026-05-02 receipts are local route health checks only. They do not claim
compiled mode is faster at this tiny shape, do not characterize steady-state
throughput, and do not establish cross-host GB10 or CUDA/Megatron parity. The
listed runs used the default cache-clear cadence, so `clear_cache_every_steps`
was null and `clear_cache_event_count` was 0.

Checkpoint/resume was also run against an explicit /tmp NPZ shard with the
same route flags. For each of M and R, eager and compiled runs saved
checkpoint-000001, resumed from step 1, advanced to step 2, and wrote a
final checkpoint with trained tokens 6. The resume cursor loaded
global_batch_offset=1 and the final checkpoint wrote global_batch_offset=2,
so checkpoint metadata now tracks batches consumed after resume without
double-counting the restored cursor.

Checkpoint/resume output:

text
M eager PASS first_loss=3.313008 resume_loss=4.124283 start/end=1/2 trained=6 resume_cursor=1 final_cursor=2
M compile PASS first_loss=3.313008 resume_loss=4.124283 start/end=1/2 trained=6 resume_cursor=1 final_cursor=2
R eager PASS first_loss=4.126661 resume_loss=3.040913 start/end=1/2 trained=6 resume_cursor=1 final_cursor=2
R compile PASS first_loss=4.126661 resume_loss=3.040913 start/end=1/2 trained=6 resume_cursor=1 final_cursor=2


## Route Risks

- The smoke shape is intentionally tiny, so tokens/s values are dominated by
  launch/compile overhead and should not be used as throughput claims.
- The train-smoke memory receipt is MLX allocator telemetry only. Active, cache,
  and peak bytes are not system-wide resident memory and are not a hardware
  memory-limit proof.
- Mamba3 and M2RNN route correctness here means finite local MLX loss,
  optimizer update, metadata, and checkpoint cursor continuity; it is not
  numerical parity against the CUDA/Megatron implementation.
- Compiled training relies on stable batch shapes and side-channel key sets.
  Changing optional structure fields or sequence shapes inside a compiled run
  can trigger recompilation or expose unsupported shape paths.
- No current receipt proves M4 Max is faster than, or equal to, GB10. Require a
  matched-run matrix with identical comparison_key.workload and
  comparison_key.software before making that comparison.

## M2RNN Correctness Slice

The 2026-04-30 M2RNN local port slice keeps the MLX path as a
correctness-first training reference. It tightens direct scan invariants for
floating input dtypes, matching recurrence dtypes, explicit h0 dtype/shape,
and non-divisible head broadcasts. The chunk wrapper still preserves sequential
recurrence semantics and now has coverage for the production-shaped
n_q=1,n_k=1,n_v=4,n_w=1,n_f=1 broadcast case.

Wave 6 lane 3 reread the local MLX M2RNN implementation/tests against the
sibling cppmega Megatron sources
cppmega/megatron/{m2rnn_spec.py,m2rnn_chunk.py,m2rnn_triton.py} and the
docs/status/m2rnn*.md status notes. The local MLX block remains a projection
+ recurrence + gate/norm/output training reference; it does not claim to port
the source Triton kernel, broadcast-view optimization, or optional depthwise
q/k/v convolution. The hybrid-R regression now checks that a next-token LM loss
produces finite gradients and an AdamW update for the M2RNN input projection,
state transition, learned decay parameters, residual D, gate norm, and output
projection inside HybridTinyLM.

The local mixer also has a value-level regression for the source-compatible
output gate broadcast used by m2rnn_spec.py: g is flattened to
num_g_heads * v_head_dim, repeated along the final feature axis to
num_heads * v_head_dim, passed through SiLU, then applied before gate norm and
output projection. This intentionally differs from repeating full
v_head_dim head vectors along a head axis.

The M2RNN h0 seam is recurrence state only. It guarantees split equivalence
for m2rnn_scan and chunked_m2rnn_scan, where q/k/v/xf are already formed,
and for the lightweight mixer only when there is no causal-conv history to carry
(conv_kernel=1). With conv_kernel > 1, callers need the explicit
M2RNNMixerState path, which carries both recurrent h and projected q/k/v
causal-conv history for segmented continuation. Passing only h0 still does not
claim arbitrary hidden-state split equivalence when convolution history matters.

No fused Metal M2RNN kernel was added in this slice. MLX exposes custom Metal
kernels, but a training-safe recurrent kernel would need a matching backward
path/custom VJP before replacing the current nn.value_and_grad route.

## Mamba3 Correctness Slice

The 2026-04-30 Mamba3 local port slice keeps the packed Author Mamba3
projection contract [z | x | B | C | dd_dt | dd_A | trap | angles] intact and
uses a bounded source-order MLX diagonal recurrence. The helper caps internal
sub-chunks at 32 tokens to bound Python work, carries only the final state
between chunks, and preserves explicit h0 continuation semantics.

The 2026-05-01 continuation slice adds a batch-shaped local cache carrier for
the source Mamba3 (angle_dt, ssm, k, v) allocator contract. The MLX block can
validate those four tensors, seed continuation from cache.ssm, accumulate the
RoPE angle offset from cache.angle_dt, and return finite final angle_dt,
ssm, k, and v tensors with source-compatible shapes. This is deliberately
not a Megatron inference-cache integration and does not claim arbitrary
full-prompt versus split-prompt equality: the local reference still does not
carry a causal-conv tail or trapezoidal dt[t + 1] boundary lookahead in the
cache.

No fused Metal Mamba3 scan kernel was added in this slice. The installed local
MLX stack exposes mx.fast.metal_kernel and mx.custom_function but no
mx.scan; the repo kernel policy still requires a custom VJP before a custom
Metal recurrent kernel can enter the differentiated training graph.

## Factual Port Boundary

As of 2026-05-02, this repository has local MLX module surfaces at
`cppmega_mlx/nn/mamba3.py` and `cppmega_mlx/nn/m2rnn.py`, with targeted
regressions in `tests/test_mamba3.py` and `tests/test_m2rnn.py`. The source
contracts checked for this note are the sibling Megatron files
`cppmega/megatron/author_mamba3_spec.py`,
`cppmega/megatron/mamba3_te_mixer.py`, and `cppmega/megatron/m2rnn_spec.py`.
That means docs may describe a local correctness/reference port with smoke and
parity-style checks; they must not claim a fused Metal Mamba3 scan, a fused
Metal M2RNN recurrence, Megatron/CUDA numerical parity, distributed Megatron
integration, or M4-vs-GB10 speed parity.

## Regression Checks

bash
./.venv/bin/python -m pytest tests/test_train_hybrid_tiny_script.py -q


Result: 14 passed in 3.26s.

bash
./.venv/bin/python -m pytest tests/test_bench_matrix.py -q


Result: 8 passed in 1.60s.

bash
./.venv/bin/python -m pytest \
  tests/test_mamba3.py \
  tests/test_m2rnn.py \
  tests/test_hybrid_lm.py \
  tests/test_bench_matrix.py \
  -q


Result: 29 passed in 1.76s.

## Output Contract

The human benchmark matrix output must keep route/backend/device metadata
visible. For these rows, the important fields are:

- hybrid-m M mlx mamba3:1 "Apple M4 Max"
- hybrid-r R mlx m2rnn:1 "Apple M4 Max"
