# Inference Modes

This document describes the local MLX inference surfaces that currently exist
in `cppmega_mlx.inference`. The scope is token-id inference on the local Mac
runtime. Text tokenization, OpenAI-compatible chat APIs, cloud serving, and
GB10/CUDA parity are separate tasks.

## Mode Matrix

| Mode | Public surface | Current status | Main knobs | Non-claims |
| --- | --- | --- | --- | --- |
| Eager full-prefix generation | `generate_tokens(` | Local greedy/sampling generation over full prefix every step | `max_new_tokens`, `temperature`, `top_k`, `top_p`, `rng_key`, `eos_token_id`, `model_kwargs` | No KV reuse, no paged serving |
| Contiguous KV generation | `generate_tokens_with_kv_cache(` | Attention-only local cache path with one MLX-LM cache per attention layer | cache shape, `num_layers`, `num_kv_heads`, `head_dim`, `max_seq_len`, sampling knobs | not model-integrated paged attention |
| Streaming chunks | `stream_generate_tokens(` | Yields token-id chunks for eager or contiguous-KV generation | `use_kv_cache`, cache shape, optional `decode_token` | Not an mlx-lm registry compatibility claim |
| Prompt cache reuse | `generate_tokens_with_prompt_cache(` | Reuses an attention-only contiguous prompt prefix via `PromptCacheEntry` | `build_prompt_cache`, suffix decode, sampling knobs | No SSM/recurrent/ngram prompt-state reuse |
| Vanilla speculative decode | `generate_tokens_speculative(` | Eager batch=1 draft-window verifier with Leviathan-style acceptance/rejection | target model, draft model, `draft_window`, sampling knobs | No KV/paged speculative serving |
| MTP self-speculative decode | `generate_tokens_mtp_self_speculative(` | Eager batch=1 path using attached `mtp_head` as draft source | `draft_window`, trained MTP depth, sampling knobs | No EAGLE-2/token-recycling claim |
| Local token-id API serving | `create_local_generation_app(` | Optional FastAPI app exposing `/health` and `/generate` over token IDs | caller-owned model, generation options, `model_kwargs_builder`, optional decoder | not an OpenAI-compatible API |
| q4 quality smoke | `scripts/bench_inference_quality.py` | Built-in ARC/MMLU/HumanEval-style token-id smoke harness over q4 linears | `--tasks-jsonl`, suites, q4 bits/group size | not a real ARC/MMLU/HumanEval leaderboard run |
| KV-q4 long-context smoke | `scripts/bench_inference_long_context.py` | Built-in NIAH/RULER-style token-id smoke harness over `QuantizedKVCache` | `--context-tokens`, suites, `--kv-bits`, `--kv-group-size`, `--quantized-kv-start` | not a real NIAH/RULER leaderboard run |

## Eager Full-Prefix

`generate_tokens(` is the lowest-assumption path. It calls the model on the
entire token prefix each step and then applies `sample_next_token`. It accepts
sequence-aligned `model_kwargs`, so side-channel tensors can be passed for the
prompt and zero-filled or repeated for generated positions according to the
generation helper rules.

Use this path when a model route is stateful, when the caller does not know KV
cache shape, or when text-only behavior is required as a baseline.

## Contiguous KV

`generate_tokens_with_kv_cache(` and `stream_generate_tokens(` with
`use_kv_cache=True` use `ContiguousKVCache`, a validated wrapper over MLX-LM
`KVCache` or `QuantizedKVCache` layers. The cache is compatible only with
attention route blocks. Hybrid routes that require SSM, recurrent, or ngram
state must stay on eager full-prefix generation until a separate state adapter
exists.

`QuantizedKVCache` is available by passing `quantized=True` plus `kv_bits` and
`kv_group_size` through the cache factory or streaming helper. The current
contiguous path supports all-KV-q4 cache construction. It is not mixed
bf16-to-q4 quantized_kv_start > 0 transition coverage.

## Prompt Cache

`build_prompt_cache` creates a `PromptCacheEntry` containing prompt token IDs,
a filled contiguous KV cache, and next-token logits. `generate_tokens_with_prompt_cache(`
can reuse that prefix for repeated prompts, then decode any suffix and new
tokens.

Prompt cache reuse is guarded by route safety checks. Attention-only models are
allowed. Mamba3, M2RNN, engram, and similar stateful roles fail closed because
their non-attention state is not represented by `PromptCacheEntry`.

## Speculative Decode

`generate_tokens_speculative(` accepts a separate draft model and verifies the
candidate window with the target model. `generate_tokens_mtp_self_speculative(`
uses the target model's attached MTP head as the draft source. Both are eager
batch=1 loops and reuse the same acceptance/rejection helper. They are useful
for validating local speculative mechanics before integrating cache-aware or
server-side paths.

EAGLE-2, token recycling, Medusa, Hydra, and paged speculative serving remain
pattern or future work until measured against these simpler local baselines.

## Serving

`create_local_generation_app(` builds an optional FastAPI app when FastAPI is
installed. It exposes:

- `/health`: basic local readiness.
- `/generate`: token-id prompt in, token IDs out.

The endpoint threads generation options into the local eager generator and
supports a caller-provided `model_kwargs_builder` for side-channel tensors. It
can also use a caller-provided token decoder to include generated text in the
response.

This is not an OpenAI-compatible API, not a tokenizer service, not a cloud
fleet endpoint, and not model-integrated paged attention.

## Benchmarks

`scripts/bench_inference_throughput.py` measures local smoke prefill/decode
throughput for Qwen3-4B-class and NAM56R-class route profiles without
allocating full multi-billion-parameter models.

`scripts/bench_inference_quality.py` quantizes eligible linears with the
repo-local q4 affine helper and evaluates token-id ARC/MMLU/HumanEval-style
smoke rows. External JSONL token-id tasks are supported, but the default output
is not a real ARC/MMLU/HumanEval leaderboard run.

`scripts/bench_inference_long_context.py` runs token-id NIAH/RULER-style smoke
rows on the local contiguous KV-q4 path. It records actual context length,
KV bits/group/start, exact-token-match metrics, timing, and memory-safety
metadata. The default output is not a real NIAH/RULER leaderboard run and not a
GB10 parity claim.

## Side Channels

Generation helpers accept side channels as caller-owned `model_kwargs`. The
serving adapter exposes `model_kwargs_builder` so an agent or app can attach
platform, syntax, structure, or other tensors when it has them. The model still
predicts next tokens; side channels condition hidden states where model
consumers are configured.

Inference enrichment from prompt text, platform context, parser adapters, or
project indexes is handled by `cppmega_mlx.inference.side_channels`. Missing
families must fail according to the configured fallback policy rather than
being silently fabricated.

## Non-Claims

- This is not an OpenAI-compatible API.
- This is not model-integrated paged attention.
- This is not a real ARC/MMLU/HumanEval leaderboard run.
- This is not a real NIAH/RULER leaderboard run.
- This is not a GB10 parity claim.
- This is not mixed bf16-to-q4 quantized_kv_start > 0 transition coverage.
