// usePresets — fetches architectures.list_presets once on mount and
// caches in module scope so subsequent App remounts skip the round-trip.

import { useEffect, useState } from "react";
import type { RpcClient } from "@/lib/rpc";

// Fallback list (cppmega_v4.architectures.PRESETS keys at the time of
// writing) used when the RPC fetch fails or the backend is offline.
// This is intentionally redundant with the backend — the hook prefers
// the live result whenever it's available.
const FALLBACK_PRESETS: readonly string[] = [
  "arcee_trinity", "deepseek_v3", "deepseek_v4_flash",
  "gemma3_270m", "gemma3_27b", "gemma4", "gemma4_31b",
  "glm_45", "glm_45_air", "glm_47", "glm_5", "glm_51",
  "gpt_oss_120b", "gpt_oss_20b", "granite_4_1", "grok25",
  "intellect_3", "kimi_k2", "kimi_linear", "laguna_xs2",
  "ling25", "ling26", "llama3_2_1b", "llama3_2_3b", "llama3_8b",
  "llama4_maverick", "longcat", "mimo_v2_5", "mimo_v2_5_pro",
  "mimo_v2_flash", "minimax_m2", "minimax_m2_5", "minimax_m2_7",
  "mistral4", "mistral_small_3_1", "nanbeige_4_1", "nemotron3",
  "olmo2_7b", "olmo3_32b", "olmo3_7b", "phi4",
  "qwen3_235b_a22b", "qwen3_30b_a3b", "qwen3_6_27b",
  "qwen3_coder_flash", "qwen3_dense_0_6b", "qwen3_dense_32b",
  "qwen3_dense_4b", "qwen3_dense_8b", "qwen3_next",
  "sarvam_105b", "sarvam_30b", "smollm3", "step3_5_flash",
  "tencent_hy3", "tiny_aya", "zaya1",
];

let _cached: readonly string[] | null = null;

interface ListPresetsResult { presets: string[]; }

export function usePresets(
  rpc: RpcClient | null,
  /** V7-H47: bump this key to force a refetch (e.g. when the backend
   *  build_id changes — see useRpc.onBackendBuildId).  Each distinct
   *  string triggers exactly one refetch. */
  invalidationKey: string | null = null,
): readonly string[] {
  const [presets, setPresets] = useState<readonly string[]>(
    _cached ?? FALLBACK_PRESETS,
  );

  useEffect(() => {
    // V7-H47: when invalidationKey advances, drop the module cache so
    // the next mount/effect refires.
    if (invalidationKey) _cached = null;
    if (_cached || !rpc) return;
    let cancelled = false;
    rpc.call<ListPresetsResult>("architectures.list_presets", {})
       .then((r) => {
         if (cancelled) return;
         if (Array.isArray(r.presets) && r.presets.length > 0) {
           _cached = r.presets;
           setPresets(r.presets);
         }
       })
       .catch(() => { /* stay with fallback */ });
    return () => { cancelled = true; };
  }, [rpc, invalidationKey]);

  return presets;
}

/** Test-only — resets module cache. Not exported in production index. */
export function _clearPresetsCache(): void { _cached = null; }
