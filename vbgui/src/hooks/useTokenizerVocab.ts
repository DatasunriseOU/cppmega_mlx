/**
 * useTokenizerVocab — given a tokenizer source path, return the
 * loaded tokenizer's vocab_size (or null while loading / on error).
 *
 * Backed by tokenizer.encode_visualize which already reports
 * capabilities.vocab_size in its response. We probe with a tiny
 * canonical text so the call is cheap, and cache by source path so
 * switching back/forth doesn't refetch.
 */

import { useEffect, useState } from "react";
import type { RpcClient } from "@/lib/rpc";

const CACHE = new Map<string, number>();
const PROBE_TEXT = "hello world";


export function useTokenizerVocab(
  rpc: RpcClient | null,
  source: string | null | undefined,
): number | null {
  const [vocab, setVocab] = useState<number | null>(
    source && CACHE.has(source) ? CACHE.get(source) ?? null : null);

  useEffect(() => {
    if (!rpc || !source) { setVocab(null); return; }
    const cached = CACHE.get(source);
    if (cached !== undefined) { setVocab(cached); return; }
    let cancelled = false;
    (async () => {
      try {
        const r = await rpc.call<{
          capabilities?: { vocab_size?: number };
        }>("tokenizer.encode_visualize",
            { tokenizer_source: source, text: PROBE_TEXT });
        const v = r?.capabilities?.vocab_size;
        if (typeof v === "number" && v > 0) {
          CACHE.set(source, v);
          if (!cancelled) setVocab(v);
        }
      } catch { /* leave null — caller falls back to default */ }
    })();
    return () => { cancelled = true; };
  }, [rpc, source]);

  return vocab;
}
