// useCatalog — lazy fetch + per-session memo for catalog.explain.

import { useEffect, useState } from "react";
import type { RpcClient } from "@/lib/rpc";

export interface ExplainEntryClient {
  category: string;
  name: string;
  summary: string;
  when_to_use: string;
  when_to_avoid: string;
  recommended_params: Record<string, unknown>;
  paper_ref: string | null;
  paper_url: string | null;
  gotchas: string[];
}

export interface ExplainResult {
  entry: ExplainEntryClient | null;
  not_found_message: string | null;
}

// Module-level memo so the cache survives component remount.
const _memo = new Map<string, Promise<ExplainResult>>();

function key(category: string, name: string): string {
  return `${category}::${name}`;
}

export function fetchExplain(
  rpc: RpcClient,
  category: string,
  name: string,
): Promise<ExplainResult> {
  const k = key(category, name);
  const cached = _memo.get(k);
  if (cached) return cached;
  const p = rpc
    .call<ExplainResult>("catalog.explain", { category, name })
    .catch((e) => ({
      entry: null,
      not_found_message: `RPC error: ${String(e)}`,
    } as ExplainResult));
  _memo.set(k, p);
  return p;
}

/** Hook returning current state for one option. Triggers fetch on
 *  first mount/category/name change. */
export function useCatalog(
  rpc: RpcClient | null,
  category: string,
  name: string | null,
  enabled = true,
): { entry: ExplainEntryClient | null; loading: boolean;
     error: string | null } {
  const [entry, setEntry] = useState<ExplainEntryClient | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled || !rpc || !name) {
      setEntry(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    fetchExplain(rpc, category, name).then((r) => {
      if (cancelled) return;
      setLoading(false);
      if (r.entry) {
        setEntry(r.entry);
        setError(null);
      } else {
        setEntry(null);
        setError(r.not_found_message);
      }
    });
    return () => { cancelled = true; };
  }, [rpc, category, name, enabled]);

  return { entry, loading, error };
}

/** Test-only — clears memo. Not exported in production index. */
export function _clearCatalogMemo(): void {
  _memo.clear();
}
