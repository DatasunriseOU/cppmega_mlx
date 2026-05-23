// V7-I07: JsonRPC LRU cache hit-rate dashboard.
//
// Polls the FastAPI /cache/stats endpoint at a fixed cadence so the
// BottomStrip CacheStats chip can render hit-rate / size / evictions
// live. The endpoint is GET-only (not JSON-RPC) — see
// cppmega_v4/jsonrpc/server.py for the route.
//
// Returns `null` until the first successful fetch so the chip can
// render a "—" placeholder rather than zeros that look real.

import { useEffect, useRef, useState } from "react";

export interface CacheStats {
  size: number;
  capacity: number;
  hits: number;
  misses: number;
  evictions: number;
  hit_rate: number;
}

export interface UseCacheStatsOptions {
  baseUrl?: string;
  /** Polling cadence in ms. Default 1500 ms (1Hz-ish, matches the
   *  backend.status heartbeat cadence in BottomStrip). */
  intervalMs?: number;
  /** Inject fetch (used by unit tests). */
  fetchImpl?: typeof fetch;
}

const DEFAULT_BASE_URL = "http://127.0.0.1:8765";

export function useCacheStats(
  opts: UseCacheStatsOptions = {},
): CacheStats | null {
  const baseUrl = opts.baseUrl ?? DEFAULT_BASE_URL;
  const intervalMs = opts.intervalMs ?? 1500;
  const [stats, setStats] = useState<CacheStats | null>(null);
  const cancelledRef = useRef(false);

  useEffect(() => {
    cancelledRef.current = false;
    const f = opts.fetchImpl ?? fetch;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      try {
        const res = await f(`${baseUrl}/cache/stats`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const body = (await res.json()) as CacheStats;
        if (!cancelledRef.current) setStats(body);
      } catch {
        // Leave the previous stats in place; the BottomStrip chip
        // already shows the backend-status dot so a separate
        // "cache-stats unreachable" indicator would be noise.
      } finally {
        if (!cancelledRef.current) {
          timer = setTimeout(tick, intervalMs);
        }
      }
    };
    tick();
    return () => {
      cancelledRef.current = true;
      if (timer) clearTimeout(timer);
    };
  }, [baseUrl, intervalMs, opts.fetchImpl]);

  return stats;
}

/** Format a hit_rate (0..1) as a percentage with one decimal. */
export function formatHitRate(rate: number): string {
  if (!Number.isFinite(rate)) return "—";
  return `${(rate * 100).toFixed(1)}%`;
}
