// V7-F58 gallery sortable report — localStorage-backed cache of
// per-preset run statistics. Survives reload; the GalleryTab and
// any "refresh" path read/write through this hook.

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "vbgui_gallery_runs_v1";

export interface GalleryEntry {
  preset: string;
  bricks: number;
  params_M?: number;
  mem_MB?: number;
  last_loss?: number;
  last_step_ms?: number;
  run_at: number;
}

export type GalleryCache = Record<string, GalleryEntry>;

function safeRead(): GalleryCache {
  try {
    if (typeof localStorage === "undefined") return {};
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return typeof parsed === "object" && parsed !== null
      ? (parsed as GalleryCache)
      : {};
  } catch {
    return {};
  }
}

function safeWrite(cache: GalleryCache) {
  try {
    if (typeof localStorage === "undefined") return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cache));
  } catch {
    // Quota exhausted or storage disabled; ignore.
  }
}

export function useGalleryCache(): {
  cache: GalleryCache;
  upsert: (entry: GalleryEntry) => void;
  clear: () => void;
} {
  const [cache, setCache] = useState<GalleryCache>(() => safeRead());

  useEffect(() => { safeWrite(cache); }, [cache]);

  const upsert = useCallback((entry: GalleryEntry) => {
    setCache((prev) => ({ ...prev, [entry.preset]: entry }));
  }, []);

  const clear = useCallback(() => setCache({}), []);

  return { cache, upsert, clear };
}
