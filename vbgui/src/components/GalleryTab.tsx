// V7-F58 per-preset sortable gallery. Renders a table of all available
// presets with cached run stats (bricks count, params_M, mem_MB,
// last_loss, last_step_ms). Click a column header to sort asc/desc.
// "Refresh" button per row runs a verify roundtrip to repopulate stats.

import { useMemo, useState } from "react";
import type { GalleryCache, GalleryEntry } from "@/hooks/useGalleryCache";

export type GalleryColumn =
  | "preset" | "bricks" | "params_M" | "mem_MB"
  | "last_loss" | "last_step_ms" | "run_at";

export interface GalleryTabProps {
  presets: readonly string[];
  cache: GalleryCache;
  onRefresh?: (preset: string) => Promise<void> | void;
  refreshing?: Set<string>;
}

const COLUMNS: { key: GalleryColumn; label: string; numeric: boolean }[] = [
  { key: "preset",         label: "Preset",       numeric: false },
  { key: "bricks",         label: "Bricks",       numeric: true  },
  { key: "params_M",       label: "Params (M)",   numeric: true  },
  { key: "mem_MB",         label: "Mem (MB)",     numeric: true  },
  { key: "last_loss",      label: "Last loss",    numeric: true  },
  { key: "last_step_ms",   label: "Step (ms)",    numeric: true  },
  { key: "run_at",         label: "Last run",     numeric: true  },
];

type SortDir = "asc" | "desc";

function compare(a: GalleryEntry, b: GalleryEntry,
                 col: GalleryColumn): number {
  if (col === "preset") return a.preset.localeCompare(b.preset);
  const av = (a as Record<GalleryColumn, unknown>)[col];
  const bv = (b as Record<GalleryColumn, unknown>)[col];
  const an = typeof av === "number" && Number.isFinite(av) ? av : null;
  const bn = typeof bv === "number" && Number.isFinite(bv) ? bv : null;
  // missing values sort last regardless of direction.
  if (an == null && bn == null) return 0;
  if (an == null) return 1;
  if (bn == null) return -1;
  return an - bn;
}

function fmt(v: unknown, col: GalleryColumn): string {
  if (v == null) return "—";
  if (col === "run_at" && typeof v === "number") {
    return new Date(v).toISOString().replace("T", " ").slice(0, 19);
  }
  if (typeof v === "number") {
    return Math.abs(v) >= 1000 || Number.isInteger(v)
      ? v.toFixed(0)
      : v.toFixed(3);
  }
  return String(v);
}

export function GalleryTab({
  presets, cache, onRefresh, refreshing,
}: GalleryTabProps): JSX.Element {
  const [sortCol, setSortCol] = useState<GalleryColumn>("preset");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const rows = useMemo<GalleryEntry[]>(() => {
    const entries = presets.map<GalleryEntry>((p) =>
      cache[p] ?? {
        preset: p, bricks: 0, run_at: 0,
      });
    // Partition first so rows with the sorted column missing always
    // sort to the bottom regardless of direction (otherwise reversing
    // an asc sort would put missing values at the top in desc).
    const isMissing = (e: GalleryEntry) => {
      if (sortCol === "preset") return false;
      const v = (e as Record<GalleryColumn, unknown>)[sortCol];
      return typeof v !== "number" || !Number.isFinite(v);
    };
    const known = entries.filter((e) => !isMissing(e));
    const missing = entries.filter(isMissing);
    known.sort((a, b) => compare(a, b, sortCol));
    if (sortDir === "desc") known.reverse();
    return [...known, ...missing];
  }, [presets, cache, sortCol, sortDir]);

  function toggleSort(col: GalleryColumn) {
    if (col === sortCol) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortCol(col);
      setSortDir("asc");
    }
  }

  return (
    <div data-testid="gallery-tab"
         style={{ padding: 12, fontFamily: "system-ui, sans-serif",
                  fontSize: 12, overflowY: "auto", flex: 1 }}>
      <h3 style={{ margin: "0 0 8px", fontSize: 14 }}>
        Per-preset gallery
      </h3>
      <p style={{ color: "#6b7280", marginTop: 0 }}>
        Click a column header to sort ascending/descending. "Refresh"
        runs a verify roundtrip and caches stats. Cache persists across
        reloads (localStorage <code>vbgui_gallery_runs_v1</code>).
      </p>
      <table data-testid="gallery-table"
             style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
            {COLUMNS.map((c) => {
              const ariaSort = sortCol === c.key
                ? (sortDir === "asc" ? "ascending" : "descending")
                : "none";
              return (
                <th key={c.key}
                    data-testid={`gallery-sort-${c.key}`}
                    aria-sort={ariaSort}
                    onClick={() => toggleSort(c.key)}
                    style={{ textAlign: c.numeric ? "right" : "left",
                             padding: "4px 6px", cursor: "pointer",
                             color: "#374151", fontWeight: 600,
                             userSelect: "none" }}>
                  {c.label}{" "}
                  {sortCol === c.key
                    ? (sortDir === "asc" ? "▲" : "▼")
                    : ""}
                </th>
              );
            })}
            <th style={{ padding: "4px 6px" }} />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.preset}
                data-testid={`gallery-row-${r.preset}`}
                style={{ borderBottom: "1px solid #f3f4f6" }}>
              {COLUMNS.map((c) => (
                <td key={c.key}
                    data-testid={`gallery-cell-${r.preset}-${c.key}`}
                    style={{ padding: "4px 6px",
                             textAlign: c.numeric ? "right" : "left",
                             color: "#374151" }}>
                  {c.key === "preset" ? r.preset
                    : fmt((r as Record<GalleryColumn, unknown>)[c.key],
                          c.key)}
                </td>
              ))}
              <td style={{ padding: "4px 6px" }}>
                {onRefresh && (
                  <button
                    data-testid={`gallery-refresh-${r.preset}`}
                    onClick={() => onRefresh(r.preset)}
                    disabled={refreshing?.has(r.preset)}
                    style={{ padding: "2px 8px" }}>
                    {refreshing?.has(r.preset) ? "…" : "Refresh"}
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
