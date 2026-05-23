// V7-Q03.2: CheckpointHistoryDropdown — operator-facing list of past
// checkpoints in a directory. Replaces copy-pasting paths into
// train-checkpoint-load-path. Driven by the ckpt.list_history RPC.
//
// Closes Lane 6 audit gap from docs/UI-TO-TRAIN-AUDIT-2026-05-23.md.

import { useCallback, useState } from "react";
import type { RpcClient } from "@/lib/rpc";

export interface CkptHistoryEntry {
  path: string;
  mtime: number;
  size_bytes: number;
  arch_hash?: string | null;
  opt_kind?: string | null;
  global_step?: number | null;
  has_opt_sidecar?: boolean;
}

interface CkptHistoryResult {
  directory: string;
  scanned: number;
  entries: CkptHistoryEntry[];
  error?: string | null;
}

interface Props {
  rpc: RpcClient | null;
  directory: string;
  onSelect: (path: string) => void;
}

function basename(p: string): string {
  const parts = p.split(/[\\/]/);
  return parts[parts.length - 1] || p;
}

function fmtMtime(mtime: number): string {
  if (!mtime || !isFinite(mtime)) return "?";
  const d = new Date(mtime * 1000);
  return d.toISOString().replace("T", " ").slice(0, 16);
}

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
}

export function CheckpointHistoryDropdown({
  rpc, directory, onSelect,
}: Props) {
  const [open, setOpen] = useState(false);
  const [entries, setEntries] = useState<CkptHistoryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!rpc) return;
    setLoading(true);
    setError(null);
    try {
      const res = await rpc.call<CkptHistoryResult>(
        "ckpt.list_history",
        { directory: directory || "." },
      );
      if (res.error) {
        setError(res.error);
        setEntries([]);
      } else {
        setEntries(res.entries || []);
      }
    } catch (exc) {
      setError(String(exc));
      setEntries([]);
    } finally {
      setLoading(false);
    }
  }, [rpc, directory]);

  return (
    <div data-testid="ckpt-history-dropdown" style={{ position: "relative" }}>
      <button
        data-testid="ckpt-history-toggle"
        type="button"
        disabled={!rpc}
        onClick={() => {
          const next = !open;
          setOpen(next);
          if (next) void refresh();
        }}
        style={{
          fontSize: 10, padding: "2px 6px",
          background: "#f3f4f6", border: "1px solid #d1d5db",
          borderRadius: 3, cursor: rpc ? "pointer" : "not-allowed",
        }}
      >
        history ▾
      </button>
      {open && (
        <div
          data-testid="ckpt-history-list"
          style={{
            position: "absolute", top: 22, left: 0, zIndex: 100,
            background: "white", border: "1px solid #d1d5db",
            borderRadius: 4, boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
            minWidth: 360, maxWidth: 480, maxHeight: 320, overflowY: "auto",
            fontSize: 10, fontFamily: "monospace",
          }}
        >
          {loading && (
            <div data-testid="ckpt-history-loading"
                 style={{ padding: 8, color: "#9ca3af" }}>
              scanning {directory || "."}…
            </div>
          )}
          {error && !loading && (
            <div data-testid="ckpt-history-error"
                 style={{ padding: 8, color: "#dc2626" }}>
              {error}
            </div>
          )}
          {!loading && !error && entries.length === 0 && (
            <div data-testid="ckpt-history-empty"
                 style={{ padding: 8, color: "#9ca3af" }}>
              no .safetensors files in {directory || "."}
            </div>
          )}
          {!loading && entries.length > 0 && (
            <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
              {entries.map((e) => (
                <li
                  key={e.path}
                  data-testid={`ckpt-history-row`}
                  data-ckpt-path={e.path}
                  onClick={() => {
                    onSelect(e.path);
                    setOpen(false);
                  }}
                  style={{
                    padding: "4px 8px", cursor: "pointer",
                    borderBottom: "1px solid #f3f4f6",
                  }}
                  onMouseEnter={(ev) =>
                    (ev.currentTarget.style.background = "#eff6ff")}
                  onMouseLeave={(ev) =>
                    (ev.currentTarget.style.background = "transparent")}
                >
                  <div style={{ fontWeight: 600, color: "#1f2937" }}>
                    {basename(e.path)}
                    {e.has_opt_sidecar && (
                      <span style={{ marginLeft: 6, color: "#059669" }}>
                        +opt
                      </span>
                    )}
                  </div>
                  <div style={{ color: "#6b7280" }}>
                    {e.arch_hash ? e.arch_hash.slice(0, 8) : "?"}
                    {" · "}
                    {e.opt_kind || "?"}
                    {" · step "}
                    {e.global_step ?? "?"}
                    {" · "}
                    {fmtMtime(e.mtime)}
                    {" · "}
                    {fmtSize(e.size_bytes)}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
