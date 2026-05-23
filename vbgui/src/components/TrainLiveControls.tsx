import { useState, useEffect, useCallback, useMemo } from "react";
import { HelpIcon } from "@/components/HelpIcon";
import type { RpcClient } from "@/lib/rpc";

export interface TrainLiveControlsProps {
  rpc: RpcClient | null;
  trainInFlight: boolean;
  activeRunId: string | null;
  onScheduleCheckpoint: (path: string) => void;
  activeLayoutState?: any;
  onLoadLayout?: (layout: any) => void;
}

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

function basename(p: string): string {
  const parts = p.split(/[\\/]/);
  return parts[parts.length - 1] || p;
}

function dirname(p: string): string {
  const parts = p.split(/[\\/]/);
  if (parts.length <= 1) return ".";
  return parts.slice(0, -1).join("/") || "/";
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

// ─────────────────────────────────────────────────────────────────────────────
// Directory Browser Modal Component (V4-F01 picker)
// ─────────────────────────────────────────────────────────────────────────────
export interface DirectoryBrowserModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
  rpc: RpcClient | null;
  initialDirectory: string;
}

export function DirectoryBrowserModal({
  isOpen,
  onClose,
  onSelect,
  rpc,
  initialDirectory,
}: DirectoryBrowserModalProps): JSX.Element | null {
  const [currentDir, setCurrentDir] = useState<string>(initialDirectory || "/tmp");
  const [subdirs, setSubdirs] = useState<string[]>([]);
  const [parentDir, setParentDir] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSubdirs = useCallback(async (dir: string) => {
    if (!rpc) return;
    setLoading(true);
    setError(null);
    try {
      const res = await rpc.call<{
        current: string;
        parent: string | null;
        subdirs: string[];
        error?: string | null;
      }>("ckpt.list_subdirs", { directory: dir });

      if (res.error) {
        setError(res.error);
        setSubdirs([]);
      } else {
        setCurrentDir(res.current);
        setParentDir(res.parent);
        setSubdirs(res.subdirs || []);
      }
    } catch (err) {
      setError(String(err));
      setSubdirs([]);
    } finally {
      setLoading(false);
    }
  }, [rpc]);

  useEffect(() => {
    if (isOpen) {
      void fetchSubdirs(initialDirectory || "/tmp");
    }
  }, [isOpen, initialDirectory, fetchSubdirs]);

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 3000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(15, 23, 42, 0.65)",
        backdropFilter: "blur(8px)",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <div
        style={{
          background: "#1e293b",
          border: "1px solid rgba(255, 255, 255, 0.15)",
          borderRadius: 12,
          width: 460,
          maxHeight: "80vh",
          display: "flex",
          flexDirection: "column",
          color: "#f8fafc",
          boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "12px 16px",
            borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            background: "#0f172a",
          }}
        >
          <span style={{ fontWeight: "bold", color: "#22d3ee", fontSize: 14 }}>
            📂 Browse Folder
          </span>
          <button
            onClick={onClose}
            style={{
              background: "transparent",
              border: "none",
              color: "#94a3b8",
              cursor: "pointer",
              fontSize: 16,
            }}
          >
            ✕
          </button>
        </div>

        {/* Current Path Input */}
        <div style={{ padding: "12px 16px 6px 16px", display: "flex", gap: 6, alignItems: "center" }}>
          <input
            type="text"
            value={currentDir}
            onChange={(e) => setCurrentDir(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                void fetchSubdirs(currentDir);
              }
            }}
            style={{
              flex: 1,
              background: "#0f172a",
              border: "1px solid #475569",
              borderRadius: 6,
              color: "#f8fafc",
              padding: "6px 10px",
              fontSize: 11,
              fontFamily: "monospace",
            }}
          />
          <button
            onClick={() => void fetchSubdirs(currentDir)}
            style={{
              background: "#334155",
              color: "white",
              border: "none",
              borderRadius: 6,
              padding: "6px 12px",
              cursor: "pointer",
              fontSize: 11,
              fontWeight: "bold",
            }}
          >
            Go
          </button>
        </div>

        {/* Directory Navigator List */}
        <div
          style={{
            flex: 1,
            padding: "8px 16px",
            overflowY: "auto",
            minHeight: 200,
            maxHeight: 320,
            display: "flex",
            flexDirection: "column",
            gap: 4,
          }}
        >
          {loading && <div style={{ color: "#94a3b8", fontSize: 11, padding: 8 }}>Loading subdirectories...</div>}
          {error && <div style={{ color: "#ef4444", fontSize: 11, padding: 8 }}>Error: {error}</div>}

          {!loading && !error && (
            <>
              {parentDir && (
                <div
                  onClick={() => void fetchSubdirs(parentDir)}
                  style={{
                    padding: "6px 8px",
                    borderRadius: 4,
                    cursor: "pointer",
                    background: "rgba(255, 255, 255, 0.03)",
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    fontSize: 11,
                    color: "#38bdf8",
                  }}
                  onMouseOver={(e) => (e.currentTarget.style.background = "rgba(255, 255, 255, 0.08)")}
                  onMouseOut={(e) => (e.currentTarget.style.background = "rgba(255, 255, 255, 0.03)")}
                >
                  <span>⬆️ .. (Parent Directory)</span>
                </div>
              )}

              {subdirs.length === 0 && !parentDir && (
                <div style={{ color: "#64748b", fontSize: 11, padding: 8, fontStyle: "italic" }}>
                  No subdirectories found.
                </div>
              )}

              {subdirs.map((subdir) => {
                const fullSubPath = currentDir === "/" ? `/${subdir}` : `${currentDir}/${subdir}`;
                return (
                  <div
                    key={subdir}
                    onClick={() => void fetchSubdirs(fullSubPath)}
                    style={{
                      padding: "6px 8px",
                      borderRadius: 4,
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      fontSize: 11,
                      color: "#e2e8f0",
                    }}
                    onMouseOver={(e) => (e.currentTarget.style.background = "rgba(255, 255, 255, 0.08)")}
                    onMouseOut={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    <span>📁 {subdir}</span>
                  </div>
                );
              })}
            </>
          )}
        </div>

        {/* Footer Actions */}
        <div
          style={{
            padding: "12px 16px",
            borderTop: "1px solid rgba(255, 255, 255, 0.1)",
            display: "flex",
            justifyContent: "flex-end",
            gap: 8,
            background: "#0f172a",
          }}
        >
          <button
            onClick={onClose}
            style={{
              background: "#334155",
              color: "#cbd5e1",
              border: "none",
              borderRadius: 6,
              padding: "6px 12px",
              cursor: "pointer",
              fontSize: 11,
              fontWeight: "bold",
            }}
          >
            Cancel
          </button>
          <button
            onClick={() => {
              onSelect(currentDir);
              onClose();
            }}
            style={{
              background: "#06b6d4",
              color: "white",
              border: "none",
              borderRadius: 6,
              padding: "6px 16px",
              cursor: "pointer",
              fontSize: 11,
              fontWeight: "bold",
              boxShadow: "0 2px 4px rgba(6, 182, 212, 0.3)",
            }}
          >
            Select Folder
          </button>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main TrainLiveControls Component
// ─────────────────────────────────────────────────────────────────────────────
export function TrainLiveControls({
  rpc,
  trainInFlight,
  activeRunId,
  onScheduleCheckpoint,
  activeLayoutState,
  onLoadLayout,
}: TrainLiveControlsProps): JSX.Element {
  // Load initial path from localStorage for memory of previous choice
  const [ckptPath, setCkptPath] = useState<string>(() => {
    return localStorage.getItem("vbgui_last_ckpt_path") || "/tmp/midrun.safetensors";
  });
  const [newLr, setNewLr] = useState<string>("");
  const [lrStatus, setLrStatus] = useState<string | null>(null);
  
  // File Tree browser state
  const [showTree, setShowTree] = useState(false);
  const [scanDir, setScanDir] = useState<string>(() => dirname(ckptPath) || "/tmp");
  const [treeEntries, setTreeEntries] = useState<CkptHistoryEntry[]>([]);
  const [loadingTree, setLoadingTree] = useState(false);
  const [treeError, setTreeError] = useState<string | null>(null);
  const [collapsedDirs, setCollapsedDirs] = useState<Set<string>>(new Set());

  // In-Browser Virtual Filesystem state
  const [isVirtualFs, setIsVirtualFs] = useState<boolean>(() => {
    return localStorage.getItem("vbgui_is_virtual_fs") === "true";
  });

  // Directory picker modal visibility
  const [showFolderPicker, setShowFolderPicker] = useState(false);

  const toggleVirtualFs = (checked: boolean) => {
    setIsVirtualFs(checked);
    localStorage.setItem("vbgui_is_virtual_fs", String(checked));
  };

  // Persistent memory of choice
  const updateCkptPath = (path: string) => {
    setCkptPath(path);
    localStorage.setItem("vbgui_last_ckpt_path", path);

    // If Virtual FS is active, instantly load the saved model layout on canvas
    if (isVirtualFs && onLoadLayout) {
      try {
        const raw = localStorage.getItem("vbgui_virtual_checkpoints_v1") || "[]";
        const vCkpts = JSON.parse(raw) as any[];
        const match = vCkpts.find((c) => c.path === path);
        if (match && match.layoutState) {
          onLoadLayout(match.layoutState);
        }
      } catch (e) {
        console.error("Failed to load virtual checkpoint layout:", e);
      }
    }
  };

  const scanDirectory = useCallback(async () => {
    if (isVirtualFs) {
      setLoadingTree(true);
      setTreeError(null);
      try {
        const raw = localStorage.getItem("vbgui_virtual_checkpoints_v1") || "[]";
        const vCkpts = JSON.parse(raw) as any[];
        
        const entries = vCkpts.map((c: any) => ({
          path: c.path,
          mtime: c.mtime,
          size_bytes: c.size_bytes || 12345,
          arch_hash: c.arch_hash || "virtual",
          opt_kind: c.opt_kind || "none",
          global_step: c.global_step ?? 0,
          has_opt_sidecar: c.has_opt_sidecar || false,
        }));
        
        // Filter checkpoints under scanDir recursively
        const filtered = entries.filter((e: any) => {
          const dir = dirname(e.path);
          return dir === scanDir || dir.startsWith(scanDir === "/" ? "/" : scanDir + "/");
        });
        
        setTreeEntries(filtered);
      } catch (exc) {
        setTreeError(String(exc));
        setTreeEntries([]);
      } finally {
        setLoadingTree(false);
      }
      return;
    }

    if (!rpc) return;
    setLoadingTree(true);
    setTreeError(null);
    try {
      const res = await rpc.call<CkptHistoryResult>(
        "ckpt.list_history",
        { directory: scanDir || "/tmp" },
      );
      if (res.error) {
        setTreeError(res.error);
        setTreeEntries([]);
      } else {
        setTreeEntries(res.entries || []);
      }
    } catch (exc) {
      setTreeError(String(exc));
      setTreeEntries([]);
    } finally {
      setLoadingTree(false);
    }
  }, [rpc, scanDir, isVirtualFs]);

  // Re-scan when tree viewer is toggled or scanDir / isVirtualFs changes
  useEffect(() => {
    if (showTree) {
      void scanDirectory();
    }
  }, [showTree, scanDirectory]);

  // Group files by parent directory recursively
  const groupedDirectories = useMemo(() => {
    const groups: Record<string, CkptHistoryEntry[]> = {};
    treeEntries.forEach((e) => {
      const dir = dirname(e.path);
      if (!groups[dir]) groups[dir] = [];
      groups[dir].push(e);
    });
    return groups;
  }, [treeEntries]);

  const toggleFolder = (dir: string) => {
    setCollapsedDirs((prev) => {
      const next = new Set(prev);
      if (next.has(dir)) {
        next.delete(dir);
      } else {
        next.add(dir);
      }
      return next;
    });
  };

  async function pushLr() {
    setLrStatus(null);
    if (!rpc || !activeRunId) {
      setLrStatus("no active run");
      return;
    }
    const parsed = parseFloat(newLr);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      setLrStatus("invalid lr");
      return;
    }
    try {
      await rpc.call<{ status: string }>(
        "pipeline.update_lr",
        { run_id: activeRunId, new_lr: parsed });
      setLrStatus(`lr → ${parsed}`);
    } catch (e) {
      setLrStatus(`error: ${(e as Error).message}`);
    }
  }

  return (
    <div
      data-testid="train-live-controls"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 6,
        padding: "8px 12px",
        background: "rgba(254, 249, 195, 0.95)",
        borderTop: "1px solid #facc15",
        fontSize: 12,
        fontFamily: "system-ui, sans-serif",
        boxShadow: "0 -4px 15px rgba(0, 0, 0, 0.05)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <strong style={{ color: "#0f172a" }}>Live</strong>
        <HelpIcon topic="train_live_controls" />

        <span
          data-testid="train-live-status"
          style={{
            color: trainInFlight ? "#92400e" : "#6b7280",
            fontWeight: "bold",
            display: "flex",
            alignItems: "center",
            gap: 4,
          }}
        >
          {trainInFlight ? "● train in flight" : "○ idle"}
        </span>

        {/* In-Browser Virtual FS Checkbox */}
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 4,
            cursor: "pointer",
            fontWeight: "bold",
            color: "#0f172a",
          }}
        >
          <input
            type="checkbox"
            checked={isVirtualFs}
            onChange={(e) => toggleVirtualFs(e.target.checked)}
            style={{ cursor: "pointer" }}
          />
          💾 In-Browser Virtual FS
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: 4, color: "#0f172a" }}>
          ckpt path
          <input
            data-testid="train-live-ckpt-path"
            type="text"
            value={ckptPath}
            onChange={(e) => updateCkptPath(e.target.value)}
            style={{
              padding: "3px 6px",
              border: "1px solid #cbd5e1",
              borderRadius: 4,
              width: 240,
              fontSize: 11,
              fontFamily: "monospace",
              color: "#0f172a",
              background: "#ffffff",
            }}
          />
        </label>

        {/* Directory Tree Toggle Button */}
        <button
          onClick={() => setShowTree(!showTree)}
          title="Browse directories and checkpoints"
          style={{
            padding: "3px 8px",
            background: showTree ? "#0891b2" : "#f1f5f9",
            color: showTree ? "white" : "#475569",
            border: "1px solid #cbd5e1",
            borderRadius: 4,
            cursor: "pointer",
            fontWeight: "bold",
            display: "flex",
            alignItems: "center",
            gap: 4,
            transition: "all 0.15s ease",
          }}
        >
          📂 Browse Tree
        </button>

        <button
          data-testid="train-live-trigger-ckpt"
          onClick={() => {
            if (!ckptPath) return;
            if (isVirtualFs) {
              try {
                const raw = localStorage.getItem("vbgui_virtual_checkpoints_v1") || "[]";
                const vCkpts = JSON.parse(raw) as any[];
                
                const step = activeLayoutState?.trainOptions?.num_steps || 0;
                const newCkpt = {
                  path: ckptPath,
                  mtime: Date.now() / 1000,
                  size_bytes: 54321,
                  arch_hash: activeLayoutState?.projectName || "virtual_model",
                  opt_kind: activeLayoutState?.trainOptions?.optimizer || "adamw",
                  global_step: step,
                  has_opt_sidecar: false,
                  layoutState: activeLayoutState,
                };
                
                const filtered = vCkpts.filter((c) => c.path !== ckptPath);
                filtered.push(newCkpt);
                localStorage.setItem("vbgui_virtual_checkpoints_v1", JSON.stringify(filtered));
                
                alert(`💾 Saved canvas layout checkpoint internally to Virtual FS at:\n${ckptPath}`);
                void scanDirectory();
              } catch (e) {
                alert(`Error saving virtual checkpoint: ${e}`);
              }
            } else {
              onScheduleCheckpoint(ckptPath);
            }
          }}
          disabled={!ckptPath}
          style={{
            padding: "3px 10px",
            background: ckptPath ? "#d97706" : "#e5e7eb",
            color: ckptPath ? "white" : "#9ca3af",
            border: "none",
            borderRadius: 4,
            cursor: ckptPath ? "pointer" : "default",
            fontWeight: "bold",
            boxShadow: ckptPath ? "0 2px 4px rgba(217, 119, 6, 0.2)" : "none",
          }}
        >
          Trigger checkpoint
        </button>

        <span style={{ color: "#cbd5e1" }}>|</span>

        <label style={{ display: "flex", alignItems: "center", gap: 4, color: "#0f172a" }}>
          live lr
          <input
            data-testid="train-live-new-lr"
            type="number"
            step="0.0001"
            min={0}
            value={newLr}
            placeholder="0.0003"
            onChange={(e) => setNewLr(e.target.value)}
            style={{
              padding: "3px 6px",
              border: "1px solid #cbd5e1",
              borderRadius: 4,
              width: 80,
              fontSize: 11,
              color: "#0f172a",
              background: "#ffffff",
            }}
          />
        </label>
        <button
          data-testid="train-live-apply-lr"
          onClick={pushLr}
          disabled={!newLr || !activeRunId}
          style={{
            padding: "3px 10px",
            background: (newLr && activeRunId) ? "#16a34a" : "#e5e7eb",
            color: (newLr && activeRunId) ? "white" : "#9ca3af",
            border: "none",
            borderRadius: 4,
            cursor: (newLr && activeRunId) ? "pointer" : "default",
            fontWeight: "bold",
          }}
        >
          Apply lr
        </button>
        {lrStatus && (
          <span
            data-testid="train-live-lr-status"
            style={{
              color: "#1e293b",
              fontWeight: 600,
              fontSize: 11,
            }}
          >
            {lrStatus}
          </span>
        )}
      </div>

      {/* Checkpoint Directory Tree Inspector Pane */}
      {showTree && (
        <div
          data-testid="ckpt-directory-tree"
          style={{
            background: "rgba(15, 23, 42, 0.95)",
            backdropFilter: "blur(12px)",
            border: "1px solid rgba(255, 255, 255, 0.15)",
            borderRadius: 8,
            padding: 12,
            marginTop: 4,
            color: "#e2e8f0",
            maxHeight: 280,
            overflowY: "auto",
            boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.4)",
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 13, fontWeight: "bold", color: "#22d3ee" }}>
                📂 Checkpoint Directory Explorer
              </span>
              <span style={{ fontSize: 10, color: "#64748b" }}>
                {isVirtualFs ? "(virtual filesystem)" : "(recursive scanning)"}
              </span>
            </div>
            
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input
                type="text"
                value={scanDir}
                onChange={(e) => setScanDir(e.target.value)}
                placeholder="Scan directory (e.g. /tmp)"
                style={{
                  background: "#1e293b",
                  border: "1px solid #475569",
                  borderRadius: 4,
                  color: "white",
                  padding: "2px 6px",
                  fontSize: 10,
                  width: 180,
                  fontFamily: "monospace",
                }}
              />
              
              {/* Directory picker browser trigger button */}
              {!isVirtualFs && (
                <button
                  onClick={() => setShowFolderPicker(true)}
                  title="Select directory using Folder Browser Picker"
                  style={{
                    background: "#334155",
                    border: "1px solid #475569",
                    borderRadius: 4,
                    color: "white",
                    padding: "2px 6px",
                    fontSize: 10,
                    cursor: "pointer",
                  }}
                >
                  📂
                </button>
              )}

              <button
                onClick={scanDirectory}
                disabled={loadingTree}
                style={{
                  background: "#334155",
                  border: "none",
                  borderRadius: 4,
                  color: "white",
                  padding: "2px 8px",
                  fontSize: 10,
                  cursor: "pointer",
                }}
              >
                {loadingTree ? "Scanning..." : "🔄 Rescan"}
              </button>
            </div>
          </div>

          {loadingTree && <div style={{ color: "#94a3b8", fontSize: 11 }}>Scanning {scanDir}...</div>}
          {treeError && <div style={{ color: "#ef4444", fontSize: 11 }}>Error: {treeError}</div>}

          {!loadingTree && !treeError && Object.keys(groupedDirectories).length === 0 && (
            <div style={{ color: "#94a3b8", fontSize: 11, fontStyle: "italic" }}>
              {isVirtualFs
                ? `No virtual checkpoints found under ${scanDir}.`
                : `No .safetensors files found recursively under ${scanDir}.`}
            </div>
          )}

          {!loadingTree && Object.keys(groupedDirectories).length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 8, paddingLeft: 4 }}>
              {Object.entries(groupedDirectories).map(([dir, files]) => {
                const isCollapsed = collapsedDirs.has(dir);
                return (
                  <div key={dir} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    {/* Directory Header Node */}
                    <div
                      onClick={() => toggleFolder(dir)}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        cursor: "pointer",
                        fontWeight: "bold",
                        fontSize: 11,
                        color: "#e2e8f0",
                        userSelect: "none",
                      }}
                    >
                      <span style={{ fontSize: 9, color: "#64748b" }}>
                        {isCollapsed ? "▶" : "▼"}
                      </span>
                      <span>📁 {dir}</span>
                      <span style={{ color: "#64748b", fontWeight: "normal", fontSize: 10 }}>
                        ({files.length} checkpoints)
                      </span>
                    </div>

                    {/* Files List Under Directory */}
                    {!isCollapsed && (
                      <div
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          gap: 3,
                          paddingLeft: 18,
                          borderLeft: "1px dashed #334155",
                        }}
                      >
                        {files.map((file) => {
                          const isSelected = ckptPath === file.path;
                          return (
                            <div
                              key={file.path}
                              onClick={() => updateCkptPath(file.path)}
                              style={{
                                padding: "4px 8px",
                                borderRadius: 4,
                                background: isSelected ? "rgba(6, 182, 212, 0.2)" : "transparent",
                                border: isSelected ? "1px solid rgba(6, 182, 212, 0.4)" : "1px solid transparent",
                                cursor: "pointer",
                                display: "flex",
                                flexDirection: "column",
                                gap: 2,
                                transition: "all 0.1s ease",
                              }}
                              onMouseOver={(e) => {
                                if (!isSelected) e.currentTarget.style.background = "#1e293b";
                              }}
                              onMouseOut={(e) => {
                                if (!isSelected) e.currentTarget.style.background = "transparent";
                              }}
                            >
                              <div style={{ display: "flex", alignItems: "center", justifyItems: "center", gap: 6 }}>
                                <span style={{ fontWeight: 600, color: isSelected ? "#22d3ee" : "#f1f5f9" }}>
                                  📄 {basename(file.path)}
                                </span>
                                {file.has_opt_sidecar && (
                                  <span
                                    title="Optimizer state sidecar present (+opt)"
                                    style={{
                                      background: "rgba(16, 185, 129, 0.2)",
                                      color: "#10b981",
                                      fontSize: 8,
                                      padding: "1px 4px",
                                      borderRadius: 3,
                                      fontWeight: "bold",
                                    }}
                                  >
                                    +opt
                                  </span>
                                )}
                              </div>
                              
                              {/* Metadata tags */}
                              <div
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: 8,
                                  fontSize: 10,
                                  color: "#94a3b8",
                                }}
                              >
                                <span>🎛️ Config: <strong style={{ color: "#cbd5e1" }}>{file.arch_hash ? file.arch_hash.slice(0, 8) : "?"}</strong></span>
                                <span>🕒 Step: <strong style={{ color: "#cbd5e1" }}>{file.global_step ?? "?"}</strong></span>
                                <span>🚀 Optim: <strong style={{ color: "#cbd5e1" }}>{file.opt_kind || "?"}</strong></span>
                                <span>📅 Date: <strong style={{ color: "#cbd5e1" }}>{fmtMtime(file.mtime)}</strong></span>
                                <span>⚖️ Size: <strong style={{ color: "#cbd5e1" }}>{fmtSize(file.size_bytes)}</strong></span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Directory picker browser modal */}
      <DirectoryBrowserModal
        isOpen={showFolderPicker}
        onClose={() => setShowFolderPicker(false)}
        onSelect={(path) => {
          setScanDir(path);
          // Automatic re-scan is handled by the useEffect watching scanDir
        }}
        rpc={rpc}
        initialDirectory={scanDir}
      />
    </div>
  );
}
