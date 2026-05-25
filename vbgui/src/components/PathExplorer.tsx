import { useEffect, useState } from "react";
import type { RpcClient } from "@/lib/rpc";
import { T } from "@/theme";

export interface PathExplorerProps {
  rpc: RpcClient | null;
  onSelect: (path: string, contentType: "text" | "code" | "parquet") => void;
  initialPath?: string;
}

interface FileItem {
  name: string;
  is_dir: boolean;
  size_bytes: number;
  extension: string;
}

interface AnalysisResult {
  path: string;
  content_type: string;
  lines: number;
  words: number;
  chars: number;
  file_count: number;
  recommendation: string;
  suggested_tokenizer: string;
}

export function PathExplorer({ rpc, onSelect, initialPath = "." }: PathExplorerProps): JSX.Element {
  const [currentPath, setCurrentPath] = useState<string>(initialPath);
  const [items, setItems] = useState<FileItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  const [contentType, setContentType] = useState<"text" | "code" | "parquet">("text");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [analyzing, setAnalyzing] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!rpc) return;
      setLoading(true);
      setError(null);
      setSelectedItem(null);
      setAnalysis(null);
      try {
        const res = await rpc.call<FileItem[]>("data.list_directory", { path: currentPath });
        if (!cancelled) {
          // Sort directories first, then files alphabetically
          const sorted = [...res].sort((a, b) => {
            if (a.is_dir && !b.is_dir) return -1;
            if (!a.is_dir && b.is_dir) return 1;
            return a.name.localeCompare(b.name);
          });
          setItems(sorted);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setItems([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [currentPath, rpc]);

  async function handleItemClick(item: FileItem) {
    if (item.is_dir) {
      const nextPath = currentPath === "." ? item.name : `${currentPath}/${item.name}`;
      setCurrentPath(nextPath);
    } else {
      setSelectedItem(item.name);
      setAnalysis(null);
      // Auto-detect content type based on extension
      const ext = item.extension.toLowerCase();
      let detectedType: "text" | "code" | "parquet" = "text";
      if ([".cpp", ".py", ".cu", ".h", ".c", ".rs", ".go", ".ts", ".tsx", ".js"].includes(ext)) {
        detectedType = "code";
      } else if (ext === ".parquet") {
        detectedType = "parquet";
      }
      setContentType(detectedType);
      const fullPath = currentPath === "." ? item.name : `${currentPath}/${item.name}`;
      onSelect(fullPath, detectedType);
    }
  }

  async function handleAnalyze() {
    if (!selectedItem || !rpc) return;
    setAnalyzing(true);
    setAnalysis(null);
    try {
      const fullPath = currentPath === "." ? selectedItem : `${currentPath}/${selectedItem}`;
      const res = await rpc.call<AnalysisResult>("data.analyze_source", {
        path: fullPath,
        content_type: contentType,
      });
      setAnalysis(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setAnalyzing(false);
    }
  }

  function handleGoUp() {
    if (currentPath === "." || !currentPath.includes("/")) {
      setCurrentPath(".");
    } else {
      const parts = currentPath.split("/");
      parts.pop();
      setCurrentPath(parts.join("/"));
    }
  }

  return (
    <div
      data-testid="path-explorer"
      className="glass-panel"
      style={{
        padding: 12,
        borderRadius: "var(--vb-radius-lg)",
        background: "var(--vb-surface-glass)",
        border: "1px solid var(--vb-border)",
        fontFamily: T.font,
        color: T.text,
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "space-between" }}>
        <strong style={{ fontSize: 11, textTransform: "uppercase", color: T.textSecondary }}>Path Browser</strong>
        <span style={{ fontSize: 10, fontFamily: T.fontMono, color: T.textMuted }}>{currentPath}</span>
      </div>

      {/* Navigation Breadcrumb */}
      <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
        <button
          onClick={handleGoUp}
          disabled={currentPath === "."}
          style={{ padding: "2px 6px", fontSize: 11 }}
        >
          parent ↰
        </button>
        <span style={{ fontSize: 11, color: T.textSecondary, overflowX: "auto", whiteSpace: "nowrap" }}>
          root / {currentPath.split("/").filter(Boolean).map((p, i) => <span key={i}> {p} /</span>)}
        </span>
      </div>

      {/* Explorer List Area */}
      <div
        style={{
          height: 140,
          overflowY: "auto",
          background: "var(--vb-surface-3)",
          border: "1px solid var(--vb-border)",
          borderRadius: "var(--vb-radius-sm)",
          padding: 4,
          display: "flex",
          flexDirection: "column",
          gap: 2,
        }}
      >
        {loading && <div style={{ fontSize: 11, color: T.textMuted, padding: 4 }}>Loading...</div>}
        {error && <div style={{ fontSize: 11, color: "var(--vb-danger)", padding: 4 }}>{error}</div>}
        {!loading && !error && items.length === 0 && (
          <div style={{ fontSize: 11, color: T.textMuted, padding: 4 }}>Empty directory</div>
        )}
        {!loading &&
          !error &&
          items.map((item, idx) => {
            const isSelected = selectedItem === item.name;
            return (
              <div
                key={idx}
                onClick={() => handleItemClick(item)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "4px 8px",
                  borderRadius: "var(--vb-radius-sm)",
                  background: isSelected ? "var(--vb-accent-soft)" : "transparent",
                  border: `1px solid ${isSelected ? "var(--vb-accent)" : "transparent"}`,
                  cursor: "pointer",
                  fontSize: 11.5,
                  userSelect: "none",
                }}
              >
                <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span>{item.is_dir ? "📁" : "📄"}</span>
                  <span style={{ color: item.is_dir ? "var(--vb-accent)" : T.text }}>{item.name}</span>
                </span>
                {!item.is_dir && (
                  <span style={{ fontSize: 9, color: T.textMuted, fontFamily: T.fontMono }}>
                    {(item.size_bytes / 1024).toFixed(1)} KB
                  </span>
                )}
              </div>
            );
          })}
      </div>

      {/* Type & Action Section */}
      {selectedItem && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "space-between" }}>
            <label style={{ fontSize: 11, color: T.textSecondary, display: "flex", gap: 4, alignItems: "center" }}>
              type
              <select
                value={contentType}
                onChange={(e) => setContentType(e.target.value as "text" | "code" | "parquet")}
                style={{
                  background: "var(--vb-surface-3)",
                  color: T.text,
                  border: "1px solid var(--vb-border)",
                  fontSize: 10,
                  padding: "1px 4px",
                }}
              >
                <option value="text">Plain Text</option>
                <option value="code">Code (C++ / Python)</option>
                <option value="parquet">Pre-tokenized Parquet</option>
              </select>
            </label>
            <button
              onClick={handleAnalyze}
              disabled={analyzing}
              style={{
                padding: "2px 8px",
                background: "var(--vb-accent)",
                color: "var(--vb-accent-contrast)",
                fontWeight: 600,
                fontSize: 11,
              }}
            >
              {analyzing ? "Analyzing…" : "Analyze"}
            </button>
          </div>

          {/* Diagnostic Recommendation Card */}
          {analysis && (
            <div
              className="glass-panel"
              style={{
                padding: 8,
                borderRadius: "var(--vb-radius-sm)",
                background: "rgba(34, 211, 238, 0.05)",
                border: "1px solid var(--vb-accent-soft)",
                fontSize: 11,
                lineHeight: 1.4,
              }}
            >
              <div style={{ fontWeight: 600, color: "var(--vb-accent)", marginBottom: 4 }}>
                ✓ Diagnostic Complete
              </div>
              <div style={{ color: T.textSecondary, fontSize: 10, display: "flex", flexWrap: "wrap", gap: "2px 8px", marginBottom: 4 }}>
                {analysis.file_count > 1 ? (
                  <span>Files: {analysis.file_count}</span>
                ) : (
                  <>
                    <span>Lines: {analysis.lines}</span>
                    <span>Words: {analysis.words}</span>
                  </>
                )}
              </div>
              <p style={{ margin: "0 0 6px 0", color: T.text }}>{analysis.recommendation}</p>
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <span
                  style={{
                    fontSize: 9,
                    fontWeight: 700,
                    padding: "1px 5px",
                    background: "var(--vb-accent-soft)",
                    color: "var(--vb-accent)",
                    borderRadius: 4,
                  }}
                >
                  Suggested: {analysis.suggested_tokenizer}
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
