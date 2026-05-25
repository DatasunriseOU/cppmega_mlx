/**
 * V8-R09 HFQuickStartModal — opens from DataInspector, calls
 * data.hf_quickstart, and renders the resulting parquet path.
 *
 * Updated with a premium glassmorphic layout, a preset catalog dropdown,
 * tokenizer presets, and a Cache Manager dashboard.
 */

import { useEffect, useState } from "react";
import type { RpcClient } from "@/lib/rpc";

export interface HFQuickStartModalProps {
  rpc: RpcClient;
  open: boolean;
  onClose: () => void;
  onResult?: (parquetPath: string, nTokens: number) => void;
}

interface QuickStartResult {
  parquet_path: string;
  n_tokens_written: number;
  n_docs_seen: number;
  elapsed_ms: number;
}

export interface CacheItem {
  file_name: string;
  parquet_path: string;
  dataset_id: string;
  tokenizer: string;
  n_tokens: number;
  split: string;
  text_field: string;
  byte_size: number;
  n_docs: number;
  elapsed_ms: number;
  category: string;
}

export interface DatasetCatalogItem {
  id: string;
  name: string;
  category: string;
  description: string;
  default_text_field: string;
  default_split: string;
}

export function HFQuickStartModal({
  rpc, open, onClose, onResult,
}: HFQuickStartModalProps): JSX.Element | null {
  const [tab, setTab] = useState<"hf" | "github" | "cache">("hf");
  const [dataset, setDataset] = useState("HuggingFaceFW/fineweb-edu");
  const [tokenizer, setTokenizer] = useState("cppmega_v3");
  const [nTokens, setNTokens] = useState(8192);
  const [repoUrl, setRepoUrl] = useState("https://github.com/karpathy/nanochat");
  const [maxCommits, setMaxCommits] = useState(50);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<QuickStartResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // Advanced metadata & cached state
  const [catalog, setCatalog] = useState<DatasetCatalogItem[]>([]);
  const [cachedItems, setCachedItems] = useState<CacheItem[]>([]);
  const [loadingCache, setLoadingCache] = useState(false);

  // Reset state when the modal closes or opens
  useEffect(() => {
    if (!open) {
      setBusy(false);
      setResult(null);
      setErr(null);
    } else {
      void fetchCatalog();
      void fetchCache();
    }
  }, [open]);

  // Refetch cache when switching to cache tab
  useEffect(() => {
    if (tab === "cache" && open) {
      void fetchCache();
    }
  }, [tab, open]);

  async function fetchCatalog() {
    try {
      const res = await rpc.call<{ catalog: DatasetCatalogItem[] }>("data.list_dataset_catalog", {});
      setCatalog(res.catalog);
    } catch (e) {
      console.error("Failed to load dataset catalog:", e);
    }
  }

  async function fetchCache() {
    setLoadingCache(true);
    try {
      const res = await rpc.call<{ items: CacheItem[] }>("data.list_cache", {});
      setCachedItems(res.items);
    } catch (e) {
      console.error("Failed to load dataset cache:", e);
    } finally {
      setLoadingCache(false);
    }
  }

  async function deleteCacheItem(fileName: string) {
    try {
      await rpc.call("data.clear_cache", { file_name: fileName });
      await fetchCache();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  async function run() {
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const jobId = `${tab}-${Date.now()}`;
      if (tab === "hf") {
        // Resolve split and text_field from catalog if possible
        const matched = catalog.find(item => item.id === dataset);
        const split = matched?.default_split || "train";
        const text_field = matched?.default_text_field || "text";

        const r = await rpc.call<QuickStartResult>(
          "data.hf_quickstart",
          {
            dataset_id: dataset,
            tokenizer: tokenizer,
            n_tokens: nTokens,
            job_id: jobId,
            split: split,
            text_field: text_field
          }
        );
        setResult(r);
        onResult?.(r.parquet_path, r.n_tokens_written);
        // Refresh cache catalog
        void fetchCache();
      } else if (tab === "github") {
        const r = await rpc.call<QuickStartResult>(
          "data.github_corpus",
          {
            repo_url: repoUrl,
            max_tokens: nTokens,
            max_commits: maxCommits,
            job_id: jobId,
            use_treesitter: true,
            use_clang: false
          }
        );
        setResult(r);
        onResult?.(r.parquet_path, r.n_tokens_written);
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  if (!open) return null;

  // Custom Category Badges
  const getBadgeStyle = (category: string) => {
    switch (category) {
      case "Pre-training":
        return { bg: "rgba(59, 130, 246, 0.15)", text: "#60a5fa", border: "1px solid rgba(59, 130, 246, 0.3)" };
      case "SFT (Instruction)":
        return { bg: "rgba(139, 92, 246, 0.15)", text: "#a78bfa", border: "1px solid rgba(139, 92, 246, 0.3)" };
      case "Math & Reasoning":
        return { bg: "rgba(245, 158, 11, 0.15)", text: "#fbbf24", border: "1px solid rgba(245, 158, 11, 0.3)" };
      case "SFT Alignment":
        return { bg: "rgba(236, 72, 153, 0.15)", text: "#f472b6", border: "1px solid rgba(236, 72, 153, 0.3)" };
      case "GitHub Code":
        return { bg: "rgba(16, 185, 129, 0.15)", text: "#34d399", border: "1px solid rgba(16, 185, 129, 0.3)" };
      default:
        return { bg: "rgba(107, 114, 128, 0.15)", text: "#9ca3af", border: "1px solid rgba(107, 114, 128, 0.3)" };
    }
  };

  return (
    <div data-testid="hf-quickstart-modal"
         style={{ position: "fixed", inset: 0, background: "rgba(2, 6, 23, 0.65)",
                  backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
                  display: "flex", alignItems: "center",
                  justifyContent: "center", zIndex: 9000 }}
         onClick={onClose}>
      <div data-testid="hf-quickstart-modal-content"
           onClick={(e) => e.stopPropagation()}
           style={{ background: "rgba(15, 23, 42, 0.85)",
                    border: "1px solid rgba(255, 255, 255, 0.08)",
                    boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.7)",
                    borderRadius: 12, padding: 24,
                    width: 580, maxHeight: "90vh", overflowY: "auto",
                    fontFamily: "system-ui, -apple-system, sans-serif",
                    color: "#f8fafc", fontSize: 13, display: "flex", flexDirection: "column",
                    gap: 16 }}>
        
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 12 }}>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600, letterSpacing: "-0.02em", color: "#f1f5f9" }}>
            ⚡ Persistent Dataset Quickstart & Cache Manager
          </h3>
          <span style={{ fontSize: 11, background: "rgba(255,255,255,0.05)", padding: "2px 8px", borderRadius: 4, color: "#94a3b8" }}>
            Cache HIT is 0.0s
          </span>
        </div>

        {/* Tab Selector */}
        <nav style={{ display: "flex", gap: 4, background: "rgba(0,0,0,0.25)", padding: 3, borderRadius: 8 }}>
          <button data-testid="hf-quickstart-tab"
                  onClick={() => setTab("hf")} disabled={busy}
                  style={{
                    flex: 1, padding: "6px 12px", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 12, fontWeight: 500,
                    transition: "all 0.2s ease",
                    background: tab === "hf" ? "rgba(255,255,255,0.08)" : "transparent",
                    color: tab === "hf" ? "#ffffff" : "#94a3b8"
                  }}>
            📂 HF Catalog & Ingestion
          </button>
          <button data-testid="github-corpus-tab"
                  onClick={() => setTab("github")} disabled={busy}
                  style={{
                    flex: 1, padding: "6px 12px", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 12, fontWeight: 500,
                    transition: "all 0.2s ease",
                    background: tab === "github" ? "rgba(255,255,255,0.08)" : "transparent",
                    color: tab === "github" ? "#ffffff" : "#94a3b8"
                  }}>
            💻 GitHub Ingest
          </button>
          <button data-testid="cache-manager-tab"
                  onClick={() => setTab("cache")} disabled={busy}
                  style={{
                    flex: 1, padding: "6px 12px", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 12, fontWeight: 500,
                    transition: "all 0.2s ease",
                    background: tab === "cache" ? "rgba(255,255,255,0.08)" : "transparent",
                    color: tab === "cache" ? "#ffffff" : "#94a3b8"
                  }}>
            ⚡ Cache Manager ({cachedItems.length})
          </button>
        </nav>

        {/* Ingest Data Tab */}
        {tab === "hf" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <label style={{ display: "flex", flexDirection: "column", gap: 4, color: "#cbd5e1" }}>
              Preset Dataset Catalog
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                disabled={busy}
                style={{
                  background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 6, padding: "8px 12px", color: "#f8fafc", outline: "none", cursor: "pointer"
                }}
              >
                {catalog.map((item) => (
                  <option key={item.id} value={item.id}>
                    [{item.category}] {item.name}
                  </option>
                ))}
                <option value="custom">-- Custom Dataset ID --</option>
              </select>
            </label>

            {dataset === "custom" ? (
              <label style={{ display: "flex", flexDirection: "column", gap: 4, color: "#cbd5e1" }}>
                Custom Dataset ID
                <input data-testid="hf-quickstart-dataset-id"
                       value={dataset === "custom" ? "" : dataset}
                       placeholder="e.g. HuggingFaceFW/fineweb-edu"
                       onChange={(e) => setDataset(e.target.value)}
                       disabled={busy}
                       style={{
                         background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)",
                         borderRadius: 6, padding: "8px 12px", color: "#f8fafc", outline: "none"
                       }} />
              </label>
            ) : (
              <div style={{ padding: "8px 12px", background: "rgba(255,255,255,0.03)", borderRadius: 6, border: "1px dashed rgba(255,255,255,0.08)", fontSize: 12, color: "#94a3b8" }}>
                💡 {catalog.find(item => item.id === dataset)?.description}
              </div>
            )}

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 4, color: "#cbd5e1" }}>
                Tokenizer Source
                <select
                  value={tokenizer}
                  onChange={(e) => setTokenizer(e.target.value)}
                  disabled={busy}
                  style={{
                    background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 6, padding: "8px 12px", color: "#f8fafc", outline: "none", cursor: "pointer"
                  }}
                >
                  <option value="cppmega_v3">cppmega_v3 (Bundled 65K)</option>
                  <option value="gpt2">gpt2 (GPT-2 Classic / xLSTM)</option>
                  <option value="meta-llama/Meta-Llama-3-8B">meta-llama/Meta-Llama-3-8B (Llama 3 / 3.2 / 4)</option>
                  <option value="deepseek-ai/DeepSeek-V3">deepseek-ai/DeepSeek-V3 (DeepSeek V3 / V4 Flash)</option>
                  <option value="Qwen/Qwen2.5-7B">Qwen/Qwen2.5-7B (Qwen3 Next / Qwen 2.5)</option>
                  <option value="google/gemma-2-9b">google/gemma-2-9b (Gemma 3 / Gemma 4)</option>
                  <option value="mistralai/Mistral-7B-v0.1">mistralai/Mistral-7B-v0.1 (Mistral / Mixtral)</option>
                  <option value="nvidia/Nemotron-3-8B">nvidia/Nemotron-3-8B (Nemotron 3)</option>
                  <option value="allenai/OLMo-1.7-7B">allenai/OLMo-1.7-7B (OLMo)</option>
                  <option value="microsoft/phi-4">microsoft/phi-4 (Phi 4)</option>
                  <option value="THUDM/glm-4-9b-chat">THUDM/glm-4-9b-chat (GLM 4 / GLM 5)</option>
                  <option value="ibm-granite/granite-3.0-8b-instruct">ibm-granite/granite-3.0-8b-instruct (Granite 4)</option>
                  <option value="HuggingFaceTB/SmolLM2-1.7B">HuggingFaceTB/SmolLM2-1.7B (SmolLM3)</option>
                  <option value="CohereForAI/aya-expanse-8b">CohereForAI/aya-expanse-8b (Tiny Aya)</option>
                  <option value="custom">-- Custom Tokenizer ID --</option>
                </select>
              </label>

              <label style={{ display: "flex", flexDirection: "column", gap: 4, color: "#cbd5e1" }}>
                n_tokens (Target Size)
                <input data-testid="hf-quickstart-n-tokens"
                       type="number" min={100} step={100}
                       value={nTokens}
                       onChange={(e) => setNTokens(Number(e.target.value))}
                       disabled={busy}
                       style={{
                         background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)",
                         borderRadius: 6, padding: "8px 12px", color: "#f8fafc", outline: "none"
                       }} />
              </label>
            </div>

            {tokenizer === "custom" && (
              <label style={{ display: "flex", flexDirection: "column", gap: 4, color: "#cbd5e1" }}>
                Custom Tokenizer Path or HuggingFace ID
                <input value={tokenizer === "custom" ? "" : tokenizer}
                       placeholder="e.g. meta-llama/Llama-3.2-3B-Instruct"
                       onChange={(e) => setTokenizer(e.target.value)}
                       disabled={busy}
                       style={{
                         background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)",
                         borderRadius: 6, padding: "8px 12px", color: "#f8fafc", outline: "none"
                       }} />
              </label>
            )}
          </div>
        )}

        {/* GitHub Ingest Tab */}
        {tab === "github" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <label style={{ display: "flex", flexDirection: "column", gap: 4, color: "#cbd5e1" }}>
              Repo URL
              <input data-testid="github-corpus-repo-url"
                     value={repoUrl}
                     onChange={(e) => setRepoUrl(e.target.value)}
                     disabled={busy}
                     style={{
                       background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)",
                       borderRadius: 6, padding: "8px 12px", color: "#f8fafc", outline: "none"
                     }} />
            </label>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 4, color: "#cbd5e1" }}>
                Max Commits
                <input data-testid="github-corpus-max-commits"
                       type="number" min={1} step={1}
                       value={maxCommits}
                       onChange={(e) => setMaxCommits(Number(e.target.value))}
                       disabled={busy}
                       style={{
                         background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)",
                         borderRadius: 6, padding: "8px 12px", color: "#f8fafc", outline: "none"
                       }} />
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 4, color: "#cbd5e1" }}>
                n_tokens (Target Limit)
                <input data-testid="hf-quickstart-n-tokens"
                       type="number" min={1} step={1}
                       value={nTokens}
                       onChange={(e) => setNTokens(Number(e.target.value))}
                       disabled={busy}
                       style={{
                         background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)",
                         borderRadius: 6, padding: "8px 12px", color: "#f8fafc", outline: "none"
                       }} />
              </label>
            </div>
          </div>
        )}

        {/* Cache Manager Tab */}
        {tab === "cache" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 12, color: "#94a3b8" }}>
                📂 Cache Location: <code style={{ color: "#38bdf8", background: "rgba(0,0,0,0.2)", padding: "2px 6px", borderRadius: 4 }}>data/cache/datasets/</code>
              </span>
              <button
                onClick={() => deleteCacheItem("")}
                disabled={cachedItems.length === 0}
                style={{
                  background: "rgba(239, 68, 68, 0.15)", color: "#f87171", border: "1px solid rgba(239, 68, 68, 0.3)",
                  padding: "4px 10px", borderRadius: 6, cursor: "pointer", fontSize: 11, fontWeight: 500, transition: "all 0.2s"
                }}
              >
                Clear All Cache
              </button>
            </div>

            {loadingCache ? (
              <div style={{ display: "flex", justifyContent: "center", padding: 24, color: "#94a3b8" }}>
                Scanning cache directory...
              </div>
            ) : cachedItems.length === 0 ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "40px 20px", background: "rgba(255,255,255,0.02)", borderRadius: 8, border: "1px dashed rgba(255,255,255,0.05)" }}>
                <span style={{ fontSize: 24, marginBottom: 8 }}>🫙</span>
                <span style={{ color: "#94a3b8", fontSize: 12 }}>No cached dataset files found. Ingest from HF to build cache.</span>
              </div>
            ) : (
              <div style={{ maxHeight: 260, overflowY: "auto", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8 }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, textAlign: "left" }}>
                  <thead>
                    <tr style={{ background: "rgba(255,255,255,0.03)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                      <th style={{ padding: "8px 12px", color: "#94a3b8", fontWeight: 500 }}>Dataset / Category</th>
                      <th style={{ padding: "8px 12px", color: "#94a3b8", fontWeight: 500 }}>Tokenizer</th>
                      <th style={{ padding: "8px 12px", color: "#94a3b8", fontWeight: 500 }}>Size / Tokens</th>
                      <th style={{ padding: "8px 12px", color: "#94a3b8", fontWeight: 500, textAlign: "right" }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cachedItems.map((item, idx) => {
                      const badge = getBadgeStyle(item.category);
                      return (
                        <tr key={idx} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)", transition: "background 0.2s", cursor: "default" }}>
                          <td style={{ padding: "10px 12px" }}>
                            <div style={{ fontWeight: 600, color: "#e2e8f0", marginBottom: 2 }}>{item.dataset_id}</div>
                            <span style={{
                              display: "inline-block", fontSize: 9, padding: "1px 5px", borderRadius: 4,
                              background: badge.bg, color: badge.text, border: badge.border, fontWeight: 600
                            }}>
                              {item.category}
                            </span>
                          </td>
                          <td style={{ padding: "10px 12px", color: "#cbd5e1" }}>
                            <code>{item.tokenizer}</code>
                          </td>
                          <td style={{ padding: "10px 12px" }}>
                            <div style={{ color: "#e2e8f0" }}>{(item.byte_size / 1024).toFixed(0)} KB</div>
                            <div style={{ color: "#64748b", fontSize: 10 }}>{item.n_tokens.toLocaleString()} tokens</div>
                          </td>
                          <td style={{ padding: "10px 12px", textAlign: "right" }}>
                            <div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
                              <button
                                onClick={() => {
                                  onResult?.(item.parquet_path, item.n_tokens);
                                  onClose();
                                }}
                                style={{
                                  background: "rgba(56, 189, 248, 0.15)", color: "#38bdf8", border: "1px solid rgba(56, 189, 248, 0.3)",
                                  padding: "3px 8px", borderRadius: 4, cursor: "pointer", fontSize: 10, fontWeight: 500
                                }}
                              >
                                Select
                              </button>
                              <button
                                onClick={() => void deleteCacheItem(item.file_name)}
                                style={{
                                  background: "rgba(248, 113, 113, 0.15)", color: "#f87171", border: "1px solid rgba(248, 113, 113, 0.3)",
                                  padding: "3px 8px", borderRadius: 4, cursor: "pointer", fontSize: 10, fontWeight: 500
                                }}
                              >
                                Delete
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Errors & Results */}
        {err && (
          <span data-testid="hf-quickstart-error"
                style={{ color: "#f87171", background: "rgba(239, 68, 68, 0.08)", padding: "8px 12px", borderRadius: 6, border: "1px solid rgba(239, 68, 68, 0.2)" }}>
            ⚠️ {err}
          </span>
        )}

        {result && (
          <div data-testid="hf-quickstart-result"
               style={{ background: "rgba(16, 185, 129, 0.08)", border: "1px solid rgba(16, 185, 129, 0.2)",
                        padding: 12, borderRadius: 8, color: "#34d399", display: "flex", flexDirection: "column", gap: 4 }}>
            <div data-testid="hf-quickstart-result-path" style={{ fontWeight: 600, fontSize: 12 }}>
              ✅ Saved Parquet: <span style={{ color: "#a7f3d0", fontFamily: "monospace" }}>{result.parquet_path}</span>
            </div>
            <div data-testid="hf-quickstart-result-tokens" style={{ fontSize: 11, color: "#a7f3d0" }}>
              ⚡ Ingested {result.n_tokens_written.toLocaleString()} tokens across {result.n_docs_seen} documents in {result.elapsed_ms.toFixed(0)} ms.
            </div>
          </div>
        )}

        {/* Footer Actions */}
        <div style={{ display: "flex", gap: 8, borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 16, justifyContent: "flex-end" }}>
          <button onClick={onClose} disabled={busy}
                  style={{
                    background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", color: "#94a3b8",
                    padding: "8px 16px", borderRadius: 6, cursor: "pointer", fontSize: 12, transition: "all 0.2s"
                  }}>
            Cancel
          </button>
          
          {tab !== "cache" && (
            <button data-testid={tab === "hf" ? "hf-quickstart-run" : "github-corpus-run"}
                    onClick={() => { void run(); }}
                    disabled={busy || (tab === "hf" && !dataset) || (tab === "github" && !repoUrl)}
                    style={{
                      background: "linear-gradient(135deg, #0284c7 0%, #0369a1 100%)", border: "none", color: "#ffffff",
                      padding: "8px 20px", borderRadius: 6, cursor: "pointer", fontSize: 12, fontWeight: 600, transition: "all 0.2s",
                      boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.2)", opacity: busy ? 0.7 : 1
                    }}>
              {busy ? "⚡ Processing..." : "🚀 Run Ingestion"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
