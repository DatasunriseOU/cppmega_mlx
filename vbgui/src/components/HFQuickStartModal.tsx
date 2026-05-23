/**
 * V8-R09 HFQuickStartModal — opens from DataInspector, calls
 * data.hf_quickstart, and renders the resulting parquet path.
 *
 * Designed as a self-contained modal: parent sets open=true and gets
 * notified via onClose when the user dismisses. The parquet path
 * lands in props.onResult so the parent can re-point training at it.
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

export function HFQuickStartModal({
  rpc, open, onClose, onResult,
}: HFQuickStartModalProps): JSX.Element | null {
  const [tab, setTab] = useState<"hf" | "github">("hf");
  const [dataset, setDataset] = useState("HuggingFaceFW/fineweb-edu");
  const [nTokens, setNTokens] = useState(8192);
  const [repoUrl, setRepoUrl] = useState(
    "https://github.com/karpathy/nanochat");
  const [maxCommits, setMaxCommits] = useState(50);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<QuickStartResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // Reset state when the modal closes.
  useEffect(() => {
    if (!open) {
      setBusy(false);
      setResult(null);
      setErr(null);
    }
  }, [open]);

  if (!open) return null;

  async function run() {
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const jobId = `${tab}-${Date.now()}`;
      const r = tab === "hf"
        ? await rpc.call<QuickStartResult>(
            "data.hf_quickstart",
            { dataset_id: dataset, n_tokens: nTokens, job_id: jobId })
        : await rpc.call<QuickStartResult>(
            "data.github_corpus",
            { repo_url: repoUrl, max_tokens: nTokens,
              max_commits: maxCommits, job_id: jobId,
              use_treesitter: true, use_clang: false });
      setResult(r);
      onResult?.(r.parquet_path, r.n_tokens_written);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div data-testid="hf-quickstart-modal"
         style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.4)",
                  display: "flex", alignItems: "center",
                  justifyContent: "center", zIndex: 9000 }}
         onClick={onClose}>
      <div data-testid="hf-quickstart-modal-content"
           onClick={(e) => e.stopPropagation()}
           style={{ background: "#fff", borderRadius: 8, padding: 24,
                    width: 480, fontFamily: "system-ui, sans-serif",
                    fontSize: 13, display: "flex", flexDirection: "column",
                    gap: 8 }}>
        <h3 style={{ margin: 0 }}>Data quickstart</h3>
        <nav style={{ display: "flex", gap: 4, marginBottom: 4 }}>
          <button data-testid="hf-quickstart-tab"
                  onClick={() => setTab("hf")} disabled={busy}
                  style={{ background: tab === "hf" ? "#dbeafe"
                                                    : "transparent" }}>
            HF Hub
          </button>
          <button data-testid="github-corpus-tab"
                  onClick={() => setTab("github")} disabled={busy}
                  style={{ background: tab === "github" ? "#dbeafe"
                                                        : "transparent" }}>
            GitHub repo (tree-sitter / clang)
          </button>
        </nav>
        {tab === "hf" && (
          <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            Dataset ID
            <input data-testid="hf-quickstart-dataset-id"
                   value={dataset}
                   onChange={(e) => setDataset(e.target.value)}
                   disabled={busy} />
          </label>
        )}
        {tab === "github" && (
          <>
            <label style={{ display: "flex", flexDirection: "column",
                            gap: 2 }}>
              Repo URL
              <input data-testid="github-corpus-repo-url"
                     value={repoUrl}
                     onChange={(e) => setRepoUrl(e.target.value)}
                     disabled={busy} />
            </label>
            <label style={{ display: "flex", flexDirection: "column",
                            gap: 2 }}>
              max_commits
              <input data-testid="github-corpus-max-commits"
                     type="number" min={1} step={1}
                     value={maxCommits}
                     onChange={(e) => setMaxCommits(Number(e.target.value))}
                     disabled={busy} />
            </label>
          </>
        )}
        <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          n_tokens (target)
          <input data-testid="hf-quickstart-n-tokens"
                 type="number" min={1} step={1}
                 value={nTokens}
                 onChange={(e) => setNTokens(Number(e.target.value))}
                 disabled={busy} />
        </label>
        <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
          <button data-testid={tab === "hf"
                                ? "hf-quickstart-run"
                                : "github-corpus-run"}
                  onClick={() => { void run(); }}
                  disabled={busy
                    || (tab === "hf" && !dataset)
                    || (tab === "github" && !repoUrl)}>
            {busy ? "running…" : "Run"}
          </button>
          <button onClick={onClose} disabled={busy}>Close</button>
        </div>
        {err && (
          <span data-testid="hf-quickstart-error"
                style={{ color: "#b91c1c" }}>{err}</span>
        )}
        {result && (
          <div data-testid="hf-quickstart-result"
               style={{ background: "#dcfce7", padding: 8, borderRadius: 4,
                        color: "#15803d", marginTop: 4 }}>
            <div data-testid="hf-quickstart-result-path">
              parquet: {result.parquet_path}
            </div>
            <div data-testid="hf-quickstart-result-tokens">
              {result.n_tokens_written} tokens, {result.n_docs_seen} docs,
              {" "}{result.elapsed_ms.toFixed(0)} ms
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
