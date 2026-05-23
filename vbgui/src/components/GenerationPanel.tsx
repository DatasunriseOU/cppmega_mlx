// V7-F01 / unwired-RPC closure — UI surface for gen.run.
// Backend has shipped greedy / temperature / top_k / top_p samplers
// for a while; the UI just hadn't exposed them. This panel lets the
// architect feed prompt_tokens (comma-separated ints) + sampler
// strategy + per-strategy hyperparams, fire gen.run, and inspect the
// generated tokens + finish_reason + elapsed_ms.

import { useState } from "react";
import type { RpcClient } from "@/lib/rpc";
import { HelpIcon } from "@/components/HelpIcon";

type Strategy = "greedy" | "temperature" | "top_k" | "top_p";

export interface GenerationPanelProps {
  rpc: RpcClient | null;
}

interface GenResult {
  tokens: number[];
  finish_reason: "eos" | "length" | "aborted";
  elapsed_ms: number;
  strategy: Strategy;
  smoke: boolean;
}

export function GenerationPanel({ rpc }: GenerationPanelProps): JSX.Element {
  const [prompt, setPrompt] = useState<string>("1, 2, 3");
  const [maxNewTokens, setMaxNewTokens] = useState<number>(16);
  const [strategy, setStrategy] = useState<Strategy>("greedy");
  const [temperature, setTemperature] = useState<number>(1.0);
  const [topK, setTopK] = useState<number>(50);
  const [topP, setTopP] = useState<number>(0.9);
  const [seed, setSeed] = useState<number>(0);
  const [vocabSize, setVocabSize] = useState<number>(32);
  const [smoke, setSmoke] = useState<boolean>(true);
  const [result, setResult] = useState<GenResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState<boolean>(false);

  async function run() {
    setError(null);
    setResult(null);
    setRunning(true);
    try {
      if (!rpc) throw new Error("rpc unavailable");
      const prompt_tokens = prompt
        .split(/[,\s]+/).map((s) => parseInt(s, 10))
        .filter((n) => Number.isFinite(n));
      const r = await rpc.call<GenResult>("gen.run", {
        prompt_tokens, max_new_tokens: maxNewTokens, strategy,
        temperature, top_k: topK, top_p: topP, seed,
        vocab_size: vocabSize, smoke,
      });
      setResult(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div data-testid="generation-panel"
         style={{ padding: 12, fontFamily: "system-ui, sans-serif",
                  fontSize: 12, display: "flex", flexDirection: "column",
                  gap: 8, flex: 1, overflowY: "auto" }}>
      <header style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>
          Inference (gen.run)
        </h3>
        <HelpIcon topic="gen_run" />
      </header>

      <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <span style={{ color: "#6b7280" }}>prompt tokens (comma sep)</span>
        <textarea data-testid="gen-prompt-tokens"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={2}
                  style={{ fontFamily: "monospace", fontSize: 12 }} />
      </label>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <label>strategy
          <select data-testid="gen-strategy" value={strategy}
                  onChange={(e) => setStrategy(e.target.value as Strategy)}
                  style={{ marginLeft: 4 }}>
            <option value="greedy">greedy</option>
            <option value="temperature">temperature</option>
            <option value="top_k">top_k</option>
            <option value="top_p">top_p</option>
          </select>
        </label>
        <label>max new
          <input data-testid="gen-max-new-tokens" type="number"
                 min={1} max={4096} value={maxNewTokens}
                 onChange={(e) => setMaxNewTokens(
                   Math.max(1, parseInt(e.target.value || "1", 10)))}
                 style={{ marginLeft: 4, width: 70 }} />
        </label>
        <label>temp
          <input data-testid="gen-temperature" type="number" step="0.1"
                 min={0.1} max={10} value={temperature}
                 onChange={(e) => setTemperature(
                   parseFloat(e.target.value || "1"))}
                 style={{ marginLeft: 4, width: 60 }} />
        </label>
        <label>top_k
          <input data-testid="gen-top-k" type="number" min={1}
                 value={topK}
                 onChange={(e) => setTopK(parseInt(e.target.value || "1", 10))}
                 style={{ marginLeft: 4, width: 60 }} />
        </label>
        <label>top_p
          <input data-testid="gen-top-p" type="number" step="0.05"
                 min={0.05} max={1.0} value={topP}
                 onChange={(e) => setTopP(parseFloat(e.target.value || "0.9"))}
                 style={{ marginLeft: 4, width: 60 }} />
        </label>
        <label>seed
          <input data-testid="gen-seed" type="number" value={seed}
                 onChange={(e) => setSeed(parseInt(e.target.value || "0", 10))}
                 style={{ marginLeft: 4, width: 70 }} />
        </label>
        <label>vocab
          <input data-testid="gen-vocab-size" type="number" min={2}
                 value={vocabSize}
                 onChange={(e) => setVocabSize(
                   parseInt(e.target.value || "32", 10))}
                 style={{ marginLeft: 4, width: 70 }} />
        </label>
        <label style={{ display: "inline-flex", gap: 4,
                        alignItems: "center" }}>
          <input data-testid="gen-smoke" type="checkbox"
                 checked={smoke}
                 onChange={(e) => setSmoke(e.target.checked)} />
          smoke
        </label>
      </div>

      <button data-testid="gen-run"
              onClick={run} disabled={running || !rpc}
              style={{ padding: "4px 12px", background: "#2563eb",
                       color: "white", border: "none", borderRadius: 4,
                       cursor: running ? "wait" : "pointer",
                       alignSelf: "flex-start" }}>
        {running ? "Running…" : "Run"}
      </button>

      {error && (
        <div data-testid="gen-error"
             style={{ color: "#991b1b", background: "#fee2e2",
                      padding: 6, borderRadius: 4 }}>
          {error}
        </div>
      )}

      {result && (
        <div data-testid="gen-result"
             style={{ background: "#f9fafb", padding: 8,
                      borderRadius: 4, border: "1px solid #e5e7eb" }}>
          <div style={{ display: "flex", gap: 8, fontSize: 11,
                        color: "#374151", marginBottom: 4 }}>
            <span data-testid="gen-finish-reason">
              finish: <code>{result.finish_reason}</code>
            </span>
            <span data-testid="gen-elapsed-ms">
              {result.elapsed_ms.toFixed(2)} ms
            </span>
            <span data-testid="gen-result-strategy">
              {result.strategy}
            </span>
            {result.smoke && (
              <span data-testid="gen-result-smoke"
                    style={{ color: "#92400e" }}>
                (smoke)
              </span>
            )}
          </div>
          <div data-testid="gen-tokens"
               style={{ fontFamily: "monospace", fontSize: 11,
                        display: "flex", gap: 2, flexWrap: "wrap" }}>
            {result.tokens.map((t, i) => (
              <span key={i}
                    data-testid={`gen-token-${i}`}
                    style={{ background: "#dbeafe", color: "#1e40af",
                             padding: "1px 4px", borderRadius: 2 }}>
                {t}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
