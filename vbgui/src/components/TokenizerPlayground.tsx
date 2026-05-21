import { useCallback, useState } from "react";
import type { RpcClient } from "@/lib/rpc";

export interface TokenSpan {
  id: number;
  text: string;
  start: number;
  end: number;
  is_special: boolean;
}

export interface EncodeVisualizeResult {
  tokens: TokenSpan[];
  token_count: number;
  bytes_total: number;
  bytes_per_token_avg: number;
  bytes_per_token_max: number;
  capabilities: {
    vocab_size: number;
    has_fim: boolean;
    has_space_nl: boolean;
    decoder_kind: "custom" | "hf" | "none";
    [k: string]: unknown;
  };
  elapsed_ms: number;
}

export interface TokenizerPanelState {
  source: string;
  result?: EncodeVisualizeResult;
  error?: string;
}

export interface TokenizerPlaygroundProps {
  rpc: RpcClient;
  initialSources?: string[];      // up to 3
  maxPanels?: number;             // default 3
  /** V4-3: callback when user picks this tokenizer for training.
   *  App stores in trainTokenizerPath; handleRunPipeline forwards via
   *  stage_options.train.tokenizer_path so backend V4-2 path can
   *  tokenize parquet text. */
  onUseForTrain?: (tokenizerSource: string) => void;
  /** V4-3: current path App is using for training (drives ✓ label). */
  trainTokenizerPath?: string | null;
}

const COLORS = ["#fde68a", "#bfdbfe", "#bbf7d0", "#fecaca", "#ddd6fe",
                "#fed7aa", "#fbcfe8", "#e5e7eb"];

function colorForId(id: number, isSpecial: boolean): string {
  if (isSpecial) return "#fca5a5";
  return COLORS[id % COLORS.length];
}

export function TokenizerPlayground({
  rpc, initialSources = [], maxPanels = 3,
  onUseForTrain, trainTokenizerPath,
}: TokenizerPlaygroundProps): JSX.Element {
  const [text, setText] = useState("Hello, world!\ndef foo():\n  return 42");
  const [hoverSpan, setHoverSpan] = useState<{ start: number; end: number } | null>(null);
  const [panels, setPanels] = useState<TokenizerPanelState[]>(
    () => initialSources.slice(0, maxPanels).map((source) => ({ source })),
  );

  const runEncode = useCallback(async (idx: number) => {
    const panel = panels[idx];
    if (!panel?.source) return;
    try {
      const result = await rpc.call<EncodeVisualizeResult>(
        "tokenizer.encode_visualize",
        { tokenizer_source: panel.source, text },
      );
      setPanels((prev) => prev.map((p, i) =>
        i === idx ? { ...p, result, error: undefined } : p));
    } catch (e) {
      setPanels((prev) => prev.map((p, i) =>
        i === idx ? { ...p, error: String(e), result: undefined } : p));
    }
  }, [panels, rpc, text]);

  const addPanel = useCallback(() => {
    setPanels((prev) => prev.length < maxPanels
      ? [...prev, { source: "" }] : prev);
  }, [maxPanels]);

  const removePanel = useCallback((idx: number) => {
    setPanels((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  const setSource = useCallback((idx: number, source: string) => {
    setPanels((prev) => prev.map((p, i) => i === idx ? { ...p, source } : p));
  }, []);

  return (
    <div data-testid="tokenizer-playground"
         style={{ display: "flex", flexDirection: "column",
                  height: "100%", padding: 12, gap: 8,
                  fontFamily: "system-ui, sans-serif" }}>
      <header style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>Tokenizer Playground</h3>
        <button data-testid="add-panel"
                disabled={panels.length >= maxPanels}
                onClick={addPanel}>+ Add tokenizer</button>
      </header>

      <textarea
        data-testid="tokenizer-input"
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={4}
        style={{ width: "100%", fontFamily: "monospace", fontSize: 12,
                 padding: 6, border: "1px solid #d1d5db", borderRadius: 4 }}
      />

      <div style={{ display: "grid",
                    gridTemplateColumns: `repeat(${Math.max(1, panels.length)}, 1fr)`,
                    gap: 8, flex: 1, minHeight: 0 }}>
        {panels.map((p, i) => (
          <TokenizerPanel key={i} index={i} state={p}
                          hoverSpan={hoverSpan}
                          onSourceChange={(s) => setSource(i, s)}
                          onEncode={() => runEncode(i)}
                          onRemove={() => removePanel(i)}
                          onHover={setHoverSpan}
                          onUseForTrain={onUseForTrain}
                          trainTokenizerPath={trainTokenizerPath} />
        ))}
      </div>
    </div>
  );
}


interface TokenizerPanelProps {
  index: number;
  state: TokenizerPanelState;
  hoverSpan: { start: number; end: number } | null;
  onSourceChange: (s: string) => void;
  onEncode: () => void;
  onRemove: () => void;
  onHover: (span: { start: number; end: number } | null) => void;
  onUseForTrain?: (tokenizerSource: string) => void;
  trainTokenizerPath?: string | null;
}

function TokenizerPanel({
  index, state, hoverSpan, onSourceChange, onEncode, onRemove, onHover,
  onUseForTrain, trainTokenizerPath,
}: TokenizerPanelProps): JSX.Element {
  return (
    <section data-testid={`tokenizer-panel-${index}`}
             style={{ border: "1px solid #e5e7eb", borderRadius: 4,
                      padding: 8, display: "flex", flexDirection: "column",
                      gap: 6, minHeight: 0 }}>
      <div style={{ display: "flex", gap: 4 }}>
        <input data-testid={`tokenizer-source-${index}`}
               type="text" placeholder="tokenizer.json path or hub id"
               value={state.source}
               onChange={(e) => onSourceChange(e.target.value)}
               style={{ flex: 1, fontFamily: "monospace", fontSize: 11 }} />
        <button data-testid={`tokenizer-encode-${index}`} onClick={onEncode}>
          Encode
        </button>
        {onUseForTrain && (
          <button data-testid={`tokenizer-use-for-train-${index}`}
                  disabled={!state.source}
                  title={trainTokenizerPath === state.source
                    ? "Currently used for training"
                    : "Send this tokenizer to stage_train"}
                  onClick={() => onUseForTrain(state.source)}
                  style={{
                    background: trainTokenizerPath === state.source
                      ? "#dcfce7" : undefined,
                    color: trainTokenizerPath === state.source
                      ? "#166534" : undefined,
                  }}>
            {trainTokenizerPath === state.source ? "✓ Train" : "→ Train"}
          </button>
        )}
        <button data-testid={`tokenizer-remove-${index}`} onClick={onRemove}>×</button>
      </div>

      {state.error && (
        <div data-testid={`tokenizer-error-${index}`}
             style={{ color: "#b91c1c", fontSize: 11 }}>
          {state.error}
        </div>
      )}

      {state.result && (
        <>
          <div data-testid={`tokenizer-metrics-${index}`}
               style={{ fontSize: 11, color: "#374151" }}>
            {state.result.token_count} tokens ·
            {" "}bytes/tok avg {state.result.bytes_per_token_avg.toFixed(2)}
            {" "}max {state.result.bytes_per_token_max} ·
            {" "}vocab {state.result.capabilities.vocab_size}
            {state.result.capabilities.has_fim ? " · FIM ✓" : ""}
          </div>
          <div data-testid={`tokenizer-chips-${index}`}
               style={{ display: "flex", flexWrap: "wrap", gap: 2,
                        overflowY: "auto", fontFamily: "monospace",
                        fontSize: 11, padding: 4,
                        background: "#f9fafb", borderRadius: 3 }}>
            {state.result.tokens.map((t, ti) => {
              const overlap = hoverSpan && t.start < hoverSpan.end &&
                              t.end > hoverSpan.start;
              return (
                <span key={ti}
                      data-testid={`tokenizer-chip-${index}-${ti}`}
                      onMouseEnter={() => onHover({ start: t.start, end: t.end })}
                      onMouseLeave={() => onHover(null)}
                      title={`id=${t.id} [${t.start}, ${t.end}]`}
                      style={{
                        background: colorForId(t.id, t.is_special),
                        padding: "1px 4px", borderRadius: 2,
                        border: overlap
                          ? "1px solid #1d4ed8" : "1px solid transparent",
                        whiteSpace: "pre",
                      }}>
                  {t.text}
                </span>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}
