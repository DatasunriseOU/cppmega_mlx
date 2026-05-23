import { useCallback, useEffect, useState } from "react";
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
    // V7-H46: byte_roundtrip is true when decode(encode(text)) == text.
    // Backend already produces this on the EncodeVisualize call; we
    // surface it as a pill so the user sees roundtrip status without
    // a separate RPC (mirrors DataInspector's data.roundtrip_check
    // indicator).
    byte_roundtrip?: boolean;
    [k: string]: unknown;
  };
  elapsed_ms: number;
}

/** V7-H46: tokenizer.roundtrip_text RPC response shape. */
export interface TokenizerRoundtripTextResult {
  matches: boolean;
  decoded: string;
  original_bytes: number;
  decoded_bytes: number;
  byte_diff: number;
  tokenizer_capability: string;
  elapsed_ms: number;
}

export interface TokenizerPanelState {
  source: string;
  result?: EncodeVisualizeResult;
  error?: string;
  // V7-H46: per-panel roundtrip-check result + running flag.
  roundtrip?: TokenizerRoundtripTextResult;
  roundtripError?: string;
  roundtripRunning?: boolean;
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
  // V7-K1: backend-driven tokenizer preset list via tokenizer.list_presets.
  // Replaces the previously hardcoded suggestion set so the UI picks up
  // PRESET_LIBRARY changes from the backend without a frontend rebuild.
  const [presets, setPresets] = useState<string[]>([]);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await rpc.call<{ presets: string[] }>(
          "tokenizer.list_presets", {});
        if (!cancelled) setPresets(res.presets ?? []);
      } catch { /* leave empty — datalist degrades gracefully */ }
    })();
    return () => { cancelled = true; };
  }, [rpc]);

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

  // V7-H46: invoke tokenizer.roundtrip_text RPC for this panel's
  // source + the current playground text; stash matches/diff onto the
  // panel state so the badge renders inline.
  const runRoundtrip = useCallback(async (idx: number) => {
    const panel = panels[idx];
    if (!panel?.source) return;
    setPanels((prev) => prev.map((p, i) =>
      i === idx ? { ...p, roundtripRunning: true,
                    roundtripError: undefined } : p));
    try {
      const result = await rpc.call<TokenizerRoundtripTextResult>(
        "tokenizer.roundtrip_text",
        { tokenizer_source: panel.source, text },
      );
      setPanels((prev) => prev.map((p, i) =>
        i === idx ? { ...p, roundtrip: result,
                      roundtripRunning: false,
                      roundtripError: undefined } : p));
    } catch (e) {
      setPanels((prev) => prev.map((p, i) =>
        i === idx ? { ...p, roundtripError: String(e),
                      roundtripRunning: false } : p));
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

      <datalist id="tokenizer-preset-suggestions"
                data-testid="tokenizer-preset-suggestions">
        {presets.map((p) => <option key={p} value={p} />)}
      </datalist>

      <div style={{ display: "grid",
                    gridTemplateColumns: `repeat(${Math.max(1, panels.length)}, 1fr)`,
                    gap: 8, flex: 1, minHeight: 0 }}>
        {panels.map((p, i) => (
          <TokenizerPanel key={i} index={i} state={p}
                          presets={presets}
                          hoverSpan={hoverSpan}
                          onSourceChange={(s) => setSource(i, s)}
                          onEncode={() => runEncode(i)}
                          onRemove={() => removePanel(i)}
                          onHover={setHoverSpan}
                          onUseForTrain={onUseForTrain}
                          onRoundtripCheck={runRoundtrip}
                          trainTokenizerPath={trainTokenizerPath} />
        ))}
      </div>
    </div>
  );
}


interface TokenizerPanelProps {
  index: number;
  state: TokenizerPanelState;
  presets?: string[];
  hoverSpan: { start: number; end: number } | null;
  onSourceChange: (s: string) => void;
  onEncode: () => void;
  onRemove: () => void;
  onHover: (span: { start: number; end: number } | null) => void;
  onUseForTrain?: (tokenizerSource: string) => void;
  /** V7-H46: fired when the user clicks the panel's Roundtrip-check
   *  button. Parent owns the RPC call (since panels share `text`). */
  onRoundtripCheck?: (idx: number) => void;
  trainTokenizerPath?: string | null;
}

interface RoundtripCheckProps {
  index: number;
  source: string;
  roundtrip?: TokenizerRoundtripTextResult;
  roundtripError?: string;
  running: boolean;
  onRun: () => void;
}

function TokenizerRoundtripCheck({
  index, source, roundtrip, roundtripError, running, onRun,
}: RoundtripCheckProps): JSX.Element {
  return (
    <div data-testid={`tokenizer-roundtrip-check-${index}`}
         style={{ display: "flex", gap: 6, alignItems: "center",
                  fontSize: 11 }}>
      <button data-testid={`tokenizer-roundtrip-run-${index}`}
              disabled={!source || running}
              onClick={onRun}>
        {running ? "Checking…" : "Roundtrip-check"}
      </button>
      {roundtripError && (
        <span data-testid={`tokenizer-roundtrip-error-${index}`}
              style={{ color: "#b91c1c" }}>
          {roundtripError}
        </span>
      )}
      {roundtrip && (
        <span data-testid={`tokenizer-roundtrip-badge-${index}`}
              data-matches={roundtrip.matches ? "yes" : "no"}
              data-capability={roundtrip.tokenizer_capability}
              style={{ padding: "2px 6px", borderRadius: 3,
                       fontFamily: "monospace",
                       background: roundtrip.matches
                         ? "#d1fae5" : "#fee2e2",
                       color: roundtrip.matches
                         ? "#065f46" : "#991b1b" }}>
          {roundtrip.matches ? "✓ byte-exact" : "✗ byte mismatch"}
          {" · "}{roundtrip.tokenizer_capability}
          {!roundtrip.matches && roundtrip.byte_diff > 0 &&
            ` · Δ${roundtrip.byte_diff}B`}
        </span>
      )}
    </div>
  );
}

function TokenizerPanel({
  index, state, presets, hoverSpan, onSourceChange, onEncode, onRemove,
  onHover, onUseForTrain, onRoundtripCheck, trainTokenizerPath,
}: TokenizerPanelProps): JSX.Element {
  return (
    <section data-testid={`tokenizer-panel-${index}`}
             style={{ border: "1px solid var(--vb-border)", borderRadius: 4,
                      padding: 8, display: "flex", flexDirection: "column",
                      gap: 6, minHeight: 0 }}>
      <div style={{ display: "flex", gap: 4 }}>
        <input data-testid={`tokenizer-source-${index}`}
               type="text" placeholder="tokenizer.json path or hub id"
               value={state.source}
               list="tokenizer-preset-suggestions"
               onChange={(e) => onSourceChange(e.target.value)}
               style={{ flex: 1, fontFamily: "monospace", fontSize: 11 }} />
        {presets && presets.length > 0 && (
          <select data-testid={`tokenizer-preset-picker-${index}`}
                  value=""
                  onChange={(e) => {
                    if (e.target.value) onSourceChange(e.target.value);
                    e.currentTarget.value = "";
                  }}
                  title="Pick a backend tokenizer preset"
                  style={{ fontSize: 11, maxWidth: 110 }}>
            <option value="">presets…</option>
            {presets.map((p) =>
              <option key={p} value={p}>{p}</option>)}
          </select>
        )}
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
               style={{ fontSize: 11, color: "var(--vb-text-secondary)",
                        display: "flex", alignItems: "center", gap: 6,
                        flexWrap: "wrap" }}>
            <span>{state.result.token_count} tokens</span>
            <span>· bytes/tok avg{" "}
              {state.result.bytes_per_token_avg.toFixed(2)}</span>
            <span>· max {state.result.bytes_per_token_max}</span>
            <span>· vocab {state.result.capabilities.vocab_size}</span>
            {state.result.capabilities.has_fim && <span>· FIM ✓</span>}
            {/* V7-H46: byte-roundtrip pill mirrors DataInspector's
                data.roundtrip_check status; backend ships the bool
                on capabilities so no extra RPC needed. */}
            {(() => {
              const rt = state.result.capabilities.byte_roundtrip;
              if (rt === undefined) return null;
              return (
                <span
                  data-testid={`tokenizer-roundtrip-${index}`}
                  data-roundtrip={rt ? "ok" : "fail"}
                  style={{ padding: "1px 6px", borderRadius: 9999,
                           fontSize: 10, fontWeight: 600,
                           background: rt ? "#dcfce7" : "#fee2e2",
                           color: rt ? "#166534" : "#991b1b",
                           border: `1px solid ${rt ? "#86efac"
                                                  : "#fca5a5"}` }}>
                  {rt ? "roundtrip ✓" : "roundtrip ✗"}
                </span>
              );
            })()}
          </div>
          {/* V7-H46: live roundtrip check via tokenizer.roundtrip_text. */}
          {onRoundtripCheck && (
            <TokenizerRoundtripCheck index={index}
              source={state.source}
              roundtrip={state.roundtrip}
              roundtripError={state.roundtripError}
              running={state.roundtripRunning ?? false}
              onRun={() => onRoundtripCheck(index)} />
          )}
          <div data-testid={`tokenizer-chips-${index}`}
               style={{ display: "flex", flexWrap: "wrap", gap: 2,
                        overflowY: "auto", fontFamily: "monospace",
                        fontSize: 11, padding: 4,
                        background: "var(--vb-surface-2)", borderRadius: 3 }}>
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
