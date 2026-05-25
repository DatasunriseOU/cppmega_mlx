import { useState, useEffect } from "react";

export interface WizardOptions {
  scaleFactor: "1" | "4" | "8" | "32"; // 1x, 1/4, 1/8, 1/32
  numLayers: number;
  tokenizer: string;
}

export interface LLMGalleryWizardModalProps {
  presetName: string | null;
  onClose: () => void;
  onGenerate: (options: WizardOptions) => void;
}

const PRESET_CANONICAL_DEFAULTS: Record<string, { H: number; layers: number; tokenizer: string }> = {
  "llama3_8b": { H: 4096, layers: 32, tokenizer: "cppmega_native_65k" },
  "llama3_2_1b": { H: 2048, layers: 16, tokenizer: "cppmega_native_65k" },
  "llama3_2_3b": { H: 3072, layers: 28, tokenizer: "cppmega_native_65k" },
  "smollm3": { H: 576, layers: 30, tokenizer: "cppmega_native_65k" },
  "phi4": { H: 3072, layers: 40, tokenizer: "cppmega_native_65k" },
  "mistral_small_3_1": { H: 4096, layers: 32, tokenizer: "cppmega_native_65k" },
  "gpt2_xl": { H: 1600, layers: 48, tokenizer: "gpt2_tiktoken" },
  "xlstm_7b": { H: 4096, layers: 36, tokenizer: "cppmega_native_65k" },
  "deepseek_v4_flash": { H: 2048, layers: 16, tokenizer: "cppmega_native_65k" },
  "deepseek_v3": { H: 4096, layers: 32, tokenizer: "cppmega_native_65k" },
  "gemma4": { H: 2048, layers: 24, tokenizer: "cppmega_native_65k" },
  "mistral4": { H: 4096, layers: 32, tokenizer: "cppmega_native_65k" },
  "ling26": { H: 2048, layers: 24, tokenizer: "cppmega_native_65k" },
  "kimi_linear": { H: 3072, layers: 28, tokenizer: "cppmega_native_65k" },
  "kimi_k2": { H: 3072, layers: 28, tokenizer: "cppmega_native_65k" },
  "longcat": { H: 2048, layers: 24, tokenizer: "cppmega_native_65k" },
  "nemotron3": { H: 3072, layers: 28, tokenizer: "cppmega_native_65k" },
  "zaya1": { H: 2048, layers: 24, tokenizer: "cppmega_native_65k" },
  "arcee_trinity": { H: 2048, layers: 24, tokenizer: "cppmega_native_65k" },
  "qwen3_next": { H: 2048, layers: 24, tokenizer: "cppmega_native_65k" },
  "gemma3_27b": { H: 5376, layers: 64, tokenizer: "cppmega_native_65k" },
  "gemma3_270m": { H: 640, layers: 18, tokenizer: "cppmega_native_65k" },
};

export function LLMGalleryWizardModal({
  presetName, onClose, onGenerate,
}: LLMGalleryWizardModalProps): JSX.Element | null {
  const [scale, setScale] = useState<"1" | "4" | "8" | "32">("32"); // default to 1/32 for fast local testing
  const [layers, setLayers] = useState<number>(4); // default lightweight layers
  const [tokenizer, setTokenizer] = useState<string>("cppmega_native_65k");

  useEffect(() => {
    if (!presetName) return;
    const defaults = PRESET_CANONICAL_DEFAULTS[presetName] || { H: 2048, layers: 12, tokenizer: "gpt2_tiktoken" };
    setTokenizer(defaults.tokenizer);
    // Suggest 4 layers by default for swift local evaluation, but allow user to adjust
    setLayers(Math.min(defaults.layers, 4));
  }, [presetName]);

  if (!presetName) return null;

  const defaults = PRESET_CANONICAL_DEFAULTS[presetName] || { H: 2048, layers: 12, tokenizer: "gpt2_tiktoken" };
  const targetH = Math.round(defaults.H / parseInt(scale));

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 2000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(15, 23, 42, 0.6)",
        backdropFilter: "blur(8px)",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <div
        data-testid="llm-gallery-wizard"
        style={{
          background: "rgba(30, 41, 59, 0.95)",
          border: "1px solid rgba(255, 255, 255, 0.15)",
          borderRadius: 16,
          padding: 24,
          width: 440,
          color: "#f8fafc",
          boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.7)",
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "#22d3ee" }}>
            📖 Sebastian Raschka LLM Gallery
          </h3>
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

        <p style={{ fontSize: 12, color: "#94a3b8", margin: 0 }}>
          Configure architecture parameters for <strong style={{ color: "#e2e8f0" }}>{presetName}</strong> before generating its visual block representation.
        </p>

        {/* 1. Scale Down Factor */}
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <label style={{ fontSize: 11, fontWeight: 600, color: "#cbd5e1" }}>
            📐 Scale Dimensions (Hidden Size H)
          </label>
          <div style={{ display: "flex", gap: 6 }}>
            {(["1", "4", "8", "32"] as const).map((s) => (
              <button
                key={s}
                onClick={() => setScale(s)}
                style={{
                  flex: 1,
                  padding: "6px 4px",
                  borderRadius: 6,
                  border: s === scale ? "1px solid #06b6d4" : "1px solid #475569",
                  background: s === scale ? "rgba(6, 182, 212, 0.15)" : "transparent",
                  color: s === scale ? "#22d3ee" : "#94a3b8",
                  cursor: "pointer",
                  fontSize: 11,
                  fontWeight: "bold",
                }}
              >
                {s === "1" ? "1x Full" : `1/${s}`}
              </button>
            ))}
          </div>
          <span style={{ fontSize: 11, color: "#64748b" }}>
            Canonical hidden size is {defaults.H}. Target H: <strong style={{ color: "#22d3ee" }}>{targetH}</strong>
          </span>
        </div>

        {/* 2. Number of Layers */}
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <label style={{ fontSize: 11, fontWeight: 600, color: "#cbd5e1" }}>
            🥞 Layer Count (🥞 Repetition blocks)
          </label>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <input
              type="range"
              min={1}
              max={defaults.layers}
              value={layers}
              onChange={(e) => setLayers(parseInt(e.target.value))}
              style={{ flex: 1, accentColor: "#06b6d4" }}
            />
            <input
              type="number"
              min={1}
              max={defaults.layers}
              value={layers}
              onChange={(e) => setLayers(Math.max(1, Math.min(defaults.layers, parseInt(e.target.value) || 1)))}
              style={{
                width: 60,
                background: "#1e293b",
                border: "1px solid #475569",
                borderRadius: 4,
                color: "white",
                padding: "2px 4px",
                textAlign: "center",
                fontSize: 12,
              }}
            />
          </div>
          <span style={{ fontSize: 11, color: "#64748b" }}>
            Canonical preset has {defaults.layers} blocks. Lightweight simulation set to <strong style={{ color: "#cbd5e1" }}>{layers}</strong> blocks.
          </span>
        </div>

        {/* 3. Tokenizer choice */}
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <label style={{ fontSize: 11, fontWeight: 600, color: "#cbd5e1" }}>
            🪙 Tokenizer Configuration
          </label>
          <select
            value={tokenizer}
            onChange={(e) => setTokenizer(e.target.value)}
            style={{
              background: "#1e293b",
              border: "1px solid #475569",
              borderRadius: 6,
              color: "white",
              padding: "6px 8px",
              fontSize: 12,
              outline: "none",
            }}
          >
            <option value="cppmega_native_65k">cppmega Native (65k BPE vocab - Local repo)</option>
            <option value="cppmega_v3">cppmega V3 (100k BPE vocab)</option>
            <option value="gpt2_tiktoken">GPT-2 (50k BPE vocab)</option>
            <option value="minimal_no_fim">Minimal (No FIM)</option>
            <option value="fim_only">FIM Only</option>
          </select>
        </div>

        {/* Footer Actions */}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 8 }}>
          <button
            onClick={onClose}
            style={{
              padding: "6px 14px",
              borderRadius: 6,
              border: "1px solid #475569",
              background: "transparent",
              color: "#94a3b8",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            Cancel
          </button>
          <button
            data-testid="llm-wizard-generate"
            onClick={() => onGenerate({ scaleFactor: scale, numLayers: layers, tokenizer })}
            style={{
              padding: "6px 14px",
              borderRadius: 6,
              border: "none",
              background: "#0891b2",
              color: "white",
              cursor: "pointer",
              fontSize: 12,
              fontWeight: "bold",
              boxShadow: "0 4px 6px -1px rgba(6, 182, 212, 0.4)",
            }}
          >
            Generate Architecture
          </button>
        </div>
      </div>
    </div>
  );
}
