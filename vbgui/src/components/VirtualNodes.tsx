import { Handle, Position, type NodeProps } from "@xyflow/react";
import { T } from "@/theme";

export interface Token {
  id: number;
  text: string;
}

export function TokenizerVirtualNode({ data }: NodeProps): JSX.Element {
  const isActive = data.isActiveNode as boolean;
  const prompt = (data.prompt as string) || "The cat sat on the mat";
  const tokens = (data.tokens as Token[]) || [];
  const onPromptChange = data.onPromptChange as (val: string) => void;

  return (
    <div
      role="group"
      aria-label="virtual tokenizer"
      data-testid="tokenizer-virtual-node"
      className={`vb-node ${isActive ? "vb-node-selected" : ""}`}
      style={{
        minWidth: 260,
        padding: "14px",
        background: "var(--vb-surface-2)",
        border: isActive
          ? "2px solid var(--vb-accent)"
          : "1px solid var(--vb-border)",
        borderRadius: "var(--vb-radius-lg)",
        boxShadow: isActive
          ? "0 0 20px var(--vb-accent-soft)"
          : "var(--vb-shadow-panel)",
        color: T.text,
        fontFamily: T.font,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <span style={{ fontSize: 16 }}>🔠</span>
        <div style={{ fontWeight: 600, fontSize: 14 }}>Tokenizer</div>
        <span
          style={{
            marginLeft: "auto",
            fontSize: 9,
            padding: "2px 6px",
            borderRadius: "var(--vb-radius-pill)",
            background: "var(--vb-accent-soft)",
            color: "var(--vb-accent)",
            fontWeight: 700,
            textTransform: "uppercase",
          }}
        >
          Virtual
        </span>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label
          htmlFor="debugger-prompt-input"
          style={{
            display: "block",
            fontSize: 10,
            color: T.textSecondary,
            marginBottom: 4,
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          Prompt Input
        </label>
        <input
          id="debugger-prompt-input"
          type="text"
          className="nodrag nopan"
          value={prompt}
          onChange={(e) => onPromptChange?.(e.target.value)}
          placeholder="Type prompt here..."
          style={{
            width: "100%",
            background: "var(--vb-surface-3)",
            border: "1px solid var(--vb-border)",
            borderRadius: "var(--vb-radius-sm)",
            color: T.text,
            padding: "6px 8px",
            fontSize: 12,
            fontFamily: T.font,
            outline: "none",
          }}
        />
      </div>

      <div>
        <div
          style={{
            fontSize: 10,
            color: T.textSecondary,
            marginBottom: 6,
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          Segmented Tokens
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {tokens.map((t, idx) => (
            <div
              key={idx}
              style={{
                background: "var(--vb-surface-3)",
                border: "1px solid var(--vb-border)",
                borderRadius: "var(--vb-radius-sm)",
                padding: "4px 8px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 2,
              }}
            >
              <span
                style={{
                  fontSize: 11.5,
                  fontWeight: 600,
                  fontFamily: T.fontMono,
                  color: "var(--vb-accent)",
                }}
              >
                &ldquo;{t.text}&rdquo;
              </span>
              <span style={{ fontSize: 9, color: T.textMuted, fontFamily: T.fontMono }}>
                ID: {t.id}
              </span>
            </div>
          ))}
        </div>
      </div>

      <Handle type="source" position={Position.Right} />
    </div>
  );
}

export function DetokenizerVirtualNode({ data }: NodeProps): JSX.Element {
  const isActive = data.isActiveNode as boolean;
  const outputToken = "mat";

  return (
    <div
      role="group"
      aria-label="virtual detokenizer"
      data-testid="detokenizer-virtual-node"
      className={`vb-node ${isActive ? "vb-node-selected" : ""}`}
      style={{
        minWidth: 260,
        padding: "14px",
        background: "var(--vb-surface-2)",
        border: isActive
          ? "2px solid var(--vb-success)"
          : "1px solid var(--vb-border)",
        borderRadius: "var(--vb-radius-lg)",
        boxShadow: isActive
          ? "0 0 20px rgba(52, 211, 153, 0.2)"
          : "var(--vb-shadow-panel)",
        color: T.text,
        fontFamily: T.font,
      }}
    >
      <Handle type="target" position={Position.Left} />

      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <span style={{ fontSize: 16 }}>🔡</span>
        <div style={{ fontWeight: 600, fontSize: 14 }}>De-Tokenizer</div>
        <span
          style={{
            marginLeft: "auto",
            fontSize: 9,
            padding: "2px 6px",
            borderRadius: "var(--vb-radius-pill)",
            background: "rgba(52, 211, 153, 0.16)",
            color: "var(--vb-success)",
            fontWeight: 700,
            textTransform: "uppercase",
          }}
        >
          Virtual
        </span>
      </div>

      <div style={{ marginBottom: 12 }}>
        <div
          style={{
            fontSize: 10,
            color: T.textSecondary,
            marginBottom: 4,
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          Next Token Logits
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {[
            { token: "mat", prob: "91.2%", color: "var(--vb-success)" },
            { token: "wall", prob: "4.1%", color: T.textSecondary },
            { token: "bed", prob: "2.3%", color: T.textSecondary },
          ].map((item, idx) => (
            <div
              key={idx}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "4px 8px",
                background: idx === 0 ? "rgba(52, 211, 153, 0.08)" : "var(--vb-surface-3)",
                border: `1px solid ${idx === 0 ? "var(--vb-success)" : "var(--vb-border)"}`,
                borderRadius: "var(--vb-radius-sm)",
                fontSize: 11,
              }}
            >
              <span style={{ fontFamily: T.fontMono, color: item.color, fontWeight: 600 }}>
                &ldquo;{item.token}&rdquo;
              </span>
              <span style={{ fontFamily: T.fontMono, color: T.textMuted }}>{item.prob}</span>
            </div>
          ))}
        </div>
      </div>

      <div>
        <div
          style={{
            fontSize: 10,
            color: T.textSecondary,
            marginBottom: 4,
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          Generated Text
        </div>
        <div
          style={{
            padding: "8px 10px",
            background: "var(--vb-surface-3)",
            border: "1px solid var(--vb-border)",
            borderRadius: "var(--vb-radius-sm)",
            fontSize: 12,
            lineHeight: 1.4,
          }}
        >
          The cat sat on the <span style={{ color: "var(--vb-success)", fontWeight: 600 }}>{outputToken}</span>
        </div>
      </div>
    </div>
  );
}
