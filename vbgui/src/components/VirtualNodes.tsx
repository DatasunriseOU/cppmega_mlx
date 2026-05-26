import { Handle, Position, type NodeProps } from "@xyflow/react";
import { T } from "@/theme";

export interface Token {
  id: number;
  text: string;
}

export interface MtpLogit {
  token: string;
  prob: string;
  color: string;
}

export interface MtpHeadData {
  position: number; // e.g. 1, 2, 3
  logits: MtpLogit[];
}

export function TokenizerVirtualNode({ data }: NodeProps): JSX.Element {
  const isActive = data.isActiveNode as boolean;
  const prompt = (data.prompt as string) || "The cat sat on the mat";
  const tokens = (data.tokens as Token[]) || [];
  const onPromptChange = data.onPromptChange as (val: string) => void;

  // Rich PathExplorer & Pre-fetch progression integration
  const selectedPath = (data.selectedPath as string) || null;
  const contentType = (data.contentType as string) || null;
  const progressPercent = (data.progressPercent as number) || 0;
  const tokenOffset = (data.tokenOffset as number) || 0;
  const downloadSpeed = (data.downloadSpeed as string) || null;

  return (
    <div
      role="group"
      aria-label="virtual tokenizer"
      data-testid="tokenizer-virtual-node"
      className={`vb-node ${isActive ? "vb-node-selected" : ""}`}
      style={{
        minWidth: 280,
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
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
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
          {contentType ? `${contentType}` : "Virtual"}
        </span>
      </div>

      {/* Path status & Pre-fetching meters */}
      {selectedPath && (
        <div
          className="glass-panel"
          style={{
            padding: 8,
            borderRadius: "var(--vb-radius-sm)",
            background: "rgba(34, 211, 238, 0.04)",
            border: "1px solid var(--vb-accent-soft)",
            display: "flex",
            flexDirection: "column",
            gap: 4,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10 }}>
            <span style={{ color: T.textSecondary, fontWeight: 600 }}>Active Source:</span>
            <span style={{ fontFamily: T.fontMono, color: "var(--vb-accent)" }}>
              {selectedPath.length > 22 ? `…${selectedPath.slice(-20)}` : selectedPath}
            </span>
          </div>

          {progressPercent > 0 && progressPercent < 100 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: T.textMuted }}>
                <span>Pre-fetching: {progressPercent.toFixed(0)}%</span>
                {downloadSpeed && <span>{downloadSpeed}</span>}
              </div>
              <div
                style={{
                  height: 4,
                  background: "var(--vb-surface-3)",
                  borderRadius: 2,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    background: "var(--vb-accent)",
                    width: `${progressPercent}%`,
                    transition: "width 200ms",
                  }}
                />
              </div>
            </div>
          )}

          {tokenOffset > 0 && (
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: T.textMuted }}>
              <span>Dataloader Offset:</span>
              <span style={{ fontFamily: T.fontMono }}>{tokenOffset.toLocaleString()} tok</span>
            </div>
          )}
        </div>
      )}

      <div>
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

      <Handle type="source" position={(data?.sourcePosition as Position) ?? Position.Right} />
    </div>
  );
}

export function DetokenizerVirtualNode({ data }: NodeProps): JSX.Element {
  const isActive = data.isActiveNode as boolean;
  
  // Dynamic single-head or multi-head MTP rendering
  const generatedText = (data.generatedText as string) || "The cat sat on the mat";
  const outputToken = (data.outputToken as string) || "mat";

  // Check if MTP logits are provided, otherwise fallback to static default logits
  const mtpHeads: MtpHeadData[] = (data.mtp_logits as MtpHeadData[]) || [
    {
      position: 1,
      logits: [
        { token: "mat", prob: "91.2%", color: "var(--vb-success)" },
        { token: "wall", prob: "4.1%", color: T.textSecondary },
        { token: "bed", prob: "2.3%", color: T.textSecondary },
      ]
    }
  ];

  return (
    <div
      role="group"
      aria-label="virtual detokenizer"
      data-testid="detokenizer-virtual-node"
      className={`vb-node ${isActive ? "vb-node-selected" : ""}`}
      style={{
        minWidth: mtpHeads.length > 1 ? 380 : 260,
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
        display: "flex",
        flexDirection: "column",
        gap: 12,
        transition: "min-width 200ms ease",
      }}
    >
      <Handle type="target" position={(data?.targetPosition as Position) ?? Position.Left} />

      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 16 }}>🔡</span>
        <div style={{ fontWeight: 600, fontSize: 14 }}>De-Tokenizer</div>
        <span
          style={{
            marginLeft: "auto",
            fontSize: 9,
            padding: "2px 6px",
            borderRadius: "var(--vb-radius-pill)",
            background: mtpHeads.length > 1 ? "rgba(34, 211, 238, 0.16)" : "rgba(52, 211, 153, 0.16)",
            color: mtpHeads.length > 1 ? "var(--vb-accent)" : "var(--vb-success)",
            fontWeight: 700,
            textTransform: "uppercase",
          }}
        >
          {mtpHeads.length > 1 ? `MTP K=${mtpHeads.length}` : "Virtual"}
        </span>
      </div>

      {/* Multi-Column or Single-Column Logits Area */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <div
          style={{
            fontSize: 10,
            color: T.textSecondary,
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          Prediction Logits
        </div>

        <div style={{ display: "grid", gridTemplateColumns: `repeat(${mtpHeads.length}, 1fr)`, gap: 8 }}>
          {mtpHeads.map((head, hIdx) => (
            <div
              key={hIdx}
              className="glass-panel"
              style={{
                padding: 6,
                borderRadius: "var(--vb-radius-sm)",
                background: "rgba(255, 255, 255, 0.02)",
                border: "1px solid var(--vb-border-soft)",
                display: "flex",
                flexDirection: "column",
                gap: 4,
              }}
            >
              <div style={{ fontSize: 9, fontWeight: 700, color: "var(--vb-accent)", textTransform: "uppercase" }}>
                Pos +{head.position}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {head.logits.map((item, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "3px 6px",
                      background: idx === 0 ? "rgba(52, 211, 153, 0.08)" : "var(--vb-surface-3)",
                      border: `1px solid ${idx === 0 ? "rgba(52, 211, 153, 0.4)" : "var(--vb-border)"}`,
                      borderRadius: "var(--vb-radius-sm)",
                      fontSize: 10,
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
          {generatedText.includes(outputToken) ? (
            <>
              {generatedText.split(outputToken)[0]}
              <span style={{ color: "var(--vb-success)", fontWeight: 600 }}>{outputToken}</span>
              {generatedText.split(outputToken)[1]}
            </>
          ) : (
            generatedText
          )}
        </div>
      </div>
    </div>
  );
}
