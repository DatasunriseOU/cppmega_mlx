// V7-H33: LossSurfaceModal — N×M heatmap of (lr_mult × wd_mult)
// triggered by RunResultModal "Explore loss surface" button. Calls
// loss_surface.run RPC, renders the best cell + Apply-best button that
// commits the chosen lr/wd multipliers back to the optim spec.

import { useState } from "react";
import type { RpcClient } from "@/lib/rpc";
import { T } from "@/theme";
import { HelpIcon } from "@/components/HelpIcon";

export interface LossSurfaceCell {
  lr_mult: number;
  wd_mult: number;
  status: string;
  final_loss?: number | null;
  throughput_tok_s?: number | null;
  mem_mb?: number | null;
  elapsed_ms?: number;
}

export interface LossSurfaceResult {
  rows: LossSurfaceCell[][];
  lr_deltas: number[];
  wd_deltas: number[];
  best_lr_mult: number | null;
  best_wd_mult: number | null;
  best_loss: number | null;
}

export interface LossSurfaceModalProps {
  rpc: RpcClient | null;
  spec: unknown;
  open: boolean;
  onClose: () => void;
  onApplyBest: (lrMult: number, wdMult: number) => void;
}

const DEFAULT_LR_DELTAS = [0.5, 1.0, 2.0];
const DEFAULT_WD_DELTAS = [0.5, 1.0, 2.0];

function cellColor(loss: number | null | undefined,
                    min: number | null, max: number | null): string {
  if (loss == null || min == null || max == null || max <= min) {
    return T.surface3;
  }
  const t = (loss - min) / (max - min);
  const r = Math.round(40 + 215 * t);
  const g = Math.round(180 - 120 * t);
  return `rgb(${r}, ${g}, 60)`;
}

export function LossSurfaceModal({
  rpc, spec, open, onClose, onApplyBest,
}: LossSurfaceModalProps): JSX.Element | null {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<LossSurfaceResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [kSteps, setKSteps] = useState<number>(2);

  if (!open) return null;

  async function run() {
    setRunning(true);
    setError(null);
    setResult(null);
    if (!rpc) {
      setError("no backend connection");
      setRunning(false);
      return;
    }
    try {
      const r = await rpc.call<LossSurfaceResult>("loss_surface.run", {
        spec,
        lr_deltas: DEFAULT_LR_DELTAS,
        wd_deltas: DEFAULT_WD_DELTAS,
        k_steps: kSteps,
      });
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  const allLosses = result?.rows.flat()
    .map((c) => c.final_loss)
    .filter((v): v is number => v != null) ?? [];
  const minLoss = allLosses.length ? Math.min(...allLosses) : null;
  const maxLoss = allLosses.length ? Math.max(...allLosses) : null;

  return (
    <div data-testid="loss-surface-modal"
         role="dialog" aria-modal="true"
         style={{ position: "fixed", inset: 0, zIndex: 1000,
                  background: "rgba(0, 0, 0, 0.7)", display: "flex",
                  alignItems: "center", justifyContent: "center",
                  padding: 24, fontFamily: T.font }}>
      <div style={{ background: T.surface,
                    border: `1px solid ${T.border}`,
                    color: T.text,
                    borderRadius: 8, padding: 18,
                    minWidth: 460, maxWidth: 720, maxHeight: "85vh",
                    boxShadow: T.shadowPop,
                    overflow: "auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between",
                      alignItems: "center", marginBottom: 8 }}>
          <h3 data-testid="loss-surface-title"
              style={{ margin: 0, fontSize: 16 }}>
            Loss surface · lr_mult × wd_mult
            <HelpIcon topic="loss_surface_explorer" />
          </h3>
          <button data-testid="loss-surface-close"
                  onClick={onClose}
                  style={{
                    background: "transparent", border: "none",
                    color: T.textSecondary, fontSize: 20, cursor: "pointer",
                    padding: 0, lineHeight: 1,
                  }}>×</button>
        </div>

        <div style={{ fontSize: 12, marginBottom: 8, color: T.textSecondary }}>
          <label style={{ marginRight: 8 }}>k_steps:</label>
          <input data-testid="loss-surface-k-steps"
                 type="number" min={1} max={64} value={kSteps}
                 onChange={(e) =>
                   setKSteps(Math.max(1, Math.min(64,
                     Number(e.target.value) || 2)))}
                 style={{
                   width: 60,
                   background: T.surface3,
                   border: `1px solid ${T.border}`,
                   color: T.text,
                   borderRadius: 4,
                   padding: "2px 6px",
                 }} />
          <button data-testid="loss-surface-run"
                  onClick={() => void run()}
                  disabled={running}
                  style={{
                    marginLeft: 10,
                    background: T.accent,
                    color: "#0f172a",
                    border: "none",
                    borderRadius: 4,
                    padding: "4px 10px",
                    fontWeight: "bold",
                    cursor: "pointer",
                  }}>
            {running ? "Running…" : "Run sweep"}
          </button>
        </div>

        {error && (
          <div data-testid="loss-surface-error"
               style={{ color: T.danger, fontSize: 12 }}>
            {error}
          </div>
        )}

        {result && (
          <div data-testid="loss-surface-result">
            <table style={{ borderCollapse: "collapse",
                            margin: "8px 0" }}>
              <thead>
                <tr>
                  <th style={{ padding: "4px 8px", fontSize: 11 }}>
                    lr↓ / wd→
                  </th>
                  {result.wd_deltas.map((wd) => (
                    <th key={wd}
                        data-testid={`loss-surface-col-${wd}`}
                        style={{ padding: "4px 8px", fontSize: 11 }}>
                      ×{wd}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row, ri) => (
                  <tr key={ri}>
                    <th data-testid={`loss-surface-row-${result.lr_deltas[ri]}`}
                        style={{ padding: "4px 8px", fontSize: 11 }}>
                      ×{result.lr_deltas[ri]}
                    </th>
                    {row.map((cell, ci) => {
                      const isBest = result.best_lr_mult === cell.lr_mult
                                   && result.best_wd_mult === cell.wd_mult;
                      return (
                        <td key={ci}
                            data-testid={`loss-surface-cell-${ri}-${ci}`}
                            style={{
                              padding: "10px 14px",
                              background: cellColor(cell.final_loss,
                                                   minLoss, maxLoss),
                              color: "white", fontSize: 11,
                              border: isBest
                                ? "3px solid #facc15" : `1px solid ${T.border}`,
                              textAlign: "center",
                              minWidth: 56,
                            }}>
                          {cell.status === "ok" && cell.final_loss != null
                            ? cell.final_loss.toFixed(3)
                            : cell.status}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>

            <div data-testid="loss-surface-best"
                 style={{ fontSize: 12, marginTop: 4, color: T.textSecondary }}>
              {result.best_loss != null ? (
                <>
                  best: lr×{result.best_lr_mult}, wd×{result.best_wd_mult}
                  {" → loss "}<strong style={{ color: T.success }}>{result.best_loss.toFixed(4)}</strong>
                </>
              ) : (
                "no successful cells"
              )}
            </div>

            {result.best_lr_mult != null && result.best_wd_mult != null && (
              <button data-testid="loss-surface-apply-best"
                      onClick={() => {
                        onApplyBest(result.best_lr_mult!,
                                    result.best_wd_mult!);
                        onClose();
                      }}
                      style={{ marginTop: 10, background: T.accent,
                               color: "#0f172a", border: "none",
                               padding: "6px 12px", borderRadius: 4,
                               cursor: "pointer", fontWeight: "bold" }}>
                Apply recommended multiplier overrides
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
