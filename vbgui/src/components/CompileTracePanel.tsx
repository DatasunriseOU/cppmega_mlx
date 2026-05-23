/**
 * V8-R06 CompileTracePanel — renders the compile.trace RPC output as
 * a per-op chip table with fused / dlpack-crossing / materialised
 * markers and an aggregate header line.
 */

import { useEffect, useState } from "react";
import type { RpcClient } from "@/lib/rpc";

interface CompileTraceOp {
  name: string;
  fused: boolean;
  group: string;
  materialised: boolean;
  dlpack_boundary: boolean;
  backend: string;
}

interface CompileTraceResult {
  ops: CompileTraceOp[];
  fused_groups: string[];
  dlpack_crossings: number;
  materialised_ops: string[];
  compile_artifact_path: string | null;
  backend: string;
}

interface SyncCheckEntry { after_op: string; reason: string }
interface SyncCheckAdvice { op: string; fix: string; confidence: string }
interface SyncCheckResult {
  necessary_syncs: SyncCheckEntry[];
  redundant_syncs: SyncCheckEntry[];
  advice: SyncCheckAdvice[];
  z3_solver_status: string;
  z3_elapsed_ms: number;
}

export interface CompileTracePanelProps {
  rpc: RpcClient;
  specPayload: unknown;
  backend?: "tilelang" | "torch_inductor" | "mlx";
}

export function CompileTracePanel({
  rpc, specPayload, backend = "mlx",
}: CompileTracePanelProps): JSX.Element {
  const [trace, setTrace] = useState<CompileTraceResult | null>(null);
  const [sync, setSync] = useState<SyncCheckResult | null>(null);
  const [showAdvice, setShowAdvice] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setErr(null);
      try {
        const [r, s] = await Promise.all([
          rpc.call<CompileTraceResult>(
            "compile.trace",
            { spec: specPayload, backend }),
          rpc.call<SyncCheckResult>(
            "sync.check",
            { spec: specPayload }).catch(() => null),
        ]);
        if (!cancelled) { setTrace(r); if (s) setSync(s); }
      } catch (e) {
        if (!cancelled) {
          setErr(e instanceof Error ? e.message : String(e));
        }
      }
    })();
    return () => { cancelled = true; };
  }, [rpc, specPayload, backend]);

  if (err) {
    return <div data-testid="compile-trace-error"
                style={{ color: "#b91c1c", padding: 12 }}>{err}</div>;
  }
  if (!trace) {
    return <div data-testid="compile-trace-loading"
                style={{ padding: 12, color: "#6b7280" }}>loading…</div>;
  }

  return (
    <div data-testid="compile-trace" style={{ padding: 12,
         fontFamily: "system-ui, sans-serif", fontSize: 12 }}>
      <header style={{ display: "flex", gap: 12, marginBottom: 8 }}>
        <span>backend: <strong>{trace.backend}</strong></span>
        <span data-testid="compile-trace-fused-count">
          fused groups: {trace.fused_groups.length}
        </span>
        <span data-testid="compile-trace-dlpack-crossings">
          dlpack crossings: {trace.dlpack_crossings}
        </span>
        <span data-testid="compile-trace-materialised-count">
          materialised: {trace.materialised_ops.length}
        </span>
        {sync && (
          <span data-testid="sync-check-badge"
                onClick={() => setShowAdvice(true)}
                style={{ marginLeft: "auto", cursor: "pointer",
                         background: sync.redundant_syncs.length > 0
                          ? "#fee2e2" : "#dcfce7",
                         color: sync.redundant_syncs.length > 0
                          ? "#991b1b" : "#166534",
                         padding: "2px 6px", borderRadius: 4 }}>
            sync: <span data-testid="sync-check-redundant-count">
              {sync.redundant_syncs.length}
            </span> redundant
          </span>
        )}
      </header>
      {showAdvice && sync && (
        <div data-testid="sync-check-advice-modal"
             onClick={() => setShowAdvice(false)}
             style={{ position: "fixed", inset: 0,
                      background: "rgba(0,0,0,0.4)", zIndex: 9000,
                      display: "flex", alignItems: "center",
                      justifyContent: "center" }}>
          <div onClick={(e) => e.stopPropagation()}
               style={{ background: "#fff", padding: 16, borderRadius: 8,
                        minWidth: 400, maxHeight: "60vh",
                        overflowY: "auto" }}>
            <h4 style={{ margin: "0 0 8px" }}>Sync-necessity advice</h4>
            <p style={{ margin: "0 0 8px", color: "#6b7280" }}>
              z3 status: {sync.z3_solver_status} ·
              {sync.z3_elapsed_ms.toFixed(1)} ms
            </p>
            {sync.advice.length === 0 ? (
              <p>No redundant syncs detected.</p>
            ) : (
              <ul>
                {sync.advice.map((a, i) => (
                  <li key={i}>
                    <strong>{a.op}</strong>: {a.fix}
                    {" "}({a.confidence})
                  </li>
                ))}
              </ul>
            )}
            <button onClick={() => setShowAdvice(false)}
                    style={{ marginTop: 8 }}>Close</button>
          </div>
        </div>
      )}
      {trace.fused_groups.map((g) => (
        <span key={g}
              data-testid={`compile-trace-fused-group-${g}`}
              style={{ display: "inline-block", marginRight: 6,
                       background: "#dbeafe", color: "#1e3a8a",
                       borderRadius: 4, padding: "2px 6px" }}>
          {g}
        </span>
      ))}
      <table style={{ marginTop: 8, borderCollapse: "collapse",
                      width: "100%" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
            <th style={th}>op</th>
            <th style={th}>backend</th>
            <th style={th}>group</th>
            <th style={th}>chips</th>
          </tr>
        </thead>
        <tbody>
          {trace.ops.map((op, i) => (
            <tr key={`${op.name}-${i}`}
                data-testid={`compile-trace-op-${i}`}>
              <td style={td}>{op.name}</td>
              <td style={td}>{op.backend}</td>
              <td style={td}>{op.group}</td>
              <td style={td}>
                {op.fused && (
                  <span style={chip("#dbeafe", "#1e3a8a")}>fused</span>
                )}
                {op.materialised && (
                  <span style={chip("#fef3c7", "#92400e")}
                        data-testid={
                          `compile-trace-op-${i}-materialised`}>
                    materialised
                  </span>
                )}
                {op.dlpack_boundary && (
                  <span style={chip("#fee2e2", "#7f1d1d")}
                        data-testid={
                          `compile-trace-op-${i}-dlpack`}>
                    dlpack
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const th: React.CSSProperties = {
  textAlign: "left", padding: "4px 6px", color: "#374151",
  fontWeight: 600,
};

const td: React.CSSProperties = {
  padding: "4px 6px", color: "#1f2937",
  borderBottom: "1px solid #f3f4f6",
};

function chip(bg: string, fg: string): React.CSSProperties {
  return {
    display: "inline-block", marginRight: 4, padding: "1px 6px",
    borderRadius: 4, background: bg, color: fg, fontSize: 10,
  };
}
