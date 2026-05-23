// AutoGroupButton — calls suggest_optim_groups and replaces draft groups.

import { useState } from "react";
import type { RpcClient } from "@/lib/rpc";
import type { Node, Edge } from "@xyflow/react";
import type { OptimKind, ParamGroupState } from "@/state/spec";

export interface ProposedGroupClient {
  matcher: string;
  optim_kind: string;
  lr: number;
  weight_decay: number;
  betas: [number, number] | null;
  ns_steps: number | null;
  param_count: number;
  rationale: string;
}

export interface AutoGroupResultClient {
  proposals: ProposedGroupClient[];
  total_params: number;
  uncovered_params: number;
}

export interface AutoGroupButtonProps {
  rpc: RpcClient | null;
  optimKind: OptimKind;
  nodes: Node[];
  edges: Edge[];
  hiddenSize?: number;
  onApply: (groups: ParamGroupState[], banner: string) => void;
  /** V7-H44: current draft groups (used for side-by-side diff render).
   *  When absent, button keeps the legacy auto-apply behaviour. */
  currentGroups?: ParamGroupState[];
}

function nodesToGraph(nodes: Node[], edges: Edge[]) {
  const modelNodes = nodes.filter(
    (n) => n.type !== "tokenizer_virtual" && n.type !== "detokenizer_virtual"
  );
  const modelNodeIds = new Set(modelNodes.map((n) => n.id));
  const modelEdges = edges.filter(
    (e) => modelNodeIds.has(e.source) && modelNodeIds.has(e.target)
  );

  return {
    nodes: modelNodes.map((n) => {
      const data = n.data as { kind?: string; params?: Record<string, unknown> };
      return { id: n.id, kind: data.kind ?? "mlp", params: data.params ?? {} };
    }),
    edges: modelEdges.map((e) => ({ src: e.source, dst: e.target })),
  };
}

function proposalsToGroups(
  proposals: ProposedGroupClient[]): ParamGroupState[] {
  return proposals.map((p) => ({
    matcher: p.matcher,
    lr: p.lr,
    weight_decay: p.weight_decay,
    betas: p.betas ?? undefined,
    ns_steps: p.ns_steps,
  }));
}

function buildBanner(result: AutoGroupResultClient): string {
  return (
    `Auto-grouped ${result.proposals.length} group${
      result.proposals.length === 1 ? "" : "s"} covering ` +
    `${result.total_params - result.uncovered_params}/` +
    `${result.total_params} params:\n` +
    result.proposals.map((p) =>
      `  • ${p.optim_kind.toUpperCase()} on ${p.matcher} — ${p.rationale}`,
    ).join("\n")
  );
}

export function AutoGroupButton({
  rpc, optimKind, nodes, edges, hiddenSize = 128, onApply,
  currentGroups,
}: AutoGroupButtonProps): JSX.Element {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // V7-H44: when currentGroups prop is supplied, fetch into staged
  // proposals instead of auto-applying. User can then diff side-by-side
  // and Accept-all / Discard / merge per-group.
  const [proposed, setProposed] = useState<AutoGroupResultClient | null>(null);

  async function run() {
    if (!rpc) { setError("rpc unavailable"); return; }
    if (nodes.length === 0) { setError("canvas is empty"); return; }
    setLoading(true);
    setError(null);
    try {
      const result = await rpc.call<AutoGroupResultClient>(
        "suggest_optim_groups",
        {
          graph: nodesToGraph(nodes, edges),
          optim_kind: optimKind,
          hidden_size: hiddenSize,
        },
      );
      if (currentGroups !== undefined) {
        // V7-H44: diff mode — stage proposals, wait for user click.
        setProposed(result);
      } else {
        // Legacy path: auto-apply.
        onApply(proposalsToGroups(result.proposals), buildBanner(result));
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <span style={{ display: "inline-flex", alignItems: "center",
                   gap: 6, flexWrap: "wrap" }}>
      <button data-testid="optim-auto-group"
              disabled={loading || nodes.length === 0}
              onClick={run}
              title={currentGroups !== undefined
                ? "Suggest matcher groups for comparison"
                : "Auto-classify params into matcher groups"}
              style={{ background: "#2563eb", color: "white",
                       border: "none", padding: "4px 8px",
                       borderRadius: 4, cursor: "pointer",
                       fontSize: 11 }}>
        {loading ? "Analysing…"
          : currentGroups !== undefined
            ? "Suggest groups"
            : "Auto-group from graph"}
      </button>
      {error && (
        <span data-testid="optim-auto-group-error"
              style={{ color: "#dc2626", fontSize: 11 }}>{error}</span>
      )}
      {proposed && currentGroups !== undefined && (
        <div data-testid="optim-auto-group-diff"
             style={{ width: "100%", marginTop: 6,
                      border: "1px solid #c7d2fe",
                      background: "#eef2ff", padding: 6,
                      fontSize: 11, fontFamily: "system-ui, sans-serif" }}>
          <div data-testid="optim-auto-group-diff-summary"
               style={{ marginBottom: 4 }}>
            Backend suggests <strong>{proposed.proposals.length}</strong> group
            {proposed.proposals.length === 1 ? "" : "s"} ·
            current has <strong>{currentGroups.length}</strong>
          </div>
          <table style={{ borderCollapse: "collapse", fontSize: 11,
                          marginBottom: 6, width: "100%" }}>
            <thead>
              <tr style={{ background: "#c7d2fe" }}>
                <th style={{ padding: "2px 6px", textAlign: "left" }}>
                  matcher
                </th>
                <th style={{ padding: "2px 6px", textAlign: "right" }}>
                  current lr
                </th>
                <th style={{ padding: "2px 6px", textAlign: "right" }}>
                  suggested lr
                </th>
                <th style={{ padding: "2px 6px", textAlign: "left" }}>
                  rationale
                </th>
              </tr>
            </thead>
            <tbody>
              {proposed.proposals.map((p) => {
                const current = currentGroups.find(
                  (g) => g.matcher === p.matcher);
                const lrDelta = current && current.lr !== p.lr;
                return (
                  <tr key={p.matcher}
                      data-testid={`optim-diff-row-${p.matcher}`}>
                    <td style={{ padding: "2px 6px" }}>{p.matcher}</td>
                    <td style={{ padding: "2px 6px", textAlign: "right",
                                 color: lrDelta ? "#b91c1c" : "#374151" }}>
                      {current ? current.lr.toExponential(2) : "—"}
                    </td>
                    <td style={{ padding: "2px 6px", textAlign: "right",
                                 color: lrDelta ? "#059669" : "#374151" }}>
                      {p.lr.toExponential(2)}
                    </td>
                    <td style={{ padding: "2px 6px",
                                 color: "#4b5563" }}>
                      {p.rationale}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <button data-testid="optim-auto-group-accept-all"
                  onClick={() => {
                    onApply(proposalsToGroups(proposed.proposals),
                            buildBanner(proposed));
                    setProposed(null);
                  }}
                  style={{ background: "#059669", color: "white",
                           border: "none", padding: "3px 8px",
                           borderRadius: 4, cursor: "pointer",
                           marginRight: 6 }}>
            Accept all
          </button>
          <button data-testid="optim-auto-group-discard"
                  onClick={() => setProposed(null)}
                  style={{ background: "transparent", color: "#374151",
                           border: "1px solid #9ca3af",
                           padding: "3px 8px", borderRadius: 4,
                           cursor: "pointer" }}>
            Discard
          </button>
        </div>
      )}
    </span>
  );
}
