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
}

function nodesToGraph(nodes: Node[], edges: Edge[]) {
  return {
    nodes: nodes.map((n) => {
      const data = n.data as { kind?: string; params?: Record<string, unknown> };
      return { id: n.id, kind: data.kind ?? "mlp", params: data.params ?? {} };
    }),
    edges: edges.map((e) => ({ src: e.source, dst: e.target })),
  };
}

export function AutoGroupButton({
  rpc, optimKind, nodes, edges, hiddenSize = 128, onApply,
}: AutoGroupButtonProps): JSX.Element {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      const groups: ParamGroupState[] = result.proposals.map((p) => ({
        matcher: p.matcher,
        lr: p.lr,
        weight_decay: p.weight_decay,
        betas: p.betas ?? undefined,
      }));
      const banner =
        `Auto-grouped ${result.proposals.length} group${
          result.proposals.length === 1 ? "" : "s"} covering ` +
        `${result.total_params - result.uncovered_params}/${result.total_params} params:\n` +
        result.proposals.map((p) =>
          `  • ${p.optim_kind.toUpperCase()} on ${p.matcher} — ${p.rationale}`,
        ).join("\n");
      onApply(groups, banner);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <button data-testid="optim-auto-group"
              disabled={loading || nodes.length === 0}
              onClick={run}
              title="Auto-classify params into matcher groups"
              style={{ background: "#2563eb", color: "white",
                       border: "none", padding: "4px 8px",
                       borderRadius: 4, cursor: "pointer",
                       fontSize: 11 }}>
        {loading ? "Analysing…" : "Auto-group from graph"}
      </button>
      {error && (
        <span data-testid="optim-auto-group-error"
              style={{ color: "#dc2626", fontSize: 11 }}>{error}</span>
      )}
    </span>
  );
}
