import { describe, it, expect } from "vitest";
import type { Node, Edge } from "@xyflow/react";
import { groupRepeatedNodes } from "@/App";

// Mirrors the Gemma preset chain that the old name-based detector failed to
// fold: a non-numbered per_layer_embed prefix in front of N identical
// gqa_sliding layers, then a different attention + moe tail.
function gemmaChain(): { nodes: Node[]; edges: Edge[] } {
  const slideParams = { sliding_window_size: 1024, head_dim: 64 };
  const nodes: Node[] = [
    { id: "ple", type: "brick", position: { x: 0, y: 0 }, data: { kind: "per_layer_embed", params: {} } },
    ...[0, 1, 2, 3, 4].map((i) => ({
      id: `gemma_sw_${i}`,
      type: "brick",
      position: { x: 0, y: 0 },
      data: { kind: "gqa_sliding", params: { ...slideParams } },
    })) as Node[],
    { id: "gemma_global", type: "brick", position: { x: 0, y: 0 }, data: { kind: "gated_attention", params: {} } },
    { id: "gemma_moe", type: "brick", position: { x: 0, y: 0 }, data: { kind: "moe", params: {} } },
  ];
  const order = ["ple", "gemma_sw_0", "gemma_sw_1", "gemma_sw_2", "gemma_sw_3", "gemma_sw_4", "gemma_global", "gemma_moe"];
  const edges: Edge[] = [];
  for (let i = 0; i < order.length - 1; i++) {
    edges.push({ id: `${order[i]}->${order[i + 1]}`, source: order[i], target: order[i + 1] });
  }
  return { nodes, edges };
}

describe("groupRepeatedNodes", () => {
  it("folds the 5 identical gqa_sliding layers into one block_group", () => {
    const { nodes, edges } = gemmaChain();
    const res = groupRepeatedNodes(nodes, edges);

    const groups = res.nodes.filter((n) => n.type === "block_group");
    expect(groups).toHaveLength(1);
    expect((groups[0].data as any).repeats).toBe(5);
    expect((groups[0].data as any).block_specs[0].kind).toBe("gqa_sliding");

    // The 5 sw bricks are gone; ple/global/moe survive.
    const ids = new Set(res.nodes.map((n) => n.id));
    expect(ids.has("gemma_sw_0")).toBe(false);
    expect(ids.has("gemma_sw_4")).toBe(false);
    expect(ids.has("ple")).toBe(true);
    expect(ids.has("gemma_global")).toBe(true);

    // Edges rewired: ple -> group -> global, no dangling internal edges.
    const gid = groups[0].id;
    expect(res.edges.some((e) => e.source === "ple" && e.target === gid)).toBe(true);
    expect(res.edges.some((e) => e.source === gid && e.target === "gemma_global")).toBe(true);
    expect(res.edges.every((e) => !e.source.startsWith("gemma_sw_") && !e.target.startsWith("gemma_sw_"))).toBe(true);
  });

  it("does not fold distinct adjacent kinds", () => {
    const nodes: Node[] = [
      { id: "a", type: "brick", position: { x: 0, y: 0 }, data: { kind: "attention", params: {} } },
      { id: "b", type: "brick", position: { x: 0, y: 0 }, data: { kind: "mlp", params: {} } },
    ];
    const edges: Edge[] = [{ id: "a->b", source: "a", target: "b" }];
    const res = groupRepeatedNodes(nodes, edges);
    expect(res.nodes.filter((n) => n.type === "block_group")).toHaveLength(0);
  });

  it("does not fold identical bricks with differing params", () => {
    const nodes: Node[] = [
      { id: "a", type: "brick", position: { x: 0, y: 0 }, data: { kind: "gqa_sliding", params: { sliding_window_size: 512 } } },
      { id: "b", type: "brick", position: { x: 0, y: 0 }, data: { kind: "gqa_sliding", params: { sliding_window_size: 1024 } } },
    ];
    const edges: Edge[] = [{ id: "a->b", source: "a", target: "b" }];
    const res = groupRepeatedNodes(nodes, edges);
    expect(res.nodes.filter((n) => n.type === "block_group")).toHaveLength(0);
  });

  it("never re-folds nodes the user explicitly unpacked", () => {
    const nodes: Node[] = [0, 1, 2].map((i) => ({
      id: `u${i}`,
      type: "brick",
      position: { x: 0, y: 0 },
      data: { kind: "gqa_sliding", params: { sliding_window_size: 1024 }, _unpacked: true },
    })) as Node[];
    const edges: Edge[] = [
      { id: "u0->u1", source: "u0", target: "u1" },
      { id: "u1->u2", source: "u1", target: "u2" },
    ];
    const res = groupRepeatedNodes(nodes, edges);
    expect(res.nodes.filter((n) => n.type === "block_group")).toHaveLength(0);
  });

  it("never folds embedding_table I/O bricks", () => {
    const nodes: Node[] = [
      { id: "in", type: "brick", position: { x: 0, y: 0 }, data: { kind: "embedding_table", params: { vocab_size: 65536 } } },
      { id: "out", type: "brick", position: { x: 0, y: 0 }, data: { kind: "embedding_table", params: { vocab_size: 65536 } } },
    ];
    const edges: Edge[] = [{ id: "in->out", source: "in", target: "out" }];
    const res = groupRepeatedNodes(nodes, edges);
    expect(res.nodes.filter((n) => n.type === "block_group")).toHaveLength(0);
  });
});
