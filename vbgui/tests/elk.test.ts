import { describe, it, expect } from "vitest";
import { layoutFlow } from "@/lib/elk";

describe("layoutFlow", () => {
  it("places three chained nodes left-to-right", async () => {
    const nodes = [
      { id: "a", type: "brick", position: { x: 0, y: 0 }, data: { kind: "mlp" } as never },
      { id: "b", type: "brick", position: { x: 0, y: 0 }, data: { kind: "mlp" } as never },
      { id: "c", type: "brick", position: { x: 0, y: 0 }, data: { kind: "mlp" } as never },
    ];
    const edges = [
      { id: "ab", source: "a", target: "b" },
      { id: "bc", source: "b", target: "c" },
    ];
    const { nodes: out } = await layoutFlow(nodes, edges);
    const xs = out.map((n) => n.position.x);
    expect(xs[0]).toBeLessThan(xs[1]);
    expect(xs[1]).toBeLessThan(xs[2]);
  });

  it("returns input nodes unchanged when graph is empty", async () => {
    const { nodes, edges } = await layoutFlow([], []);
    expect(nodes).toEqual([]);
    expect(edges).toEqual([]);
  });
});
