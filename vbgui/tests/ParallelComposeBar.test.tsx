import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ParallelComposeBar } from "@/components/ParallelComposeBar";

describe("V7-F57 ParallelComposeBar", () => {
  it("renders the compose button", () => {
    render(<ParallelComposeBar onCompose={() => {}} />);
    expect(screen.getByTestId("parallel-compose-tiny-aya"))
      .toBeDefined();
  });

  it("Compose fires with 5 nodes (input, attn, mlp, join, norm)", () => {
    const onC = vi.fn();
    render(<ParallelComposeBar onCompose={onC} />);
    fireEvent.click(screen.getByTestId("parallel-compose-tiny-aya"));
    expect(onC).toHaveBeenCalledTimes(1);
    const [nodes, edges] = onC.mock.calls[0];
    expect(nodes.length).toBe(5);
    const ids = nodes.map((n: { id: string }) => n.id);
    expect(ids).toEqual([
      "aya_input", "aya_attn", "aya_mlp", "aya_join", "aya_norm",
    ]);
    // 5 edges describing the parallel fan-out + join + norm.
    expect(edges.length).toBe(5);
  });

  it("emits parallel fan-out edges (input → attn, input → mlp)", () => {
    const onC = vi.fn();
    render(<ParallelComposeBar onCompose={onC} />);
    fireEvent.click(screen.getByTestId("parallel-compose-tiny-aya"));
    const [, edges] = onC.mock.calls[0];
    const fromInput = edges.filter(
      (e: { source: string }) => e.source === "aya_input");
    expect(fromInput.length).toBe(2);
    const fromInputTargets = fromInput
      .map((e: { target: string }) => e.target).sort();
    expect(fromInputTargets).toEqual(["aya_attn", "aya_mlp"]);
  });

  it("emits the residual-add join with two inbound edges", () => {
    const onC = vi.fn();
    render(<ParallelComposeBar onCompose={onC} />);
    fireEvent.click(screen.getByTestId("parallel-compose-tiny-aya"));
    const [, edges] = onC.mock.calls[0];
    const intoJoin = edges.filter(
      (e: { target: string }) => e.target === "aya_join");
    expect(intoJoin.length).toBe(2);
    const sources = intoJoin
      .map((e: { source: string }) => e.source).sort();
    expect(sources).toEqual(["aya_attn", "aya_mlp"]);
  });
});
