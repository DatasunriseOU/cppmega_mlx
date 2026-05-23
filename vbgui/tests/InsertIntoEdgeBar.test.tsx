import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { InsertIntoEdgeBar } from "@/components/InsertIntoEdgeBar";

describe("V7-F51 InsertIntoEdgeBar", () => {
  it("renders brick dropdown listing the bricks catalog", () => {
    render(<InsertIntoEdgeBar edges={[]} onInsert={() => {}} />);
    const sel = screen.getByTestId(
      "insert-edge-brick-kind") as HTMLSelectElement;
    const opts = Array.from(sel.options).map((o) => o.value);
    expect(opts).toContain("mlstm");
    expect(opts).toContain("attention");
  });

  it("defaults brick selection to mlstm (the F51 honest closure target)", () => {
    render(<InsertIntoEdgeBar edges={[]} onInsert={() => {}} />);
    const sel = screen.getByTestId(
      "insert-edge-brick-kind") as HTMLSelectElement;
    expect(sel.value).toBe("mlstm");
  });

  it("Insert button is disabled when there are no edges", () => {
    render(<InsertIntoEdgeBar edges={[]} onInsert={() => {}} />);
    const btn = screen.getByTestId("insert-edge-go") as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });

  it("Insert fires onInsert with the chosen kind + edge pair", () => {
    const onI = vi.fn();
    render(<InsertIntoEdgeBar
      edges={[{ source: "attn", target: "mlp" }]}
      onInsert={onI}
    />);
    fireEvent.click(screen.getByTestId("insert-edge-go"));
    expect(onI).toHaveBeenCalledTimes(1);
    const [kind, edge] = onI.mock.calls[0];
    expect(kind).toBe("mlstm");
    expect(edge).toEqual({ source: "attn", target: "mlp" });
  });

  it("lists each edge as an option labelled src → target", () => {
    render(<InsertIntoEdgeBar
      edges={[
        { source: "attn", target: "mlp" },
        { source: "mlp", target: "head" },
      ]}
      onInsert={() => {}}
    />);
    const sel = screen.getByTestId(
      "insert-edge-target") as HTMLSelectElement;
    expect(sel.options.length).toBe(2);
    expect(sel.options[0].textContent).toContain("attn → mlp");
    expect(sel.options[1].textContent).toContain("mlp → head");
  });

  it("changing brick + edge selection feeds into the onInsert call", () => {
    const onI = vi.fn();
    render(<InsertIntoEdgeBar
      edges={[
        { source: "a", target: "b" },
        { source: "b", target: "c" },
      ]}
      onInsert={onI}
    />);
    fireEvent.change(screen.getByTestId("insert-edge-brick-kind"),
                     { target: { value: "gated_attention" } });
    fireEvent.change(screen.getByTestId("insert-edge-target"),
                     { target: { value: "b->c" } });
    fireEvent.click(screen.getByTestId("insert-edge-go"));
    expect(onI).toHaveBeenCalledWith("gated_attention",
                                     { source: "b", target: "c" });
  });
});
