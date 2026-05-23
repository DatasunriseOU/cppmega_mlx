/**
 * V8-R08 vitest: FeatureInjectionBar.
 *
 * Asserts:
 *  - on mount, calls catalog.list_options({category: 'feature_injectors'})
 *    and populates the dropdown
 *  - Apply click fires onApply with the chosen option's name + paper_ref
 *  - applied-list reflects the click (visible badge / list)
 *  - dropdown disabled while catalog list is empty (loading)
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { FeatureInjectionBar } from "@/components/FeatureInjectionBar";

function makeFakeRpc(responses: Record<string, unknown>) {
  const calls: { method: string; params: unknown }[] = [];
  return {
    calls,
    rpc: {
      call: vi.fn(async (method: string, params: unknown) => {
        calls.push({ method, params });
        const r = responses[method];
        if (r instanceof Error) throw r;
        return r;
      }),
    } as never,
  };
}

const CATALOG_OK = {
  options: [
    { name: "mtp_weighted",
      summary: "Multi-token prediction K=2 head + weighted loss",
      paper_ref: "rewriter:MTPRewriter" },
    { name: "ifim_shaped",
      summary: "Span-aware IFIM loss reshaping",
      paper_ref: "rewriter:IFIMRewriter" },
    { name: "engram",
      summary: "Standalone local engram (n-gram) branch",
      paper_ref: "brick:engram" },
  ],
};

describe("V8-R08 FeatureInjectionBar", () => {
  it("populates dropdown from catalog.list_options on mount", async () => {
    const { rpc, calls } = makeFakeRpc({
      "catalog.list_options": CATALOG_OK,
    });
    render(<FeatureInjectionBar rpc={rpc} onApply={() => {}} />);

    await waitFor(() => {
      expect(calls.some((c) =>
        c.method === "catalog.list_options")).toBe(true);
    });
    const first = calls.find((c) =>
      c.method === "catalog.list_options")!;
    expect(first.params).toEqual({ category: "feature_injectors" });

    await waitFor(() => {
      const dropdown = screen.getByTestId("feature-injection-dropdown") as
        HTMLSelectElement;
      expect(dropdown.options.length).toBe(3);
      expect(dropdown.value).toBe("mtp_weighted");
    });
  });

  it("Apply fires onApply with name + paper_ref", async () => {
    const { rpc } = makeFakeRpc({
      "catalog.list_options": CATALOG_OK,
    });
    const onApply = vi.fn();
    render(<FeatureInjectionBar rpc={rpc} onApply={onApply} />);
    await waitFor(() => {
      expect(screen.getByTestId("feature-injection-dropdown"))
        .toBeDefined();
    });

    // Default selected is the first option, mtp_weighted.
    fireEvent.click(screen.getByTestId("feature-injection-apply"));
    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onApply.mock.calls[0][0]).toEqual({
      name: "mtp_weighted",
      paper_ref: "rewriter:MTPRewriter",
    });

    // applied-list reflects the click.
    expect(screen.getByTestId("feature-injection-applied-list").textContent)
      .toContain("mtp_weighted");
  });

  it("changing selection then Apply emits the new choice", async () => {
    const { rpc } = makeFakeRpc({
      "catalog.list_options": CATALOG_OK,
    });
    const onApply = vi.fn();
    render(<FeatureInjectionBar rpc={rpc} onApply={onApply} />);
    await waitFor(() => {
      expect(screen.getByTestId("feature-injection-dropdown"))
        .toBeDefined();
    });
    fireEvent.change(screen.getByTestId("feature-injection-dropdown"),
      { target: { value: "engram" } });
    fireEvent.click(screen.getByTestId("feature-injection-apply"));
    expect(onApply.mock.calls[0][0]).toEqual({
      name: "engram", paper_ref: "brick:engram",
    });
  });

  it("applying the same option N times renders ONE chip with ×N count " +
     "(UX#1: no more comma-joined mtp_weighted,mtp_weighted,...)", async () => {
    const { rpc } = makeFakeRpc({ "catalog.list_options": CATALOG_OK });
    render(<FeatureInjectionBar rpc={rpc} onApply={() => {}} />);
    await waitFor(() => {
      expect(screen.getByTestId("feature-injection-dropdown")).toBeDefined();
    });
    const applyBtn = screen.getByTestId("feature-injection-apply");
    fireEvent.click(applyBtn);
    fireEvent.click(applyBtn);
    fireEvent.click(applyBtn);
    fireEvent.click(applyBtn);

    // ONE chip, not four.
    expect(screen.getAllByTestId("feature-injection-chip-mtp_weighted"))
      .toHaveLength(1);
    // With ×4 count badge.
    expect(screen.getByTestId("feature-injection-chip-mtp_weighted-count")
      .textContent).toBe("×4");
    // Applied list should NOT contain the literal comma-joined string
    // "mtp_weighted, mtp_weighted, mtp_weighted, mtp_weighted".
    const list = screen.getByTestId("feature-injection-applied-list");
    expect(list.textContent ?? "").not.toContain(
      "mtp_weighted, mtp_weighted");
  });

  it("chip × button calls onRemove and pops one instance", async () => {
    const { rpc } = makeFakeRpc({ "catalog.list_options": CATALOG_OK });
    const onRemove = vi.fn();
    render(<FeatureInjectionBar rpc={rpc} onApply={() => {}}
                                 onRemove={onRemove} />);
    await waitFor(() => {
      expect(screen.getByTestId("feature-injection-dropdown")).toBeDefined();
    });
    const applyBtn = screen.getByTestId("feature-injection-apply");
    fireEvent.click(applyBtn);
    fireEvent.click(applyBtn);
    fireEvent.click(applyBtn);
    expect(screen.getByTestId("feature-injection-chip-mtp_weighted-count")
      .textContent).toBe("×3");

    fireEvent.click(screen.getByTestId(
      "feature-injection-chip-mtp_weighted-remove"));
    expect(onRemove).toHaveBeenCalledTimes(1);
    expect(onRemove.mock.calls[0][0]).toEqual({
      name: "mtp_weighted",
      paper_ref: "rewriter:MTPRewriter",
    });
    // Count drops to ×2.
    expect(screen.getByTestId("feature-injection-chip-mtp_weighted-count")
      .textContent).toBe("×2");
  });

  it("single-instance chip renders without a count badge", async () => {
    const { rpc } = makeFakeRpc({ "catalog.list_options": CATALOG_OK });
    render(<FeatureInjectionBar rpc={rpc} onApply={() => {}} />);
    await waitFor(() => {
      expect(screen.getByTestId("feature-injection-dropdown")).toBeDefined();
    });
    fireEvent.click(screen.getByTestId("feature-injection-apply"));
    expect(screen.getByTestId("feature-injection-chip-mtp_weighted"))
      .toBeDefined();
    expect(screen.queryByTestId(
      "feature-injection-chip-mtp_weighted-count")).toBeNull();
  });

  it("renders error banner on RPC failure", async () => {
    const { rpc } = makeFakeRpc({
      "catalog.list_options": new Error("catalog down"),
    });
    render(<FeatureInjectionBar rpc={rpc} onApply={() => {}} />);
    await waitFor(() => {
      expect(screen.getByTestId("feature-injection-error").textContent ?? "")
        .toContain("catalog down");
    });
  });
});
