import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ExplainModal } from "@/components/ExplainModal";
import { _clearCatalogMemo } from "@/hooks/useCatalog";

const LION = {
  entry: {
    category: "optimizer",
    name: "lion",
    summary: "Sign-based momentum.",
    when_to_use: "Memory-constrained.",
    when_to_avoid: "Small batch.",
    recommended_params: { lr: 1e-4, weight_decay: 0.01 },
    paper_ref: "Chen 2023",
    paper_url: "https://arxiv.org/abs/2302.06675",
    gotchas: ["lr > 5e-4 → NaN", "Less stable <100M"],
  },
  not_found_message: null,
};
const NOT_FOUND = {
  entry: null,
  not_found_message: "no entry for category='optimizer', name='sophia'",
};

function fakeRpc(payload: unknown) {
  return {
    call: vi.fn(async () => payload),
  } as never;
}

beforeEach(() => _clearCatalogMemo());

describe("ExplainModal", () => {
  it("renders title with category badge", async () => {
    render(<ExplainModal rpc={fakeRpc(LION)} category="optimizer"
                          name="lion" onClose={() => {}} />);
    await waitFor(() => {
      expect(screen.getByTestId("explain-modal-title").textContent)
        .toContain("lion");
      expect(screen.getByTestId("explain-modal-title").textContent)
        .toContain("optimizer");
    });
  });

  it("renders all 5 sections of an entry", async () => {
    render(<ExplainModal rpc={fakeRpc(LION)} category="optimizer"
                          name="lion" onClose={() => {}} />);
    await waitFor(() => {
      expect(screen.getByTestId("explain-modal-summary").textContent)
        .toContain("Sign-based");
    });
    expect(screen.getByTestId("explain-modal-when-to-use")).toBeTruthy();
    expect(screen.getByTestId("explain-modal-when-to-avoid")).toBeTruthy();
    expect(screen.getByTestId("explain-modal-recommended")).toBeTruthy();
    expect(screen.getByTestId("explain-modal-gotchas")).toBeTruthy();
  });

  it("renders paper link with target=_blank", async () => {
    render(<ExplainModal rpc={fakeRpc(LION)} category="optimizer"
                          name="lion" onClose={() => {}} />);
    await waitFor(() => {
      const a = screen.getByTestId("explain-modal-paper") as HTMLAnchorElement;
      expect(a.href).toContain("arxiv.org/abs/2302.06675");
      expect(a.target).toBe("_blank");
    });
  });

  it("close button + backdrop click fire onClose", async () => {
    const onClose = vi.fn();
    render(<ExplainModal rpc={fakeRpc(LION)} category="optimizer"
                          name="lion" onClose={onClose} />);
    fireEvent.click(screen.getByTestId("explain-modal-close"));
    expect(onClose).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByTestId("explain-modal-backdrop"));
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("clicking inside the modal does NOT close", async () => {
    const onClose = vi.fn();
    render(<ExplainModal rpc={fakeRpc(LION)} category="optimizer"
                          name="lion" onClose={onClose} />);
    await waitFor(() => screen.getByTestId("explain-modal"));
    fireEvent.click(screen.getByTestId("explain-modal"));
    expect(onClose).not.toHaveBeenCalled();
  });

  it("Apply recommended button fires callback with params", async () => {
    const onApply = vi.fn();
    const onClose = vi.fn();
    render(<ExplainModal rpc={fakeRpc(LION)} category="optimizer"
                          name="lion" onClose={onClose}
                          onApplyRecommended={onApply} />);
    await waitFor(() => screen.getByTestId("explain-modal-apply"));
    fireEvent.click(screen.getByTestId("explain-modal-apply"));
    expect(onApply).toHaveBeenCalledWith({ lr: 1e-4, weight_decay: 0.01 });
    expect(onClose).toHaveBeenCalled();
  });

  it("shows error envelope when entry not found", async () => {
    render(<ExplainModal rpc={fakeRpc(NOT_FOUND)} category="optimizer"
                          name="sophia" onClose={() => {}} />);
    await waitFor(() => {
      expect(screen.getByTestId("explain-modal-error").textContent)
        .toContain("no entry");
    });
  });

  it("renders 2 gotchas as list items", async () => {
    render(<ExplainModal rpc={fakeRpc(LION)} category="optimizer"
                          name="lion" onClose={() => {}} />);
    await waitFor(() => {
      const items = screen.getByTestId("explain-modal-gotchas")
                          .querySelectorAll("li");
      expect(items.length).toBe(2);
    });
  });
});
