import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { BrickContextPanel } from "@/components/BrickContextPanel";

const RPC = { call: vi.fn(async () =>
                ({ entry: null, not_found_message: "x" })) } as never;

describe("BrickContextPanel", () => {
  it("shows activation dropdown for mlp brick", () => {
    render(<BrickContextPanel rpc={RPC} brickId="mlp_0" brickKind="mlp"
                                params={{}}
                                onApply={() => {}} onClose={() => {}} />);
    expect(screen.getByTestId("brick-context-mlp_0-activation")).toBeTruthy();
  });

  it("hides activation dropdown for attention brick", () => {
    render(<BrickContextPanel rpc={RPC} brickId="attn_0"
                                brickKind="attention" params={{}}
                                onApply={() => {}} onClose={() => {}} />);
    expect(screen.queryByTestId("brick-context-attn_0-activation")).toBeNull();
  });

  it("shows pre_norm / post_norm dropdowns for attention", () => {
    render(<BrickContextPanel rpc={RPC} brickId="attn_0"
                                brickKind="attention" params={{}}
                                onApply={() => {}} onClose={() => {}} />);
    expect(screen.getByTestId("brick-context-attn_0-pre-norm")).toBeTruthy();
    expect(screen.getByTestId("brick-context-attn_0-post-norm")).toBeTruthy();
  });

  it("V7-F52: shows swap dropdown listing same-category bricks", () => {
    const onSwap = vi.fn();
    render(<BrickContextPanel rpc={RPC} brickId="attn_0"
                                brickKind="attention" params={{}}
                                onApply={() => {}}
                                onSwapKind={onSwap}
                                onClose={() => {}} />);
    const sel = screen.getByTestId(
      "brick-context-attn_0-swap-target") as HTMLSelectElement;
    expect(sel).toBeDefined();
    const opts = Array.from(sel.options).map((o) => o.value);
    expect(opts).toContain("attention");
    expect(opts).toContain("gated_attention");
    // sdpa_attention category includes mla / mistral4_mla as well.
    expect(opts).toContain("mla");
  });

  it("V7-F52: Swap kind button disabled when target == current", () => {
    render(<BrickContextPanel rpc={RPC} brickId="attn_0"
                                brickKind="attention" params={{}}
                                onApply={() => {}}
                                onSwapKind={() => {}}
                                onClose={() => {}} />);
    const btn = screen.getByTestId(
      "brick-context-attn_0-swap-apply") as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });

  it("V7-F52: picking a new kind enables the Swap button + fires callback", () => {
    const onSwap = vi.fn();
    const onClose = vi.fn();
    render(<BrickContextPanel rpc={RPC} brickId="attn_0"
                                brickKind="attention" params={{}}
                                onApply={() => {}}
                                onSwapKind={onSwap}
                                onClose={onClose} />);
    fireEvent.change(screen.getByTestId("brick-context-attn_0-swap-target"),
                     { target: { value: "gated_attention" } });
    const btn = screen.getByTestId(
      "brick-context-attn_0-swap-apply") as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
    fireEvent.click(btn);
    expect(onSwap).toHaveBeenCalledWith("gated_attention");
    expect(onClose).toHaveBeenCalled();
  });

  it("V7-F52: swap dropdown hidden when onSwapKind not provided", () => {
    render(<BrickContextPanel rpc={RPC} brickId="attn_0"
                                brickKind="attention" params={{}}
                                onApply={() => {}} onClose={() => {}} />);
    expect(screen.queryByTestId("brick-context-attn_0-swap-target"))
      .toBeNull();
  });

  it("hides norm dropdowns for embedding brick (no support)", () => {
    render(<BrickContextPanel rpc={RPC} brickId="emb_0"
                                brickKind="abs_pos_embed" params={{}}
                                onApply={() => {}} onClose={() => {}} />);
    expect(screen.queryByTestId("brick-context-emb_0-pre-norm")).toBeNull();
  });

  it("Apply button fires onApply with merged params + onClose", () => {
    const onApply = vi.fn();
    const onClose = vi.fn();
    render(<BrickContextPanel rpc={RPC} brickId="mlp_0" brickKind="mlp"
                                params={{ intermediate_size: 256 }}
                                onApply={onApply} onClose={onClose} />);
    fireEvent.change(
      screen.getByTestId("brick-context-mlp_0-activation"),
      { target: { value: "swiglu" } },
    );
    fireEvent.click(screen.getByTestId("brick-context-mlp_0-apply"));
    expect(onApply).toHaveBeenCalledWith(expect.objectContaining({
      intermediate_size: 256,
      activation: "swiglu",
    }));
    expect(onClose).toHaveBeenCalled();
  });

  it("Close button fires onClose without onApply", () => {
    const onApply = vi.fn();
    const onClose = vi.fn();
    render(<BrickContextPanel rpc={RPC} brickId="mlp_0" brickKind="mlp"
                                params={{}}
                                onApply={onApply} onClose={onClose} />);
    fireEvent.click(screen.getByTestId("brick-context-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onApply).not.toHaveBeenCalled();
  });

  it("shows disabled 'No trainable weights' button for residual brick when histogram requested", () => {
    const onInspect = vi.fn();
    render(<BrickContextPanel rpc={RPC} brickId="res_0" brickKind="residual"
                                params={{}}
                                onApply={() => {}} onClose={() => {}}
                                onInspectHistogram={onInspect} />);
    const btn = screen.getByTestId("brick-context-res_0-histogram-disabled") as HTMLButtonElement;
    expect(btn).toBeTruthy();
    expect(btn.disabled).toBe(true);
    expect(btn.textContent).toBe("No trainable weights");
    expect(screen.queryByTestId("brick-context-res_0-histogram-fetch")).toBeNull();
  });

  it("shows enabled 'Inspect weight histogram' button for linear_bridge brick when histogram requested", () => {
    const onInspect = vi.fn();
    render(<BrickContextPanel rpc={RPC} brickId="lb_0" brickKind="linear_bridge"
                                params={{}}
                                onApply={() => {}} onClose={() => {}}
                                onInspectHistogram={onInspect} />);
    const btn = screen.getByTestId("brick-context-lb_0-histogram-fetch") as HTMLButtonElement;
    expect(btn).toBeTruthy();
    expect(btn.disabled).toBe(false);
    expect(btn.textContent).toBe("Inspect weight histogram");
    expect(screen.queryByTestId("brick-context-lb_0-histogram-disabled")).toBeNull();
  });
});
