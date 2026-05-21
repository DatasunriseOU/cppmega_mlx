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
});
