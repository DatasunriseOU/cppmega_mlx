import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { LossTab } from "@/components/sidebar/LossTab";
import { OptimTab } from "@/components/sidebar/OptimTab";
import { RewritersTab } from "@/components/sidebar/RewritersTab";
import { ShardingTab } from "@/components/sidebar/ShardingTab";
import { GotchasTab } from "@/components/sidebar/GotchasTab";
import { Sidebar } from "@/components/Sidebar";
import { INITIAL_SPEC } from "@/state/spec";

describe("LossTab", () => {
  it("renders all 5 loss kinds", () => {
    render(<LossTab loss={INITIAL_SPEC.loss} onApply={() => {}} />);
    const select = screen.getByTestId("loss-kind") as HTMLSelectElement;
    expect(select.options).toHaveLength(5);
  });

  it("MTP shows k + beta inputs", () => {
    render(<LossTab loss={INITIAL_SPEC.loss} onApply={() => {}} />);
    fireEvent.change(screen.getByTestId("loss-kind"),
      { target: { value: "mtp_weighted" } });
    expect(screen.getByTestId("loss-mtp-k")).toBeTruthy();
    expect(screen.getByTestId("loss-mtp-beta")).toBeTruthy();
  });

  it("IFIM shows lambda_fim input", () => {
    render(<LossTab loss={INITIAL_SPEC.loss} onApply={() => {}} />);
    fireEvent.change(screen.getByTestId("loss-kind"),
      { target: { value: "ifim_shaped" } });
    expect(screen.getByTestId("loss-ifim-lambda")).toBeTruthy();
  });

  it("Apply fires onApply with the draft", () => {
    const onApply = vi.fn();
    render(<LossTab loss={INITIAL_SPEC.loss} onApply={onApply} />);
    fireEvent.change(screen.getByTestId("loss-kind"),
      { target: { value: "mhc_attn_bias" } });
    fireEvent.click(screen.getByTestId("loss-apply"));
    expect(onApply).toHaveBeenCalledWith(expect.objectContaining({
      kind: "mhc_attn_bias",
    }));
  });
});

describe("OptimTab", () => {
  it("renders the seeded group", () => {
    render(<OptimTab optim={INITIAL_SPEC.optim} onApply={() => {}} />);
    expect(screen.getByTestId("optim-group-0")).toBeTruthy();
  });

  it("Add group appends a row", () => {
    render(<OptimTab optim={INITIAL_SPEC.optim} onApply={() => {}} />);
    fireEvent.click(screen.getByTestId("optim-add-group"));
    expect(screen.getByTestId("optim-group-1")).toBeTruthy();
  });

  it("Remove group drops the row", () => {
    render(<OptimTab optim={INITIAL_SPEC.optim} onApply={() => {}} />);
    fireEvent.click(screen.getByTestId("optim-add-group"));
    fireEvent.click(screen.getByTestId("optim-group-1-remove"));
    expect(screen.queryByTestId("optim-group-1")).toBeNull();
  });

  it("Apply fires onApply with the draft", () => {
    const onApply = vi.fn();
    render(<OptimTab optim={INITIAL_SPEC.optim} onApply={onApply} />);
    fireEvent.click(screen.getByTestId("optim-apply"));
    expect(onApply).toHaveBeenCalled();
  });
});

describe("RewritersTab", () => {
  it("renders add buttons for MTP/IFIM/MHC", () => {
    render(<RewritersTab rewriters={[]}
                         onAdd={() => {}} onRemove={() => {}}
                         onReorder={() => {}} />);
    expect(screen.getByTestId("rewriter-add-MTPRewriter")).toBeTruthy();
    expect(screen.getByTestId("rewriter-add-IFIMRewriter")).toBeTruthy();
    expect(screen.getByTestId("rewriter-add-MHCRewriter")).toBeTruthy();
  });

  it("Add calls onAdd with default params", () => {
    const onAdd = vi.fn();
    render(<RewritersTab rewriters={[]}
                         onAdd={onAdd} onRemove={() => {}}
                         onReorder={() => {}} />);
    fireEvent.click(screen.getByTestId("rewriter-add-MTPRewriter"));
    expect(onAdd).toHaveBeenCalledWith({
      name: "MTPRewriter", params: { k: 2, beta: 0.6 },
    });
  });

  it("reorder buttons are disabled at ends", () => {
    const seeded = [{ name: "MTPRewriter" as const, params: {} },
                    { name: "IFIMRewriter" as const, params: {} }];
    render(<RewritersTab rewriters={seeded}
                         onAdd={() => {}} onRemove={() => {}}
                         onReorder={() => {}} />);
    expect((screen.getByTestId("rewriter-up-0") as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId("rewriter-down-1") as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("ShardingTab", () => {
  it("renders proposals + Accept buttons", () => {
    const onAccept = vi.fn();
    render(<ShardingTab sharding={INITIAL_SPEC.sharding}
                        proposals={[{
                          strategy_name: "fsdp2_only", fits: true,
                          estimated_per_rank_bytes: 1_000_000_000,
                          reason: "fits",
                        }]}
                        onAccept={onAccept}
                        onChange={() => {}} />);
    fireEvent.click(screen.getByTestId("sharding-accept-0"));
    expect(onAccept).toHaveBeenCalledWith(0);
  });

  it("Add axis appends row via onChange", () => {
    const onChange = vi.fn();
    render(<ShardingTab sharding={INITIAL_SPEC.sharding}
                        proposals={[]} onAccept={() => {}}
                        onChange={onChange} />);
    fireEvent.click(screen.getByTestId("sharding-add-axis"));
    expect(onChange).toHaveBeenCalled();
    const last = onChange.mock.calls.at(-1)?.[0];
    expect(last.axis_assignments).toHaveLength(2);
  });

  it("toggles flip fp8_enabled / activation_checkpointing", () => {
    const onChange = vi.fn();
    render(<ShardingTab sharding={INITIAL_SPEC.sharding}
                        proposals={[]} onAccept={() => {}}
                        onChange={onChange} />);
    fireEvent.click(screen.getByTestId("sharding-toggle-fp8_enabled"));
    expect(onChange.mock.calls.at(-1)?.[0].fp8_enabled).toBe(true);
  });
});

describe("GotchasTab", () => {
  it("renders gotchas grouped by severity", () => {
    render(<GotchasTab gotchas={[
      { id: "e1", severity: "error",   message: "boom" },
      { id: "w1", severity: "warning", message: "careful" },
      { id: "i1", severity: "info",    message: "fyi" },
    ]} />);
    expect(screen.getByTestId("gotchas-error")).toBeTruthy();
    expect(screen.getByTestId("gotchas-warning")).toBeTruthy();
    expect(screen.getByTestId("gotchas-info")).toBeTruthy();
  });

  it("shows Auto-fix for known fixable gotchas", () => {
    const onAutoFix = vi.fn();
    render(<GotchasTab onAutoFix={onAutoFix} gotchas={[
      { id: "fsdp2_whole_compile", severity: "error", message: "x" },
    ]} />);
    fireEvent.click(screen.getByTestId("gotcha-fsdp2_whole_compile-autofix"));
    expect(onAutoFix).toHaveBeenCalledWith("fsdp2_whole_compile");
  });
});

describe("Sidebar", () => {
  const stubs = {
    loss: INITIAL_SPEC.loss, optim: INITIAL_SPEC.optim,
    rewriters: INITIAL_SPEC.rewriters, sharding: INITIAL_SPEC.sharding,
    gotchas: INITIAL_SPEC.gotchas, proposals: [],
    onLossApply: () => {}, onOptimApply: () => {},
    onRewriterAdd: () => {}, onRewriterRemove: () => {},
    onRewriterReorder: () => {},
    onShardingChange: () => {}, onShardingAccept: () => {},
  };

  it("renders all 5 tab buttons", () => {
    render(<Sidebar {...stubs} />);
    for (const k of ["loss", "optim", "rewriters", "sharding", "gotchas"]) {
      expect(screen.getByTestId(`sidebar-tab-${k}`)).toBeTruthy();
    }
  });

  it("switches active tab on click", () => {
    render(<Sidebar {...stubs} />);
    fireEvent.click(screen.getByTestId("sidebar-tab-optim"));
    expect(screen.getByTestId("optim-tab")).toBeTruthy();
    fireEvent.click(screen.getByTestId("sidebar-tab-gotchas"));
    expect(screen.getByTestId("gotchas-tab")).toBeTruthy();
  });
});
