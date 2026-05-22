import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { LossTab } from "@/components/sidebar/LossTab";
import { OptimTab } from "@/components/sidebar/OptimTab";
import { RewritersTab } from "@/components/sidebar/RewritersTab";
import { ShardingTab } from "@/components/sidebar/ShardingTab";
import { GotchasTab } from "@/components/sidebar/GotchasTab";
import { SideChannelsTab } from "@/components/sidebar/SideChannelsTab";
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

  it("kind dropdown exposes all 7 OptimKind options (E7-12)", () => {
    render(<OptimTab optim={INITIAL_SPEC.optim} onApply={() => {}} />);
    const select = screen.getByTestId("optim-kind") as HTMLSelectElement;
    const options = Array.from(select.options).map((o) => o.value);
    expect(options).toEqual([
      "adamw", "muon", "muon_adamw_hybrid",
      "lion", "lion8bit", "adam8bit", "sgd",
    ]);
  });

  it("can select Lion from the kind dropdown (E7-12)", () => {
    const onApply = vi.fn();
    render(<OptimTab optim={INITIAL_SPEC.optim} onApply={onApply} />);
    fireEvent.change(screen.getByTestId("optim-kind"),
                     { target: { value: "lion" } });
    fireEvent.click(screen.getByTestId("optim-apply"));
    const arg = onApply.mock.calls[0][0];
    expect(arg.kind).toBe("lion");
  });

  it("can select Lion8bit and Adam8bit from the dropdown (E7-12)", () => {
    const onApply = vi.fn();
    render(<OptimTab optim={INITIAL_SPEC.optim} onApply={onApply} />);
    for (const kind of ["lion8bit", "adam8bit"]) {
      fireEvent.change(screen.getByTestId("optim-kind"),
                       { target: { value: kind } });
      fireEvent.click(screen.getByTestId("optim-apply"));
    }
    const kinds = onApply.mock.calls.map((c) => c[0].kind);
    expect(kinds).toContain("lion8bit");
    expect(kinds).toContain("adam8bit");
  });
});

describe("OptimTab — RECOMMENDED_LR (E7-12)", () => {
  it("exposes recommended lr per kind including Lion=1e-4", async () => {
    const mod = await import("@/components/sidebar/OptimTab");
    expect(mod.RECOMMENDED_LR.lion).toBe(1e-4);
    expect(mod.RECOMMENDED_LR.lion8bit).toBe(1e-4);
    expect(mod.RECOMMENDED_LR.adam8bit).toBe(3e-4);
    expect(mod.RECOMMENDED_LR.adamw).toBe(3e-4);
    expect(mod.RECOMMENDED_LR.muon).toBe(1e-2);
    expect(mod.RECOMMENDED_LR.sgd).toBe(1e-2);
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

describe("SideChannelsTab", () => {
  it("updates family policy and applies the draft", () => {
    const onApply = vi.fn();
    render(<SideChannelsTab sideChannels={INITIAL_SPEC.side_channels}
                            availableChannels={["doc_ids", "token_ids"]}
                            selectedTrainChannels={[]}
                            gotchas={[]}
                            onApply={onApply}
                            onTrainChannelsChange={() => {}} />);
    fireEvent.change(screen.getByTestId("side-channel-family-platform-mode"),
      { target: { value: "require" } });
    fireEvent.change(screen.getByTestId("side-channel-family-platform-dropout"),
      { target: { value: "0.2" } });
    fireEvent.change(screen.getByTestId("side-channel-family-platform-fallback"),
      { target: { value: "error" } });
    fireEvent.click(screen.getByTestId("side-channels-apply"));
    const next = onApply.mock.calls[0][0];
    expect(next.families.platform.mode).toBe("require");
    expect(next.families.platform.dropout).toBe(0.2);
    expect(next.families.platform.fallback).toBe("error");
  });

  it("renders inference and platform preview controls", () => {
    render(<SideChannelsTab sideChannels={INITIAL_SPEC.side_channels}
                            availableChannels={["platform_ids"]}
                            selectedTrainChannels={[]}
                            gotchas={[]}
                            onApply={() => {}}
                            onTrainChannelsChange={() => {}} />);
    fireEvent.change(screen.getByTestId("side-channel-inference-source"),
      { target: { value: "parse_if_possible" } });
    fireEvent.change(screen.getByTestId("side-channel-platform-os"),
      { target: { value: "linux" } });
    expect(screen.getByTestId("side-channel-platform-preview").textContent)
      .toContain("os=linux");
    expect(screen.getByTestId("side-channel-preview").textContent)
      .toContain("source=parse_if_possible");
  });

  it("surfaces required-family contract probe errors", () => {
    render(<SideChannelsTab sideChannels={INITIAL_SPEC.side_channels}
                            availableChannels={[]}
                            selectedTrainChannels={[]}
                            gotchas={[{
                              id: "side_channel_required_platform",
                              severity: "error",
                              message: "required side-channel family 'platform'",
                            }]}
                            onApply={() => {}}
                            onTrainChannelsChange={() => {}} />);
    expect(screen.getByTestId(
      "side-channel-probe-error-side_channel_required_platform",
    ).textContent).toContain("platform");
  });

  it("selects train side-channel inputs inside the side-channel tab", () => {
    const onTrainChannelsChange = vi.fn();
    render(<SideChannelsTab sideChannels={INITIAL_SPEC.side_channels}
                            availableChannels={["doc_ids", "token_ids"]}
                            selectedTrainChannels={["doc_ids"]}
                            gotchas={[]}
                            onApply={() => {}}
                            onTrainChannelsChange={onTrainChannelsChange} />);
    expect(screen.getByTestId("side-channel-train-doc_ids"))
      .toHaveProperty("checked", true);
    fireEvent.click(screen.getByTestId("side-channel-train-token_ids"));
    expect(onTrainChannelsChange).toHaveBeenCalledWith([
      "doc_ids", "token_ids",
    ]);
  });
});

describe("Sidebar", () => {
  const stubs = {
    loss: INITIAL_SPEC.loss, optim: INITIAL_SPEC.optim,
    rewriters: INITIAL_SPEC.rewriters,
    sideChannels: INITIAL_SPEC.side_channels,
    availableSideChannels: ["doc_ids", "token_ids"],
    selectedTrainSideChannels: [],
    sharding: INITIAL_SPEC.sharding,
    gotchas: INITIAL_SPEC.gotchas, proposals: [],
    onLossApply: () => {}, onOptimApply: () => {},
    onRewriterAdd: () => {}, onRewriterRemove: () => {},
    onRewriterReorder: () => {},
    onSideChannelsApply: () => {},
    onTrainSideChannelsChange: () => {},
    onShardingChange: () => {}, onShardingAccept: () => {},
  };

  it("renders side-channel tab button with existing tabs", () => {
    render(<Sidebar {...stubs} />);
    for (const k of [
      "loss", "optim", "rewriters", "side_channels", "sharding", "gotchas",
    ]) {
      expect(screen.getByTestId(`sidebar-tab-${k}`)).toBeTruthy();
    }
  });

  it("switches active tab on click", () => {
    render(<Sidebar {...stubs} />);
    fireEvent.click(screen.getByTestId("sidebar-tab-optim"));
    expect(screen.getByTestId("optim-tab")).toBeTruthy();
    fireEvent.click(screen.getByTestId("sidebar-tab-side_channels"));
    expect(screen.getByTestId("side-channels-tab")).toBeTruthy();
    fireEvent.click(screen.getByTestId("sidebar-tab-gotchas"));
    expect(screen.getByTestId("gotchas-tab")).toBeTruthy();
  });
});
