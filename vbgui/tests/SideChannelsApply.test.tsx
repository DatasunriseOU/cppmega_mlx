// V7-H11: SideChannelsTab Apply button calls side_channels.apply RPC
// and renders backend's per-family + gotcha verdict inline.

import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { SideChannelsTab } from "@/components/sidebar/SideChannelsTab";
import type { RpcClient } from "@/lib/rpc";
import type { SideChannelState } from "@/state/spec";

const baseState: SideChannelState = {
  mode: "if_available",
  families: {
    platform: { mode: "require", columns: ["platform_ids"],
                 embedding: "categorical", dropout: 0,
                 residual_scale: 1, fallback: "drop_family",
                 language_scope: ["any"] },
  },
  inference: {
    source: "auto", fail_policy: "drop_family",
    timeout_ms: 500, cache_enabled: true,
  },
} as unknown as SideChannelState;

function makeRpc(reply: object): RpcClient {
  return {
    call: vi.fn(async () => reply as never),
  } as unknown as RpcClient;
}

describe("V7-H11 SideChannelsTab Apply RPC", () => {
  it("calls side_channels.apply on click and renders families + summary",
     async () => {
    const rpc = makeRpc({
      ok: true, active_count: 1, inactive_count: 0,
      families: [{
        family: "platform", mode: "require", active: true,
        reason: "all requested columns present",
        columns_requested: ["platform_ids"],
        columns_present: ["platform_ids"], columns_missing: [],
      }],
      gotchas: [], elapsed_ms: 2.1,
    });

    render(
      <SideChannelsTab
        sideChannels={baseState}
        availableChannels={["platform_ids", "doc_ids"]}
        selectedTrainChannels={[]}
        gotchas={[]}
        rpc={rpc}
        tokenizerSource="dummy"
        onApply={vi.fn()}
        onTrainChannelsChange={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByTestId("side-channels-apply"));

    await waitFor(() => {
      expect(rpc.call).toHaveBeenCalledWith(
        "side_channels.apply",
        expect.objectContaining({
          side_channels: expect.any(Object),
          available_side_channels: ["platform_ids", "doc_ids"],
        }),
      );
    });

    expect((await screen.findByTestId("side-channels-apply-summary"))
             .textContent).toContain("applied: 1 active");
    expect((screen.getByTestId("side-channels-apply-summary"))
             .textContent).toContain("ok");
    expect((screen.getByTestId("side-channels-apply-family-platform"))
             .textContent).toContain("active");
  });

  it("renders error severity gotcha from backend with red colour", async () => {
    const rpc = makeRpc({
      ok: false, active_count: 0, inactive_count: 1,
      families: [{
        family: "platform", mode: "require", active: false,
        reason: "required columns missing: platform_ids",
        columns_requested: ["platform_ids"],
        columns_present: [], columns_missing: ["platform_ids"],
      }],
      gotchas: [{ id: "side_channel_required_platform",
                  severity: "error",
                  message: "required side-channel family 'platform' is missing platform_ids" }],
      elapsed_ms: 1.3,
    });

    render(
      <SideChannelsTab
        sideChannels={baseState}
        availableChannels={["doc_ids"]}
        selectedTrainChannels={[]}
        gotchas={[]}
        rpc={rpc}
        tokenizerSource="dummy"
        onApply={vi.fn()}
        onTrainChannelsChange={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByTestId("side-channels-apply"));

    expect((
      await screen.findByTestId(
        "side-channels-apply-gotcha-side_channel_required_platform")
    ).textContent).toContain("required side-channel family");
    expect((screen.getByTestId("side-channels-apply-summary"))
             .textContent).toContain("errors");
  });

  it("falls back to local-only apply when no rpc client", async () => {
    const onApply = vi.fn();
    render(
      <SideChannelsTab
        sideChannels={baseState}
        availableChannels={["platform_ids"]}
        selectedTrainChannels={[]}
        gotchas={[]}
        rpc={null}
        tokenizerSource={null}
        onApply={onApply}
        onTrainChannelsChange={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByTestId("side-channels-apply"));

    await waitFor(() => {
      expect(onApply).toHaveBeenCalledTimes(1);
    });
    expect((await screen.findByTestId("side-channels-apply-error"))
             .textContent).toContain("no backend connection");
  });
});
