// V7-H33: LossSurfaceModal opens, calls loss_surface.run, renders grid,
// Apply-best invokes onApplyBest with chosen multipliers + closes.

import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { LossSurfaceModal } from "@/components/LossSurfaceModal";
import type { RpcClient } from "@/lib/rpc";

function makeRpc(reply: object): RpcClient {
  return { call: vi.fn(async () => reply as never) } as unknown as RpcClient;
}

const RESULT = {
  rows: [
    [
      { lr_mult: 0.5, wd_mult: 1.0, status: "ok", final_loss: 2.1 },
      { lr_mult: 0.5, wd_mult: 2.0, status: "ok", final_loss: 2.4 },
    ],
    [
      { lr_mult: 1.0, wd_mult: 1.0, status: "ok", final_loss: 1.8 },
      { lr_mult: 1.0, wd_mult: 2.0, status: "ok", final_loss: 2.0 },
    ],
  ],
  lr_deltas: [0.5, 1.0],
  wd_deltas: [1.0, 2.0],
  best_lr_mult: 1.0,
  best_wd_mult: 1.0,
  best_loss: 1.8,
};

describe("V7-H33 LossSurfaceModal", () => {
  it("renders nothing when closed", () => {
    const { container } = render(
      <LossSurfaceModal rpc={makeRpc(RESULT)} spec={{}} open={false}
                        onClose={vi.fn()} onApplyBest={vi.fn()} />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("calls loss_surface.run on Run sweep + renders cells + best", async () => {
    const rpc = makeRpc(RESULT);
    render(
      <LossSurfaceModal rpc={rpc} spec={{ x: 1 }} open={true}
                        onClose={vi.fn()} onApplyBest={vi.fn()} />,
    );
    fireEvent.click(screen.getByTestId("loss-surface-run"));
    await waitFor(() => {
      expect(rpc.call).toHaveBeenCalledWith("loss_surface.run",
        expect.objectContaining({ spec: { x: 1 }, k_steps: 2 }));
    });
    expect((await screen.findByTestId("loss-surface-cell-0-0"))
             .textContent).toBe("2.100");
    expect(screen.getByTestId("loss-surface-best").textContent)
      .toContain("best: lr×1");
    expect(screen.getByTestId("loss-surface-best").textContent)
      .toContain("1.8000");
  });

  it("Apply-best invokes onApplyBest with best multipliers + closes",
     async () => {
    const onApplyBest = vi.fn();
    const onClose = vi.fn();
    render(
      <LossSurfaceModal rpc={makeRpc(RESULT)} spec={{}} open={true}
                        onClose={onClose} onApplyBest={onApplyBest} />,
    );
    fireEvent.click(screen.getByTestId("loss-surface-run"));
    fireEvent.click(await screen.findByTestId("loss-surface-apply-best"));
    expect(onApplyBest).toHaveBeenCalledWith(1.0, 1.0);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("renders error when rpc client missing", async () => {
    render(
      <LossSurfaceModal rpc={null} spec={{}} open={true}
                        onClose={vi.fn()} onApplyBest={vi.fn()} />,
    );
    fireEvent.click(screen.getByTestId("loss-surface-run"));
    expect((await screen.findByTestId("loss-surface-error"))
             .textContent).toContain("no backend");
  });
});
