import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { TrainLiveControls } from "@/components/TrainLiveControls";

function fakeRpc(handler: (m: string, p: unknown) => Promise<unknown>) {
  return { call: async <T,>(m: string, p: unknown) =>
                  (await handler(m, p)) as T } as never;
}

describe("V7-K9/K10 TrainLiveControls", () => {
  it("status idle when trainInFlight=false", () => {
    render(<TrainLiveControls rpc={null} trainInFlight={false}
                                activeRunId={null}
                                onScheduleCheckpoint={() => {}} />);
    expect(screen.getByTestId("train-live-status").textContent)
      .toContain("idle");
  });

  it("status flips to 'in flight' when trainInFlight=true", () => {
    render(<TrainLiveControls rpc={null} trainInFlight={true}
                                activeRunId="r-1"
                                onScheduleCheckpoint={() => {}} />);
    expect(screen.getByTestId("train-live-status").textContent)
      .toContain("in flight");
  });

  it("Trigger checkpoint forwards the path (K9)", () => {
    const onS = vi.fn();
    render(<TrainLiveControls rpc={null} trainInFlight={false}
                                activeRunId={null}
                                onScheduleCheckpoint={onS} />);
    fireEvent.change(screen.getByTestId("train-live-ckpt-path"),
                     { target: { value: "/tmp/my.safetensors" } });
    fireEvent.click(screen.getByTestId("train-live-trigger-ckpt"));
    expect(onS).toHaveBeenCalledWith("/tmp/my.safetensors");
  });

  it("Apply lr disabled without activeRunId (K10)", () => {
    render(<TrainLiveControls rpc={null} trainInFlight={false}
                                activeRunId={null}
                                onScheduleCheckpoint={() => {}} />);
    fireEvent.change(screen.getByTestId("train-live-new-lr"),
                     { target: { value: "0.0001" } });
    const btn = screen.getByTestId(
      "train-live-apply-lr") as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });

  it("Apply lr calls pipeline.update_lr with run_id + new_lr (K10)", async () => {
    const rpc = fakeRpc(async () => ({ status: "ok" }));
    render(<TrainLiveControls rpc={rpc}
                                trainInFlight={true}
                                activeRunId="r-42"
                                onScheduleCheckpoint={() => {}} />);
    fireEvent.change(screen.getByTestId("train-live-new-lr"),
                     { target: { value: "0.0001" } });
    fireEvent.click(screen.getByTestId("train-live-apply-lr"));
    await waitFor(() => {
      expect(screen.getByTestId("train-live-lr-status").textContent)
        .toContain("0.0001");
    });
  });

  it("invalid lr surfaces status without RPC call", async () => {
    const callSpy = vi.fn(async () => ({ status: "ok" }));
    const rpc = fakeRpc(callSpy);
    render(<TrainLiveControls rpc={rpc} trainInFlight={true}
                                activeRunId="r-1"
                                onScheduleCheckpoint={() => {}} />);
    fireEvent.change(screen.getByTestId("train-live-new-lr"),
                     { target: { value: "-0.5" } });
    fireEvent.click(screen.getByTestId("train-live-apply-lr"));
    await waitFor(() => {
      expect(screen.getByTestId("train-live-lr-status").textContent)
        .toContain("invalid");
    });
    expect(callSpy).not.toHaveBeenCalled();
  });
});
