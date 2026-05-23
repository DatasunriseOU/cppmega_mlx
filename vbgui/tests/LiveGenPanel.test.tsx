// V7-H42: LiveGenPanel renders rolling token tail + finish toast.

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { LiveGenPanel } from "@/components/LiveGenPanel";
import type { GenTokenEvent } from "@/hooks/useGenerateStream";

const EVS: GenTokenEvent[] = [
  { step: 0, token_id: 100 },
  { step: 1, token_id: 200 },
  { step: 2, token_id: 300 },
];

describe("V7-H42 LiveGenPanel", () => {
  it("returns empty when nothing happened yet", () => {
    const { container } = render(
      <LiveGenPanel events={[]} genInFlight={false} />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders waiting-for-first-token placeholder when active + empty",
     () => {
    render(<LiveGenPanel events={[]} genInFlight={true} />);
    expect(screen.getByTestId("live-gen-panel-empty")).toBeTruthy();
  });

  it("renders token count + last id + rolling tail", () => {
    render(<LiveGenPanel events={EVS} genInFlight={true} />);
    expect(screen.getByTestId("live-gen-panel-token-count").textContent)
      .toContain("3");
    expect(screen.getByTestId("live-gen-panel-last-token").textContent)
      .toContain("300");
    expect(screen.getByTestId("live-gen-panel-tail").textContent)
      .toBe("100 200 300");
  });

  it("respects tail window — only last N tokens rendered", () => {
    const many: GenTokenEvent[] = Array.from({ length: 80 },
      (_, i) => ({ step: i, token_id: i }));
    render(<LiveGenPanel events={many} genInFlight={true} tail={5} />);
    // Last 5 tokens: 75..79.
    expect(screen.getByTestId("live-gen-panel-tail").textContent)
      .toBe("75 76 77 78 79");
  });

  it("shows finish toast and dismiss button", () => {
    const onDismiss = vi.fn();
    render(<LiveGenPanel events={EVS} genInFlight={false}
                          finishToast={true}
                          onDismissToast={onDismiss} />);
    expect(screen.getByTestId("live-gen-panel-toast").textContent)
      .toContain("gen done");
    expect(screen.getByTestId("live-gen-panel-toast-dismiss"))
      .toBeTruthy();
  });

  it("shows reconnect counter when reconnectAttempts > 0", () => {
    render(<LiveGenPanel events={EVS} genInFlight={true}
                          reconnectAttempts={3} />);
    expect(screen.getByTestId("live-gen-panel-reconnects").textContent)
      .toContain("3");
  });
});
