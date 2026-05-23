// UX-redesign #6: compact MemoryBar pill + per-rank cluster mode.

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryBar } from "@/components/MemoryBar";
import { INITIAL_SPEC } from "@/state/spec";

const STATE = {
  ...INITIAL_SPEC,
  worst_rank_bytes: 2 * 1024 ** 3, // 2 GB
  device_hbm_bytes: 80 * 1024 ** 3,
};

describe("UX-redesign #6 MemoryBar", () => {
  it("legacy mode (no compact prop): flex:1 horizontal bar unchanged",
     () => {
    render(<MemoryBar state={STATE} />);
    const bar = screen.getByTestId("memory-bar");
    expect(bar.getAttribute("data-mode")).toBeNull();
    // estimate testid still exists at the existing path so legacy
    // consumers (Sidebar tabs, other panels) keep working.
    expect(screen.getByTestId("memory-bar-estimate")
            .getAttribute("data-bytes")).toBe(String(STATE.worst_rank_bytes));
  });

  it("compact single-device: fixed 100×18 pill with no flex:1", () => {
    render(<MemoryBar state={STATE} compact={true} />);
    const bar = screen.getByTestId("memory-bar");
    expect(bar.getAttribute("data-mode")).toBe("compact");
    expect(bar.style.width).toBe("100px");
    expect(bar.style.height).toBe("18px");
  });

  it("compact + perRankBytes (cluster): N mini-bars + cluster label",
     () => {
    const ranks = [
      2 * 1024 ** 3, 3 * 1024 ** 3, 2 * 1024 ** 3, 4 * 1024 ** 3,
      2 * 1024 ** 3, 3 * 1024 ** 3, 2 * 1024 ** 3, 4 * 1024 ** 3,
    ];
    render(<MemoryBar state={STATE} compact={true}
                       perRankBytes={ranks} />);
    const bar = screen.getByTestId("memory-bar");
    expect(bar.getAttribute("data-mode")).toBe("compact-cluster");
    for (let i = 0; i < 8; i++) {
      const rank = screen.getByTestId(`memory-bar-rank-${i}`);
      expect(rank.getAttribute("data-bytes")).toBe(String(ranks[i]));
    }
    // Label shows 8× max <peak>.
    const label = screen.getByTestId("memory-bar-cluster-label");
    expect(label.textContent).toContain("8×");
    expect(label.textContent).toContain("4.00 GB");
  });

  it("perRankBytes with length 1: falls back to single-device compact",
     () => {
    render(<MemoryBar state={STATE} compact={true}
                       perRankBytes={[2 * 1024 ** 3]} />);
    expect(screen.getByTestId("memory-bar").getAttribute("data-mode"))
      .toBe("compact");
    expect(screen.queryByTestId("memory-bar-rank-0")).toBeNull();
  });

  it("tooltip enriched with per-rank breakdown in cluster mode", () => {
    const ranks = [1 * 1024 ** 3, 2 * 1024 ** 3];
    render(<MemoryBar state={STATE} compact={true}
                       perRankBytes={ranks} />);
    const bar = screen.getByTestId("memory-bar");
    const tip = bar.getAttribute("title") ?? "";
    expect(tip).toContain("per-rank:");
    expect(tip).toContain("r0=1.00 GB");
    expect(tip).toContain("r1=2.00 GB");
  });
});
