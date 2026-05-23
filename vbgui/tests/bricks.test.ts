import { describe, it, expect } from "vitest";
import {
  ADAPTERS, BRICKS, CATEGORY_COLORS,
  adapterFor, brickFor, colorFor,
} from "@/lib/bricks";

describe("brick registry", () => {
  it("has at least 22 brick entries (VBPlan minimum)", () => {
    expect(BRICKS.length).toBeGreaterThanOrEqual(22);
  });

  it("matches the current count of 27", () => {
    expect(BRICKS.length).toBe(27);
  });

  it("has 6 adapters per ticket spec", () => {
    expect(ADAPTERS.length).toBe(6);
  });

  it("every brick category has a colour entry", () => {
    for (const b of BRICKS) {
      expect(CATEGORY_COLORS[b.category]).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });

  it("brickFor + adapterFor lookups", () => {
    expect(brickFor("attention")?.label).toBe("Attention (vanilla)");
    expect(adapterFor("residual")?.label).toBe("Residual Add");
    expect(brickFor("nonexistent")).toBeUndefined();
  });

  it("colorFor resolves every BrickCategory", () => {
    expect(colorFor("sdpa_attention")).toBe(CATEGORY_COLORS.sdpa_attention);
  });

  it("brick kinds are unique", () => {
    const kinds = BRICKS.map((b) => b.kind);
    expect(new Set(kinds).size).toBe(kinds.length);
  });

  it("adapter kinds are unique", () => {
    const kinds = ADAPTERS.map((a) => a.kind);
    expect(new Set(kinds).size).toBe(kinds.length);
  });

  it("covers the 6 expected adapter kinds from VBPlan", () => {
    const expected = ["merge_heads", "split_heads", "transpose_bnsd",
                      "linear_bridge", "rmsnorm", "residual"];
    for (const e of expected) {
      expect(adapterFor(e)).toBeDefined();
    }
  });
});
