/**
 * brickDims utility — parameter count + shape annotations per kind.
 * Used by BrickContextPanel's "Dimensions & params" block.
 */

import { describe, it, expect } from "vitest";
import { computeBrickDims, fmtParamCount } from "@/lib/brickDims";


describe("computeBrickDims", () => {
  it("attention with H=4096, nh=32 hits ~67M params", () => {
    const d = computeBrickDims("attention", 4096, {
      num_attention_heads: 32, num_key_value_heads: 32, head_dim: 128,
    });
    expect(d.input).toBe("(B, S, H)");
    expect(d.output).toBe("(B, S, H)");
    expect(d.n_params).toBeGreaterThan(60_000_000);
    expect(d.n_params).toBeLessThan(80_000_000);
  });

  it("moe scales linearly with num_experts", () => {
    const d4 = computeBrickDims("moe", 4096,
      { num_experts: 4, top_k: 2, intermediate_size: 16384 });
    const d8 = computeBrickDims("moe", 4096,
      { num_experts: 8, top_k: 2, intermediate_size: 16384 });
    expect(d8.n_params).toBeCloseTo(2 * d4.n_params, -6);
    expect(d8.formula).toContain("8 experts");
  });

  it("rmsnorm: params = H", () => {
    const d = computeBrickDims("rmsnorm", 1024, {});
    expect(d.n_params).toBe(1024);
    expect(d.formula).toMatch(/γ/);
  });

  it("layernorm: params = 2H", () => {
    const d = computeBrickDims("layernorm", 1024, {});
    expect(d.n_params).toBe(2048);
  });

  it("mlp(swiglu): params = 3·H·d_ff", () => {
    const d = computeBrickDims("mlp", 1024, { intermediate_size: 4096 });
    expect(d.n_params).toBe(3 * 1024 * 4096);
  });

  it("abs_pos_embed: params = max_pos·H", () => {
    const d = computeBrickDims("abs_pos_embed", 768,
      { max_position_embeddings: 1024 });
    expect(d.n_params).toBe(1024 * 768);
  });

  it("embedding_table uses vocab × H", () => {
    const d = computeBrickDims("embedding_table", 4096, {}, 65536);
    expect(d.n_params).toBe(65536 * 4096);
    expect(d.input).toContain("token ids");
  });

  it("fmtParamCount formats K / M / B suffixes", () => {
    expect(fmtParamCount(500)).toBe("500");
    expect(fmtParamCount(1500)).toBe("1.5 K");
    expect(fmtParamCount(1_500_000)).toBe("1.50 M");
    expect(fmtParamCount(7_000_000_000)).toBe("7.00 B");
  });

  it("unknown kind returns 0 params with hint formula", () => {
    const d = computeBrickDims("zzz_unknown", 256, {});
    expect(d.n_params).toBe(0);
    expect(d.formula).toContain("no formula");
  });
});
