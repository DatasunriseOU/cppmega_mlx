import { describe, it, expect } from "vitest";
import {
  INITIAL_SPEC, memoryColor, memoryFillRatio, specReducer,
} from "@/state/spec";

describe("specReducer", () => {
  it("loss.set replaces loss", () => {
    const s = specReducer(INITIAL_SPEC, {
      type: "loss.set",
      loss: { kind: "mtp_weighted", head_outputs: ["h0", "h1"],
              params: { k: 2 } },
    });
    expect(s.loss.kind).toBe("mtp_weighted");
    expect(s.loss.head_outputs).toEqual(["h0", "h1"]);
  });

  it("optim.add_group appends group", () => {
    const s = specReducer(INITIAL_SPEC, {
      type: "optim.add_group",
      group: { matcher: "regex:.*moe.*", lr: 1e-4, weight_decay: 0.0 },
    });
    expect(s.optim.groups).toHaveLength(2);
  });

  it("optim.remove_group drops the right index", () => {
    const seeded = specReducer(INITIAL_SPEC, {
      type: "optim.add_group",
      group: { matcher: "x", lr: 1e-5, weight_decay: 0.0 },
    });
    const s = specReducer(seeded, { type: "optim.remove_group", index: 0 });
    expect(s.optim.groups).toHaveLength(1);
    expect(s.optim.groups[0].matcher).toBe("x");
  });

  it("rewriters.reorder swaps elements", () => {
    const seeded = specReducer(INITIAL_SPEC, {
      type: "rewriters.add",
      rewriter: { name: "MTPRewriter", params: { k: 2 } },
    });
    const with2 = specReducer(seeded, {
      type: "rewriters.add",
      rewriter: { name: "IFIMRewriter", params: { lambda_fim: 0.1 } },
    });
    const s = specReducer(with2, { type: "rewriters.reorder", from: 0, to: 1 });
    expect(s.rewriters[0].name).toBe("IFIMRewriter");
    expect(s.rewriters[1].name).toBe("MTPRewriter");
  });

  it("sharding.set replaces sharding", () => {
    const s = specReducer(INITIAL_SPEC, {
      type: "sharding.set",
      sharding: { ...INITIAL_SPEC.sharding, fp8_enabled: true },
    });
    expect(s.sharding.fp8_enabled).toBe(true);
  });

  it("verify.complete stores latency + brick count", () => {
    const s = specReducer(INITIAL_SPEC, {
      type: "verify.complete", elapsed_ms: 4.2, brick_count: 22,
    });
    expect(s.last_verify_ms).toBe(4.2);
    expect(s.brick_count).toBe(22);
  });

  it("backend.status updates connection state", () => {
    const s = specReducer(INITIAL_SPEC,
      { type: "backend.status", status: "connected" });
    expect(s.backend_status).toBe("connected");
  });

  it("side_channels.set replaces generic side-channel policy", () => {
    const s = specReducer(INITIAL_SPEC, {
      type: "side_channels.set",
      side_channels: {
        ...INITIAL_SPEC.side_channels,
        families: {
          ...INITIAL_SPEC.side_channels.families,
          platform: {
            ...INITIAL_SPEC.side_channels.families.platform,
            mode: "require",
            dropout: 0.2,
            fallback: "error",
          },
        },
      },
    });
    expect(s.side_channels.families.platform.mode).toBe("require");
    expect(s.side_channels.families.platform.dropout).toBe(0.2);
    expect(s.side_channels.families.platform.fallback).toBe("error");
  });
});

describe("side-channel defaults", () => {
  it("INITIAL_SPEC exposes language-neutral side-channel families", () => {
    expect(INITIAL_SPEC.side_channels.mode).toBe("auto");
    expect(Object.keys(INITIAL_SPEC.side_channels.families)).toEqual([
      "platform",
      "syntax",
      "structure",
      "semantic_graph",
      "temporal_diff",
    ]);
    expect(INITIAL_SPEC.side_channels.families.platform.columns).toEqual([
      "platform_ids",
      "source_platform_ids",
    ]);
    expect(INITIAL_SPEC.side_channels.inference.fail_policy).toBe("drop_family");
  });

  it("INITIAL_SPEC exposes packed-row materialization defaults", () => {
    expect(INITIAL_SPEC.data_materialization).toEqual({
      packing_policy: "best_fit",
      max_seq_len: 4096,
      pad_to_max: true,
      include_provenance: true,
      required_token_fields: [
        "input_ids",
        "target_ids",
        "loss_mask",
        "doc_ids",
        "pack_id",
        "valid_token_count",
        "num_docs",
      ],
    });
  });
});

describe("memory helpers", () => {
  it("memoryFillRatio = worst / hbm", () => {
    const s = { ...INITIAL_SPEC,
                worst_rank_bytes: 40 * 1024 ** 3,
                device_hbm_bytes: 80 * 1024 ** 3 };
    expect(memoryFillRatio(s)).toBeCloseTo(0.5);
  });

  it("memoryColor thresholds match VBPlan §4.2", () => {
    const make = (r: number) => ({
      ...INITIAL_SPEC,
      worst_rank_bytes: r * 80 * 1024 ** 3,
      device_hbm_bytes: 80 * 1024 ** 3,
    });
    expect(memoryColor(make(0.5))).toBe("green");
    expect(memoryColor(make(0.8))).toBe("yellow");
    expect(memoryColor(make(0.95))).toBe("red");
  });

  it("memoryFillRatio handles zero HBM", () => {
    expect(memoryFillRatio({ ...INITIAL_SPEC, device_hbm_bytes: 0 })).toBe(0);
  });
});
