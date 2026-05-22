import { describe, it, expect } from "vitest";
import {
  CURRENT_SCHEMA_VERSION, FutureSchemaError, migrate,
} from "@/state/migrations";

describe("V7-H04 spec migrations", () => {
  it("stamps schema_version on a v0 (pre-versioning) spec", () => {
    const v0 = { projectName: "demo", spec: {}, nodes: [], edges: [] };
    const out = migrate(v0);
    expect(out.schema_version).toBe(CURRENT_SCHEMA_VERSION);
    expect(out.projectName).toBe("demo");
  });

  it("returns spec unchanged when already at current version", () => {
    const cur = { schema_version: CURRENT_SCHEMA_VERSION,
                  projectName: "demo" };
    const out = migrate(cur);
    expect(out.schema_version).toBe(CURRENT_SCHEMA_VERSION);
    expect(out.projectName).toBe("demo");
  });

  it("throws FutureSchemaError on a spec from a newer build", () => {
    expect(() => migrate({ schema_version: 9_999 }))
      .toThrowError(FutureSchemaError);
  });

  it("preserves nested fields through migration", () => {
    const v0 = {
      projectName: "P",
      spec: { loss: { kind: "cross_entropy" }, optim: { kind: "adamw" } },
      nodes: [{ id: "a" }],
    };
    const out = migrate(v0) as typeof v0 & {
      schema_version: number;
      spec: Record<string, unknown>;
    };
    expect(out.schema_version).toBe(CURRENT_SCHEMA_VERSION);
    expect(out.spec.loss).toEqual(v0.spec.loss);
    expect(out.spec.optim).toEqual(v0.spec.optim);
    expect(out.spec.data_materialization).toBeDefined();
    expect(out.nodes).toEqual(v0.nodes);
  });

  it("adds data materialization defaults when loading v1 specs", () => {
    const v1 = {
      schema_version: 1,
      spec: { loss: { kind: "cross_entropy" } },
    };
    const out = migrate(v1) as typeof v1 & {
      spec: { data_materialization?: unknown };
    };
    expect(out.schema_version).toBe(CURRENT_SCHEMA_VERSION);
    expect(out.spec.data_materialization).toEqual({
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
