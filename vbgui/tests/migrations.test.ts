import { describe, it, expect } from "vitest";
import {
  CURRENT_SCHEMA_VERSION, FutureSchemaError, migrate,
} from "@/state/migrations";

describe("V7-H04 spec migrations", () => {
  it("stamps schema_version on a v0 (pre-versioning) spec", () => {
    const v0 = { projectName: "demo", spec: {}, nodes: [], edges: [] };
    const out = migrate(v0);
    expect(out.schema_version).toBe(1);
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
    const out = migrate(v0) as typeof v0 & { schema_version: number };
    expect(out.schema_version).toBe(1);
    expect(out.spec).toEqual(v0.spec);
    expect(out.nodes).toEqual(v0.nodes);
  });
});
