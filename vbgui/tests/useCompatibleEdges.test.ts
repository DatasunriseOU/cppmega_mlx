// E-AUDIT-02 vitest: isValidConnection predicate semantics.

import { describe, expect, it } from "vitest";
import { makeIsValidConnection } from "../src/hooks/useCompatibleEdges";


describe("makeIsValidConnection", () => {
  it("allows everything when pair set is empty (server fallback)", () => {
    const isValid = makeIsValidConnection(
      new Set(),
      () => "mlp",
    );
    expect(isValid({ source: "n1", target: "n2" })).toBe(true);
  });

  it("rejects when pair is not in the compatible set", () => {
    const isValid = makeIsValidConnection(
      new Set(["attention→mlp"]),
      (id) => (id === "src" ? "attention" : "tokenizer"),
    );
    expect(isValid({ source: "src", target: "dst" })).toBe(false);
  });

  it("accepts when pair is in the compatible set", () => {
    const isValid = makeIsValidConnection(
      new Set(["attention→mlp"]),
      (id) => (id === "src" ? "attention" : "mlp"),
    );
    expect(isValid({ source: "src", target: "dst" })).toBe(true);
  });

  it("rejects when source or target is unknown", () => {
    const isValid = makeIsValidConnection(
      new Set(["attention→mlp"]),
      () => null,
    );
    expect(isValid({ source: "x", target: "y" })).toBe(false);
    expect(isValid({ source: null, target: "y" })).toBe(false);
  });

  it("always allows connection if source is tokenizer", () => {
    const isValid = makeIsValidConnection(
      new Set(["attention→mlp"]),
      (id) => (id === "src" ? "tokenizer" : "attention"),
    );
    expect(isValid({ source: "src", target: "dst" })).toBe(true);
  });

  it("always allows connection if target is detokenizer", () => {
    const isValid = makeIsValidConnection(
      new Set(["attention→mlp"]),
      (id) => (id === "src" ? "mlp" : "detokenizer"),
    );
    expect(isValid({ source: "src", target: "dst" })).toBe(true);
  });
});
