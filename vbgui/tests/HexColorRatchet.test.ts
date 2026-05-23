// UX#8 — ratchet test for hex colour literals in vbgui components.
//
// 518 hex literals across 45 component files (commit 5a4e314 baseline).
// Replacing all in one go would be a giant risky diff, so this test
// runs as a ratchet: every file has a max-allowed hex count locked in
// tests/fixtures/hex_color_baseline.json. New code is forbidden from:
//
//   - Exceeding the baseline count for an existing file (ratchet down).
//   - Adding a NEW file with hex literals not present in the baseline.
//
// Migration path is to replace `#xxxxxx` literals with `T.surface`,
// `T.text`, `T.success`, etc. from `@/theme`, then drop the file's
// baseline number. Drop the entry entirely when count hits 0.

import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import baseline from "./fixtures/hex_color_baseline.json";

const HERE = dirname(fileURLToPath(import.meta.url));
const COMPONENTS_DIR = resolve(HERE, "..", "src", "components");
const HEX_RE = /#[0-9a-fA-F]{3,8}\b/g;
const IGNORE_FILES = new Set(["theme.ts"]);

function walk(dir: string, rel = ""): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    if (IGNORE_FILES.has(entry)) continue;
    const full = join(dir, entry);
    const relPath = rel ? `${rel}/${entry}` : entry;
    if (statSync(full).isDirectory()) {
      out.push(...walk(full, relPath));
    } else if (entry.endsWith(".ts") || entry.endsWith(".tsx")) {
      out.push(`components/${relPath}`);
    }
  }
  return out;
}

function countHex(absPath: string): number {
  const src = readFileSync(absPath, "utf-8");
  // Ignore line + block comments so docs/JSDoc with example colours
  // don't dominate the count.
  const stripped = src
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/[^\n]*/g, (_m, p1) => p1);
  return (stripped.match(HEX_RE) ?? []).length;
}

const baselineMap = baseline as Record<string, number>;

describe("UX#8: hex color ratchet (only counts can decrease)", () => {
  const files = walk(COMPONENTS_DIR);

  it("does not introduce hex colors in new files", () => {
    const offenders: string[] = [];
    for (const f of files) {
      if (f in baselineMap) continue;
      const abs = join(COMPONENTS_DIR, f.replace(/^components\//, ""));
      const n = countHex(abs);
      if (n > 0) {
        offenders.push(`${f}: ${n} hex literal${n === 1 ? "" : "s"}`);
      }
    }
    if (offenders.length > 0) {
      throw new Error(
        "UX#8: new files contain hex literals — use theme tokens " +
        "from @/theme (T.surface, T.text, T.success, etc.) instead. " +
        "If you must add a hex literal, bump the baseline at " +
        "tests/fixtures/hex_color_baseline.json:\n  " +
        offenders.join("\n  "),
      );
    }
  });

  it("does not exceed per-file baseline hex counts", () => {
    const offenders: string[] = [];
    for (const [rel, limit] of Object.entries(baselineMap)) {
      const abs = join(COMPONENTS_DIR, rel.replace(/^components\//, ""));
      try {
        const n = countHex(abs);
        if (n > limit) {
          offenders.push(`${rel}: ${n} > baseline ${limit}`);
        }
      } catch {
        // File deleted — that's fine, baseline becomes stale but
        // doesn't fail; CI can drop the entry on next commit.
      }
    }
    if (offenders.length > 0) {
      throw new Error(
        "UX#8: hex literal counts exceed baseline — please replace " +
        "with theme tokens (T.surface/T.text/etc) from @/theme, or " +
        "if the addition is intentional, update the baseline at " +
        "tests/fixtures/hex_color_baseline.json:\n  " +
        offenders.join("\n  "),
      );
    }
  });

  it("baseline file is non-empty and tracks > 0 violations " +
     "(starting reality, not aspiration)", () => {
    const total = Object.values(baselineMap)
      .reduce((a, b) => a + b, 0);
    expect(Object.keys(baselineMap).length).toBeGreaterThan(0);
    expect(total).toBeGreaterThan(0);
  });
});
