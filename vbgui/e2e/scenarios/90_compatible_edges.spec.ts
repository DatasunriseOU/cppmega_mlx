// E-AUDIT-02: FlowCanvas isValidConnection wired via
// catalog.list_options('compatible_edges'). After preset load, the
// edge-compatibility hook has fetched the pair set; FlowCanvas
// passes it to React Flow's isValidConnection prop. We assert that
// the catalog endpoint returns at least one pair and that no
// runtime errors leak into the page (incompatible-drop UI behavior
// is covered by the vitest unit test).

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";

test("E-AUDIT-02: compatible_edges populated after preset load",
  async ({ page }) => {
    test.setTimeout(30_000);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    await gotoApp(page);

    const catalogResp = await page.evaluate(async () => {
      const r = await fetch("http://127.0.0.1:8767/rpc", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          jsonrpc: "2.0", id: "c1",
          method: "catalog.list_options",
          params: { category: "compatible_edges" },
        }),
      });
      return await r.json();
    });
    expect(catalogResp.error).toBeUndefined();
    const opts = catalogResp.result.options as Array<{
      name: string; summary: string; paper_ref: string;
    }>;
    expect(opts.length).toBeGreaterThan(0);
    expect(opts[0].name).toContain("->");
    expect(opts[0].paper_ref).toBe(opts[0].name.split("->")[0]);

    await selectPreset(page, "llama3_8b");
    // No page errors after preset load + hook fetch.
    expect(errors).toEqual([]);
  });
