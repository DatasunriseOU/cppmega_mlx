// V3-10: Roundtrip FAIL surfaces as red badge in DataInspector but
// does NOT block Train. Synthetic-target stage_train (V3-2 deferred)
// is independent of tokenizer correctness, so a roundtrip failure
// is informational, not fatal.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, clickTab } from "../fixtures";
import { loadMatrix } from "../utils/matrix";

test("V3-10: roundtrip FAIL badge shows; Train remains enabled",
  async ({ page }) => {
    const matrix = loadMatrix();
    // Inject FAIL into data.roundtrip_check via route shim.
    await page.route("**/rpc", async (route, request) => {
      const body = JSON.parse(request.postData() ?? "{}");
      if (body.method !== "data.roundtrip_check") {
        await route.continue();
        return;
      }
      const ups = await page.request.fetch(request);
      const json = await ups.json();
      if (json.result?.rows) {
        json.result.rows = json.result.rows.map((r: {
          row_index: number; byte_diff: number; matches: boolean;
          decoded_preview: string;
        }) => ({
          ...r, matches: false, byte_diff: 7,
          decoded_preview: "fake-roundtrip-fail",
        }));
      }
      await route.fulfill({
        status: 200, contentType: "application/json",
        body: JSON.stringify(json),
      });
    });

    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await clickTab(page, "data");
    await page.getByTestId("data-inspector").waitFor();
    await page.getByTestId("data-path")
      .fill(matrix.parquets.T2_gpt2_small__P1_minimal.path);
    await page.getByTestId("data-load").click();
    await page.getByTestId("data-row-0").waitFor({ timeout: 8_000 });
    await page.getByTestId("data-tokenizer-path")
      .fill(matrix.tokenizers.T2_gpt2_small.path);
    await page.getByTestId("data-roundtrip").click();
    await page.getByTestId("data-roundtrip-0").waitFor({ timeout: 8_000 });

    // FAIL badge present
    const badge = page.getByTestId("data-roundtrip-0");
    await expect(badge).toContainText("Roundtrip FAIL");

    // Train still enabled — roundtrip is a warning, not a block.
    await clickTab(page, "canvas");
    await page.getByTestId("run-pipeline-toggle").click();
    await expect(page.getByTestId("run-pipeline-train")).toBeEnabled();
    await expect(page.getByTestId("top-bar-train-disabled-reason"))
      .toHaveCount(0);
  });
