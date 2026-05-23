// H20: accepting a sharding proposal in ShardingTab makes App auto-
// derive fake_ranks from the product of axis degrees. Train extras
// then carry fake_ranks > 1 + non-zero gradient_reduce_ms.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("H20: accept sharding → extras.fake_ranks > 1 + reduce_ms > 0",
  async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("sidebar-tab-sharding").click();
    await page.getByTestId("sharding-tab").waitFor();
    const accept = page.locator(
      "[data-testid^='sharding-accept-']").first();
    await accept.waitFor({ timeout: 8_000 });
    await accept.click();

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();
    await page.getByTestId("run-result-extras-row-train").waitFor();

    const fr = parseInt(((await page.getByTestId(
      "run-result-extras-train-fake_ranks").textContent()) ?? "0").trim(), 10);
    expect(fr).toBeGreaterThan(1);

    const reduceMs = parseFloat(((await page.getByTestId(
      "run-result-extras-train-gradient_reduce_ms").textContent()) ?? "0")
      .trim());
    expect(reduceMs).toBeGreaterThan(0);

    // Verify actual backend cross-device collective execution simulation (math-effect 🟢)
    const isSimulated = await page.getByTestId(
      "run-result-extras-train-is_simulated").textContent();
    expect(isSimulated?.trim()).toBe("true");

    const shardDim = parseInt(
      (await page.getByTestId(
        "run-result-extras-train-sharding_applied-shard_dim")
        .textContent()) ?? "0", 10);
    expect(shardDim).toBeGreaterThan(1);

    const perRankParam = parseInt(
      (await page.getByTestId(
        "run-result-extras-train-sharding_applied-per_rank_param_bytes")
        .textContent()) ?? "0", 10);
    const totalParam = parseInt(
      (await page.getByTestId(
        "run-result-extras-train-sharding_applied-total_param_bytes")
        .textContent()) ?? "0", 10);
    expect(perRankParam).toBe(Math.floor(totalParam / shardDim));

    await closeModal(page);
  });
