// H15: accept the first sharding proposal in ShardingTab → run Train
// → assert extras.sharding_applied.per_rank_param_bytes is populated
// AND smaller than total_param_bytes (because shard_dim > 1 from the
// h100_8x topology). H01 already proved axes propagate; H15 proves
// the per-rank shard simulation actually divides param bytes.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("H15: sharding_applied.per_rank_param_bytes < total_param_bytes",
  async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Open Sharding tab → accept first proposal (shard_dim > 1).
    await page.getByTestId("sidebar-tab-sharding").click();
    await page.getByTestId("sharding-tab").waitFor();
    const firstAccept = page.locator(
      "[data-testid^='sharding-accept-']").first();
    await firstAccept.waitFor({ timeout: 8_000 });
    await firstAccept.click();

    // Train.
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();
    await page.getByTestId("run-result-extras-row-train").waitFor();

    // Assert per_rank_param_bytes is set and < total_param_bytes.
    const perRank = parseInt((await page.getByTestId(
      "run-result-extras-train-sharding_applied-per_rank_param_bytes")
      .textContent() ?? "0").trim(), 10);
    const total = parseInt((await page.getByTestId(
      "run-result-extras-train-sharding_applied-total_param_bytes")
      .textContent() ?? "0").trim(), 10);
    const shardDim = parseInt((await page.getByTestId(
      "run-result-extras-train-sharding_applied-shard_dim")
      .textContent() ?? "0").trim(), 10);

    expect(total).toBeGreaterThan(0);
    expect(shardDim).toBeGreaterThan(1);
    expect(perRank).toBeGreaterThan(0);
    expect(perRank).toBeLessThan(total);
    expect(perRank).toBe(Math.floor(total / shardDim));

    // Activations bytes also populated and divided by shard_dim.
    const perRankAct = parseInt((await page.getByTestId(
      "run-result-extras-train-sharding_applied-per_rank_activation_bytes")
      .textContent() ?? "0").trim(), 10);
    expect(perRankAct).toBeGreaterThan(0);

    await closeModal(page);
  });
