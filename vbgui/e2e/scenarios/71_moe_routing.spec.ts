// H18: MoE forward hook surfaces real routing metrics.
//
// Pick a MoE preset (mixtral_like / gpt_oss / etc.) → Train → assert
// extras.moe carries non-null routing_entropy, load_balance_loss,
// per_expert_load (replacing V5-G25 static config-only display).

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

const MOE_PRESET = "qwen3_235b_a22b";  // mixtral_like → MoE brick

test("H18: MoE preset → extras.moe.routing_entropy populated",
  async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, MOE_PRESET);

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();
    await page.getByTestId("run-result-extras-row-train").waitFor();

    // num_experts + top_k always present.
    const numExperts = parseInt(((await page.getByTestId(
      "run-result-extras-train-moe-num_experts").textContent()) ?? "0")
      .trim(), 10);
    expect(numExperts).toBeGreaterThan(1);

    // routing_entropy must be > 0 and ≤ log(num_experts).
    const entropyText = ((await page.getByTestId(
      "run-result-extras-train-moe-routing_entropy").textContent()) ?? "")
      .trim();
    expect(entropyText).not.toBe("null");
    const entropy = parseFloat(entropyText);
    expect(entropy).toBeGreaterThan(0);
    expect(entropy).toBeLessThanOrEqual(Math.log(numExperts) + 1e-6);

    // load_balance_loss non-null and >= 0.
    const lbText = ((await page.getByTestId(
      "run-result-extras-train-moe-load_balance_loss").textContent()) ?? "")
      .trim();
    expect(lbText).not.toBe("null");
    expect(parseFloat(lbText)).toBeGreaterThanOrEqual(0);

    await closeModal(page);
  });
