// V3-7: AblationsTab axis variants must produce real loss / weight
// divergence — not just N rows in a table. Closes the "AblationsTab
// Run produces results table with variants" vacuous gap from v2.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";

async function runAblation(
  page: import("@playwright/test").Page,
  axis: "activation" | "optimizer" | "norm" | "schedule",
  variants: string[],
  numSteps: number,
): Promise<void> {
  await page.getByTestId("sidebar-tab-ablations").click();
  await page.getByTestId("ablations-tab").waitFor();
  await page.getByTestId("ablation-axis").selectOption(axis);

  // Tick only the requested variants — uncheck whatever's pre-selected.
  const allLabels = page.locator("[data-testid^='ablation-variant-']");
  const count = await allLabels.count();
  for (let i = 0; i < count; i++) {
    const label = allLabels.nth(i);
    const tid = await label.getAttribute("data-testid");
    const v = tid?.replace("ablation-variant-", "") ?? "";
    const checkbox = label.locator("input[type='checkbox']");
    const want = variants.includes(v);
    const isChecked = await checkbox.isChecked();
    if (want !== isChecked) await checkbox.click();
  }

  await page.getByTestId("ablation-num-steps").fill(String(numSteps));
  await page.getByTestId("ablation-run").click();
  await page.getByTestId("ablation-results").waitFor({ timeout: 60_000 });
}

async function rowFinal(
  page: import("@playwright/test").Page, variant: string,
): Promise<number> {
  const text = await page.getByTestId(`ablation-final-${variant}`)
    .textContent();
  return parseFloat(text?.trim() ?? "NaN");
}

// ---------------------------------------------------------------------------
// Activation axis: different activations should produce DIFFERENT final loss
// ---------------------------------------------------------------------------

test("activation ablation: glu vs swiglu vs gelu produce different losses",
  async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await runAblation(page, "activation", ["glu", "swiglu", "gelu"], 4);

    const glu     = await rowFinal(page, "glu");
    const swiglu  = await rowFinal(page, "swiglu");
    const gelu    = await rowFinal(page, "gelu");

    expect([glu, swiglu, gelu].every(Number.isFinite)).toBe(true);

    // At least one pair must differ by > 1e-3 in absolute terms — proves
    // the activation swap actually reached the model. Synthetic data
    // makes exact ordering noisy but identical results are impossible.
    const maxDiff = Math.max(
      Math.abs(glu - swiglu),
      Math.abs(swiglu - gelu),
      Math.abs(glu - gelu),
    );
    expect(maxDiff).toBeGreaterThan(1e-3);
  });

// ---------------------------------------------------------------------------
// Optimizer axis: AdamW vs Lion → different final loss (math divergence)
// ---------------------------------------------------------------------------

test("optimizer ablation: adamw vs lion final losses differ", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");
  await runAblation(page, "optimizer", ["adamw", "lion"], 4);

  const adamw = await rowFinal(page, "adamw");
  const lion  = await rowFinal(page, "lion");

  expect(Number.isFinite(adamw) && Number.isFinite(lion)).toBe(true);

  // AdamW and Lion are fundamentally different update rules — sign-only
  // vs second-moment adaptive. Their final losses after 4 steps must
  // differ. V2 only counted rows, this is the math-divergence proof.
  expect(Math.abs(adamw - lion)).toBeGreaterThan(1e-4);
});
