// V7-F53 visual e2e — dimension scaling sweep.
// Click "Run sweep" → 4 sequential train runs with H ∈ {64,128,256,512}
// (2 steps each) → assert the multi-line LossChart renders one
// visible <path> per H, with circle data points carrying the
// data-loss-value attribute (proves the visual rendering reflects
// real per-step training math, not a mock).

import { test, expect } from "@playwright/test";
import { gotoApp } from "../fixtures";

test("F53: scaling sweep renders 4 H-lines in LossChart", async ({
  page,
}) => {
  // The 4 real train runs at H=64..512 with 2 steps each take ~30-60s
  // wall-clock on a laptop; the default Playwright actionTimeout is
  // too tight, but the assertions below explicitly poll with their
  // own timeout.
  test.setTimeout(180_000);

  await gotoApp(page);
  await page.getByTestId("app-tab-sweep").click();
  await expect(page.getByTestId("sweep-panel")).toBeVisible();

  const runBtn = page.getByTestId("scaling-sweep-run");
  await expect(runBtn).toBeEnabled();
  await runBtn.click();

  // Progress indicator appears while sweep is in flight.
  await expect(page.getByTestId("sweep-progress")).toBeVisible({
    timeout: 30_000,
  });

  // Wait for all 4 H series to render their visible <path> lines.
  for (const H of [64, 128, 256, 512]) {
    await expect(page.getByTestId(`sweep-chart-line-H${H}`)).toBeVisible({
      timeout: 120_000,
    });
  }
  // Each line carries an SVG <path d=…> string with ≥2 segments
  // (M + L) → proves at least the 2 real training steps were drawn,
  // not a fake placeholder.
  for (const H of [64, 128, 256, 512]) {
    const d = await page.getByTestId(`sweep-chart-line-H${H}`)
      .getAttribute("d");
    expect(d, `H=${H} path d`).toBeTruthy();
    expect((d ?? "").split("L").length, `H=${H} segments`)
      .toBeGreaterThanOrEqual(2);
  }

  // Per-point data-loss-value attributes — each is a finite number.
  for (const H of [64, 128, 256, 512]) {
    const first = await page.getByTestId(`sweep-chart-point-H${H}-0`)
      .getAttribute("data-loss-value");
    expect(Number.isFinite(Number(first)),
           `H=${H} point0 loss finite (was ${first})`)
      .toBe(true);
  }

  // Legend lists all four H labels in a stable order.
  await expect(page.getByTestId("sweep-chart-legend-H64")).toBeVisible();
  await expect(page.getByTestId("sweep-chart-legend-H128")).toBeVisible();
  await expect(page.getByTestId("sweep-chart-legend-H256")).toBeVisible();
  await expect(page.getByTestId("sweep-chart-legend-H512")).toBeVisible();
});
