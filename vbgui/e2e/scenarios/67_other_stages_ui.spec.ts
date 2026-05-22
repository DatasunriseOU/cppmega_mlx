// H13: Drive the Full pipeline through the UI and walk every stage
// row in the result modal, asserting the extras V5-G21 emits are
// surfaced for non-train stages too (dry_forward / loss_smoke /
// optimizer_smoke). loss_smoke + optimizer_smoke are FULL_STAGES
// only, hence Full not Smoke.
//
// Negative path: rows with status="skipped" don't crash when
// expand-toggled (they may have no extras at all).

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("H13: Full modal exposes every stage's extras", async ({ page }) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Full pipeline (toggle → run-pipeline-full) includes loss_smoke /
  // optimizer_smoke / input_parity_check on top of the Smoke list.
  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("run-pipeline-full").click();
  await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });

  // dry_forward extras: batch / seq_len / hidden / num_nodes.
  await page.getByTestId("run-result-expand-dry_forward").click();
  await page.getByTestId("run-result-extras-row-dry_forward").waitFor();
  for (const k of ["batch", "seq_len", "hidden", "num_nodes"]) {
    const cell = page.getByTestId(`run-result-extras-dry_forward-${k}`);
    await expect(cell).toBeVisible();
    expect(parseInt(
      ((await cell.textContent()) ?? "0").trim(), 10)).toBeGreaterThan(0);
  }

  // loss_smoke: loss_value + loss_finite.
  await page.getByTestId("run-result-expand-loss_smoke").click();
  await page.getByTestId("run-result-extras-row-loss_smoke").waitFor();
  const lossVal = parseFloat(((await page.getByTestId(
    "run-result-extras-loss_smoke-loss_value").textContent()) ?? "NaN").trim());
  expect(Number.isFinite(lossVal)).toBe(true);
  const lossFinite = ((await page.getByTestId(
    "run-result-extras-loss_smoke-loss_finite").textContent()) ?? "")
    .trim().toLowerCase();
  expect(lossFinite).toBe("true");

  // optimizer_smoke: optimizer_kind + num_groups.
  await page.getByTestId("run-result-expand-optimizer_smoke").click();
  await page.getByTestId("run-result-extras-row-optimizer_smoke").waitFor();
  const optKind = ((await page.getByTestId(
    "run-result-extras-optimizer_smoke-optimizer_kind").textContent()) ?? "")
    .trim();
  expect(optKind.length).toBeGreaterThan(0);
  const numGroups = parseInt(((await page.getByTestId(
    "run-result-extras-optimizer_smoke-num_groups").textContent()) ?? "0")
    .trim(), 10);
  expect(numGroups).toBeGreaterThan(0);

  // Negative: skipped stages — if input_parity_check was skipped (no
  // tokenizer wired in Smoke), expanding (or its absence) must not
  // throw. We assert either there's no expand button (no extras) OR
  // clicking it doesn't crash.
  const skippedRow = page.getByTestId("run-result-stage-input_parity_check");
  if (await skippedRow.count() > 0) {
    const status = (await skippedRow.textContent()) ?? "";
    if (status.includes("skipped")) {
      const expandBtn = page.getByTestId(
        "run-result-expand-input_parity_check");
      const btnCount = await expandBtn.count();
      if (btnCount > 0) {
        await expandBtn.click();
        // No throw; modal still mounted.
        await expect(page.getByTestId("run-result-modal")).toBeVisible();
      }
    }
  }

  await closeModal(page);
});
