// E2E coverage for new UI components added by bb0 epic (v2):
// DimensionsTab, AblationsTab, ScheduleEditor, BrickContextPanel,
// Tooltip + ExplainModal, AutoGroupButton, Roundtrip badge.

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, clickTab, dropBrickViaPalette,
} from "../fixtures";
import { snapshot } from "../utils/screenshot";
import { loadMatrix } from "../utils/matrix";

// ---------------------------------------------------------------------------
// New sidebar tabs (Dimensions / Ablations)
// ---------------------------------------------------------------------------

test.describe("new sidebar tabs (E7-2 / E7-11-UI)", () => {
  test("Sidebar exposes Dimensions + Ablations tabs", async ({ page }) => {
    await gotoApp(page);
    await expect(page.getByTestId("sidebar-tab-dimensions")).toBeVisible();
    await expect(page.getByTestId("sidebar-tab-ablations")).toBeVisible();
  });

  test("DimensionsTab populates after preset drop", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("sidebar-tab-dimensions").click();
    await page.getByTestId("dimensions-tab").waitFor();
    // Wait for inference_log to populate (after verify completes).
    await expect.poll(async () =>
      await page.locator("[data-testid^='dim-row-']").count(),
      { timeout: 8_000 },
    ).toBeGreaterThan(0);
    await snapshot(page, "10_new_ui", "dimensions_populated");
  });

  test("DimensionsTab source filter narrows rows", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("sidebar-tab-dimensions").click();
    await page.getByTestId("dimensions-tab").waitFor();
    await expect.poll(async () =>
      await page.locator("[data-testid^='dim-row-']").count(),
      { timeout: 8_000 },
    ).toBeGreaterThan(0);
    const before = await page.locator("[data-testid^='dim-row-']").count();
    await page.getByTestId("dimensions-filter-source")
              .selectOption("user");
    const after = await page.locator("[data-testid^='dim-row-']").count();
    expect(after).toBeLessThanOrEqual(before);
  });

  test("AblationsTab axis dropdown + variants render", async ({ page }) => {
    await gotoApp(page);
    await page.getByTestId("sidebar-tab-ablations").click();
    await page.getByTestId("ablations-tab").waitFor();
    await expect(page.getByTestId("ablation-axis")).toBeVisible();
    await expect(page.getByTestId("ablation-run")).toBeVisible();
    // Default activation axis exposes glu + swiglu variants pre-ticked.
    await expect(page.getByTestId("ablation-variant-glu")).toBeVisible();
    await expect(page.getByTestId("ablation-variant-swiglu")).toBeVisible();
  });

  test("AblationsTab axis switch reveals new variant list", async ({ page }) => {
    await gotoApp(page);
    await page.getByTestId("sidebar-tab-ablations").click();
    await page.getByTestId("ablations-tab").waitFor();
    await page.getByTestId("ablation-axis").selectOption("optimizer");
    await expect(page.getByTestId("ablation-variant-lion")).toBeVisible();
    await expect(page.getByTestId("ablation-variant-muon")).toBeVisible();
  });

  test("AblationsTab Run with empty canvas surfaces error", async ({ page }) => {
    await gotoApp(page);
    await page.getByTestId("sidebar-tab-ablations").click();
    await page.getByTestId("ablations-tab").waitFor();
    await page.getByTestId("ablation-run").click();
    await expect(page.getByTestId("ablation-error")).toBeVisible({ timeout: 4_000 });
  });
});

// ---------------------------------------------------------------------------
// ScheduleEditor inside OptimTab (E7-9)
// ---------------------------------------------------------------------------

test.describe("ScheduleEditor in OptimTab (E7-9)", () => {
  test("schedule toggle button reveals editor", async ({ page }) => {
    await gotoApp(page);
    await page.getByTestId("sidebar-tab-optim").click();
    await page.getByTestId("optim-tab").waitFor();
    await page.getByTestId("optim-group-0-schedule-toggle").click();
    await expect(page.getByTestId("schedule-editor-0")).toBeVisible();
  });

  test("selecting cosine reveals total_steps + sparkline", async ({ page }) => {
    await gotoApp(page);
    await page.getByTestId("sidebar-tab-optim").click();
    await page.getByTestId("optim-group-0-schedule-toggle").click();
    await page.getByTestId("schedule-kind-0").selectOption("cosine");
    await expect(page.getByTestId("schedule-total-0")).toBeVisible();
    await expect(page.getByTestId("schedule-sparkline")).toBeVisible();
    await snapshot(page, "10_new_ui", "schedule_cosine");
  });
});

// ---------------------------------------------------------------------------
// AutoGroupButton in OptimTab (E7-4)
// ---------------------------------------------------------------------------

test.describe("AutoGroupButton (E7-4)", () => {
  test("button visible in OptimTab; disabled until canvas has bricks", async ({ page }) => {
    await gotoApp(page);
    await page.getByTestId("sidebar-tab-optim").click();
    const btn = page.getByTestId("optim-auto-group");
    await expect(btn).toBeVisible();
    await expect(btn).toBeDisabled();
  });

  test("Auto-group after preset emits banner with grouping", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("sidebar-tab-optim").click();
    await page.getByTestId("optim-auto-group").click();
    await expect(page.getByTestId("optim-auto-group-banner"))
      .toBeVisible({ timeout: 8_000 });
  });
});

// ---------------------------------------------------------------------------
// BrickContextPanel (E7-5/E7-6)
// ---------------------------------------------------------------------------

test.describe("BrickContextPanel (E7-5 + E7-6)", () => {
  test("clicking a brick opens activation/norm panel", async ({ page }) => {
    await gotoApp(page);
    await dropBrickViaPalette(page, "mlp");
    // pick first brick-node
    const firstNode = page.locator("[data-testid^='brick-node-']").first();
    await firstNode.click();
    // The panel testid uses the node id; locate any brick-context-* root
    const panel = page.locator("[data-testid^='brick-context-']").first();
    await expect(panel).toBeVisible({ timeout: 4_000 });
    // mlp should expose activation dropdown
    const actDropdown = page.locator(
      "[data-testid$='-activation']").first();
    await expect(actDropdown).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// Roundtrip badge (E7-3-UI)
// ---------------------------------------------------------------------------

test("Roundtrip badge surfaces per-row OK/FAIL (E7-3-UI)", async ({ page }) => {
  const matrix = loadMatrix();
  await gotoApp(page);
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
  const text = await page.getByTestId("data-roundtrip-0").textContent();
  expect(text).toMatch(/Roundtrip (OK|FAIL)/);
  await snapshot(page, "10_new_ui", "roundtrip_badge");
});
