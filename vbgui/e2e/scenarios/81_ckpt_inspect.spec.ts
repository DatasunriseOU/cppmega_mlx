// V7-C03: ckpt.inspect RPC + TopBar metadata display on Load.
// Scenario: run a Train with checkpoint_save_path, then a second
// Train with checkpoint_load_path pointing at the same file —
// before clicking Train, the TopBar must render arch_hash /
// opt_kind / version parsed out of the just-written metadata.

import { test, expect } from "@playwright/test";
import path from "node:path";
import os from "node:os";
import fs from "node:fs";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

function tmpCkptPath(): string {
  const tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), "v7c03-"));
  return path.join(tmpdir, "ckpt.safetensors");
}

test("V7-C03: ckpt.inspect renders arch_hash + opt_kind + version on Load",
  async ({ page }) => {
    test.setTimeout(90_000);
    const ckptPath = tmpCkptPath();
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Wave 1: train + save.
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("train-checkpoint-save-path").fill(ckptPath);
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await closeModal(page);

    // Wave 2: type the same path into ckpt-load. The TopBar's
    // debounced effect must fire ckpt.inspect and surface metadata.
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-checkpoint-load-path").fill(ckptPath);

    // Wait for inspect block.
    await expect(page.getByTestId("ckpt-inspect-info"))
      .toBeVisible({ timeout: 5_000 });

    const archHashEl = page.getByTestId("ckpt-inspect-arch-hash");
    await expect(archHashEl).toBeVisible();
    const archHashText = await archHashEl.textContent();
    // truncated to 12 chars + ellipsis, but must start with "arch: "
    expect(archHashText).toMatch(/^arch: [0-9a-f]{12}…$/);

    const optKindEl = page.getByTestId("ckpt-inspect-opt-kind");
    const optKindText = await optKindEl.textContent();
    expect(optKindText).toContain("opt: ");
    expect(optKindText).not.toContain("opt: ?");

    const stepEl = page.getByTestId("ckpt-inspect-step");
    const stepText = await stepEl.textContent();
    expect(stepText).not.toContain("step: ?");

    const versionEl = page.getByTestId("ckpt-inspect-version");
    const versionText = await versionEl.textContent();
    expect(versionText).not.toContain("v: ?");
  });

test("V7-C03: ckpt.inspect renders 'file not found' for missing path",
  async ({ page }) => {
    test.setTimeout(30_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-checkpoint-load-path")
      .fill("/tmp/definitely_does_not_exist_v7c03.safetensors");

    await expect(page.getByTestId("ckpt-inspect-missing"))
      .toBeVisible({ timeout: 5_000 });
  });
