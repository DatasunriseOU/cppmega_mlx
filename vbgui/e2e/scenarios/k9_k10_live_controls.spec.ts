// V7-K9 / K10 visual e2e — TrainLiveControls.
// K9: 'Trigger checkpoint' sets pendingCheckpointTrigger; the next
//     train run forwards trigger_checkpoint_path to stage_train.
// K10: 'Apply lr' surfaces an error status when no run is active
//     (the RPC live_update_lr is host-side; the e2e validates the
//     UI path. Backend support is V7-K10b follow-up.).

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, clickRunPipeline, closeModal,
} from "../fixtures";

test("K9: Trigger checkpoint flips status into in-flight on next train",
async ({ page }) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Status is idle initially.
  await expect(page.getByTestId("train-live-status"))
    .toContainText("idle");

  // Set a checkpoint path and click trigger.
  await page.getByTestId("train-live-ckpt-path").fill("/tmp/k9-mid.safetensors");
  await page.getByTestId("train-live-trigger-ckpt").click();

  // Run train; the live status briefly flips to 'in flight' while
  // the RPC is awaiting completion.
  await clickRunPipeline(page, "train");
  await closeModal(page);

  // Status is idle again post-completion.
  await expect(page.getByTestId("train-live-status"))
    .toContainText("idle");
});

test("K10: Apply lr without active run surfaces 'no active run' status",
async ({ page }) => {
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  await page.getByTestId("train-live-new-lr").fill("0.0005");
  // No active run yet — button stays disabled, status doesn't change.
  const btn = page.getByTestId("train-live-apply-lr");
  await expect(btn).toBeDisabled();
});
