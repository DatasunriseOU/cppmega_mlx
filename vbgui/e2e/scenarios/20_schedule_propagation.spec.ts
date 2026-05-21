// V4-6: All 6 schedule kinds reach stage_train and shape lr_trajectory
// correctly. V3-5 proved only linear_warmup; v4-6 covers the remaining
// 5 + verifies analytical trajectory shape per kind.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";

type Setup = {
  warmup?: string;        // for non-constant schedules
  total?: string;         // cosine / wsd / polynomial
  decay?: string;         // wsd only
  power?: string;         // polynomial only
};

interface Scenario {
  kind: string;
  setup: Setup;
  /** Asserts lr_trajectory shape for N=8 steps with base_lr governed
   *  by spec default group 0 lr (3e-4 for AdamW). */
  assertTrajectory: (lr: number[]) => void;
}

const SCENARIOS: Scenario[] = [
  {
    kind: "constant",
    setup: {},
    assertTrajectory: (lr) => {
      // All values equal (within rounding to 6 dp).
      const first = lr[0];
      for (const v of lr) expect(v).toBeCloseTo(first, 6);
    },
  },
  {
    kind: "linear_warmup",
    setup: { warmup: "4" },
    assertTrajectory: (lr) => {
      // First step is ramp from 0; step 4+ is at base_lr.
      expect(lr[0]).toBeCloseTo(0, 6);
      expect(lr[4]).toBeGreaterThan(lr[0] + 1e-7);
      expect(lr[4]).toBeGreaterThanOrEqual(lr[3]);
    },
  },
  {
    kind: "cosine",
    setup: { warmup: "2", total: "8" },
    assertTrajectory: (lr) => {
      // After warmup, lr monotonically decays toward min_lr_ratio*base.
      expect(lr[0]).toBeCloseTo(0, 6);
      expect(lr[7]).toBeLessThan(lr[2]);
    },
  },
  {
    kind: "wsd",
    setup: { warmup: "2", total: "8", decay: "4" },
    assertTrajectory: (lr) => {
      // Warmup → stable → decay. Last < peak.
      expect(lr[0]).toBeCloseTo(0, 6);
      expect(lr[7]).toBeLessThan(Math.max(...lr));
    },
  },
  {
    kind: "inv_sqrt",
    setup: { warmup: "2" },
    assertTrajectory: (lr) => {
      // Warmup ramps then 1/sqrt decay.
      expect(lr[0]).toBeCloseTo(0, 6);
      expect(lr[7]).toBeLessThan(lr[2] + 1e-9);
    },
  },
  {
    kind: "polynomial",
    setup: { warmup: "2", total: "8", power: "2" },
    assertTrajectory: (lr) => {
      // (1 - t/T)^power monotonic decay after warmup.
      expect(lr[0]).toBeCloseTo(0, 6);
      expect(lr[7]).toBeLessThan(lr[2]);
    },
  },
];

for (const { kind, setup, assertTrajectory } of SCENARIOS) {
  test(`V4-6: schedule '${kind}' propagates kind + lr_trajectory shape`,
    async ({ page }) => {
      test.setTimeout(60_000);
      await gotoApp(page);
      await selectPreset(page, "llama3_8b");

      await page.getByTestId("sidebar-tab-optim").click();
      await page.getByTestId("optim-group-0-schedule-toggle").click();
      await page.getByTestId("schedule-kind-0").selectOption(kind);
      if (setup.warmup) {
        await page.getByTestId("schedule-warmup-0").fill(setup.warmup);
      }
      if (setup.total) {
        await page.getByTestId("schedule-total-0").fill(setup.total);
      }
      if (setup.decay) {
        await page.getByTestId("schedule-decay-0").fill(setup.decay);
      }
      if (setup.power) {
        await page.getByTestId("schedule-power-0").fill(setup.power);
      }
      await page.getByTestId("optim-apply").click();

      // Run Train with N=8 so trajectory shape is observable.
      await page.getByTestId("run-pipeline-toggle").click();
      await page.getByTestId("train-num-steps").fill("8");
      await page.getByTestId("run-pipeline-train").click();
      const modal = page.getByTestId("run-result-modal");
      await modal.waitFor({ timeout: 60_000 });

      const extras = await readTrainExtras(page);

      // Propagation
      expect(extras.schedule_kind).toBe(kind);
      expect(extras.model_summary.schedule_kind).toBe(kind);
      expect(extras.num_steps).toBe(8);
      expect(extras.lr_trajectory.length).toBe(8);

      // Shape
      assertTrajectory(extras.lr_trajectory);

      await closeModal(page);
    });
}
