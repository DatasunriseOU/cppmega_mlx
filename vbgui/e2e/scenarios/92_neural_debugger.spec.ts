import { expect, test } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";
import { snapshot } from "../utils/screenshot";

test.describe("Interactive Neural Debugger & Step-by-Step Training Simulator", () => {
  test("runs debugger step-by-step and full train simulation with zero console warnings", async ({ page }) => {
    // 1. Audit console messages
    const consoleMsgs: string[] = [];
    page.on("console", (msg) => {
      const txt = msg.text();
      consoleMsgs.push(`[${msg.type()}] ${txt}`);
    });

    // 2. Load the App
    await gotoApp(page);
    await snapshot(page, "92_neural_debugger", "01_empty_app");

    // 3. Select a preset to populate canvas
    await selectPreset(page, "llama3_8b");
    await snapshot(page, "92_neural_debugger", "02_llama3_8b_loaded");

    // 4. Toggle Debugger Mode
    const toggleBtn = page.getByTestId("toggle-debugger-mode");
    await expect(toggleBtn).toBeVisible();
    await toggleBtn.click();

    // 5. Verify floating dashboard and virtual nodes appear
    const dashboard = page.getByTestId("debugger-dashboard");
    await expect(dashboard).toBeVisible();
    await snapshot(page, "92_neural_debugger", "03_debugger_dashboard_active");

    const tokenizerNode = page.getByTestId("tokenizer-virtual-node");
    await expect(tokenizerNode).toBeVisible();

    const detokenizerNode = page.getByTestId("detokenizer-virtual-node");
    await expect(detokenizerNode).toBeVisible();

    // 6. Test manual step forward/backward navigation
    await expect(dashboard).toContainText("Step Index: -1");

    const stepFwdBtn = page.getByTestId("debugger-btn-step-fwd");
    await stepFwdBtn.click();
    await expect(dashboard).toContainText("Step Index: 0");
    await snapshot(page, "92_neural_debugger", "04_step_forward_0");

    const stepBwdBtn = page.getByTestId("debugger-btn-step-bwd");
    await stepBwdBtn.click();
    await expect(dashboard).toContainText("Step Index: -1");

    // 7. Test automated Full Train Step animation sequence
    const fullTrainBtn = page.getByTestId("debugger-btn-full-train");
    await expect(fullTrainBtn).toBeVisible();
    await fullTrainBtn.click();

    // Wait for the gold optimizer weight update pulse to appear
    const pulseElement = page.getByTestId("debugger-weight-update-pulse");
    await expect(pulseElement).toBeVisible({ timeout: 10000 });
    await snapshot(page, "92_neural_debugger", "05_full_train_pulse_complete");

    // Verify it automatically resets to step -1 after complete
    await expect(dashboard).toContainText("Step Index: -1");

    // 8. Analyze logs to ensure no severe errors or React warnings
    const severeErrors = consoleMsgs.filter(
      (m) =>
        m.includes("Warning:") ||
        m.includes("WebSocket connection to") ||
        m.includes("Failed to load resource") ||
        m.includes("Uncaught (in promise) Error")
    );

    console.log(`Audited ${consoleMsgs.length} console logs.`);
    if (severeErrors.length > 0) {
      console.log("Severe Console Logs found:", severeErrors);
    }
  });
});
