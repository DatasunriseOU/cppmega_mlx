# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: e2e/scenarios/screenshot_generator.spec.ts >> All Presets Canvas Verification >> Capture Wizard & Canvas for Preset: deepseek_v4_flash
- Location: e2e/scenarios/screenshot_generator.spec.ts:22:5

# Error details

```
Error: page.goto: Protocol error (Page.navigate): Cannot navigate to invalid URL
Call log:
  - navigating to "/", waiting until "load"

```

# Test source

```ts
  1   | import { test, expect } from "@playwright/test";
  2   | 
  3   | const ALL_PRESETS = [
  4   |   "arcee_trinity", "deepseek_v3", "deepseek_v4_flash", "gemma3_270m", "gemma3_27b",
  5   |   "gemma4", "gemma4_31b", "glm_45", "glm_45_air", "glm_47", "glm_5", "glm_51",
  6   |   "gpt_oss_120b", "gpt_oss_20b", "granite_4_1", "grok25", "intellect_3", "kimi_k2",
  7   |   "kimi_linear", "laguna_xs2", "ling25", "ling26", "llama3_2_1b", "llama3_2_3b",
  8   |   "llama3_8b", "llama4_maverick", "longcat", "mimo_v2_5", "mimo_v2_5_pro", "mimo_v2_flash",
  9   |   "minimax_m2", "minimax_m2_5", "minimax_m2_7", "mistral4", "mistral_small_3_1",
  10  |   "nanbeige_4_1", "nemotron3", "olmo2_7b", "olmo3_32b", "olmo3_7b", "phi4",
  11  |   "qwen3_235b_a22b", "qwen3_30b_a3b", "qwen3_6_27b", "qwen3_coder_flash", "qwen3_dense_0_6b",
  12  |   "qwen3_dense_32b", "qwen3_dense_4b", "qwen3_dense_8b", "qwen3_next", "sarvam_105b",
  13  |   "sarvam_30b", "smollm3", "step3_5_flash", "tencent_hy3", "tiny_aya", "zaya1",
  14  |   "gpt2_xl", "tiny_aya_parallel", "xlstm_7b", "gemma_4_e2b", "gemma_4_e4b"
  15  | ] as const;
  16  | 
  17  | const CORE_PRESETS = ["llama3_8b", "xlstm_7b", "nemotron3", "deepseek_v4_flash"] as const;
  18  | const ART_DIR = "/Users/dave/.gemini/antigravity-cli/brain/adc0dd7f-7358-462b-b86f-95349940d112";
  19  | 
  20  | test.describe("All Presets Canvas Verification", () => {
  21  |   for (const preset of ALL_PRESETS) {
  22  |     test(`Capture Wizard & Canvas for Preset: ${preset}`, async ({ page }) => {
  23  |       // Fast 30-second timeout for canvas resolution
  24  |       test.setTimeout(30000);
  25  | 
  26  |       // Connect directly to the dev server
> 27  |       await page.goto("/");
      |                  ^ Error: page.goto: Protocol error (Page.navigate): Cannot navigate to invalid URL
  28  |       await page.waitForTimeout(1000);
  29  | 
  30  |       // 1. Select Preset in dropdown launcher
  31  |       await page.getByTestId("preset-launcher").selectOption(preset);
  32  | 
  33  |       // Await wizard modal to become visible
  34  |       const wizard = page.getByTestId("llm-wizard-generate");
  35  |       await expect(wizard).toBeVisible({ timeout: 5000 });
  36  | 
  37  |       // Downscale to 1/32 and set layers to 2 for instant compilation
  38  |       await page.locator("button:has-text('1/32')").click();
  39  |       await page.locator("input[type='number']").fill("2");
  40  | 
  41  |       // Take step 1 screenshot: Wizard Active
  42  |       await page.screenshot({ path: `${ART_DIR}/step1_wizard_${preset}.png` });
  43  | 
  44  |       // Click Generate to build specs dynamically on the backend
  45  |       await page.getByTestId("llm-wizard-generate").click();
  46  | 
  47  |       // Await canvas render showing visual brick nodes
  48  |       await expect.poll(async () =>
  49  |         await page.locator("[data-testid^='brick-node-']").count(),
  50  |         { timeout: 10000 }
  51  |       ).toBeGreaterThan(0);
  52  | 
  53  |       // Take step 2 screenshot: Canvas Loaded
  54  |       await page.screenshot({ path: `${ART_DIR}/step2_canvas_${preset}.png` });
  55  |     });
  56  |   }
  57  | });
  58  | 
  59  | test.describe("Core Representative Presets Ingest & Train", () => {
  60  |   for (const preset of CORE_PRESETS) {
  61  |     test(`Capture Ingest & Train for Preset: ${preset}`, async ({ page }) => {
  62  |       // 120-second timeout to allow local compilation of custom GPU kernels
  63  |       test.setTimeout(120000);
  64  | 
  65  |       // Connect directly to the dev server
  66  |       await page.goto("/");
  67  |       await page.waitForTimeout(1500);
  68  | 
  69  |       // 1. Select Preset
  70  |       await page.getByTestId("preset-launcher").selectOption(preset);
  71  | 
  72  |       const wizard = page.getByTestId("llm-wizard-generate");
  73  |       await expect(wizard).toBeVisible({ timeout: 5000 });
  74  | 
  75  |       // Downscale & Set 2 layers
  76  |       await page.locator("button:has-text('1/32')").click();
  77  |       await page.locator("input[type='number']").fill("2");
  78  |       await page.getByTestId("llm-wizard-generate").click();
  79  | 
  80  |       // Await canvas render
  81  |       await expect.poll(async () =>
  82  |         await page.locator("[data-testid^='brick-node-']").count(),
  83  |         { timeout: 10000 }
  84  |       ).toBeGreaterThan(0);
  85  | 
  86  |       // 2. Select data tab and trigger quickstart ingestion
  87  |       await page.getByTestId("app-tab-data").click();
  88  |       await page.getByTestId("hf-quickstart-modal-open").click();
  89  | 
  90  |       // Configure default quickstart options inside the modal
  91  |       const modalContent = page.getByTestId("hf-quickstart-modal-content");
  92  |       await modalContent.locator("select").first().selectOption("HuggingFaceFW/fineweb-edu");
  93  |       await modalContent.locator("select").nth(1).selectOption("cppmega_v3");
  94  |       await page.getByTestId("hf-quickstart-n-tokens").fill("8192");
  95  | 
  96  |       // Ingest
  97  |       await page.getByTestId("hf-quickstart-run").click();
  98  |       await expect(page.getByTestId("hf-quickstart-result")).toBeVisible({ timeout: 25000 });
  99  | 
  100 |       // Take step 3 screenshot: Data Cached
  101 |       await page.screenshot({ path: `${ART_DIR}/step3_data_${preset}.png` });
  102 | 
  103 |       // Close modal
  104 |       await modalContent.locator("button:has-text('Cancel')").click();
  105 | 
  106 |       // Go back to Canvas tab
  107 |       await page.getByTestId("app-tab-canvas").click();
  108 | 
  109 |       // 3. Select train ops, set loss scale, and start training
  110 |       await page.getByTestId("sidebar-tab-trainops").click();
  111 |       await page.getByTestId("train-options-toggle").click();
  112 |       await page.getByTestId("train-opt-loss_scaler_init_scale").fill("1.0");
  113 | 
  114 |       await page.getByTestId("run-pipeline-toggle").click();
  115 |       if (preset === "nemotron3") {
  116 |         await page.getByTestId("top-bar-precision-mode").selectOption("fp32");
  117 |       }
  118 |       await page.getByTestId("train-num-steps").fill("3");
  119 | 
  120 |       // Run pipeline
  121 |       await page.getByTestId("run-pipeline-train").click();
  122 | 
  123 |       // Wait for training telemetry to appear in live panel
  124 |       const livePanel = page.getByTestId("live-train-panel");
  125 |       await expect(livePanel).toBeVisible({ timeout: 10000 });
  126 | 
  127 |       if (preset !== "deepseek_v4_flash") {
```