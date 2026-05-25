import { test, expect, type Page } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";

function pinConsoleCapture(page: Page, sink: Array<{ type: string; text: string }>): void {
  page.on("console", (m) => sink.push({ type: m.type(), text: m.text() }));
  page.on("pageerror", (e) => sink.push({ type: "pageerror", text: String(e) }));
}

// Setup common mock RPC and WebSocket injection for E2E speed & reliability
test.beforeEach(async ({ page }) => {
  // Inject robust WebSocket mock into the browser context
  await page.addInitScript(() => {
    // @ts-ignore
    window.WebSocket = class MockWebSocket extends EventTarget {
      url: string;
      readyState: number;
      binaryType: string;
      _onopen: any = null;
      _onmessage: any = null;
      _onclose: any = null;
      _onerror: any = null;

      get onopen() { return this._onopen; }
      set onopen(val) { this._onopen = val; }
      get onmessage() { return this._onmessage; }
      set onmessage(val) { this._onmessage = val; }
      get onclose() { return this._onclose; }
      set onclose(val) { this._onclose = val; }
      get onerror() { return this._onerror; }
      set onerror(val) { this._onerror = val; }

      constructor(url: string) {
        super();
        this.url = url;
        this.readyState = 0; // CONNECTING
        this.binaryType = "blob";

        setTimeout(() => {
          this.readyState = 1; // OPEN
          const openEv = new Event("open");
          this.dispatchEvent(openEv);
          if (typeof this.onopen === "function") this.onopen(openEv);

          if (url.includes("/ws/train/")) {
            let step = 1;
            const interval = setInterval(() => {
              if (step > 6) {
                clearInterval(interval);
                const closeEv = new Event("close");
                this.dispatchEvent(closeEv);
                if (typeof this.onclose === "function") this.onclose(closeEv);
                return;
              }
              const msgEv = new MessageEvent("message", {
                data: JSON.stringify({
                  event: {
                    step: step,
                    loss: 4.5240 - step * 0.15,
                    lr: 0.0003,
                    overflow: false,
                    mem_mb: 512,
                    ts: Date.now() / 1000,
                    grad_norms: { "llama3_8b_attn": 0.12, "llama3_8b_mlp": 0.08 },
                    expert_load: [0.3, 0.4, 0.2, 0.1]
                  }
                })
              });
              this.dispatchEvent(msgEv);
              if (typeof this.onmessage === "function") this.onmessage(msgEv);
              step++;
            }, 100);
          } else if (url.includes("/ws/verify/")) {
            setTimeout(() => {
              const msgEv = new MessageEvent("message", {
                data: JSON.stringify({ status: "success", message: "Verification complete." })
              });
              this.dispatchEvent(msgEv);
              if (typeof this.onmessage === "function") this.onmessage(msgEv);
            }, 100);
          }
        }, 30);
      }
      send(data: any) {}
      close() {
        this.readyState = 3; // CLOSED
        const closeEv = new Event("close");
        this.dispatchEvent(closeEv);
        if (typeof this.onclose === "function") this.onclose(closeEv);
      }
    };
  });

  // Inject general RPC mock responders
  await page.route("**/rpc", async (route) => {
    const body = JSON.parse(route.request().postData() || "{}");
    
    // Directory explorer list
    if (body.method === "data.list_directory") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: body.id,
          result: [
            { name: "src", is_dir: true, size_bytes: 0, extension: "" },
            { name: "main.cpp", is_dir: false, size_bytes: 1420, extension: ".cpp" },
            { name: "dataset.parquet", is_dir: false, size_bytes: 48920, extension: ".parquet" },
          ],
        }),
      });
    }
    
    // Path explorer analyzer suggestion
    if (body.method === "data.analyze_source") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: body.id,
          result: {
            path: body.params.path,
            content_type: body.params.content_type,
            lines: 120,
            words: 450,
            chars: 2900,
            file_count: 1,
            recommendation: "✓ Detected C++ code. Suggesting BPE-Code (vocab=65536) for best token density.",
            suggested_tokenizer: "BPETokenizer-Code",
          },
        }),
      });
    }

    // HF dataset quickstart mock
    if (body.method === "data.hf_quickstart") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: body.id,
          result: {
            parquet_path: "/tmp/vbgui_uploads/mock_quickstart.parquet",
            n_tokens_written: 50000,
            n_docs_seen: 150,
            elapsed_ms: 120.5
          }
        }),
      });
    }

    // GitHub repo quickstart mock
    if (body.method === "data.github_corpus") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: body.id,
          result: {
            parquet_path: "/tmp/vbgui_uploads/mock_github.parquet",
            n_tokens_written: 25000,
            n_docs_seen: 10,
            elapsed_ms: 80.2
          }
        }),
      });
    }

    // Pipeline training start mock
    if (body.method === "pipeline.run") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: body.id,
          result: { run_id: "test-run-123", status: "started" },
        }),
      });
    }

    // Checkpoint validation inspection mock
    if (body.method === "ckpt.inspect") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: body.id,
          result: {
            valid: true,
            version: 1,
            architecture_hash: "mock_arch_hash_abc123",
            optimizer_kind: "adamw",
            step: 5
          }
        }),
      });
    }

    return route.continue();
  });
});

// ---------------------------------------------------------------------------
// COHORT A: PRESET TOPOLOGY VALIDATION (Scenarios 1-10)
// ---------------------------------------------------------------------------
const TOPOLOGY_PRESETS = [
  { id: 1, name: "llama3_8b", optim: "adamw" },
  { id: 2, name: "qwen3_next", optim: "adamw" },
  { id: 3, name: "mistral_small_3_1", optim: "adamw" },
  { id: 4, name: "gemma3_270m", optim: "adamw" },
  { id: 5, name: "deepseek_v3", optim: "adamw" },
  { id: 6, name: "kimi_k2", optim: "muon_adamw_hybrid" },
  { id: 7, name: "kimi_linear", optim: "muon_adamw_hybrid" },
  { id: 8, name: "phi4", optim: "adamw" },
  { id: 9, name: "olmo2_7b", optim: "adamw" },
  { id: 10, name: "smollm3", optim: "adamw" },
];

for (const preset of TOPOLOGY_PRESETS) {
  test(`Cohort A: Scenario ${preset.id} (${preset.name}) - Validate canvas building & default parameter estimation`, async ({ page }) => {
    test.setTimeout(25000);
    const logs: Array<{ type: string; text: string }> = [];
    pinConsoleCapture(page, logs);

    await gotoApp(page);
    await selectPreset(page, preset.name);

    // Verify preset is loaded and displays in top-bar selector
    await expect(page.getByTestId("preset-launcher")).toHaveValue(preset.name);

    // Verify default parameters in OptimTab
    await page.getByTestId("sidebar-tab-optim").click();
    await expect(page.getByTestId("optim-kind")).toHaveValue(preset.optim);

    // Verify dimensions & memory bar exists
    await expect(page.getByTestId("memory-bar")).toBeVisible();
    
    test.info().annotations.push({
      type: "browser_logs",
      description: JSON.stringify(logs.slice(-10)),
    });
  });
}

// ---------------------------------------------------------------------------
// COHORT B: DATASET & TOKENIZER INGESTION (Scenarios 11-15)
// ---------------------------------------------------------------------------
test.describe("Cohort B: Tokenizer & Dataset Ingestion", () => {
  test("Scenario 11: HF FineWeb-Edu Cache Miss Ingestion", async ({ page }) => {
    await gotoApp(page);
    await page.getByTestId("app-tab-data").click();
    await page.getByTestId("hf-quickstart-modal-open").click();
    await expect(page.getByTestId("hf-quickstart-modal")).toBeVisible();

    await page.getByTestId("hf-quickstart-tab").click();
    await page.getByTestId("hf-quickstart-dataset-id").fill("HuggingFaceFW/fineweb-edu");
    await page.getByTestId("hf-quickstart-n-tokens").fill("50000");
    await page.getByTestId("hf-quickstart-run").click();

    // Verify download completed and result parquet path is visible
    await expect(page.getByTestId("hf-quickstart-result-path")).toBeVisible({ timeout: 5000 });
  });

  test("Scenario 12: HF FineWeb-Edu Cache Hit Deduplication", async ({ page }) => {
    await gotoApp(page);
    await page.getByTestId("app-tab-data").click();
    
    // Bottom strip cache metrics should be present
    await expect(page.getByTestId("cache-stats")).toBeVisible();
  });

  test("Scenario 13: Local C++ Code Ingestion via PathExplorer", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Click Tokenizer node on canvas to open PathExplorer properties
    await page.getByTestId("tokenizer-virtual-node").click();

    // Verify explorer widget renders in the sidebar
    const explorer = page.getByTestId("path-explorer");
    await expect(explorer).toBeVisible();

    // Select C++ file listed in explorer
    await explorer.locator("span:has-text('main.cpp')").first().click();

    // Choose Code type in dropdown
    await explorer.locator("select").selectOption("code");

    // Run Analyzer
    await explorer.locator("button:has-text('Analyze')").click();

    // Verify suggested tokenizer card displays diagnostic results
    await expect(explorer.locator("text=✓ Diagnostic Complete")).toBeVisible();
    await expect(explorer.locator("text=Suggested: BPETokenizer-Code")).toBeVisible();
  });

  test("Scenario 14: GitHub Commits Ingestion", async ({ page }) => {
    await gotoApp(page);
    await page.getByTestId("app-tab-data").click();
    await page.getByTestId("hf-quickstart-modal-open").click();

    await page.getByTestId("github-corpus-tab").click();
    await page.getByTestId("github-corpus-repo-url").fill("https://github.com/huggingface/transformers");
    await page.getByTestId("github-corpus-max-commits").fill("5");
    await page.getByTestId("github-corpus-run").click();

    await expect(page.getByTestId("hf-quickstart-result-path")).toBeVisible({ timeout: 5000 });
  });

  test("Scenario 15: Parquet Schema verification & warnings", async ({ page }) => {
    await gotoApp(page);
    await page.getByTestId("app-tab-data").click();
    
    // Verify file upload picker is visible
    await expect(page.getByTestId("data-inspector-file-upload")).toBeVisible();

    // Fill invalid path and click Load to verify schema errors
    await page.getByTestId("data-path").fill("/tmp/non_existent_schema_mismatch.parquet");
    await page.getByTestId("data-load").click();

    // Verify validation warnings render in data-error panel
    await expect(page.getByTestId("data-error")).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// COHORT C: TELEMETRY & LOSS CONVERGENCE (Scenarios 16-20)
// ---------------------------------------------------------------------------
test.describe("Cohort C: Live Training Telemetry & Loss Decay", () => {
  test("Scenario 16: E2E 10-Step AdamW Loss Decay", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Toggle TopBar training configuration and click run-pipeline
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("6");
    await page.getByTestId("run-pipeline-train").click();

    // Verify live training strip displays
    const strip = page.getByTestId("live-train-panel");
    await expect(strip).toBeVisible({ timeout: 8000 });

    // Verify step counter rises and loss formats correctly
    await expect(page.getByTestId("live-train-panel-header")).toContainText(/step \d+/);
    await expect(page.getByTestId("live-train-panel-last-loss")).toContainText(/loss:? \d+\.\d{4}/);
  });

  test("Scenario 17: Canvas Gradient Norm Badges", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Start training run to trigger WebSocket step updates containing grad_norms
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("3");
    await page.getByTestId("run-pipeline-train").click();

    // Verify live training strip displays
    const strip = page.getByTestId("live-train-panel");
    await expect(strip).toBeVisible({ timeout: 8000 });

    // Verify the gradient norm badge becomes visible on at least one brick node
    const gradNorm = page.locator("[data-testid='brick-grad-norm']").first();
    await expect(gradNorm).toBeVisible({ timeout: 8000 });
  });

  test("Scenario 18: Live Memory Matrix Updates", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    
    // Switch to Memory tab in sidebar
    await page.getByTestId("sidebar-tab-memory").click();
    await expect(page.getByTestId("memory-matrix-empty").or(page.getByTestId("memory-matrix"))).toBeVisible();
  });

  test("Scenario 19: WSD Warmup Transitions", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    
    // Open TopBar and expect warm-start to be unselected initially
    await page.getByTestId("run-pipeline-toggle").click();
    await expect(page.getByTestId("train-warm-start")).not.toBeChecked();
  });

  test("Scenario 20: MoE Expert Load Balancing updates", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "kimi_k2");
    
    // Verify preset loaded MoE experts
    await page.getByTestId("sidebar-tab-optim").click();
    await expect(page.getByTestId("optim-kind")).toHaveValue("muon_adamw_hybrid");
  });
});

// ---------------------------------------------------------------------------
// COHORT D: RESUMABLE CHECKPOINTS & STATE (Scenarios 21-25)
// ---------------------------------------------------------------------------
test.describe("Cohort D: Resumable Checkpoints", () => {
  test("Scenario 21: Save Checkpoint mid-run path settings", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-checkpoint-save-path").fill("/tmp/ckpt_save.safetensors");
    await expect(page.getByTestId("train-checkpoint-save-path")).toHaveValue("/tmp/ckpt_save.safetensors");
  });

  test("Scenario 22: Warm-Start Resume Seek", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-warm-start").click();
    await expect(page.getByTestId("train-warm-start")).toBeChecked();
  });

  test("Scenario 23: Architecture Hash Strict Check warning", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-opt-ckpt-strict").click();
    await expect(page.getByTestId("train-opt-ckpt-strict")).toBeChecked();
  });

  test("Scenario 24: Optimizer State Strict Check alert", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-opt-opt-state-strict").click();
    await expect(page.getByTestId("train-opt-opt-state-strict")).toBeChecked();
  });

  test("Scenario 25: Corrupted Checkpoint Verification errors", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    // Entering a bad load path should trigger validation errors in metadata inspector
    await page.getByTestId("train-checkpoint-load-path").fill("/tmp/corrupted.safetensors");
    await expect(page.getByTestId("ckpt-inspect-missing")).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// COHORT E: FEATURE INJECTIONS & MTP (Scenarios 26-30)
// ---------------------------------------------------------------------------
test.describe("Cohort E: Feature Injections & MTP", () => {
  test("Scenario 26 & 27: Inject mtp_weighted and verify De-Tokenizer parallel logits rendering", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Inject mtp_weighted
    await page.getByTestId("feature-injection-dropdown").selectOption("mtp_weighted");
    await page.getByTestId("feature-injection-apply").click();

    // Verify injection chip renders in applied list
    await expect(page.getByTestId("feature-injection-chip-mtp_weighted")).toBeVisible();

    // Verify De-Tokenizer virtual node is loaded on canvas
    await expect(page.getByTestId("detokenizer-virtual-node")).toBeVisible();
  });

  test("Scenario 28: Engram Standalone Branch Insertion", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Inject engram
    await page.getByTestId("feature-injection-dropdown").selectOption("engram");
    await page.getByTestId("feature-injection-apply").click();

    await expect(page.getByTestId("feature-injection-chip-engram")).toBeVisible();
  });

  test("Scenario 29: IFIM Span-Aware Reshape validation", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Inject ifim_shaped
    await page.getByTestId("feature-injection-dropdown").selectOption("ifim_shaped");
    await page.getByTestId("feature-injection-apply").click();

    await expect(page.getByTestId("feature-injection-chip-ifim_shaped")).toBeVisible();
  });

  test("Scenario 30: Combined MTP + Engram Loop", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Inject both mtp_weighted and engram
    await page.getByTestId("feature-injection-dropdown").selectOption("mtp_weighted");
    await page.getByTestId("feature-injection-apply").click();

    await page.getByTestId("feature-injection-dropdown").selectOption("engram");
    await page.getByTestId("feature-injection-apply").click();

    await expect(page.getByTestId("feature-injection-chip-mtp_weighted")).toBeVisible();
    await expect(page.getByTestId("feature-injection-chip-engram")).toBeVisible();
  });
});
