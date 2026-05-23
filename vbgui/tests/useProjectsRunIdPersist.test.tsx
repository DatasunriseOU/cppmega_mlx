// V7-H41/H42: localStorage helpers for activeTrainRunId + trainPaused
// rehydrate.

import { describe, it, expect, beforeEach } from "vitest";
import {
  loadActiveTrainRunId, saveActiveTrainRunId,
  loadTrainPaused, saveTrainPaused,
} from "@/hooks/useProjects";

describe("V7-H41/H42 useProjects run-state persistence", () => {
  beforeEach(() => {
    localStorage.removeItem("vbgui_active_train_run_id_v1");
    localStorage.removeItem("vbgui_train_paused_v1");
  });

  it("loadActiveTrainRunId returns null when unset", () => {
    expect(loadActiveTrainRunId()).toBeNull();
  });

  it("saveActiveTrainRunId round-trips through localStorage", () => {
    saveActiveTrainRunId("run-42");
    expect(loadActiveTrainRunId()).toBe("run-42");
  });

  it("saveActiveTrainRunId(null) clears the key", () => {
    saveActiveTrainRunId("run-43");
    expect(loadActiveTrainRunId()).toBe("run-43");
    saveActiveTrainRunId(null);
    expect(loadActiveTrainRunId()).toBeNull();
  });

  it("loadTrainPaused defaults to false", () => {
    expect(loadTrainPaused()).toBe(false);
  });

  it("saveTrainPaused round-trips through localStorage", () => {
    saveTrainPaused(true);
    expect(loadTrainPaused()).toBe(true);
    saveTrainPaused(false);
    expect(loadTrainPaused()).toBe(false);
  });
});
