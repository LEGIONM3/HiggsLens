import { render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LeaderboardView } from "./LeaderboardView";

describe("LeaderboardView Component", () => {
  const mockModels = [
    {
      model_id: "xgboost",
      display_name: "XGBoost Champion",
      roc_auc: 0.91234,
      ams_score: 3.57541,
      optimal_threshold: 0.81182,
      status: "available",
      weights_available: true,
      device: "cuda:0 (RTX 5070 Ti)",
      training_run_origin: "R005 (GPU Sweep)",
      subsample_notes: "Full 250k train set",
      dataset_provenance: "ATLAS open data (record 328, DOI 10.7483/OPENDATA.ATLAS.ZBP2.M5T8)",
    },
    {
      model_id: "lightgbm",
      display_name: "LightGBM Runner Up",
      roc_auc: 0.91142,
      ams_score: 3.49812,
      optimal_threshold: 0.79211,
      status: "available",
      weights_available: true,
      device: "cuda:0 (RTX 5070 Ti)",
      training_run_origin: "R005 (GPU Sweep)",
      subsample_notes: "Full 250k train set",
    },
  ];

  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        if (url.includes("/api/v1/models")) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ models: mockModels }),
          });
        }
        if (url.includes("/thresholds")) {
          return Promise.resolve({
            ok: true,
            json: () =>
              Promise.resolve({
                model_id: "xgboost",
                points: [
                  { threshold: 0.5, ams: 2.5 },
                  { threshold: 0.81182, ams: 3.57541 },
                ],
              }),
          });
        }
        return Promise.reject(new Error("Unknown URL"));
      })
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders frozen banner, QML callout, and certified models table dynamically", async () => {
    render(<LeaderboardView />);

    await waitFor(() => {
      expect(screen.getByText(/OFFICIALLY FROZEN/i)).toBeDefined();
    });

    // QML callout
    expect(
      screen.getByText(/Quantum ML models \(qml_vqc, qml_qaoa\) are experimental research benchmarks/i)
    ).toBeDefined();

    // Model rows
    expect(screen.getByText("XGBoost Champion")).toBeDefined();
    expect(screen.getByText("LightGBM Runner Up")).toBeDefined();

    // Provenance line
    expect(screen.getByText(/record 328, DOI 10.7483\/OPENDATA.ATLAS.ZBP2.M5T8/i)).toBeDefined();
  });
});
