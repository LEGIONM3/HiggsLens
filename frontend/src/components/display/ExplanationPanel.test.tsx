import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";
import { ExplanationPanel } from "./ExplanationPanel";

describe("ExplanationPanel Component", () => {
  const mockAttributions = [
    { feature: "DER_mass_MMC", value: 120.5, contribution: 0.85 },
    { feature: "PRI_tau_pt", value: 45.2, contribution: 0.32 },
    { feature: "DER_mass_jet_jet", value: -999.0, contribution: -0.15 },
    { feature: "PRI_met", value: 33.1, contribution: -0.42 },
  ];

  const mockGroups = [
    { group: "tau", total_abs_contribution: 0.32, signed_contribution: 0.32 },
    { group: "met", total_abs_contribution: 0.42, signed_contribution: -0.42 },
    { group: "global", total_abs_contribution: 1.0, signed_contribution: 0.7 },
  ];

  it("renders model info, probability, and verbatim interpretation notes", () => {
    render(
      <ExplanationPanel
        modelId="xgboost"
        probability={0.812}
        predictedLabel="signal"
        threshold={0.8118}
        baseValue={-0.25}
        margin={1.46}
        attributions={mockAttributions}
        objectGroups={mockGroups}
      />
    );

    // Header & probability
    expect(screen.getByText("xgboost")).toBeDefined();
    expect(screen.getByText("81.2% (signal)")).toBeDefined();

    // Verbatim Mandatory Note 1
    expect(
      screen.getByText(
        /Feature attributions describe how the model reached its score\. They are not statements of physical causation\./i
      )
    ).toBeDefined();

    // Verbatim Mandatory Note 2 Axis Label
    expect(
      screen.getByText(/contribution to model score \(log-odds\)/i)
    ).toBeDefined();

    // Sentinel badge
    expect(screen.getByText("not available (-999)")).toBeDefined();
  });
});
