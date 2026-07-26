import React, { useState } from "react";

export interface FeatureAttributionData {
  feature: string;
  value: number;
  contribution: number;
}

export interface ObjectGroupAttributionData {
  group: string;
  total_abs_contribution: number;
  signed_contribution: number;
}

export interface ExplanationPanelProps {
  modelId: string;
  probability: number;
  predictedLabel: string;
  threshold: number;
  baseValue: number;
  margin: number;
  attributions: FeatureAttributionData[];
  objectGroups: ObjectGroupAttributionData[];
  selectedGroupFilter?: string | null;
  onClearGroupFilter?: () => void;
  isLoading?: boolean;
}

export const ExplanationPanel: React.FC<ExplanationPanelProps> = ({
  modelId,
  probability,
  predictedLabel,
  threshold,
  baseValue,
  margin,
  attributions,
  objectGroups,
  selectedGroupFilter,
  onClearGroupFilter,
  isLoading = false,
}) => {
  const [showAll, setShowAll] = useState(false);

  if (isLoading) {
    return (
      <div className="p-4 bg-slate-900/90 border border-slate-800 rounded-xl text-slate-300 animate-pulse">
        <div className="text-sm font-medium mb-2">Loading TreeSHAP attributions...</div>
        <div className="h-2 bg-slate-700 rounded w-3/4 mb-2"></div>
        <div className="h-2 bg-slate-700 rounded w-1/2"></div>
      </div>
    );
  }

  // Filter attributions by group if selectedGroupFilter is set
  const filteredAttributions = selectedGroupFilter
    ? attributions.filter((a) => {
        if (selectedGroupFilter === "tau") return ["PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi", "DER_pt_ratio_lep_tau", "DER_deltar_tau_lep"].includes(a.feature);
        if (selectedGroupFilter === "lepton") return ["PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi"].includes(a.feature);
        if (selectedGroupFilter === "leading_jet") return ["PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi"].includes(a.feature);
        if (selectedGroupFilter === "subleading_jet") return ["PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi"].includes(a.feature);
        if (selectedGroupFilter === "met") return ["PRI_met", "PRI_met_phi", "PRI_met_sumet", "DER_mass_transverse_met_lep", "DER_met_phi_centrality"].includes(a.feature);
        return ["DER_mass_MMC", "DER_mass_vis", "DER_pt_h", "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet", "DER_pt_tot", "DER_sum_pt", "DER_lep_eta_centrality", "PRI_jet_num", "PRI_jet_all_pt"].includes(a.feature);
      })
    : attributions;

  const displayAttributions = showAll ? filteredAttributions : filteredAttributions.slice(0, 10);
  const maxAbsContrib = Math.max(...attributions.map((a) => Math.abs(a.contribution)), 0.1);

  return (
    <div className="p-4 bg-slate-950/95 border border-slate-800 rounded-xl text-slate-100 shadow-xl space-y-4 max-w-xl">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div>
          <span className="text-xs uppercase tracking-wider text-cyan-400 font-mono font-semibold">
            TreeSHAP Feature Attributions
          </span>
          <h3 className="text-sm font-semibold text-slate-200">
            Model: <span className="text-white font-mono">{modelId}</span>
          </h3>
        </div>
        <div className="text-right">
          <div className="text-xs text-slate-400">Signal Probability</div>
          <div className="text-base font-bold font-mono text-cyan-300">
            {(probability * 100).toFixed(1)}% ({predictedLabel})
          </div>
        </div>
      </div>

      {/* Verbatim Mandatory Note 1 */}
      <div className="p-2.5 bg-amber-950/40 border border-amber-800/60 rounded-lg text-amber-200 text-xs leading-relaxed">
        <strong>Interpretation Note:</strong> Feature attributions describe how the model reached its score. They are not statements of physical causation.
      </div>

      {/* Group Filter Badge */}
      {selectedGroupFilter && (
        <div className="flex items-center justify-between bg-cyan-950/60 border border-cyan-800/60 px-3 py-1.5 rounded-lg text-xs text-cyan-200">
          <span>
            Filtering by 3D object: <strong>{selectedGroupFilter}</strong>
          </span>
          <button
            onClick={onClearGroupFilter}
            className="text-cyan-400 hover:text-white text-xs underline font-mono"
          >
            Show All Groups
          </button>
        </div>
      )}

      {/* Log-odds summary */}
      <div className="grid grid-cols-3 gap-2 bg-slate-900/60 p-2.5 rounded-lg border border-slate-800/80 text-xs font-mono">
        <div>
          <span className="text-slate-400 block text-[10px]">Base (Bias)</span>
          <span className="text-slate-200">{baseValue.toFixed(3)}</span>
        </div>
        <div>
          <span className="text-slate-400 block text-[10px]">Margin (Log-Odds)</span>
          <span className="text-slate-200">{margin.toFixed(3)}</span>
        </div>
        <div>
          <span className="text-slate-400 block text-[10px]">Threshold</span>
          <span className="text-cyan-300">{threshold.toFixed(3)}</span>
        </div>
      </div>

      {/* Axis Label - Verbatim Mandatory Note 2 */}
      <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1">
        <span>← Push to Background</span>
        <span className="font-semibold text-slate-300 uppercase tracking-tight">
          contribution to model score (log-odds)
        </span>
        <span>Push to Signal →</span>
      </div>

      {/* Horizontal Bar Chart */}
      <div className="space-y-2">
        {displayAttributions.map((attr) => {
          const widthPct = Math.min((Math.abs(attr.contribution) / maxAbsContrib) * 50, 50);
          const isPositive = attr.contribution >= 0;
          const isSentinel = attr.value === -999.0;

          return (
            <div key={attr.feature} className="text-xs space-y-0.5">
              <div className="flex justify-between items-center text-[11px]">
                <span className="font-mono text-slate-300 truncate max-w-[200px]" title={attr.feature}>
                  {attr.feature}
                </span>
                <span className="font-mono text-slate-400 text-[10px]">
                  {isSentinel ? (
                    <span className="text-amber-400 font-sans">not available (-999)</span>
                  ) : (
                    `val: ${attr.value.toFixed(2)}`
                  )}
                  {" | "}
                  <span className={isPositive ? "text-cyan-400 font-semibold" : "text-rose-400 font-semibold"}>
                    {attr.contribution > 0 ? `+${attr.contribution.toFixed(3)}` : attr.contribution.toFixed(3)}
                  </span>
                </span>
              </div>

              {/* Bar track */}
              <div className="h-2.5 bg-slate-900 rounded-full relative overflow-hidden flex items-center border border-slate-800">
                {/* Center zero axis line */}
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-700 z-10"></div>
                {isPositive ? (
                  <div
                    className="absolute left-1/2 top-0 bottom-0 bg-cyan-500 rounded-r-full"
                    style={{ width: `${widthPct}%` }}
                  ></div>
                ) : (
                  <div
                    className="absolute right-1/2 top-0 bottom-0 bg-rose-500 rounded-l-full"
                    style={{ width: `${widthPct}%` }}
                  ></div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Show All Toggle */}
      {filteredAttributions.length > 10 && (
        <button
          onClick={() => setShowAll(!showAll)}
          className="w-full py-1.5 text-xs text-slate-400 hover:text-white bg-slate-900/80 hover:bg-slate-800 border border-slate-800 rounded-lg transition-colors font-medium"
        >
          {showAll ? "Show Top 10 Features" : `Show All ${filteredAttributions.length} Features`}
        </button>
      )}
    </div>
  );
};
