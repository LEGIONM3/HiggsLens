import React, { useEffect, useState } from "react";
import { ThresholdExplorer } from "./ThresholdExplorer";
import type { ThresholdScanPoint } from "../../lib/thresholdExplorer";

export interface ModelSummary {
  model_id: string;
  display_name: string;
  roc_auc: number;
  ams_score: number;
  optimal_threshold: number;
  status: string;
  weights_available: boolean;
  device?: string;
  training_run_origin?: string;
  subsample_notes?: string;
  dataset_provenance?: string;
}

export const LeaderboardView: React.FC = () => {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("xgboost");
  const [scanPoints, setScanPoints] = useState<ThresholdScanPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/v1/models")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        return res.json();
      })
      .then((data) => {
        const sorted = (data.models as ModelSummary[]).sort((a, b) => b.roc_auc - a.roc_auc);
        setModels(sorted);
        setLoading(false);
      })
      .catch((err) => {
        setError(`Failed to load models leaderboard: ${err.message}`);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!selectedModelId) return;

    fetch(`/api/v1/metrics/${selectedModelId}/thresholds`)
      .then((res) => {
        if (!res.ok) return { points: [] };
        return res.json();
      })
      .then((data) => {
        setScanPoints(data.points || []);
      })
      .catch(() => {
        setScanPoints([]);
      });
  }, [selectedModelId]);

  const selectedModel = models.find((m) => m.model_id === selectedModelId) || models[0];

  if (loading) {
    return (
      <div className="p-8 text-center text-slate-300 font-mono animate-pulse">
        Loading Official Frozen Leaderboard...
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-rose-950/60 border border-rose-800 text-rose-200 rounded-xl">
        {error}
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto text-slate-100">
      {/* Frozen Official Banner */}
      <div className="p-4 bg-cyan-950/80 border border-cyan-700/80 rounded-xl shadow-lg flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">🧊</span>
            <h2 className="text-lg font-bold uppercase tracking-wider text-cyan-200 font-mono">
              OFFICIALLY FROZEN — 2026-07-26
            </h2>
          </div>
          <p className="text-xs text-cyan-300/90 mt-1 max-w-3xl">
            {selectedModel?.dataset_provenance ||
              "ATLAS open data (record 328, DOI 10.7483/OPENDATA.ATLAS.ZBP2.M5T8) — official ATLAS simulated events, classified by certified pre-trained models."}
          </p>
        </div>
        <div className="text-xs text-right font-mono text-cyan-400 bg-slate-900/80 px-3 py-2 rounded-lg border border-cyan-800/60">
          Leaderboard Status: <strong className="text-white">READ-ONLY CERTIFIED</strong>
        </div>
      </div>

      {/* QML Honesty Callout Box */}
      <div className="p-4 bg-slate-900/90 border border-slate-800 rounded-xl text-xs leading-relaxed text-slate-300">
        <strong className="text-amber-400 font-mono uppercase tracking-wider block mb-1">
          Quantum ML Benchmark Disclaimer
        </strong>
        Quantum ML models (qml_vqc, qml_qaoa) are experimental research benchmarks evaluated on a 100-event budget that scored at chance level on this tabular task — an honest negative result reported deliberately.
      </div>

      {/* Leaderboard Table */}
      <div className="bg-slate-950/90 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
        <div className="px-5 py-4 border-b border-slate-800 flex justify-between items-center">
          <h3 className="text-base font-semibold text-slate-200">
            Certified Model Leaderboard (12 Benchmarked Candidates)
          </h3>
          <span className="text-xs text-slate-400 font-mono">
            Sorted by Test ROC-AUC Descending
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-900/80 text-slate-400 border-b border-slate-800 font-mono uppercase tracking-wider">
                <th className="py-3 px-4">Rank</th>
                <th className="py-3 px-4">Model ID</th>
                <th className="py-3 px-4">Test ROC-AUC</th>
                <th className="py-3 px-4">AMS @ Threshold</th>
                <th className="py-3 px-4">Device</th>
                <th className="py-3 px-4">Training Run Origin</th>
                <th className="py-3 px-4">Subsample Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {models.map((m, idx) => {
                const isChampion = idx === 0;
                const isSelected = m.model_id === selectedModelId;

                return (
                  <tr
                    key={m.model_id}
                    onClick={() => setSelectedModelId(m.model_id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? "bg-cyan-950/60 text-white"
                        : "hover:bg-slate-900/60 text-slate-300"
                    }`}
                  >
                    <td className="py-3 px-4 font-bold">
                      #{idx + 1} {isChampion && <span className="ml-1 text-amber-400">🏆</span>}
                    </td>
                    <td className="py-3 px-4 font-bold text-white">
                      {m.display_name} <span className="text-[10px] text-slate-500 font-normal">({m.model_id})</span>
                    </td>
                    <td className="py-3 px-4 text-cyan-300 font-bold">{m.roc_auc.toFixed(4)}</td>
                    <td className="py-3 px-4 text-emerald-300 font-bold">{m.ams_score.toFixed(4)} @ {m.optimal_threshold.toFixed(4)}</td>
                    <td className="py-3 px-4 text-slate-300">{m.device || "CPU"}</td>
                    <td className="py-3 px-4 text-slate-300">{m.training_run_origin || "R004"}</td>
                    <td className="py-3 px-4 text-slate-400">{m.subsample_notes || "Full 250k"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Threshold / AMS Explorer */}
      {selectedModel && (
        <ThresholdExplorer
          modelId={selectedModel.model_id}
          officialThreshold={selectedModel.optimal_threshold}
          officialAms={selectedModel.ams_score}
          scanPoints={scanPoints}
        />
      )}
    </div>
  );
};

export default LeaderboardView;
