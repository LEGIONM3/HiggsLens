import React from 'react';
import { Trophy, Award, CheckCircle2 } from 'lucide-react';

interface LeaderboardItem {
  id: string;
  name: string;
  type: string;
  ams: number;
  rocAuc: number;
  accuracy: number;
  status: string;
}

const CERTIFIED_MODELS: LeaderboardItem[] = [
  { id: 'xgboost_tuned', name: 'XGBoost Tuned Baseline', type: 'Gradient Boosted Trees', ams: 3.642, rocAuc: 0.914, accuracy: 0.842, status: 'Certified Benchmark' },
  { id: 'lightgbm_tuned', name: 'LightGBM Native Missing', type: 'Gradient Boosted Trees', ams: 3.618, rocAuc: 0.912, accuracy: 0.839, status: 'Certified Benchmark' },
  { id: 'nn_mlp_deep', name: 'Deep Residual MLP', type: 'Neural Network', ams: 3.510, rocAuc: 0.905, accuracy: 0.828, status: 'Certified Benchmark' },
  { id: 'random_forest', name: 'Random Forest Baseline', type: 'Ensemble', ams: 3.120, rocAuc: 0.884, accuracy: 0.795, status: 'Reference' },
];

export const LeaderboardView: React.FC = () => {
  return (
    <div className="flex flex-col gap-6 animate-fadeIn">
      <div className="bg-[#090d16] border border-slate-800 rounded-2xl p-6 flex flex-col gap-6 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400">
              <Trophy className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                Official Pretrained Model Leaderboard
                <span className="px-2 py-0.5 rounded bg-amber-950 text-amber-300 text-[10px] font-mono font-semibold">
                  ATLAS Challenge Metric
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Evaluated on Test Partition (v) &bull; Approximate Median Significance (AMS, b_r = 10)
              </p>
            </div>
          </div>
        </div>

        {/* Model Ranking Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px]">
                <th className="py-3 px-4">Rank &amp; Model Identity</th>
                <th className="py-3 px-4">Architecture</th>
                <th className="py-3 px-4">AMS Score</th>
                <th className="py-3 px-4">ROC-AUC</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">Certification</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {CERTIFIED_MODELS.map((m, rank) => (
                <tr key={m.id} className="hover:bg-slate-900/60 transition-colors text-slate-200">
                  <td className="py-3.5 px-4 font-bold flex items-center gap-2 text-white">
                    <span className="w-6 h-6 rounded-full bg-slate-800 flex items-center justify-center text-[11px] text-cyan-400 font-mono">
                      #{rank + 1}
                    </span>
                    {m.name}
                  </td>
                  <td className="py-3.5 px-4 text-slate-400">{m.type}</td>
                  <td className="py-3.5 px-4 text-amber-400 font-bold text-sm">{m.ams.toFixed(3)}</td>
                  <td className="py-3.5 px-4 text-cyan-300">{m.rocAuc.toFixed(3)}</td>
                  <td className="py-3.5 px-4 text-slate-300">{(m.accuracy * 100).toFixed(1)}%</td>
                  <td className="py-3.5 px-4">
                    <span className="px-2 py-1 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[10px] font-semibold flex items-center gap-1 w-fit">
                      <CheckCircle2 className="w-3 h-3" /> {m.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
