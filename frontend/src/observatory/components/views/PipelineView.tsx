import React from 'react';
import { Database, ShieldCheck, RefreshCw, FileText, Lock } from 'lucide-react';
import { DatasetStatus } from '../../../types';

interface PipelineViewProps {
  status: DatasetStatus | null;
  onRefresh?: () => void;
}

export const PipelineView: React.FC<PipelineViewProps> = ({ status, onRefresh }) => {
  return (
    <div className="flex flex-col gap-6 animate-fadeIn">
      <div className="glass-panel p-6 border-l-4 border-l-cyan-500 flex flex-col gap-4 bg-[#090d16] border border-slate-800 rounded-2xl shadow-2xl">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <Database className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                CERN Open Data Pipeline
                <span className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 text-[10px] font-mono font-semibold">
                  RECORD 328
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                ATLAS H &rarr; &tau;&tau; Machine Learning Challenge 2014 &bull; DOI: 10.7483/OPENDATA.ATLAS.ZBP2.M5T8
              </p>
            </div>
          </div>

          {onRefresh && (
            <button
              onClick={onRefresh}
              className="px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 font-semibold text-xs transition-colors flex items-center gap-1.5"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Refresh Pipeline Status
            </button>
          )}
        </div>

        {/* Partition Details & Cache Status */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl bg-[#05070c] border border-slate-800 flex flex-col gap-1">
            <span className="text-slate-400 text-[10px] font-mono uppercase">DATASET CACHE STATUS</span>
            <span className="text-emerald-400 font-bold text-sm flex items-center gap-1">
              <ShieldCheck className="w-4 h-4" /> Cached Locally
            </span>
            <span className="text-slate-400 text-xs font-mono">Size: 177.86 MiB (818,238 Events)</span>
          </div>

          <div className="p-4 rounded-xl bg-[#05070c] border border-slate-800 flex flex-col gap-1">
            <span className="text-slate-400 text-[10px] font-mono uppercase">PARTITION MAP (KAGGLESET)</span>
            <span className="text-cyan-300 font-bold text-sm font-mono">t: Train | b: Val | v: Test | u: Holdout</span>
            <span className="text-slate-400 text-xs">Strict zero data leakage isolation</span>
          </div>

          <div className="p-4 rounded-xl bg-[#05070c] border border-slate-800 flex flex-col gap-1">
            <span className="text-slate-400 text-[10px] font-mono uppercase">SENTINEL IMPUTATION RULE</span>
            <span className="text-amber-400 font-bold text-sm font-mono">Preserved (-999.0)</span>
            <span className="text-slate-400 text-xs">Preserves jet multiplicity missingness</span>
          </div>
        </div>

        {/* Scientific Integrity Briefing */}
        <div className="p-5 rounded-xl bg-[#05070c] border border-cyan-500/20 flex flex-col gap-2">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Lock className="w-4 h-4 text-cyan-400" /> Scientific Integrity &amp; Partitioning Rules
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            The HiggsLens pipeline enforces zero data leakage across splits. Feature standard scalers and median imputers are fitted strictly on the training partition (`t`). Sentinel missing values (`-999.0`) for jet properties under low multiplicity (`PRI_jet_num` &lt; 2) are passed directly to models that natively support missing values (LightGBM, XGBoost) or imputed using binary indicators for neural baseline models.
          </p>
        </div>
      </div>
    </div>
  );
};
