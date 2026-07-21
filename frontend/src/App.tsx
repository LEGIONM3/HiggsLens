import React, { useState, useEffect } from 'react';
import { DatasetStatus, ModelInfo } from './types';
import { fetchDatasetStatus, fetchModelRegistry } from './services/api';
import { DatasetCardComponent } from './components/DatasetCard';
import { ModelArenaComponent } from './components/ModelArena';
import { DetectorViewComponent } from './components/DetectorView';
import { Atom, Terminal, Shield, Sparkles, ExternalLink } from 'lucide-react';

export const App: React.FC = () => {
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [activeTab, setActiveTab] = useState<'pipeline' | 'arena' | 'detector'>('pipeline');

  const loadInitialData = async () => {
    try {
      const [ds, mods] = await Promise.all([
        fetchDatasetStatus().catch(() => null),
        fetchModelRegistry().catch(() => ({})),
      ]);
      if (ds) setDatasetStatus(ds);
      if (mods) setModels(mods);
    } catch (err) {
      console.error('Error loading initial data:', err);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  return (
    <div className="min-h-screen flex flex-col justify-between">
      {/* Header */}
      <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-br from-cyan-500 to-purple-600 text-slate-950 shadow-lg shadow-cyan-500/20">
              <Atom className="w-6 h-6 stroke-[2.5]" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                HiggsLens
                <span className="badge badge-purple text-[10px]">CERN Open Data 328</span>
              </h1>
              <p className="text-xs text-slate-400 font-medium">
                ATLAS $H \to \tau\tau$ Machine Learning Challenge 2014 & Reproducible Arena
              </p>
            </div>
          </div>

          <nav className="flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveTab('pipeline')}
              className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'pipeline'
                  ? 'bg-gradient-to-r from-cyan-500 to-cyan-600 text-slate-950 shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              1. Data Pipeline
            </button>
            <button
              onClick={() => setActiveTab('arena')}
              className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'arena'
                  ? 'bg-gradient-to-r from-purple-500 to-purple-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              2. Model Arena
            </button>
            <button
              onClick={() => setActiveTab('detector')}
              className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'detector'
                  ? 'bg-gradient-to-r from-emerald-500 to-emerald-600 text-slate-950 shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              3. Detector View
            </button>
          </nav>

          <div className="hidden lg:flex items-center gap-4 text-xs text-slate-400 mono">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              API: Active (Port 8000)
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full flex flex-col gap-6">
        {activeTab === 'pipeline' && (
          <div className="flex flex-col gap-6 animate-fadeIn">
            <DatasetCardComponent status={datasetStatus} onRefresh={loadInitialData} />
            
            {/* Scientific Briefing Card */}
            <div className="glass-panel p-6 border-l-4 border-l-cyan-500 flex flex-col gap-3">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Shield className="w-5 h-5 text-cyan-400" />
                Scientific Integrity & Preprocessing Rules
              </h3>
              <p className="text-sm text-slate-300 leading-relaxed">
                The HiggsLens pipeline guarantees zero data leakage across partitions (`KaggleSet` mapping: `t` for training, `b` for validation & threshold selection, `v` for test evaluation). Imputers and standard scalers are fitted exclusively on training data (`t`). Sentinel values (`-999.0`) corresponding to leading/subleading jet measurements under jet multiplicity (`PRI_jet_num`) are preserved or imputed with binary missingness indicators based on candidate architecture requirements.
              </p>
              <div className="flex flex-wrap gap-4 pt-2 text-xs mono text-slate-400">
                <span>DOI: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`</span>
                <span>•</span>
                <span>818,238 Full-Detector Simulated Events</span>
                <span>•</span>
                <span>Approximate Median Significance ($b_r = 10$)</span>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'arena' && (
          <div className="animate-fadeIn">
            <ModelArenaComponent models={models} onRefresh={loadInitialData} />
          </div>
        )}

        {activeTab === 'detector' && (
          <div className="animate-fadeIn">
            <DetectorViewComponent models={models} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/60 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-500" />
            <span>HiggsLens · Educational & Machine Learning Research Platform</span>
          </div>
          <div className="flex items-center gap-6">
            <a
              href="https://opendata.cern.ch/record/328"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-cyan-400 transition-colors flex items-center gap-1"
            >
              CERN Record 328 <ExternalLink className="w-3 h-3" />
            </a>
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-cyan-400 transition-colors flex items-center gap-1"
            >
              OpenAPI Docs <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};
