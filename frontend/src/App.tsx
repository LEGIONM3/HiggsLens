import React, { useState, useEffect, Suspense, lazy } from 'react';
import { DatasetStatus, ModelInfo } from './types';
import { fetchDatasetStatus, fetchModelRegistry } from './services/api';
import { DatasetCardComponent } from './components/DatasetCard';
import { ModelArenaComponent } from './components/ModelArena';
import { LabComponent } from './components/LabComponent';
import { EducationProvider } from './context/EducationContext';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import { Atom, Terminal, Shield, Sparkles, ExternalLink, RotateCcw } from 'lucide-react';

const EventDisplay3D = lazy(() => import('./components/display/EventDisplay3D'));
const AcceleratorJourneyView = lazy(() => import('./components/journey/AcceleratorJourneyView'));
const LeaderboardView = lazy(() => import('./components/leaderboard/LeaderboardView'));
const GalleryView = lazy(() => import('./components/gallery/GalleryView').then(m => ({ default: m.GalleryView })));

export const App: React.FC = () => {
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [activeTab, setActiveTab] = useState<'pipeline' | 'journey' | 'leaderboard' | 'gallery' | 'arena' | 'detector' | 'lab'>('pipeline');
  const [selectedEventId, setSelectedEventId] = useState<number | null>(null);

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
    <EducationProvider>
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
                ATLAS $H \to \tau\tau$ Machine Learning Challenge 2014 &amp; Reproducible Arena
              </p>
            </div>
          </div>

          <nav className="flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveTab('pipeline')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'pipeline'
                  ? 'bg-gradient-to-r from-cyan-500 to-cyan-600 text-slate-950 shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              1. Data Pipeline
            </button>
            <button
              onClick={() => setActiveTab('journey')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'journey'
                  ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              2. Accelerator Journey
            </button>
            <button
              onClick={() => setActiveTab('leaderboard')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'leaderboard'
                  ? 'bg-gradient-to-r from-cyan-400 to-blue-500 text-slate-950 shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              3. Model Leaderboard
            </button>
            <button
              onClick={() => setActiveTab('gallery')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'gallery'
                  ? 'bg-gradient-to-r from-amber-500 to-amber-600 text-slate-950 shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              4. Event Gallery
            </button>
            <button
              onClick={() => setActiveTab('arena')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'arena'
                  ? 'bg-gradient-to-r from-purple-500 to-purple-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              5. Model Arena
            </button>
            <button
              onClick={() => setActiveTab('detector')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'detector'
                  ? 'bg-gradient-to-r from-emerald-500 to-emerald-600 text-slate-950 shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              6. 3D Event Display
            </button>
            <button
              onClick={() => setActiveTab('lab')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'lab'
                  ? 'bg-gradient-to-r from-purple-500 to-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              7. Lab (Experimental)
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
                Scientific Integrity &amp; Preprocessing Rules
              </h3>
              <p className="text-sm text-slate-300 leading-relaxed">
                The HiggsLens pipeline guarantees zero data leakage across partitions (`KaggleSet` mapping: `t` for training, `b` for validation &amp; threshold selection, `v` for test evaluation). Imputers and standard scalers are fitted exclusively on training data (`t`). Sentinel values (`-999.0`) corresponding to leading/subleading jet measurements under jet multiplicity (`PRI_jet_num`) are preserved or imputed with binary missingness indicators based on candidate architecture requirements.
              </p>
              <div className="flex flex-wrap gap-4 pt-2 text-xs mono text-slate-400">
                <span>DOI: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`</span>
                <span>&bull;</span>
                <span>818,238 Full-Detector Reconstruction Events</span>
                <span>&bull;</span>
                <span>Approximate Median Significance ($b_r = 10$)</span>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'journey' && (
          <div className="animate-fadeIn">
            <Suspense
              fallback={
                <div className="glass-panel p-12 flex flex-col items-center justify-center gap-3 text-cyan-400">
                  <RotateCcw className="w-8 h-8 animate-spin" />
                  <span className="text-sm font-medium">Loading Accelerator Journey...</span>
                </div>
              }
            >
              <AcceleratorJourneyView />
            </Suspense>
          </div>
        )}

        {activeTab === 'leaderboard' && (
          <div className="animate-fadeIn">
            <ErrorBoundary fallbackTitle="Leaderboard View Error">
              <Suspense
                fallback={
                  <div className="glass-panel p-12 flex flex-col items-center justify-center gap-3 text-cyan-400">
                    <RotateCcw className="w-8 h-8 animate-spin" />
                    <span className="text-sm font-medium">Loading Official Leaderboard...</span>
                  </div>
                }
              >
                <LeaderboardView />
              </Suspense>
            </ErrorBoundary>
          </div>
        )}

        {activeTab === 'gallery' && (
          <div className="animate-fadeIn">
            <ErrorBoundary fallbackTitle="Event Gallery Error">
              <Suspense
                fallback={
                  <div className="glass-panel p-12 flex flex-col items-center justify-center gap-3 text-amber-400">
                    <RotateCcw className="w-8 h-8 animate-spin" />
                    <span className="text-sm font-medium">Loading Curated Event Gallery...</span>
                  </div>
                }
              >
                <GalleryView
                  onSelectEventForDisplay={(eventId) => {
                    setSelectedEventId(eventId);
                    setActiveTab('detector');
                  }}
                />
              </Suspense>
            </ErrorBoundary>
          </div>
        )}

        {activeTab === 'arena' && (
          <div className="animate-fadeIn">
            <ModelArenaComponent models={models} onRefresh={loadInitialData} />
          </div>
        )}

        {activeTab === 'detector' && (
          <div className="animate-fadeIn">
            <Suspense
              fallback={
                <div className="glass-panel p-12 flex flex-col items-center justify-center gap-3 text-cyan-400">
                  <RotateCcw className="w-8 h-8 animate-spin" />
                  <span className="text-sm font-medium">Loading 3D Event Display...</span>
                </div>
              }
            >
              <EventDisplay3D />
            </Suspense>
          </div>
        )}

        {activeTab === 'lab' && (
          <div className="animate-fadeIn">
            <LabComponent models={models} />
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
  </EducationProvider>
);
};
