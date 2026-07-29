import React, { useState, useEffect, Suspense, lazy } from 'react';
import { DatasetStatus, ModelInfo } from './types';
import { fetchDatasetStatus, fetchModelRegistry } from './services/api';
import { DatasetCardComponent } from './components/DatasetCard';
import { ModelArenaComponent } from './components/ModelArena';
import { LabComponent } from './components/LabComponent';
import { EducationProvider } from './context/EducationContext';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import { ObservatoryShell } from './observatory/components/shell/ObservatoryShell';
import { ModeType } from './observatory/tokens/theme';
import { Shield, RotateCcw } from 'lucide-react';

const EventDisplay3D = lazy(() => import('./components/display/EventDisplay3D'));
const AcceleratorJourneyView = lazy(() => import('./components/journey/AcceleratorJourneyView'));
const LeaderboardView = lazy(() => import('./components/leaderboard/LeaderboardView'));
const GalleryView = lazy(() => import('./components/gallery/GalleryView').then(m => ({ default: m.GalleryView })));

export const App: React.FC = () => {
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
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
      <ObservatoryShell>
        {(currentMode, setMode) => (
          <>
            {currentMode === 'pipeline' && (
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
                  <div className="flex flex-wrap gap-4 pt-2 text-xs font-mono text-slate-400">
                    <span>DOI: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`</span>
                    <span>&bull;</span>
                    <span>818,238 Full-Detector Reconstruction Events</span>
                    <span>&bull;</span>
                    <span>Approximate Median Significance (b_r = 10)</span>
                  </div>
                </div>
              </div>
            )}

            {currentMode === 'journey' && (
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

            {(currentMode === 'studio' || currentMode === ('detector' as ModeType)) && (
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

            {currentMode === 'leaderboard' && (
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

            {currentMode === 'gallery' && (
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
                        setMode('studio');
                      }}
                    />
                  </Suspense>
                </ErrorBoundary>
              </div>
            )}

            {currentMode === 'arena' && (
              <div className="animate-fadeIn">
                <ModelArenaComponent models={models} onRefresh={loadInitialData} />
              </div>
            )}

            {currentMode === 'lab' && (
              <div className="animate-fadeIn">
                <LabComponent models={models} />
              </div>
            )}
          </>
        )}
      </ObservatoryShell>
    </EducationProvider>
  );
};
