import React, { useState, useEffect } from 'react';
import { DatasetStatus, ModelInfo } from './types';
import { fetchDatasetStatus, fetchModelRegistry } from './services/api';
import { ObservatoryShell } from './observatory/components/shell/ObservatoryShell';
import { AcceleratorJourneyView } from './observatory/components/journey/AcceleratorJourneyView';
import { EventStudioView } from './observatory/components/studio/EventStudioView';
import { PipelineView } from './observatory/components/views/PipelineView';
import { LeaderboardView } from './observatory/components/views/LeaderboardView';
import { GalleryView } from './observatory/components/views/GalleryView';
import { ArenaView } from './observatory/components/views/ArenaView';
import { LabView } from './observatory/components/views/LabView';

export const App: React.FC = () => {
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [selectedEventId, setSelectedEventId] = useState<number>(100001);

  // Check window URL location for permalinks (/display/:eventId or /studio/event/:id)
  useEffect(() => {
    const path = window.location.pathname;
    const match = path.match(/\/(display|studio\/event)\/(\d+)/);
    if (match && match[2]) {
      const parsedId = parseInt(match[2], 10);
      if (!isNaN(parsedId)) {
        setSelectedEventId(parsedId);
      }
    }
  }, []);

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
    <ObservatoryShell>
      {(currentMode, setMode) => (
        <>
          {currentMode === 'journey' && (
            <AcceleratorJourneyView onEnterStudio={() => setMode('studio')} />
          )}

          {currentMode === 'studio' && (
            <EventStudioView
              eventId={selectedEventId}
              onOpenDetails={(id) => {
                setSelectedEventId(id);
                window.history.pushState({}, '', `/display/${id}`);
              }}
            />
          )}

          {currentMode === 'leaderboard' && <LeaderboardView />}

          {currentMode === 'pipeline' && (
            <PipelineView status={datasetStatus} onRefresh={loadInitialData} />
          )}

          {currentMode === 'gallery' && (
            <GalleryView
              onSelectEventForDisplay={(id) => {
                setSelectedEventId(id);
                setMode('studio');
                window.history.pushState({}, '', `/display/${id}`);
              }}
            />
          )}

          {currentMode === 'arena' && <ArenaView models={models} onRefresh={loadInitialData} />}

          {currentMode === 'lab' && <LabView models={models} />}
        </>
      )}
    </ObservatoryShell>
  );
};
