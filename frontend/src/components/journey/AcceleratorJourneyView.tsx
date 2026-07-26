import React, { useState, useEffect, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { AcceleratorRing } from './AcceleratorRing';
import { BeamControls } from './BeamControls';
import { EventEditor } from './EventEditor';
import { EventDisplay3D } from '../display/EventDisplay3D';
import { JourneyStateMachine, JourneyState } from '../../lib/journeyState';

export interface EventData {
  event_id: number;
  label: string;
  weight: number;
  features: Record<string, number>;
  prediction: {
    model_id: string;
    probability: number;
    predicted_label: string;
    threshold: number;
  };
}

export const AcceleratorJourneyView: React.FC = () => {
  const [protonsPerBunch, setProtonsPerBunch] = useState<number>(1.15e11);
  const [bunchCount, setBunchCount] = useState<number>(2808);

  const [stateMachine] = useState(() => new JourneyStateMachine());
  const [journeyState, setJourneyState] = useState<JourneyState>('idle');

  const [currentEvent, setCurrentEvent] = useState<EventData | null>(null);
  const [editedFeatures, setEditedFeatures] = useState<Record<string, number> | null>(null);
  const [selectedLandmark, setSelectedLandmark] = useState<{ name: string; desc: string } | null>(null);

  // Fetch real event from GET /api/v1/events/sample?n=1
  const fetchSampleEvent = useCallback(async () => {
    try {
      const seed = Math.floor(Math.random() * 10000);
      const resp = await fetch(`/api/v1/events/sample?n=1&seed=${seed}`);
      if (resp.ok) {
        const data = await resp.json();
        if (data.events && data.events.length > 0) {
          const sampled = data.events[0];
          setCurrentEvent(sampled);
          setEditedFeatures(sampled.features);
        }
      }
    } catch (err) {
      console.error('Failed to sample collision event:', err);
    }
  }, []);

  // Auto Run Orchestration sequence
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;

    if (journeyState === 'injecting') {
      timer = setTimeout(() => {
        setJourneyState(stateMachine.transition('INJECTION_COMPLETE'));
      }, 1000);
    } else if (journeyState === 'accelerating') {
      timer = setTimeout(() => {
        setJourneyState(stateMachine.transition('ACCELERATION_COMPLETE'));
      }, 2000);
    } else if (journeyState === 'colliding') {
      fetchSampleEvent();
      timer = setTimeout(() => {
        setJourneyState(stateMachine.transition('COLLISION_COMPLETE'));
      }, 1000);
    } else if (journeyState === 'zooming') {
      timer = setTimeout(() => {
        setJourneyState(stateMachine.transition('ZOOM_COMPLETE'));
      }, 1000);
    }

    return () => clearTimeout(timer);
  }, [journeyState, stateMachine, fetchSampleEvent]);

  const handleStartAutoRun = () => {
    if (stateMachine.canTransition('START_AUTO_RUN')) {
      setJourneyState(stateMachine.transition('START_AUTO_RUN'));
    }
  };

  const handleTriggerCollision = () => {
    fetchSampleEvent();
    if (stateMachine.canTransition('START_AUTO_RUN')) {
      stateMachine.transition('START_AUTO_RUN');
      stateMachine.transition('INJECTION_COMPLETE');
      stateMachine.transition('ACCELERATION_COMPLETE');
      stateMachine.transition('COLLISION_COMPLETE');
      setJourneyState(stateMachine.transition('ZOOM_COMPLETE'));
    } else {
      setJourneyState('displaying');
    }
  };

  const handleReset = () => {
    stateMachine.reset();
    setJourneyState('idle');
    setCurrentEvent(null);
    setEditedFeatures(null);
  };

  // Callback when EventEditor re-derives features
  const handleDeriveComplete = (
    fullFeatures: Record<string, number>,
    prediction: {
      model_id: string;
      probability: number;
      predicted_label: string;
      threshold: number;
    }
  ) => {
    setEditedFeatures(fullFeatures);
    if (currentEvent) {
      setCurrentEvent({
        ...currentEvent,
        features: fullFeatures,
        prediction,
      });
    }
  };

  return (
    <div className="space-y-6 text-slate-100">
      {/* View Header Briefing Card */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 backdrop-blur-md shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-cyan-400 flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
            Accelerator Journey &amp; Event Editor
          </h2>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Interactive stylized LHC accelerator visualization. Set proton beam parameters, trigger counter-rotating beam collisions at &radic;s = 8 TeV, inspect real ATLAS open data events, and re-derive features in the Event Editor.
          </p>
        </div>
        <div className="text-right">
          <span className="text-[11px] font-mono px-3 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
            Current Phase: {journeyState.toUpperCase()}
          </span>
        </div>
      </div>

      {/* 3D LHC Accelerator Ring Canvas */}
      <div className="relative w-full h-[400px] bg-slate-950 rounded-xl border border-slate-800 overflow-hidden shadow-2xl">
        <Canvas camera={{ position: [0, 8, 12], fov: 50 }}>
          <color attach="background" args={['#020617']} />
          <ambientLight intensity={0.6} />
          <directionalLight position={[10, 15, 10]} intensity={1.2} />
          <pointLight position={[6, 0, 0]} intensity={2.0} color="#06B6D4" />

          <AcceleratorRing
            bunchCount={bunchCount}
            protonsPerBunch={protonsPerBunch}
            journeyState={journeyState}
            onSelectLandmark={(name, desc) => setSelectedLandmark({ name, desc })}
          />

          <OrbitControls enablePan={true} maxPolarAngle={Math.PI / 2 - 0.05} />
        </Canvas>

        {/* Floating Canvas Badges */}
        <div className="absolute top-3 left-3 bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 text-xs text-slate-300 font-mono">
          Stylized illustrative accelerator geometry &mdash; not to scale
        </div>

        {/* Landmark Modal Tooltip Overlay */}
        {selectedLandmark && (
          <div className="absolute bottom-3 left-3 right-3 md:left-auto md:right-3 md:max-w-md bg-slate-900/95 border border-cyan-800/80 rounded-xl p-4 shadow-2xl backdrop-blur-md text-xs">
            <div className="flex justify-between items-center mb-1">
              <h4 className="font-semibold text-cyan-300 text-sm">{selectedLandmark.name}</h4>
              <button
                onClick={() => setSelectedLandmark(null)}
                className="text-slate-400 hover:text-white text-base font-mono"
              >
                &times;
              </button>
            </div>
            <p className="text-slate-300 leading-relaxed">{selectedLandmark.desc}</p>
          </div>
        )}
      </div>

      {/* Beam Steering Control Panel */}
      <BeamControls
        protonsPerBunch={protonsPerBunch}
        setProtonsPerBunch={setProtonsPerBunch}
        bunchCount={bunchCount}
        setBunchCount={setBunchCount}
        journeyState={journeyState}
        onStartAutoRun={handleStartAutoRun}
        onTriggerCollision={handleTriggerCollision}
        onReset={handleReset}
      />

      {/* Event Display Handoff (Reused F1 3D Scene Components) */}
      {currentEvent && (
        <div className="space-y-6 pt-4 border-t border-slate-800">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-cyan-400">
              Reconstructed Collision Event #{currentEvent.event_id}
            </h3>
            <span className="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono">
              Label: {currentEvent.label.toUpperCase()}
            </span>
          </div>

          <EventDisplay3D />

          {/* Advanced PRI->DER Event Editor */}
          <EventEditor
            initialPriFeatures={editedFeatures || currentEvent.features}
            baseEventId={currentEvent.event_id}
            onDeriveComplete={handleDeriveComplete}
          />
        </div>
      )}
    </div>
  );
};

export default AcceleratorJourneyView;
