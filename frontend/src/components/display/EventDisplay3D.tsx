import React, { useState, useEffect, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import {
  Atom,
  RotateCcw,
  Sliders,
  Sparkles,
  Info,
  Layers,
  ShieldAlert,
  FileText,
  Eye,
  Box,
  Focus,
} from 'lucide-react';

import {
  RenderableObject,
  getRenderablePhysicsObjects,
} from '../../lib/kinematics';
import { DetectorScene } from './DetectorScene';
import { EventObjects3D } from './EventObjects3D';
import { CameraControls, CameraPreset } from './CameraControls';
import { EducationMode } from '../education/EducationMode';
import { ResearchReportModal } from './ResearchReportModal';
import { useEducation } from '../../context/EducationContext';

export interface EventData {
  event_id: number;
  true_label: 'signal' | 'background';
  features: Record<string, number>;
  prediction: {
    model_id: string;
    probability: number;
    predicted_label: string;
    threshold: number;
  };
}

export interface EventSampleApiResponse {
  events: EventData[];
  count: number;
  seed: number;
  label_filter: string;
}

export const EventDisplay3D: React.FC = () => {
  const { toggleDrawer } = useEducation();
  const [events, setEvents] = useState<EventData[]>([]);
  const [selectedEventIndex, setSelectedEventIndex] = useState<number>(0);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [seed, setSeed] = useState<number>(42);
  const [labelFilter, setLabelFilter] = useState<string>('any');
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>('observatory');
  const [showAxisHelper, setShowAxisHelper] = useState<boolean>(false);
  const [isReportModalOpen, setIsReportModalOpen] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSampledEvents = async (s: number, l: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/v1/events/sample?n=12&seed=${s}&label=${l}`);
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP Error ${res.status}`);
      }
      const data: EventSampleApiResponse = await res.json();
      setEvents(data.events);
      setSelectedEventIndex(0);
      setSelectedObjectId(null);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch sampled collision events');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSampledEvents(seed, labelFilter);
  }, []);

  const currentEvent = events[selectedEventIndex] || null;

  // Derive 3D renderable objects for current event (memoized for 60fps performance)
  const renderableObjects = useMemo(() => {
    if (!currentEvent) return [];
    return getRenderablePhysicsObjects(currentEvent.features);
  }, [currentEvent]);

  const selectedObject = useMemo(() => {
    if (!selectedObjectId) return null;
    return renderableObjects.find((o) => o.id === selectedObjectId) || null;
  }, [selectedObjectId, renderableObjects]);

  const selectedObjectPosition = useMemo<[number, number, number] | null>(() => {
    if (!selectedObject) return null;
    return [
      selectedObject.direction[0] * selectedObject.length * 0.7,
      selectedObject.direction[1] * selectedObject.length * 0.7,
      selectedObject.direction[2] * selectedObject.length * 0.7,
    ];
  }, [selectedObject]);

  const handleSampleClick = () => {
    const nextSeed = Math.floor(Math.random() * 10000);
    setSeed(nextSeed);
    fetchSampledEvents(nextSeed, labelFilter);
  };

  const handleFilterChange = (newFilter: string) => {
    setLabelFilter(newFilter);
    fetchSampledEvents(seed, newFilter);
  };

  return (
    <div className="flex flex-col gap-6 w-full animate-fadeIn">
      {/* Scientific Telemetry Control Studio Header */}
      <div className="glass-panel p-5 flex flex-col xl:flex-row xl:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="p-3 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Atom className="w-6 h-6 stroke-[2.5]" />
          </div>
          <div>
            <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2 flex-wrap">
              3D Event Display & Kinematics
              <span className="badge badge-cyan">ATLAS Open Data</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Reconstruction visualization of ATLAS open-data event recorded kinematics inside detector-inspired illustrative geometry (not to scale).
            </p>
          </div>
        </div>

        {/* Toolbar Controls */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Label Filter Form Field */}
          <div className="flex items-center gap-2 bg-slate-900/90 px-3 py-1.5 rounded-xl border border-slate-800 text-xs">
            <Sliders className="w-3.5 h-3.5 text-slate-400" />
            <label htmlFor="event-label-filter" className="text-slate-400 font-medium">Filter:</label>
            <select
              id="event-label-filter"
              name="event-label-filter"
              value={labelFilter}
              onChange={(e) => handleFilterChange(e.target.value)}
              className="bg-slate-950 text-white rounded px-2.5 py-1 border border-slate-800 focus:outline-none focus:border-cyan-500 text-xs font-mono"
            >
              <option value="any">Any Label</option>
              <option value="signal">Signal Only</option>
              <option value="background">Background Only</option>
            </select>
          </div>

          {/* Seed Input Form Field */}
          <div className="flex items-center gap-2 bg-slate-900/90 px-3 py-1.5 rounded-xl border border-slate-800 text-xs">
            <label htmlFor="event-seed-input" className="text-slate-400 font-medium">Seed:</label>
            <input
              id="event-seed-input"
              name="event-seed-input"
              type="number"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value, 10) || 42)}
              className="w-20 bg-slate-950 text-white rounded px-2 py-1 border border-slate-800 focus:outline-none focus:border-cyan-500 text-center text-xs font-mono"
            />
          </div>

          {/* Sample Next Set Button */}
          <button
            onClick={handleSampleClick}
            disabled={loading}
            className="px-3.5 py-1.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs transition-all flex items-center gap-1.5 shadow-md shadow-cyan-600/20 disabled:opacity-50"
          >
            <RotateCcw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Next Random Sample</span>
          </button>

          {/* Education Mode Trigger */}
          <button
            onClick={toggleDrawer}
            className="px-3.5 py-1.5 rounded-xl border border-cyan-500/40 bg-cyan-950/40 hover:bg-cyan-900/60 text-cyan-300 font-semibold text-xs flex items-center gap-1.5 transition-all"
            aria-label="Toggle Physics Education Mode side panel"
          >
            <span>🎓</span>
            <span>Education Mode</span>
          </button>
        </div>
      </div>

      {/* Event Selector Carousel */}
      {events.length > 0 && (
        <div className="flex items-center gap-2.5 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-slate-800">
          {events.map((ev, idx) => {
            const isSelected = idx === selectedEventIndex;
            const probPct = (ev.prediction.probability * 100).toFixed(1);
            return (
              <button
                key={ev.event_id}
                onClick={() => {
                  setSelectedEventIndex(idx);
                  setSelectedObjectId(null);
                }}
                className={`flex-shrink-0 p-3 rounded-xl border text-left transition-all flex flex-col gap-1 min-w-[145px] ${
                  isSelected
                    ? 'bg-slate-900 border-cyan-500 shadow-md shadow-cyan-500/10'
                    : 'bg-slate-950/60 border-slate-800/80 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono text-slate-300 font-semibold">#{ev.event_id}</span>
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                      ev.true_label === 'signal'
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {ev.true_label}
                  </span>
                </div>
                <div className="text-[11px] text-slate-400 flex items-center justify-between mt-1">
                  <span>XGBoost:</span>
                  <span className="font-mono font-bold text-cyan-400">{probPct}%</span>
                </div>
              </button>
            );
          })}
        </div>
      )}

      {/* Main 3D Display Grid Container */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 3D Canvas Telemetry Area (2 Columns) */}
        <div className="lg:col-span-2 glass-panel p-4 flex flex-col gap-4 min-h-[500px] relative">
          {/* Canvas Camera & View Controls Overlay */}
          <div className="absolute top-6 left-6 z-10 flex flex-wrap items-center gap-2 bg-slate-950/90 backdrop-blur px-3 py-1.5 rounded-xl border border-slate-800/80">
            <span className="text-[11px] text-slate-400 font-semibold flex items-center gap-1">
              <Eye className="w-3.5 h-3.5 text-cyan-400" /> Camera:
            </span>
            <button
              onClick={() => setCameraPreset('observatory')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                cameraPreset === 'observatory' || cameraPreset === 'perspective'
                  ? 'bg-cyan-500 text-slate-950 font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Observatory
            </button>
            <button
              onClick={() => setCameraPreset('eventFocus')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                cameraPreset === 'eventFocus'
                  ? 'bg-cyan-500 text-slate-950 font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Event Focus
            </button>
            <button
              onClick={() => setCameraPreset('barrelSlice')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                cameraPreset === 'barrelSlice' || cameraPreset === 'transverse'
                  ? 'bg-cyan-500 text-slate-950 font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Barrel Slice
            </button>
            <button
              onClick={() => setCameraPreset('longitudinal')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                cameraPreset === 'longitudinal' || cameraPreset === 'side'
                  ? 'bg-cyan-500 text-slate-950 font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Longitudinal
            </button>
            {selectedObject && (
              <button
                onClick={() => setCameraPreset('inspection')}
                className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all flex items-center gap-1 ${
                  cameraPreset === 'inspection'
                    ? 'bg-amber-500 text-slate-950 font-bold'
                    : 'text-amber-400 hover:text-amber-300'
                }`}
              >
                <Focus className="w-3 h-3" />
                Inspect Target
              </button>
            )}
            <div className="h-4 w-px bg-slate-800 mx-1" />
            <button
              onClick={() => setShowAxisHelper(!showAxisHelper)}
              className={`px-2 py-1 rounded text-[11px] font-medium border transition-all ${
                showAxisHelper
                  ? 'bg-purple-500/20 text-purple-300 border-purple-500/40'
                  : 'text-slate-400 border-slate-800 hover:text-white'
              }`}
            >
              Axes (XYZ)
            </button>
          </div>

          {/* 3D WebGL Canvas */}
          <div className="w-full h-[480px] bg-[#070b12] rounded-xl overflow-hidden relative border border-slate-800/80">
            {loading ? (
              <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 z-20">
                <div className="flex items-center gap-3 text-cyan-400 text-sm font-medium">
                  <RotateCcw className="w-5 h-5 animate-spin" />
                  Loading ATLAS collision events...
                </div>
              </div>
            ) : error ? (
              <div className="absolute inset-0 flex items-center justify-center bg-slate-950/90 z-20 p-6">
                <div className="flex flex-col items-center gap-2 text-rose-400 text-center max-w-md">
                  <ShieldAlert className="w-8 h-8" />
                  <span className="font-bold text-sm">Error Loading Events</span>
                  <span className="text-xs text-slate-400">{error}</span>
                </div>
              </div>
            ) : (
              <Canvas camera={{ position: [7.5, 5.5, 7.5], fov: 45 }}>
                <ambientLight intensity={0.6} />
                <directionalLight position={[10, 15, 10]} intensity={1.2} />
                <directionalLight position={[-10, -10, -10]} intensity={0.4} />

                <DetectorScene showAxisHelper={showAxisHelper} />
                <EventObjects3D
                  objects={renderableObjects}
                  selectedObjectId={selectedObjectId}
                  onSelectObject={(obj) => {
                    setSelectedObjectId(obj.id);
                    setCameraPreset('inspection');
                  }}
                />
                <CameraControls
                  preset={cameraPreset}
                  selectedObjectPosition={selectedObjectPosition}
                />
              </Canvas>
            )}

            {/* Mandatory Illustrative Geometry Disclaimer Badge */}
            <div className="absolute bottom-3 left-3 bg-slate-950/85 backdrop-blur px-3 py-1.5 rounded-lg text-[11px] text-slate-300 border border-slate-800 flex items-center gap-1.5 font-mono">
              <Box className="w-3.5 h-3.5 text-cyan-400" />
              Detector-inspired illustrative geometry — not to scale
            </div>
          </div>

          {/* Kinematic Object Symbol Legend */}
          <div className="flex flex-wrap items-center justify-center gap-4 text-xs pt-1">
            <span className="flex items-center gap-1.5 font-medium text-slate-300">
              <span className="w-3 h-3 rounded-full bg-orange-500 shadow-sm" />
              Hadronic Tau (τ_had)
            </span>
            <span className="flex items-center gap-1.5 font-medium text-slate-300">
              <span className="w-3 h-3 rounded-full bg-cyan-400 shadow-sm" />
              Lepton (e/μ)
            </span>
            <span className="flex items-center gap-1.5 font-medium text-slate-300">
              <span className="w-3 h-3 rounded-full bg-emerald-500 shadow-sm" />
              Leading Jet (j₁)
            </span>
            <span className="flex items-center gap-1.5 font-medium text-slate-300">
              <span className="w-3 h-3 rounded-full bg-rose-500 shadow-sm" />
              Missing Energy (E_T^miss)
            </span>
          </div>
        </div>

        {/* Event Detail & Kinematic Inspector Panel */}
        <div className="flex flex-col gap-5">
          {currentEvent ? (
            <>
              {/* Event Reconstruction Overview Card */}
              <div className="glass-panel p-5 flex flex-col gap-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div>
                    <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                      Event Reconstruction
                    </div>
                    <div className="text-xl font-bold font-mono text-white">
                      #{currentEvent.event_id}
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-slate-400 block font-mono">True Label (ATLAS Open Data)</span>
                    <span
                      className={`inline-block px-2 py-0.5 rounded text-xs font-bold font-mono ${
                        currentEvent.true_label === 'signal'
                          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                          : 'bg-slate-800 text-slate-400 border border-slate-700'
                      }`}
                    >
                      {currentEvent.true_label.toUpperCase()}
                    </span>
                  </div>
                </div>

                {/* Champion Model Signal Probability Gauge */}
                <div className="flex flex-col gap-1.5 bg-slate-900/80 p-3.5 rounded-xl border border-slate-800">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-300 font-medium flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                      Champion Model Signal Prob:
                    </span>
                    <span className="font-mono font-bold text-cyan-400 text-sm">
                      {(currentEvent.prediction.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800 relative">
                    <div
                      className="bg-gradient-to-r from-cyan-500 to-indigo-500 h-full rounded-full transition-all duration-500"
                      style={{ width: `${currentEvent.prediction.probability * 100}%` }}
                    />
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-rose-500 z-10"
                      style={{ left: `${currentEvent.prediction.threshold * 100}%` }}
                      title={`Decision Threshold: ${currentEvent.prediction.threshold}`}
                    />
                  </div>
                  <div className="flex justify-between items-center text-[10px] font-mono text-slate-500 mt-0.5">
                    <span>Background (0.0)</span>
                    <span className="text-rose-400 font-semibold">Threshold: {currentEvent.prediction.threshold}</span>
                    <span>Signal (1.0)</span>
                  </div>
                </div>

                {/* Visible & MMC Mass Cards */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <div className="text-[10px] text-slate-400 font-mono">DER_mass_vis</div>
                    <div className="text-base font-bold font-mono text-slate-100">
                      {currentEvent.features['DER_mass_vis'] !== undefined && currentEvent.features['DER_mass_vis'] !== -999.0
                        ? `${currentEvent.features['DER_mass_vis'].toFixed(2)} GeV`
                        : 'N/A'}
                    </div>
                  </div>
                  <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                    <div className="text-[10px] text-slate-400 font-mono">DER_mass_MMC</div>
                    <div className="text-base font-bold font-mono text-slate-100">
                      {currentEvent.features['DER_mass_MMC'] === -999.0 ? (
                        <span className="text-amber-400 text-xs font-semibold" title="MMC algorithm did not converge for this event kinematics">
                          -999.0 (No Conv)
                        </span>
                      ) : currentEvent.features['DER_mass_MMC'] !== undefined ? (
                        `${currentEvent.features['DER_mass_MMC'].toFixed(2)} GeV`
                      ) : (
                        'N/A'
                      )}
                    </div>
                  </div>
                </div>

                {/* View Event Analysis Report Modal Trigger */}
                <button
                  onClick={() => setIsReportModalOpen(true)}
                  className="w-full py-2.5 rounded-xl border border-cyan-500/30 bg-cyan-950/20 hover:bg-cyan-900/40 text-cyan-300 font-semibold text-xs transition-all flex items-center justify-center gap-2 shadow-sm"
                >
                  <FileText className="w-4 h-4 text-cyan-400" />
                  <span>View Event Analysis Report</span>
                </button>
              </div>

              {/* Kinematic Inspector Panel */}
              <div className="glass-panel p-5 flex flex-col gap-3">
                <div className="text-xs font-bold text-slate-200 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  Kinematic Object Inspector
                </div>

                {selectedObject ? (
                  <div className="flex flex-col gap-2.5 bg-slate-900/80 p-4 rounded-xl border border-cyan-500/30 animate-fadeIn">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-bold text-white flex items-center gap-2">
                        <span
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: selectedObject.color }}
                        />
                        {selectedObject.name}
                      </span>
                      <button
                        onClick={() => setSelectedObjectId(null)}
                        className="text-[10px] text-slate-400 hover:text-white underline font-mono"
                      >
                        Deselect
                      </button>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs font-mono pt-1">
                      <div className="bg-slate-950 p-2 rounded border border-slate-800">
                        <span className="text-slate-500 block text-[10px]">pT (Transverse Momentum)</span>
                        <span className="text-slate-200 font-bold">{selectedObject.pt.toFixed(2)} GeV</span>
                      </div>
                      <div className="bg-slate-950 p-2 rounded border border-slate-800">
                        <span className="text-slate-500 block text-[10px]">Pseudorapidity (η)</span>
                        <span className="text-slate-200 font-bold">{selectedObject.eta.toFixed(3)}</span>
                      </div>
                      <div className="bg-slate-950 p-2 rounded border border-slate-800">
                        <span className="text-slate-500 block text-[10px]">Azimuthal Angle (φ)</span>
                        <span className="text-slate-200 font-bold">{selectedObject.phi.toFixed(3)} rad</span>
                      </div>
                      <div className="bg-slate-950 p-2 rounded border border-slate-800">
                        <span className="text-slate-500 block text-[10px]">Reconstructed Type</span>
                        <span className="text-cyan-400 font-bold uppercase">{selectedObject.type}</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-4 rounded-xl border border-slate-800/80 bg-slate-950/40 text-xs text-slate-400 text-center">
                    Click any 3D object in the detector view to inspect its kinematic vector properties.
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="glass-panel p-6 text-center text-xs text-slate-400">
              No event selected.
            </div>
          )}
        </div>
      </div>

      {/* Physics Education Side Drawer Modal */}
      <EducationMode />

      {/* Certified Event Research Report Modal */}
      {currentEvent && (
        <ResearchReportModal
          eventId={currentEvent.event_id}
          isOpen={isReportModalOpen}
          onClose={() => setIsReportModalOpen(false)}
        />
      )}
    </div>
  );
};

export default EventDisplay3D;
