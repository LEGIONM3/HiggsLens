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
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>('perspective');
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
              onClick={() => setCameraPreset('perspective')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                cameraPreset === 'perspective'
                  ? 'bg-cyan-500 text-slate-950 font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              3/4 View
            </button>
            <button
              onClick={() => setCameraPreset('transverse')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                cameraPreset === 'transverse'
                  ? 'bg-cyan-500 text-slate-950 font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Transverse (z=0)
            </button>
            <button
              onClick={() => setCameraPreset('side')}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                cameraPreset === 'side'
                  ? 'bg-cyan-500 text-slate-950 font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Side View
            </button>
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
          <div className="w-full h-[480px] bg-[#090d16] rounded-xl overflow-hidden relative border border-slate-800/80">
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
              <Canvas camera={{ position: [7, 6, 7], fov: 45 }}>
                <ambientLight intensity={0.6} />
                <directionalLight position={[10, 15, 10]} intensity={1.2} />
                <directionalLight position={[-10, -10, -10]} intensity={0.4} />

                <DetectorScene showAxisHelper={showAxisHelper} />
                <EventObjects3D
                  objects={renderableObjects}
                  selectedObjectId={selectedObjectId}
                  onSelectObject={(obj) => setSelectedObjectId(obj.id)}
                />
                <CameraControls preset={cameraPreset} />
              </Canvas>
            )}

            {/* Illustrative Geometry Disclaimer Badge */}
            <div className="absolute bottom-3 left-3 bg-slate-950/85 backdrop-blur px-3 py-1.5 rounded-lg text-[11px] text-slate-300 border border-slate-800 flex items-center gap-1.5 font-mono">
              <Box className="w-3.5 h-3.5 text-cyan-400" />
              Detector-inspired illustrative geometry, not to scale
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
              <span className="w-3 h-3 rounded-full bg-yellow-500 shadow-sm" />
              Leading Jet
            </span>
            <span className="flex items-center gap-1.5 font-medium text-slate-300">
              <span className="w-3 h-3 rounded-full bg-amber-500 shadow-sm" />
              Subleading Jet
            </span>
            <span className="flex items-center gap-1.5 font-medium text-slate-300">
              <span className="w-3 h-3 rounded-full bg-rose-500 shadow-sm" />
              MET (E_T^miss)
            </span>
          </div>
        </div>

        {/* Sidebar: Current Event HUD & Object Telemetry Inspector */}
        <div className="flex flex-col gap-4">
          {/* Current Event HUD & Prediction Gauge */}
          {currentEvent && (
            <div className="glass-panel p-5 flex flex-col gap-4">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                <div>
                  <div className="text-[11px] text-slate-400 font-semibold uppercase tracking-wider">
                    Event Reconstruction
                  </div>
                  <div className="text-lg font-bold font-mono text-white">
                    #{currentEvent.event_id}
                  </div>
                </div>
                <div className="flex flex-col items-end">
                  <span className="text-[10px] text-slate-400">True Label (ATLAS Open Data)</span>
                  <span
                    className={`px-2.5 py-0.5 rounded text-xs font-bold font-mono ${
                      currentEvent.true_label === 'signal'
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-slate-800 text-slate-300 border border-slate-700'
                    }`}
                  >
                    {currentEvent.true_label.toUpperCase()}
                  </span>
                </div>
              </div>

              {/* Certified Model Prediction Gauge */}
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-300 font-medium flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4 text-cyan-400" />
                    Champion Model Signal Prob:
                  </span>
                  <span className="font-mono font-bold text-cyan-400 text-sm">
                    {(currentEvent.prediction.probability * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Progress Bar & Decision Threshold Marker */}
                <div className="relative w-full h-3 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-300"
                    style={{ width: `${currentEvent.prediction.probability * 100}%` }}
                  />
                  {/* Decision Threshold Marker */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-rose-500 z-10"
                    style={{ left: `${currentEvent.prediction.threshold * 100}%` }}
                    title={`Decision Threshold: ${currentEvent.prediction.threshold.toFixed(4)}`}
                  />
                </div>

                <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                  <span>Background (0.0)</span>
                  <span className="text-rose-400">
                    Threshold: {currentEvent.prediction.threshold.toFixed(4)}
                  </span>
                  <span>Signal (1.0)</span>
                </div>
              </div>

              {/* DER Feature Summary */}
              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800/80 text-xs">
                <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/80">
                  <div className="text-[10px] text-slate-400 font-mono">DER_mass_vis</div>
                  <div className="font-mono font-bold text-slate-200">
                    {currentEvent.features['DER_mass_vis']?.toFixed(2) ?? 'N/A'} GeV
                  </div>
                </div>
                <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/80">
                  <div className="text-[10px] text-slate-400 font-mono">DER_mass_MMC</div>
                  <div className="font-mono font-bold text-slate-200">
                    {currentEvent.features['DER_mass_MMC'] === -999.0
                      ? '-999.0 (sentinel)'
                      : `${currentEvent.features['DER_mass_MMC']?.toFixed(2)} GeV`}
                  </div>
                </div>
              </div>
              {currentEvent.features['DER_mass_MMC'] === -999.0 && (
                <div className="text-[10px] text-amber-400 flex items-center gap-1 font-mono bg-amber-950/30 p-2 rounded-lg border border-amber-800/40">
                  <Info className="w-3.5 h-3.5 flex-shrink-0" />
                  <span>Note: -999.0 indicates MMC mass algorithm did not converge for this event.</span>
                </div>
              )}

              {/* Research Report Action Button */}
              <button
                onClick={() => setIsReportModalOpen(true)}
                className="mt-1 w-full py-2.5 bg-sky-900/30 hover:bg-sky-800/40 border border-sky-500/50 hover:border-sky-400 text-sky-200 hover:text-white text-xs font-semibold rounded-xl shadow transition-all flex items-center justify-center gap-2 focus:ring-2 focus:ring-sky-400"
                aria-label={`View Research Report for Event ${currentEvent.event_id}`}
              >
                <FileText className="w-4 h-4 text-sky-400" />
                <span>View Event Analysis Report</span>
              </button>
            </div>
          )}

          {/* Selected Object Kinematics Inspector */}
          <div className="glass-panel p-5 flex flex-col gap-3">
            <h3 className="text-sm font-bold text-white flex items-center gap-2 border-b border-slate-800/80 pb-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              Kinematic Object Inspector
            </h3>

            {selectedObject ? (
              <div className="flex flex-col gap-3 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-sm font-mono" style={{ color: selectedObject.color }}>
                    {selectedObject.name}
                  </span>
                  <span className="badge badge-cyan text-[10px]">Recorded Vector</span>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center bg-slate-950/80 p-2.5 rounded-xl border border-slate-800/80 font-mono">
                  <div>
                    <div className="text-[10px] text-slate-400">pT (GeV)</div>
                    <div className="font-bold text-white">{selectedObject.pt.toFixed(1)}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-400">η (eta)</div>
                    <div className="font-bold text-white">{selectedObject.eta.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-400">φ (phi)</div>
                    <div className="font-bold text-white">{selectedObject.phi.toFixed(2)}</div>
                  </div>
                </div>

                <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80 font-mono text-[11px] text-slate-300">
                  <div>px: {selectedObject.cartesian.px.toFixed(2)} GeV/c</div>
                  <div>py: {selectedObject.cartesian.py.toFixed(2)} GeV/c</div>
                  <div>pz: {selectedObject.cartesian.pz.toFixed(2)} GeV/c</div>
                </div>

                <div className="bg-cyan-950/30 border border-cyan-500/20 p-3 rounded-xl text-cyan-200 text-xs leading-relaxed">
                  <div className="font-bold mb-1 flex items-center gap-1 text-cyan-300">
                    <Info className="w-3.5 h-3.5" /> Physics Context:
                  </div>
                  {selectedObject.tooltip}
                </div>
              </div>
            ) : (
              <div className="text-xs text-slate-400 py-6 text-center italic">
                Click any 3D object in the detector view to inspect its kinematic vector properties.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Research Report Modal */}
      {currentEvent && (
        <ResearchReportModal
          eventId={currentEvent.event_id}
          isOpen={isReportModalOpen}
          onClose={() => setIsReportModalOpen(false)}
        />
      )}

      {/* Physics Education Side Drawer */}
      <EducationMode
        features={currentEvent?.features}
        signalProbability={currentEvent?.prediction.probability}
        threshold={currentEvent?.prediction.threshold}
      />

      {/* Dataset Provenance Footer */}
      <footer className="text-center text-xs text-slate-400 py-3 border-t border-slate-800/80 font-mono">
        ATLAS open data (record 328, DOI 10.7483/OPENDATA.ATLAS.ZBP2.M5T8) — official ATLAS simulated events, classified by certified pre-trained models.
      </footer>
    </div>
  );
};

export default EventDisplay3D;
