import React from 'react';
import { Target, ShieldCheck, Play, Pause, RotateCcw, ZoomIn, ZoomOut, Zap, Gauge } from 'lucide-react';
import { ReconstructedObject } from '../../../types';
import { etaPhiPtToCartesian, isSentinel } from '../../../lib/kinematics';
import { formatPhysicsValue } from '../../../lib/sentinel';
import { SpeedMode, SpeedMultiplier, SelectedObjectPayload, IllustrativeParticle } from '../DetectorScene3D';

interface NonWebGLPolarViewProps {
  eventId?: number;
  realObjects?: ReconstructedObject[];
  metData?: { magnitude: number; phi: number } | null;
  isPlaying?: boolean;
  speedMode?: SpeedMode;
  speedMultiplier?: SpeedMultiplier;
  zoomLevel?: number;
  onTogglePlay?: () => void;
  onRestart?: () => void;
  onZoomIn?: () => void;
  onZoomOut?: () => void;
  onChangeSpeedMode?: (mode: SpeedMode) => void;
  onChangeSpeedMultiplier?: (mult: SpeedMultiplier) => void;
  onSelectObject?: (payload: SelectedObjectPayload) => void;
  onEnterStudio?: () => void;
  onOpenDetails?: (eventId: number) => void;
}

export const NonWebGLPolarView: React.FC<NonWebGLPolarViewProps> = ({
  eventId = 100001,
  realObjects = [],
  metData,
  isPlaying = true,
  speedMode = 'VIEWABLE',
  speedMultiplier = 1.0,
  zoomLevel = 22,
  onTogglePlay,
  onRestart,
  onZoomIn,
  onZoomOut,
  onChangeSpeedMode,
  onChangeSpeedMultiplier,
  onSelectObject,
  onEnterStudio,
  onOpenDetails,
}) => {
  const cx = 160;
  const cy = 160;
  // Radius scaled by zoomLevel
  const baseRadius = 140;
  const zoomFactor = 22 / Math.max(5, Math.min(50, zoomLevel));
  const maxRadius = Math.min(155, Math.max(70, baseRadius * zoomFactor));

  const getObjectColor = (type: string) => {
    switch (type) {
      case 'tau':
        return '#06b6d4';
      case 'lepton':
        return '#f8fafc';
      case 'jet_leading':
        return '#f59e0b';
      case 'jet_subleading':
        return '#fbbf24';
      default:
        return '#cbd5e1';
    }
  };

  return (
    <div className="w-full flex flex-col gap-4 animate-fadeIn">
      <div className="flex flex-wrap items-center justify-between gap-3 bg-[#090d16] border border-slate-800 p-3 rounded-xl">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-mono">
            Non-WebGL 2D Fallback
          </div>
          <span className="text-xs text-slate-300 font-mono">2D Polar Projection View (&phi; angular &bull; pT radial)</span>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
          <span>Zoom Scale: {zoomFactor.toFixed(2)}x</span>
        </div>
      </div>

      <div className="relative w-full h-[480px] bg-[#05070c] rounded-xl border border-slate-800 flex items-center justify-center p-4">
        <svg width="340" height="340" viewBox="0 0 320 320" className="max-w-full">
          {/* Detector Concentric Rings */}
          <circle cx={cx} cy={cy} r={maxRadius * 0.25} fill="none" stroke="#06b6d4" strokeWidth="1" strokeDasharray="3 3" opacity={0.4} />
          <circle cx={cx} cy={cy} r={maxRadius * 0.5} fill="none" stroke="#38bdf8" strokeWidth="1" strokeDasharray="3 3" opacity={0.3} />
          <circle cx={cx} cy={cy} r={maxRadius * 0.75} fill="none" stroke="#f59e0b" strokeWidth="1" strokeDasharray="3 3" opacity={0.2} />
          <circle cx={cx} cy={cy} r={maxRadius} fill="none" stroke="#64748b" strokeWidth="1.5" opacity={0.5} />

          {/* Axes Crosshair */}
          <line x1={cx - maxRadius} y1={cy} x2={cx + maxRadius} y2={cy} stroke="#1e293b" strokeWidth="1" />
          <line x1={cx} y1={cy - maxRadius} x2={cx} y2={cy + maxRadius} stroke="#1e293b" strokeWidth="1" />

          {/* Center Beam Tube & Interaction Core */}
          <circle cx={cx} cy={cy} r="6" fill="#38bdf8" opacity={0.8} />

          {/* Real Event Vectors */}
          {realObjects.map((obj, i) => {
            const r = Math.min((obj.pt / 100) * maxRadius, maxRadius);
            const vx = cx + r * Math.cos(obj.phi);
            const vy = cy - r * Math.sin(obj.phi);
            const color = getObjectColor(obj.object_type);
            const cart = etaPhiPtToCartesian(obj.pt, obj.eta, obj.phi);

            return (
              <g
                key={i}
                className="cursor-pointer hover:opacity-80 transition-opacity"
                onClick={() =>
                  onSelectObject?.({
                    isApiBacked: true,
                    realObject: obj,
                    cartesianPxPyPz: { px: cart.x, py: cart.y, pz: cart.z },
                  })
                }
              >
                <line x1={cx} y1={cy} x2={vx} y2={vy} stroke={color} strokeWidth="3" />
                <circle cx={vx} cy={vy} r="6" fill={color} />
                <text x={vx + 8} y={vy + 4} fill={color} fontSize="10" fontFamily="monospace" fontWeight="bold">
                  {obj.label}
                </text>
              </g>
            );
          })}

          {/* MET Vector (Dashed Magenta) */}
          {metData && metData.magnitude > 0 && (() => {
            const r = Math.min((metData.magnitude / 100) * maxRadius, maxRadius);
            const mx = cx + r * Math.cos(metData.phi);
            const my = cy - r * Math.sin(metData.phi);
            return (
              <g
                className="cursor-pointer hover:opacity-80"
                onClick={() =>
                  onSelectObject?.({
                    isApiBacked: true,
                    realObject: {
                      object_type: 'met',
                      label: 'MET / missing transverse energy',
                      pt: metData.magnitude,
                      eta: 0,
                      phi: metData.phi,
                      present: true,
                    },
                    cartesianPxPyPz: {
                      px: metData.magnitude * Math.cos(metData.phi),
                      py: metData.magnitude * Math.sin(metData.phi),
                      pz: 0,
                    },
                  })
                }
              >
                <line x1={cx} y1={cy} x2={mx} y2={my} stroke="#ec4899" strokeWidth="2.5" strokeDasharray="4 4" />
                <circle cx={mx} cy={my} r="5" fill="#ec4899" />
                <text x={mx + 8} y={my + 4} fill="#ec4899" fontSize="10" fontFamily="monospace" fontWeight="bold">
                  MET ({metData.magnitude.toFixed(1)} GeV)
                </text>
              </g>
            );
          })()}
        </svg>

        <div className="absolute bottom-3 left-3 text-[10px] font-mono text-slate-400">
          Click any 2D vector ray to view detailed kinematics namecard.
        </div>
      </div>
    </div>
  );
};
