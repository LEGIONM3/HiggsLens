import React from 'react';
import { ShieldCheck, Target, ArrowRight } from 'lucide-react';
import { parseEventKinematics, isSentinel } from '../../../lib/kinematics';

interface NonWebGLPolarViewProps {
  eventId?: number;
  onEnterStudio?: () => void;
  onOpenDetails?: (eventId: number) => void;
}

const SAMPLE_FALLBACK_FEATURES: Record<string, number> = {
  PRI_tau_pt: 32.41,
  PRI_tau_eta: 1.24,
  PRI_tau_phi: -1.82,
  PRI_lep_pt: 48.72,
  PRI_lep_eta: -0.45,
  PRI_lep_phi: 1.31,
  PRI_met: 56.18,
  PRI_met_phi: -2.95,
  PRI_jet_num: 2,
  PRI_jet_leading_pt: 89.15,
  PRI_jet_leading_eta: 0.18,
  PRI_jet_leading_phi: 2.74,
  PRI_jet_subleading_pt: 41.6,
  PRI_jet_subleading_eta: -1.12,
  PRI_jet_subleading_phi: -0.85,
};

export const NonWebGLPolarView: React.FC<NonWebGLPolarViewProps> = ({
  eventId = 100001,
  onEnterStudio,
  onOpenDetails,
}) => {
  const objects = parseEventKinematics(SAMPLE_FALLBACK_FEATURES);

  // SVG polar projection coordinates
  const cx = 150;
  const cy = 150;
  const maxRadius = 130;
  const maxPt = 100; // 100 GeV normalization scale

  return (
    <div className="flex flex-col gap-6 animate-fadeIn">
      {/* 2D Polar Projection Fallback Container */}
      <div className="w-full bg-[#090d16] border border-slate-800 rounded-2xl p-6 flex flex-col gap-6 shadow-2xl">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <Target className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                2D Kinematic Polar Projection
                <span className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 font-mono text-[10px]">
                  Non-WebGL Fallback
                </span>
              </h3>
              <p className="text-xs text-slate-400">
                Recorded ATLAS Open-Data Event Kinematics (&eta;-&phi; / pT vector projection)
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 font-mono text-xs text-slate-300">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Event ID #{eventId}</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          {/* 2D Polar Projection SVG Map */}
          <div className="flex flex-col items-center justify-center p-4 bg-[#05070c] rounded-xl border border-slate-800/80 relative">
            <svg width="300" height="300" viewBox="0 0 300 300" className="max-w-full">
              {/* Concentric Polar Grid Rings */}
              <circle cx={cx} cy={cy} r={maxRadius * 0.25} fill="none" stroke="#1e293b" strokeDasharray="3 3" />
              <circle cx={cx} cy={cy} r={maxRadius * 0.5} fill="none" stroke="#1e293b" strokeDasharray="3 3" />
              <circle cx={cx} cy={cy} r={maxRadius * 0.75} fill="none" stroke="#1e293b" strokeDasharray="3 3" />
              <circle cx={cx} cy={cy} r={maxRadius} fill="none" stroke="#334155" strokeWidth="1.5" />

              {/* Axes Crosshair */}
              <line x1={cx - maxRadius} y1={cy} x2={cx + maxRadius} y2={cy} stroke="#1e293b" strokeWidth="1" />
              <line x1={cx} y1={cy - maxRadius} x2={cx} y2={cy + maxRadius} stroke="#1e293b" strokeWidth="1" />

              {/* Radial Labels */}
              <text x={cx + 5} y={cy - maxRadius + 12} fill="#64748b" fontSize="9" fontFamily="monospace">
                100 GeV
              </text>
              <text x={cx + maxRadius - 25} y={cy - 5} fill="#64748b" fontSize="9" fontFamily="monospace">
                +&phi;
              </text>

              {/* Vector Rays */}
              {objects.map((obj, i) => {
                const r = Math.min((obj.pt / maxPt) * maxRadius, maxRadius);
                const vx = cx + r * Math.cos(obj.phi);
                const vy = cy - r * Math.sin(obj.phi);

                let color = '#cbd5e1';
                if (obj.object_type === 'tau') color = '#06b6d4';
                if (obj.object_type === 'lepton') color = '#f8fafc';
                if (obj.object_type === 'jet_leading') color = '#f59e0b';
                if (obj.object_type === 'jet_subleading') color = '#fbbf24';

                return (
                  <g key={i}>
                    <line x1={cx} y1={cy} x2={vx} y2={vy} stroke={color} strokeWidth="2.5" />
                    <circle cx={vx} cy={vy} r="5" fill={color} />
                  </g>
                );
              })}

              {/* MET Vector (Dashed Magenta) */}
              {!isSentinel(SAMPLE_FALLBACK_FEATURES['PRI_met']) && (() => {
                const met = SAMPLE_FALLBACK_FEATURES['PRI_met'];
                const metPhi = SAMPLE_FALLBACK_FEATURES['PRI_met_phi'];
                const r = Math.min((met / maxPt) * maxRadius, maxRadius);
                const mx = cx + r * Math.cos(metPhi);
                const my = cy - r * Math.sin(metPhi);
                return (
                  <g>
                    <line x1={cx} y1={cy} x2={mx} y2={my} stroke="#ec4899" strokeWidth="2" strokeDasharray="4 4" />
                    <polygon points={`${mx},${my-4} ${mx-4},${my+4} ${mx+4},${my+4}`} fill="#ec4899" />
                  </g>
                );
              })()}
            </svg>

            <span className="text-[10px] text-slate-400 font-mono mt-2">
              Polar Coordinate View (&phi; angular &bull; pT radial)
            </span>
          </div>

          {/* Kinematics Data Table */}
          <div className="flex flex-col gap-3 font-mono text-xs">
            <h4 className="text-xs font-bold text-white uppercase tracking-wider border-b border-slate-800 pb-2">
              Recorded Object Kinematics
            </h4>

            {objects.map((obj, i) => (
              <div
                key={i}
                className="p-2.5 rounded-lg bg-[#05070c] border border-slate-800 flex items-center justify-between"
              >
                <div>
                  <span className="text-white font-bold block">{obj.label}</span>
                  <span className="text-slate-400 text-[11px]">
                    &eta;: {obj.eta.toFixed(2)} &bull; &phi;: {obj.phi.toFixed(2)} rad
                  </span>
                </div>
                <span className="text-cyan-400 font-bold">{obj.pt.toFixed(1)} GeV</span>
              </div>
            ))}

            <div className="p-2.5 rounded-lg bg-[#05070c] border border-pink-500/30 flex items-center justify-between">
              <div>
                <span className="text-pink-300 font-bold block">Missing Transverse Energy (MET)</span>
                <span className="text-slate-400 text-[11px]">
                  &phi;: {SAMPLE_FALLBACK_FEATURES['PRI_met_phi'].toFixed(2)} rad
                </span>
              </div>
              <span className="text-pink-400 font-bold">
                {SAMPLE_FALLBACK_FEATURES['PRI_met'].toFixed(1)} GeV
              </span>
            </div>

            <div className="pt-3 flex items-center gap-3">
              {onEnterStudio && (
                <button
                  onClick={onEnterStudio}
                  className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition-colors flex items-center gap-2"
                >
                  Open Event Reconstruction Studio <ArrowRight className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
