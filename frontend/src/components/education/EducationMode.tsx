import React from 'react';
import { EducationLevel, useEducation } from '../../context/EducationContext';

interface EducationModeProps {
  features?: Record<string, number>;
  signalProbability?: number;
  threshold?: number;
  onNavigateToLeaderboard?: () => void;
}

export const EducationMode: React.FC<EducationModeProps> = ({
  features = {},
  signalProbability = 0.5,
  threshold = 0.5,
  onNavigateToLeaderboard,
}) => {
  const { level, setLevel, isOpen, setIsOpen } = useEducation();

  if (!isOpen) return null;

  return (
    <aside
      aria-label="Education mode physics guide side drawer"
      className="fixed right-0 top-16 bottom-0 w-96 bg-slate-900/95 backdrop-blur-md border-l border-slate-800 shadow-2xl z-40 flex flex-col p-5 overflow-y-auto text-slate-200 text-xs"
    >
      {/* Header & Controls */}
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-xl">🎓</span>
          <h2 className="font-bold text-sm text-cyan-300">Physics Education Mode</h2>
        </div>
        <button
          onClick={() => setIsOpen(false)}
          className="text-slate-400 hover:text-white text-lg px-2 rounded"
          aria-label="Close education panel"
        >
          ✕
        </button>
      </div>

      {/* Level Selector Tabs */}
      <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800 mb-5">
        {([1, 2, 3] as EducationLevel[]).map((lvl) => (
          <button
            key={lvl}
            onClick={() => setLevel(lvl)}
            className={`flex-1 py-1.5 text-center font-medium rounded-md transition-all ${
              level === lvl
                ? 'bg-cyan-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Level {lvl}
          </button>
        ))}
      </div>

      {/* LEVEL 1: BEGINNER */}
      {level === 1 && (
        <div className="space-y-4">
          <div className="p-3 bg-cyan-950/40 border border-cyan-800/50 rounded-xl">
            <h3 className="font-semibold text-cyan-300 mb-1 text-xs">🌱 Level 1: What am I looking at?</h3>
            <p className="text-slate-300 text-[11px] leading-relaxed">
              This 3D event display shows subatomic particles produced when high-energy protons collide inside the Large Hadron Collider.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-semibold text-slate-300 text-[11px]">Key Visual Components:</h4>
            <ul className="space-y-2 text-[11px] text-slate-300">
              <li className="flex items-start gap-2 p-2 bg-slate-800/50 rounded-lg">
                <span className="text-amber-400 font-bold">🟡 Tau Jet (Cyan/Gold Cone):</span>
                <span>Decay cone from a tau lepton (tau), key signature of Higgs decay.</span>
              </li>
              <li className="flex items-start gap-2 p-2 bg-slate-800/50 rounded-lg">
                <span className="text-blue-400 font-bold">🔵 Lepton Track (Red/Blue):</span>
                <span>Charged particle track curved by the magnetic field.</span>
              </li>
              <li className="flex items-start gap-2 p-2 bg-slate-800/50 rounded-lg">
                <span className="text-emerald-400 font-bold">🟢 Hadronic Jets (Green Cones):</span>
                <span>Collimated spray of hadrons produced by quarks/gluons.</span>
              </li>
              <li className="flex items-start gap-2 p-2 bg-slate-800/50 rounded-lg">
                <span className="text-purple-400 font-bold">🟣 MET Arrow (Purple Dash):</span>
                <span>Missing Transverse Energy carried away by invisible neutrinos (nu).</span>
              </li>
            </ul>
          </div>

          <div className="p-3 bg-slate-800/40 border border-slate-700/60 rounded-xl text-[10px] text-slate-400 leading-normal">
            <span className="font-semibold text-slate-300 block mb-1">Dataset Provenance:</span>
            ATLAS open data (record 328, DOI 10.7483/OPENDATA.ATLAS.ZBP2.M5T8) — official ATLAS simulated events, classified by certified pre-trained models.
          </div>
        </div>
      )}

      {/* LEVEL 2: INTERMEDIATE */}
      {level === 2 && (
        <div className="space-y-4">
          <div className="p-3 bg-cyan-950/40 border border-cyan-800/50 rounded-xl">
            <h3 className="font-semibold text-cyan-300 mb-1 text-xs">🔬 Level 2: How does the model decide?</h3>
            <p className="text-slate-300 text-[11px] leading-relaxed">
              The certified XGBoost champion model evaluates 30 physical features to classify events into signal (H to tau tau) vs background (Z to tau tau, t anti-t).
            </p>
          </div>

          <div className="p-3 bg-slate-950 border border-slate-800 rounded-xl space-y-2">
            <div className="flex justify-between items-center text-[11px]">
              <span className="text-slate-400">Signal Probability:</span>
              <span className="font-mono font-bold text-cyan-300">{(signalProbability * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center text-[11px]">
              <span className="text-slate-400">Decision Threshold:</span>
              <span className="font-mono text-slate-300">{threshold.toFixed(4)}</span>
            </div>
            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-1">
              <div
                className={`h-full ${signalProbability >= threshold ? 'bg-cyan-500' : 'bg-orange-500'}`}
                style={{ width: `${Math.min(100, signalProbability * 100)}%` }}
              />
            </div>
          </div>

          <div className="p-3 bg-amber-950/40 border border-amber-800/50 rounded-xl text-[10px] text-amber-200/90 leading-relaxed">
            <span className="font-bold text-amber-300 block mb-1">Interpretation Honesty Note:</span>
            Feature attributions describe how the model reached its score. They are not statements of physical causation.
          </div>
        </div>
      )}

      {/* LEVEL 3: ADVANCED */}
      {level === 3 && (
        <div className="space-y-4">
          <div className="p-3 bg-cyan-950/40 border border-cyan-800/50 rounded-xl">
            <h3 className="font-semibold text-cyan-300 mb-1 text-xs">⚛️ Level 3: Show me the physics</h3>
            <p className="text-slate-300 text-[11px] leading-relaxed">
              Full 30-feature vector with explicit physical units and quantum ML research benchmarks.
            </p>
          </div>

          {/* Physical Units Summary Table */}
          <div className="border border-slate-800 rounded-xl overflow-hidden bg-slate-950">
            <div className="p-2 bg-slate-900 font-semibold text-cyan-400 text-[11px] border-b border-slate-800">
              Physics Units Standard
            </div>
            <div className="p-2 space-y-1 text-[10px] font-mono text-slate-300">
              <div className="flex justify-between border-b border-slate-900 py-0.5">
                <span>pT, m, met</span>
                <span className="text-cyan-300">GeV (Gigaelectronvolts)</span>
              </div>
              <div className="flex justify-between border-b border-slate-900 py-0.5">
                <span>phi, phi_met</span>
                <span className="text-cyan-300">rad (Radians, [-pi, pi])</span>
              </div>
              <div className="flex justify-between border-b border-slate-900 py-0.5">
                <span>eta, delta_R</span>
                <span className="text-cyan-300">dimensionless (Pseudorapidity)</span>
              </div>
              <div className="flex justify-between py-0.5">
                <span>Sentinels</span>
                <span className="text-amber-400">-999.0 (Undefined/Missing)</span>
              </div>
            </div>
          </div>

          <div className="p-3 bg-slate-950 border border-slate-800 rounded-xl text-[10px] text-slate-300 leading-relaxed">
            <span className="font-bold text-purple-300 block mb-1">Quantum ML Research Callout:</span>
            Quantum ML models (qml_vqc, qml_qaoa) are experimental research benchmarks evaluated on a 100-event budget that scored at chance level on this tabular task — an honest negative result reported deliberately.
          </div>

          {onNavigateToLeaderboard && (
            <button
              onClick={onNavigateToLeaderboard}
              className="w-full py-2 bg-purple-900/60 hover:bg-purple-800/80 text-purple-200 border border-purple-700/60 rounded-xl font-medium text-xs transition-colors flex items-center justify-center gap-2"
            >
              <span>🏆</span>
              <span>View Certified 12-Model Leaderboard</span>
            </button>
          )}
        </div>
      )}
    </aside>
  );
};
