import React from 'react';

export interface BeamControlsProps {
  protonsPerBunch: number;
  setProtonsPerBunch: (val: number) => void;
  bunchCount: number;
  setBunchCount: (val: number) => void;
  journeyState: string;
  onStartAutoRun: () => void;
  onTriggerCollision: () => void;
  onReset: () => void;
}

export const BeamControls: React.FC<BeamControlsProps> = ({
  protonsPerBunch,
  setProtonsPerBunch,
  bunchCount,
  setBunchCount,
  journeyState,
  onStartAutoRun,
  onTriggerCollision,
  onReset,
}) => {
  // Relative intensity formula: bunches/2808 * (protons / 1.15e11)^2
  const nominalProtons = 1.15e11;
  const nominalBunches = 2808;
  const relativeIntensity =
    (bunchCount / nominalBunches) * Math.pow(protonsPerBunch / nominalProtons, 2);

  const isBusy = journeyState !== 'idle' && journeyState !== 'displaying';

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 backdrop-blur-md shadow-xl text-slate-100 space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 className="text-base font-semibold text-cyan-400 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
          LHC Beam Steering & Controls
        </h3>
        <span className="text-xs px-2.5 py-1 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono">
          4 TeV + 4 TeV (&radic;s = 8 TeV)
        </span>
      </div>

      {/* Control Sliders */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Protons per Bunch */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <label htmlFor="protons-per-bunch-slider" className="text-slate-300 font-medium">Protons per Bunch</label>
            <span className="font-mono text-cyan-300">
              {(protonsPerBunch / 1e11).toFixed(2)} &times; 10<sup>11</sup>
            </span>
          </div>
          <input
            id="protons-per-bunch-slider"
            type="range"
            min={1.0e11}
            max={1.5e11}
            step={0.01e11}
            value={protonsPerBunch}
            onChange={(e) => setProtonsPerBunch(parseFloat(e.target.value))}
            disabled={isBusy}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500 disabled:opacity-50"
          />
        </div>

        {/* Number of Bunches */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <label htmlFor="bunch-count-slider" className="text-slate-300 font-medium">Number of Bunches</label>
            <span className="font-mono text-cyan-300">{bunchCount}</span>
          </div>
          <input
            id="bunch-count-slider"
            type="range"
            min={1000}
            max={2808}
            step={1}
            value={bunchCount}
            onChange={(e) => setBunchCount(parseInt(e.target.value, 10))}
            disabled={isBusy}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500 disabled:opacity-50"
          />
        </div>
      </div>

      {/* Calculated Readouts */}
      <div className="flex items-center justify-between bg-slate-950/60 rounded-lg p-3 border border-slate-800/80 text-xs">
        <div>
          <span className="text-slate-400">Particle Type:</span>{' '}
          <span className="text-cyan-300 font-semibold">Protons (p⁺)</span>
          <span className="text-slate-500 ml-2 text-[10px]">
            (Neutrons cannot be accelerated as they lack electric charge)
          </span>
        </div>
        <div>
          <span className="text-slate-400">Relative Collision Rate:</span>{' '}
          <span className="font-mono text-emerald-400 font-bold">
            {(relativeIntensity * 100).toFixed(1)}% ({relativeIntensity.toFixed(2)}x nominal)
          </span>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-3 pt-1">
        <button
          onClick={onStartAutoRun}
          disabled={isBusy}
          className="flex-1 py-2.5 px-4 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:from-slate-700 disabled:to-slate-800 text-white font-medium text-sm rounded-lg shadow-md transition-all flex items-center justify-center gap-2 disabled:cursor-not-allowed"
        >
          {isBusy ? (
            <>
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
              State: {journeyState.toUpperCase()}...
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              </svg>
              Auto Run Acceleration &amp; Collision
            </>
          )}
        </button>

        <button
          onClick={onTriggerCollision}
          disabled={isBusy}
          className="py-2.5 px-4 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 font-medium text-sm rounded-lg border border-slate-700 transition-all flex items-center gap-2"
        >
          Trigger Collision Now
        </button>

        <button
          onClick={onReset}
          className="py-2.5 px-3 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 text-xs rounded-lg border border-slate-800 transition-all"
        >
          Reset View
        </button>
      </div>

      {/* Mandatory Verbatim Honesty Note (Non-negotiable 4) */}
      <div className="p-2.5 bg-cyan-950/30 border border-cyan-800/40 rounded-lg text-xs text-cyan-200/90 leading-relaxed flex items-start gap-2">
        <svg className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span>
          Beam parameters change the collision rate, not the physics outcomes. Every collision shown is a real ATLAS open data event.
        </span>
      </div>
    </div>
  );
};
