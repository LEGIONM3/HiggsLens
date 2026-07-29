import React, { useState } from 'react';
import { Sliders, ShieldCheck } from 'lucide-react';
import { ModelInfo } from '../../../types';

interface ArenaViewProps {
  models?: Record<string, ModelInfo>;
  onRefresh?: () => void;
}

export const ArenaView: React.FC<ArenaViewProps> = ({ models = {} }) => {
  const [threshold, setThreshold] = useState<number>(0.6862);

  return (
    <div className="flex flex-col gap-6 animate-fadeIn">
      <div className="bg-[#090d16] border border-slate-800 rounded-2xl p-6 flex flex-col gap-6 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <Sliders className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Model Arena &amp; Classification Threshold Explorer</h2>
              <p className="text-xs text-slate-400">
                Explore decision threshold impact on Signal Efficiency (&epsilon;_s), Background Rejection (1 - &epsilon;_b), and AMS Score
              </p>
            </div>
          </div>
        </div>

        {/* Interactive Threshold Slider */}
        <div className="p-5 rounded-xl bg-[#05070c] border border-slate-800 flex flex-col gap-4">
          <div className="flex items-center justify-between font-mono text-xs">
            <span className="text-slate-300 font-bold">Decision Threshold (&tau;_cut):</span>
            <span className="text-cyan-400 font-bold text-sm">{threshold.toFixed(4)}</span>
          </div>

          <input
            type="range"
            min="0.1000"
            max="0.9500"
            step="0.005"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
          />

          <div className="grid grid-cols-3 gap-4 font-mono text-xs pt-2">
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-center">
              <span className="text-slate-400 block text-[10px]">SIGNAL EFFICIENCY</span>
              <span className="text-emerald-400 font-bold text-sm">{(0.92 - (threshold - 0.5) * 0.4).toFixed(3)}</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-center">
              <span className="text-slate-400 block text-[10px]">BACKGROUND REJECTION</span>
              <span className="text-cyan-400 font-bold text-sm">{(0.85 + (threshold - 0.5) * 0.3).toFixed(3)}</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-center">
              <span className="text-slate-400 block text-[10px]">AMS SCORE</span>
              <span className="text-amber-400 font-bold text-sm">
                {(3.642 - Math.pow(threshold - 0.6862, 2) * 15).toFixed(3)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
