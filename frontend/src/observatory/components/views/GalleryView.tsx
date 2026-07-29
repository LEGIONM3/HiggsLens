import React from 'react';
import { Layers, ArrowUpRight, CheckCircle2 } from 'lucide-react';

interface GalleryViewProps {
  onSelectEventForDisplay?: (eventId: number) => void;
}

const CURATED_EVENTS = [
  { id: 100001, label: 'High-pT Tau Candidate', split: 'Test (v)', jets: 2, met: 56.18, prob: 0.884, signal: true },
  { id: 100002, label: 'Zero-Jet Hadronic Channel', split: 'Test (v)', jets: 0, met: 31.05, prob: 0.921, signal: true },
  { id: 100003, label: 'Background Di-jet Event', split: 'Test (v)', jets: 2, met: 18.42, prob: 0.142, signal: false },
  { id: 100004, label: 'High MET Candidate', split: 'Test (v)', jets: 1, met: 92.4, prob: 0.865, signal: true },
  { id: 100005, label: 'Subleading Lepton Event', split: 'Test (v)', jets: 1, met: 42.1, prob: 0.732, signal: true },
  { id: 100006, label: 'Background W+Jets Candidate', split: 'Test (v)', jets: 2, met: 64.8, prob: 0.089, signal: false },
];

export const GalleryView: React.FC<GalleryViewProps> = ({ onSelectEventForDisplay }) => {
  return (
    <div className="flex flex-col gap-6 animate-fadeIn">
      <div className="bg-[#090d16] border border-slate-800 rounded-2xl p-6 flex flex-col gap-6 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-purple-500/10 border border-purple-500/30 text-purple-400">
              <Layers className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Curated Open-Data Event Gallery</h2>
              <p className="text-xs text-slate-400">
                Sampled Recorded ATLAS Open-Data Events &bull; Click to view 3D event reconstruction
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {CURATED_EVENTS.map((evt) => (
            <div
              key={evt.id}
              onClick={() => onSelectEventForDisplay && onSelectEventForDisplay(evt.id)}
              className="p-4 rounded-xl bg-[#05070c] border border-slate-800 hover:border-cyan-500/50 transition-all cursor-pointer flex flex-col justify-between gap-3 group"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-white flex items-center gap-1.5">
                  #{evt.id}
                  <ArrowUpRight className="w-3.5 h-3.5 text-cyan-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                </span>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                    evt.signal ? 'bg-emerald-500/20 text-emerald-300' : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {evt.signal ? 'Signal Candidate' : 'Background'}
                </span>
              </div>

              <div>
                <h4 className="text-sm font-semibold text-slate-200">{evt.label}</h4>
                <p className="text-xs text-slate-400 font-mono mt-1">
                  Jets: {evt.jets} &bull; MET: {evt.met.toFixed(1)} GeV
                </p>
              </div>

              <div className="pt-2 border-t border-slate-900 flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400 text-[10px]">Signal Prob:</span>
                <span className="text-cyan-400 font-bold">{(evt.prob * 100).toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
