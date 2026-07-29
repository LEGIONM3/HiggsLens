import React from 'react';
import { Info, ShieldCheck } from 'lucide-react';
import { ModeType } from '../../tokens/theme';

interface QualityDisclaimerProps {
  mode: ModeType;
  onSkipToStudio?: () => void;
}

export const QualityDisclaimer: React.FC<QualityDisclaimerProps> = ({ mode, onSkipToStudio }) => {
  if (mode === 'journey') {
    return (
      <aside
        aria-label="Accelerator Journey Quality Disclosure"
        tabIndex={0}
        className="w-full bg-[#090d16]/95 border-b border-cyan-500/20 px-4 py-2 text-xs flex flex-wrap items-center justify-between gap-2 text-cyan-200/90 focus:outline-none focus:ring-1 focus:ring-cyan-400"
      >
        <div className="flex items-center gap-2 max-w-full">
          <Info className="w-4 h-4 text-cyan-400 shrink-0" aria-hidden="true" />
          <span className="font-medium leading-tight">
            Illustrative accelerator journey. HiggsLens next displays a recorded ATLAS open-data event.
          </span>
        </div>
        {onSkipToStudio && (
          <button
            onClick={onSkipToStudio}
            className="px-2.5 py-1 rounded bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-300 font-semibold text-[11px] transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-400 shrink-0 mt-0.5 sm:mt-0"
            aria-label="Skip Accelerator Journey and open Event Studio directly"
          >
            Skip to Event Studio &rarr;
          </button>
        )}
      </aside>
    );
  }

  if (mode === 'studio') {
    return (
      <aside
        aria-label="Event Reconstruction Studio Quality Disclosure"
        tabIndex={0}
        className="w-full bg-[#090d16]/95 border-b border-slate-800 px-4 py-2 text-xs flex items-center justify-between gap-2 text-slate-300 focus:outline-none focus:ring-1 focus:ring-cyan-400"
      >
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" aria-hidden="true" />
          <span className="leading-tight">
            Detector-inspired illustrative geometry — not to scale. Recorded event kinematics supplied by API; frozen certified model outputs provide classification only.
          </span>
        </div>
      </aside>
    );
  }

  return null;
};
