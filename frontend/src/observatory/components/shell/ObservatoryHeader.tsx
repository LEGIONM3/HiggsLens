import React from 'react';
import { Atom, Radio, ExternalLink } from 'lucide-react';
import { ModeType } from '../../tokens/theme';

interface ObservatoryHeaderProps {
  activeMode: ModeType;
  apiConnected?: boolean;
}

export const ObservatoryHeader: React.FC<ObservatoryHeaderProps> = ({
  activeMode,
  apiConnected = true,
}) => {
  const getModeBadgeTitle = (mode: ModeType) => {
    switch (mode) {
      case 'journey':
        return 'Experience A: Accelerator Journey';
      case 'studio':
        return 'Experience B: Event Studio';
      case 'leaderboard':
        return 'Model Leaderboard';
      case 'pipeline':
        return 'Data Pipeline';
      case 'gallery':
        return 'Event Gallery';
      case 'arena':
        return 'Model Arena';
      case 'lab':
        return 'Experimental Lab';
      default:
        return 'Observatory';
    }
  };

  return (
    <header className="w-full bg-[#05070c]/95 border-b border-slate-800/80 backdrop-blur-md sticky top-0 z-50 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex flex-wrap items-center justify-between gap-3">
        {/* Left Branding */}
        <div className="flex items-center gap-3">
          <div
            className="p-2 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 shadow-sm"
            aria-hidden="true"
          >
            <Atom className="w-6 h-6 stroke-[2.2]" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2 font-sans">
                HiggsLens
              </h1>
              <span className="px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-500/30 text-cyan-300 text-[10px] font-mono font-semibold tracking-wide">
                CERN OPEN DATA 328
              </span>
              <span className="px-2 py-0.5 rounded-full bg-slate-900 border border-slate-700 text-slate-300 text-[10px] font-mono font-medium">
                {getModeBadgeTitle(activeMode)}
              </span>
            </div>
            <p className="text-xs text-slate-400 font-medium hidden sm:block">
              ATLAS H &rarr; &tau;&tau; Open Data Research &amp; Visual Observatory
            </p>
          </div>
        </div>

        {/* Right System Telemetry */}
        <div className="flex items-center gap-4 text-xs text-slate-400 font-mono">
          <div
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900/80 border border-slate-800"
            role="status"
            aria-live="polite"
          >
            <Radio className={`w-3.5 h-3.5 ${apiConnected ? 'text-emerald-400 animate-pulse' : 'text-amber-400'}`} />
            <span className={apiConnected ? 'text-emerald-400 font-medium' : 'text-amber-400 font-medium'}>
              {apiConnected ? 'API Active (Port 8000)' : 'API Offline (Fallback)'}
            </span>
          </div>

          <a
            href="https://opendata.cern.ch/record/328"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden md:flex items-center gap-1 text-slate-400 hover:text-cyan-400 transition-colors focus:outline-none focus:ring-1 focus:ring-cyan-400 rounded px-1.5 py-0.5"
            aria-label="View CERN Record 328 in new tab"
          >
            DOI: 10.7483 <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      </div>
    </header>
  );
};
