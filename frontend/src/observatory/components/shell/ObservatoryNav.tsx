import React from 'react';
import { ModeType } from '../../tokens/theme';

interface ObservatoryNavProps {
  activeMode: ModeType;
  onSelectMode: (mode: ModeType) => void;
}

interface NavItem {
  id: ModeType;
  label: string;
  badge?: string;
  ariaLabel: string;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'journey', label: '1. Accelerator Journey', badge: 'Exp A', ariaLabel: 'Navigate to Experience A: Accelerator Journey' },
  { id: 'studio', label: '2. Event Studio', badge: 'Exp B', ariaLabel: 'Navigate to Experience B: Event Reconstruction Studio' },
  { id: 'leaderboard', label: '3. Leaderboard', ariaLabel: 'Navigate to Official Model Leaderboard' },
  { id: 'pipeline', label: '4. Data Pipeline', ariaLabel: 'Navigate to CERN Open Data Pipeline' },
  { id: 'gallery', label: '5. Event Gallery', ariaLabel: 'Navigate to Event Gallery' },
  { id: 'arena', label: '6. Model Arena', ariaLabel: 'Navigate to Model Arena' },
  { id: 'lab', label: '7. Experimental Lab', ariaLabel: 'Navigate to Experimental Lab' },
];

export const ObservatoryNav: React.FC<ObservatoryNavProps> = ({ activeMode, onSelectMode }) => {
  return (
    <nav
      className="w-full bg-[#05070c]/80 border-b border-slate-800/80 px-4 py-2 sticky top-[57px] z-40 backdrop-blur-sm"
      aria-label="Observatory Navigation"
    >
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-2">
        <div
          role="tablist"
          aria-label="Observatory Main Views"
          className="flex items-center gap-1.5 overflow-x-auto scrollbar-thin py-1 w-full"
        >
          {NAV_ITEMS.map((item) => {
            const isActive = activeMode === item.id;
            return (
              <button
                key={item.id}
                role="tab"
                id={`tab-${item.id}`}
                aria-selected={isActive}
                aria-controls={`panel-${item.id}`}
                aria-label={item.ariaLabel}
                onClick={() => onSelectMode(item.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all flex items-center gap-1.5 focus:outline-none focus:ring-2 focus:ring-cyan-400 ${
                  isActive
                    ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <span>{item.label}</span>
                {item.badge && (
                  <span
                    className={`px-1.5 py-0.2 rounded text-[9px] font-mono uppercase ${
                      isActive ? 'bg-slate-950/20 text-slate-950 font-bold' : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
};
