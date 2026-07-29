import React, { useState, useEffect, useCallback } from 'react';
import { ModeType } from '../../tokens/theme';
import { ObservatoryHeader } from './ObservatoryHeader';
import { ObservatoryNav } from './ObservatoryNav';
import { QualityDisclaimer } from './QualityDisclaimer';
import { Sparkles, ExternalLink } from 'lucide-react';

const STORAGE_KEY = 'higgslens_preferred_mode';

interface ObservatoryShellProps {
  initialMode?: ModeType;
  activeMode?: ModeType;
  onModeChange?: (mode: ModeType) => void;
  children: (currentMode: ModeType, setMode: (m: ModeType) => void) => React.ReactNode;
}

export const ObservatoryShell: React.FC<ObservatoryShellProps> = ({
  initialMode,
  activeMode: controlledMode,
  onModeChange,
  children,
}) => {
  // Local state initialized with local storage persistence check
  const [internalMode, setInternalMode] = useState<ModeType>(() => {
    if (initialMode) return initialMode;
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved && ['journey', 'studio', 'leaderboard', 'pipeline', 'gallery', 'arena', 'lab'].includes(saved)) {
        return saved as ModeType;
      }
    } catch {
      // Ignore storage error
    }
    return 'journey'; // Default first-time visitor experience
  });

  const currentMode = controlledMode ?? internalMode;

  const handleModeChange = useCallback(
    (newMode: ModeType) => {
      setInternalMode(newMode);
      try {
        localStorage.setItem(STORAGE_KEY, newMode);
      } catch {
        // Ignore storage write error
      }
      if (onModeChange) {
        onModeChange(newMode);
      }
    },
    [onModeChange]
  );

  // Keyboard shortcut listener (Number keys 1-7 for navigation when not in an editable field)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeEl = document.activeElement as HTMLElement | null;
      const isInput =
        activeEl &&
        (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA' || activeEl.isContentEditable);
      if (isInput) return;

      const keyMap: Record<string, ModeType> = {
        '1': 'journey',
        '2': 'studio',
        '3': 'leaderboard',
        '4': 'pipeline',
        '5': 'gallery',
        '6': 'arena',
        '7': 'lab',
      };

      if (e.key in keyMap) {
        handleModeChange(keyMap[e.key]);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleModeChange]);

  return (
    <div className="min-h-screen flex flex-col justify-between bg-[#05070c] text-slate-100 font-sans relative overflow-x-hidden selection:bg-cyan-500 selection:text-slate-950">
      {/* Subtle Background Radial Observatory Grid */}
      <div
        className="fixed inset-0 pointer-events-none z-0 opacity-20 bg-[radial-gradient(#1e293b_1px,transparent_1px)] [background-size:24px_24px]"
        aria-hidden="true"
      />

      {/* Outer Shell Wrapper */}
      <div className="relative z-10 flex flex-col flex-1">
        {/* Observatory Header */}
        <ObservatoryHeader activeMode={currentMode} />

        {/* Observatory Navigation */}
        <ObservatoryNav activeMode={currentMode} onSelectMode={handleModeChange} />

        {/* Quality Disclaimer Banner */}
        <QualityDisclaimer
          mode={currentMode}
          onSkipToStudio={() => handleModeChange('studio')}
        />

        {/* Main Observatory Viewport Container */}
        <main
          id={`panel-${currentMode}`}
          role="tabpanel"
          aria-labelledby={`tab-${currentMode}`}
          className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex-1 w-full flex flex-col gap-6"
        >
          {children(currentMode, handleModeChange)}
        </main>
      </div>

      {/* Observatory Scientific Footer */}
      <footer className="relative z-10 border-t border-slate-900 bg-[#05070c]/90 py-5 text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-400" aria-hidden="true" />
            <span className="font-medium">
              HiggsLens &middot; Orbital Scientific Observatory &amp; ATLAS ML Classification Platform
            </span>
          </div>

          <div className="flex items-center gap-6 text-slate-400 font-mono text-[11px]">
            <a
              href="https://opendata.cern.ch/record/328"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-cyan-400 transition-colors flex items-center gap-1 focus:outline-none focus:ring-1 focus:ring-cyan-400 rounded px-1"
            >
              CERN Record 328 <ExternalLink className="w-3 h-3" />
            </a>
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-cyan-400 transition-colors flex items-center gap-1 focus:outline-none focus:ring-1 focus:ring-cyan-400 rounded px-1"
            >
              OpenAPI Docs <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};
