export const OBSERVATORY_THEME = {
  colors: {
    bg: {
      deep: '#05070c',
      panel: '#090d16',
      surface: '#0f172a',
      overlay: 'rgba(5, 7, 12, 0.85)',
      card: 'rgba(15, 23, 42, 0.65)',
    },
    border: {
      subtle: 'rgba(255, 255, 255, 0.08)',
      medium: 'rgba(255, 255, 255, 0.15)',
      active: '#06b6d4',
      highlight: 'rgba(6, 182, 212, 0.5)',
    },
    semantics: {
      beam: '#06b6d4',       // Cyan for Accelerator Beam & Tau candidates
      beamGlow: '#22d3ee',
      lepton: '#f8fafc',     // Detector White for Leptons (electrons/muons)
      leptonSubtle: '#cbd5e1',
      jetLeading: '#f59e0b',  // Gold/Amber for Leading Jet
      jetSubleading: '#fbbf24',// Warm Gold for Subleading Jet
      met: '#ec4899',        // Magenta for Missing Transverse Energy (MET)
      structure: '#1e293b',  // Neutral Slate for Detector Chambers
      success: '#10b981',    // Emerald green for verified states
      warning: '#f59e0b',    // Amber for unverified/sentinel states
    },
    text: {
      primary: '#f8fafc',
      secondary: '#94a3b8',
      muted: '#64748b',
      accent: '#38bdf8',
    },
  },
  typography: {
    fontSans: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontMono: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  },
  shadows: {
    glowCyan: '0 0 20px rgba(6, 182, 212, 0.25)',
    glowMagenta: '0 0 20px rgba(236, 72, 153, 0.25)',
    glowAmber: '0 0 20px rgba(245, 158, 11, 0.25)',
  },
} as const;

export type ModeType = 'journey' | 'studio' | 'leaderboard' | 'pipeline' | 'gallery' | 'arena' | 'lab';
