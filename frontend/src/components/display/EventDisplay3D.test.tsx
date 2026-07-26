import { describe, expect, it, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import React from 'react';
import { EventDisplay3D } from './EventDisplay3D';

// Mock Canvas & Three.js components for Vitest DOM environment
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="r3f-canvas">{children}</div>
  ),
  useThree: () => ({ camera: { position: { set: vi.fn() }, lookAt: vi.fn() } }),
}));

vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls" />,
  Line: () => <div data-testid="drei-line" />,
}));

describe('EventDisplay3D Component Smoke Test', () => {
  it('renders top toolbar, controls, and canvas container cleanly', async () => {
    // Mock fetch for API /events/sample
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        events: [
          {
            event_id: 300001,
            true_label: 'signal',
            features: {
              PRI_tau_pt: 45.2,
              PRI_tau_eta: 0.5,
              PRI_tau_phi: 1.2,
              PRI_lep_pt: 32.1,
              PRI_lep_eta: -0.8,
              PRI_lep_phi: -2.1,
              PRI_met: 50.0,
              PRI_met_phi: 0.1,
              PRI_jet_num: 1,
              PRI_jet_leading_pt: 85.4,
              PRI_jet_leading_eta: 1.1,
              PRI_jet_leading_phi: -0.5,
              DER_mass_vis: 78.4,
              DER_mass_MMC: 125.0,
            },
            prediction: {
              model_id: 'xgboost',
              probability: 0.8912,
              predicted_label: 'signal',
              threshold: 0.8118,
            },
          },
        ],
        count: 1,
        seed: 42,
        label_filter: 'any',
      }),
    }) as any;

    await act(async () => {
      render(<EventDisplay3D />);
    });

    expect(screen.getByText(/3D Event Display & Kinematics/i)).toBeDefined();
    expect(screen.getByText(/Sample New Set/i)).toBeDefined();
  });
});
