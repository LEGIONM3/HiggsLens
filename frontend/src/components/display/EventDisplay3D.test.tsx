import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, act, fireEvent } from '@testing-library/react';
import React from 'react';
import { EventDisplay3D } from './EventDisplay3D';
import { EducationProvider } from '../../context/EducationContext';

// Mock Canvas & Three.js components for Vitest DOM environment
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="r3f-canvas">{children}</div>
  ),
  useThree: () => ({ camera: { position: { set: vi.fn() }, lookAt: vi.fn() } }),
  useFrame: vi.fn(),
}));

vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls" />,
  Line: () => <div data-testid="drei-line" />,
}));

const mockEventSampleResponse = {
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
        threshold: 0.85,
      },
    },
  ],
  count: 1,
  seed: 42,
  label_filter: 'any',
};

describe('EventDisplay3D & Release B Spatial Observatory Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders top toolbar, controls, and Spatial Observatory canvas cleanly', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockEventSampleResponse,
    }) as any;

    await act(async () => {
      render(
        <EducationProvider>
          <EventDisplay3D />
        </EducationProvider>
      );
    });

    expect(screen.getByText(/3D Event Display & Kinematics/i)).toBeDefined();
    expect(screen.getByText(/Next Random Sample/i)).toBeDefined();
    expect(screen.getByText(/Detector-inspired illustrative geometry — not to scale/i)).toBeDefined();
  });

  it('provides accessible form labels, ids, and names for filter and seed controls', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockEventSampleResponse,
    }) as any;

    await act(async () => {
      render(
        <EducationProvider>
          <EventDisplay3D />
        </EducationProvider>
      );
    });

    const filterSelect = screen.getByLabelText(/Filter:/i) as HTMLSelectElement;
    expect(filterSelect).toBeDefined();
    expect(filterSelect.id).toBe('event-label-filter');
    expect(filterSelect.name).toBe('event-label-filter');

    const seedInput = screen.getByLabelText(/Seed:/i) as HTMLInputElement;
    expect(seedInput).toBeDefined();
    expect(seedInput.id).toBe('event-seed-input');
    expect(seedInput.name).toBe('event-seed-input');
  });

  it('handles Release B camera preset changes (Observatory, Event Focus, Barrel Slice, Longitudinal) and axes toggle', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockEventSampleResponse,
    }) as any;

    await act(async () => {
      render(
        <EducationProvider>
          <EventDisplay3D />
        </EducationProvider>
      );
    });

    const observatoryBtn = screen.getByRole('button', { name: /Observatory/i });
    const eventFocusBtn = screen.getByRole('button', { name: /Event Focus/i });
    const barrelSliceBtn = screen.getByRole('button', { name: /Barrel Slice/i });
    const longitudinalBtn = screen.getByRole('button', { name: /Longitudinal/i });
    const axesBtn = screen.getByRole('button', { name: /Axes \(XYZ\)/i });

    expect(observatoryBtn).toBeDefined();
    expect(eventFocusBtn).toBeDefined();
    expect(barrelSliceBtn).toBeDefined();
    expect(longitudinalBtn).toBeDefined();
    expect(axesBtn).toBeDefined();

    act(() => {
      fireEvent.click(eventFocusBtn);
      fireEvent.click(barrelSliceBtn);
      fireEvent.click(longitudinalBtn);
      fireEvent.click(observatoryBtn);
      fireEvent.click(axesBtn);
    });
  });

  it('renders an explicit error state when event fetching fails', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({ detail: 'API connection failed' }),
    }) as any;

    await act(async () => {
      render(
        <EducationProvider>
          <EventDisplay3D />
        </EducationProvider>
      );
    });

    expect(screen.getByText(/Error Loading Events/i)).toBeDefined();
    expect(screen.getByText(/API connection failed/i)).toBeDefined();
  });
});
