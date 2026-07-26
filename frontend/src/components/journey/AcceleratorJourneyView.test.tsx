import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import { AcceleratorJourneyView } from './AcceleratorJourneyView';

// Mock Three.js / Canvas to prevent WebGL canvas initialization errors in jsdom
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="mock-three-canvas">{children}</div>
  ),
  useFrame: vi.fn(),
}));

vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="mock-orbit-controls" />,
}));

describe('AcceleratorJourneyView', () => {
  it('renders heading briefing card, beam controls, and LHC ring disclaimer', () => {
    render(<AcceleratorJourneyView />);

    expect(screen.getByText('Accelerator Journey & Event Editor')).toBeDefined();
    expect(screen.getByText('LHC Beam Steering & Controls')).toBeDefined();
    expect(
      screen.getByText(/Beam parameters change the collision rate, not the physics outcomes/)
    ).toBeDefined();
  });
});
