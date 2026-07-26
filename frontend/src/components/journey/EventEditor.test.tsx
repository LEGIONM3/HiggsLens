import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { EventEditor } from './EventEditor';

describe('EventEditor', () => {
  const initialPri = {
    PRI_tau_pt: 45.2,
    PRI_tau_eta: 1.2,
    PRI_tau_phi: -0.7,
    PRI_lep_pt: 32.1,
    PRI_lep_eta: 0.5,
    PRI_lep_phi: 2.9,
    PRI_met: 40.0,
    PRI_met_phi: 0.1,
    PRI_met_sumet: 180.0,
    PRI_jet_num: 0,
    PRI_jet_leading_pt: -999.0,
    PRI_jet_leading_eta: -999.0,
    PRI_jet_leading_phi: -999.0,
    PRI_jet_subleading_pt: -999.0,
    PRI_jet_subleading_eta: -999.0,
    PRI_jet_subleading_phi: -999.0,
    PRI_jet_all_pt: 0.0,
  };

  it('renders collapsed state by default and expands on click', () => {
    render(<EventEditor initialPriFeatures={initialPri} onDeriveComplete={vi.fn()} />);

    expect(screen.getByText('Advanced — Event Kinematics Editor')).toBeDefined();
    expect(screen.queryByText('MMC Policy:')).toBeNull();

    // Expand
    fireEvent.click(screen.getByText('Advanced — Event Kinematics Editor'));
    expect(screen.getByText(/MMC Policy:/)).toBeDefined();
  });

  it('renders MMC sentinel caveat notice verbatim when expanded', () => {
    render(<EventEditor initialPriFeatures={initialPri} onDeriveComplete={vi.fn()} />);
    fireEvent.click(screen.getByText('Advanced — Event Kinematics Editor'));

    expect(
      screen.getByText(/DER_mass_MMC cannot be recomputed from edited kinematics/)
    ).toBeDefined();
  });
});
