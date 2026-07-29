import { describe, it, expect } from 'vitest';
import { etaPhiPtToCartesian, parseEventKinematics, isSentinel } from '../kinematics';

describe('Kinematics Utilities & Vector Math', () => {
  it('converts eta, phi, pt to Cartesian Vector3D correctly', () => {
    const pt = 50;
    const eta = 0;
    const phi = 0; // x-axis direction
    const vec = etaPhiPtToCartesian(pt, eta, phi);

    expect(vec.x).toBeCloseTo(50, 4);
    expect(vec.y).toBeCloseTo(0, 4);
    expect(vec.z).toBeCloseTo(0, 4);
  });

  it('identifies sentinel -999.0 missing values', () => {
    expect(isSentinel(-999.0)).toBe(true);
    expect(isSentinel(undefined)).toBe(true);
    expect(isSentinel(null)).toBe(true);
    expect(isSentinel(42.5)).toBe(false);
  });

  it('parses recorded event kinematics omitting missing objects', () => {
    const features = {
      PRI_tau_pt: 30.0,
      PRI_tau_eta: 1.0,
      PRI_tau_phi: 0.5,
      PRI_lep_pt: 45.0,
      PRI_lep_eta: -0.5,
      PRI_lep_phi: 2.0,
      PRI_jet_leading_pt: -999.0, // Sentinel missing
      PRI_jet_leading_eta: -999.0,
      PRI_jet_leading_phi: -999.0,
    };

    const parsed = parseEventKinematics(features);
    expect(parsed).toHaveLength(2); // Only Tau & Lepton present
    expect(parsed[0].object_type).toBe('tau');
    expect(parsed[1].object_type).toBe('lepton');
  });
});
