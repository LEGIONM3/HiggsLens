import { describe, expect, it } from 'vitest';
import {
  computeCartesianMomentum,
  computeDirectionVector,
  computeLengthScale,
  computeMETDirectionVector,
} from './kinematics';

describe('Kinematics pure math functions', () => {
  it('computes cartesian momentum for positive eta', () => {
    // Worked example: pt=57.6, eta=1.2, phi=-0.7
    const pt = 57.6;
    const eta = 1.2;
    const phi = -0.7;

    const cart = computeCartesianMomentum(pt, eta, phi);
    expect(cart.px).toBeCloseTo(44.0549, 3);
    expect(cart.py).toBeCloseTo(-37.1069, 3);
    expect(cart.pz).toBeCloseTo(86.9449, 3);
    expect(cart.p).toBeCloseTo(104.2937, 3);
  });

  it('computes cartesian momentum for negative eta', () => {
    const pt = 45.0;
    const eta = -1.5;
    const phi = 0.5;

    const cart = computeCartesianMomentum(pt, eta, phi);
    expect(cart.px).toBeCloseTo(45.0 * Math.cos(0.5), 3);
    expect(cart.py).toBeCloseTo(45.0 * Math.sin(0.5), 3);
    expect(cart.pz).toBeCloseTo(45.0 * Math.sinh(-1.5), 3);
    expect(cart.pz).toBeLessThan(0);
  });

  it('normalizes direction unit vector', () => {
    const dir = computeDirectionVector(30, 40, 0);
    expect(dir[0]).toBeCloseTo(0.6, 5);
    expect(dir[1]).toBeCloseTo(0.8, 5);
    expect(dir[2]).toBeCloseTo(0.0, 5);

    const norm = Math.hypot(dir[0], dir[1], dir[2]);
    expect(norm).toBeCloseTo(1.0, 5);
  });

  it('computes MET direction vector strictly in transverse z=0 plane', () => {
    const met_phi = 1.23;
    const dir = computeMETDirectionVector(met_phi);
    expect(dir[0]).toBeCloseTo(Math.cos(met_phi), 5);
    expect(dir[1]).toBeCloseTo(Math.sin(met_phi), 5);
    expect(dir[2]).toBe(0.0);
  });

  it('enforces monotonic length scaling rule L = L_min + k * log10(1 + pt)', () => {
    const l30 = computeLengthScale(30);
    const l300 = computeLengthScale(300);

    expect(l30).toBeCloseTo(1.5 + 1.5 * Math.log10(31), 3);
    expect(l300).toBeCloseTo(1.5 + 1.5 * Math.log10(301), 3);
    expect(l300).toBeGreaterThan(l30);
  });
});
