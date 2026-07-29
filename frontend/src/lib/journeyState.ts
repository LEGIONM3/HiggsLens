/**
 * Deterministic PRNG using Mulberry32 for reproducible accelerator packet animations.
 */
export function createDeterministicPRNG(seed: number) {
  let s = seed >>> 0;
  return function () {
    let t = (s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface BeamPacketConfig {
  bunchCount: number;
  protonsPerBunch: number;
  energyGev: number;
  speed: number;
}

export const DEFAULT_BEAM_CONFIG: BeamPacketConfig = {
  bunchCount: 2808,
  protonsPerBunch: 1.15e11,
  energyGev: 4000, // 4 TeV per beam => sqrt(s) = 8 TeV
  speed: 1.0,
};
