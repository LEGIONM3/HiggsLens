/**
 * Kinematics & Coordinate Geometry Library for HiggsLens 3D Event Display.
 * Pure mathematical functions for computing Cartesian momenta, direction vectors,
 * logarithmic length scaling, MET transverse vectors, and sentinel-guided object derivation.
 */

export interface CartesianMomentum {
  px: number;
  py: number;
  pz: number;
  p: number;
}

export interface RenderableObject {
  id: string;
  name: string;
  type: 'tau' | 'lepton' | 'leading_jet' | 'subleading_jet' | 'met';
  color: string;
  pt: number;
  eta: number;
  phi: number;
  direction: [number, number, number];
  length: number;
  cartesian: CartesianMomentum;
  tooltip: string;
}

export const L_MIN_DEFAULT = 1.5;
export const K_SCALE_DEFAULT = 1.5;

/**
 * Computes Cartesian momentum components (px, py, pz, p) from transverse momentum (pt),
 * pseudorapidity (eta), and azimuthal angle (phi).
 */
export function computeCartesianMomentum(
  pt: number,
  eta: number,
  phi: number
): CartesianMomentum {
  const px = pt * Math.cos(phi);
  const py = pt * Math.sin(phi);
  const pz = pt * Math.sinh(eta);
  const p = pt * Math.cosh(eta);
  return { px, py, pz, p };
}

/**
 * Computes a normalized 3D direction unit vector [dx, dy, dz] from Cartesian momentum components.
 */
export function computeDirectionVector(
  px: number,
  py: number,
  pz: number
): [number, number, number] {
  const norm = Math.hypot(px, py, pz);
  if (norm === 0) return [0, 0, 0];
  return [px / norm, py / norm, pz / norm];
}

/**
 * Computes scaled 3D visual length using logarithmic scaling rule:
 * L = L_min + k * log10(1 + pt)
 */
export function computeLengthScale(
  pt: number,
  L_min: number = L_MIN_DEFAULT,
  k: number = K_SCALE_DEFAULT
): number {
  if (pt <= 0) return L_min;
  return L_min + k * Math.log10(1 + pt);
}

/**
 * Computes transverse-only direction unit vector for MET in the z=0 plane.
 */
export function computeMETDirectionVector(met_phi: number): [number, number, number] {
  return [Math.cos(met_phi), Math.sin(met_phi), 0];
}

/**
 * Derives the list of 3D renderable physics objects from raw event features.
 * Enforces strict Sentinel Rule (-999.0 values and PRI_jet_num rules).
 */
export function getRenderablePhysicsObjects(
  features: Record<string, number>
): RenderableObject[] {
  const objects: RenderableObject[] = [];

  // 1. Hadronic Tau (PRI_tau_pt, PRI_tau_eta, PRI_tau_phi)
  const tau_pt = features['PRI_tau_pt'] ?? -999.0;
  if (tau_pt !== -999.0 && tau_pt > 0) {
    const tau_eta = features['PRI_tau_eta'] ?? 0;
    const tau_phi = features['PRI_tau_phi'] ?? 0;
    const cart = computeCartesianMomentum(tau_pt, tau_eta, tau_phi);
    const dir = computeDirectionVector(cart.px, cart.py, cart.pz);
    const len = computeLengthScale(tau_pt);
    objects.push({
      id: 'tau',
      name: 'Hadronic Tau (\u03c4_had)',
      type: 'tau',
      color: '#F97316', // Orange
      pt: tau_pt,
      eta: tau_eta,
      phi: tau_phi,
      direction: dir,
      length: len,
      cartesian: cart,
      tooltip:
        'Hadronic tau decay candidate. The Higgs boson (H \u2192 \u03c4\u03c4) decays into a pair of tau leptons, producing narrow collimated hadronic jets.'
    });
  }

  // 2. Lepton (PRI_lep_pt, PRI_lep_eta, PRI_lep_phi)
  const lep_pt = features['PRI_lep_pt'] ?? -999.0;
  if (lep_pt !== -999.0 && lep_pt > 0) {
    const lep_eta = features['PRI_lep_eta'] ?? 0;
    const lep_phi = features['PRI_lep_phi'] ?? 0;
    const cart = computeCartesianMomentum(lep_pt, lep_eta, lep_phi);
    const dir = computeDirectionVector(cart.px, cart.py, cart.pz);
    const len = computeLengthScale(lep_pt);
    objects.push({
      id: 'lepton',
      name: 'Light Lepton (e / \u03bc)',
      type: 'lepton',
      color: '#06B6D4', // Cyan
      pt: lep_pt,
      eta: lep_eta,
      phi: lep_phi,
      direction: dir,
      length: len,
      cartesian: cart,
      tooltip:
        'Isolated light lepton (electron or muon) from the leptonic decay branch of the tau pair.'
    });
  }

  // 3. Jets driven strictly by PRI_jet_num sentinel rule
  const jet_num = Math.round(features['PRI_jet_num'] ?? 0);

  if (jet_num >= 1) {
    const jet_lead_pt = features['PRI_jet_leading_pt'] ?? -999.0;
    if (jet_lead_pt !== -999.0 && jet_lead_pt > 0) {
      const jet_lead_eta = features['PRI_jet_leading_eta'] ?? 0;
      const jet_lead_phi = features['PRI_jet_leading_phi'] ?? 0;
      const cart = computeCartesianMomentum(jet_lead_pt, jet_lead_eta, jet_lead_phi);
      const dir = computeDirectionVector(cart.px, cart.py, cart.pz);
      const len = computeLengthScale(jet_lead_pt);
      objects.push({
        id: 'leading_jet',
        name: 'Leading Hadronic Jet',
        type: 'leading_jet',
        color: '#EAB308', // Yellow
        pt: jet_lead_pt,
        eta: jet_lead_eta,
        phi: jet_lead_phi,
        direction: dir,
        length: len,
        cartesian: cart,
        tooltip:
          'Highest-pT reconstructed jet of hadrons produced by initial/final state radiation or vector boson fusion.'
      });
    }
  }

  if (jet_num >= 2) {
    const jet_sub_pt = features['PRI_jet_subleading_pt'] ?? -999.0;
    if (jet_sub_pt !== -999.0 && jet_sub_pt > 0) {
      const jet_sub_eta = features['PRI_jet_subleading_eta'] ?? 0;
      const jet_sub_phi = features['PRI_jet_subleading_phi'] ?? 0;
      const cart = computeCartesianMomentum(jet_sub_pt, jet_sub_eta, jet_sub_phi);
      const dir = computeDirectionVector(cart.px, cart.py, cart.pz);
      const len = computeLengthScale(jet_sub_pt);
      objects.push({
        id: 'subleading_jet',
        name: 'Subleading Hadronic Jet',
        type: 'subleading_jet',
        color: '#F59E0B', // Amber
        pt: jet_sub_pt,
        eta: jet_sub_eta,
        phi: jet_sub_phi,
        direction: dir,
        length: len,
        cartesian: cart,
        tooltip:
          'Second highest-pT reconstructed jet. In VBF Higgs production, two forward jets are characteristic.'
      });
    }
  }

  // 4. Missing Transverse Energy (MET)
  const met_pt = features['PRI_met'] ?? -999.0;
  if (met_pt !== -999.0 && met_pt > 0) {
    const met_phi = features['PRI_met_phi'] ?? 0;
    const dir = computeMETDirectionVector(met_phi);
    const len = computeLengthScale(met_pt);
    objects.push({
      id: 'met',
      name: 'Missing Transverse Energy (E_T^miss)',
      type: 'met',
      color: '#EF4444', // Red
      pt: met_pt,
      eta: 0, // Transverse only
      phi: met_phi,
      direction: dir,
      length: len,
      cartesian: { px: met_pt * Math.cos(met_phi), py: met_pt * Math.sin(met_phi), pz: 0, p: met_pt },
      tooltip:
        'Unbalanced momentum in the transverse plane. Indicates undetected weakly-interacting particles (neutrinos from tau decays).'
    });
  }

  return objects;
}
