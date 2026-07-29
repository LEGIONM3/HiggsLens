import { ReconstructedObject } from '../types';

export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export function isSentinel(val: number | null | undefined): boolean {
  if (val === null || val === undefined) return true;
  return Math.abs(val - (-999.0)) < 1e-3 || isNaN(val);
}

/**
 * Converts recorded ATLAS kinematic coordinates (pT, eta, phi) to 3D Cartesian momentum vector.
 */
export function etaPhiPtToCartesian(pt: number, eta: number, phi: number): Vector3D {
  const px = pt * Math.cos(phi);
  const py = pt * Math.sin(phi);
  const pz = pt * Math.sinh(eta);
  return { x: px, y: py, z: pz };
}

/**
 * Parses raw ATLAS event features dictionary into certified reconstructed event objects.
 * Omits sentinel (-999.0) or unmeasured missing objects.
 */
export function parseEventKinematics(features: Record<string, number>): ReconstructedObject[] {
  const objects: ReconstructedObject[] = [];

  // Tau Candidate
  const tauPt = features['PRI_tau_pt'];
  const tauEta = features['PRI_tau_eta'];
  const tauPhi = features['PRI_tau_phi'];
  if (!isSentinel(tauPt) && !isSentinel(tauEta) && !isSentinel(tauPhi)) {
    objects.push({
      object_type: 'tau',
      label: 'Tau Candidate (τ)',
      pt: tauPt,
      eta: tauEta,
      phi: tauPhi,
      present: true,
    });
  }

  // Lepton (Electron / Muon)
  const lepPt = features['PRI_lep_pt'];
  const lepEta = features['PRI_lep_eta'];
  const lepPhi = features['PRI_lep_phi'];
  if (!isSentinel(lepPt) && !isSentinel(lepEta) && !isSentinel(lepPhi)) {
    objects.push({
      object_type: 'lepton',
      label: 'Light Lepton (e/μ)',
      pt: lepPt,
      eta: lepEta,
      phi: lepPhi,
      present: true,
    });
  }

  // Leading Jet
  const jetLeadingPt = features['PRI_jet_leading_pt'];
  const jetLeadingEta = features['PRI_jet_leading_eta'];
  const jetLeadingPhi = features['PRI_jet_leading_phi'];
  if (!isSentinel(jetLeadingPt) && !isSentinel(jetLeadingEta) && !isSentinel(jetLeadingPhi)) {
    objects.push({
      object_type: 'jet_leading',
      label: 'Leading Jet (j1)',
      pt: jetLeadingPt,
      eta: jetLeadingEta,
      phi: jetLeadingPhi,
      present: true,
    });
  }

  // Subleading Jet
  const jetSubleadingPt = features['PRI_jet_subleading_pt'];
  const jetSubleadingEta = features['PRI_jet_subleading_eta'];
  const jetSubleadingPhi = features['PRI_jet_subleading_phi'];
  if (!isSentinel(jetSubleadingPt) && !isSentinel(jetSubleadingEta) && !isSentinel(jetSubleadingPhi)) {
    objects.push({
      object_type: 'jet_subleading',
      label: 'Subleading Jet (j2)',
      pt: jetSubleadingPt,
      eta: jetSubleadingEta,
      phi: jetSubleadingPhi,
      present: true,
    });
  }

  return objects;
}
