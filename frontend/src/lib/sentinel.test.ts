import { describe, expect, it } from 'vitest';
import { getRenderablePhysicsObjects } from './kinematics';

describe('Sentinel rule and jet multiplicity filtering', () => {
  it('renders 0 jets when PRI_jet_num == 0', () => {
    const features: Record<string, number> = {
      PRI_tau_pt: 45.2,
      PRI_tau_eta: 0.5,
      PRI_tau_phi: 1.2,
      PRI_lep_pt: 32.1,
      PRI_lep_eta: -0.8,
      PRI_lep_phi: -2.1,
      PRI_met: 50.0,
      PRI_met_phi: 0.1,
      PRI_jet_num: 0,
      PRI_jet_leading_pt: -999.0,
      PRI_jet_leading_eta: -999.0,
      PRI_jet_leading_phi: -999.0,
      PRI_jet_subleading_pt: -999.0,
      PRI_jet_subleading_eta: -999.0,
      PRI_jet_subleading_phi: -999.0,
    };

    const objs = getRenderablePhysicsObjects(features);
    const jetObjs = objs.filter((o) => o.type.includes('jet'));
    expect(jetObjs).toHaveLength(0);
    expect(objs.map((o) => o.id)).toEqual(['tau', 'lepton', 'met']);
  });

  it('renders leading jet only when PRI_jet_num == 1', () => {
    const features: Record<string, number> = {
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
      PRI_jet_subleading_pt: -999.0,
      PRI_jet_subleading_eta: -999.0,
      PRI_jet_subleading_phi: -999.0,
    };

    const objs = getRenderablePhysicsObjects(features);
    const jetObjs = objs.filter((o) => o.type.includes('jet'));
    expect(jetObjs).toHaveLength(1);
    expect(jetObjs[0].id).toBe('leading_jet');
    expect(jetObjs[0].pt).toBe(85.4);
  });

  it('renders leading and subleading jets when PRI_jet_num >= 2', () => {
    const features: Record<string, number> = {
      PRI_tau_pt: 45.2,
      PRI_tau_eta: 0.5,
      PRI_tau_phi: 1.2,
      PRI_lep_pt: 32.1,
      PRI_lep_eta: -0.8,
      PRI_lep_phi: -2.1,
      PRI_met: 50.0,
      PRI_met_phi: 0.1,
      PRI_jet_num: 2,
      PRI_jet_leading_pt: 110.2,
      PRI_jet_leading_eta: 1.1,
      PRI_jet_leading_phi: -0.5,
      PRI_jet_subleading_pt: 65.8,
      PRI_jet_subleading_eta: -1.4,
      PRI_jet_subleading_phi: 2.2,
    };

    const objs = getRenderablePhysicsObjects(features);
    const jetObjs = objs.filter((o) => o.type.includes('jet'));
    expect(jetObjs).toHaveLength(2);
    expect(jetObjs[0].id).toBe('leading_jet');
    expect(jetObjs[1].id).toBe('subleading_jet');
  });

  it('never creates an object for sentinel -999.0 feature values', () => {
    const features: Record<string, number> = {
      PRI_tau_pt: -999.0,
      PRI_lep_pt: -999.0,
      PRI_met: -999.0,
      PRI_jet_num: 0,
    };

    const objs = getRenderablePhysicsObjects(features);
    expect(objs).toHaveLength(0);
  });
});
