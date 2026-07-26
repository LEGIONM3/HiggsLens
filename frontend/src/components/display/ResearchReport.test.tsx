import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import { ResearchReportModal } from './ResearchReportModal';
import { ReproducibilitySection } from '../leaderboard/ReproducibilitySection';

const mockReport = {
  report_version: '1.0',
  generated_at: '2026-07-26T22:00:00Z',
  event: {
    event_id: 300000,
    features: { DER_mass_MMC: 15.97, PRI_tau_pt: 32.21 },
    source_split: 'test',
    gallery: { category: 'background', rank: 1, selection_method: 'top_probability' },
  },
  classification: {
    model_id: 'xgboost',
    signal_probability: 0.1234,
    predicted_label: 'background',
    threshold: 0.8118,
  },
  explanation: {
    base_value: -1.23,
    margin: -2.45,
    attributions: [
      { feature: 'DER_mass_MMC', value: 15.97, contribution: -0.92 },
      { feature: 'PRI_tau_pt', value: 32.21, contribution: 0.45 },
    ],
    object_groups: [
      { group: 'tau', total_abs_contribution: 0.45, signed_contribution: 0.45 },
    ],
  },
  reproducibility: {
    report_contract_version: '1.0',
    dataset: {
      record: '328',
      doi: '10.7483/OPENDATA.ATLAS.ZBP2.M5T8',
      content_hash: '54242acf28a78ce303ea48bcf7002f0a44df08448271477e0a63331486c4f316',
    },
    inference_contract: {
      feature_count: 30,
      sentinel_value: -999.0,
      prediction_path: 'certified PredictionService',
      explanation_path: 'native XGBoost pred_contribs=True',
    },
    certified_models: [
      {
        model_id: 'xgboost',
        feature_schema_version: 'v1',
        device: 'CPU (scikit-learn)',
        training_run_origin: 'Official baseline',
        subsample_notes: 'Full dataset',
      },
    ],
    frozen_leaderboard_status: 'Certified & Frozen',
    server_capabilities: {
      supported_report_formats: ['json', 'html'],
      retraining_supported: false,
      public_holdout_access: false,
      champion_model_id: 'xgboost',
    },
  },
  provenance: {
    statement: 'ATLAS open data (record 328, DOI 10.7483/OPENDATA.ATLAS.ZBP2.M5T8) — official ATLAS simulated events, classified by certified pre-trained models.',
  },
};

describe('ResearchReportModal & Reproducibility Frontend Controls', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('1. Renders nothing when isOpen is false', () => {
    const { container } = render(
      <ResearchReportModal eventId={300000} isOpen={false} onClose={vi.fn()} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('2. Fetches report on demand and renders structured preview when opened', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockReport,
    } as Response);

    render(<ResearchReportModal eventId={300000} isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('📄 Research Report — Event #300000')).toBeDefined();
      expect(screen.getByText('BACKGROUND')).toBeDefined();
      expect(screen.getByText('0.1234')).toBeDefined();
      expect(screen.getByText('DER_mass_MMC')).toBeDefined();
    });
  });

  it('3. Download JSON button triggers blob download with expected filename', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockReport,
    } as Response);

    const createObjectURLMock = vi.fn().mockReturnValue('blob:test-url');
    const revokeObjectURLMock = vi.fn();
    globalThis.URL.createObjectURL = createObjectURLMock;
    globalThis.URL.revokeObjectURL = revokeObjectURLMock;

    render(<ResearchReportModal eventId={300000} isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('📄 Research Report — Event #300000')).toBeDefined();
    });

    const downloadBtn = screen.getByLabelText('Download JSON Report');
    fireEvent.click(downloadBtn);

    expect(createObjectURLMock).toHaveBeenCalled();
  });

  it('4. Print button opens canonical HTML report URL directly on user gesture', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockReport,
    } as Response);

    const windowOpenSpy = vi.spyOn(window, 'open').mockImplementation(() => null);

    render(<ResearchReportModal eventId={300000} isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('📄 Research Report — Event #300000')).toBeDefined();
    });

    const printBtn = screen.getByLabelText('Print or Save PDF in Browser');
    fireEvent.click(printBtn);

    expect(windowOpenSpy).toHaveBeenCalledWith('/api/v1/events/300000/report?format=html', '_blank');
  });

  it('5. ReproducibilitySection fetches manifest on expand and renders dynamic facts', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockReport.reproducibility,
    } as Response);

    render(<ReproducibilitySection />);

    const toggleBtn = screen.getByLabelText('Toggle Reproducibility Details Section');
    fireEvent.click(toggleBtn);

    await waitFor(() => {
      expect(screen.getByText('328')).toBeDefined();
      expect(screen.getByText('10.7483/OPENDATA.ATLAS.ZBP2.M5T8')).toBeDefined();
      expect(screen.getByText('Certified & Frozen')).toBeDefined();
    });
  });
});
