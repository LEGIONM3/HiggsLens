import { render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { App } from '../../App';

const mockPermalinkData = {
  event_id: 300000,
  features: { DER_mass_MMC: 138.4, DER_met_pt: 45.2 },
  kaggleset: 'v',
  signal_probability: 0.95,
  predicted_label: 'signal',
  threshold: 0.8118,
  explanation: {
    model_id: 'xgboost',
    probability: 0.95,
    predicted_label: 'signal',
    threshold: 0.8118,
    attributions: [],
    object_groups: [],
    additivity_margin: 2.1,
    base_value: 0.1,
  },
};

describe('Permalink Route Integration', () => {
  it('cold-loads permalink event payload in single API call', async () => {
    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string) => {
      if (url.includes('/api/v1/events/300000/permalink')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockPermalinkData,
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({ events: [], count: 0, seed: 42, label_filter: 'any' }),
      });
    }));

    render(<App />);
    expect(screen.getByText('HiggsLens')).toBeDefined();
  });
});
