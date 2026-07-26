import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { GalleryView } from './GalleryView';

const mockGalleryData = {
  events: [
    {
      event_id: 300000,
      features: { DER_mass_MMC: 138.4, DER_met_pt: 45.2 },
      signal_probability: 0.95,
      predicted_label: 'signal',
      threshold: 0.8118,
      gallery_category: 'signal',
      gallery_rank: 1,
      selection_method: 'top_probability',
    },
    {
      event_id: 300001,
      features: { DER_mass_MMC: 80.1, DER_met_pt: 12.5 },
      signal_probability: 0.12,
      predicted_label: 'background',
      threshold: 0.8118,
      gallery_category: 'background',
      gallery_rank: 1,
      selection_method: 'top_probability',
    },
  ],
  total_count: 2,
  categories: { signal: 1, background: 1, interesting: 0 },
  selection_method: 'top_probability',
};

describe('GalleryView Component', () => {
  it('renders loading skeleton initially and fetches gallery data', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockGalleryData,
    }));

    const onSelectMock = vi.fn();
    render(<GalleryView onSelectEventForDisplay={onSelectMock} />);

    await waitFor(() => {
      expect(screen.getAllByText(/300000/)[0]).toBeDefined();
    });

    expect(screen.getAllByText(/300000/)[0]).toBeDefined();
    expect(screen.getAllByText(/300001/)[0]).toBeDefined();
  });

  it('triggers onSelectEventForDisplay when View in 3D button is clicked', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockGalleryData,
    }));

    const onSelectMock = vi.fn();
    render(<GalleryView onSelectEventForDisplay={onSelectMock} />);

    await waitFor(() => {
      expect(screen.getAllByText(/300000/)[0]).toBeDefined();
    });

    const buttons = screen.getAllByRole('button', { name: /View Event in 3D/i });
    fireEvent.click(buttons[0]);

    expect(onSelectMock).toHaveBeenCalledWith(300000);
  });
});
