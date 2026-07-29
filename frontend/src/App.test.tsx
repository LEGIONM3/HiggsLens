import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { App } from './App';
import { CernEventObservatory } from './observatory/components/CernEventObservatory';

// Polyfill ResizeObserver & Canvas for JSDOM canvas rendering
if (typeof window !== 'undefined' && !window.ResizeObserver) {
  window.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

describe('HIGGSLENS EVENT OBSERVATORY Single-Page Experience', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it('1. Renders single-page header, title, and mandatory educational disclaimer', async () => {
    render(<App />);

    expect(screen.getByText('HIGGSLENS EVENT OBSERVATORY')).toBeDefined();
    expect(
      screen.getByText('CERN/ATLAS open-data event visualization + frozen ML classification')
    ).toBeDefined();
    expect(
      screen.getByText(/Illustrative collision animation. Real event classification uses CERN\/ATLAS open-data kinematics/i)
    ).toBeDefined();
    expect(screen.getByText(/CERN Record 328 \(DOI 10.7483\)/i)).toBeDefined();
  });

  it('2. Start / Pause / Restart controls function', async () => {
    render(<CernEventObservatory initialEventId={100001} />);

    // Initial state is playing => displays Pause
    const pauseBtn = screen.getByText('Pause');
    expect(pauseBtn).toBeDefined();
    fireEvent.click(pauseBtn);

    // After pause => displays Start
    const startBtn = screen.getByText('Start');
    expect(startBtn).toBeDefined();

    const restartBtn = screen.getByText('Restart');
    expect(restartBtn).toBeDefined();
    fireEvent.click(restartBtn);
  });

  it('3. Speed mode switch: L Speed vs Viewable', async () => {
    render(<CernEventObservatory initialEventId={100001} />);

    const lSpeedBtn = screen.getByRole('button', { name: 'L Speed' });
    fireEvent.click(lSpeedBtn);
    expect(screen.getByText(/L Speed: compressed educational visualization/i)).toBeDefined();

    const viewableBtn = screen.getByRole('button', { name: 'Viewable' });
    fireEvent.click(viewableBtn);
    expect(screen.queryByText(/L Speed: compressed educational visualization/i)).toBeNull();
  });

  it('4. Viewable speed buttons (0.2x, 0.5x, 1x, 2x) display and select', async () => {
    render(<CernEventObservatory initialEventId={100001} />);

    const btn02 = screen.getByRole('button', { name: '0.2x' });
    const btn05 = screen.getByRole('button', { name: '0.5x' });
    const btn1 = screen.getByRole('button', { name: '1x' });
    const btn2 = screen.getByRole('button', { name: '2x' });

    expect(btn02).toBeDefined();
    expect(btn05).toBeDefined();
    expect(btn1).toBeDefined();
    expect(btn2).toBeDefined();

    fireEvent.click(btn2);
    expect(btn2.className).toContain('text-cyan-300');
  });

  it('5. Zoom In and Zoom Out buttons clamp bounds', async () => {
    render(<CernEventObservatory initialEventId={100001} />);

    const zoomInBtn = screen.getByRole('button', { name: /Zoom In/i });
    const zoomOutBtn = screen.getByRole('button', { name: /Zoom Out/i });

    expect(zoomInBtn).toBeDefined();
    expect(zoomOutBtn).toBeDefined();

    // Click Zoom In multiple times
    fireEvent.click(zoomInBtn);
    fireEvent.click(zoomInBtn);
    expect(screen.getByText('14u')).toBeDefined();

    // Click Zoom Out
    fireEvent.click(zoomOutBtn);
    expect(screen.getByText('18u')).toBeDefined();
  });

  it('6. Namecards distinguish API-backed vs Illustrative particles', async () => {
    render(<CernEventObservatory initialEventId={100001} />);

    // Wait for event to load
    await waitFor(() => {
      expect(screen.getByText('Tau Candidate (τ)')).toBeDefined();
    });

    // Click Tau candidate vector label
    fireEvent.click(screen.getByText('Tau Candidate (τ)'));

    // Verify Namecard distinguishes API-backed event object
    expect(screen.getAllByText('API-backed event object').length).toBeGreaterThan(0);
    expect(screen.getByText('TRANSVERSE MOMENTUM (pT)')).toBeDefined();

    // Close modal
    fireEvent.click(screen.getByText('Close Namecard'));
  });

  it('7. /display/:eventId route renders the one-page Event Observatory', async () => {
    window.history.pushState({}, '', '/display/100002');
    render(<App />);

    expect(screen.getByText('HIGGSLENS EVENT OBSERVATORY')).toBeDefined();

    await waitFor(() => {
      const input = screen.getByRole('spinbutton') as HTMLInputElement;
      expect(input.value).toBe('100002');
    });
  });
});
