import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ObservatoryShell } from '../ObservatoryShell';

describe('ObservatoryShell & Navigation Controls', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it('renders observatory header and default journey mode for first-time visitors', () => {
    render(
      <ObservatoryShell>
        {(currentMode) => <div data-testid="content">{currentMode}</div>}
      </ObservatoryShell>
    );

    expect(screen.getByText('HiggsLens')).toBeDefined();
    expect(screen.getByText('CERN OPEN DATA 328')).toBeDefined();
    expect(screen.getByTestId('content').textContent).toBe('journey');
    expect(
      screen.getByText(/Illustrative accelerator journey. HiggsLens next displays a recorded ATLAS open-data event./i)
    ).toBeDefined();
  });

  it('switches to Event Studio when "Skip to Event Studio" button is clicked and persists to localStorage', () => {
    render(
      <ObservatoryShell>
        {(currentMode) => <div data-testid="content">{currentMode}</div>}
      </ObservatoryShell>
    );

    const skipButton = screen.getByRole('button', { name: /Skip Accelerator Journey and open Event Studio directly/i });
    fireEvent.click(skipButton);

    expect(screen.getByTestId('content').textContent).toBe('studio');
    expect(localStorage.getItem('higgslens_preferred_mode')).toBe('studio');
  });

  it('restores preferred mode from localStorage on initial render', () => {
    localStorage.setItem('higgslens_preferred_mode', 'leaderboard');

    render(
      <ObservatoryShell>
        {(currentMode) => <div data-testid="content">{currentMode}</div>}
      </ObservatoryShell>
    );

    expect(screen.getByTestId('content').textContent).toBe('leaderboard');
  });
});
