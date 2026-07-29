import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ObservatoryShell } from '../ObservatoryShell';

describe('ObservatoryShell & Shell Controls', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it('renders observatory header, navigation tabs, and default journey experience for first-time visitors', () => {
    render(
      <ObservatoryShell>
        {(currentMode) => <div data-testid="content">{currentMode}</div>}
      </ObservatoryShell>
    );

    expect(screen.getByText('HiggsLens')).toBeDefined();
    expect(screen.getByText('CERN OPEN DATA 328')).toBeDefined();
    expect(screen.getByRole('tablist', { name: /Observatory Main Views/i })).toBeDefined();

    // Verify first-time visitor default is 'journey'
    expect(screen.getByTestId('content').textContent).toBe('journey');

    // Disclosure badge check for Journey
    expect(
      screen.getByText(/Illustrative accelerator journey. HiggsLens next displays a recorded ATLAS open-data event./i)
    ).toBeDefined();

    // "Skip to Event Studio" button check
    expect(screen.getByRole('button', { name: /Skip Accelerator Journey and open Event Studio directly/i })).toBeDefined();
  });

  it('navigates to Event Studio when "Skip to Event Studio" button is clicked and persists to localStorage', () => {
    render(
      <ObservatoryShell>
        {(currentMode) => <div data-testid="content">{currentMode}</div>}
      </ObservatoryShell>
    );

    const skipButton = screen.getByRole('button', { name: /Skip Accelerator Journey and open Event Studio directly/i });
    fireEvent.click(skipButton);

    expect(screen.getByTestId('content').textContent).toBe('studio');
    expect(localStorage.getItem('higgslens_preferred_mode')).toBe('studio');

    // Experience B disclosure should be present
    expect(
      screen.getByText(/Detector-inspired illustrative geometry — not to scale. Recorded event kinematics supplied by API;/i)
    ).toBeDefined();
  });

  it('restores visitor preferred mode from localStorage on initial load', () => {
    localStorage.setItem('higgslens_preferred_mode', 'leaderboard');

    render(
      <ObservatoryShell>
        {(currentMode) => <div data-testid="content">{currentMode}</div>}
      </ObservatoryShell>
    );

    expect(screen.getByTestId('content').textContent).toBe('leaderboard');
  });

  it('supports tab navigation clicks and updates tab accessibility states', () => {
    render(
      <ObservatoryShell>
        {(currentMode) => <div data-testid="content">{currentMode}</div>}
      </ObservatoryShell>
    );

    const leaderboardTab = screen.getByRole('tab', { name: /Navigate to Official Model Leaderboard/i });
    expect(leaderboardTab.getAttribute('aria-selected')).toBe('false');

    fireEvent.click(leaderboardTab);

    expect(leaderboardTab.getAttribute('aria-selected')).toBe('true');
    expect(screen.getByTestId('content').textContent).toBe('leaderboard');
  });

  it('handles number key shortcuts (1-7) to quickly switch views when focus is outside inputs', () => {
    render(
      <ObservatoryShell>
        {(currentMode) => <div data-testid="content">{currentMode}</div>}
      </ObservatoryShell>
    );

    fireEvent.keyDown(window, { key: '2' });
    expect(screen.getByTestId('content').textContent).toBe('studio');

    fireEvent.keyDown(window, { key: '4' });
    expect(screen.getByTestId('content').textContent).toBe('pipeline');

    fireEvent.keyDown(window, { key: '7' });
    expect(screen.getByTestId('content').textContent).toBe('lab');
  });
});
