import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { App } from './App';

// Polyfill ResizeObserver for JSDOM R3F Canvas rendering
if (typeof window !== 'undefined' && !window.ResizeObserver) {
  window.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

describe('HiggsLens Observatory App Integration', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('renders Observatory branding, header telemetry, and navigation tabs', () => {
    render(<App />);

    expect(screen.getByText('HiggsLens')).toBeDefined();
    expect(screen.getByText('CERN OPEN DATA 328')).toBeDefined();
    expect(screen.getByRole('tablist', { name: /Observatory Main Views/i })).toBeDefined();
  });
});
