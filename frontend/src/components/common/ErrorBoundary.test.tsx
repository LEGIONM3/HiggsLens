import { render, screen } from '@testing-library/react';
import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { ErrorBoundary } from './ErrorBoundary';

const ProblemChild: React.FC = () => {
  throw new Error("Test component crash!");
};

describe('ErrorBoundary', () => {
  it('renders fallback UI when child component throws an error', () => {
    // Suppress console.error during test
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
      <ErrorBoundary fallbackTitle="Custom Fallback Title">
        <ProblemChild />
      </ErrorBoundary>
    );

    expect(screen.getByText('Custom Fallback Title')).toBeDefined();
    expect(screen.getByText('Test component crash!')).toBeDefined();

    spy.mockRestore();
  });
});
