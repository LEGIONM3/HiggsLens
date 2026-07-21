import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import { DatasetCardComponent } from './components/DatasetCard';

describe('Frontend Components Test Suite', () => {
  it('renders DatasetCardComponent correctly with correct dataset and partition facts', () => {
    const mockStatus = {
      record_id: 328,
      doi: '10.7483/OPENDATA.ATLAS.ZBP2.M5T8',
      exists: true,
      file_size_bytes: 186500000,
      validation_report: null
    };

    render(
      <DatasetCardComponent
        status={mockStatus as any}
        onRefresh={() => {}}
      />
    );

    expect(screen.getByText(/Record 328/i)).toBeDefined();
    expect(screen.getByText(/10\.7483\/OPENDATA\.ATLAS\.ZBP2\.M5T8/i)).toBeDefined();
    expect(screen.getByText(/t: Train \| b: Val \| v: Test \| u: Holdout/i)).toBeDefined();
  });
});
