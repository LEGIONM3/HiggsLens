import { fireEvent, render, screen } from '@testing-library/react';
import React, { useEffect } from 'react';
import { describe, expect, it } from 'vitest';
import { EducationMode } from './EducationMode';
import { EducationProvider, useEducation } from '../../context/EducationContext';

const DrawerOpener: React.FC = () => {
  const { setIsOpen } = useEducation();
  useEffect(() => {
    setIsOpen(true);
  }, [setIsOpen]);
  return null;
};

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <EducationProvider>
    <DrawerOpener />
    {children}
  </EducationProvider>
);

describe('EducationMode Component', () => {
  it('renders Level 1 Beginner content and verbatim provenance sentence when opened', () => {
    render(
      <TestWrapper>
        <EducationMode />
      </TestWrapper>
    );

    expect(screen.getByText('Physics Education Mode')).toBeDefined();
    expect(screen.getByText(/Level 1: What am I looking at\?/i)).toBeDefined();
    expect(
      screen.getByText(/ATLAS open data \(record 328, DOI 10.7483\/OPENDATA.ATLAS.ZBP2.M5T8\) — official ATLAS simulated events, classified by certified pre-trained models./i)
    ).toBeDefined();
  });

  it('switches to Level 2 Intermediate and displays verbatim honesty note', () => {
    render(
      <TestWrapper>
        <EducationMode signalProbability={0.92} threshold={0.8118} />
      </TestWrapper>
    );

    const level2Btn = screen.getByRole('button', { name: 'Level 2' });
    fireEvent.click(level2Btn);

    expect(screen.getByText(/Level 2: How does the model decide\?/i)).toBeDefined();
    expect(
      screen.getByText(/Feature attributions describe how the model reached its score. They are not statements of physical causation./i)
    ).toBeDefined();
  });

  it('switches to Level 3 Advanced and displays physical units table', () => {
    render(
      <TestWrapper>
        <EducationMode />
      </TestWrapper>
    );

    const level3Btn = screen.getByRole('button', { name: 'Level 3' });
    fireEvent.click(level3Btn);

    expect(screen.getByText(/Level 3: Show me the physics/i)).toBeDefined();
    expect(screen.getByText('Physics Units Standard')).toBeDefined();
    expect(screen.getByText(/Quantum ML models \(qml_vqc, qml_qaoa\) are experimental research benchmarks/i)).toBeDefined();
  });
});
