import { describe, it, expect } from 'vitest';
import { JourneyStateMachine } from './journeyState';

describe('JourneyStateMachine', () => {
  it('should initialize in idle state', () => {
    const sm = new JourneyStateMachine();
    expect(sm.getState()).toBe('idle');
  });

  it('should follow happy path auto-run transition sequence', () => {
    const sm = new JourneyStateMachine();

    expect(sm.canTransition('START_AUTO_RUN')).toBe(true);
    expect(sm.transition('START_AUTO_RUN')).toBe('injecting');

    expect(sm.canTransition('INJECTION_COMPLETE')).toBe(true);
    expect(sm.transition('INJECTION_COMPLETE')).toBe('accelerating');

    expect(sm.canTransition('ACCELERATION_COMPLETE')).toBe(true);
    expect(sm.transition('ACCELERATION_COMPLETE')).toBe('colliding');

    expect(sm.canTransition('COLLISION_COMPLETE')).toBe(true);
    expect(sm.transition('COLLISION_COMPLETE')).toBe('zooming');

    expect(sm.canTransition('ZOOM_COMPLETE')).toBe(true);
    expect(sm.transition('ZOOM_COMPLETE')).toBe('displaying');
  });

  it('should reject invalid transition attempts', () => {
    const sm = new JourneyStateMachine();

    expect(sm.canTransition('INJECTION_COMPLETE')).toBe(false);
    expect(() => sm.transition('INJECTION_COMPLETE')).toThrowError(
      /Invalid journey transition/
    );
  });

  it('should allow RESET from any active state', () => {
    const sm = new JourneyStateMachine('accelerating');
    expect(sm.canTransition('RESET')).toBe(true);
    expect(sm.transition('RESET')).toBe('idle');
  });
});
