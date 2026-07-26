/**
 * Pure TypeScript state machine for Accelerator Journey flow.
 * Zero Three.js or DOM dependencies.
 */

export type JourneyState =
  | 'idle'
  | 'injecting'
  | 'accelerating'
  | 'colliding'
  | 'zooming'
  | 'displaying';

export type JourneyEventType =
  | 'START_AUTO_RUN'
  | 'INJECTION_COMPLETE'
  | 'ACCELERATION_COMPLETE'
  | 'COLLISION_COMPLETE'
  | 'ZOOM_COMPLETE'
  | 'RESET';

export interface JourneyEvent {
  type: JourneyEventType;
}

const ALLOWED_TRANSITIONS: Record<JourneyState, Partial<Record<JourneyEventType, JourneyState>>> = {
  idle: {
    START_AUTO_RUN: 'injecting',
    RESET: 'idle',
  },
  injecting: {
    INJECTION_COMPLETE: 'accelerating',
    RESET: 'idle',
  },
  accelerating: {
    ACCELERATION_COMPLETE: 'colliding',
    RESET: 'idle',
  },
  colliding: {
    COLLISION_COMPLETE: 'zooming',
    RESET: 'idle',
  },
  zooming: {
    ZOOM_COMPLETE: 'displaying',
    RESET: 'idle',
  },
  displaying: {
    START_AUTO_RUN: 'injecting',
    RESET: 'idle',
  },
};

export class JourneyStateMachine {
  private currentState: JourneyState;

  constructor(initialState: JourneyState = 'idle') {
    this.currentState = initialState;
  }

  public getState(): JourneyState {
    return this.currentState;
  }

  public canTransition(event: JourneyEventType): boolean {
    const transitions = ALLOWED_TRANSITIONS[this.currentState];
    return transitions[event] !== undefined;
  }

  public transition(event: JourneyEventType): JourneyState {
    const nextState = ALLOWED_TRANSITIONS[this.currentState]?.[event];
    if (!nextState) {
      throw new Error(
        `Invalid journey transition '${event}' from current state '${this.currentState}'.`
      );
    }
    this.currentState = nextState;
    return this.currentState;
  }

  public reset(): JourneyState {
    this.currentState = 'idle';
    return this.currentState;
  }
}
