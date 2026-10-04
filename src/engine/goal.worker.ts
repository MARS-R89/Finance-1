/**
 * Web Worker for Financial Goal Solver
 * Offloads parametric MC & bootstrap bisection iterations from the main thread.
 */
import { solveGoal } from './backtest.ts';
import { GoalSolverInput } from './types.ts';

interface GoalWorkerMessage {
  requestId: number;
  input: GoalSolverInput;
}

self.onmessage = (event: MessageEvent<GoalWorkerMessage>) => {
  const { requestId, input } = event.data;
  try {
    const result = solveGoal(input);
    self.postMessage({ requestId, result });
  } catch (error: any) {
    self.postMessage({
      requestId,
      error: error?.message || 'Goal solver worker execution failed',
    });
  }
};
