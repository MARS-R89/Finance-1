/**
 * Goal Solver Web Worker Service
 * Offloads solveGoal to a Web Worker, terminates stale instances on new runs,
 * and guards with requestIds to prevent race conditions.
 * No main-thread fallback: errors reject cleanly so UI can display them.
 */
import { GoalSolverInput, GoalSolverResult } from './types.ts';

let activeWorker: Worker | null = null;
let currentRequestId = 0;

export function runGoalSolverAsync(input: GoalSolverInput): Promise<GoalSolverResult> {
  return new Promise<GoalSolverResult>((resolve, reject) => {
    // Terminate the previous worker when a new run starts
    if (activeWorker) {
      activeWorker.terminate();
      activeWorker = null;
    }

    if (typeof window === 'undefined' || typeof Worker === 'undefined') {
      reject(new Error('Web Workers are not supported in this environment.'));
      return;
    }

    const requestId = ++currentRequestId;

    try {
      const worker = new Worker(new URL('./goal.worker.ts', import.meta.url), {
        type: 'module',
      });
      activeWorker = worker;

      worker.onmessage = (event: MessageEvent) => {
        const data = event.data;
        // Ignore response if request id is older
        if (!data || data.requestId !== requestId) {
          return;
        }

        if (activeWorker === worker) {
          activeWorker = null;
        }
        worker.terminate();

        if (data.error) {
          reject(new Error(data.error));
        } else {
          resolve(data.result);
        }
      };

      worker.onerror = (err) => {
        if (activeWorker === worker) {
          activeWorker = null;
        }
        worker.terminate();
        reject(new Error(err.message || 'Goal solver worker failed unexpectedly.'));
      };

      worker.postMessage({ requestId, input });
    } catch (err: any) {
      if (activeWorker) {
        activeWorker.terminate();
        activeWorker = null;
      }
      reject(new Error(err?.message || 'Failed to initialize goal solver worker.'));
    }
  });
}

export function terminateGoalSolver(): void {
  if (activeWorker) {
    activeWorker.terminate();
    activeWorker = null;
  }
}
