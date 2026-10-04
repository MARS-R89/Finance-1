/**
 * Optimizer Web Worker Service
 * Offloads maximizeSharpe, maximizeReturn, and computeEfficientFrontier to a Web Worker,
 * terminates stale worker instances on new runs, and guards with requestIds to prevent race conditions.
 * No main-thread fallback: errors reject cleanly so UI can display them.
 */
import { OptimizerPoint, OptimizerConstraints, OptimizableAsset } from './types.ts';

export interface OptimizerRunInput {
  optMode: 'max_sharpe' | 'max_return';
  assets: OptimizableAsset[];
  covarianceMatrix: number[][];
  constraints: OptimizerConstraints;
  frontierPointsCount?: number;
}

export interface OptimizerRunResult {
  optimalResult: OptimizerPoint;
  frontierPoints: OptimizerPoint[];
}

let activeWorker: Worker | null = null;
let currentRequestId = 0;

export function runOptimizerAsync(input: OptimizerRunInput): Promise<OptimizerRunResult> {
  return new Promise<OptimizerRunResult>((resolve, reject) => {
    // Terminate any previous worker when a new run starts
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
      const worker = new Worker(new URL('./optimizer.worker.ts', import.meta.url), {
        type: 'module',
      });
      activeWorker = worker;

      worker.onmessage = (event: MessageEvent) => {
        const data = event.data;
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
        reject(new Error(err.message || 'Optimizer worker failed unexpectedly.'));
      };

      worker.postMessage({ requestId, input });
    } catch (err: any) {
      if (activeWorker) {
        activeWorker.terminate();
        activeWorker = null;
      }
      reject(new Error(err?.message || 'Failed to initialize optimizer worker.'));
    }
  });
}

export function terminateOptimizer(): void {
  if (activeWorker) {
    activeWorker.terminate();
    activeWorker = null;
  }
}
