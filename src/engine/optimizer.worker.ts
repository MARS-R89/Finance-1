/**
 * Web Worker for Portfolio Optimizer & Efficient Frontier
 * Offloads maximizeSharpe, maximizeReturn, and computeEfficientFrontier from the main thread.
 */
import { maximizeReturn, maximizeSharpe, computeEfficientFrontier } from './optimizer.ts';
import { OptimizerRunInput } from './optimizerService.ts';

interface OptimizerWorkerMessage {
  requestId: number;
  input: OptimizerRunInput;
}

self.onmessage = (event: MessageEvent<OptimizerWorkerMessage>) => {
  const { requestId, input } = event.data;
  try {
    const { optMode, assets, covarianceMatrix, constraints, frontierPointsCount = 30 } = input;
    const optimalResult =
      optMode === 'max_return'
        ? maximizeReturn(assets, covarianceMatrix, constraints)
        : maximizeSharpe(assets, covarianceMatrix, constraints);
    const frontierPoints = computeEfficientFrontier(
      assets,
      covarianceMatrix,
      frontierPointsCount,
      constraints
    );
    self.postMessage({ requestId, result: { optimalResult, frontierPoints } });
  } catch (error: any) {
    self.postMessage({
      requestId,
      error: error?.message || 'Optimizer worker execution failed',
    });
  }
};
