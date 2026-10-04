/**
 * Web Worker for Strategy Lab Scorecard & Walk-Forward Simulations
 * Offloads compute-heavy walk-forward backtests, parametric MC, and bootstrap MC.
 */
import { runStrategyScorecard, StrategyScorecardOptions } from './backtest.ts';

self.onmessage = (event: MessageEvent<StrategyScorecardOptions>) => {
  try {
    const options = event.data;
    const result = runStrategyScorecard({
      ...options,
      onProgress: (strategyId: string, doneCount: number, total: number) => {
        self.postMessage({ type: 'progress', strategyId, doneCount, total });
      },
    });
    self.postMessage({ type: 'complete', result });
  } catch (error: any) {
    self.postMessage({ error: error?.message || 'Strategy Lab worker execution failed' });
  }
};
