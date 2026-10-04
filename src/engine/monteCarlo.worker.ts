/**
 * Web Worker for Monte Carlo Simulation Execution
 * Keeps computation off the React main thread for silky smooth UI responsiveness.
 */
import { runMonteCarloSimulation } from './monteCarlo.ts';
import { MonteCarloConfig } from './types.ts';

self.onmessage = (event: MessageEvent<MonteCarloConfig>) => {
  try {
    const config = event.data;
    const result = runMonteCarloSimulation(config);
    self.postMessage({ result });
  } catch (error: any) {
    self.postMessage({ error: error?.message || 'Monte Carlo worker execution failed' });
  }
};
