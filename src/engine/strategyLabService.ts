/**
 * Strategy Lab Async Service with Dedicated Worker Execution
 * Keeps heavy 11-strategy walk-forward backtests off the main thread.
 * No main-thread fallback: errors reject cleanly so UI displays error message.
 */
import { StrategyScorecardResult, StrategyScorecardOptions } from './types.ts';

export async function runStrategyLabScorecardAsync(
  options: StrategyScorecardOptions,
  onProgress?: (pct: number) => void
): Promise<StrategyScorecardResult> {
  if (typeof window !== 'undefined' && typeof Worker !== 'undefined') {
    return new Promise<StrategyScorecardResult>((resolve, reject) => {
      let progressTimer: any;
      let cur = 10;
      if (onProgress) {
        onProgress(cur);
        progressTimer = setInterval(() => {
          cur = Math.min(92, cur + Math.floor(Math.random() * 12) + 4);
          onProgress(cur);
        }, 300);
      }
      try {
        const worker = new Worker(new URL('./strategyLab.worker.ts', import.meta.url), {
          type: 'module',
        });
        worker.onmessage = (event: MessageEvent) => {
          if (event.data?.type === 'progress') {
            const { doneCount, total } = event.data;
            if (onProgress && total > 0) {
              onProgress(Math.round((doneCount / total) * 100));
            }
            return;
          }
          clearInterval(progressTimer);
          worker.terminate();
          if (onProgress) onProgress(100);
          if (event.data?.error) {
            reject(new Error(event.data.error));
          } else {
            resolve(event.data.result);
          }
        };
        worker.onerror = (err) => {
          clearInterval(progressTimer);
          worker.terminate();
          if (onProgress) onProgress(100);
          reject(new Error(err.message || 'Strategy Lab worker execution failed'));
        };
        worker.postMessage(options);
      } catch (err: any) {
        clearInterval(progressTimer);
        if (onProgress) onProgress(100);
        reject(new Error(err?.message || 'Failed to initialize Strategy Lab worker'));
      }
    });
  }

  throw new Error('Web Workers are not supported in this environment.');
}
