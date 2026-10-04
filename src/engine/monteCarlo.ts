/**
 * Monte Carlo Simulation Engine (Geometric Brownian Motion with Correlated Returns)
 * Uses Cholesky Decomposition of the Covariance Matrix, stochastic default events for Bonds/FD,
 * and extracts annual percentile bands (P5-P95), probability of loss, and inflation-beating stats.
 */
import {
  MonteCarloConfig,
  MonteCarloResult,
  MonteCarloPercentilePoint,
  MonteCarloSummaryStats,
} from './types.ts';
import { analyzeBond } from './bonds.ts';
import { normalizeAllocation, calculateInflationTarget } from './projections.ts';
import { FALLBACK_ASSET_HISTORY } from './fallbackData.ts';
import { createMulberry32 } from './prng.ts';
import { historicalBootstrap } from './backtest.ts';

/**
 * Computes the lower-triangular Cholesky factor L such that L * L^T = Sigma.
 * Adds a small diagonal ridge epsilon for positive-definite numerical stability.
 */
export function choleskyDecomposition(matrix: number[][], epsilon = 1e-7): number[][] {
  const n = matrix.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += L[i][k] * L[j][k];
      }
      if (i === j) {
        // Diagonal element: ensure positive radicand with regularization
        const diagVal = matrix[i][i] + epsilon - sum;
        L[i][j] = Math.sqrt(Math.max(1e-12, diagVal));
      } else {
        // Off-diagonal element
        L[i][j] = (matrix[i][j] - sum) / L[j][j];
      }
    }
  }
  return L;
}

/**
 * Standard Normal Random Variate Generator using Box-Muller transform.
 * Supports seeded PRNG for reproducible Monte Carlo simulations.
 */
export class FastNormalRNG {
  private hasSpare = false;
  private spare = 0;
  private uniformRng: () => number;

  constructor(rng?: () => number) {
    this.uniformRng = rng || Math.random;
  }

  public next(): number {
    if (this.hasSpare) {
      this.hasSpare = false;
      return this.spare;
    }
    let u = 0;
    let v = 0;
    let s = 0;
    do {
      u = this.uniformRng() * 2.0 - 1.0;
      v = this.uniformRng() * 2.0 - 1.0;
      s = u * u + v * v;
    } while (s >= 1.0 || s === 0);
    const mul = Math.sqrt((-2.0 * Math.log(s)) / s);
    this.spare = v * mul;
    this.hasSpare = true;
    return u * mul;
  }
}

/**
 * Core pure function for Monte Carlo simulation.
 */
export function runMonteCarloSimulation(config: MonteCarloConfig): MonteCarloResult {
  if (config.model === 'bootstrap') {
    return historicalBootstrap(config);
  }

  const startTime = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const {
    allocation,
    years,
    assetStats,
    covarianceMatrix,
    assetIds,
    bondConfig,
    nSims = 5000,
    inflationRate = 0.06,
    seed,
    fatTails = false,
    computeStressComparison = true,
  } = config;

  const effectiveFatTails = config.model === 'student_t' ? true : fatTails;

  if (
    (assetIds.length > 0 || Object.keys(allocation).some((id) => id !== 'bonds_fd' && (allocation[id] ?? 0) > 0)) &&
    (!covarianceMatrix || covarianceMatrix.length !== assetIds.length)
  ) {
    throw new Error(
      `Dimension mismatch in runMonteCarloSimulation: covarianceMatrix dimension (${covarianceMatrix?.length ?? 0}) != assetIds.length (${assetIds?.length ?? 0})`
    );
  }

  if (years <= 0) throw new Error('Tenure must be at least 1 year.');

  // Initialize seeded PRNG (mulberry32) if seed is provided, else fallback to Math.random
  const uniformRng = seed !== undefined ? createMulberry32(seed) : Math.random;

  // Resolve capital for lumpsum and SIP
  let lumpsum = config.lumpsumAmount ?? 0;
  let sip = config.monthlySip ?? 0;

  // Backward compatibility with legacy config (amount + mode)
  if (config.lumpsumAmount === undefined && config.monthlySip === undefined) {
    const legacyAmt = config.amount ?? 0;
    if (config.mode === 'sip') {
      sip = legacyAmt;
      lumpsum = 0;
    } else {
      lumpsum = legacyAmt;
      sip = 0;
    }
  }

  if (lumpsum <= 0 && sip <= 0) {
    throw new Error('Investment amount must be positive.');
  }

  const weights = normalizeAllocation(allocation);
  const totalMonths = Math.round(years * 12);

  // Separate stochastic assets (stocks/MFs/gold) from deterministic/credit bond
  const activeStochasticIds = assetIds.filter(
    (id) => (weights[id] || 0) > 0 && id !== 'bonds_fd'
  );
  const numStochastic = activeStochasticIds.length;
  const bondWeight = weights['bonds_fd'] || 0;

  // Build sub-covariance matrix for active stochastic assets
  const activeIndices = activeStochasticIds.map((id) => assetIds.indexOf(id));
  const activeCovMatrix = (config.stressMode && config.bearCovarianceMatrix)
    ? config.bearCovarianceMatrix
    : covarianceMatrix;

  const subCovAnnual: number[][] = Array.from({ length: numStochastic }, () =>
    new Array(numStochastic).fill(0)
  );
  for (let i = 0; i < numStochastic; i++) {
    for (let j = 0; j < numStochastic; j++) {
      const origI = activeIndices[i];
      const origJ = activeIndices[j];
      let val = activeCovMatrix[origI]?.[origJ] ?? 0;
      if (config.stressMode && !config.bearCovarianceMatrix) {
        val *= 1.4; // 40% heightened variance under stress mode fallback
      }
      subCovAnnual[i][j] = val;
    }
  }

  // Monthly covariance: Sigma_monthly = Sigma_annual / 12
  const subCovMonthly: number[][] = Array.from({ length: numStochastic }, () =>
    new Array(numStochastic).fill(0)
  );
  for (let i = 0; i < numStochastic; i++) {
    for (let j = 0; j < numStochastic; j++) {
      subCovMonthly[i][j] = subCovAnnual[i][j] / 12;
    }
  }

  // Cholesky decomposition L of subCovMonthly
  const L = choleskyDecomposition(subCovMonthly);

  // Monthly drift: ln(1 + cagr) / 12 (with stress return haircut if enabled)
  const monthlyDrift: number[] = new Array(numStochastic).fill(0);
  const haircut = config.stressMode ? (config.bearReturnHaircut ?? 0.03) : 0;
  for (let i = 0; i < numStochastic; i++) {
    const id = activeStochasticIds[i];
    const rawCagr = assetStats[id]?.cagr ?? FALLBACK_ASSET_HISTORY[id]?.cagr ?? 0.12;
    const cagr = Math.max(-0.4, rawCagr - haircut);
    monthlyDrift[i] = Math.log(1 + cagr) / 12;
  }

  // Bond parameters
  const bondAnalysis = analyzeBond(
    bondConfig || {
      couponRate: 0.075,
      tenureYears: years,
      payoutFrequency: 'annual',
      creditRisk: { defaultProbabilityAnnual: 0.005, recoveryRate: 0.5 },
      reinvestmentRateAfterMaturity: 0.065,
    }
  );
  const monthlyBondYield = Math.pow(1 + bondAnalysis.ytm, 1 / 12) - 1;
  const annualDefaultProb = bondConfig?.creditRisk?.defaultProbabilityAnnual ?? 0.005;
  const recoveryRate = bondConfig?.creditRisk?.recoveryRate ?? 0.5;
  const monthlyDefaultProb = 1 - Math.pow(1 - annualDefaultProb, 1 / 12);
  const bondTenure = bondConfig?.tenureYears ?? years;
  const maturityMonth = Math.round(bondTenure * 12);
  const reinvestmentRate = bondConfig?.reinvestmentRateAfterMaturity ?? 0.065;
  const monthlyReinvestmentYield = Math.pow(1 + reinvestmentRate, 1 / 12) - 1;

  // Exact total invested calculation
  const totalInvested = lumpsum + sip * totalMonths;

  // Exact inflation target value calculation using shared helper
  const inflationTargetValue = calculateInflationTarget(lumpsum, sip, years, inflationRate);

  // Simulator helper function
  const runSimPaths = (useFatTails: boolean, simsCount: number, pathSeed?: number) => {
    const uRng = pathSeed !== undefined ? createMulberry32(pathSeed) : Math.random;
    const rng = new FastNormalRNG(uRng);
    const yearValues: number[][] = Array.from({ length: years }, () => new Array(simsCount).fill(0));
    const Z = new Float64Array(numStochastic);
    const W = new Float64Array(numStochastic);

    let lossCount = 0;
    let beatInflationCount = 0;
    let defaultsCount = 0;

    for (let s = 0; s < simsCount; s++) {
      const assetCapital = new Float64Array(numStochastic);
      let bondCapital = 0;
      let redeployedBondCapital = 0;
      let bondHasDefaulted = false;

      if (lumpsum > 0) {
        for (let i = 0; i < numStochastic; i++) {
          const id = activeStochasticIds[i];
          assetCapital[i] = lumpsum * weights[id];
        }
        bondCapital = lumpsum * bondWeight;
      }

      for (let m = 0; m < totalMonths; m++) {
        if (sip > 0) {
          for (let i = 0; i < numStochastic; i++) {
            const id = activeStochasticIds[i];
            assetCapital[i] += sip * weights[id];
          }
          if (bondHasDefaulted) {
            redeployedBondCapital += sip * bondWeight;
          } else {
            bondCapital += sip * bondWeight;
          }
        }

        // Shocks generation: standard normal vs Student-t(5) fat tails
        if (useFatTails) {
          for (let i = 0; i < numStochastic; i++) {
            const z = rng.next();
            const x1 = rng.next();
            const x2 = rng.next();
            const x3 = rng.next();
            const x4 = rng.next();
            const x5 = rng.next();
            const v = x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 + x5 * x5;
            Z[i] = z * Math.sqrt(3 / Math.max(1e-6, v));
          }
        } else {
          for (let i = 0; i < numStochastic; i++) {
            Z[i] = rng.next();
          }
        }

        for (let i = 0; i < numStochastic; i++) {
          let shock = 0;
          for (let k = 0; k <= i; k++) {
            shock += L[i][k] * Z[k];
          }
          W[i] = shock;
        }

        for (let i = 0; i < numStochastic; i++) {
          const returnFactor = Math.exp(monthlyDrift[i] + W[i]);
          assetCapital[i] *= returnFactor;
        }

        if (bondWeight > 0) {
          if (!bondHasDefaulted) {
            if (uRng() < monthlyDefaultProb) {
              bondCapital *= recoveryRate;
              bondHasDefaulted = true;
              defaultsCount++;
            } else {
              const currentYield = m < maturityMonth ? monthlyBondYield : monthlyReinvestmentYield;
              bondCapital *= 1 + currentYield;
            }
          }
          if (redeployedBondCapital > 0) {
            redeployedBondCapital *= 1 + monthlyReinvestmentYield;
          }
        }

        const monthNum = m + 1;
        if (monthNum % 12 === 0) {
          const yIndex = monthNum / 12 - 1;
          let totalVal = bondCapital + redeployedBondCapital;
          for (let i = 0; i < numStochastic; i++) {
            totalVal += assetCapital[i];
          }
          yearValues[yIndex][s] = totalVal;
        }
      }

      const terminalVal = yearValues[years - 1][s];
      if (terminalVal < totalInvested) {
        lossCount++;
      }
      if (terminalVal >= inflationTargetValue) {
        beatInflationCount++;
      }
    }

    const percentiles: MonteCarloPercentilePoint[] = [];
    const getPercentile = (sorted: number[], p: number) => {
      const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((p / 100) * sorted.length)));
      return sorted[idx];
    };

    for (let y = 0; y < years; y++) {
      const sorted = yearValues[y].slice().sort((a, b) => a - b);
      percentiles.push({
        year: y + 1,
        p5: Number(getPercentile(sorted, 5).toFixed(2)),
        p10: Number(getPercentile(sorted, 10).toFixed(2)),
        p25: Number(getPercentile(sorted, 25).toFixed(2)),
        p50: Number(getPercentile(sorted, 50).toFixed(2)),
        p75: Number(getPercentile(sorted, 75).toFixed(2)),
        p90: Number(getPercentile(sorted, 90).toFixed(2)),
        p95: Number(getPercentile(sorted, 95).toFixed(2)),
      });
    }

    const finalSorted = yearValues[years - 1].slice().sort((a, b) => a - b);
    const sumFinal = finalSorted.reduce((a, b) => a + b, 0);
    const meanFinal = sumFinal / simsCount;
    const varianceFinal =
      finalSorted.reduce((sum, v) => sum + Math.pow(v - meanFinal, 2), 0) / (simsCount - 1);
    const stdDevFinal = Math.sqrt(varianceFinal);

    const stats: MonteCarloSummaryStats = {
      min: Number(finalSorted[0].toFixed(2)),
      p25: Number(getPercentile(finalSorted, 25).toFixed(2)),
      median: Number(getPercentile(finalSorted, 50).toFixed(2)),
      p75: Number(getPercentile(finalSorted, 75).toFixed(2)),
      mean: Number(meanFinal.toFixed(2)),
      max: Number(finalSorted[finalSorted.length - 1].toFixed(2)),
      stdDev: Number(stdDevFinal.toFixed(2)),
    };

    const terminalValues = new Float64Array(finalSorted);

    return {
      percentiles,
      stats,
      probLoss: Number(((lossCount / simsCount) * 100).toFixed(2)),
      probInflation: Number(((beatInflationCount / simsCount) * 100).toFixed(2)),
      defaultsCount,
      terminalValues,
    };
  };

  const mainResult = runSimPaths(effectiveFatTails, nSims, seed);

  // Compute side-by-side normal vs stress-aware comparison if requested
  let stressComparison: MonteCarloResult['stressComparison'] = undefined;
  if (computeStressComparison) {
    if (effectiveFatTails) {
      const normalRun = runSimPaths(false, Math.min(2500, nSims), seed !== undefined ? seed + 101 : undefined);
      stressComparison = {
        normalP5: normalRun.percentiles[years - 1].p5,
        normalP10: normalRun.percentiles[years - 1].p10,
        normalP50: normalRun.percentiles[years - 1].p50,
        normalProbLoss: normalRun.probLoss,
        stressP5: mainResult.percentiles[years - 1].p5,
        stressP10: mainResult.percentiles[years - 1].p10,
        stressP50: mainResult.percentiles[years - 1].p50,
        stressProbLoss: mainResult.probLoss,
        isStressAware: true,
        isBearRegimeStress: config.stressMode || false,
      };
    } else {
      const stressRun = runSimPaths(true, Math.min(2500, nSims), seed !== undefined ? seed + 101 : undefined);
      stressComparison = {
        normalP5: mainResult.percentiles[years - 1].p5,
        normalP10: mainResult.percentiles[years - 1].p10,
        normalP50: mainResult.percentiles[years - 1].p50,
        normalProbLoss: mainResult.probLoss,
        stressP5: stressRun.percentiles[years - 1].p5,
        stressP10: stressRun.percentiles[years - 1].p10,
        stressP50: stressRun.percentiles[years - 1].p50,
        stressProbLoss: stressRun.probLoss,
        isStressAware: false,
        isBearRegimeStress: config.stressMode || false,
      };
    }
  }

  const endTime = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const termVals = mainResult.terminalValues;

  return {
    percentilesByYear: mainResult.percentiles,
    probabilityOfLoss: mainResult.probLoss,
    probabilityOfBeatingInflation: mainResult.probInflation,
    inflationRate,
    totalInvested: Number(totalInvested.toFixed(2)),
    terminalStats: mainResult.stats,
    nSims,
    simulationTimeMs: Number((endTime - startTime).toFixed(1)),
    defaultsTriggeredCount: mainResult.defaultsCount,
    stressComparison,
    terminalValues: termVals,
    probabilityAtLeast: (target: number) => {
      let count = 0;
      for (let i = 0; i < nSims; i++) {
        if (termVals[i] >= target) count++;
      }
      return Number(((count / nSims) * 100).toFixed(1));
    },
  };
}

interface ActiveWorkerEntry {
  worker: Worker;
  reject: (reason: any) => void;
}

const activeWorkersByKey = new Map<string, ActiveWorkerEntry>();

export function isSimulationCancelled(err: unknown): boolean {
  return !!(
    err &&
    typeof err === 'object' &&
    ((err as any).isCancelled === true || (err as any).name === 'SimulationCancelledError')
  );
}

export function cancelSimulation(key: string): void {
  const existing = activeWorkersByKey.get(key);
  if (existing) {
    existing.worker.terminate();
    const cancelErr = new Error(`Simulation cancelled for key "${key}"`);
    (cancelErr as any).isCancelled = true;
    (cancelErr as any).name = 'SimulationCancelledError';
    existing.reject(cancelErr);
    activeWorkersByKey.delete(key);
  }
}

/**
 * Runs Monte Carlo inside a Web Worker with cancellation support and fallback to main thread.
 * If key is provided, any existing in-flight run for that key is aborted.
 */
export async function runMonteCarlo(config: MonteCarloConfig, key?: string): Promise<MonteCarloResult> {
  if (typeof window !== 'undefined' && typeof Worker !== 'undefined') {
    if (key) {
      cancelSimulation(key);
    }

    return new Promise<MonteCarloResult>((resolve, reject) => {
      let worker: Worker;
      try {
        worker = new Worker(new URL('./monteCarlo.worker.ts', import.meta.url), {
          type: 'module',
        });
      } catch (e) {
        console.warn('runMonteCarlo worker creation failed, using main-thread fallback capped at 1000 sims:', e);
        resolve(runMonteCarloSimulation({ ...config, nSims: Math.min(config.nSims ?? 5000, 1000) }));
        return;
      }

      if (key) {
        activeWorkersByKey.set(key, { worker, reject });
      }

      const cleanup = () => {
        if (key && activeWorkersByKey.get(key)?.worker === worker) {
          activeWorkersByKey.delete(key);
        }
      };

      worker.onmessage = (event: MessageEvent) => {
        cleanup();
        worker.terminate();
        if (event.data?.error) {
          console.warn('runMonteCarlo worker reported error, using main-thread fallback capped at 1000 sims:', event.data.error);
          resolve(runMonteCarloSimulation({ ...config, nSims: Math.min(config.nSims ?? 5000, 1000) }));
        } else {
          resolve(event.data.result);
        }
      };

      worker.onerror = (err) => {
        cleanup();
        worker.terminate();
        console.warn('runMonteCarlo worker errored, using main-thread fallback capped at 1000 sims:', err);
        resolve(runMonteCarloSimulation({ ...config, nSims: Math.min(config.nSims ?? 5000, 1000) }));
      };

      worker.postMessage(config);
    });
  }

  return runMonteCarloSimulation(config);
}
