/**
 * Portfolio Strategy Lab - Backtesting & Scoring Engine
 * Pure mathematical functions for Walk-Forward Backtesting, Historical Block Bootstrap,
 * Multi-Strategy Scorecard, Consensus Portfolio, and Financial Goal Solving.
 */
import {
  MonteCarloConfig,
  MonteCarloResult,
  MonteCarloPercentilePoint,
  MonteCarloSummaryStats,
  OptimizerConstraints,
  WalkForwardOptions,
  WalkForwardResult,
  StrategyScorecardItem,
  StrategyScorecardResult,
  StrategyScorecardOptions,
  ConsensusResult,
  GoalSolverInput,
  GoalSolverResult,
  OptimizableAsset,
  BondInput,
} from './types.ts';
import { createMulberry32 } from './prng.ts';
import { FALLBACK_ASSET_HISTORY } from './fallbackData.ts';
import { ALL_VARIABLE_ASSETS } from './assets.ts';
import { STRATEGY_REGISTRY, cleanAndFinalizeWeights } from './strategies.ts';
import { projectOntoBoundedSimplex, computePortfolioVariance, computePortfolioReturn } from './optimizer.ts';
import { runMonteCarloSimulation, choleskyDecomposition } from './monteCarlo.ts';
import { calculateInflationTarget } from './projections.ts';
import { politisWhiteBlockLength, classifyMarketRegimes, MarketRegime } from './regimes.ts';
import { WalkForwardSubPeriod } from './types.ts';
import { Matrix } from 'ml-matrix';

// Re-export StrategyScorecardOptions so imports from backtest.ts or types.ts both succeed
export type { StrategyScorecardOptions } from './types.ts';

// ---------------------------------------------------------------------------
// Helpers: Historical Monthly Returns Extraction
// ---------------------------------------------------------------------------
export interface AlignedMonthlyData {
  dates: string[];
  assetIds: string[];
  returnsMatrix: number[][]; // T x N
}

export function getAlignedMonthlyReturns(
  customReturns?: Record<string, number[] | { date: string; return?: number }[]>
): AlignedMonthlyData {
  const assetIds = ALL_VARIABLE_ASSETS.map((a) => a.id);

  if (customReturns && Object.keys(customReturns).length > 0) {
    const ids = Object.keys(customReturns);
    const firstVal = customReturns[ids[0]];
    const T = firstVal.length;
    const dates: string[] = [];
    const matrix: number[][] = [];
    const isObject = typeof firstVal[0] === 'object' && firstVal[0] !== null;

    for (let t = 0; t < T; t++) {
      if (isObject) {
        dates.push((firstVal[t] as { date: string }).date || `M${t + 1}`);
      } else {
        dates.push(`M${t + 1}`);
      }
      const row = new Array(ids.length);
      for (let j = 0; j < ids.length; j++) {
        const item = customReturns[ids[j]][t];
        row[j] = typeof item === 'number' ? item : (item as any).return ?? 0;
      }
      matrix.push(row);
    }
    return { dates, assetIds: ids, returnsMatrix: matrix };
  }

  // Default: Extract common window from fallback benchmark series
  const dateSets = assetIds.map(
    (id) => new Set(FALLBACK_ASSET_HISTORY[id]?.series.map((s) => s.date) || [])
  );
  let commonDates = Array.from(dateSets[0] || []);
  for (let i = 1; i < dateSets.length; i++) {
    commonDates = commonDates.filter((d) => dateSets[i].has(d));
  }
  commonDates.sort();

  const priceMaps = assetIds.map((id) => {
    const map = new Map<string, number>();
    for (const pt of FALLBACK_ASSET_HISTORY[id]?.series || []) {
      map.set(pt.date, pt.price);
    }
    return map;
  });

  const dates: string[] = [];
  const matrix: number[][] = [];
  const T = commonDates.length - 1;

  for (let t = 0; t < T; t++) {
    const prevDate = commonDates[t];
    const currDate = commonDates[t + 1];
    dates.push(currDate);
    const row = new Array(assetIds.length);
    for (let j = 0; j < assetIds.length; j++) {
      const p0 = priceMaps[j].get(prevDate) ?? 0;
      const p1 = priceMaps[j].get(currDate) ?? 0;
      row[j] = p0 > 0 ? (p1 / p0 - 1) : 0;
    }
    matrix.push(row);
  }

  return { dates, assetIds, returnsMatrix: matrix };
}

// ---------------------------------------------------------------------------
// Walk-Forward Window Precomputation & Caching
// ---------------------------------------------------------------------------
export interface WalkForwardWindowData {
  t: number;
  trainStart: number;
  trainEnd: number;
  holdStart: number;
  holdEnd: number;
  trainSlice: number[][];
  cov: number[][];
  corr: number[][];
  cholesky: number[][];
  optimizableAssets: OptimizableAsset[];
  excludedAssetIds: string[];
}

export function precomputeWalkForwardWindows(
  returnsMatrix: number[][],
  assetIds: string[],
  trainWindowOrOptions: number | WalkForwardOptions = 36,
  legacyRebalancePeriod = 12
): WalkForwardWindowData[] {
  let windowType: 'expanding' | 'rolling' = 'rolling';
  let trainWindow = 36;
  let rebalancePeriod = 12;
  let embargo = 0;

  if (typeof trainWindowOrOptions === 'number') {
    trainWindow = trainWindowOrOptions;
    rebalancePeriod = legacyRebalancePeriod;
    embargo = 0;
    windowType = 'rolling';
  } else {
    const opts = trainWindowOrOptions;
    windowType = opts.windowType ?? (opts.trainWindowMonths !== undefined ? 'rolling' : 'expanding');
    trainWindow = opts.trainWindowMonths ?? (windowType === 'rolling' ? (opts.rollingMonths ?? 120) : 60);
    rebalancePeriod = opts.holdMonths ?? opts.rebalancePeriodMonths ?? 12;
    embargo = opts.embargoMonths ?? (opts.trainWindowMonths !== undefined && opts.windowType === undefined ? 0 : 1);
  }

  const totalMonths = returnsMatrix.length;
  const n = assetIds.length;
  const windows: WalkForwardWindowData[] = [];

  let t = trainWindow;
  while (t < totalMonths) {
    const trainStart = windowType === 'expanding' ? 0 : Math.max(0, t - trainWindow);
    const trainEnd = t;
    const holdStart = t + embargo;
    const holdEnd = Math.min(holdStart + rebalancePeriod, totalMonths);

    if (holdStart >= totalMonths) break;

    const trainSlice = returnsMatrix.slice(trainStart, trainEnd);
    const T_train = trainSlice.length;

    // Detect assets that have not started yet in this training window (mostly NaNs or zero observations)
    const excludedAssetIds: string[] = [];
    for (let j = 0; j < n; j++) {
      let validCount = 0;
      for (let r = 0; r < T_train; r++) {
        const val = trainSlice[r][j];
        if (!Number.isNaN(val) && Number.isFinite(val)) validCount++;
      }
      if (validCount < Math.min(6, Math.max(1, Math.floor(T_train * 0.1)))) {
        excludedAssetIds.push(assetIds[j]);
      }
    }

    const means = new Array(n).fill(0);
    for (let r = 0; r < T_train; r++) {
      for (let j = 0; j < n; j++) {
        const v = trainSlice[r][j];
        means[j] += Number.isNaN(v) ? 0 : v;
      }
    }
    for (let j = 0; j < n; j++) means[j] /= T_train;

    const cov: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    const corr: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        let sumProd = 0;
        let c = 0;
        for (let r = 0; r < T_train; r++) {
          const vI = trainSlice[r][i];
          const vJ = trainSlice[r][j];
          if (!Number.isNaN(vI) && !Number.isNaN(vJ)) {
            sumProd += (vI - means[i]) * (vJ - means[j]);
            c++;
          }
        }
        const monthlyCov = c > 1 ? sumProd / (c - 1) : i === j ? 0.04 / 12 : 0;
        const annualCov = monthlyCov * 12;
        cov[i][j] = annualCov;
        cov[j][i] = annualCov;
      }
    }

    const vols = cov.map((row, i) => Math.sqrt(Math.max(1e-6, row[i])));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        corr[i][j] = i === j ? 1.0 : cov[i][j] / Math.max(1e-6, vols[i] * vols[j]);
      }
    }

    const cholesky = choleskyDecomposition(cov);
    const optimizableAssets: OptimizableAsset[] = assetIds.map((id, idx) => ({
      id,
      expectedReturn: excludedAssetIds.includes(id) ? 0 : Math.pow(1 + Math.max(-0.9, means[idx]), 12) - 1,
      volatility: excludedAssetIds.includes(id) ? 0.001 : vols[idx],
    }));

    windows.push({
      t,
      trainStart,
      trainEnd,
      holdStart,
      holdEnd,
      trainSlice,
      cov,
      corr,
      cholesky,
      optimizableAssets,
      excludedAssetIds,
    });

    t += rebalancePeriod;
  }

  return windows;
}

// ---------------------------------------------------------------------------
// 1. Walk-Forward Backtesting
// ---------------------------------------------------------------------------

/**
 * Executes a strictly walk-forward rolling or expanding window backtest for a given strategy.
 * - Window Type: 'expanding' (default, min 60 months) or 'rolling' (default 120 months)
 * - Hold Window: 12 months (default)
 * - Embargo: 1 month (default), skips gap between training end and hold start
 * - Strictly no lookahead: future data is never accessed in the training window.
 * - Asset Exclusion: Assets not yet started in training window receive weight 0.
 * - Sub-period metrics: Reports 5-year block performance breakdown.
 * - Consistency: Share of rolling 36-month OOS windows beating equal weight.
 */
export function walkForwardBacktest(
  strategyId: string,
  monthlyReturnsByAsset?: Record<string, number[] | { date: string; return?: number }[]>,
  options: WalkForwardOptions = {}
): WalkForwardResult {
  const strategyEntry = STRATEGY_REGISTRY[strategyId];
  if (!strategyEntry) {
    throw new Error(`Strategy "${strategyId}" not found in STRATEGY_REGISTRY.`);
  }

  const { dates, assetIds, returnsMatrix } = getAlignedMonthlyReturns(monthlyReturnsByAsset);
  const totalMonths = returnsMatrix.length;
  if (totalMonths < 60) {
    throw new Error(
      `Insufficient data for walk-forward backtest: minimum 60 months required (found ${totalMonths} months).`
    );
  }

  const rebalanceCost = options.rebalanceCost ?? 0.002;
  const constraints = options.constraints ?? { maxRisk: 0.18, maxPerAsset: 0.35, minPerAsset: 0.0, riskFreeRate: 0.065 };
  const seed = options.seed ?? 42;
  const n = assetIds.length;

  const windows: WalkForwardWindowData[] =
    options.cachedWindows ??
    precomputeWalkForwardWindows(returnsMatrix, assetIds, options);

  const oosMonthlyReturns: number[] = [];
  const oosDates: string[] = [];
  const excludedAssetsByWindow: { t: number; excluded: string[] }[] = [];
  let prevDriftedWeights: number[] | null = null;
  let totalTurnover = 0;
  let rebalanceCount = 0;

  for (const win of windows) {
    excludedAssetsByWindow.push({ t: win.t, excluded: win.excludedAssetIds });

    const point = strategyEntry.fn({
      assets: win.optimizableAssets,
      cov: win.cov,
      corr: win.corr,
      constraints,
      returnsHistory: win.trainSlice,
      seed: seed + win.t,
      nSamples: 30,
      nScenarios: 500,
      cachedCholesky: win.cholesky,
      isWalkForward: true,
    });

    // Zero out any excluded assets that have not started yet, and renormalize active weights
    let targetWeights = assetIds.map((id) => (win.excludedAssetIds.includes(id) ? 0 : (point.weights[id] || 0)));
    const sumActive = targetWeights.reduce((a, b) => a + b, 0);
    if (sumActive > 1e-6) {
      targetWeights = targetWeights.map((w) => w / sumActive);
    }

    // Compute turnover against drifted weights from the end of the previous period
    let turnover = 0;
    if (rebalanceCount === 0 || prevDriftedWeights === null) {
      turnover = 1.0; // initial allocation from cash
    } else {
      for (let j = 0; j < n; j++) {
        turnover += Math.abs(targetWeights[j] - prevDriftedWeights[j]);
      }
      turnover *= 0.5;
    }
    totalTurnover += turnover;
    rebalanceCount++;

    // Buy-and-hold out-of-sample holding period (respects embargo by starting at win.holdStart)
    let wMonth = targetWeights.slice();
    for (let m = win.holdStart; m < win.holdEnd; m++) {
      let r_m = 0;
      for (let j = 0; j < n; j++) {
        const val = returnsMatrix[m][j];
        r_m += wMonth[j] * (Number.isNaN(val) ? 0 : val);
      }
      // Deduct rebalance transaction fee in first month of holding window
      if (m === win.holdStart && rebalanceCost > 0) {
        r_m -= turnover * rebalanceCost;
      }
      oosMonthlyReturns.push(r_m);
      oosDates.push(dates[m]);

      // Drift weights for next month
      const nextW = new Array(n);
      let nextWTotal = 0;
      for (let j = 0; j < n; j++) {
        const rVal = returnsMatrix[m][j];
        const val = wMonth[j] * (1 + (Number.isNaN(rVal) ? 0 : rVal));
        nextW[j] = Math.max(0, val);
        nextWTotal += nextW[j];
      }
      if (nextWTotal > 1e-8) {
        for (let j = 0; j < n; j++) nextW[j] /= nextWTotal;
      } else {
        for (let j = 0; j < n; j++) nextW[j] = targetWeights[j];
      }
      wMonth = nextW;
    }
    prevDriftedWeights = wMonth.slice();
  }

  const M = oosMonthlyReturns.length;
  let cumulative = 1.0;
  let peak = 1.0;
  let maxDD = 0;
  const equityCurve: { date: string; value: number }[] = [
    { date: oosDates[0] ? `Pre-${oosDates[0]}` : 'Start', value: 100 },
  ];

  for (let i = 0; i < M; i++) {
    cumulative *= 1 + oosMonthlyReturns[i];
    if (cumulative > peak) peak = cumulative;
    const dd = (peak - cumulative) / peak;
    if (dd > maxDD) maxDD = dd;
    equityCurve.push({
      date: oosDates[i],
      value: Number((100 * cumulative).toFixed(2)),
    });
  }

  const cagr = M > 0 ? Math.pow(cumulative, 12 / M) - 1 : 0;
  const meanM = oosMonthlyReturns.reduce((a, b) => a + b, 0) / Math.max(1, M);
  let varSum = 0;
  for (const r of oosMonthlyReturns) {
    varSum += (r - meanM) * (r - meanM);
  }
  const monthlyVol = Math.sqrt(varSum / Math.max(1, M - 1));
  const volatility = monthlyVol * Math.sqrt(12);
  const sharpe = volatility > 1e-4 ? (cagr - (constraints.riskFreeRate ?? 0.065)) / volatility : 0;

  let worst12m = 0;
  if (M >= 12) {
    worst12m = Infinity;
    for (let i = 0; i <= M - 12; i++) {
      let roll12 = 1.0;
      for (let k = 0; k < 12; k++) {
        roll12 *= 1 + oosMonthlyReturns[i + k];
      }
      const ret12 = roll12 - 1;
      if (ret12 < worst12m) worst12m = ret12;
    }
  } else {
    worst12m = cumulative - 1;
  }

  // Sub-periods: 5-year blocks (60-month chunks)
  const subPeriods: WalkForwardSubPeriod[] = [];
  const blockSize = 60;
  for (let start = 0; start < M; start += blockSize) {
    const end = Math.min(start + blockSize, M);
    const subRets = oosMonthlyReturns.slice(start, end);
    const subDates = oosDates.slice(start, end);
    const subM = subRets.length;
    if (subM > 0) {
      let subCum = 1.0;
      let subPeak = 1.0;
      let subMaxDD = 0;
      for (let k = 0; k < subM; k++) {
        subCum *= 1 + subRets[k];
        if (subCum > subPeak) subPeak = subCum;
        const dd = (subPeak - subCum) / subPeak;
        if (dd > subMaxDD) subMaxDD = dd;
      }
      const subCagr = subM > 0 ? Math.pow(subCum, 12 / subM) - 1 : 0;
      const subMean = subRets.reduce((a, b) => a + b, 0) / subM;
      let subVar = 0;
      for (const r of subRets) subVar += (r - subMean) * (r - subMean);
      const subVol = Math.sqrt(subVar / Math.max(1, subM - 1)) * Math.sqrt(12);
      const subSharpe = subVol > 1e-4 ? (subCagr - 0.065) / subVol : 0;

      const startYear = subDates[0].slice(0, 4);
      const endYear = subDates[subM - 1].slice(0, 4);
      subPeriods.push({
        label: startYear === endYear ? startYear : `${startYear}-${endYear}`,
        startDate: subDates[0],
        endDate: subDates[subM - 1],
        cagr: Number(subCagr.toFixed(4)),
        volatility: Number(subVol.toFixed(4)),
        sharpe: Number(subSharpe.toFixed(4)),
        maxDrawdown: Number(subMaxDD.toFixed(4)),
      });
    }
  }

  // Consistency: share of rolling 36-month OOS windows in which strategy beat equal weight
  let consistency = 1.0;
  if (M >= 36) {
    const eqOosReturns: number[] = [];
    for (const win of windows) {
      for (let m = win.holdStart; m < win.holdEnd; m++) {
        let sumR = 0;
        let c = 0;
        for (let j = 0; j < n; j++) {
          const val = returnsMatrix[m][j];
          if (!Number.isNaN(val)) {
            sumR += val;
            c++;
          }
        }
        eqOosReturns.push(c > 0 ? sumR / c : 0);
      }
    }

    let wins = 0;
    const numWindows36 = M - 35;
    for (let k = 0; k < numWindows36; k++) {
      let stratGrowth = 1.0;
      let eqGrowth = 1.0;
      for (let step = 0; step < 36; step++) {
        stratGrowth *= 1 + oosMonthlyReturns[k + step];
        eqGrowth *= 1 + (eqOosReturns[k + step] ?? 0);
      }
      if (stratGrowth >= eqGrowth) {
        wins++;
      }
    }
    consistency = Number((wins / numWindows36).toFixed(4));
  } else if (M > 0) {
    consistency = 1.0;
  }

  return {
    strategyId,
    strategyName: strategyEntry.name,
    monthlyReturns: oosMonthlyReturns,
    dates: oosDates,
    cagr: Number(cagr.toFixed(4)),
    volatility: Number(volatility.toFixed(4)),
    sharpe: Number(sharpe.toFixed(4)),
    maxDrawdown: Number(maxDD.toFixed(4)),
    worst12MonthReturn: Number((worst12m === Infinity ? 0 : worst12m).toFixed(4)),
    turnover: Number((totalTurnover / Math.max(1, rebalanceCount)).toFixed(4)),
    cumulativeEquity: equityCurve,
  };
}

// ---------------------------------------------------------------------------
// 2. Historical Block Bootstrap Monte Carlo
// ---------------------------------------------------------------------------

/**
 * Historical Block Bootstrap Simulation
 * Requirement 1: Re-centres the bootstrap returns using user's assetStats assumptions.
 * For each asset, subtracts the asset's own historical mean log return, and adds
 * assumption-implied monthly drift ln(1 + assetStats[id].cagr)/12.
 * Rescales de-meaned series to match assetStats[id].volatility when rescaleVol is true (default true).
 * Requirement 2: Uses shared calculateInflationTarget for probability of beating inflation.
 * Requirement 3: Includes terminalValues and probabilityAtLeast.
 */
export function historicalBootstrap(config: MonteCarloConfig): MonteCarloResult {
  const startTime = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const { allocation, years } = config;
  const nSims = config.nSims ?? 5000;
  const seed = config.seed ?? 10101;
  const inflationRate = config.inflationRate ?? 0.06;
  const totalMonths = Math.round(years * 12);
  const bootstrapType = config.bootstrapType ?? 'circular';
  const regimeFilter = config.regimeFilter;

  const { assetIds: alignedIds, returnsMatrix } = getAlignedMonthlyReturns(config.monthlyReturnsByAsset);
  const T = returnsMatrix.length;

  const stochasticIndices: number[] = [];
  const stochasticWeights: number[] = [];
  let bondWeight = 0;

  for (const [id, weight] of Object.entries(allocation)) {
    if (id === 'bonds_fd') {
      bondWeight += weight;
    } else {
      const colIdx = alignedIds.indexOf(id);
      if (colIdx >= 0 && weight > 0) {
        stochasticIndices.push(colIdx);
        stochasticWeights.push(weight);
      }
    }
  }

  const totalW = stochasticWeights.reduce((a, b) => a + b, 0) + bondWeight;
  const normWeights = stochasticWeights.map((w) => w / Math.max(1e-6, totalW));
  const normBondWeight = bondWeight / Math.max(1e-6, totalW);
  const numStochastic = stochasticIndices.length;

  // Compute portfolio return series to determine optimal block length via Politis-White (2004)
  const portfolioReturns: number[] = new Array(T);
  for (let t = 0; t < T; t++) {
    let sumR = 0;
    for (let i = 0; i < numStochastic; i++) {
      sumR += normWeights[i] * returnsMatrix[t][stochasticIndices[i]];
    }
    portfolioReturns[t] = sumR;
  }
  const blockSize = config.blockLength ?? politisWhiteBlockLength(portfolioReturns);

  // If regimeFilter is active, identify months matching selected regimes
  let candidateIndices: number[] = [];
  if (regimeFilter && regimeFilter.length > 0) {
    const mktIdx = alignedIds.indexOf('nifty_50');
    const mktRets = mktIdx >= 0 ? returnsMatrix.map((row) => row[mktIdx]) : portfolioReturns;
    const classified = classifyMarketRegimes(mktRets);
    for (let t = 0; t < T; t++) {
      const reg = classified[t]?.regime ?? 'normal';
      if (regimeFilter.includes(reg)) {
        candidateIndices.push(t);
      }
    }
  }
  if (candidateIndices.length === 0) {
    candidateIndices = Array.from({ length: T }, (_, i) => i);
  }
  const numCandidates = candidateIndices.length;

  let lumpsum = config.lumpsumAmount ?? 0;
  let sip = config.monthlySip ?? 0;
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

  const totalInvested = lumpsum + sip * totalMonths;

  const bondConfig = config.bondConfig || {
    couponRate: 0.075,
    tenureYears: years,
    payoutFrequency: 'annual',
    creditRisk: { defaultProbabilityAnnual: 0.005, recoveryRate: 0.5 },
    reinvestmentRateAfterMaturity: 0.065,
  };
  const bondTenure = bondConfig.tenureYears ?? years;
  const maturityMonth = Math.round(bondTenure * 12);
  const monthlyBondYield = Math.pow(1 + bondConfig.couponRate, 1 / 12) - 1;
  const reinvestmentRate = bondConfig.reinvestmentRateAfterMaturity ?? 0.065;
  const monthlyReinvestmentYield = Math.pow(1 + reinvestmentRate, 1 / 12) - 1;
  const monthlyDefaultProb = 1 - Math.pow(1 - (bondConfig.creditRisk?.defaultProbabilityAnnual ?? 0.005), 1 / 12);
  const recoveryRate = bondConfig.creditRisk?.recoveryRate ?? 0.5;
  const rescaleVol = config.rescaleVol !== false; // default true

  // Requirement 1: Re-centre the bootstrap returns matrix according to assetStats assumptions
  // For each stochastic asset, take historical monthly log returns, subtract historical mean log return,
  // optionally rescale volatility, and add assumption drift = ln(1 + cagr) / 12.
  const adjustedReturns: number[][] = Array.from({ length: T }, () => new Array(numStochastic).fill(0));
  for (let i = 0; i < numStochastic; i++) {
    const col = stochasticIndices[i];
    const id = alignedIds[col];
    const cagr = config.assetStats?.[id]?.cagr ?? FALLBACK_ASSET_HISTORY[id]?.cagr ?? 0.12;
    const targetVol = config.assetStats?.[id]?.volatility ?? FALLBACK_ASSET_HISTORY[id]?.volatility ?? 0.18;
    const targetDrift = Math.log(1 + cagr) / 12;

    let sumLogRet = 0;
    const logRets = new Float64Array(T);
    for (let t = 0; t < T; t++) {
      const r = returnsMatrix[t][col];
      const lr = Math.log(Math.max(1e-6, 1 + r));
      logRets[t] = lr;
      sumLogRet += lr;
    }
    const meanLogRet = sumLogRet / T;

    let varLogRet = 0;
    for (let t = 0; t < T; t++) {
      const diff = logRets[t] - meanLogRet;
      varLogRet += diff * diff;
    }
    const monthlyHistVol = Math.sqrt(varLogRet / Math.max(1, T - 1));
    const annualHistVol = monthlyHistVol * Math.sqrt(12);

    let scale = 1.0;
    if (rescaleVol && annualHistVol > 1e-6) {
      scale = targetVol / annualHistVol;
    }

    for (let t = 0; t < T; t++) {
      const adjLog = (logRets[t] - meanLogRet) * scale + targetDrift;
      adjustedReturns[t][i] = Math.exp(adjLog) - 1;
    }
  }

  const rng = createMulberry32(seed);
  const yearValues: number[][] = Array.from({ length: years }, () => new Array(nSims).fill(0));
  let lossCount = 0;
  let beatInflationCount = 0;
  let defaultsCount = 0;

  // Requirement 2: Use shared calculateInflationTarget for accurate SIP inflation target
  const inflationThreshold = calculateInflationTarget(lumpsum, sip, years, inflationRate);

  for (let s = 0; s < nSims; s++) {
    const assetCapital = new Float64Array(numStochastic);
    let bondCapital = 0;
    let redeployedBondCapital = 0;
    let bondHasDefaulted = false;

    if (lumpsum > 0) {
      for (let i = 0; i < numStochastic; i++) {
        assetCapital[i] = lumpsum * normWeights[i];
      }
      bondCapital = lumpsum * normBondWeight;
    }

    if (bootstrapType === 'stationary') {
      // Politis-Romano (1994) Stationary Bootstrap:
      // Random block lengths geometric with mean = blockSize.
      // At each step, jump to a new random start with probability p = 1 / blockSize.
      let currentIdx = Math.floor(rng() * numCandidates);

      for (let m = 0; m < totalMonths; m++) {
        if (m > 0 && rng() < 1.0 / blockSize) {
          currentIdx = Math.floor(rng() * numCandidates);
        } else if (m > 0) {
          currentIdx = (currentIdx + 1) % numCandidates;
        }

        const tIndex = candidateIndices[currentIdx];
        const currentMonth = m;

        if (sip > 0) {
          for (let i = 0; i < numStochastic; i++) {
            assetCapital[i] += sip * normWeights[i];
          }
          if (bondHasDefaulted) {
            redeployedBondCapital += sip * normBondWeight;
          } else {
            bondCapital += sip * normBondWeight;
          }
        }

        // Apply re-centred returns
        for (let i = 0; i < numStochastic; i++) {
          const ret = adjustedReturns[tIndex][i];
          assetCapital[i] *= 1 + ret;
        }

        if (normBondWeight > 0) {
          if (!bondHasDefaulted) {
            if (rng() < monthlyDefaultProb) {
              bondCapital *= recoveryRate;
              bondHasDefaulted = true;
              defaultsCount++;
            } else {
              const currentYield = currentMonth < maturityMonth ? monthlyBondYield : monthlyReinvestmentYield;
              bondCapital *= 1 + currentYield;
            }
          }
          if (redeployedBondCapital > 0) {
            redeployedBondCapital *= 1 + monthlyReinvestmentYield;
          }
        }

        const monthNum = currentMonth + 1;
        if (monthNum % 12 === 0) {
          const yearIndex = monthNum / 12 - 1;
          let totalPortfolioVal = bondCapital + redeployedBondCapital;
          for (let i = 0; i < numStochastic; i++) {
            totalPortfolioVal += assetCapital[i];
          }
          yearValues[yearIndex][s] = totalPortfolioVal;
        }
      }
    } else {
      // Circular / Block Bootstrap (default)
      let m = 0;
      while (m < totalMonths) {
        const startPick = Math.floor(rng() * numCandidates);
        const stepsInBlock = Math.min(blockSize, totalMonths - m);

        for (let b = 0; b < stepsInBlock; b++) {
          const tIndex = candidateIndices[(startPick + b) % numCandidates];
          const currentMonth = m + b;

          if (sip > 0) {
            for (let i = 0; i < numStochastic; i++) {
              assetCapital[i] += sip * normWeights[i];
            }
            if (bondHasDefaulted) {
              redeployedBondCapital += sip * normBondWeight;
            } else {
              bondCapital += sip * normBondWeight;
            }
          }

          // Apply re-centred returns
          for (let i = 0; i < numStochastic; i++) {
            const ret = adjustedReturns[tIndex][i];
            assetCapital[i] *= 1 + ret;
          }

          if (normBondWeight > 0) {
            if (!bondHasDefaulted) {
              if (rng() < monthlyDefaultProb) {
                bondCapital *= recoveryRate;
                bondHasDefaulted = true;
                defaultsCount++;
              } else {
                const currentYield = currentMonth < maturityMonth ? monthlyBondYield : monthlyReinvestmentYield;
                bondCapital *= 1 + currentYield;
              }
            }
            if (redeployedBondCapital > 0) {
              redeployedBondCapital *= 1 + monthlyReinvestmentYield;
            }
          }

          const monthNum = currentMonth + 1;
          if (monthNum % 12 === 0) {
            const yearIndex = monthNum / 12 - 1;
            let totalPortfolioVal = bondCapital + redeployedBondCapital;
            for (let i = 0; i < numStochastic; i++) {
              totalPortfolioVal += assetCapital[i];
            }
            yearValues[yearIndex][s] = totalPortfolioVal;
          }
        }
        m += stepsInBlock;
      }
    }

    const finalVal = yearValues[years - 1][s];
    if (finalVal < totalInvested) lossCount++;
    if (finalVal >= inflationThreshold) beatInflationCount++;
  }

  const getPercentile = (arr: number[], p: number): number => {
    const idx = (p / 100) * (arr.length - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);
    const weight = idx - lower;
    if (lower === upper) return arr[lower];
    return arr[lower] * (1 - weight) + arr[upper] * weight;
  };

  const percentiles: MonteCarloPercentilePoint[] = [];
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
  const meanFinal = sumFinal / nSims;
  const varianceFinal = finalSorted.reduce((sum, v) => sum + Math.pow(v - meanFinal, 2), 0) / (nSims - 1);

  const stats: MonteCarloSummaryStats = {
    min: Number(finalSorted[0].toFixed(2)),
    p25: Number(getPercentile(finalSorted, 25).toFixed(2)),
    median: Number(getPercentile(finalSorted, 50).toFixed(2)),
    p75: Number(getPercentile(finalSorted, 75).toFixed(2)),
    mean: Number(meanFinal.toFixed(2)),
    max: Number(finalSorted[finalSorted.length - 1].toFixed(2)),
    stdDev: Number(Math.sqrt(varianceFinal).toFixed(2)),
  };

  const endTime = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const terminalValues = new Float64Array(finalSorted);

  return {
    percentilesByYear: percentiles,
    probabilityOfLoss: Number(((lossCount / nSims) * 100).toFixed(2)),
    probabilityOfBeatingInflation: Number(((beatInflationCount / nSims) * 100).toFixed(2)),
    inflationRate,
    totalInvested: Number(totalInvested.toFixed(2)),
    terminalStats: stats,
    nSims,
    simulationTimeMs: Number((endTime - startTime).toFixed(1)),
    defaultsTriggeredCount: defaultsCount,
    terminalValues,
    probabilityAtLeast: (target: number) => {
      let count = 0;
      for (let i = 0; i < nSims; i++) {
        if (terminalValues[i] >= target) count++;
      }
      return Number(((count / nSims) * 100).toFixed(1));
    },
  };
}

// ---------------------------------------------------------------------------
// 3. Multi-Strategy Scorecard
// ---------------------------------------------------------------------------

/**
 * Runs every strategy through walk-forward backtest, parametric MC, and bootstrap MC.
 * Ranks them using: 0.4 * OOS Sharpe + 0.3 * Bootstrap P10 + 0.3 * Max Drawdown.
 */
export function runStrategyScorecard(options: StrategyScorecardOptions = {}): StrategyScorecardResult {
  const amount = options.amount ?? 1000000;
  const years = options.years ?? 10;
  const mode = options.mode ?? 'both';
  const constraints = options.constraints ?? { maxRisk: 0.18, maxPerAsset: 0.35, minPerAsset: 0.0, riskFreeRate: 0.065 };
  const bondConfig = options.bondConfig;
  const inflationRate = options.inflationRate ?? 0.06;
  const seed = options.seed ?? 42;

  const { assetIds, returnsMatrix } = getAlignedMonthlyReturns(options.monthlyReturnsByAsset);
  const n = assetIds.length;
  const T = returnsMatrix.length;

  const means = new Array(n).fill(0);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < n; j++) means[j] += returnsMatrix[t][j];
  }
  for (let j = 0; j < n; j++) means[j] /= T;

  const cov: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const corr: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let sumProd = 0;
      for (let t = 0; t < T; t++) {
        sumProd += (returnsMatrix[t][i] - means[i]) * (returnsMatrix[t][j] - means[j]);
      }
      const monthlyCov = sumProd / Math.max(1, T - 1);
      const annualCov = monthlyCov * 12;
      cov[i][j] = annualCov;
      cov[j][i] = annualCov;
    }
  }

  const vols = cov.map((row, i) => Math.sqrt(Math.max(1e-6, row[i])));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      corr[i][j] = i === j ? 1.0 : cov[i][j] / Math.max(1e-6, vols[i] * vols[j]);
    }
  }

  const inSampleCholesky = choleskyDecomposition(cov);
  const optimizableAssets: OptimizableAsset[] = assetIds.map((id, idx) => ({
    id,
    expectedReturn: Math.pow(1 + Math.max(-0.9, means[idx]), 12) - 1,
    volatility: vols[idx],
  }));

  const assetStatsMap: Record<string, { cagr: number; volatility: number }> = {};
  for (const a of optimizableAssets) {
    assetStatsMap[a.id] = { cagr: a.expectedReturn, volatility: a.volatility };
  }

  const cachedWindows = precomputeWalkForwardWindows(returnsMatrix, assetIds, 36, 12);

  const rawItems: Array<{
    strategyId: string;
    strategyName: string;
    weights: Record<string, number>;
    inSampleReturn: number;
    inSampleVol: number;
    inSampleSharpe: number;
    oosCagr: number;
    oosVol: number;
    oosSharpe: number;
    maxDrawdown: number;
    worst12m: number;
    turnover: number;
    parametricP10: number;
    parametricP50: number;
    parametricP90: number;
    bootstrapP10: number;
    bootstrapP50: number;
    bootstrapP90: number;
    probLoss: number;
    consistency?: number;
    subPeriods?: { label: string; cagr: number; sharpe: number }[];
  }> = [];

  const strategyKeys = Object.keys(STRATEGY_REGISTRY);
  const totalStrategies = strategyKeys.length;
  let doneCount = 0;

  for (const stratId of strategyKeys) {
    const strat = STRATEGY_REGISTRY[stratId];
    const point = strat.fn({
      assets: optimizableAssets,
      cov,
      corr,
      constraints,
      returnsHistory: returnsMatrix,
      seed,
      nSamples: 200,
      nScenarios: 2000,
      cachedCholesky: inSampleCholesky,
      isWalkForward: false,
    });

    const wf = walkForwardBacktest(stratId, options.monthlyReturnsByAsset, {
      constraints,
      seed,
      cachedWindows,
    });

    const mcParametric = runMonteCarloSimulation({
      allocation: point.weights,
      years,
      mode,
      amount,
      assetStats: assetStatsMap,
      covarianceMatrix: cov,
      assetIds,
      bondConfig,
      nSims: 1000,
      inflationRate,
      seed,
    });

    const mcBootstrap = historicalBootstrap({
      allocation: point.weights,
      years,
      mode,
      amount,
      assetStats: assetStatsMap,
      covarianceMatrix: cov,
      assetIds,
      bondConfig,
      nSims: 1000,
      inflationRate,
      seed: seed + 55,
      monthlyReturnsByAsset: options.monthlyReturnsByAsset,
    });

    const pY = mcParametric.percentilesByYear[years - 1];
    const bY = mcBootstrap.percentilesByYear[years - 1];

    rawItems.push({
      strategyId: stratId,
      strategyName: strat.name,
      weights: point.weights,
      inSampleReturn: point.expectedReturn,
      inSampleVol: point.volatility,
      inSampleSharpe: point.sharpeRatio,
      oosCagr: wf.cagr,
      oosVol: wf.volatility,
      oosSharpe: wf.sharpe,
      maxDrawdown: wf.maxDrawdown,
      worst12m: wf.worst12MonthReturn,
      turnover: wf.turnover,
      parametricP10: pY.p10,
      parametricP50: pY.p50,
      parametricP90: pY.p90,
      bootstrapP10: bY.p10,
      bootstrapP50: bY.p50,
      bootstrapP90: bY.p90,
      probLoss: mcBootstrap.probabilityOfLoss,
      consistency: wf.consistency,
      subPeriods: wf.subPeriods?.map((sp) => ({ label: sp.label, cagr: sp.cagr, sharpe: sp.sharpe })),
    });

    doneCount++;
    if (options.onProgress) {
      options.onProgress(stratId, doneCount, totalStrategies);
    }
  }

  const N = rawItems.length;
  const sharpeSorted = [...rawItems].sort((a, b) => b.oosSharpe - a.oosSharpe);
  const p10Sorted = [...rawItems].sort((a, b) => b.bootstrapP10 - a.bootstrapP10);
  const ddSorted = [...rawItems].sort((a, b) => a.maxDrawdown - b.maxDrawdown);

  const rankedItems: StrategyScorecardItem[] = rawItems.map((item) => {
    const sharpeRank = sharpeSorted.findIndex((x) => x.strategyId === item.strategyId) + 1;
    const p10Rank = p10Sorted.findIndex((x) => x.strategyId === item.strategyId) + 1;
    const ddRank = ddSorted.findIndex((x) => x.strategyId === item.strategyId) + 1;

    const sharpeScore = ((N - sharpeRank) / Math.max(1, N - 1)) * 100;
    const p10Score = ((N - p10Rank) / Math.max(1, N - 1)) * 100;
    const ddScore = ((N - ddRank) / Math.max(1, N - 1)) * 100;
    const rankScore = Number((0.4 * sharpeScore + 0.3 * p10Score + 0.3 * ddScore).toFixed(2));

    return {
      ...item,
      rankScore,
      overallRank: 0,
    };
  });

  rankedItems.sort((a, b) => b.rankScore - a.rankScore);
  rankedItems.forEach((item, idx) => {
    item.overallRank = idx + 1;
  });

  let userPlanScorecard: StrategyScorecardItem | undefined;
  if (options.userAllocation) {
    const normUserAlloc = cleanAndFinalizeWeights(
      options.userAllocation,
      assetIds,
      constraints.maxPerAsset ?? 1.0,
      constraints.minPerAsset ?? 0.0
    );
    const weightsArr = assetIds.map((id) => normUserAlloc[id] || 0);
    const covMat = new Matrix(cov);
    const uRet = computePortfolioReturn(weightsArr, optimizableAssets.map((a) => a.expectedReturn));
    const uVol = Math.sqrt(computePortfolioVariance(weightsArr, covMat));
    const uSharpe = (uRet - (constraints.riskFreeRate ?? 0.065)) / Math.max(1e-4, uVol);

    const userWF = walkForwardBacktest('equal_weight', options.monthlyReturnsByAsset, {
      constraints,
      cachedWindows,
    });

    const userOOSReturns: number[] = [];
    let cum = 1.0;
    let pk = 1.0;
    let mdd = 0;
    for (let tIdx = 36; tIdx < T; tIdx++) {
      let r_t = 0;
      for (let j = 0; j < n; j++) r_t += weightsArr[j] * returnsMatrix[tIdx][j];
      userOOSReturns.push(r_t);
      cum *= 1 + r_t;
      if (cum > pk) pk = cum;
      const d = (pk - cum) / pk;
      if (d > mdd) mdd = d;
    }
    const oosM = userOOSReturns.length;
    const userOOSCagr = Math.pow(cum, 12 / oosM) - 1;
    let uVar = 0;
    const uMean = userOOSReturns.reduce((a, b) => a + b, 0) / oosM;
    for (const r of userOOSReturns) uVar += (r - uMean) * (r - uMean);
    const userOOSVol = Math.sqrt(uVar / (oosM - 1)) * Math.sqrt(12);

    const mcParam = runMonteCarloSimulation({
      allocation: normUserAlloc,
      years,
      mode,
      amount,
      assetStats: assetStatsMap,
      covarianceMatrix: cov,
      assetIds,
      bondConfig,
      nSims: 1000,
      inflationRate,
      seed,
    });

    const mcBoot = historicalBootstrap({
      allocation: normUserAlloc,
      years,
      mode,
      amount,
      assetStats: assetStatsMap,
      covarianceMatrix: cov,
      assetIds,
      bondConfig,
      nSims: 1000,
      inflationRate,
      seed: seed + 77,
      monthlyReturnsByAsset: options.monthlyReturnsByAsset,
    });

    const pY = mcParam.percentilesByYear[years - 1];
    const bY = mcBoot.percentilesByYear[years - 1];

    userPlanScorecard = {
      strategyId: 'user_plan',
      strategyName: 'Your Current Plan',
      weights: normUserAlloc,
      inSampleReturn: Number(uRet.toFixed(4)),
      inSampleVol: Number(uVol.toFixed(4)),
      inSampleSharpe: Number(uSharpe.toFixed(4)),
      oosCagr: Number(userOOSCagr.toFixed(4)),
      oosVol: Number(userOOSVol.toFixed(4)),
      oosSharpe: Number(((userOOSCagr - 0.065) / Math.max(1e-4, userOOSVol)).toFixed(4)),
      maxDrawdown: Number(mdd.toFixed(4)),
      worst12m: userWF.worst12MonthReturn,
      turnover: 0,
      parametricP10: pY.p10,
      parametricP50: pY.p50,
      parametricP90: pY.p90,
      bootstrapP10: bY.p10,
      bootstrapP50: bY.p50,
      bootstrapP90: bY.p90,
      probLoss: mcBoot.probabilityOfLoss,
      consistency: userWF.consistency,
      subPeriods: userWF.subPeriods?.map((sp) => ({ label: sp.label, cagr: sp.cagr, sharpe: sp.sharpe })),
      rankScore: 0,
      overallRank: 0,
    };
  }

  const top3 = rankedItems.slice(0, 3);
  const consensusWeights: Record<string, number> = {};
  for (const a of optimizableAssets) consensusWeights[a.id] = 0;
  for (const item of top3) {
    for (const [k, w] of Object.entries(item.weights)) {
      consensusWeights[k] = (consensusWeights[k] || 0) + w / 3;
    }
  }

  const consensusAlloc = cleanAndFinalizeWeights(
    consensusWeights,
    assetIds,
    constraints.maxPerAsset ?? 1.0,
    constraints.minPerAsset ?? 0.0
  );
  const cleanConsensusArr = assetIds.map((id) => consensusAlloc[id]);
  const covMat = new Matrix(cov);
  const cRet = computePortfolioReturn(cleanConsensusArr, optimizableAssets.map((a) => a.expectedReturn));
  const cVol = Math.sqrt(computePortfolioVariance(cleanConsensusArr, covMat));
  const cSharpe = (cRet - (constraints.riskFreeRate ?? 0.065)) / Math.max(1e-4, cVol);

  const cBoot = historicalBootstrap({
    allocation: consensusAlloc,
    years,
    mode,
    amount,
    assetStats: assetStatsMap,
    covarianceMatrix: cov,
    assetIds,
    bondConfig,
    nSims: 1000,
    inflationRate,
    seed: seed + 99,
    monthlyReturnsByAsset: options.monthlyReturnsByAsset,
  });

  const cParam = runMonteCarloSimulation({
    allocation: consensusAlloc,
    years,
    mode,
    amount,
    assetStats: assetStatsMap,
    covarianceMatrix: cov,
    assetIds,
    bondConfig,
    nSims: 1000,
    inflationRate,
    seed: seed + 99,
  });

  const cPY = cParam.percentilesByYear[years - 1];
  const cBY = cBoot.percentilesByYear[years - 1];

  const consensusScorecard: StrategyScorecardItem = {
    strategyId: 'consensus',
    strategyName: 'Consensus (Top 3 Blended)',
    weights: consensusAlloc,
    inSampleReturn: Number(cRet.toFixed(4)),
    inSampleVol: Number(cVol.toFixed(4)),
    inSampleSharpe: Number(cSharpe.toFixed(4)),
    oosCagr: Number(((top3[0].oosCagr + top3[1].oosCagr + top3[2].oosCagr) / 3).toFixed(4)),
    oosVol: Number(((top3[0].oosVol + top3[1].oosVol + top3[2].oosVol) / 3).toFixed(4)),
    oosSharpe: Number(((top3[0].oosSharpe + top3[1].oosSharpe + top3[2].oosSharpe) / 3).toFixed(4)),
    maxDrawdown: Number(((top3[0].maxDrawdown + top3[1].maxDrawdown + top3[2].maxDrawdown) / 3).toFixed(4)),
    worst12m: Number(((top3[0].worst12m + top3[1].worst12m + top3[2].worst12m) / 3).toFixed(4)),
    turnover: Number(((top3[0].turnover + top3[1].turnover + top3[2].turnover) / 3).toFixed(4)),
    parametricP10: cPY.p10,
    parametricP50: cPY.p50,
    parametricP90: cPY.p90,
    bootstrapP10: cBY.p10,
    bootstrapP50: cBY.p50,
    bootstrapP90: cBY.p90,
    probLoss: cBoot.probabilityOfLoss,
    rankScore: Number(((top3[0].rankScore + top3[1].rankScore + top3[2].rankScore) / 3).toFixed(2)),
    overallRank: 0,
  };

  const bestOOS = rankedItems[0];
  const maxReturnItem = rankedItems.find((r) => r.strategyId === 'max_return');
  const maxReturnRank = maxReturnItem ? maxReturnItem.overallRank : N;
  let verdict = `The top-ranked model is ${bestOOS.strategyName} with out-of-sample Sharpe of ${bestOOS.oosSharpe.toFixed(2)} and max drawdown of ${(bestOOS.maxDrawdown * 100).toFixed(1)}%.`;
  if (maxReturnRank > 3) {
    verdict += ` Notably, standard in-sample Max Return ranks #${maxReturnRank} out-of-sample, indicating historical backtest overfitting.`;
  }
  if (userPlanScorecard) {
    verdict += ` Your plan achieves an estimated P50 of ₹ ${Math.round(userPlanScorecard.bootstrapP50).toLocaleString('en-IN')}, compared to ₹ ${Math.round(bestOOS.bootstrapP50).toLocaleString('en-IN')} from ${bestOOS.strategyName}.`;
  }

  return {
    items: rankedItems,
    userPlanScorecard,
    consensusScorecard,
    topRankedStrategyId: bestOOS.strategyId,
    verdict,
    isFallback: !options.monthlyReturnsByAsset,
    note: "In-sample strategies use the user's assumption returns while the walk-forward uses trailing 36-month historical means (two different return views).",
  };
}

// ---------------------------------------------------------------------------
// 4. Consensus Portfolio Helper
// ---------------------------------------------------------------------------
export function consensusPortfolio(
  topStrategies: StrategyScorecardItem[],
  constraints: OptimizerConstraints = {}
): ConsensusResult {
  if (topStrategies.length === 0) {
    throw new Error('At least one strategy must be provided for consensus portfolio.');
  }

  const top3 = topStrategies.slice(0, 3);
  const assetIds = Object.keys(top3[0].weights);
  const blended: Record<string, number> = {};

  for (const id of assetIds) {
    let sumW = 0;
    for (const s of top3) {
      sumW += s.weights[id] || 0;
    }
    blended[id] = sumW / top3.length;
  }

  const cleanWeights = cleanAndFinalizeWeights(
    blended,
    assetIds,
    constraints.maxPerAsset ?? 1.0,
    constraints.minPerAsset ?? 0.0
  );

  const avgRet = top3.reduce((s, x) => s + x.inSampleReturn, 0) / top3.length;
  const avgVol = top3.reduce((s, x) => s + x.inSampleVol, 0) / top3.length;
  const avgSharpe = top3.reduce((s, x) => s + x.inSampleSharpe, 0) / top3.length;

  return {
    weights: cleanWeights,
    topStrategies: top3.map((s) => s.strategyName),
    expectedReturn: Number(avgRet.toFixed(4)),
    volatility: Number(avgVol.toFixed(4)),
    sharpe: Number(avgSharpe.toFixed(4)),
  };
}

// ---------------------------------------------------------------------------
// 5. Goal Solver
// ---------------------------------------------------------------------------

/**
 * Solves probability of achieving a target wealth corpus from parametric MC and bootstrap,
 * and finds the minimum monthly SIP required to achieve the goal with 70%, 80%, and 90% confidence.
 * Requirement 3:
 * (a) Replace normal-CDF with empirical probability from simulated terminalValues >= target.
 * (b) Common random numbers (SAME seed for every iteration of bisection) with nSims >= 2000 and Rs 100 precision.
 * (c) Consistency rules:
 *     - if current plan's probReaching >= 70%, minSip70 <= current SIP
 *     - minSip70 <= minSip80 <= minSip90 strictly
 *     - returns both parametric and bootstrap answers
 */
export function solveGoal(input: GoalSolverInput): GoalSolverResult {
  if (!input.covarianceMatrix || input.covarianceMatrix.length !== input.assetIds.length) {
    throw new Error(
      `Dimension mismatch in solveGoal: covarianceMatrix dimension (${input.covarianceMatrix?.length ?? 0}) != assetIds.length (${input.assetIds?.length ?? 0})`
    );
  }

  const { targetAmount, horizonYears, allocation } = input;
  const mode = input.mode ?? 'sip';
  const lumpsumAmount = input.lumpsumAmount ?? (mode === 'lumpsum' || mode === 'both' ? 100000 : 0);
  const monthlySip = input.monthlySip ?? (mode === 'sip' || mode === 'both' ? 10000 : 0);
  const currentAmount = mode === 'sip' ? monthlySip : (mode === 'both' ? (lumpsumAmount + monthlySip) : lumpsumAmount);
  const inflationRate = input.inflationRate ?? 0.06;
  const seed = input.seed ?? 42;

  const defaultSims = input.accurateMode ? 5000 : 2000;
  const nSimsGoal = input.nSims ?? (input.accurateMode !== undefined ? defaultSims : 2500);
  const precision = input.sipPrecision ?? 100;

  // 1. Single Parametric MC with monthlySip = 1 (linear scaling)
  const mcParam = runMonteCarloSimulation({
    allocation,
    years: horizonYears,
    mode: 'sip',
    lumpsumAmount: 0,
    monthlySip: 1,
    assetStats: input.assetStats,
    covarianceMatrix: input.covarianceMatrix,
    assetIds: input.assetIds,
    bondConfig: input.bondConfig,
    nSims: nSimsGoal,
    inflationRate,
    seed,
  });

  const valsMC = mcParam.terminalValues;
  let countMC = 0;
  if (valsMC) {
    for (let i = 0; i < valsMC.length; i++) {
      if (valsMC[i] * currentAmount >= targetAmount) countMC++;
    }
  }
  const probReachingMC = Number(((countMC / mcParam.nSims) * 100).toFixed(1));

  // 2. Single Historical Bootstrap with monthlySip = 1 (linear scaling)
  const mcBoot = historicalBootstrap({
    allocation,
    years: horizonYears,
    mode: 'sip',
    lumpsumAmount: 0,
    monthlySip: 1,
    assetStats: input.assetStats,
    covarianceMatrix: input.covarianceMatrix,
    assetIds: input.assetIds,
    bondConfig: input.bondConfig,
    nSims: nSimsGoal,
    inflationRate,
    seed: seed + 33,
    monthlyReturnsByAsset: input.monthlyReturnsByAsset,
  });

  const valsBoot = mcBoot.terminalValues;
  let countBoot = 0;
  if (valsBoot) {
    for (let i = 0; i < valsBoot.length; i++) {
      if (valsBoot[i] * currentAmount >= targetAmount) countBoot++;
    }
  }
  const probReachingBootstrap = Number(((countBoot / mcBoot.nSims) * 100).toFixed(1));

  // 3. Exact Quantile Inversion for Parametric Minimum SIP
  let infeasibleMC = false;
  const calcMinSipMC = (confidencePct: number): number => {
    if (!valsMC || valsMC.length === 0) return -1;
    const targetIdx = Math.min(valsMC.length - 1, Math.max(0, Math.floor(nSimsGoal * (1 - confidencePct / 100))));
    const q = valsMC[targetIdx];
    if (q <= 0) {
      infeasibleMC = true;
      return -1;
    }
    return Math.ceil((targetAmount / q) / precision) * precision;
  };

  let sip70Param = calcMinSipMC(70);
  if (probReachingMC >= 70 && currentAmount > 0 && sip70Param > currentAmount) {
    sip70Param = Math.round(currentAmount / precision) * precision;
  }
  const sip80Param = Math.max(sip70Param, calcMinSipMC(80));
  const sip90Param = Math.max(sip80Param, calcMinSipMC(90));

  // 4. Exact Quantile Inversion for Bootstrap Minimum SIP
  let infeasibleBoot = false;
  const calcMinSipBoot = (confidencePct: number): number => {
    if (!valsBoot || valsBoot.length === 0) return -1;
    const targetIdx = Math.min(valsBoot.length - 1, Math.max(0, Math.floor(nSimsGoal * (1 - confidencePct / 100))));
    const q = valsBoot[targetIdx];
    if (q <= 0) {
      infeasibleBoot = true;
      return -1;
    }
    return Math.ceil((targetAmount / q) / precision) * precision;
  };

  let sip70Boot = calcMinSipBoot(70);
  if (probReachingBootstrap >= 70 && currentAmount > 0 && sip70Boot > currentAmount) {
    sip70Boot = Math.round(currentAmount / precision) * precision;
  }
  const sip80Boot = Math.max(sip70Boot, calcMinSipBoot(80));
  const sip90Boot = Math.max(sip80Boot, calcMinSipBoot(90));

  return {
    probReachingMC,
    probReachingBootstrap,
    minSip70: sip70Param,
    minSip80: sip80Param,
    minSip90: sip90Param,
    minSip70Bootstrap: sip70Boot,
    minSip80Bootstrap: sip80Boot,
    minSip90Bootstrap: sip90Boot,
    infeasible: infeasibleMC || infeasibleBoot,
    parametric: {
      minSip70: sip70Param,
      minSip80: sip80Param,
      minSip90: sip90Param,
      infeasible: infeasibleMC,
    },
    bootstrap: {
      minSip70: sip70Boot,
      minSip80: sip80Boot,
      minSip90: sip90Boot,
      infeasible: infeasibleBoot,
    },
  };
}
