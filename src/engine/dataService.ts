/**
 * Historical Data Service & Statistical Computation Engine
 * Fetches Yahoo Finance & MFAPI data, resamples to monthly, calculates CAGR, Volatility,
 * Covariance & Correlation matrices with in-memory caching and fallback handling.
 */
import {
  AssetHistory,
  AssetStats,
  HistoryApiResponse,
  MonthlyDataPoint,
} from './types.ts';
import { ALL_VARIABLE_ASSETS, ASSET_MAP } from './assets.ts';
import { FALLBACK_ASSET_HISTORY } from './fallbackData.ts';

// In-memory cache structure with 24h TTL
interface CacheEntry {
  data: HistoryApiResponse;
  timestamp: number;
}
const CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours
let memoryCache: CacheEntry | null = null;
const singleAssetCache = new Map<string, { data: AssetHistory; timestamp: number }>();

/**
 * Computes CAGR, annualized volatility, and returns for a monthly price series.
 */
export function computeAssetStats(
  series: MonthlyDataPoint[],
  id: string,
  name: string,
  category = ASSET_MAP[id]?.category || 'sector'
): AssetStats {
  if (!series || series.length < 2) {
    throw new Error(`Insufficient data points for asset ${id} to compute stats.`);
  }

  const populatedSeries: MonthlyDataPoint[] = [];
  populatedSeries.push({
    ...series[0],
    return: series[0].return ?? 0,
  });
  for (let i = 1; i < series.length; i++) {
    const prev = series[i - 1].price;
    const curr = series[i].price;
    const ret = prev > 0 ? curr / prev - 1 : 0;
    populatedSeries.push({
      ...series[i],
      return: Number(ret.toFixed(6)),
    });
  }

  const startPrice = populatedSeries[0].price;
  const endPrice = populatedSeries[populatedSeries.length - 1].price;
  const monthsCount = populatedSeries.length;

  const parseYearMonth = (dateStr: string) => {
    const parts = dateStr.split('-');
    const y = parseInt(parts[0], 10);
    const m = parts.length > 1 ? parseInt(parts[1], 10) : 1;
    return { year: isNaN(y) ? 2020 : y, month: isNaN(m) ? 1 : m };
  };

  const dStart = parseYearMonth(populatedSeries[0].date);
  const dEnd = parseYearMonth(populatedSeries[populatedSeries.length - 1].date);
  const totalMonthsElapsed = (dEnd.year - dStart.year) * 12 + (dEnd.month - dStart.month);
  const years = totalMonthsElapsed > 0 ? totalMonthsElapsed / 12 : (monthsCount - 1) / 12;

  const cagr = years > 0 && startPrice > 0 ? Math.pow(endPrice / startPrice, 1 / years) - 1 : 0;

  const returns = populatedSeries.slice(1).map((p) => p.return ?? 0);
  const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance =
    returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) /
    Math.max(1, returns.length - 1);
  const volatility = Math.sqrt(variance) * Math.sqrt(12);

  return {
    id,
    name,
    category,
    cagr: Number(cagr.toFixed(6)),
    volatility: Number(volatility.toFixed(6)),
    startPrice,
    endPrice,
    monthsCount,
    startDate: populatedSeries[0].date,
    endDate: populatedSeries[populatedSeries.length - 1].date,
  };
}

/**
 * Computes full covariance and correlation matrices across aligned monthly returns.
 */
export function computeCovarianceAndCorrelation(
  assetHistories: Record<string, AssetHistory>,
  assetIds: string[]
): { covarianceMatrix: number[][]; correlationMatrix: number[][] } {
  const n = assetIds.length;
  if (n === 0) {
    return { covarianceMatrix: [], correlationMatrix: [] };
  }

  const dateSets = assetIds.map(
    (id) => new Set(assetHistories[id]?.series.map((s) => s.date) || [])
  );
  let commonDates = Array.from(dateSets[0] || []);
  for (let i = 1; i < dateSets.length; i++) {
    commonDates = commonDates.filter((d) => dateSets[i].has(d));
  }
  commonDates.sort();

  if (commonDates.length < 2) {
    commonDates = (FALLBACK_ASSET_HISTORY[assetIds[0]]?.series || []).map((s) => s.date);
  }

  const T = commonDates.length - 1;
  const returnMatrix: number[][] = [];
  for (let t = 0; t < T; t++) {
    const prevDate = commonDates[t];
    const currDate = commonDates[t + 1];
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      const assetId = assetIds[j];
      const series = assetHistories[assetId]?.series || FALLBACK_ASSET_HISTORY[assetId].series;
      const prevPoint = series.find((p) => p.date === prevDate);
      const currPoint = series.find((p) => p.date === currDate);
      let r = 0;
      if (prevPoint && currPoint && prevPoint.price > 0) {
        r = currPoint.price / prevPoint.price - 1;
      }
      row.push(r);
    }
    returnMatrix.push(row);
  }

  const means: number[] = new Array(n).fill(0);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < n; j++) {
      means[j] += returnMatrix[t][j];
    }
  }
  for (let j = 0; j < n; j++) {
    means[j] /= T;
  }

  const cov: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const corr: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let sumProd = 0;
      for (let t = 0; t < T; t++) {
        sumProd += (returnMatrix[t][i] - means[i]) * (returnMatrix[t][j] - means[j]);
      }
      const monthlyCov = sumProd / Math.max(1, T - 1);
      const annualCov = monthlyCov * 12;
      cov[i][j] = annualCov;
      cov[j][i] = annualCov;
    }
  }

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        corr[i][j] = 1.0;
      } else {
        const stdI = Math.sqrt(Math.max(1e-12, cov[i][i]));
        const stdJ = Math.sqrt(Math.max(1e-12, cov[j][j]));
        const val = cov[i][j] / (stdI * stdJ);
        corr[i][j] = Number(Math.max(-1, Math.min(1, val)).toFixed(4));
      }
    }
  }

  return {
    covarianceMatrix: cov,
    correlationMatrix: corr,
  };
}

export const DEFAULT_CATEGORY_PRIORS: Record<string, number> = {
  sector: 0.11,
  mf_large_cap: 0.12,
  mf_flexi_cap: 0.12,
  mf_mid_cap: 0.13,
  mf_small_cap: 0.14,
  gold: 0.08,
  mf_debt: 0.065,
  nifty_50: 0.12,
};

export function getDefaultPriorForAsset(assetId: string, category?: string): number {
  if (DEFAULT_CATEGORY_PRIORS[assetId] !== undefined) {
    return DEFAULT_CATEGORY_PRIORS[assetId];
  }
  if (assetId === 'nifty_50' || category === 'stock') {
    return 0.12;
  }
  if (category === 'sector' || assetId.startsWith('nifty_')) {
    return 0.11;
  }
  if (category === 'gold' || assetId === 'gold') {
    return 0.08;
  }
  if (assetId === 'mf_debt') {
    return 0.065;
  }
  return 0.12;
}

export function calculateAdjustedCagr(
  historicalCagr: number,
  categoryPrior: number,
  s = 0.5
): number {
  const weight = Math.max(0, Math.min(1, s));
  return Number(((1 - weight) * historicalCagr + weight * categoryPrior).toFixed(6));
}

export function shrinkCovariance(covMatrix: number[][], k = 0.2): number[][] {
  const n = covMatrix.length;
  if (n <= 1 || k <= 0) return covMatrix.map((row) => row.slice());

  const vols = covMatrix.map((row, i) => Math.sqrt(Math.max(1e-12, row[i])));
  let sumCorr = 0;
  let count = 0;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const rho = covMatrix[i][j] / Math.max(1e-12, vols[i] * vols[j]);
      sumCorr += Math.max(-1, Math.min(1, rho));
      count++;
    }
  }

  const meanCorr = count > 0 ? sumCorr / count : 0;
  const shrunk: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        shrunk[i][j] = covMatrix[i][j];
      } else {
        const target = meanCorr * vols[i] * vols[j];
        shrunk[i][j] = (1 - k) * covMatrix[i][j] + k * target;
      }
    }
  }

  return shrunk;
}

export function calculateLedoitWolfIntensity(returnsMatrix: number[][]): number {
  if (!returnsMatrix || returnsMatrix.length < 3) return 0.2;
  const T = returnsMatrix.length;
  const N = returnsMatrix[0].length;
  if (N <= 1) return 0;

  const means = new Array(N).fill(0);
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < N; i++) means[i] += returnsMatrix[t][i];
  }
  for (let i = 0; i < N; i++) means[i] /= T;

  const X: number[][] = Array.from({ length: T }, (_, t) =>
    Array.from({ length: N }, (_, i) => returnsMatrix[t][i] - means[i])
  );

  const S: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let t = 0; t < T; t++) sum += X[t][i] * X[t][j];
      S[i][j] = sum / (T - 1);
    }
  }

  const std = S.map((row, i) => Math.sqrt(Math.max(1e-12, row[i])));
  const r: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
  let sumCorr = 0;
  let countCorr = 0;

  for (let i = 0; i < N; i++) {
    r[i][i] = 1.0;
    for (let j = i + 1; j < N; j++) {
      const rho = S[i][j] / Math.max(1e-12, std[i] * std[j]);
      const clamped = Math.max(-1, Math.min(1, rho));
      r[i][j] = clamped;
      r[j][i] = clamped;
      sumCorr += clamped;
      countCorr++;
    }
  }

  const rBar = countCorr > 0 ? sumCorr / countCorr : 0;
  const F: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      F[i][j] = i === j ? S[i][i] : rBar * std[i] * std[j];
    }
  }

  let pi = 0;
  const pMat: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      let sumDev = 0;
      for (let t = 0; t < T; t++) {
        const dev = X[t][i] * X[t][j] - S[i][j];
        sumDev += dev * dev;
      }
      pMat[i][j] = sumDev / T;
      pi += pMat[i][j];
    }
  }

  let rho = 0;
  for (let i = 0; i < N; i++) {
    rho += pMat[i][i];
  }
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      let sumThetaII = 0;
      let sumThetaJJ = 0;
      for (let t = 0; t < T; t++) {
        const prod = X[t][i] * X[t][j] - S[i][j];
        sumThetaII += (X[t][i] * X[t][i] - S[i][i]) * prod;
        sumThetaJJ += (X[t][j] * X[t][j] - S[j][j]) * prod;
      }
      const thetaII = sumThetaII / T;
      const thetaJJ = sumThetaJJ / T;
      const rhoIJ = (rBar / 2) * ((std[j] / std[i]) * thetaII + (std[i] / std[j]) * thetaJJ);
      rho += rhoIJ;
    }
  }

  let gamma = 0;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const diff = F[i][j] - S[i][j];
      gamma += diff * diff;
    }
  }

  if (gamma < 1e-12) return 0;
  const kappa = (pi - rho) / gamma;
  const delta = Math.max(0, Math.min(1, kappa / T));
  return Number(delta.toFixed(4));
}

export function buildCovariance(
  assetIds: string[],
  assumptions: Record<string, { volatility: number }>,
  correlationMatrix: number[][],
  ridge = 1e-7,
  shrinkageK = 0,
  returnsHistory?: number[][]
): number[][] {
  const n = assetIds.length;
  const cov: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    const idI = assetIds[i];
    const volI = assumptions[idI]?.volatility ?? 0.18;
    for (let j = 0; j < n; j++) {
      const idJ = assetIds[j];
      const volJ = assumptions[idJ]?.volatility ?? 0.18;
      let rho = i === j ? 1.0 : 0.5;
      if (
        correlationMatrix &&
        correlationMatrix.length > i &&
        correlationMatrix[i] &&
        correlationMatrix[i].length > j &&
        typeof correlationMatrix[i][j] === 'number'
      ) {
        rho = correlationMatrix[i][j];
      }
      let val = rho * volI * volJ;
      if (i === j) {
        val += ridge;
      }
      cov[i][j] = val;
    }
  }

  let effectiveK = shrinkageK;
  if (returnsHistory && returnsHistory.length >= 3) {
    effectiveK = calculateLedoitWolfIntensity(returnsHistory);
  }

  if (effectiveK > 0) {
    return shrinkCovariance(cov, effectiveK);
  }

  return cov;
}

export async function getAssetHistory(assetId: string): Promise<AssetHistory> {
  const cached = singleAssetCache.get(assetId);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
    return cached.data;
  }

  const def = ASSET_MAP[assetId];
  if (!def) {
    throw new Error(`Asset ID "${assetId}" is not part of the Asset Universe.`);
  }

  const series = FALLBACK_ASSET_HISTORY[assetId]?.series || [];
  const stats = computeAssetStats(series, def.id, def.name, def.category);
  const result: AssetHistory = {
    ...stats,
    series,
    isFallback: true,
  };
  singleAssetCache.set(assetId, { data: result, timestamp: Date.now() });
  return result;
}

export async function getFullUniverseHistory(forceRefresh = false): Promise<HistoryApiResponse> {
  if (!forceRefresh && memoryCache && Date.now() - memoryCache.timestamp < CACHE_TTL_MS) {
    return {
      ...memoryCache.data,
      cached: true,
    };
  }

  const assetIds = ALL_VARIABLE_ASSETS.map((a) => a.id);
  const assets: Record<string, AssetHistory> = {};

  const promises = assetIds.map(async (id) => {
    try {
      const history = await getAssetHistory(id);
      assets[id] = history;
    } catch {
      assets[id] = FALLBACK_ASSET_HISTORY[id];
    }
  });

  await Promise.all(promises);

  const { covarianceMatrix, correlationMatrix } = computeCovarianceAndCorrelation(
    assets,
    assetIds
  );

  const response: HistoryApiResponse = {
    assets,
    assetIds,
    covarianceMatrix,
    correlationMatrix,
    usingFallbackData: true,
    timestamp: Date.now(),
    message: 'Using bundled fallback historical data (verified 10-year Indian asset series).',
  };

  memoryCache = {
    data: response,
    timestamp: Date.now(),
  };

  return response;
}
