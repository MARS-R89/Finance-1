/**
 * Stability Diagnostics Engine (Pure Functions)
 * Implements:
 * 1) rollingStats(series, window=60): rolling 5-year annualized return, volatility, and correlation to Nifty 50
 * 2) chowTest(y, x, breakIndex): Chow structural break test in alpha/beta with regularized incomplete beta F CDF
 *    supSupFScan(y, x, trim=0.15): scans candidate break dates for maximum F statistic
 * 3) parameterStability(assetId): summary flag 'stable' | 'watch' | 'unstable' with documented thresholds
 */

import { MonthlyDataPoint } from './types.ts';
import { FALLBACK_ASSET_HISTORY } from './fallbackData.ts';

export interface RollingStatPoint {
  date: string;
  annualizedReturn: number;
  annualizedVol: number;
  correlation: number;
}

export interface ChowTestResult {
  fStat: number;
  pValue: number;
  breakIndex: number;
  breakDate?: string;
  df1: number;
  df2: number;
  rssFull: number;
  rssPre: number;
  rssPost: number;
  preBeta: number;
  postBeta: number;
  preAlpha: number;
  postAlpha: number;
  isSignificant: boolean; // pValue < 0.05
  disclaimer: string;
}

export interface SupFScanResult {
  maxF: number;
  maxFBreakIndex: number;
  maxFBreakDate?: string;
  pValue: number;
  testedDatesCount: number;
  knownBreakResults: { date: string; label: string; result: ChowTestResult | null }[];
  disclaimer: string;
}

export interface ParameterStabilityResult {
  assetId: string;
  status: 'stable' | 'watch' | 'unstable';
  fullSampleVol: number;
  fullSampleCorr: number;
  maxVolRelativeDrift: number; // e.g. 0.28 for 28% deviation
  maxCorrAbsoluteDrift: number; // e.g. 0.15 for 0.15 correlation shift
  minRollingVol: number;
  maxRollingVol: number;
  minRollingCorr: number;
  maxRollingCorr: number;
  reason: string;
}

export const KNOWN_HISTORICAL_BREAK_DATES = [
  { date: '2008-10', label: '2008 Lehman / Global Financial Crisis' },
  { date: '2016-11', label: '2016 India Demonetisation' },
  { date: '2020-03', label: '2020 COVID-19 Market Crash' },
  { date: '2022-02', label: '2022 Geopolitical / Global Rate Hike Shock' },
];

// ---------------------------------------------------------------------------
// 1. Rolling 5-Year (60-Month) Statistics
// ---------------------------------------------------------------------------

/**
 * Computes rolling 5-year (default window=60 months) annualized return, volatility,
 * and correlation to Nifty 50.
 */
export function rollingStats(
  series: (MonthlyDataPoint | number)[],
  window = 60,
  marketSeries?: (MonthlyDataPoint | number)[]
): RollingStatPoint[] {
  // Normalize inputs to monthly returns array and dates array
  const { returns: assetReturns, dates } = extractReturnsAndDates(series);
  const { returns: mktReturns } = marketSeries
    ? extractReturnsAndDates(marketSeries)
    : { returns: extractReturnsAndDates(FALLBACK_ASSET_HISTORY['nifty_50'].series).returns };

  const T = assetReturns.length;
  if (T < window) {
    return [];
  }

  const results: RollingStatPoint[] = [];

  for (let t = window; t <= T; t++) {
    const subAsset = assetReturns.slice(t - window, t);
    const date = dates[t - 1] || `M${t}`;

    // Cumulative product of (1 + r) to compute annualized geometric return
    let cumProd = 1.0;
    for (let i = 0; i < window; i++) {
      cumProd *= Math.max(1e-6, 1 + subAsset[i]);
    }
    const annReturn = Math.pow(cumProd, 12 / window) - 1;

    // Volatility
    const meanA = subAsset.reduce((a, b) => a + b, 0) / window;
    let varA = 0;
    for (let i = 0; i < window; i++) {
      const d = subAsset[i] - meanA;
      varA += d * d;
    }
    const annVol = Math.sqrt(varA / (window - 1)) * Math.sqrt(12);

    // Correlation with market (matching last window months)
    let corr = 0;
    if (mktReturns.length >= t) {
      const subMkt = mktReturns.slice(t - window, t);
      const meanM = subMkt.reduce((a, b) => a + b, 0) / window;
      let varM = 0;
      let covAM = 0;
      for (let i = 0; i < window; i++) {
        const da = subAsset[i] - meanA;
        const dm = subMkt[i] - meanM;
        covAM += da * dm;
        varM += dm * dm;
      }
      const denom = Math.sqrt(varA * varM);
      corr = denom > 1e-12 ? Math.max(-1, Math.min(1, covAM / denom)) : 0;
    }

    results.push({
      date,
      annualizedReturn: Number(annReturn.toFixed(4)),
      annualizedVol: Number(annVol.toFixed(4)),
      correlation: Number(corr.toFixed(4)),
    });
  }

  return results;
}

// ---------------------------------------------------------------------------
// 2. Chow Test for Structural Break & Sup-F Scan
// ---------------------------------------------------------------------------

/**
 * Evaluates OLS regression y = alpha + beta * x and returns RSS and coefficients.
 */
function runOLS(y: number[], x: number[]): { alpha: number; beta: number; rss: number } {
  const n = y.length;
  if (n < 2) return { alpha: 0, beta: 0, rss: 0 };

  const meanX = x.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;

  let covXY = 0;
  let varX = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    covXY += dx * dy;
    varX += dx * dx;
  }

  const beta = varX > 1e-14 ? covXY / varX : 0;
  const alpha = meanY - beta * meanX;

  let rss = 0;
  for (let i = 0; i < n; i++) {
    const res = y[i] - (alpha + beta * x[i]);
    rss += res * res;
  }

  return { alpha, beta, rss };
}

/**
 * Regularized Incomplete Beta Function I_x(a, b)
 * Used to compute the exact cumulative distribution function of the F-distribution:
 * F_CDF(F; d1, d2) = 1 - I_{d2 / (d2 + d1 * F)}(d2 / 2, d1 / 2)
 */
export function incompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;

  // For Chow test with 2 regressors (d1 = 2, b = d1/2 = 1):
  // I_x(a, 1) = x^a directly by analytical integration
  if (Math.abs(b - 1) < 1e-9) {
    return Math.pow(x, a);
  }

  // General continued fraction (Lentz's method)
  const lnBeta = logGamma(a) + logGamma(b) - logGamma(a + b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lnBeta) / a;

  // Symmetry transformation if x > (a + 1) / (a + b + 2)
  if (x > (a + 1) / (a + b + 2)) {
    return 1 - incompleteBeta(1 - x, b, a);
  }

  // Continued fraction expansion
  let f = 1.0;
  let c = 1.0;
  let d = 0.0;
  const tiny = 1e-30;

  for (let m = 1; m <= 200; m++) {
    // Even step: m = 2k
    const m2 = 2 * m;
    let num = -(a + m - 1) * (a + b + m - 1) * x / ((a + m2 - 2) * (a + m2 - 1));
    d = 1.0 + num * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1.0 + num / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1.0 / d;
    f *= c * d;

    // Odd step: m = 2k + 1
    num = m * (b - m) * x / ((a + m2 - 1) * (a + m2));
    d = 1.0 + num * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1.0 + num / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const delta = c * d;
    f *= delta;

    if (Math.abs(delta - 1.0) < 1e-12) break;
  }

  return front * (f - 1.0);
}

/**
 * Log-gamma function (Lanczos approximation)
 */
function logGamma(z: number): number {
  const p = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.138571095836524,
    9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  z -= 1;
  let x = 0.99999999999980993;
  for (let i = 0; i < p.length; i++) {
    x += p[i] / (z + i + 1);
  }
  const t = z + p.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

/**
 * F-distribution Cumulative Survival Function: P(F_stat >= F)
 */
export function fSurvival(fStat: number, d1: number, d2: number): number {
  if (fStat <= 0) return 1.0;
  if (!Number.isFinite(fStat) || Number.isNaN(fStat)) return 1.0;
  const x = d2 / (d2 + d1 * fStat);
  return Math.max(0, Math.min(1, incompleteBeta(x, d2 / 2, d1 / 2)));
}

/**
 * Chow Test for Structural Break in Alpha and Beta.
 * Tests H0: alpha1 = alpha2 AND beta1 = beta2 against H1: structural break at breakIndex.
 */
export function chowTest(
  y: number[],
  x: number[],
  breakIndex: number,
  dates?: string[]
): ChowTestResult {
  const T = y.length;
  const k = 2; // intercept and slope
  const disclaimer = 'indicative only (multiple testing, short samples)';

  if (breakIndex < k + 1 || T - breakIndex < k + 1) {
    return {
      fStat: 0,
      pValue: 1.0,
      breakIndex,
      breakDate: dates?.[breakIndex],
      df1: k,
      df2: Math.max(1, T - 2 * k),
      rssFull: 0,
      rssPre: 0,
      rssPost: 0,
      preBeta: 0,
      postBeta: 0,
      preAlpha: 0,
      postAlpha: 0,
      isSignificant: false,
      disclaimer,
    };
  }

  const yPre = y.slice(0, breakIndex);
  const xPre = x.slice(0, breakIndex);
  const yPost = y.slice(breakIndex);
  const xPost = x.slice(breakIndex);

  const olsFull = runOLS(y, x);
  const olsPre = runOLS(yPre, xPre);
  const olsPost = runOLS(yPost, xPost);

  const rssCombined = olsPre.rss + olsPost.rss;
  const df1 = k;
  const df2 = T - 2 * k;

  let fStat = 0;
  if (rssCombined > 1e-14 && df2 > 0) {
    const numerator = (olsFull.rss - rssCombined) / df1;
    const denominator = rssCombined / df2;
    fStat = Math.max(0, numerator / denominator);
  }

  const pValue = fSurvival(fStat, df1, df2);

  return {
    fStat: Number(fStat.toFixed(4)),
    pValue: Number(pValue.toFixed(6)),
    breakIndex,
    breakDate: dates?.[breakIndex],
    df1,
    df2,
    rssFull: Number(olsFull.rss.toFixed(6)),
    rssPre: Number(olsPre.rss.toFixed(6)),
    rssPost: Number(olsPost.rss.toFixed(6)),
    preBeta: Number(olsPre.beta.toFixed(4)),
    postBeta: Number(olsPost.beta.toFixed(4)),
    preAlpha: Number(olsPre.alpha.toFixed(6)),
    postAlpha: Number(olsPost.alpha.toFixed(6)),
    isSignificant: pValue < 0.05,
    disclaimer,
  };
}

/**
 * Scans candidate break dates between [trim * T, (1 - trim) * T] for the maximum Chow F statistic.
 * Evaluates known historical break dates as well.
 */
export function supSupFScan(
  y: number[],
  x: number[],
  dates?: string[],
  trim = 0.15
): SupFScanResult {
  const T = y.length;
  const startIdx = Math.max(4, Math.floor(trim * T));
  const endIdx = Math.min(T - 4, Math.floor((1 - trim) * T));
  const disclaimer = 'indicative only (multiple testing, short samples)';

  let maxF = -1;
  let maxIdx = startIdx;
  let maxP = 1.0;
  let count = 0;

  for (let idx = startIdx; idx <= endIdx; idx++) {
    const res = chowTest(y, x, idx, dates);
    count++;
    if (res.fStat > maxF) {
      maxF = res.fStat;
      maxIdx = idx;
      maxP = res.pValue;
    }
  }

  // Also test known historical events if date strings are present
  const knownResults = KNOWN_HISTORICAL_BREAK_DATES.map((known) => {
    if (!dates) return { date: known.date, label: known.label, result: null };
    const idx = dates.findIndex((d) => d.startsWith(known.date));
    if (idx >= 4 && idx <= T - 4) {
      return {
        date: known.date,
        label: known.label,
        result: chowTest(y, x, idx, dates),
      };
    }
    return { date: known.date, label: known.label, result: null };
  });

  return {
    maxF: Number(Math.max(0, maxF).toFixed(4)),
    maxFBreakIndex: maxIdx,
    maxFBreakDate: dates?.[maxIdx],
    pValue: Number(maxP.toFixed(6)),
    testedDatesCount: count,
    knownBreakResults: knownResults,
    disclaimer,
  };
}

// ---------------------------------------------------------------------------
// 3. Parameter Stability Diagnostic (Documented Thresholds)
// ---------------------------------------------------------------------------

/**
 * Evaluates parameter stability ('stable' | 'watch' | 'unstable') based on how much
 * rolling 5-year volatility and market correlation move relative to the full-sample value.
 *
 * DOCUMENTED THRESHOLDS:
 * - Volatility Relative Drift = max |rollingVol - fullSampleVol| / fullSampleVol
 *   - Threshold: > 0.40 (40% relative swing) -> 'unstable'
 *   - Threshold: > 0.20 (20% relative swing) -> 'watch'
 *   - Otherwise: 'stable'
 *
 * - Correlation Absolute Drift = max |rollingCorr - fullSampleCorr|
 *   - Threshold: > 0.40 (0.40 absolute shift) -> 'unstable'
 *   - Threshold: > 0.20 (0.20 absolute shift) -> 'watch'
 *   - Otherwise: 'stable'
 *
 * Overall status is the worst of the volatility and correlation flags.
 */
export function parameterStability(
  assetId: string,
  series?: (MonthlyDataPoint | number)[],
  marketSeries?: (MonthlyDataPoint | number)[]
): ParameterStabilityResult {
  const assetData = series || FALLBACK_ASSET_HISTORY[assetId]?.series;
  if (!assetData) {
    return {
      assetId,
      status: 'stable',
      fullSampleVol: 0,
      fullSampleCorr: 0,
      maxVolRelativeDrift: 0,
      maxCorrAbsoluteDrift: 0,
      minRollingVol: 0,
      maxRollingVol: 0,
      minRollingCorr: 0,
      maxRollingCorr: 0,
      reason: 'No series available for evaluation.',
    };
  }

  const { returns: assetReturns } = extractReturnsAndDates(assetData);
  const mktData = marketSeries || FALLBACK_ASSET_HISTORY['nifty_50'].series;
  const { returns: mktReturns } = extractReturnsAndDates(mktData);

  const T = Math.min(assetReturns.length, mktReturns.length);
  const rA = assetReturns.slice(assetReturns.length - T);
  const rM = mktReturns.slice(mktReturns.length - T);

  // Full sample stats
  const meanA = rA.reduce((a, b) => a + b, 0) / T;
  const meanM = rM.reduce((a, b) => a + b, 0) / T;

  let varA = 0;
  let varM = 0;
  let covAM = 0;
  for (let i = 0; i < T; i++) {
    const da = rA[i] - meanA;
    const dm = rM[i] - meanM;
    varA += da * da;
    varM += dm * dm;
    covAM += da * dm;
  }
  const fullSampleVol = Math.sqrt(varA / (T - 1)) * Math.sqrt(12);
  const denom = Math.sqrt(varA * varM);
  const fullSampleCorr = denom > 1e-12 ? covAM / denom : 0;

  // Rolling 60-month statistics
  const rStats = rollingStats(assetData, 60, mktData);

  if (rStats.length === 0) {
    return {
      assetId,
      status: 'stable',
      fullSampleVol: Number(fullSampleVol.toFixed(4)),
      fullSampleCorr: Number(fullSampleCorr.toFixed(4)),
      maxVolRelativeDrift: 0,
      maxCorrAbsoluteDrift: 0,
      minRollingVol: Number(fullSampleVol.toFixed(4)),
      maxRollingVol: Number(fullSampleVol.toFixed(4)),
      minRollingCorr: Number(fullSampleCorr.toFixed(4)),
      maxRollingCorr: Number(fullSampleCorr.toFixed(4)),
      reason: 'Short history (< 60 months); rolling stability assessment deferred.',
    };
  }

  let minVol = Infinity;
  let maxVol = -Infinity;
  let minCorr = Infinity;
  let maxCorr = -Infinity;
  let maxVolDrift = 0;
  let maxCorrDrift = 0;

  for (const pt of rStats) {
    if (pt.annualizedVol < minVol) minVol = pt.annualizedVol;
    if (pt.annualizedVol > maxVol) maxVol = pt.annualizedVol;
    if (pt.correlation < minCorr) minCorr = pt.correlation;
    if (pt.correlation > maxCorr) maxCorr = pt.correlation;

    const volRelDiff = fullSampleVol > 1e-4 ? Math.abs(pt.annualizedVol - fullSampleVol) / fullSampleVol : 0;
    if (volRelDiff > maxVolDrift) maxVolDrift = volRelDiff;

    const corrAbsDiff = Math.abs(pt.correlation - fullSampleCorr);
    if (corrAbsDiff > maxCorrDrift) maxCorrDrift = corrAbsDiff;
  }

  // Apply Documented Thresholds
  let status: 'stable' | 'watch' | 'unstable' = 'stable';
  const reasons: string[] = [];

  if (maxVolDrift > 0.40 || maxCorrDrift > 0.40) {
    status = 'unstable';
    if (maxVolDrift > 0.40) reasons.push(`High rolling volatility drift (${(maxVolDrift * 100).toFixed(1)}% > 40%)`);
    if (maxCorrDrift > 0.40) reasons.push(`High rolling correlation swing (${maxCorrDrift.toFixed(2)} > 0.40)`);
  } else if (maxVolDrift > 0.20 || maxCorrDrift > 0.20) {
    status = 'watch';
    if (maxVolDrift > 0.20) reasons.push(`Moderate rolling volatility drift (${(maxVolDrift * 100).toFixed(1)}% > 20%)`);
    if (maxCorrDrift > 0.20) reasons.push(`Moderate rolling correlation shift (${maxCorrDrift.toFixed(2)} > 0.20)`);
  } else {
    reasons.push('Parameters exhibit strong temporal stability across 5-year rolling windows.');
  }

  return {
    assetId,
    status,
    fullSampleVol: Number(fullSampleVol.toFixed(4)),
    fullSampleCorr: Number(fullSampleCorr.toFixed(4)),
    maxVolRelativeDrift: Number(maxVolDrift.toFixed(4)),
    maxCorrAbsoluteDrift: Number(maxCorrDrift.toFixed(4)),
    minRollingVol: Number(minVol.toFixed(4)),
    maxRollingVol: Number(maxVol.toFixed(4)),
    minRollingCorr: Number(minCorr.toFixed(4)),
    maxRollingCorr: Number(maxCorr.toFixed(4)),
    reason: reasons.join('; '),
  };
}

// ---------------------------------------------------------------------------
// Helper: Extract monthly returns & dates
// ---------------------------------------------------------------------------
export function extractReturnsAndDates(
  series: (MonthlyDataPoint | number)[]
): { returns: number[]; dates: string[] } {
  if (series.length === 0) return { returns: [], dates: [] };

  if (typeof series[0] === 'number') {
    return {
      returns: series as number[],
      dates: (series as number[]).map((_, i) => `M${i + 1}`),
    };
  }

  const dataPoints = series as MonthlyDataPoint[];
  const dates: string[] = [];
  const returns: number[] = [];

  for (let i = 1; i < dataPoints.length; i++) {
    const prev = dataPoints[i - 1].price;
    const curr = dataPoints[i].price;
    const ret = dataPoints[i].return !== undefined ? dataPoints[i].return! : prev > 0 ? curr / prev - 1 : 0;
    returns.push(ret);
    dates.push(dataPoints[i].date);
  }

  return { returns, dates };
}
