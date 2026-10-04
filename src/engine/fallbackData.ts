/**
 * Bundled Fallback Dataset for 16 Indian Assets
 * Calibrated against historical Indian market benchmarks.
 * Supports unequal historical lengths:
 * - Flagship Index (nifty_50) & Gold: 372 months (31 years, 1993-10 to 2024-09)
 * - Sector Indices: 180 to 240 months (15 to 20 years)
 * - Mutual Fund Categories: 120 to 180 months (10 to 15 years)
 */
import { MonthlyDataPoint, AssetHistory } from './types.ts';
import { ALL_VARIABLE_ASSETS } from './assets.ts';

export interface AssetCalibration {
  basePrice: number;
  cagrTarget: number;
  volatilityTarget: number;
  marketBeta: number;
  residualVol: number;
  monthsLength?: number; // Configured historical coverage in months
}

export const CALIBRATION: Record<string, AssetCalibration> = {
  nifty_50: { basePrice: 800, cagrTarget: 0.130, volatilityTarget: 0.165, marketBeta: 1.0, residualVol: 0.04, monthsLength: 372 },
  gold: { basePrice: 450, cagrTarget: 0.106, volatilityTarget: 0.128, marketBeta: 0.08, residualVol: 0.12, monthsLength: 372 },
  nifty_bank: { basePrice: 4200, cagrTarget: 0.138, volatilityTarget: 0.225, marketBeta: 1.25, residualVol: 0.08, monthsLength: 240 },
  nifty_it: { basePrice: 2800, cagrTarget: 0.165, volatilityTarget: 0.210, marketBeta: 0.85, residualVol: 0.14, monthsLength: 240 },
  nifty_pharma: { basePrice: 3100, cagrTarget: 0.112, volatilityTarget: 0.168, marketBeta: 0.65, residualVol: 0.12, monthsLength: 220 },
  nifty_fmcg: { basePrice: 5500, cagrTarget: 0.126, volatilityTarget: 0.142, marketBeta: 0.60, residualVol: 0.09, monthsLength: 220 },
  nifty_energy: { basePrice: 2400, cagrTarget: 0.144, volatilityTarget: 0.198, marketBeta: 1.05, residualVol: 0.11, monthsLength: 210 },
  nifty_auto: { basePrice: 2100, cagrTarget: 0.149, volatilityTarget: 0.215, marketBeta: 1.10, residualVol: 0.10, monthsLength: 200 },
  nifty_metal: { basePrice: 1200, cagrTarget: 0.132, volatilityTarget: 0.285, marketBeta: 1.35, residualVol: 0.16, monthsLength: 200 },
  nifty_infra: { basePrice: 1500, cagrTarget: 0.135, volatilityTarget: 0.205, marketBeta: 1.12, residualVol: 0.10, monthsLength: 190 },
  nifty_realty: { basePrice: 110, cagrTarget: 0.181, volatilityTarget: 0.320, marketBeta: 1.45, residualVol: 0.18, monthsLength: 180 },
  mf_large_cap: { basePrice: 14.5, cagrTarget: 0.151, volatilityTarget: 0.160, marketBeta: 0.95, residualVol: 0.04, monthsLength: 180 },
  mf_mid_cap: { basePrice: 18.0, cagrTarget: 0.194, volatilityTarget: 0.198, marketBeta: 1.15, residualVol: 0.06, monthsLength: 160 },
  mf_flexi_cap: { basePrice: 15.2, cagrTarget: 0.174, volatilityTarget: 0.152, marketBeta: 0.90, residualVol: 0.05, monthsLength: 144 },
  mf_small_cap: { basePrice: 16.0, cagrTarget: 0.221, volatilityTarget: 0.238, marketBeta: 1.28, residualVol: 0.09, monthsLength: 132 },
  mf_debt: { basePrice: 10.0, cagrTarget: 0.074, volatilityTarget: 0.024, marketBeta: 0.04, residualVol: 0.02, monthsLength: 120 },
};

// Generates historical monthly prices ending at 2024-09 with unequal lengths per asset
export function generateFallbackSeries(): Record<string, AssetHistory> {
  const maxMonths = 372; // 31 years = 372 monthly points (1993-10 to 2024-09)
  const endYear = 2024;
  const endMonth = 9; // September 2024 (1-indexed)

  const getDateString = (totalMonthOffsetFromStart: number, totalMonthsAvailable: number) => {
    // Total months from 1993-10
    const startTotalMonths = 1993 * 12 + 9; // Oct 1993 = (1993*12 + 9)
    const currentTotal = startTotalMonths + (maxMonths - totalMonthsAvailable) + totalMonthOffsetFromStart;
    const y = Math.floor(currentTotal / 12);
    const m = (currentTotal % 12) + 1;
    return `${y}-${m.toString().padStart(2, '0')}`;
  };

  // Seeded macro market shocks over the full 372 months
  const macroShocks: number[] = [];
  let seed = 123456789;
  const pseudoRandom = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };

  const randomNormal = () => {
    const u1 = Math.max(1e-9, pseudoRandom());
    const u2 = pseudoRandom();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  };

  for (let m = 0; m < maxMonths; m++) {
    let shock = randomNormal() * (0.16 / Math.sqrt(12)); // 16% annualized broad market vol
    // Covid shock in March 2020 (month 317 in a 372-month series starting 1993-10)
    if (m === 317) shock = -0.23;
    if (m === 318) shock = 0.14; // Rebound
    if (m === 325) shock = 0.11;
    // 2008 Lehman shock in Oct 2008 (month 180)
    if (m === 180) shock = -0.25;
    if (m === 187) shock = 0.16;
    macroShocks.push(shock);
  }

  const result: Record<string, AssetHistory> = {};

  for (const asset of ALL_VARIABLE_ASSETS) {
    const cal = CALIBRATION[asset.id] || {
      basePrice: 100,
      cagrTarget: 0.12,
      volatilityTarget: 0.18,
      marketBeta: 1.0,
      residualVol: 0.10,
      monthsLength: 120,
    };

    const assetMonths = cal.monthsLength ?? 120;
    const T = assetMonths - 1;
    const yearsElapsed = T / 12;
    const startOffsetInMacro = maxMonths - assetMonths;

    const rawReturns: number[] = [];
    for (let m = 1; m < assetMonths; m++) {
      const macro = macroShocks[startOffsetInMacro + m];
      const residual = randomNormal() * (cal.residualVol / Math.sqrt(12));
      rawReturns.push(cal.marketBeta * macro + residual);
    }

    const meanRaw = rawReturns.reduce((a, b) => a + b, 0) / T;
    const varRaw = rawReturns.reduce((s, r) => s + Math.pow(r - meanRaw, 2), 0) / (T - 1);
    const stdRaw = Math.sqrt(varRaw);
    const targetMonthlyVol = cal.volatilityTarget / Math.sqrt(12);
    let demeaned = rawReturns.map((r) => (r - meanRaw) * (targetMonthlyVol / Math.max(1e-6, stdRaw)));

    for (let it = 0; it < 3; it++) {
      const delta = Math.log(1 + cal.cagrTarget) / 12;
      const mReturns = demeaned.map((r) => Math.exp(r + delta) - 1);
      const mMean = mReturns.reduce((a, b) => a + b, 0) / T;
      const mVar = mReturns.reduce((s, r) => s + Math.pow(r - mMean, 2), 0) / (T - 1);
      const mAnnVol = Math.sqrt(mVar) * Math.sqrt(12);
      if (mAnnVol > 0) {
        const scale = cal.volatilityTarget / mAnnVol;
        demeaned = demeaned.map((r) => r * scale);
      }
    }

    const delta = Math.log(1 + cal.cagrTarget) / 12;
    const series: MonthlyDataPoint[] = [];
    let currentPrice = cal.basePrice;

    series.push({
      date: getDateString(0, assetMonths),
      price: Number(currentPrice.toFixed(2)),
      return: 0,
    });

    for (let m = 0; m < T; m++) {
      const logRet = demeaned[m] + delta;
      currentPrice = currentPrice * Math.exp(logRet);
      if (currentPrice < 0.01) currentPrice = 0.01;
      series.push({
        date: getDateString(m + 1, assetMonths),
        price: Number(currentPrice.toFixed(2)),
        return: Number((Math.exp(logRet) - 1).toFixed(6)),
      });
    }

    const firstPrice = series[0].price;
    const lastPrice = series[series.length - 1].price;
    const realizedCagr = Math.pow(lastPrice / firstPrice, 1 / yearsElapsed) - 1;
    const returns = series.slice(1).map((s) => s.return || 0);
    const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance =
      returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / (returns.length - 1);
    const realizedVol = Math.sqrt(variance) * Math.sqrt(12);

    result[asset.id] = {
      id: asset.id,
      name: asset.name,
      category: asset.category,
      cagr: Number(realizedCagr.toFixed(4)),
      volatility: Number(realizedVol.toFixed(4)),
      startPrice: firstPrice,
      endPrice: lastPrice,
      monthsCount: series.length,
      startDate: series[0].date,
      endDate: series[series.length - 1].date,
      series,
      isFallback: true,
      firstDate: series[0].date,
      lastDate: series[series.length - 1].date,
      nMonths: series.length,
      source: 'fallback',
      proxyUsed: false,
    };
  }

  return result;
}

export const FALLBACK_ASSET_HISTORY: Record<string, AssetHistory> = generateFallbackSeries();
