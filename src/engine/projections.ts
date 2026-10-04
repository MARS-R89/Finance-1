/**
 * Deterministic Portfolio Projections Engine (Lumpsum & SIP)
 * Simulates month-by-month deterministic asset compounding and calculates
 * portfolio CAGR, year-by-year snapshots, and SIP XIRR.
 */
import {
  PortfolioAllocation,
  DeterministicProjectionResult,
  YearSnapshot,
  BondInput,
  CashFlow,
} from './types.ts';
import { FALLBACK_ASSET_HISTORY } from './fallbackData.ts';
import { analyzeBond } from './bonds.ts';
import { xirr } from './xirr.ts';

// Default bond configuration if none provided
const DEFAULT_BOND_CONFIG: BondInput = {
  couponRate: 0.075,
  tenureYears: 5,
  payoutFrequency: 'annual',
  creditRisk: {
    defaultProbabilityAnnual: 0.005,
    recoveryRate: 0.50,
  },
};

/**
 * Calculates the inflation-adjusted target wealth required to preserve purchasing power.
 * - Lumpsum compounds for the full tenure: lumpsum * (1 + inflationRate)^years
 * - SIP contributions compound from the month deposited:
 *   sum_{m=0}^{totalMonths-1} sip * (1 + inflationRate)^((totalMonths - m) / 12)
 */
export function calculateInflationTarget(
  lumpsum: number,
  sip: number,
  years: number,
  inflationRate: number
): number {
  let target = 0;
  if (lumpsum > 0) {
    target += lumpsum * Math.pow(1 + inflationRate, years);
  }
  if (sip > 0) {
    const totalMonths = Math.round(years * 12);
    for (let m = 0; m < totalMonths; m++) {
      const remainingYears = (totalMonths - m) / 12;
      target += sip * Math.pow(1 + inflationRate, remainingYears);
    }
  }
  return target;
}

/**
 * Normalizes allocation weights so they strictly sum to 1.0.
 */
export function normalizeAllocation(allocation: PortfolioAllocation): PortfolioAllocation {
  const keys = Object.keys(allocation);
  if (keys.length === 0) return {};
  const total = keys.reduce((sum, k) => sum + Math.max(0, allocation[k]), 0);
  if (total <= 0) {
    const equalWeight = 1 / keys.length;
    return keys.reduce((acc, k) => {
      acc[k] = equalWeight;
      return acc;
    }, {} as PortfolioAllocation);
  }
  const normalized: PortfolioAllocation = {};
  for (const k of keys) {
    normalized[k] = Math.max(0, allocation[k]) / total;
  }
  return normalized;
}

/**
 * Resolves annual expected return for an asset (variable or bond/FD).
 */
export function getAssetExpectedReturn(
  assetId: string,
  assetStats?: Record<string, { cagr: number }>,
  bondConfig?: BondInput
): number {
  if (assetId === 'bonds_fd') {
    const analysis = analyzeBond(bondConfig || DEFAULT_BOND_CONFIG);
    return analysis.netExpectedYield;
  }
  if (assetStats && assetStats[assetId] !== undefined) {
    return assetStats[assetId].cagr;
  }
  if (FALLBACK_ASSET_HISTORY[assetId]) {
    return FALLBACK_ASSET_HISTORY[assetId].cagr;
  }
  return 0.12; // 12% default expected return fallback
}

/**
 * Projects a lumpsum investment month-by-month over a given tenure.
 * @param amount Initial investment amount in currency units (e.g., 100,000)
 * @param years Tenure in years
 * @param allocation Portfolio weights mapping assetId -> weight
 * @param assetStats Optional map of annualized expected returns / CAGRs
 * @param bondConfig Optional bond/FD configuration
 */
export function projectLumpsum(
  amount: number,
  years: number,
  allocation: PortfolioAllocation,
  assetStats?: Record<string, { cagr: number }>,
  bondConfig?: BondInput
): DeterministicProjectionResult {
  if (amount <= 0) throw new Error('Investment amount must be greater than zero.');
  if (years <= 0) throw new Error('Investment tenure must be at least 1 year.');

  const weights = normalizeAllocation(allocation);
  const assetIds = Object.keys(weights);
  const totalMonths = Math.round(years * 12);

  // Calculate monthly compounding rates for each asset: (1 + annualCAGR)^(1/12) - 1
  const monthlyRates: Record<string, number> = {};
  const currentAssetValues: Record<string, number> = {};
  const bondTenure = bondConfig?.tenureYears ?? years;
  const bondMaturityMonth = Math.round(bondTenure * 12);
  const reinvestmentRate = bondConfig?.reinvestmentRateAfterMaturity ?? 0.065;
  const monthlyReinvestmentRate = Math.pow(1 + reinvestmentRate, 1 / 12) - 1;

  for (const id of assetIds) {
    const annualReturn = getAssetExpectedReturn(id, assetStats, bondConfig);
    monthlyRates[id] = Math.pow(1 + annualReturn, 1 / 12) - 1;
    currentAssetValues[id] = amount * weights[id];
  }

  const yearByYear: YearSnapshot[] = [];

  // Month-by-month deterministic iteration
  for (let m = 1; m <= totalMonths; m++) {
    for (const id of assetIds) {
      const rate =
        id === 'bonds_fd' && m > bondMaturityMonth ? monthlyReinvestmentRate : monthlyRates[id];
      currentAssetValues[id] *= 1 + rate;
    }

    // Capture annual snapshots
    if (m % 12 === 0) {
      const year = m / 12;
      const totalVal = Object.values(currentAssetValues).reduce((sum, v) => sum + v, 0);
      yearByYear.push({
        year,
        investedCapital: amount,
        portfolioValue: Number(totalVal.toFixed(2)),
        netGains: Number((totalVal - amount).toFixed(2)),
        assetValues: Object.fromEntries(
          Object.entries(currentAssetValues).map(([k, v]) => [k, Number(v.toFixed(2))])
        ),
      });
    }
  }

  const finalValue = Object.values(currentAssetValues).reduce((sum, v) => sum + v, 0);
  const cagr = Math.pow(finalValue / amount, 1 / years) - 1;

  return {
    mode: 'lumpsum',
    initialInvestment: amount,
    years,
    totalInvested: amount,
    finalValue: Number(finalValue.toFixed(2)),
    netGains: Number((finalValue - amount).toFixed(2)),
    cagrOrXirr: Number(cagr.toFixed(6)),
    yearByYear,
    allocation: weights,
  };
}

/**
 * Projects a Systematic Investment Plan (SIP) month-by-month over a given tenure.
 * @param monthlyAmount Regular monthly SIP contribution
 * @param years Tenure in years
 * @param allocation Portfolio weights mapping assetId -> weight
 * @param assetStats Optional map of annualized expected returns / CAGRs
 * @param bondConfig Optional bond/FD configuration
 */
export function projectSIP(
  monthlyAmount: number,
  years: number,
  allocation: PortfolioAllocation,
  assetStats?: Record<string, { cagr: number }>,
  bondConfig?: BondInput
): DeterministicProjectionResult {
  if (monthlyAmount <= 0) throw new Error('Monthly SIP amount must be greater than zero.');
  if (years <= 0) throw new Error('Investment tenure must be at least 1 year.');

  const weights = normalizeAllocation(allocation);
  const assetIds = Object.keys(weights);
  const totalMonths = Math.round(years * 12);
  const bondTenure = bondConfig?.tenureYears ?? years;
  const bondMaturityMonth = Math.round(bondTenure * 12);
  const reinvestmentRate = bondConfig?.reinvestmentRateAfterMaturity ?? 0.065;
  const monthlyReinvestmentRate = Math.pow(1 + reinvestmentRate, 1 / 12) - 1;

  const monthlyRates: Record<string, number> = {};
  const currentAssetValues: Record<string, number> = {};

  for (const id of assetIds) {
    const annualReturn = getAssetExpectedReturn(id, assetStats, bondConfig);
    monthlyRates[id] = Math.pow(1 + annualReturn, 1 / 12) - 1;
    currentAssetValues[id] = 0;
  }

  const yearByYear: YearSnapshot[] = [];
  const cashflows: CashFlow[] = [];
  const startDate = new Date();
  const startYear = startDate.getFullYear();
  const startMonth = startDate.getMonth();

  // Month-by-month SIP execution
  for (let m = 0; m < totalMonths; m++) {
    // Investment at start of each month
    for (const id of assetIds) {
      currentAssetValues[id] += monthlyAmount * weights[id];
    }

    // Cash outflow recorded for XIRR
    const installmentDate = new Date(startYear, startMonth + m, 1);
    cashflows.push({
      date: installmentDate,
      amount: -monthlyAmount,
    });

    // Asset growth over the month
    for (const id of assetIds) {
      const rate =
        id === 'bonds_fd' && m >= bondMaturityMonth ? monthlyReinvestmentRate : monthlyRates[id];
      currentAssetValues[id] *= 1 + rate;
    }

    const monthNum = m + 1;
    if (monthNum % 12 === 0) {
      const year = monthNum / 12;
      const totalVal = Object.values(currentAssetValues).reduce((sum, v) => sum + v, 0);
      const investedSoFar = monthNum * monthlyAmount;
      yearByYear.push({
        year,
        investedCapital: investedSoFar,
        portfolioValue: Number(totalVal.toFixed(2)),
        netGains: Number((totalVal - investedSoFar).toFixed(2)),
        assetValues: Object.fromEntries(
          Object.entries(currentAssetValues).map(([k, v]) => [k, Number(v.toFixed(2))])
        ),
      });
    }
  }

  const finalValue = Object.values(currentAssetValues).reduce((sum, v) => sum + v, 0);
  const totalInvested = totalMonths * monthlyAmount;

  // Add final terminal value inflow on completion date for XIRR
  const endDate = new Date(startYear, startMonth + totalMonths, 1);
  cashflows.push({
    date: endDate,
    amount: finalValue,
  });

  let portfolioXirr = 0;
  try {
    portfolioXirr = xirr(cashflows, 0.12);
  } catch {
    // Approximate annualized SIP return fallback
    const totalGains = finalValue - totalInvested;
    const avgHoldingYears = years / 2;
    portfolioXirr = avgHoldingYears > 0 ? totalGains / (totalInvested * avgHoldingYears) : 0;
  }

  return {
    mode: 'sip',
    monthlySip: monthlyAmount,
    years,
    totalInvested,
    finalValue: Number(finalValue.toFixed(2)),
    netGains: Number((finalValue - totalInvested).toFixed(2)),
    cagrOrXirr: Number(portfolioXirr.toFixed(6)),
    yearByYear,
    allocation: weights,
  };
}
