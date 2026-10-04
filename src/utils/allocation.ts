/**
 * Indian Rupee (INR) Allocation & Currency Conversion Layer
 * Converts between Rupee allocations (₹ per month) and decimal weights [0, 1].
 */
import { ALL_ASSET_DEFINITIONS } from '../engine/assets.ts';

export interface AssetCategoryGroup {
  id: string;
  name: string;
  description: string;
  assetIds: string[];
}

export const ASSET_GROUPS: AssetCategoryGroup[] = [
  {
    id: 'direct_stocks',
    name: 'Direct Stocks',
    description: 'Flagship index proxy for direct equity holdings',
    assetIds: ['nifty_50'],
  },
  {
    id: 'mutual_funds',
    name: 'Mutual Funds',
    description: 'Diversified actively and passively managed funds',
    assetIds: ['mf_flexi_cap', 'mf_large_cap', 'mf_mid_cap', 'mf_small_cap', 'mf_debt'],
  },
  {
    id: 'sectors',
    name: 'Sector Indices',
    description: 'Thematic Indian industry and sector benchmarks',
    assetIds: [
      'nifty_bank',
      'nifty_it',
      'nifty_pharma',
      'nifty_fmcg',
      'nifty_auto',
      'nifty_metal',
      'nifty_energy',
      'nifty_realty',
      'nifty_infra',
    ],
  },
  {
    id: 'commodities',
    name: 'Gold & Commodities',
    description: 'Physical gold commodity / sovereign gold hedge',
    assetIds: ['gold'],
  },
  {
    id: 'fixed_income',
    name: 'Bonds & Fixed Deposit',
    description: 'Fixed income instruments with coupon schedule and credit default modeling',
    assetIds: ['bonds_fd'],
  },
];

/**
 * Converts a map of Rupee amounts (₹/month) to normalized weights summing strictly to 1.0.
 * Any negative or invalid amounts are treated as 0.
 * If total Rupee amount is 0, defaults to equal weights across active assets.
 */
export function rupeeAllocationToWeights(allocationInr: Record<string, number>): Record<string, number> {
  const cleanAmounts: Record<string, number> = {};
  let totalInr = 0;
  for (const [id, val] of Object.entries(allocationInr)) {
    const amt = typeof val === 'number' && !isNaN(val) && val > 0 ? val : 0;
    if (amt > 0) {
      cleanAmounts[id] = amt;
      totalInr += amt;
    }
  }

  const weights: Record<string, number> = {};
  if (totalInr <= 0) {
    // If empty or zero, assign 0 weights
    for (const asset of ALL_ASSET_DEFINITIONS) {
      weights[asset.id] = 0;
    }
    return weights;
  }

  let weightSum = 0;
  const entries = Object.entries(cleanAmounts);
  for (let i = 0; i < entries.length; i++) {
    const [id, amt] = entries[i];
    const w = amt / totalInr;
    weights[id] = w;
    weightSum += w;
  }

  // Renormalize slightly to prevent floating point inaccuracies
  if (Math.abs(weightSum - 1.0) > 1e-9 && weightSum > 0) {
    for (const id of Object.keys(weights)) {
      weights[id] /= weightSum;
    }
  }

  return weights;
}

/**
 * Converts decimal portfolio weights to Rupee allocations (₹/month) rounded to nearest ₹100.
 * Any rounding residual is allocated to the largest holding so the sum equals monthlyAmount exactly.
 */
export function weightsToRupeeAllocation(
  weights: Record<string, number>,
  monthlyAmount: number
): Record<string, number> {
  if (monthlyAmount <= 0) {
    const zeroAlloc: Record<string, number> = {};
    for (const id of Object.keys(weights)) zeroAlloc[id] = 0;
    return zeroAlloc;
  }

  // Clean weights
  const validWeights: Record<string, number> = {};
  let sumWeight = 0;
  for (const [id, w] of Object.entries(weights)) {
    const cleanW = typeof w === 'number' && !isNaN(w) && w > 0 ? w : 0;
    if (cleanW > 0) {
      validWeights[id] = cleanW;
      sumWeight += cleanW;
    }
  }

  if (sumWeight <= 0) {
    return { nifty_50: monthlyAmount };
  }

  // Normalize
  for (const id of Object.keys(validWeights)) {
    validWeights[id] /= sumWeight;
  }

  const inrAllocation: Record<string, number> = {};
  let currentAllocatedSum = 0;
  let largestAssetId = Object.keys(validWeights)[0];
  let largestWeight = -1;

  for (const [id, w] of Object.entries(validWeights)) {
    // Round to nearest 100
    const rawRupees = w * monthlyAmount;
    const roundedRupees = Math.round(rawRupees / 100) * 100;
    inrAllocation[id] = roundedRupees;
    currentAllocatedSum += roundedRupees;
    if (w > largestWeight) {
      largestWeight = w;
      largestAssetId = id;
    }
  }

  // Residual adjustment on largest asset to ensure exact sum = monthlyAmount
  const residual = monthlyAmount - currentAllocatedSum;
  if (residual !== 0 && largestAssetId) {
    inrAllocation[largestAssetId] = Math.max(0, (inrAllocation[largestAssetId] || 0) + residual);
  }

  return inrAllocation;
}

/**
 * Calculates subtotals per asset group.
 */
export function computeGroupSubtotals(
  allocationInr: Record<string, number>
): Record<string, { totalInr: number; percentage: number }> {
  const total = Object.values(allocationInr).reduce((sum, v) => sum + (v > 0 ? v : 0), 0);
  const result: Record<string, { totalInr: number; percentage: number }> = {};
  for (const group of ASSET_GROUPS) {
    let groupSum = 0;
    for (const id of group.assetIds) {
      groupSum += allocationInr[id] > 0 ? allocationInr[id] : 0;
    }
    result[group.id] = {
      totalInr: groupSum,
      percentage: total > 0 ? Number(((groupSum / total) * 100).toFixed(1)) : 0,
    };
  }
  return result;
}
