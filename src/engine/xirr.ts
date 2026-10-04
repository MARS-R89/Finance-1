/**
 * High-precision XIRR (Extended Internal Rate of Return) Implementation
 * Uses hybrid Newton-Raphson with bounded Bisection fallback for irregular cash flow series.
 */
import { CashFlow } from './types.ts';

const MAX_ITERATIONS = 100;
const TOLERANCE = 1e-7;

/**
 * Calculates the Extended Internal Rate of Return (XIRR) for a schedule of cash flows.
 * @param cashflows List of cash flows with date and amount (negative = outflow, positive = inflow).
 * @param guess Initial annualized discount rate guess (default 0.1 for 10%).
 */
export function xirr(cashflows: CashFlow[], guess = 0.1): number {
  if (!cashflows || cashflows.length < 2) {
    throw new Error('XIRR requires at least 2 cash flows.');
  }

  // Parse and sort cash flows chronologically
  const parsedFlows = cashflows
    .map((cf) => ({
      time: typeof cf.date === 'string' ? new Date(cf.date).getTime() : cf.date.getTime(),
      amount: cf.amount,
    }))
    .filter((cf) => !isNaN(cf.time) && isFinite(cf.amount) && cf.amount !== 0)
    .sort((a, b) => a.time - b.time);

  if (parsedFlows.length < 2) {
    throw new Error('XIRR requires at least 2 non-zero valid cash flows.');
  }

  // Must have at least one positive and one negative cash flow
  const hasPositive = parsedFlows.some((cf) => cf.amount > 0);
  const hasNegative = parsedFlows.some((cf) => cf.amount < 0);
  if (!hasPositive || !hasNegative) {
    throw new Error('XIRR cash flows must include both negative outflows and positive inflows.');
  }

  const d0 = parsedFlows[0].time;

  // Year fraction t_i = (d_i - d_0) / 365.0 days
  const items = parsedFlows.map((cf) => ({
    years: (cf.time - d0) / (365.0 * 24 * 60 * 60 * 1000),
    amount: cf.amount,
  }));

  // Net Present Value function: NPV(r) = sum( C_i * (1 + r)^(-t_i) )
  const npv = (r: number): number => {
    if (r <= -1.0) return Number.NEGATIVE_INFINITY;
    let sum = 0;
    for (let i = 0; i < items.length; i++) {
      sum += items[i].amount * Math.pow(1 + r, -items[i].years);
    }
    return sum;
  };

  // First derivative of NPV with respect to r
  const dNpv = (r: number): number => {
    if (r <= -1.0) return 0;
    let sum = 0;
    for (let i = 0; i < items.length; i++) {
      sum += -items[i].years * items[i].amount * Math.pow(1 + r, -items[i].years - 1);
    }
    return sum;
  };

  // Step 1: Newton-Raphson iteration
  let r = guess;
  for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
    const val = npv(r);
    if (Math.abs(val) < TOLERANCE) {
      return Number(r.toFixed(6));
    }
    const deriv = dNpv(r);
    if (Math.abs(deriv) < 1e-12) {
      break; // Derivative too flat; switch to bisection
    }
    const nextR = r - val / deriv;
    // Prevent divergence beyond realistic boundaries (-0.999 to 100.0)
    if (nextR <= -0.999 || nextR > 100.0 || isNaN(nextR)) {
      break; // Fallback to bisection
    }
    if (Math.abs(nextR - r) < TOLERANCE) {
      return Number(nextR.toFixed(6));
    }
    r = nextR;
  }

  // Step 2: Robust Bisection method across brackets [-0.99, 10.0]
  let low = -0.99;
  let high = 10.0;
  let npvLow = npv(low);
  let npvHigh = npv(high);

  // Expand high bracket if needed
  if (npvLow * npvHigh > 0) {
    high = 100.0;
    npvHigh = npv(high);
  }

  if (npvLow * npvHigh > 0) {
    // If sign doesn't bracket root, return current best approximation
    return Number(r.toFixed(6));
  }

  for (let iter = 0; iter < 120; iter++) {
    const mid = (low + high) / 2;
    const npvMid = npv(mid);
    if (Math.abs(npvMid) < TOLERANCE || (high - low) / 2 < TOLERANCE) {
      return Number(mid.toFixed(6));
    }
    if (npvLow * npvMid < 0) {
      high = mid;
      npvHigh = npvMid;
    } else {
      low = mid;
      npvLow = npvMid;
    }
  }

  return Number(((low + high) / 2).toFixed(6));
}
