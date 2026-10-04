/**
 * Bonds & Fixed Deposit Engine
 * Deterministic cash flow schedule, Yield to Maturity (YTM) calculation,
 * credit-risk haircut, and stochastic default event handling for Monte Carlo.
 */
import { BondInput, BondCashFlow, BondAnalysis, CashFlow } from './types.ts';
import { xirr } from './xirr.ts';

/**
 * Returns periods per year for a given payout frequency.
 */
export function getPeriodsPerYear(frequency: BondInput['payoutFrequency']): number {
  switch (frequency) {
    case 'monthly':
      return 12;
    case 'quarterly':
      return 4;
    case 'semi-annual':
      return 2;
    case 'annual':
      return 1;
    case 'cumulative':
      return 1; // Handled as single bullet payoff
  }
}

/**
 * Generates deterministic cash flow schedule for a bond / FD.
 * @param input Bond configuration parameters
 * @param principal Principal investment amount (default: 100,000)
 */
export function generateBondCashflows(
  input: BondInput,
  principal = 100000
): BondCashFlow[] {
  const { couponRate, tenureYears, payoutFrequency } = input;
  const cashflows: BondCashFlow[] = [];
  const startDate = new Date();
  const startYear = startDate.getFullYear();
  const startMonth = startDate.getMonth(); // 0-indexed

  if (payoutFrequency === 'cumulative') {
    // Standard Indian bank FD quarterly compounding: A = P * (1 + r/4)^(4 * t)
    const quarterlyRate = couponRate / 4;
    const totalQuarters = Math.round(tenureYears * 4);
    const maturityAmount = principal * Math.pow(1 + quarterlyRate, totalQuarters);
    const maturityDate = new Date(startYear + tenureYears, startMonth, 1);
    cashflows.push({
      period: 1,
      date: maturityDate.toISOString().slice(0, 10),
      couponPayment: maturityAmount - principal,
      principalPayment: principal,
      totalCashFlow: maturityAmount,
      remainingPrincipal: 0,
    });
    return cashflows;
  }

  const periodsPerYear = getPeriodsPerYear(payoutFrequency);
  const totalPeriods = Math.round(tenureYears * periodsPerYear);
  const monthsPerPeriod = 12 / periodsPerYear;
  const periodicCoupon = principal * (couponRate / periodsPerYear);

  for (let p = 1; p <= totalPeriods; p++) {
    const elapsedMonths = p * monthsPerPeriod;
    const periodDate = new Date(startYear, startMonth + elapsedMonths, 1);
    const isMaturity = p === totalPeriods;
    const principalRepayment = isMaturity ? principal : 0;
    const total = periodicCoupon + principalRepayment;

    cashflows.push({
      period: p,
      date: periodDate.toISOString().slice(0, 10),
      couponPayment: periodicCoupon,
      principalPayment: principalRepayment,
      totalCashFlow: total,
      remainingPrincipal: isMaturity ? 0 : principal,
    });
  }

  return cashflows;
}

/**
 * Computes Yield to Maturity (YTM) and risk-adjusted metrics for a Bond / FD.
 */
export function analyzeBond(input: BondInput, principal = 100000): BondAnalysis {
  const cashflows = generateBondCashflows(input, principal);
  const startDate = new Date();

  const xirrFlows: CashFlow[] = [
    { date: startDate, amount: -principal },
    ...cashflows.map((cf) => ({
      date: new Date(cf.date),
      amount: cf.totalCashFlow,
    })),
  ];

  let computedYtm = 0;
  try {
    computedYtm = xirr(xirrFlows, input.couponRate);
  } catch {
    // If exact XIRR convergence fails, compute effective annual rate: (1 + r/m)^m - 1
    const m = getPeriodsPerYear(input.payoutFrequency);
    computedYtm = Math.pow(1 + input.couponRate / m, m) - 1;
  }

  const totalCoupons = cashflows.reduce((sum, cf) => sum + cf.couponPayment, 0);
  const totalReceived = cashflows.reduce((sum, cf) => sum + cf.totalCashFlow, 0);

  // Credit risk expected annual haircut
  const { defaultProbabilityAnnual, recoveryRate } = input.creditRisk;
  const expectedAnnualLossRate = defaultProbabilityAnnual * (1 - recoveryRate);
  const netExpectedYield = Math.max(0, computedYtm - expectedAnnualLossRate);

  return {
    cashflows,
    ytm: Number(computedYtm.toFixed(6)),
    totalCoupons: Number(totalCoupons.toFixed(2)),
    totalReceived: Number(totalReceived.toFixed(2)),
    expectedAnnualLossRate: Number(expectedAnnualLossRate.toFixed(6)),
    netExpectedYield: Number(netExpectedYield.toFixed(6)),
  };
}

/**
 * Monthly simulation step for Bond / FD inside Monte Carlo simulation.
 * Calculates default hazard check and coupon accrual/reinvestment.
 */
export function stepBondMonthly(
  currentValue: number,
  monthlyYield: number,
  monthlyDefaultProb: number,
  recoveryRate: number,
  hasDefaulted: boolean,
  randUniform: number
): { newValue: number; defaultedThisMonth: boolean; hasDefaulted: boolean } {
  if (hasDefaulted) {
    return { newValue: currentValue, defaultedThisMonth: false, hasDefaulted: true };
  }

  // Credit risk default check
  if (randUniform < monthlyDefaultProb) {
    // Haircut applied: remaining value is only the recovery percentage
    const postDefaultValue = currentValue * recoveryRate;
    return {
      newValue: postDefaultValue,
      defaultedThisMonth: true,
      hasDefaulted: true,
    };
  }

  // Regular growth via accrued yield
  const newValue = currentValue * (1 + monthlyYield);
  return {
    newValue,
    defaultedThisMonth: false,
    hasDefaulted: false,
  };
}
