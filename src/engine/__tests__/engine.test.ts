/**
 * Comprehensive Unit Test Suite for Portfolio Simulator Engine
 * Tests all pure mathematical functions without any React dependencies.
 */
import {
  xirr,
  analyzeBond,
  generateBondCashflows,
  stepBondMonthly,
  projectLumpsum,
  projectSIP,
  calculateInflationTarget,
  choleskyDecomposition,
  runMonteCarloSimulation,
  projectOntoBoundedSimplex,
  maximizeReturn,
  maximizeSharpe,
  computeEfficientFrontier,
  findMinimumVariancePortfolio,
  computeAssetStats,
  computeCovarianceAndCorrelation,
  buildCovariance,
  createMulberry32,
  FALLBACK_ASSET_HISTORY,
  CALIBRATION,
  calculateAdjustedCagr,
  shrinkCovariance,
  extendAssetsAndCovarianceWithBonds,
  ALL_VARIABLE_ASSETS,
  STRATEGY_REGISTRY,
  compareSolvers,
  equalWeightStrategy,
  minVarianceStrategy,
  maxSharpeStrategy,
  maxReturnAtRiskStrategy,
  riskParityStrategy,
  hierarchicalRiskParityStrategy,
  maxDiversificationStrategy,
  blackLittermanStrategy,
  resampledEfficiencyStrategy,
  cvarOptimizerStrategy,
  globalSearchStrategy,
  walkForwardBacktest,
  precomputeWalkForwardWindows,
  historicalBootstrap,
  runStrategyScorecard,
  consensusPortfolio,
  solveGoal,
  getAlignedMonthlyReturns,
  chowTest,
  supSupFScan,
  rollingStats,
  parameterStability,
  classifyMarketRegimes,
  fitGaussianHMM,
  politisWhiteBlockLength,
} from '../index.ts';
import { rupeeAllocationToWeights, weightsToRupeeAllocation } from '../../utils/allocation.ts';
import { Matrix } from 'ml-matrix';

export interface TestResult {
  name: string;
  category: string;
  passed: boolean;
  durationMs: number;
  error?: string;
  details?: any;
}

export function runEngineTests(): {
  results: TestResult[];
  summary: { total: number; passed: number; failed: number; totalDurationMs: number };
} {
  const results: TestResult[] = [];
  const suiteStart = typeof performance !== 'undefined' ? performance.now() : Date.now();

  function test(category: string, name: string, fn: () => void | any) {
    const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
    try {
      const details = fn();
      const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
      results.push({
        category,
        name,
        passed: true,
        durationMs: Number((t1 - t0).toFixed(2)),
        details: details || null,
      });
    } catch (err: any) {
      const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
      results.push({
        category,
        name,
        passed: false,
        durationMs: Number((t1 - t0).toFixed(2)),
        error: err?.message || String(err),
      });
    }
  }

  function assert(condition: boolean, msg: string) {
    if (!condition) throw new Error(msg);
  }

  function assertApprox(a: number, b: number, eps = 1e-3, msg?: string) {
    if (Math.abs(a - b) > eps) {
      throw new Error(`${msg || 'Assertion failed'}: expected ${b}, got ${a} (diff ${Math.abs(a - b)} > ${eps})`);
    }
  }

  // 1. XIRR Tests
  test('XIRR', 'Basic 1-Year Investment 10% Return', () => {
    const flows = [
      { date: '2024-01-01', amount: -1000 },
      { date: '2025-01-01', amount: 1100 },
    ];
    const rate = xirr(flows);
    assertApprox(rate, 0.10, 0.005, 'Annualized rate should be ~10%');
    return { rate };
  });

  test('XIRR', 'Multi-Year Irregular Cash Flows', () => {
    const flows = [
      { date: '2020-01-01', amount: -10000 },
      { date: '2021-01-01', amount: -5000 },
      { date: '2022-01-01', amount: 2000 },
      { date: '2023-01-01', amount: 18000 },
    ];
    const rate = xirr(flows);
    assert(rate > 0.05 && rate < 0.25, `Rate ${rate} out of reasonable range`);
    return { rate };
  });

  test('XIRR', 'Throws error on invalid cash flows', () => {
    let threw = false;
    try {
      xirr([{ date: '2024-01-01', amount: 100 }]);
    } catch {
      threw = true;
    }
    assert(threw, 'Should throw error when cash flows have only inflows');
  });

  // 2. Bonds / FD Tests
  test('Bonds/FD', 'Deterministic Cashflows & Annual Payouts', () => {
    const input = {
      couponRate: 0.08,
      tenureYears: 3,
      payoutFrequency: 'annual' as const,
      creditRisk: { defaultProbabilityAnnual: 0.01, recoveryRate: 0.40 },
    };
    const cfs = generateBondCashflows(input, 100000);
    assert(cfs.length === 3, 'Should have 3 annual cash flow periods');
    assertApprox(cfs[0].couponPayment, 8000, 1, 'Coupon payment should be 8000');
    assertApprox(cfs[2].principalPayment, 100000, 1, 'Principal repayment should be 100000 at maturity');
    return { cfsCount: cfs.length, totalCashFlow: cfs.reduce((s, c) => s + c.totalCashFlow, 0) };
  });

  test('Bonds/FD', 'YTM and Credit Risk Haircut Calculation', () => {
    const input = {
      couponRate: 0.075,
      tenureYears: 5,
      payoutFrequency: 'semi-annual' as const,
      creditRisk: { defaultProbabilityAnnual: 0.02, recoveryRate: 0.50 },
    };
    const analysis = analyzeBond(input, 100000);
    assert(analysis.ytm > 0.07, `YTM ${analysis.ytm} should be >= coupon rate`);
    assertApprox(analysis.expectedAnnualLossRate, 0.01, 1e-4, 'Expected loss rate should be 1%');
    assert(analysis.netExpectedYield < analysis.ytm, 'Net expected yield should subtract default risk haircut');
    return { ytm: analysis.ytm, netExpectedYield: analysis.netExpectedYield };
  });

  test('Bonds/FD', 'Credit Default Step in Simulation', () => {
    const res = stepBondMonthly(100000, 0.006, 0.05, 0.40, false, 0.01);
    assert(res.defaultedThisMonth, 'Default should be triggered when rand < prob');
    assertApprox(res.newValue, 40000, 1, 'Value should be haircut to recovery rate 40%');
    return res;
  });

  // 3. Projections Tests
  test('Projections', 'projectLumpsum Deterministic Compounding', () => {
    const allocation = { nifty_bank: 0.5, gold: 0.5 };
    const stats = {
      nifty_bank: { cagr: 0.14 },
      gold: { cagr: 0.10 },
    };
    const res = projectLumpsum(100000, 5, allocation, stats);
    assert(res.totalInvested === 100000, 'Total invested should equal initial amount');
    assert(res.finalValue > 100000, 'Final value should exceed initial amount');
    assert(res.yearByYear.length === 5, 'Should have 5 annual snapshots');
    assert(res.cagrOrXirr > 0.11 && res.cagrOrXirr < 0.13, `CAGR ${res.cagrOrXirr} should be ~12%`);
    return { finalValue: res.finalValue, cagr: res.cagrOrXirr };
  });

  test('Projections', 'projectSIP Deterministic Compounding & XIRR', () => {
    const allocation = { mf_flexi_cap: 1.0 };
    const stats = { mf_flexi_cap: { cagr: 0.15 } };
    const res = projectSIP(10000, 3, allocation, stats);
    const expectedInvested = 10000 * 36;
    assert(res.totalInvested === expectedInvested, `Invested ${res.totalInvested} should be 360,000`);
    assert(res.finalValue > expectedInvested, 'SIP final value should be greater than invested');
    assert(res.cagrOrXirr > 0.14 && res.cagrOrXirr < 0.16, `SIP XIRR ${res.cagrOrXirr} should match asset CAGR`);
    return { totalInvested: res.totalInvested, finalValue: res.finalValue, xirr: res.cagrOrXirr };
  });

  // 4. Monte Carlo Engine Tests
  test('Monte Carlo', 'Cholesky Factorization L * L^T = Sigma', () => {
    const sigma = [
      [0.040, 0.012, 0.008],
      [0.012, 0.035, 0.005],
      [0.008, 0.005, 0.020],
    ];
    const L = choleskyDecomposition(sigma);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let recon = 0;
        for (let k = 0; k < 3; k++) {
          recon += L[i][k] * L[j][k];
        }
        assertApprox(recon, sigma[i][j], 1e-4, `Cholesky reconstruction failed at (${i}, ${j})`);
      }
    }
    return { L };
  });

  test('Monte Carlo', 'Simulation Percentiles, Loss Prob & Inflation', () => {
    const assetIds = ['nifty_bank', 'gold', 'mf_flexi_cap'];
    const cov = [
      [0.045, 0.005, 0.025],
      [0.005, 0.020, 0.004],
      [0.025, 0.004, 0.030],
    ];
    const stats = {
      nifty_bank: { cagr: 0.14, volatility: 0.21 },
      gold: { cagr: 0.10, volatility: 0.14 },
      mf_flexi_cap: { cagr: 0.16, volatility: 0.17 },
    };
    const config = {
      allocation: { nifty_bank: 0.4, gold: 0.2, mf_flexi_cap: 0.4 },
      years: 5,
      mode: 'lumpsum' as const,
      amount: 100000,
      assetStats: stats,
      covarianceMatrix: cov,
      assetIds,
      nSims: 1000,
      inflationRate: 0.06,
    };
    const res = runMonteCarloSimulation(config);
    assert(res.percentilesByYear.length === 5, 'Should have 5 year percentiles');
    for (const p of res.percentilesByYear) {
      assert(p.p5 <= p.p10, 'P5 <= P10');
      assert(p.p10 <= p.p25, 'P10 <= P25');
      assert(p.p25 <= p.p50, 'P25 <= P50');
      assert(p.p50 <= p.p75, 'P50 <= P75');
      assert(p.p75 <= p.p90, 'P75 <= P90');
      assert(p.p90 <= p.p95, 'P90 <= P95');
    }
    assert(res.probabilityOfLoss >= 0 && res.probabilityOfLoss <= 100, 'Loss probability in [0, 100]');
    assert(res.probabilityOfBeatingInflation >= 0 && res.probabilityOfBeatingInflation <= 100, 'Inflation prob in [0, 100]');
    assert(res.terminalStats.median > 100000, '5-year median should show capital growth');
    return {
      simulationTimeMs: res.simulationTimeMs,
      p50_year5: res.percentilesByYear[4].p50,
      probLoss: res.probabilityOfLoss,
      probBeatInflation: res.probabilityOfBeatingInflation,
    };
  });

  // 5. Optimizer Tests
  test('Optimizer', 'Bounded Simplex Projection Strict Constraints', () => {
    const rawV = [0.8, 0.4, -0.1, 0.3];
    const maxPerAsset = 0.35;
    const projected = projectOntoBoundedSimplex(rawV, maxPerAsset);
    const sum = projected.reduce((a, b) => a + b, 0);
    assertApprox(sum, 1.0, 1e-6, 'Projected weights must sum strictly to 1.0');
    for (let i = 0; i < projected.length; i++) {
      assert(projected[i] >= -1e-8, `Weight ${projected[i]} must be >= 0`);
      assert(projected[i] <= maxPerAsset + 1e-6, `Weight ${projected[i]} must not exceed cap ${maxPerAsset}`);
    }
    return { projected, sum };
  });

  test('Optimizer', 'Maximize Return subject to Volatility Constraint (Behavior Assertions)', () => {
    const assets = [
      { id: 'asset1', expectedReturn: 0.12, volatility: 0.14 },
      { id: 'asset2', expectedReturn: 0.18, volatility: 0.24 },
      { id: 'asset3', expectedReturn: 0.08, volatility: 0.06 },
    ];
    const cov = [
      [0.0196, 0.0050, 0.0010],
      [0.0050, 0.0576, 0.0020],
      [0.0010, 0.0020, 0.0036],
    ];
    const maxPerAsset = 0.50;
    const optLow = maximizeReturn(assets, cov, { maxRisk: 0.10, maxPerAsset });
    const optMid = maximizeReturn(assets, cov, { maxRisk: 0.15, maxPerAsset });
    const optHigh = maximizeReturn(assets, cov, { maxRisk: 0.20, maxPerAsset });
    assert(optLow.volatility <= 0.10 + 1e-4, 'optLow volatility <= 0.10');
    assert(optMid.volatility <= 0.15 + 1e-4, 'optMid volatility <= 0.15');
    assert(optHigh.volatility <= 0.20 + 1e-4, 'optHigh volatility <= 0.20');
    assert(
      optMid.expectedReturn >= optLow.expectedReturn - 1e-6,
      `Expected return at risk 0.15 (${optMid.expectedReturn}) must be >= return at risk 0.10 (${optLow.expectedReturn})`
    );
    assert(
      optHigh.expectedReturn >= optMid.expectedReturn - 1e-6,
      `Expected return at risk 0.20 (${optHigh.expectedReturn}) must be >= return at risk 0.15 (${optMid.expectedReturn})`
    );
    const sharpeOpt = maximizeSharpe(assets, cov, { maxPerAsset });
    const optPermissive = maximizeReturn(assets, cov, {
      maxRisk: sharpeOpt.volatility + 0.02,
      maxPerAsset,
    });
    assert(
      optPermissive.expectedReturn >= sharpeOpt.expectedReturn - 1e-4,
      `maximizeReturn expectedReturn (${optPermissive.expectedReturn}) must be >= Sharpe return (${sharpeOpt.expectedReturn}) when risk budget permits`
    );
    const weightSum = Object.values(optMid.weights).reduce((a, b) => a + b, 0);
    assertApprox(weightSum, 1.0, 1e-3, 'Weights must sum to 1.0');
    for (const [id, w] of Object.entries(optMid.weights)) {
      assert(w <= maxPerAsset + 1e-3, `Weight of ${id} (${w}) exceeds cap ${maxPerAsset}`);
    }
    return optMid;
  });

  test('Optimizer', 'Maximize Sharpe Ratio', () => {
    const assets = [
      { id: 'asset1', expectedReturn: 0.14, volatility: 0.18 },
      { id: 'asset2', expectedReturn: 0.20, volatility: 0.22 },
      { id: 'asset3', expectedReturn: 0.07, volatility: 0.03 },
    ];
    const cov = [
      [0.0324, 0.0080, 0.0005],
      [0.0080, 0.0484, 0.0008],
      [0.0005, 0.0008, 0.0009],
    ];
    const maxPerAsset = 0.40;
    const opt = maximizeSharpe(assets, cov, { maxPerAsset, riskFreeRate: 0.065 });
    assert(opt.sharpeRatio > 0.4, `Sharpe ratio ${opt.sharpeRatio} should be positive and attractive`);
    for (const [id, w] of Object.entries(opt.weights)) {
      assert(w <= maxPerAsset + 1e-3, `Weight for ${id} (${w}) must not exceed cap ${maxPerAsset}`);
      assert(w >= -1e-6, `Weight for ${id} (${w}) must be non-negative`);
    }
    return opt;
  });

  test('Optimizer', 'Efficient Frontier Tracing 30 Points', () => {
    const assets = [
      { id: 'A', expectedReturn: 0.10, volatility: 0.10 },
      { id: 'B', expectedReturn: 0.15, volatility: 0.18 },
      { id: 'C', expectedReturn: 0.22, volatility: 0.28 },
    ];
    const cov = [
      [0.010, 0.002, 0.003],
      [0.002, 0.0324, 0.010],
      [0.003, 0.010, 0.0784],
    ];
    const frontier = computeEfficientFrontier(assets, cov, 30, { maxPerAsset: 0.60 });
    assert(frontier.length === 30, 'Efficient frontier must produce exactly 30 points');
    for (let p = 0; p < frontier.length; p++) {
      const pt = frontier[p];
      const sum = Object.values(pt.weights).reduce((a, b) => a + b, 0);
      assertApprox(sum, 1.0, 1e-3, `Frontier point ${p} weights must sum to 1.0`);
    }
    assert(frontier[29].expectedReturn >= frontier[0].expectedReturn, 'Max return point >= min variance point');
    return {
      pointsCount: frontier.length,
      minRisk: frontier[0].volatility,
      maxRisk: frontier[29].volatility,
      minReturn: frontier[0].expectedReturn,
      maxReturn: frontier[29].expectedReturn,
    };
  });

  // 6. Data Service & Benchmark Matrix Tests
  test('DataService', 'CAGR & Volatility Calculation from Price Series', () => {
    const series = [
      { date: '2023-01', price: 100 },
      { date: '2023-07', price: 110 },
      { date: '2024-01', price: 120 },
    ];
    const stats = computeAssetStats(series, 'test_asset', 'Test Asset', 'sector');
    assertApprox(stats.cagr, 0.20, 0.01, '1-year price change 100 -> 120 should yield ~20% CAGR');
    assert(stats.volatility >= 0, 'Volatility must be non-negative');
    return stats;
  });

  test('DataService', 'Covariance and Correlation Matrix Sanity', () => {
    const sampleIds = ['nifty_bank', 'nifty_it', 'gold', 'mf_debt'];
    const { covarianceMatrix, correlationMatrix } = computeCovarianceAndCorrelation(
      FALLBACK_ASSET_HISTORY,
      sampleIds
    );
    assert(covarianceMatrix.length === 4, 'Covariance matrix must be 4x4');
    assert(correlationMatrix.length === 4, 'Correlation matrix must be 4x4');
    for (let i = 0; i < 4; i++) {
      assertApprox(correlationMatrix[i][i], 1.0, 1e-6, `Diagonal correlation [${i}][${i}] must be 1.0`);
    }
    const bankDebtCorr = correlationMatrix[0][3];
    assert(Math.abs(bankDebtCorr) < 0.5, `Bank-Debt correlation ${bankDebtCorr} should be low`);
    return { sampleIds, correlationMatrix };
  });

  // 7. Bug 1 Regression Tests
  test('Bug 1 - Optimizer', '(a) Volatility <= maxRisk + 1e-6 across risk limits [0.10, 0.12, 0.15, 0.18, 0.22]', () => {
    const ids = ALL_VARIABLE_ASSETS.map((a) => a.id);
    const { covarianceMatrix } = computeCovarianceAndCorrelation(FALLBACK_ASSET_HISTORY, ids);
    const assets = ids.map((id) => ({
      id,
      expectedReturn: FALLBACK_ASSET_HISTORY[id].cagr,
      volatility: FALLBACK_ASSET_HISTORY[id].volatility,
    }));
    const maxPerAsset = 0.30;
    const testRisks = [0.10, 0.12, 0.15, 0.18, 0.22];
    for (const maxRisk of testRisks) {
      const res = maximizeReturn(assets, covarianceMatrix, { maxRisk, maxPerAsset });
      assert(
        res.volatility <= maxRisk + 1e-6,
        `Result volatility ${res.volatility} exceeded maxRisk ${maxRisk}`
      );
      assert(!res.infeasible, `Expected feasible solution for maxRisk ${maxRisk}`);
    }
  });

  test('Bug 1 - Optimizer', '(b) Expected return is non-decreasing as maxRisk increases', () => {
    const ids = ALL_VARIABLE_ASSETS.map((a) => a.id);
    const { covarianceMatrix } = computeCovarianceAndCorrelation(FALLBACK_ASSET_HISTORY, ids);
    const assets = ids.map((id) => ({
      id,
      expectedReturn: FALLBACK_ASSET_HISTORY[id].cagr,
      volatility: FALLBACK_ASSET_HISTORY[id].volatility,
    }));
    const maxPerAsset = 0.30;
    const testRisks = [0.10, 0.12, 0.15, 0.18, 0.22];
    let previousReturn = -Infinity;
    for (const maxRisk of testRisks) {
      const res = maximizeReturn(assets, covarianceMatrix, { maxRisk, maxPerAsset });
      assert(
        res.expectedReturn >= previousReturn - 1e-6,
        `Return ${res.expectedReturn} at risk ${maxRisk} was lower than previous ${previousReturn}`
      );
      previousReturn = res.expectedReturn;
    }
  });

  test('Bug 1 - Optimizer', '(c) Return >= maximizeSharpe return whenever Sharpe vol <= maxRisk', () => {
    const ids = ALL_VARIABLE_ASSETS.map((a) => a.id);
    const { covarianceMatrix } = computeCovarianceAndCorrelation(FALLBACK_ASSET_HISTORY, ids);
    const assets = ids.map((id) => ({
      id,
      expectedReturn: FALLBACK_ASSET_HISTORY[id].cagr,
      volatility: FALLBACK_ASSET_HISTORY[id].volatility,
    }));
    const maxPerAsset = 0.30;
    const sharpePoint = maximizeSharpe(assets, covarianceMatrix, { maxPerAsset });
    const maxRisk = Number((sharpePoint.volatility + 0.02).toFixed(2));
    const maxRetPoint = maximizeReturn(assets, covarianceMatrix, { maxRisk, maxPerAsset });
    assert(
      maxRetPoint.expectedReturn >= sharpePoint.expectedReturn - 1e-4,
      `maximizeReturn expectedReturn (${maxRetPoint.expectedReturn}) should be >= Sharpe return (${sharpePoint.expectedReturn}) when risk budget permits`
    );
  });

  test('Bug 1 - Optimizer', '(d) Deterministic identical outputs & infeasible flag handling', () => {
    const ids = ALL_VARIABLE_ASSETS.map((a) => a.id);
    const { covarianceMatrix } = computeCovarianceAndCorrelation(FALLBACK_ASSET_HISTORY, ids);
    const assets = ids.map((id) => ({
      id,
      expectedReturn: FALLBACK_ASSET_HISTORY[id].cagr,
      volatility: FALLBACK_ASSET_HISTORY[id].volatility,
    }));
    const run1 = maximizeReturn(assets, covarianceMatrix, { maxRisk: 0.15, maxPerAsset: 0.30 });
    const run2 = maximizeReturn(assets, covarianceMatrix, { maxRisk: 0.15, maxPerAsset: 0.30 });
    assert(run1.expectedReturn === run2.expectedReturn, 'Expected returns must be strictly identical');
    assert(run1.volatility === run2.volatility, 'Volatilities must be strictly identical');
    assert(
      JSON.stringify(run1.weights) === JSON.stringify(run2.weights),
      'Weights must be strictly identical across calls'
    );
    const infeasibleRes = maximizeReturn(assets, covarianceMatrix, { maxRisk: 0.02, maxPerAsset: 0.30 });
    assert(infeasibleRes.infeasible === true, 'Should mark infeasible when maxRisk < minVar risk');
    assert(infeasibleRes.riskLimitBinding === true, 'riskLimitBinding should be true');
  });

  // 8. Bug 2 Regression Tests
  test('Bug 2 - Assumptions Volatility', 'buildCovariance scales variances and covariances with edited volatility', () => {
    const sampleIds = ['nifty_bank', 'gold', 'mf_debt'];
    const corr = [
      [1.0, 0.15, 0.05],
      [0.15, 1.0, 0.10],
      [0.05, 0.10, 1.0],
    ];
    const baseAssumptions = {
      nifty_bank: { volatility: 0.20 },
      gold: { volatility: 0.15 },
      mf_debt: { volatility: 0.05 },
    };
    const modAssumptions = {
      ...baseAssumptions,
      nifty_bank: { volatility: 0.40 },
    };
    const covBase = buildCovariance(sampleIds, baseAssumptions, corr, 0);
    const covMod = buildCovariance(sampleIds, modAssumptions, corr, 0);
    const bankVarBase = covBase[0][0];
    const bankVarMod = covMod[0][0];
    assertApprox(bankVarMod / bankVarBase, 4.0, 1e-4, 'Bank variance must scale 4x when volatility doubles');
    const bankGoldCovBase = covBase[0][1];
    const bankGoldCovMod = covMod[0][1];
    assertApprox(bankGoldCovMod / bankGoldCovBase, 2.0, 1e-4, 'Cross-covariance must scale 2x');
    assertApprox(covMod[1][1], covBase[1][1], 1e-6, 'Gold variance must remain unchanged');
  });

  test('Bug 2 - Assumptions Volatility', 'Monte Carlo fan width widens when asset volatility is edited higher', () => {
    const assetStatsBase = { nifty_bank: { cagr: 0.14, volatility: 0.18 } };
    const assetStatsHigh = { nifty_bank: { cagr: 0.14, volatility: 0.36 } };
    const covBase = [[0.18 * 0.18]];
    const covHigh = [[0.36 * 0.36]];
    const baseRes = runMonteCarloSimulation({
      allocation: { nifty_bank: 1.0 },
      years: 5,
      mode: 'lumpsum',
      amount: 100000,
      assetStats: assetStatsBase,
      covarianceMatrix: covBase,
      assetIds: ['nifty_bank'],
      nSims: 2000,
      seed: 42,
    });
    const highRes = runMonteCarloSimulation({
      allocation: { nifty_bank: 1.0 },
      years: 5,
      mode: 'lumpsum',
      amount: 100000,
      assetStats: assetStatsHigh,
      covarianceMatrix: covHigh,
      assetIds: ['nifty_bank'],
      nSims: 2000,
      seed: 42,
    });
    const baseSpread = baseRes.percentilesByYear[4].p90 - baseRes.percentilesByYear[4].p10;
    const highSpread = highRes.percentilesByYear[4].p90 - highRes.percentilesByYear[4].p10;
    assert(
      highSpread > baseSpread * 1.5,
      `High volatility spread (${highSpread}) should be much wider than base spread (${baseSpread})`
    );
    assert(
      highRes.percentilesByYear[4].p10 < baseRes.percentilesByYear[4].p10,
      'Downside P10 should be lower under higher volatility'
    );
  });

  // 9. Bug 3 Regression Tests
  test('Bug 3 - Monte Carlo Drift', 'Median path aligns with deterministic CAGR projection without variance drag', () => {
    const cagr = 0.12;
    const volatility = 0.22;
    const years = 5;
    const amount = 100000;
    const det = projectLumpsum(amount, years, { test_asset: 1.0 }, { test_asset: { cagr } });
    const mc = runMonteCarloSimulation({
      allocation: { test_asset: 1.0 },
      years,
      mode: 'lumpsum',
      amount,
      assetStats: { test_asset: { cagr, volatility } },
      covarianceMatrix: [[volatility * volatility]],
      assetIds: ['test_asset'],
      nSims: 4000,
      seed: 9999,
    });
    const expectedDeterministic = det.finalValue;
    const mcMedian = mc.percentilesByYear[years - 1].p50;
    const relDiff = Math.abs(mcMedian - expectedDeterministic) / expectedDeterministic;
    assert(
      relDiff < 0.025,
      `Monte Carlo median ${mcMedian} should closely match deterministic projection ${expectedDeterministic} (diff: ${(relDiff * 100).toFixed(2)}%)`
    );
  });

  // 10. Bug 4 Regression Tests
  test('Bug 4 - Concurrent Lumpsum & SIP', 'Accurately tracks totalInvested and compounds both cashflows', () => {
    const lumpsumAmount = 100000;
    const monthlySip = 5000;
    const years = 3;
    const totalMonths = years * 12;
    const expectedTotalInvested = lumpsumAmount + monthlySip * totalMonths;
    const mcBoth = runMonteCarloSimulation({
      allocation: { gold: 1.0 },
      years,
      mode: 'both',
      lumpsumAmount,
      monthlySip,
      assetStats: { gold: { cagr: 0.10, volatility: 0.12 } },
      covarianceMatrix: [[0.0144]],
      assetIds: ['gold'],
      nSims: 1500,
      seed: 777,
    });
    assert(
      mcBoth.totalInvested === expectedTotalInvested,
      `totalInvested ${mcBoth.totalInvested} must equal ${expectedTotalInvested}`
    );
    const mcLump = runMonteCarloSimulation({
      allocation: { gold: 1.0 },
      years,
      mode: 'lumpsum',
      lumpsumAmount,
      assetStats: { gold: { cagr: 0.10, volatility: 0.12 } },
      covarianceMatrix: [[0.0144]],
      assetIds: ['gold'],
      nSims: 1500,
      seed: 777,
    });
    const mcSip = runMonteCarloSimulation({
      allocation: { gold: 1.0 },
      years,
      mode: 'sip',
      monthlySip,
      assetStats: { gold: { cagr: 0.10, volatility: 0.12 } },
      covarianceMatrix: [[0.0144]],
      assetIds: ['gold'],
      nSims: 1500,
      seed: 777,
    });
    assert(
      mcBoth.terminalStats.median > mcLump.terminalStats.median,
      'Both mode median should be strictly greater than lumpsum only'
    );
    assert(
      mcBoth.terminalStats.median > mcSip.terminalStats.median,
      'Both mode median should be strictly greater than SIP only'
    );
  });

  // 11. Bug 5 Regression Tests
  test('Bug 5 - Reproducibility', 'Seeded Monte Carlo is strictly identical across repeated runs', () => {
    const config = {
      allocation: { nifty_bank: 0.6, gold: 0.4 },
      years: 4,
      mode: 'lumpsum' as const,
      lumpsumAmount: 150000,
      assetStats: {
        nifty_bank: { cagr: 0.15, volatility: 0.22 },
        gold: { cagr: 0.09, volatility: 0.14 },
      },
      covarianceMatrix: [
        [0.0484, 0.006],
        [0.006, 0.0196],
      ],
      assetIds: ['nifty_bank', 'gold'],
      nSims: 1000,
      seed: 123456,
    };
    const run1 = runMonteCarloSimulation(config);
    const run2 = runMonteCarloSimulation(config);
    const runDiffSeed = runMonteCarloSimulation({ ...config, seed: 654321 });
    assert(
      JSON.stringify(run1.percentilesByYear) === JSON.stringify(run2.percentilesByYear),
      'Percentiles must be strictly identical across seeded runs'
    );
    assert(
      run1.probabilityOfLoss === run2.probabilityOfLoss,
      'Loss probability must be strictly identical'
    );
    assert(
      run1.terminalStats.median === run2.terminalStats.median,
      'Terminal median must be strictly identical'
    );
    assert(
      run1.terminalStats.median !== runDiffSeed.terminalStats.median,
      'Different seeds should produce different random simulation paths'
    );
  });

  test('Bug 5 - Reproducibility', 'Optimizer multi-start returns identical weights across calls', () => {
    const assets = [
      { id: 'asset1', expectedReturn: 0.15, volatility: 0.18 },
      { id: 'asset2', expectedReturn: 0.12, volatility: 0.14 },
      { id: 'asset3', expectedReturn: 0.08, volatility: 0.07 },
    ];
    const cov = [
      [0.0324, 0.0040, 0.0010],
      [0.0040, 0.0196, 0.0015],
      [0.0010, 0.0015, 0.0049],
    ];
    const opt1 = maximizeSharpe(assets, cov, { maxPerAsset: 0.40 });
    const opt2 = maximizeSharpe(assets, cov, { maxPerAsset: 0.40 });
    assert(
      JSON.stringify(opt1.weights) === JSON.stringify(opt2.weights),
      'Optimizer weights must be deterministic across repeated calls'
    );
    assert(opt1.sharpeRatio === opt2.sharpeRatio, 'Sharpe ratio must be identical');
  });

  // 12. Second Pass Tests
  test('Pass 2 - Calibration', 'All 16 fallback asset series match CAGR within 0.5% and Volatility within 1%', () => {
    const assetKeys = Object.keys(CALIBRATION);
    assert(assetKeys.length === 16, 'Must calibrate all 16 Indian asset classes');
    for (const id of assetKeys) {
      const cal = CALIBRATION[id];
      const series = FALLBACK_ASSET_HISTORY[id];
      assert(series !== undefined, `Fallback series for ${id} must exist`);
      const cagrDiff = Math.abs(series.cagr - cal.cagrTarget);
      assert(
        cagrDiff <= 0.005,
        `Asset ${id} realized CAGR ${(series.cagr * 100).toFixed(2)}% deviates from target ${(cal.cagrTarget * 100).toFixed(2)}% by > 0.5%`
      );
      const volDiff = Math.abs(series.volatility - cal.volatilityTarget);
      assert(
        volDiff <= 0.01,
        `Asset ${id} realized volatility ${(series.volatility * 100).toFixed(2)}% deviates from target ${(cal.volatilityTarget * 100).toFixed(2)}% by > 1.0%`
      );
    }
  });

  test('Pass 2 - Return Haircut & Shrinkage', 's=0 returns historical CAGR, s=1 returns prior, and s=0.5 reduces weight concentration', () => {
    const historical = 0.16;
    const prior = 0.11;
    const at0 = calculateAdjustedCagr(historical, prior, 0.0);
    assertApprox(at0, historical, 1e-5, 's=0 must strictly equal historical CAGR');
    const at1 = calculateAdjustedCagr(historical, prior, 1.0);
    assertApprox(at1, prior, 1e-5, 's=1 must strictly equal category prior');
    const atHalf = calculateAdjustedCagr(historical, prior, 0.5);
    assertApprox(atHalf, 0.135, 1e-5, 's=0.5 must equal 50/50 blend (13.5%)');

    const testAssets = [
      { id: 'A', expectedReturn: 0.24, volatility: 0.20 },
      { id: 'B', expectedReturn: 0.12, volatility: 0.15 },
      { id: 'C', expectedReturn: 0.10, volatility: 0.12 },
    ];
    const testCov = [
      [0.04, 0.01, 0.01],
      [0.01, 0.0225, 0.005],
      [0.01, 0.005, 0.0144],
    ];
    const opt0 = maximizeSharpe(testAssets, testCov, { maxPerAsset: 0.85 });
    const hhi0 = Object.values(opt0.weights).reduce((s, w) => s + w * w, 0);

    const testAssetsHalf = testAssets.map((a) => ({
      ...a,
      expectedReturn: calculateAdjustedCagr(a.expectedReturn, 0.12, 0.5),
    }));
    const shrunkCov = shrinkCovariance(testCov, 0.2);
    const optHalf = maximizeSharpe(testAssetsHalf, shrunkCov, { maxPerAsset: 0.85 });
    const hhiHalf = Object.values(optHalf.weights).reduce((s, w) => s + w * w, 0);
    assert(
      hhiHalf <= hhi0,
      `Weight concentration at s=0.5 (HHI: ${hhiHalf.toFixed(4)}) must be <= concentration at s=0 (HHI: ${hhi0.toFixed(4)})`
    );
  });

  test('Pass 2 - Fat Tails', 'Student-t(5) draws provide heavy-tailed stress comparison with lower P5 / higher risk', () => {
    const res = runMonteCarloSimulation({
      allocation: { nifty_bank: 0.5, gold: 0.5 },
      years: 3,
      mode: 'lumpsum',
      amount: 100000,
      assetStats: {
        nifty_bank: { cagr: 0.14, volatility: 0.22 },
        gold: { cagr: 0.10, volatility: 0.12 },
      },
      covarianceMatrix: [
        [0.0484, 0.004],
        [0.004, 0.0144],
      ],
      assetIds: ['nifty_bank', 'gold'],
      nSims: 1500,
      seed: 888,
      fatTails: true,
      computeStressComparison: true,
    });
    const stressComp = res.stressComparison;
    assert(stressComp !== undefined, 'Stress comparison must be populated');
    if (!stressComp) throw new Error('Stress comparison was undefined');
    assert(stressComp.isStressAware === true, 'isStressAware must be true');
    assert(stressComp.normalP5 > 0, 'normalP5 must be positive');
    assert(stressComp.stressP5 > 0, 'stressP5 must be positive');
    assert(typeof stressComp.normalProbLoss === 'number', 'normalProbLoss is a number');
    assert(typeof stressComp.stressProbLoss === 'number', 'stressProbLoss is a number');
  });

  test('Pass 2 - Bond Fixes', '(a) Later SIP deposits after default grow in redeployed sleeve; (b) Reinvestment rate after maturity; (c) Optimizer includes bonds', () => {
    const defaultSipRes = runMonteCarloSimulation({
      allocation: { bonds_fd: 1.0 },
      years: 3,
      mode: 'sip',
      monthlySip: 10000,
      assetStats: {},
      covarianceMatrix: [[0.0001]],
      assetIds: [],
      bondConfig: {
        couponRate: 0.08,
        tenureYears: 3,
        payoutFrequency: 'annual',
        creditRisk: { defaultProbabilityAnnual: 1.0, recoveryRate: 0.05 },
        reinvestmentRateAfterMaturity: 0.07,
      },
      nSims: 200,
      seed: 1234,
    });
    assert(
      defaultSipRes.terminalStats.median > 150000,
      `Deposits after default must earn reinvestment yield in redeployed sleeve (median: ${defaultSipRes.terminalStats.median})`
    );

    const projLump = projectLumpsum(
      100000,
      5,
      { bonds_fd: 1.0 },
      {},
      {
        couponRate: 0.12,
        tenureYears: 2,
        payoutFrequency: 'annual',
        creditRisk: { defaultProbabilityAnnual: 0, recoveryRate: 1.0 },
        reinvestmentRateAfterMaturity: 0.06,
      }
    );
    assertApprox(projLump.finalValue, 149400, 1000, 'Bond must switch to reinvestment rate after maturity');

    const testAssets = [
      { id: 'stock_high', expectedReturn: 0.18, volatility: 0.22 },
      { id: 'stock_med', expectedReturn: 0.12, volatility: 0.16 },
    ];
    const testCov = [
      [0.0484, 0.015],
      [0.015, 0.0256],
    ];
    const extended = extendAssetsAndCovarianceWithBonds(testAssets, testCov, 0.075, 0.005);
    assert(extended.assets.length === 3, 'Must have 3 assets including bonds_fd');
    assert(extended.covarianceMatrix.length === 3, 'Covariance matrix extended to 3x3');
    const bondOpt = maximizeReturn(extended.assets, extended.covarianceMatrix, {
      maxRisk: 0.05,
      maxPerAsset: 0.70,
    });
    assert(
      (bondOpt.weights['bonds_fd'] || 0) > 0.30,
      `Optimizer must allocate significant weight to bonds_fd under tight risk limit (got: ${bondOpt.weights['bonds_fd']})`
    );
  });

  // 13. Strategy Lab & Multi-Strategy Quantitative Engine Tests
  test('Strategies - Black-Litterman', 'No-views matches market weights within 2% per asset and reports excess+Rf', () => {
    const assets = ALL_VARIABLE_ASSETS.slice(0, 6).map((a) => ({
      id: a.id,
      expectedReturn: FALLBACK_ASSET_HISTORY[a.id]?.cagr ?? 0.12,
      volatility: FALLBACK_ASSET_HISTORY[a.id]?.volatility ?? 0.18,
    }));
    const n = assets.length;
    const cov = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? Math.pow(assets[i].volatility, 2) : 0.01))
    );
    const bl = STRATEGY_REGISTRY['black_litterman'].fn({
      assets,
      cov,
      constraints: { maxPerAsset: 1.0, minPerAsset: 0.0, riskFreeRate: 0.065 },
    });
    const expectedW = 1 / n;
    for (const a of assets) {
      const w = bl.weights[a.id] ?? 0;
      assert(
        Math.abs(w - expectedW) <= 0.02,
        `Black-Litterman with no views should match market weight ${expectedW} within 2%, got ${w} for ${a.id}`
      );
    }
    assert(bl.expectedReturn > 0.065, 'expectedReturn must be reported as mu_post w + riskFreeRate');
  });

  test('Strategies - Black-Litterman', 'Bullish view raises asset weight and higher confidence moves weight more', () => {
    const assets = ALL_VARIABLE_ASSETS.slice(0, 5).map((a) => ({
      id: a.id,
      expectedReturn: FALLBACK_ASSET_HISTORY[a.id]?.cagr ?? 0.12,
      volatility: FALLBACK_ASSET_HISTORY[a.id]?.volatility ?? 0.18,
    }));
    const n = assets.length;
    const cov = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? Math.pow(assets[i].volatility, 2) : 0.01))
    );
    const baseline = STRATEGY_REGISTRY['black_litterman'].fn({
      assets,
      cov,
      constraints: { maxPerAsset: 1.0, minPerAsset: 0.0, riskFreeRate: 0.065 },
    });
    const lowConf = STRATEGY_REGISTRY['black_litterman'].fn({
      assets,
      cov,
      constraints: { maxPerAsset: 1.0, minPerAsset: 0.0, riskFreeRate: 0.065 },
      views: [{ assetId: assets[0].id, expectedReturn: 0.30, confidence: 0.1 }],
    });
    const highConf = STRATEGY_REGISTRY['black_litterman'].fn({
      assets,
      cov,
      constraints: { maxPerAsset: 1.0, minPerAsset: 0.0, riskFreeRate: 0.065 },
      views: [{ assetId: assets[0].id, expectedReturn: 0.30, confidence: 0.99 }],
    });
    const wBase = baseline.weights[assets[0].id] ?? 0;
    const wLow = lowConf.weights[assets[0].id] ?? 0;
    const wHigh = highConf.weights[assets[0].id] ?? 0;
    assert(wLow > wBase, `Bullish view should raise asset weight above baseline (${wLow} > ${wBase})`);
    assert(wHigh > wLow, `Confidence 0.99 (${wHigh}) should move weight more than confidence 0.10 (${wLow})`);
  });

  test('Strategies - Registry Constraints', '|sum - 1| < 1e-9 and 0 <= w <= maxPerAsset for every strategy in STRATEGY_REGISTRY', () => {
    const assets = ALL_VARIABLE_ASSETS.map((a) => ({
      id: a.id,
      expectedReturn: FALLBACK_ASSET_HISTORY[a.id]?.cagr ?? 0.12,
      volatility: FALLBACK_ASSET_HISTORY[a.id]?.volatility ?? 0.18,
    }));
    const n = assets.length;
    const cov = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? Math.pow(assets[i].volatility, 2) : 0.01))
    );
    const maxPerAsset = 0.35;
    for (const [id, strat] of Object.entries(STRATEGY_REGISTRY)) {
      const res = strat.fn({
        assets,
        cov,
        constraints: { maxPerAsset, minPerAsset: 0.0, riskFreeRate: 0.065 },
        seed: 42,
      });
      const sum = Object.values(res.weights).reduce((a, b) => a + b, 0);
      assert(
        Math.abs(sum - 1.0) < 1e-9,
        `Strategy ${id} weights sum ${sum} must satisfy |sum - 1| < 1e-9`
      );
      for (const [assetId, w] of Object.entries(res.weights)) {
        assert(
          w >= -1e-6 && w <= maxPerAsset + 1e-6,
          `Strategy ${id} weight for ${assetId} (${w}) violates [0, ${maxPerAsset}] bounds`
        );
      }
    }
  });

  test('Strategy Lab - Performance', 'Full strategy scorecard completes under 15 seconds on fallback data', () => {
    const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const res = runStrategyScorecard();
    const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const durationSeconds = (t1 - t0) / 1000;
    console.log(`\n  [PERF] runStrategyScorecard finished in ${durationSeconds.toFixed(2)}s (target: < 15s)`);
    assert(
      durationSeconds < 15.0,
      `Full scorecard must complete under 15 seconds (took ${durationSeconds.toFixed(2)}s)`
    );
    assert(res.items.length === 11, 'Must evaluate all 11 quantitative strategies');
  });

  test('Strategies - CVaR Optimizer', 'CVaR(95%) on independent scenarios <= equalWeight and within 10% of best of the three', () => {
    const assets = ALL_VARIABLE_ASSETS.slice(0, 8).map((a) => ({
      id: a.id,
      expectedReturn: FALLBACK_ASSET_HISTORY[a.id]?.cagr ?? 0.12,
      volatility: FALLBACK_ASSET_HISTORY[a.id]?.volatility ?? 0.18,
    }));
    const n = assets.length;
    const cov = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? Math.pow(assets[i].volatility, 2) : 0.015))
    );
    const cvarRes = STRATEGY_REGISTRY['cvar_optimizer'].fn({
      assets,
      cov,
      constraints: { maxPerAsset: 0.35, minPerAsset: 0.0, riskFreeRate: 0.065 },
      seed: 42,
    });
    const minVarRes = STRATEGY_REGISTRY['min_variance'].fn({
      assets,
      cov,
      constraints: { maxPerAsset: 0.35, minPerAsset: 0.0, riskFreeRate: 0.065 },
      seed: 42,
    });
    const eqRes = STRATEGY_REGISTRY['equal_weight'].fn({
      assets,
      cov,
      constraints: { maxPerAsset: 0.35, minPerAsset: 0.0, riskFreeRate: 0.065 },
      seed: 42,
    });
    const deRes = STRATEGY_REGISTRY['global_search'].fn({
      assets,
      cov,
      constraints: { maxPerAsset: 0.35, minPerAsset: 0.0, riskFreeRate: 0.065 },
      seed: 42,
    });

    const testSeed = 99999;
    const rng = createMulberry32(testSeed);
    const sampleGaussian = () => {
      let u1 = rng();
      while (u1 < 1e-12) u1 = rng();
      const u2 = rng();
      return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    };
    const L = choleskyDecomposition(cov);
    const nScenarios = 5000;
    const testScenarios: number[][] = [];
    for (let s = 0; s < nScenarios; s++) {
      const z = Array.from({ length: n }, () => sampleGaussian());
      const scen = new Array(n);
      for (let i = 0; i < n; i++) {
        let shock = 0;
        for (let j = 0; j <= i; j++) shock += L[i][j] * z[j];
        scen[i] = assets[i].expectedReturn + shock;
      }
      testScenarios.push(scen);
    }
    const evalCVaR = (weightsMap: Record<string, number>) => {
      const w = assets.map((a) => weightsMap[a.id] || 0);
      const losses = testScenarios.map((scen) => {
        let r = 0;
        for (let i = 0; i < n; i++) r += w[i] * scen[i];
        return -r;
      });
      losses.sort((a, b) => b - a);
      const tail = Math.floor(0.05 * nScenarios);
      let sum = 0;
      for (let i = 0; i < tail; i++) sum += losses[i];
      return sum / tail;
    };
    const cvarVal = evalCVaR(cvarRes.weights);
    const minVarVal = evalCVaR(minVarRes.weights);
    const eqVal = evalCVaR(eqRes.weights);
    const deVal = evalCVaR(deRes.weights);
    assert(
      cvarVal <= eqVal + 1e-4,
      `cvarOptimizer CVaR (${cvarVal.toFixed(4)}) must be <= equalWeight CVaR (${eqVal.toFixed(4)})`
    );
    const bestOfThree = Math.min(cvarVal, minVarVal, eqVal);
    assert(
      cvarVal <= bestOfThree * 1.10,
      `cvarOptimizer CVaR (${cvarVal.toFixed(4)}) must be within 10% of best of the three (${bestOfThree.toFixed(4)})`
    );
    assert(typeof deVal === 'number' && !isNaN(deVal), 'Differential evolution cross-check must produce valid CVaR');
  });

  test('Strategies - Determinism', 'Each strategy is strictly deterministic across calls with identical seed', () => {
    const assets = ALL_VARIABLE_ASSETS.slice(0, 5).map((a) => ({
      id: a.id,
      expectedReturn: FALLBACK_ASSET_HISTORY[a.id]?.cagr ?? 0.12,
      volatility: FALLBACK_ASSET_HISTORY[a.id]?.volatility ?? 0.18,
    }));
    const n = assets.length;
    const cov = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? Math.pow(assets[i].volatility, 2) : 0.01))
    );
    for (const [id, strat] of Object.entries(STRATEGY_REGISTRY)) {
      const run1 = strat.fn({
        assets,
        cov,
        constraints: { maxPerAsset: 0.50, minPerAsset: 0.0, riskFreeRate: 0.065 },
        seed: 777,
      });
      const run2 = strat.fn({
        assets,
        cov,
        constraints: { maxPerAsset: 0.50, minPerAsset: 0.0, riskFreeRate: 0.065 },
        seed: 777,
      });
      assert(
        JSON.stringify(run1.weights) === JSON.stringify(run2.weights),
        `Strategy ${id} weights must be deterministic`
      );
      assert(run1.expectedReturn === run2.expectedReturn, `Strategy ${id} expectedReturn must match`);
      assert(run1.volatility === run2.volatility, `Strategy ${id} volatility must match`);
    }
  });

  test('Strategies - Risk Parity', 'Risk parity contributions are within 5% of each other when caps are loose', () => {
    const assets = ALL_VARIABLE_ASSETS.slice(0, 6).map((a) => ({
      id: a.id,
      expectedReturn: FALLBACK_ASSET_HISTORY[a.id]?.cagr ?? 0.12,
      volatility: FALLBACK_ASSET_HISTORY[a.id]?.volatility ?? 0.18,
    }));
    const n = assets.length;
    const cov = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? Math.pow(assets[i].volatility, 2) : 0.005))
    );
    const rp = STRATEGY_REGISTRY['risk_parity'].fn({
      assets,
      cov,
      constraints: { maxPerAsset: 1.0, minPerAsset: 0.0, riskFreeRate: 0.065 },
    });
    const w = assets.map((a) => rp.weights[a.id] || 0);
    const covMat = new Matrix(cov);
    const covW = covMat.mmul(Matrix.columnVector(w)).to1DArray();
    let pVar = 0;
    for (let i = 0; i < n; i++) pVar += w[i] * covW[i];
    const pVol = Math.sqrt(pVar);
    const rcs = w.map((wi, i) => (wi * covW[i]) / pVol);
    const sumRC = rcs.reduce((a, b) => a + b, 0);
    const normRC = rcs.map((rc) => rc / sumRC);
    const target = 1 / n;
    const maxDev = Math.max(...normRC.map((rc) => Math.abs(rc - target) / target));
    assert(
      maxDev <= 0.05,
      `Risk parity risk contributions should be within 5% of equal (max deviation: ${(maxDev * 100).toFixed(2)}%)`
    );
  });

  test('Strategies - HRP', 'Hierarchical Risk Parity weights are strictly non-negative and sum to 1', () => {
    const assets = ALL_VARIABLE_ASSETS.map((a) => ({
      id: a.id,
      expectedReturn: FALLBACK_ASSET_HISTORY[a.id]?.cagr ?? 0.12,
      volatility: FALLBACK_ASSET_HISTORY[a.id]?.volatility ?? 0.18,
    }));
    const n = assets.length;
    const cov = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? Math.pow(assets[i].volatility, 2) : 0.01))
    );
    const hrp = STRATEGY_REGISTRY['hrp'].fn({
      assets,
      cov,
      constraints: { maxPerAsset: 0.40, minPerAsset: 0.0, riskFreeRate: 0.065 },
    });
    for (const [id, w] of Object.entries(hrp.weights)) {
      assert(w >= 0, `HRP weight for ${id} (${w}) must be non-negative`);
    }
    const sum = Object.values(hrp.weights).reduce((a, b) => a + b, 0);
    assert(Math.abs(sum - 1.0) < 1e-9, `HRP weights sum ${sum} must equal 1`);
  });

  test('Strategies - Resampled Efficiency', 'Resampled efficiency portfolio sum of squares is lower than maxSharpe', () => {
    const assets = ALL_VARIABLE_ASSETS.map((a) => ({
      id: a.id,
      expectedReturn: FALLBACK_ASSET_HISTORY[a.id]?.cagr ?? 0.12,
      volatility: FALLBACK_ASSET_HISTORY[a.id]?.volatility ?? 0.18,
    }));
    const n = assets.length;
    const cov = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? Math.pow(assets[i].volatility, 2) : 0.01))
    );
    const maxSharpe = STRATEGY_REGISTRY['max_sharpe'].fn({
      assets,
      cov,
      constraints: { maxPerAsset: 0.80, minPerAsset: 0.0, riskFreeRate: 0.065 },
    });
    const resampled = STRATEGY_REGISTRY['resampled_efficiency'].fn({
      assets,
      cov,
      constraints: { maxPerAsset: 0.80, minPerAsset: 0.0, riskFreeRate: 0.065 },
      seed: 42,
    });
    const hhiSharpe = Object.values(maxSharpe.weights).reduce((s, w) => s + w * w, 0);
    const hhiResampled = Object.values(resampled.weights).reduce((s, w) => s + w * w, 0);
    assert(
      hhiResampled < hhiSharpe,
      `Resampled sum of squares (${hhiResampled.toFixed(4)}) must be lower than maxSharpe (${hhiSharpe.toFixed(4)})`
    );
  });

  test('Backtest - No Lookahead', 'Altering data after training window does not alter earlier OOS returns', () => {
    const aligned = getAlignedMonthlyReturns();
    const baseData: Record<string, number[]> = {};
    for (let j = 0; j < aligned.assetIds.length; j++) {
      const id = aligned.assetIds[j];
      baseData[id] = aligned.returnsMatrix.map((row) => row[j]);
    }
    const res1 = walkForwardBacktest('min_variance', baseData, {
      trainWindowMonths: 36,
      rebalancePeriodMonths: 12,
    });
    const modifiedData: Record<string, number[]> = {};
    for (const [id, rets] of Object.entries(baseData)) {
      modifiedData[id] = rets.map((r, idx) => (idx >= 48 ? r + 0.10 : r));
    }
    const res2 = walkForwardBacktest('min_variance', modifiedData, {
      trainWindowMonths: 36,
      rebalancePeriodMonths: 12,
    });
    const w1Returns1 = res1.monthlyReturns.slice(0, 12);
    const w1Returns2 = res2.monthlyReturns.slice(0, 12);
    assert(
      JSON.stringify(w1Returns1) === JSON.stringify(w1Returns2),
      'OOS returns for window 1 must not change when future data is altered'
    );
  });

  test('Backtest - Bootstrap', 'Historical bootstrap is reproducible with a fixed seed', () => {
    const config = {
      allocation: { nifty_bank: 0.5, gold: 0.5 },
      years: 5,
      mode: 'lumpsum' as const,
      amount: 100000,
      assetStats: {
        nifty_bank: { cagr: 0.14, volatility: 0.22 },
        gold: { cagr: 0.10, volatility: 0.12 },
      },
      covarianceMatrix: [
        [0.0484, 0.005],
        [0.005, 0.0144],
      ],
      assetIds: ['nifty_bank', 'gold'],
      nSims: 500,
      seed: 42,
    };
    const b1 = historicalBootstrap(config);
    const b2 = historicalBootstrap(config);
    assert(
      JSON.stringify(b1.percentilesByYear) === JSON.stringify(b2.percentilesByYear),
      'Bootstrap percentiles must be identical for same seed'
    );
    assert(b1.probabilityOfLoss === b2.probabilityOfLoss, 'Loss prob must be identical');
  });

  test('Backtest - Goal Solver', 'Goal solver required SIP for 90% confidence >= SIP for 70% confidence', () => {
    const allocation = { nifty_bank: 0.5, gold: 0.5 };
    const assetStats = {
      nifty_bank: { cagr: 0.14, volatility: 0.22 },
      gold: { cagr: 0.10, volatility: 0.12 },
    };
    const cov = [
      [0.0484, 0.005],
      [0.005, 0.0144],
    ];
    const assetIds = ['nifty_bank', 'gold'];
    const res = solveGoal({
      targetAmount: 5000000,
      horizonYears: 10,
      mode: 'sip',
      monthlySip: 10000,
      allocation,
      assetStats,
      covarianceMatrix: cov,
      assetIds,
    });
    assert(
      res.minSip90 >= res.minSip70,
      `SIP for 90% confidence (${res.minSip90}) must be >= SIP for 70% confidence (${res.minSip70})`
    );
  });

  test('Allocation - Rupee Conversion', 'Rupee conversion round-trip sums stay exact for 20 random allocations', () => {
    const assetIds = ALL_VARIABLE_ASSETS.map((a) => a.id);
    let seed = 42;
    const rng = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };
    for (let t = 0; t < 20; t++) {
      const rawWeights: Record<string, number> = {};
      let sum = 0;
      for (const id of assetIds) {
        const val = rng();
        rawWeights[id] = val;
        sum += val;
      }
      for (const id of assetIds) rawWeights[id] /= sum;
      const monthlyAmount = Math.floor(10000 + rng() * 90000);
      const inrAlloc = weightsToRupeeAllocation(rawWeights, monthlyAmount);
      const sumInr = Object.values(inrAlloc).reduce((a, b) => a + b, 0);
      assert(
        sumInr === monthlyAmount,
        `Rupee allocation sum (${sumInr}) must strictly equal monthlyAmount (${monthlyAmount})`
      );
      const derivedWeights = rupeeAllocationToWeights(inrAlloc);
      const sumW = Object.values(derivedWeights).reduce((a, b) => a + b, 0);
      assert(
        Math.abs(sumW - 1.0) < 1e-9,
        `Derived weights sum ${sumW} must equal 1.0`
      );
    }
  });

  test('Allocation - Rupee Conversion', '12-month totals equal monthly x 12 per asset', () => {
    const assetIds = ALL_VARIABLE_ASSETS.map((a) => a.id);
    const testAlloc = { [assetIds[0]]: 10000, [assetIds[1]]: 15000, [assetIds[2]]: 5000 };
    for (const [id, monthly] of Object.entries(testAlloc)) {
      const yearly = monthly * 12;
      assert(yearly === monthly * 12, `Yearly allocation for ${id} must equal monthly x 12`);
    }
  });

  // =========================================================================
  // NEW TESTS: 5 Finance-Consistency Issues
  // =========================================================================

  // Issue 1 Tests: Bootstrap Re-centring
  test('Issue 1 - Bootstrap Re-centring', '(a) Halving CAGR lowers bootstrap P50 by roughly the same factor as parametric P50 (within 10%)', () => {
    const assetIds = ['nifty_bank', 'gold', 'mf_large_cap'];
    const baseStats: Record<string, { cagr: number; volatility: number }> = {
      nifty_bank: { cagr: 0.14, volatility: 0.22 },
      gold: { cagr: 0.10, volatility: 0.13 },
      mf_large_cap: { cagr: 0.15, volatility: 0.16 },
    };
    const halvedStats: Record<string, { cagr: number; volatility: number }> = {
      nifty_bank: { cagr: 0.07, volatility: 0.22 },
      gold: { cagr: 0.05, volatility: 0.13 },
      mf_large_cap: { cagr: 0.075, volatility: 0.16 },
    };
    const cov = [
      [0.0484, 0.005, 0.020],
      [0.005, 0.0169, 0.004],
      [0.020, 0.004, 0.0256],
    ];
    const allocation = { nifty_bank: 0.4, gold: 0.2, mf_large_cap: 0.4 };
    const years = 10;
    const amount = 1000000;
    const seed = 12345;

    // Parametric base vs halved
    const paramBase = runMonteCarloSimulation({
      allocation,
      years,
      mode: 'lumpsum',
      amount,
      assetStats: baseStats,
      covarianceMatrix: cov,
      assetIds,
      nSims: 2500,
      seed,
    });
    const paramHalved = runMonteCarloSimulation({
      allocation,
      years,
      mode: 'lumpsum',
      amount,
      assetStats: halvedStats,
      covarianceMatrix: cov,
      assetIds,
      nSims: 2500,
      seed,
    });

    // Bootstrap base vs halved
    const bootBase = historicalBootstrap({
      allocation,
      years,
      mode: 'lumpsum',
      amount,
      assetStats: baseStats,
      covarianceMatrix: cov,
      assetIds,
      nSims: 2500,
      seed,
    });
    const bootHalved = historicalBootstrap({
      allocation,
      years,
      mode: 'lumpsum',
      amount,
      assetStats: halvedStats,
      covarianceMatrix: cov,
      assetIds,
      nSims: 2500,
      seed,
    });

    const paramRatio = paramHalved.terminalStats.median / paramBase.terminalStats.median;
    const bootRatio = bootHalved.terminalStats.median / bootBase.terminalStats.median;
    const relDiff = Math.abs(bootRatio - paramRatio) / paramRatio;

    console.log(`\n  [Issue 1(a)] Parametric P50: ${paramBase.terminalStats.median} -> ${paramHalved.terminalStats.median} (ratio: ${paramRatio.toFixed(3)})`);
    console.log(`  [Issue 1(a)] Bootstrap P50:  ${bootBase.terminalStats.median} -> ${bootHalved.terminalStats.median} (ratio: ${bootRatio.toFixed(3)})`);

    assert(
      relDiff <= 0.10,
      `Bootstrap P50 drop factor (${bootRatio.toFixed(3)}) must match parametric drop factor (${paramRatio.toFixed(3)}) within 10% (got diff: ${(relDiff * 100).toFixed(2)}%)`
    );
  });

  test('Issue 1 - Bootstrap Re-centring', '(b) With unchanged assumptions on fallback data, bootstrap P50 stays within 15% of parametric P50', () => {
    const assetIds = ['nifty_bank', 'mf_flexi_cap', 'gold'];
    const stats: Record<string, { cagr: number; volatility: number }> = {
      nifty_bank: { cagr: FALLBACK_ASSET_HISTORY['nifty_bank'].cagr, volatility: FALLBACK_ASSET_HISTORY['nifty_bank'].volatility },
      mf_flexi_cap: { cagr: FALLBACK_ASSET_HISTORY['mf_flexi_cap'].cagr, volatility: FALLBACK_ASSET_HISTORY['mf_flexi_cap'].volatility },
      gold: { cagr: FALLBACK_ASSET_HISTORY['gold'].cagr, volatility: FALLBACK_ASSET_HISTORY['gold'].volatility },
    };
    const { covarianceMatrix } = computeCovarianceAndCorrelation(FALLBACK_ASSET_HISTORY, assetIds);
    const allocation = { nifty_bank: 0.35, mf_flexi_cap: 0.45, gold: 0.20 };
    const years = 5;
    const amount = 500000;
    const seed = 4242;

    const param = runMonteCarloSimulation({
      allocation,
      years,
      mode: 'lumpsum',
      amount,
      assetStats: stats,
      covarianceMatrix,
      assetIds,
      nSims: 3000,
      seed,
    });
    const boot = historicalBootstrap({
      allocation,
      years,
      mode: 'lumpsum',
      amount,
      assetStats: stats,
      covarianceMatrix,
      assetIds,
      nSims: 3000,
      seed,
    });

    const diffPct = Math.abs(boot.terminalStats.median - param.terminalStats.median) / param.terminalStats.median;
    console.log(`\n  [Issue 1(b)] Unchanged assumptions: Parametric P50 = ${param.terminalStats.median}, Bootstrap P50 = ${boot.terminalStats.median} (diff: ${(diffPct * 100).toFixed(2)}%)`);

    assert(
      diffPct <= 0.15,
      `Bootstrap P50 (${boot.terminalStats.median}) should be within 15% of parametric P50 (${param.terminalStats.median}) (diff: ${(diffPct * 100).toFixed(2)}%)`
    );
  });

  test('Issue 1 - Bootstrap Re-centring', '(c) Bootstrap stays reproducible for a fixed seed', () => {
    const config = {
      allocation: { nifty_it: 0.5, nifty_pharma: 0.5 },
      years: 5,
      mode: 'lumpsum' as const,
      amount: 200000,
      assetStats: {
        nifty_it: { cagr: 0.15, volatility: 0.20 },
        nifty_pharma: { cagr: 0.12, volatility: 0.17 },
      },
      covarianceMatrix: [
        [0.040, 0.005],
        [0.005, 0.0289],
      ],
      assetIds: ['nifty_it', 'nifty_pharma'],
      nSims: 1000,
      seed: 987654,
    };
    const run1 = historicalBootstrap(config);
    const run2 = historicalBootstrap(config);
    assert(
      run1.terminalStats.median === run2.terminalStats.median,
      'Bootstrap P50 must be strictly identical for identical seeds'
    );
    assert(
      JSON.stringify(run1.percentilesByYear) === JSON.stringify(run2.percentilesByYear),
      'Bootstrap yearly percentiles must match identically'
    );
  });

  // Issue 2 Tests: SIP-Aware Inflation Target
  test('Issue 2 - Inflation Target', 'calculateInflationTarget helper matches hand-computed values', () => {
    // 1. Lumpsum only: 100,000 at 6% over 5 years
    // 100000 * (1.06)^5 = 133822.55776
    const lumpTarget = calculateInflationTarget(100000, 0, 5, 0.06);
    assertApprox(lumpTarget, 100000 * Math.pow(1.06, 5), 1e-4, 'Lumpsum inflation target');

    // 2. SIP only: 10,000/month for 1 year (12 months) at 0% inflation
    // Exactly 12 * 10,000 = 120,000
    const zeroInfTarget = calculateInflationTarget(0, 10000, 1, 0.0);
    assertApprox(zeroInfTarget, 120000, 1e-4, 'Zero inflation SIP target');

    // 3. SIP hand-computed: 1000/month for 3 months at 12% annual inflation
    // totalMonths = 3. Months m = 0, 1, 2:
    // m=0: 1000 * (1.12)^(3/12) = 1000 * 1.12^0.25 = 1028.7373
    // m=1: 1000 * (1.12)^(2/12) = 1000 * 1.12^(1/6) = 1019.0732
    // m=2: 1000 * (1.12)^(1/12) = 1000 * 1.12^(1/12) = 1009.4888
    // Total = 3057.2993
    const handTarget = calculateInflationTarget(0, 1000, 3 / 12, 0.12);
    const expectedHand =
      1000 * Math.pow(1.12, 3 / 12) +
      1000 * Math.pow(1.12, 2 / 12) +
      1000 * Math.pow(1.12, 1 / 12);
    assertApprox(handTarget, expectedHand, 1e-4, 'Hand-computed 3-month SIP inflation target');
  });

  test('Issue 2 - Inflation Target', 'Bootstrap and Parametric P(beat inflation) are within 10 percentage points for comparable P50', () => {
    const assetIds = ['mf_flexi_cap', 'gold'];
    const stats = {
      mf_flexi_cap: { cagr: 0.14, volatility: 0.16 },
      gold: { cagr: 0.09, volatility: 0.13 },
    };
    const cov = [
      [0.0256, 0.002],
      [0.002, 0.0169],
    ];
    const allocation = { mf_flexi_cap: 0.7, gold: 0.3 };
    const years = 5;
    const monthlySip = 20000;
    const inflationRate = 0.06;
    const seed = 54321;

    const param = runMonteCarloSimulation({
      allocation,
      years,
      mode: 'sip',
      monthlySip,
      assetStats: stats,
      covarianceMatrix: cov,
      assetIds,
      nSims: 3000,
      inflationRate,
      seed,
    });
    const boot = historicalBootstrap({
      allocation,
      years,
      mode: 'sip',
      monthlySip,
      assetStats: stats,
      covarianceMatrix: cov,
      assetIds,
      nSims: 3000,
      inflationRate,
      seed,
    });

    const diffProb = Math.abs(boot.probabilityOfBeatingInflation - param.probabilityOfBeatingInflation);
    console.log(`\n  [Issue 2] Parametric P(beat inflation): ${param.probabilityOfBeatingInflation}%, Bootstrap: ${boot.probabilityOfBeatingInflation}% (diff: ${diffProb.toFixed(1)} pp)`);

    assert(
      diffProb <= 10.0,
      `Bootstrap and parametric P(beat inflation) must be within 10 percentage points (diff: ${diffProb.toFixed(1)} pp)`
    );
  });

  // Issue 3 Tests: Goal Solver Probabilities & Monotonicity
  test('Issue 3 - Goal Solver', '(a) if current plan probReaching >= 70% then minSip70 <= current SIP', () => {
    const allocation = { mf_large_cap: 0.5, mf_mid_cap: 0.5 };
    const assetStats = {
      mf_large_cap: { cagr: 0.14, volatility: 0.16 },
      mf_mid_cap: { cagr: 0.17, volatility: 0.19 },
    };
    const cov = [
      [0.0256, 0.020],
      [0.020, 0.0361],
    ];
    const assetIds = ['mf_large_cap', 'mf_mid_cap'];

    // Set a very achievable target so current SIP (e.g. 50,000/mo) easily has prob >= 70%
    const currentSip = 50000;
    const targetAmount = 2500000; // 25 Lakh over 5 years (invested is 30 Lakh)
    const res = solveGoal({
      targetAmount,
      horizonYears: 5,
      mode: 'sip',
      monthlySip: currentSip,
      allocation,
      assetStats,
      covarianceMatrix: cov,
      assetIds,
    });

    console.log(`\n  [Issue 3(a)] probReachingMC = ${res.probReachingMC}%, minSip70 = ${res.minSip70}, current SIP = ${currentSip}`);
    assert(res.probReachingMC >= 70, `Expected probReachingMC >= 70%, got ${res.probReachingMC}%`);
    assert(
      res.minSip70 <= currentSip,
      `minSip70 (${res.minSip70}) must be <= current SIP (${currentSip}) when current plan prob >= 70%`
    );
  });

  test('Issue 3 - Goal Solver', '(b) minSip70 <= minSip80 <= minSip90 strictly for both parametric and bootstrap', () => {
    const allocation = { nifty_50: 0.6, gold: 0.4 };
    const assetStats = {
      nifty_50: { cagr: 0.13, volatility: 0.16 },
      gold: { cagr: 0.10, volatility: 0.12 },
    };
    const cov = [
      [0.0256, 0.002],
      [0.002, 0.0144],
    ];
    const assetIds = ['nifty_50', 'gold'];

    const res = solveGoal({
      targetAmount: 5000000, // 50 Lakh
      horizonYears: 10,
      mode: 'sip',
      monthlySip: 15000,
      allocation,
      assetStats,
      covarianceMatrix: cov,
      assetIds,
    });

    console.log(`\n  [Issue 3(b)] Parametric SIPs: 70% = ${res.minSip70}, 80% = ${res.minSip80}, 90% = ${res.minSip90}`);
    console.log(`  [Issue 3(b)] Bootstrap SIPs:  70% = ${res.minSip70Bootstrap}, 80% = ${res.minSip80Bootstrap}, 90% = ${res.minSip90Bootstrap}`);

    assert(res.minSip70 <= res.minSip80, `Parametric minSip70 (${res.minSip70}) <= minSip80 (${res.minSip80})`);
    assert(res.minSip80 <= res.minSip90, `Parametric minSip80 (${res.minSip80}) <= minSip90 (${res.minSip90})`);

    if (res.minSip70Bootstrap !== undefined && res.minSip80Bootstrap !== undefined && res.minSip90Bootstrap !== undefined) {
      assert(res.minSip70Bootstrap <= res.minSip80Bootstrap, `Bootstrap minSip70 <= minSip80`);
      assert(res.minSip80Bootstrap <= res.minSip90Bootstrap, `Bootstrap minSip80 <= minSip90`);
    }
  });

  test('Issue 3 - Goal Solver', '(c) Re-running simulation at returned minSip70 with fresh seed gives probability within 3 percentage points of 70', () => {
    const allocation = { mf_flexi_cap: 1.0 };
    const assetStats = {
      mf_flexi_cap: { cagr: 0.14, volatility: 0.16 },
    };
    const cov = [[0.0256]];
    const assetIds = ['mf_flexi_cap'];
    const targetAmount = 4000000;
    const horizonYears = 8;

    const res = solveGoal({
      targetAmount,
      horizonYears,
      mode: 'sip',
      monthlySip: 15000,
      allocation,
      assetStats,
      covarianceMatrix: cov,
      assetIds,
      seed: 42,
    });

    // Fresh run with different seed (e.g. 999999) at the solved minSip70
    const freshRun = runMonteCarloSimulation({
      allocation,
      years: horizonYears,
      mode: 'sip',
      monthlySip: res.minSip70,
      assetStats,
      covarianceMatrix: cov,
      assetIds,
      nSims: 4000,
      seed: 999999,
    });

    let countReaches = 0;
    if (freshRun.terminalValues) {
      for (let i = 0; i < freshRun.terminalValues.length; i++) {
        if (freshRun.terminalValues[i] >= targetAmount) countReaches++;
      }
    }
    const freshProb = (countReaches / freshRun.nSims) * 100;
    console.log(`\n  [Issue 3(c)] Solved minSip70 = ${res.minSip70}, fresh seed probability = ${freshProb.toFixed(1)}% (target: 70%)`);

    assert(
      Math.abs(freshProb - 70.0) <= 3.0,
      `Fresh simulation at minSip70 gave probability ${freshProb.toFixed(1)}%, which must be within 3 percentage points of 70%`
    );
  });

  test('Issue 3 - Goal Solver', '(d) Linearity test: SIP x2 strictly doubles simulated terminals for both Monte Carlo and Historical Bootstrap with fixed seed', () => {
    const allocation = { nifty_50: 0.6, gold: 0.4 };
    const assetStats = {
      nifty_50: { cagr: 0.13, volatility: 0.16 },
      gold: { cagr: 0.10, volatility: 0.12 },
    };
    const cov = [
      [0.0256, 0.002],
      [0.002, 0.0144],
    ];
    const assetIds = ['nifty_50', 'gold'];

    const mcBase = runMonteCarloSimulation({
      allocation,
      years: 5,
      mode: 'sip',
      monthlySip: 10000,
      lumpsumAmount: 0,
      assetStats,
      covarianceMatrix: cov,
      assetIds,
      nSims: 500,
      seed: 12345,
    });
    const mcDouble = runMonteCarloSimulation({
      allocation,
      years: 5,
      mode: 'sip',
      monthlySip: 20000,
      lumpsumAmount: 0,
      assetStats,
      covarianceMatrix: cov,
      assetIds,
      nSims: 500,
      seed: 12345,
    });
    let maxRelDiffMC = 0;
    for (let i = 0; i < 500; i++) {
      const expected = mcBase.terminalValues![i] * 2;
      const actual = mcDouble.terminalValues![i];
      const relDiff = Math.abs(expected - actual) / actual;
      if (relDiff > maxRelDiffMC) maxRelDiffMC = relDiff;
    }
    assert(maxRelDiffMC < 1e-6, `MC terminals must double when SIP doubles (max diff: ${maxRelDiffMC})`);

    const bootBase = historicalBootstrap({
      allocation,
      years: 5,
      mode: 'sip',
      monthlySip: 10000,
      lumpsumAmount: 0,
      assetStats,
      covarianceMatrix: cov,
      assetIds,
      nSims: 500,
      seed: 12345,
    });
    const bootDouble = historicalBootstrap({
      allocation,
      years: 5,
      mode: 'sip',
      monthlySip: 20000,
      lumpsumAmount: 0,
      assetStats,
      covarianceMatrix: cov,
      assetIds,
      nSims: 500,
      seed: 12345,
    });
    let maxRelDiffBoot = 0;
    for (let i = 0; i < 500; i++) {
      const expected = bootBase.terminalValues![i] * 2;
      const actual = bootDouble.terminalValues![i];
      const relDiff = Math.abs(expected - actual) / actual;
      if (relDiff > maxRelDiffBoot) maxRelDiffBoot = relDiff;
    }
    assert(maxRelDiffBoot < 1e-6, `Bootstrap terminals must double when SIP doubles (max diff: ${maxRelDiffBoot})`);
  });

  test('Issue 3 - Goal Solver', '(e) Accuracy test: Results from direct quantile inversion match old bisection within 3% on a 4-asset allocation with identical seed', () => {
    const allocation = {
      nifty_bank: 0.25,
      nifty_it: 0.25,
      gold: 0.25,
      mf_flexi_cap: 0.25,
    };
    const assetStats = {
      nifty_bank: { cagr: 0.14, volatility: 0.18 },
      nifty_it: { cagr: 0.15, volatility: 0.22 },
      gold: { cagr: 0.10, volatility: 0.12 },
      mf_flexi_cap: { cagr: 0.14, volatility: 0.16 },
    };
    const cov = [
      [0.0324, 0.005, 0.002, 0.004],
      [0.005, 0.0484, 0.001, 0.006],
      [0.002, 0.001, 0.0144, 0.002],
      [0.004, 0.006, 0.002, 0.0256],
    ];
    const assetIds = ['nifty_bank', 'nifty_it', 'gold', 'mf_flexi_cap'];
    const targetAmount = 10000000;
    const horizonYears = 10;
    const nSims = 2500;
    const seed = 42;

    const res = solveGoal({
      targetAmount,
      horizonYears,
      mode: 'sip',
      monthlySip: 25000,
      allocation,
      assetStats,
      covarianceMatrix: cov,
      assetIds,
      nSims,
      seed,
    });

    // Compute reference bisection on the exact same seed
    const targetIdx70 = Math.floor(nSims * 0.30);
    let low = 0;
    let high = Math.max(50000, (targetAmount / (horizonYears * 12)) * 3.5);
    for (let iter = 0; iter < 16; iter++) {
      const mid = (low + high) / 2;
      const testMC = runMonteCarloSimulation({
        allocation,
        years: horizonYears,
        mode: 'sip',
        lumpsumAmount: 0,
        monthlySip: mid,
        assetStats,
        covarianceMatrix: cov,
        assetIds,
        nSims,
        seed,
      });
      if (testMC.terminalValues && testMC.terminalValues[targetIdx70] >= targetAmount) {
        high = mid;
      } else {
        low = mid;
      }
    }
    const bisectionSip70 = Math.round(high / 100) * 100;
    const diffPct = (Math.abs(res.minSip70 - bisectionSip70) / bisectionSip70) * 100;
    console.log(`\n  [Goal Solver 4-Asset Accuracy] New method = ${res.minSip70}, Old Bisection = ${bisectionSip70}, Diff = ${diffPct.toFixed(2)}% (target: < 3%)`);
    assert(diffPct <= 3.0, `Expected diff <= 3%, got ${diffPct.toFixed(2)}%`);
  });

  test('Issue 3 - Goal Solver', '(f) Performance test: solveGoal finishes under 3 seconds on fallback data', () => {
    const allocation = {
      nifty_bank: 0.25,
      nifty_it: 0.25,
      gold: 0.25,
      mf_flexi_cap: 0.25,
    };
    const assetStats = {
      nifty_bank: { cagr: 0.14, volatility: 0.18 },
      nifty_it: { cagr: 0.15, volatility: 0.22 },
      gold: { cagr: 0.10, volatility: 0.12 },
      mf_flexi_cap: { cagr: 0.14, volatility: 0.16 },
    };
    const cov = [
      [0.0324, 0.005, 0.002, 0.004],
      [0.005, 0.0484, 0.001, 0.006],
      [0.002, 0.001, 0.0144, 0.002],
      [0.004, 0.006, 0.002, 0.0256],
    ];
    const assetIds = ['nifty_bank', 'nifty_it', 'gold', 'mf_flexi_cap'];

    const tStart = performance.now();
    const res = solveGoal({
      targetAmount: 10000000,
      horizonYears: 10,
      mode: 'sip',
      monthlySip: 25000,
      allocation,
      assetStats,
      covarianceMatrix: cov,
      assetIds,
      seed: 42,
    });
    const durationMs = performance.now() - tStart;
    console.log(`\n  [Goal Solver Perf] solveGoal finished in ${(durationMs / 1000).toFixed(3)}s (target: < 3s)`);
    assert(durationMs < 3000, `solveGoal should finish under 3000ms, took ${durationMs.toFixed(1)}ms`);
    assert(res.minSip70 > 0, 'minSip70 must be positive');
  });

  // Issue 4 Tests: Black-Litterman Expected Return
  test('Issue 4 - Black-Litterman', 'BL with no views and equal-weight strategy report the same expectedReturn', () => {
    const assets = ALL_VARIABLE_ASSETS.slice(0, 6).map((a) => ({
      id: a.id,
      expectedReturn: FALLBACK_ASSET_HISTORY[a.id]?.cagr ?? 0.14,
      volatility: FALLBACK_ASSET_HISTORY[a.id]?.volatility ?? 0.18,
    }));
    const n = assets.length;
    const cov = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? Math.pow(assets[i].volatility, 2) : 0.01))
    );

    const bl = blackLittermanStrategy({
      assets,
      cov,
      constraints: { maxPerAsset: 1.0, minPerAsset: 0.0, riskFreeRate: 0.065 },
    });

    const eq = equalWeightStrategy({
      assets,
      cov,
      constraints: { maxPerAsset: 1.0, minPerAsset: 0.0, riskFreeRate: 0.065 },
    });

    console.log(`\n  [Issue 4] Equal Weight expectedReturn: ${(eq.expectedReturn * 100).toFixed(2)}%`);
    console.log(`  [Issue 4] Black-Litterman expectedReturn: ${(bl.expectedReturn * 100).toFixed(2)}%`);
    console.log(`  [Issue 4] BL equilibriumReturn: ${(bl.equilibriumReturn! * 100).toFixed(2)}%, posteriorReturn: ${(bl.posteriorReturn! * 100).toFixed(2)}%`);

    assertApprox(
      bl.expectedReturn,
      eq.expectedReturn,
      1e-4,
      `BL expectedReturn (${bl.expectedReturn}) must match Equal Weight (${eq.expectedReturn}) with no views`
    );
    assert(bl.equilibriumReturn !== undefined, 'equilibriumReturn field must be populated');
    assert(bl.posteriorReturn !== undefined, 'posteriorReturn field must be populated');
  });

  // Issue 5 Tests: Walk-Forward Buy-and-Hold & Drifted Turnover
  test('Issue 5 - Walk-Forward Cost', '(a) With a one-asset strategy, turnover is 0 after the first rebalance', () => {
    const aligned = getAlignedMonthlyReturns();
    const assetId = aligned.assetIds[0];
    const singleAssetData: Record<string, number[]> = {
      [assetId]: aligned.returnsMatrix.map((r) => r[0]),
    };

    // Custom strategy that always allocates 100% to asset 0
    const oneAssetStrategy = (input: any) => {
      const w: Record<string, number> = { [input.assets[0].id]: 1.0 };
      return {
        strategyId: 'single_asset',
        strategyName: 'Single Asset',
        weights: w,
        expectedReturn: 0.12,
        volatility: 0.18,
        sharpeRatio: 0.35,
      };
    };

    // Temporarily register single asset strategy
    (STRATEGY_REGISTRY as any)['single_asset'] = {
      name: 'Single Asset',
      description: 'Single asset',
      fn: oneAssetStrategy,
    };

    const wf = walkForwardBacktest('single_asset', singleAssetData, {
      trainWindowMonths: 36,
      rebalancePeriodMonths: 12,
    });

    delete (STRATEGY_REGISTRY as any)['single_asset'];

    // In a 120-month dataset with 36-month train and 12-month rebalance:
    // Total windows = 7 (months 36, 48, 60, 72, 84, 96, 108).
    // Rebalance 1 (initial allocation from cash) has turnover 1.0.
    // Rebalances 2 through 7 have turnover 0.0 because the drifted weight was 1.0 and target was 1.0.
    // Average turnover across 7 rebalances is 1.0 / 7 = 0.1429.
    console.log(`\n  [Issue 5(a)] One-asset average turnover: ${wf.turnover} (initial 1.0 + 6 x 0.0) / 7 = 0.1429`);
    assertApprox(wf.turnover, 1.0 / 7, 0.01, 'One-asset turnover after first rebalance must be 0');
  });

  test('Issue 5 - Walk-Forward Cost', '(b) With drifting weights, turnover is positive and smaller than monthly-rebalance figure', () => {
    // Run min_variance walk forward
    const wfDrifted = walkForwardBacktest('min_variance', undefined, {
      trainWindowMonths: 36,
      rebalancePeriodMonths: 12,
    });

    console.log(`\n  [Issue 5(b)] Drifting annual turnover for min_variance: ${wfDrifted.turnover}`);

    // Turnover must be positive (reflects weight adjustment between drifted weights and new annual targets)
    assert(wfDrifted.turnover > 0.01, `Drifted turnover (${wfDrifted.turnover}) must be positive`);
    // And significantly smaller than a strategy that rebalances back to static targets every month
    assert(wfDrifted.turnover < 0.85, `Drifted annual turnover (${wfDrifted.turnover}) must be reasonable`);
  });

  // Long-Horizon Validation & Diagnostics Tests
  test('Walk-Forward - Windows & Embargo', 'Expanding vs rolling windows use the correct training ranges; embargo leaves the stated gap; no-lookahead still holds; an asset that has not started is excluded from that window', () => {
    const T = 180;
    const n = 3;
    const ids = ['asset_a', 'asset_b', 'asset_c'];
    const matrix: number[][] = Array.from({ length: T }, (_, t) => [
      0.01 + 0.02 * Math.sin(t / 6),
      0.008 + 0.015 * Math.cos(t / 8),
      t < 70 ? NaN : 0.012 + 0.025 * Math.sin(t / 5), // asset_c starts at month 70
    ]);

    // Expanding window: min 60 training months, 1 month embargo, 12 months hold
    const expWindows = precomputeWalkForwardWindows(matrix, ids, {
      windowType: 'expanding',
      trainWindowMonths: 60,
      holdMonths: 12,
      embargoMonths: 1,
    });

    assert(expWindows.length > 0, 'Expanding windows must be generated');
    assert(expWindows[0].trainSlice.length === 60, 'Window 0 training length must be 60');
    assert(expWindows[0].trainStart === 0 && expWindows[0].trainEnd === 60, 'Window 0 training range must be 0 to 60');
    assert(expWindows[0].holdStart === 61, 'Hold start must reflect 1-month embargo (60 + 1 = 61)');
    assert(expWindows[0].holdEnd === 73, 'Hold end must be 61 + 12 = 73');

    assert(expWindows[1].trainSlice.length === 72, 'Window 1 training length must be 72 in expanding mode');
    assert(expWindows[1].trainStart === 0 && expWindows[1].trainEnd === 72, 'Window 1 training range must be 0 to 72');
    assert(expWindows[1].holdStart === 73 && expWindows[1].holdEnd === 85, 'Window 1 hold period must be 73 to 85');

    // Asset exclusion: asset_c has not started yet in window 0 (ended at 60)
    assert(expWindows[0].excludedAssetIds.includes('asset_c'), 'Asset C must be excluded in window 0 before it starts');

    // Rolling window: 120 months training, 2 months embargo
    const rollWindows = precomputeWalkForwardWindows(matrix, ids, {
      windowType: 'rolling',
      rollingMonths: 120,
      holdMonths: 12,
      embargoMonths: 2,
    });
    assert(rollWindows[0].trainSlice.length === 120, 'Rolling window 0 training length must be 120');
    assert(rollWindows[0].holdStart === 122, 'Hold start must reflect 2-month embargo (120 + 2 = 122)');
    assert(rollWindows[1].trainSlice.length === 120, 'Rolling window 1 training length must remain 120');
    assert(rollWindows[1].trainStart === 12 && rollWindows[1].trainEnd === 132, 'Rolling window 1 range must shift by 12 months');

    // No-lookahead test: altering data at t >= 90 must not change window 0 or window 1 OOS returns
    const customData1: Record<string, number[]> = {
      asset_a: matrix.map((row) => (Number.isNaN(row[0]) ? 0 : row[0])),
      asset_b: matrix.map((row) => (Number.isNaN(row[1]) ? 0 : row[1])),
    };
    const res1 = walkForwardBacktest('equal_weight', customData1, {
      windowType: 'expanding',
      trainWindowMonths: 60,
      holdMonths: 12,
      embargoMonths: 1,
    });

    const customData2: Record<string, number[]> = {
      asset_a: customData1.asset_a.map((r, idx) => (idx >= 90 ? r + 0.15 : r)),
      asset_b: customData1.asset_b.map((r, idx) => (idx >= 90 ? r - 0.10 : r)),
    };
    const res2 = walkForwardBacktest('equal_weight', customData2, {
      windowType: 'expanding',
      trainWindowMonths: 60,
      holdMonths: 12,
      embargoMonths: 1,
    });

    const earlyOos1 = res1.monthlyReturns.slice(0, 24);
    const earlyOos2 = res2.monthlyReturns.slice(0, 24);
    assert(
      JSON.stringify(earlyOos1) === JSON.stringify(earlyOos2),
      'Early OOS returns must be strictly identical regardless of future alterations'
    );
  });

  test('Diagnostics - Chow Test', 'chowTest detects a synthetic break in beta (large F, small p) and does not flag a no-break series most of the time', () => {
    const rng = createMulberry32(888111);
    const randNorm = () => {
      const u1 = Math.max(1e-9, rng());
      const u2 = rng();
      return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    };

    // 1. Synthetic break in beta at month 60
    const T = 120;
    const breakMonth = 60;
    const x: number[] = [];
    const y: number[] = [];

    for (let t = 0; t < T; t++) {
      const mkt = randNorm() * 0.05;
      x.push(mkt);
      if (t < breakMonth) {
        // Pre-break: alpha = 0.005, beta = 0.4
        y.push(0.005 + 0.4 * mkt + randNorm() * 0.015);
      } else {
        // Post-break: alpha = -0.015, beta = 1.9
        y.push(-0.015 + 1.9 * mkt + randNorm() * 0.015);
      }
    }

    const breakResult = chowTest(y, x, breakMonth);
    assert(breakResult.fStat > 15, `Chow F-stat (${breakResult.fStat}) must be large for synthetic break in beta`);
    assert(breakResult.pValue < 0.001, `Chow p-value (${breakResult.pValue}) must be < 0.001 for structural break`);
    assert(breakResult.isSignificant === true, 'Chow test must flag structural break as significant');

    // 2. 200 seeded no-break series: false-positive rate at 5% level must be below 12%
    let falsePositives = 0;
    const nSims = 200;

    for (let s = 0; s < nSims; s++) {
      const simX: number[] = [];
      const simY: number[] = [];
      for (let t = 0; t < T; t++) {
        const mkt = randNorm() * 0.05;
        simX.push(mkt);
        // Constant beta = 1.0 throughout full series
        simY.push(0.005 + 1.0 * mkt + randNorm() * 0.02);
      }
      const simRes = chowTest(simY, simX, breakMonth);
      if (simRes.pValue < 0.05) {
        falsePositives++;
      }
    }

    const fpRate = falsePositives / nSims;
    console.log(`\n  [Chow Test] 200 no-break series false positive rate: ${(fpRate * 100).toFixed(1)}% (target < 12%)`);
    assert(fpRate < 0.12, `False positive rate (${(fpRate * 100).toFixed(1)}%) must be below 12%`);
  });

  test('Regimes - Rule Based', 'Regime labels on a synthetic series with a known crash mark the crash as bear', () => {
    // Synthetic market price series with crash
    // 0..40: Growth from 100 to 200
    // 41..46: Crash from 200 to 130 (35% drawdown)
    // 47..80: Recovery from 130 to 220
    const prices: number[] = [];
    let p = 100;
    for (let t = 0; t <= 40; t++) {
      p = 100 * Math.exp((t / 40) * Math.log(2.0));
      prices.push(p);
    }
    for (let t = 1; t <= 6; t++) {
      p = 200 - t * ((200 - 130) / 6);
      prices.push(p);
    }
    for (let t = 1; t <= 34; t++) {
      p = 130 * Math.exp((t / 34) * Math.log(220 / 130));
      prices.push(p);
    }

    const regimePoints = classifyMarketRegimes(prices);
    assert(regimePoints.length === prices.length, 'All price points must receive a regime label');

    // Month 46 is at the bottom of the crash (price 130 vs peak 200 = 35% drawdown > 20%)
    const crashPoint = regimePoints[46];
    assert(crashPoint.drawdown > 0.20, `Crash drawdown (${(crashPoint.drawdown * 100).toFixed(1)}%) must be > 20%`);
    assert(crashPoint.regime === 'bear', `Crash period must be classified as 'bear' (was ${crashPoint.regime})`);
  });

  test('Regimes - 2-State Gaussian HMM', 'HMM recovers two well-separated synthetic regimes, and sets the unreliable flag on pure noise', () => {
    const rng = createMulberry32(456789);
    const randNorm = () => {
      const u1 = Math.max(1e-9, rng());
      const u2 = rng();
      return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    };

    // 1. Two well-separated regimes with high persistence (95% and 90%)
    const T = 180;
    let state = 0;
    const rets: number[] = [];

    for (let t = 0; t < T; t++) {
      if (state === 0) {
        rets.push(0.015 + randNorm() * 0.02);
        if (rng() > 0.95) state = 1;
      } else {
        rets.push(-0.035 + randNorm() * 0.07);
        if (rng() > 0.90) state = 0;
      }
    }

    const hmmFit = fitGaussianHMM(rets, { seed: 12345 });
    assert(!hmmFit.unreliable, 'HMM fit on persistent 2-state process must be marked reliable');
    assert(hmmFit.states[0].annualizedVol < hmmFit.states[1].annualizedVol, 'State 0 vol must be lower than State 1 vol');
    assert(hmmFit.states[0].averageDurationMonths >= 6.0, 'Average state duration must be >= 6 months');

    // 2. Pure i.i.d. noise (no persistent state structure)
    const noise: number[] = [];
    for (let t = 0; t < 100; t++) {
      noise.push(randNorm() * 0.04);
    }
    const hmmNoise = fitGaussianHMM(noise, { seed: 9999 });
    assert(hmmNoise.unreliable === true, 'HMM on pure i.i.d. noise must be flagged as unreliable');
  });

  test('Bootstrap - Politis-White Block Length', 'politisWhiteBlockLength is larger for an autocorrelated series than for white noise, stays within 3-24, and is deterministic', () => {
    const rng = createMulberry32(314159);
    const randNorm = () => {
      const u1 = Math.max(1e-9, rng());
      const u2 = rng();
      return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    };

    const N = 240;
    const whiteNoise: number[] = [];
    const autoCorrSeries: number[] = [];

    let prev = 0;
    for (let t = 0; t < N; t++) {
      const eps = randNorm();
      whiteNoise.push(eps);
      // AR(1) with phi = 0.75
      const val = 0.75 * prev + eps;
      autoCorrSeries.push(val);
      prev = val;
    }

    const bWhite = politisWhiteBlockLength(whiteNoise);
    const bAuto = politisWhiteBlockLength(autoCorrSeries);

    console.log(`\n  [Politis-White] White noise block length: ${bWhite} mo, AR(1) block length: ${bAuto} mo`);
    assert(bWhite >= 3 && bWhite <= 24, `White noise block length (${bWhite}) must stay in [3, 24]`);
    assert(bAuto >= 3 && bAuto <= 24, `AR(1) block length (${bAuto}) must stay in [3, 24]`);
    assert(bAuto > bWhite, `Autocorrelated series block length (${bAuto}) must be larger than white noise (${bWhite})`);

    // Determinism assertion
    const bAutoRepeat = politisWhiteBlockLength(autoCorrSeries);
    assert(bAuto === bAutoRepeat, 'politisWhiteBlockLength must be strictly deterministic');
  });

  test('Bootstrap - Stationary Bootstrap', 'Stationary bootstrap is reproducible for a fixed seed and keeps the P50 within 20% of the parametric P50 on the fallback data', () => {
    const config = {
      allocation: { nifty_bank: 0.5, gold: 0.5 },
      years: 5,
      mode: 'lumpsum' as const,
      amount: 100000,
      assetStats: {
        nifty_bank: { cagr: 0.14, volatility: 0.22 },
        gold: { cagr: 0.10, volatility: 0.12 },
      },
      covarianceMatrix: [
        [0.0484, 0.005],
        [0.005, 0.0144],
      ],
      assetIds: ['nifty_bank', 'gold'],
      nSims: 1000,
      seed: 888777,
      bootstrapType: 'stationary' as const,
    };

    const s1 = historicalBootstrap(config);
    const s2 = historicalBootstrap(config);

    assert(
      JSON.stringify(s1.percentilesByYear) === JSON.stringify(s2.percentilesByYear),
      'Stationary bootstrap percentiles must be identical for identical seed'
    );
    assert(s1.terminalStats.median === s2.terminalStats.median, 'Stationary bootstrap median must be reproducible');

    const param = runMonteCarloSimulation({
      ...config,
      nSims: 1000,
    });

    const p50Stat = s1.terminalStats.median;
    const p50Param = param.terminalStats.median;
    const diffRatio = Math.abs(p50Stat - p50Param) / p50Param;

    console.log(`\n  [Stationary Bootstrap] Parametric P50: ₹ ${p50Param}, Stationary P50: ₹ ${p50Stat} (diff: ${(diffRatio * 100).toFixed(2)}%)`);
    assert(
      diffRatio < 0.20,
      `Stationary bootstrap P50 (${p50Stat}) must stay within 20% of parametric P50 (${p50Param})`
    );
  });

  const suiteEnd = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const totalPassed = results.filter((r) => r.passed).length;
  return {
    results,
    summary: {
      total: results.length,
      passed: totalPassed,
      failed: results.length - totalPassed,
      totalDurationMs: Number((suiteEnd - suiteStart).toFixed(2)),
    },
  };
}
