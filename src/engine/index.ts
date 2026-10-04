/**
 * Portfolio Simulator Engine - Public API
 * Complete quantitative calculation and historical data engine for portfolio simulations.
 */

// Domain Types
export type {
  AssetCategory,
  AssetDefinition,
  MonthlyDataPoint,
  AssetStats,
  AssetHistory,
  HistoryApiResponse,
  PayoutFrequency,
  CreditRisk,
  BondInput,
  BondCashFlow,
  BondAnalysis,
  PortfolioAllocation,
  YearSnapshot,
  DeterministicProjectionResult,
  MonteCarloPercentilePoint,
  MonteCarloSummaryStats,
  MonteCarloResult,
  MonteCarloConfig,
  CashFlow,
  OptimizerPoint,
  OptimizerConstraints,
  OptimizableAsset,
  Strategy,
  StrategyInput,
  StrategyOutput,
  StrategyRegistry,
  BLView,
  WalkForwardOptions,
  WalkForwardResult,
  StrategyScorecardItem,
  StrategyScorecardOptions,
  StrategyScorecardResult,
  ConsensusResult,
  GoalSolverInput,
  GoalSolverResult,
} from './types.ts';

// Asset Universe
export {
  SECTOR_INDICES,
  COMMODITY_ASSETS,
  MUTUAL_FUND_CATEGORIES,
  DIRECT_STOCKS_ASSET,
  BONDS_FD_DEFINITION,
  ALL_VARIABLE_ASSETS,
  ALL_ASSET_DEFINITIONS,
  ASSET_MAP,
} from './assets.ts';

// Fallback Benchmark Data
export {
  FALLBACK_ASSET_HISTORY,
  CALIBRATION,
  generateFallbackSeries,
} from './fallbackData.ts';

// Historical Data & Matrix Calculations
export {
  computeAssetStats,
  computeCovarianceAndCorrelation,
  buildCovariance,
  shrinkCovariance,
  calculateLedoitWolfIntensity,
  DEFAULT_CATEGORY_PRIORS,
  getDefaultPriorForAsset,
  calculateAdjustedCagr,
  getAssetHistory,
  getFullUniverseHistory,
} from './dataService.ts';

// PRNG
export { createMulberry32 } from './prng.ts';

// Bonds & Fixed Deposits
export {
  getPeriodsPerYear,
  generateBondCashflows,
  analyzeBond,
  stepBondMonthly,
} from './bonds.ts';

// Deterministic Projections
export {
  projectLumpsum,
  projectSIP,
  normalizeAllocation,
  getAssetExpectedReturn,
  calculateInflationTarget,
} from './projections.ts';

// Monte Carlo (GBM with Cholesky Correlated Returns)
export {
  runMonteCarlo,
  runMonteCarloSimulation,
  choleskyDecomposition,
  isSimulationCancelled,
  cancelSimulation,
} from './monteCarlo.ts';

// High-Precision XIRR
export { xirr } from './xirr.ts';

// Markowitz Optimizer & Efficient Frontier
export {
  projectOntoBoundedSimplex,
  computePortfolioVariance,
  computePortfolioReturn,
  computeVarianceGradient,
  findMinimumVariancePortfolio,
  estimateMaxEigenvalue,
  maximizeReturn,
  maximizeSharpe,
  computeEfficientFrontier,
  extendAssetsAndCovarianceWithBonds,
} from './optimizer.ts';

export {
  runOptimizerAsync,
  terminateOptimizer,
} from './optimizerService.ts';

// Strategy Lab - Pure Allocation Strategies
export {
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
  compareSolvers,
  STRATEGY_REGISTRY,
} from './strategies.ts';

// Backtesting, Bootstrap, Scorecard & Goal Solver
export {
  getAlignedMonthlyReturns,
  walkForwardBacktest,
  precomputeWalkForwardWindows,
  historicalBootstrap,
  runStrategyScorecard,
  consensusPortfolio,
  solveGoal,
} from './backtest.ts';

export {
  runGoalSolverAsync,
  terminateGoalSolver,
} from './goalService.ts';

// Long-Horizon Estimation & Shrinkage Techniques
export {
  perAssetGeometricMean,
  emMeanCovariance,
  pairwiseCovariance,
  nearestPSD,
  jamesSteinShrinkMeans,
  ledoitWolfConstantCorrelation,
  proxyBackfill,
} from './estimation.ts';

// Stability Diagnostics & Chow Tests
export {
  rollingStats,
  chowTest,
  supSupFScan,
  parameterStability,
  fSurvival,
  incompleteBeta,
  KNOWN_HISTORICAL_BREAK_DATES,
  extractReturnsAndDates,
} from './diagnostics.ts';

// Market Regimes, HMM, and Automatic Block Length
export {
  classifyMarketRegimes,
  fitGaussianHMM,
  computeRegimeStats,
  politisWhiteBlockLength,
} from './regimes.ts';
