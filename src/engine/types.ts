/**
 * Portfolio Simulator Engine - Type Definitions
 */

export type AssetCategory = 'sector' | 'gold' | 'mf' | 'fixed_income' | 'stock';

export interface AssetDefinition {
  id: string;
  name: string;
  category: AssetCategory;
  tickerOrCode: string;
  description: string;
  benchmark?: string;
}

export interface MonthlyDataPoint {
  date: string; // YYYY-MM
  price: number;
  return?: number; // Monthly simple return (price_t / price_{t-1} - 1)
}

export interface AssetStats {
  id: string;
  name: string;
  category: AssetCategory;
  cagr: number; // Annualized mean return (CAGR) e.g. 0.14 for 14%
  volatility: number; // Annualized standard deviation of monthly returns e.g. 0.18 for 18%
  startPrice: number;
  endPrice: number;
  monthsCount: number;
  startDate: string;
  endDate: string;
  firstDate?: string;
  lastDate?: string;
  nMonths?: number;
  source?: string;
  proxyUsed?: boolean;
  proxyName?: string;
  pctBackfilled?: number;
  backfillR2?: number;
}

export interface AssetHistory extends AssetStats {
  series: MonthlyDataPoint[];
  isFallback?: boolean;
}

export interface UnbalancedPanel {
  dates: string[]; // Chronologically sorted YYYY-MM
  assetIds: string[];
  matrix: number[][]; // T rows (dates) x N columns (assets), NaN where missing
}

export type EstimationMethod = 'common' | 'per_asset' | 'em_shrinkage';

export interface HistoryApiResponse {
  assets: Record<string, AssetHistory>;
  assetIds: string[];
  covarianceMatrix: number[][]; // Annualized covariance matrix (N x N)
  correlationMatrix: number[][]; // Correlation matrix (N x N)
  usingFallbackData: boolean;
  timestamp: number;
  cached?: boolean;
  message?: string;
}

// Bonds / Fixed Deposit Types
export type PayoutFrequency = 'monthly' | 'quarterly' | 'semi-annual' | 'annual' | 'cumulative';

export interface CreditRisk {
  defaultProbabilityAnnual: number; // Annual probability of default e.g. 0.02 for 2%
  recoveryRate: number; // Recovery percentage e.g. 0.40 for 40% recovery
}

export interface BondInput {
  couponRate: number; // Annual coupon rate e.g. 0.075 for 7.5%
  tenureYears: number; // Tenure in years e.g. 5
  payoutFrequency: PayoutFrequency;
  creditRisk: CreditRisk;
  faceValue?: number; // Default 100 or 1000
  reinvestmentRateAfterMaturity?: number; // Default 0.065 (6.5%) applied after tenure ends
}

export interface BondCashFlow {
  period: number;
  date: string;
  couponPayment: number;
  principalPayment: number;
  totalCashFlow: number;
  remainingPrincipal: number;
}

export interface BondAnalysis {
  cashflows: BondCashFlow[];
  ytm: number; // Annualized Yield to Maturity (XIRR)
  totalCoupons: number;
  totalReceived: number;
  expectedAnnualLossRate: number; // defaultProbability * (1 - recoveryRate)
  netExpectedYield: number; // YTM - expectedAnnualLossRate
}

// Projections Types
export type PortfolioAllocation = Record<string, number>; // assetId -> weight (0 to 1, summing to 1)

export interface YearSnapshot {
  year: number;
  investedCapital: number;
  portfolioValue: number;
  netGains: number;
  assetValues: Record<string, number>;
}

export interface DeterministicProjectionResult {
  mode: 'lumpsum' | 'sip';
  initialInvestment?: number;
  monthlySip?: number;
  years: number;
  totalInvested: number;
  finalValue: number;
  netGains: number;
  cagrOrXirr: number; // CAGR for lumpsum, XIRR for SIP
  yearByYear: YearSnapshot[];
  allocation: PortfolioAllocation;
}

// Monte Carlo Types
export interface MonteCarloPercentilePoint {
  year: number;
  p5: number;
  p10: number;
  p25: number;
  p50: number; // Median
  p75: number;
  p90: number;
  p95: number;
}

export interface MonteCarloSummaryStats {
  min: number;
  p25: number;
  median: number;
  p75: number;
  mean: number;
  max: number;
  stdDev: number;
}

export interface MonteCarloStressComparison {
  normalP5: number;
  normalP10?: number;
  normalP50?: number;
  normalProbLoss: number;
  stressP5: number;
  stressP10?: number;
  stressP50?: number;
  stressProbLoss: number;
  isStressAware: boolean;
  isBearRegimeStress?: boolean;
}

export interface MonteCarloResult {
  percentilesByYear: MonteCarloPercentilePoint[];
  probabilityOfLoss: number; // % of runs where terminal value < total invested capital
  probabilityOfBeatingInflation: number; // % of runs where terminal real value > inflation threshold
  inflationRate: number;
  totalInvested: number;
  terminalStats: MonteCarloSummaryStats;
  nSims: number;
  simulationTimeMs: number;
  defaultsTriggeredCount?: number;
  stressComparison?: MonteCarloStressComparison;
  terminalValues?: Float64Array; // Array of all simulated terminal values
  probabilityAtLeast?: (target: number) => number; // Fraction of simulated terminal values >= target
}

export interface MonteCarloConfig {
  allocation: PortfolioAllocation;
  years: number;
  mode?: 'lumpsum' | 'sip' | 'both';
  amount?: number; // initial lumpsum or monthly SIP amount (backward compatibility)
  lumpsumAmount?: number; // Initial lumpsum investment (if any)
  monthlySip?: number; // Regular monthly SIP contribution (if any)
  seed?: number; // Optional seed for deterministic, reproducible simulation
  assetStats: Record<string, { cagr: number; volatility: number }>;
  covarianceMatrix: number[][];
  assetIds: string[];
  bondConfig?: BondInput;
  nSims?: number; // Default 5000
  inflationRate?: number; // Default 0.06 (6%)
  fatTails?: boolean; // When true, uses Student-t(5) unit-variance scaled shocks (Stress-aware returns)
  computeStressComparison?: boolean; // When true, evaluates both Normal and Student-t(5) stats
  model?: 'parametric' | 'student_t' | 'bootstrap'; // Simulation shock generator model
  bootstrapType?: 'circular' | 'stationary'; // Stationary (Politis-Romano) or Circular block bootstrap
  blockLength?: number; // Chosen bootstrap block length (Politis-White)
  regimeFilter?: ('bear' | 'high_vol' | 'normal')[]; // When set, resamples exclusively from selected market regimes
  stressMode?: boolean; // When true, replaces covariance with bear-regime covariance and applies haircut
  bearCovarianceMatrix?: number[][]; // Stress covariance matrix from bear periods
  bearReturnHaircut?: number; // Stress return haircut (e.g. 0.03 for 3% reduction in cagr)
  monthlyReturnsByAsset?: Record<string, number[]>; // Historical returns for bootstrap
  rescaleVol?: boolean; // When true (default), rescales bootstrap shocks to match assetStats volatility
}

// Cashflow for XIRR
export interface CashFlow {
  date: Date | string;
  amount: number; // Negative for outflows (investments), positive for inflows
}

// Optimizer Types
export interface OptimizerPoint {
  weights: Record<string, number>;
  expectedReturn: number; // Annualized portfolio return
  volatility: number; // Annualized portfolio risk (standard deviation)
  sharpeRatio: number;
  infeasible?: boolean; // True if even minimum-variance portfolio violates maxRisk
  riskLimitBinding?: boolean; // True if optimal portfolio is constrained by maxRisk
}

export interface OptimizerConstraints {
  maxRisk?: number; // Annualized max volatility constraint
  maxCVaR?: number; // Annualized max CVaR(95) ceiling for mean-CVaR strategy
  maxPerAsset?: number; // Upper bound per asset weight (e.g., 0.30 for 30%)
  minPerAsset?: number; // Lower bound per asset weight (default 0)
  riskFreeRate?: number; // Default 0.065 (6.5% RBI repo / risk-free benchmark)
}

// Strategy Lab Types
export interface BLView {
  assetId: string;
  expectedReturn: number;
  confidence: number; // 0 to 1
}

export interface OptimizableAsset {
  id: string;
  expectedReturn: number;
  volatility: number;
}

export interface StrategyInput {
  assets: OptimizableAsset[];
  cov: number[][];
  corr?: number[][];
  constraints: OptimizerConstraints;
  returnsHistory?: number[][]; // T x N matrix of returns
  views?: BLView[];
  seed?: number;
  nSamples?: number; // For Resampled Efficiency (default 200)
  nScenarios?: number; // For CVaR optimizer
  cachedCholesky?: number[][]; // Optional precomputed Cholesky factor
  isWalkForward?: boolean; // Signal whether running in walk-forward backtest
}

export type StrategyOutput = OptimizerPoint & {
  strategyId: string;
  strategyName: string;
  equilibriumReturn?: number;
  posteriorReturn?: number;
};

export type Strategy = (input: StrategyInput) => StrategyOutput;

export type StrategyRegistry = Record<
  string,
  {
    name: string;
    description: string;
    fn: Strategy;
  }
>;

// Backtesting Types
export interface WalkForwardSubPeriod {
  label: string; // e.g. "2015-2019" or "2020-2024"
  startDate: string;
  endDate: string;
  cagr: number;
  volatility: number;
  sharpe: number;
  maxDrawdown: number;
}

export interface WalkForwardOptions {
  windowType?: 'expanding' | 'rolling'; // Default 'expanding' (min 60 months), 'rolling' uses rollingMonths
  rollingMonths?: number; // Default 120 months for rolling window
  holdMonths?: number; // Holding/rebalance period (default 12)
  trainWindowMonths?: number; // Legacy/explicit train window size (e.g. 36 for 36/12 legacy mode)
  rebalancePeriodMonths?: number; // Legacy alias for holdMonths (default 12)
  embargoMonths?: number; // Months to skip between end of training and start of hold (default 1)
  rebalanceCost?: number; // Default 0.002 (0.2% turnover fee)
  constraints?: OptimizerConstraints;
  seed?: number;
  cachedWindows?: any[];
  isWalkForward?: boolean;
}

export interface WalkForwardResult {
  strategyId: string;
  strategyName: string;
  monthlyReturns: number[];
  dates: string[];
  cagr: number;
  volatility: number;
  sharpe: number;
  maxDrawdown: number;
  worst12MonthReturn: number;
  turnover: number;
  cumulativeEquity: { date: string; value: number }[]; // Value starting at 100
  consistency?: number; // Share of rolling 36-month OOS windows in which strategy beat equal weight
  subPeriods?: WalkForwardSubPeriod[]; // Metrics by sub-period (5-year blocks / decades)
  excludedAssetsByWindow?: { t: number; excluded: string[] }[]; // Assets not yet started in training window
}

export interface StrategyScorecardItem {
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
  rankScore: number;
  overallRank: number;
  consistency?: number; // Share of rolling 36-month OOS windows where strategy beat equal weight
  subPeriods?: { label: string; cagr: number; sharpe: number }[];
}

export interface StrategyScorecardOptions {
  amount?: number;
  years?: number;
  mode?: 'lumpsum' | 'sip' | 'both';
  constraints?: OptimizerConstraints;
  monthlyReturnsByAsset?: Record<string, number[]>;
  bondConfig?: BondInput;
  inflationRate?: number;
  userAllocation?: Record<string, number>;
  seed?: number;
  onProgress?: (strategyId: string, doneCount: number, total: number) => void;
}

export interface StrategyScorecardResult {
  items: StrategyScorecardItem[];
  userPlanScorecard?: StrategyScorecardItem;
  consensusScorecard?: StrategyScorecardItem;
  topRankedStrategyId: string;
  verdict: string;
  isFallback: boolean;
  note?: string;
}

export interface ConsensusResult {
  weights: Record<string, number>;
  topStrategies: string[];
  expectedReturn: number;
  volatility: number;
  sharpe: number;
}

export interface GoalSolverInput {
  targetAmount: number;
  horizonYears: number;
  allocation: PortfolioAllocation;
  mode: 'lumpsum' | 'sip' | 'both';
  lumpsumAmount?: number;
  monthlySip?: number;
  assetStats: Record<string, { cagr: number; volatility: number }>;
  covarianceMatrix: number[][];
  assetIds: string[];
  monthlyReturnsByAsset?: Record<string, number[]>;
  bondConfig?: BondInput;
  inflationRate?: number;
  seed?: number;
  nSims?: number;
  sipPrecision?: number;
  accurateMode?: boolean;
}

export interface GoalSolverResult {
  probReachingMC: number;
  probReachingBootstrap: number;
  minSip70: number;
  minSip80: number;
  minSip90: number;
  minSip70Bootstrap?: number;
  minSip80Bootstrap?: number;
  minSip90Bootstrap?: number;
  infeasible?: boolean;
  parametric?: {
    minSip70: number;
    minSip80: number;
    minSip90: number;
    infeasible?: boolean;
  };
  bootstrap?: {
    minSip70: number;
    minSip80: number;
    minSip90: number;
    infeasible?: boolean;
  };
}
