/**
 * Optimizer Tab Component
 * Mean-Variance Markowitz Optimization & Efficient Frontier Visualization
 * Allows finding Max Return or Max Sharpe portfolios subject to risk and asset-cap constraints,
 * comparing with the current portfolio, and applying optimal weights back to the simulator.
 */
import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  Sparkles,
  TrendingUp,
  Shield,
  Target,
  ArrowRight,
  Info,
  CheckCircle2,
  RefreshCw,
  Sliders,
} from 'lucide-react';
import {
  computePortfolioReturn,
  computePortfolioVariance,
  runMonteCarlo,
  runOptimizerAsync,
  isSimulationCancelled,
  ALL_ASSET_DEFINITIONS,
  ASSET_MAP,
  OptimizerPoint,
  MonteCarloResult,
  BondInput,
  buildCovariance,
  analyzeBond,
  extendAssetsAndCovarianceWithBonds,
} from '../engine/index.ts';
import { Matrix } from 'ml-matrix';
import { formatINR, getAssetColor } from '../utils/formatters.ts';
import { weightsToRupeeAllocation } from '../utils/allocation.ts';
import { AssetAssumption } from './LeftPanel.tsx';

interface OptimizerTabProps {
  totalAmount: number;
  monthlyAmount?: number;
  horizonYears: number;
  currentAllocation: Record<string, number>; // assetId -> percent (0 to 100)
  assumptions: Record<string, AssetAssumption>;
  covarianceMatrix: number[][];
  correlationMatrix: number[][];
  assetIds: string[];
  bondConfig: BondInput;
  inflationRate: number; // decimal e.g. 0.06
  onApplyAllocation: (newAllocation: Record<string, number>) => void;
  maxRisk?: number;
  onMaxRiskChange?: (risk: number) => void;
  maxPerAsset?: number;
  onMaxPerAssetChange?: (cap: number) => void;
  isActive?: boolean;
}

export const OptimizerTab: React.FC<OptimizerTabProps> = React.memo(({
  totalAmount: initialAmount,
  monthlyAmount: initialMonthlyAmount,
  horizonYears: initialYears,
  currentAllocation,
  assumptions,
  covarianceMatrix,
  correlationMatrix,
  assetIds,
  bondConfig,
  inflationRate,
  onApplyAllocation,
  maxRisk: propMaxRisk,
  onMaxRiskChange,
  maxPerAsset: propMaxPerAsset,
  onMaxPerAssetChange,
  isActive = true,
}) => {
  const [totalAmount, setTotalAmount] = useState<number>(initialAmount);
  const [horizonYears, setHorizonYears] = useState<number>(initialYears);
  const [optMode, setOptMode] = useState<'max_sharpe' | 'max_return'>('max_sharpe');

  const [localMaxRisk, setLocalMaxRisk] = useState<number>(propMaxRisk ?? 15);
  const maxRisk = propMaxRisk ?? localMaxRisk;
  const setMaxRisk = (val: number) => {
    setLocalMaxRisk(val);
    if (onMaxRiskChange) onMaxRiskChange(val);
  };

  const [localMaxPerAsset, setLocalMaxPerAsset] = useState<number>(propMaxPerAsset ?? 30);
  const maxPerAsset = propMaxPerAsset ?? localMaxPerAsset;
  const setMaxPerAsset = (val: number) => {
    setLocalMaxPerAsset(val);
    if (onMaxPerAssetChange) onMaxPerAssetChange(val);
  };

  const [includeBonds, setIncludeBonds] = useState<boolean>(true);

  const [isOptimizing, setIsOptimizing] = useState<boolean>(false);
  const [optimalResult, setOptimalResult] = useState<OptimizerPoint | null>(null);
  const [frontierPoints, setFrontierPoints] = useState<OptimizerPoint[]>([]);
  const [mcResult, setMcResult] = useState<MonteCarloResult | null>(null);
  const [appliedFeedback, setAppliedFeedback] = useState<boolean>(false);

  const bondAnalysis = useMemo(() => {
    return analyzeBond(
      bondConfig || {
        couponRate: 0.075,
        tenureYears: horizonYears,
        payoutFrequency: 'annual',
        creditRisk: { defaultProbabilityAnnual: 0.005, recoveryRate: 0.5 },
        reinvestmentRateAfterMaturity: 0.065,
      }
    );
  }, [bondConfig, horizonYears]);

  const optimizableAssets = useMemo(() => {
    const ids = assetIds.length > 0 ? assetIds : Object.keys(assumptions);
    return ids.map((id) => {
      const assumption = assumptions[id] || { cagr: 0.12, volatility: 0.18 };
      return {
        id,
        expectedReturn: assumption.cagr,
        volatility: assumption.volatility,
      };
    });
  }, [assetIds, assumptions]);

  const { effectiveAssets, effectiveCov } = useMemo(() => {
    const ids = optimizableAssets.map((a) => a.id);
    const baseCov = buildCovariance(ids, assumptions, correlationMatrix, 1e-7, 0.2);
    if (includeBonds) {
      const extended = extendAssetsAndCovarianceWithBonds(
        optimizableAssets,
        baseCov,
        bondAnalysis.netExpectedYield,
        0.005
      );
      return {
        effectiveAssets: extended.assets,
        effectiveCov: extended.covarianceMatrix,
      };
    }
    return {
      effectiveAssets: optimizableAssets,
      effectiveCov: baseCov,
    };
  }, [optimizableAssets, assumptions, correlationMatrix, includeBonds, bondAnalysis]);

  const currentPortfolioStats = useMemo(() => {
    const n = effectiveAssets.length;
    const weightsVector: number[] = new Array(n).fill(0);
    const returnsVector = effectiveAssets.map((a) => a.expectedReturn);
    let activeSum = 0;
    for (let i = 0; i < n; i++) {
      const w = (currentAllocation[effectiveAssets[i].id] || 0) / 100;
      weightsVector[i] = w;
      activeSum += w;
    }
    if (activeSum > 0) {
      for (let i = 0; i < n; i++) weightsVector[i] /= activeSum;
    } else {
      weightsVector[0] = 1.0;
    }
    const ret = computePortfolioReturn(weightsVector, returnsVector);
    const covMat = new Matrix(effectiveCov);
    const vol = Math.sqrt(computePortfolioVariance(weightsVector, covMat));
    const sharpe = (ret - 0.065) / Math.max(1e-4, vol);
    return {
      returnPct: Number((ret * 100).toFixed(2)),
      volPct: Number((vol * 100).toFixed(2)),
      sharpe: Number(sharpe.toFixed(2)),
    };
  }, [effectiveAssets, currentAllocation, effectiveCov]);

  const lastRunInputsRef = useRef<string>('');
  const hasRunOnceRef = useRef<boolean>(false);

  const handleRunOptimizer = useCallback(async () => {
    if (effectiveAssets.length === 0) return;

    const currentParams = JSON.stringify({
      optMode,
      includeBonds,
      maxRisk,
      maxPerAsset,
      horizonYears,
      totalAmount,
      assetsCount: effectiveAssets.length,
    });
    lastRunInputsRef.current = currentParams;
    hasRunOnceRef.current = true;

    const DEBUG_PERF = false;
    if (DEBUG_PERF) performance.mark('optimizer-start');

    setIsOptimizing(true);
    setAppliedFeedback(false);

    try {
      const constraints = {
        maxRisk: maxRisk / 100,
        maxPerAsset: Math.max(1 / effectiveAssets.length, maxPerAsset / 100),
        riskFreeRate: 0.065,
      };

      const { optimalResult: result, frontierPoints: frontier } = await runOptimizerAsync({
        optMode,
        assets: effectiveAssets,
        covarianceMatrix: effectiveCov,
        constraints,
        frontierPointsCount: 30,
      });

      setOptimalResult(result);
      setFrontierPoints(frontier);

      const mc = await runMonteCarlo(
        {
          allocation: result.weights,
          years: horizonYears,
          mode: 'lumpsum',
          amount: totalAmount,
          assetStats: assumptions,
          covarianceMatrix: effectiveCov,
          assetIds: effectiveAssets.map((a) => a.id),
          bondConfig,
          nSims: 2000,
          inflationRate,
        },
        'optimizer'
      );
      setMcResult(mc);
    } catch (err: any) {
      if (isSimulationCancelled(err)) return;
    } finally {
      setIsOptimizing(false);
      if (DEBUG_PERF) {
        performance.mark('optimizer-end');
        performance.measure('optimizer-run', 'optimizer-start', 'optimizer-end');
        const entries = performance.getEntriesByName('optimizer-run');
        if (entries.length > 0) {
          console.log(`[PERF] Optimizer run: ${entries[entries.length - 1].duration.toFixed(2)}ms`);
        }
      }
    }
  }, [
    optMode,
    includeBonds,
    maxRisk,
    maxPerAsset,
    horizonYears,
    totalAmount,
    effectiveAssets,
    effectiveCov,
    assumptions,
    bondConfig,
    inflationRate,
  ]);

  useEffect(() => {
    const currentParams = JSON.stringify({
      optMode,
      includeBonds,
      maxRisk,
      maxPerAsset,
      horizonYears,
      totalAmount,
      assetsCount: effectiveAssets.length,
    });

    if (!isActive) {
      return;
    }

    if (hasRunOnceRef.current && lastRunInputsRef.current === currentParams) {
      return;
    }

    if (effectiveAssets.length === 0) return;

    const timer = setTimeout(() => {
      handleRunOptimizer();
    }, 400);

    return () => clearTimeout(timer);
  }, [
    isActive,
    optMode,
    includeBonds,
    maxRisk,
    maxPerAsset,
    effectiveAssets,
    handleRunOptimizer,
  ]);

  const handleApplyToSimulator = () => {
    if (!optimalResult) return;
    const mAmount = initialMonthlyAmount || 25000;
    const inrAlloc = weightsToRupeeAllocation(optimalResult.weights, mAmount);
    onApplyAllocation(inrAlloc);
    setAppliedFeedback(true);
    setTimeout(() => setAppliedFeedback(false), 3000);
  };

  const frontierChartData = useMemo(() => {
    return frontierPoints.map((pt, idx) => ({
      x: Number((pt.volatility * 100).toFixed(2)),
      y: Number((pt.expectedReturn * 100).toFixed(2)),
      sharpe: pt.sharpeRatio,
      index: idx,
    }));
  }, [frontierPoints]);

  const recommendedPointData = useMemo(() => {
    if (!optimalResult) return [];
    return [
      {
        x: Number((optimalResult.volatility * 100).toFixed(2)),
        y: Number((optimalResult.expectedReturn * 100).toFixed(2)),
        name: 'Recommended Optimal Portfolio',
        sharpe: optimalResult.sharpeRatio,
      },
    ];
  }, [optimalResult]);

  const currentPointData = useMemo(() => {
    return [
      {
        x: currentPortfolioStats.volPct,
        y: currentPortfolioStats.returnPct,
        name: 'Current Simulator Allocation',
        sharpe: currentPortfolioStats.sharpe,
      },
    ];
  }, [currentPortfolioStats]);

  const activeRecommendedAssets = useMemo(() => {
    if (!optimalResult) return [];
    return Object.entries(optimalResult.weights)
      .filter(([_, w]) => w > 0.005)
      .map(([id, w]) => {
        const pct = Number((w * 100).toFixed(1));
        return {
          id,
          name: ASSET_MAP[id]?.name || id,
          value: pct,
          color: getAssetColor(id),
          expectedReturn: assumptions[id]?.cagr ?? 0.12,
        };
      })
      .sort((a, b) => b.value - a.value);
  }, [optimalResult, assumptions]);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <section className="lg:col-span-5 bg-neutral-900/90 border border-neutral-800 rounded-xl p-5 space-y-5">
          <div className="flex items-center gap-2 border-b border-neutral-800 pb-3">
            <Sliders className="w-4 h-4 text-emerald-400" />
            <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-200">
              Optimizer Constraints &amp; Targets
            </h2>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label htmlFor="opt-capital-input" className="text-xs text-neutral-400">Total Investment</label>
              <div className="relative">
                <span className="absolute left-2.5 top-2 text-neutral-500 text-xs">₹</span>
                <input
                  id="opt-capital-input"
                  type="number"
                  min={10000}
                  step={50000}
                  value={totalAmount}
                  onChange={(e) => setTotalAmount(Math.max(10000, Number(e.target.value)))}
                  className="w-full bg-neutral-950 border border-neutral-800 rounded px-2 pl-6 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-emerald-500"
                />
              </div>
              <span className="text-[10px] text-neutral-500 font-mono">
                {formatINR(totalAmount)}
              </span>
            </div>
            <div className="space-y-1">
              <label htmlFor="opt-horizon-input" className="text-xs text-neutral-400">Horizon (Years)</label>
              <input
                id="opt-horizon-input"
                type="number"
                min={1}
                max={30}
                value={horizonYears}
                onChange={(e) => setHorizonYears(Math.min(30, Math.max(1, Number(e.target.value))))}
                className="w-full bg-neutral-950 border border-neutral-800 rounded px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-emerald-500"
              />
              <span className="text-[10px] text-neutral-500 font-mono">
                {horizonYears * 12} Months
              </span>
            </div>
          </div>

          <div className="space-y-2">
            <span className="text-xs text-neutral-300 font-medium block">Optimization Objective</span>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <button
                onClick={() => setOptMode('max_sharpe')}
                className={`p-2.5 rounded-lg border text-left transition cursor-pointer ${
                  optMode === 'max_sharpe'
                    ? 'bg-emerald-950/70 border-emerald-500/70 text-emerald-300'
                    : 'bg-neutral-950 border-neutral-800 text-neutral-400 hover:text-neutral-200'
                }`}
              >
                <div className="font-semibold text-xs text-white">Best Risk-Adjusted</div>
                <div className="text-[10px] text-neutral-400 mt-0.5">Maximize Sharpe Ratio</div>
              </button>
              <button
                onClick={() => setOptMode('max_return')}
                className={`p-2.5 rounded-lg border text-left transition cursor-pointer ${
                  optMode === 'max_return'
                    ? 'bg-emerald-950/70 border-emerald-500/70 text-emerald-300'
                    : 'bg-neutral-950 border-neutral-800 text-neutral-400 hover:text-neutral-200'
                }`}
              >
                <div className="font-semibold text-xs text-white">Max Return for Risk</div>
                <div className="text-[10px] text-neutral-400 mt-0.5">Cap at volatility ceiling</div>
              </button>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-neutral-300 font-medium">Max Risk (Annual Volatility Limit)</span>
              <span className="font-mono text-amber-400 font-bold bg-neutral-950 px-2 py-0.5 rounded border border-neutral-800">
                {maxRisk}% Volatility
              </span>
            </div>
            <input
              type="range"
              min={8}
              max={30}
              step={1}
              value={maxRisk}
              onChange={(e) => setMaxRisk(Number(e.target.value))}
              className="w-full accent-amber-500 cursor-pointer h-1.5 bg-neutral-800 rounded-lg"
            />
            <div className="flex justify-between text-[10px] text-neutral-500 font-mono">
              <span>8% (Low)</span>
              <span>15% (Moderate)</span>
              <span>22% (Equity)</span>
              <span>30% (Aggressive)</span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <div>
                <span className="text-neutral-300 font-medium block">Max Per-Asset Cap</span>
                <span className="text-[10px] text-neutral-500">Prevents hyper-concentrated allocations</span>
              </div>
              <span className="font-mono text-emerald-400 font-bold bg-neutral-950 px-2 py-0.5 rounded border border-neutral-800">
                {maxPerAsset}% Max
              </span>
            </div>
            <input
              type="range"
              min={10}
              max={100}
              step={5}
              value={maxPerAsset}
              onChange={(e) => setMaxPerAsset(Number(e.target.value))}
              className="w-full accent-emerald-500 cursor-pointer h-1.5 bg-neutral-800 rounded-lg"
            />
            <div className="flex justify-between text-[10px] text-neutral-500 font-mono">
              <span>10% (Diversified)</span>
              <span>30% (Recommended)</span>
              <span>50%</span>
              <span>100% (Unconstrained)</span>
            </div>
          </div>

          <div className="pt-2 border-t border-neutral-800/80 flex items-center justify-between text-xs">
            <div>
              <span className="text-neutral-300 font-medium block">Include Bonds / FD</span>
              <span className="text-[10px] text-neutral-500">
                {(bondAnalysis.netExpectedYield * 100).toFixed(1)}% yield • 0.5% risk
              </span>
            </div>
            <label className="relative inline-flex items-center cursor-pointer shrink-0">
              <input
                type="checkbox"
                checked={includeBonds}
                onChange={(e) => setIncludeBonds(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-9 h-5 bg-neutral-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-neutral-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-emerald-500"></div>
            </label>
          </div>

          <div className="pt-2">
            <button
              onClick={handleRunOptimizer}
              disabled={isOptimizing}
              className="w-full py-2.5 px-4 bg-emerald-500 hover:bg-emerald-400 text-neutral-950 font-semibold rounded-lg text-xs flex items-center justify-center gap-2 transition cursor-pointer shadow-md shadow-emerald-500/20 active:scale-[0.99] disabled:opacity-50"
            >
              {isOptimizing ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Optimizing Frontier...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Find Best Allocation</span>
                </>
              )}
            </button>
          </div>
        </section>

        <section className="lg:col-span-7 bg-neutral-900/90 border border-neutral-800 rounded-xl p-5 space-y-5">
          <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
            <div>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-200">
                Optimal Allocation Summary
              </h2>
              <span className="text-[11px] text-neutral-500">
                Subject to {maxPerAsset}% asset cap &amp; {maxRisk}% volatility ceiling
              </span>
            </div>
            <button
              onClick={handleApplyToSimulator}
              disabled={!optimalResult}
              className={`px-3.5 py-1.5 text-xs font-semibold rounded-lg flex items-center gap-1.5 transition cursor-pointer ${
                appliedFeedback
                  ? 'bg-emerald-900/80 text-emerald-300 border border-emerald-700'
                  : 'bg-neutral-100 hover:bg-white text-neutral-950 shadow-sm'
              }`}
            >
              {appliedFeedback ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Applied to Simulator!</span>
                </>
              ) : (
                <>
                  <span>Apply to Simulator</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </>
              )}
            </button>
          </div>

          {optimalResult?.infeasible && (
            <div className="p-3 rounded-lg bg-amber-950/70 border border-amber-800 text-xs text-amber-300 flex items-start gap-2">
              <Info className="w-4 h-4 shrink-0 mt-0.5 text-amber-400" />
              <div>
                <strong>Infeasible Risk Constraint:</strong> Your maximum risk limit ({maxRisk}%) is lower
                than the minimum possible portfolio volatility ({(optimalResult.volatility * 100).toFixed(1)}%)
                under the {maxPerAsset}% asset cap. Showing the minimum-variance allocation.
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-neutral-950 border border-neutral-800/80 p-3 rounded-lg">
              <div className="text-[11px] text-neutral-400 flex items-center justify-between">
                <span>Expected Return</span>
                <TrendingUp className="w-3 h-3 text-emerald-400" />
              </div>
              <div className="text-lg font-bold font-mono text-emerald-400 mt-0.5">
                {optimalResult ? `${(optimalResult.expectedReturn * 100).toFixed(1)}%` : '--'}
              </div>
              <div className="text-[10px] text-neutral-500 font-mono">CAGR p.a.</div>
            </div>
            <div className="bg-neutral-950 border border-neutral-800/80 p-3 rounded-lg">
              <div className="text-[11px] text-neutral-400 flex items-center justify-between">
                <span>Volatility (Risk)</span>
                <Shield className="w-3 h-3 text-amber-400" />
              </div>
              <div className="text-lg font-bold font-mono text-amber-400 mt-0.5">
                {optimalResult ? `${(optimalResult.volatility * 100).toFixed(1)}%` : '--'}
              </div>
              <div className="text-[10px] text-neutral-500 font-mono">Annual Std Dev</div>
            </div>
            <div className="bg-neutral-950 border border-neutral-800/80 p-3 rounded-lg">
              <div className="text-[11px] text-neutral-400 flex items-center justify-between">
                <span>Sharpe Ratio</span>
                <Target className="w-3 h-3 text-sky-400" />
              </div>
              <div className="text-lg font-bold font-mono text-sky-300 mt-0.5">
                {optimalResult ? optimalResult.sharpeRatio.toFixed(2) : '--'}
              </div>
              <div className="text-[10px] text-neutral-500 font-mono">Rf = 6.5%</div>
            </div>
            <div className="bg-neutral-950 border border-neutral-800/80 p-3 rounded-lg">
              <div className="text-[11px] text-neutral-400 flex items-center justify-between">
                <span>Monte Carlo (P50)</span>
                <Sparkles className="w-3 h-3 text-teal-400" />
              </div>
              <div className="text-lg font-bold font-mono text-teal-300 mt-0.5">
                {mcResult ? formatINR(mcResult.terminalStats.median) : '--'}
              </div>
              <div className="text-[10px] text-neutral-500 font-mono">
                {horizonYears}Y Terminal Median
              </div>
            </div>
          </div>

          {mcResult && (
            <div className="bg-neutral-950/60 border border-neutral-800/60 p-3 rounded-lg flex items-center justify-between text-xs font-mono">
              <div className="text-neutral-400">
                P10 Crash:{' '}
                <span className="text-rose-400 font-bold">
                  {formatINR(mcResult.percentilesByYear[mcResult.percentilesByYear.length - 1].p10)}
                </span>
              </div>
              <div className="text-neutral-400">
                P50 Median:{' '}
                <span className="text-emerald-400 font-bold">
                  {formatINR(mcResult.terminalStats.median)}
                </span>
              </div>
              <div className="text-neutral-400">
                P90 Boom:{' '}
                <span className="text-teal-300 font-bold">
                  {formatINR(mcResult.percentilesByYear[mcResult.percentilesByYear.length - 1].p90)}
                </span>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-center">
            <div className="sm:col-span-5 h-[170px] w-full flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={activeRecommendedAssets}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={45}
                    outerRadius={70}
                    paddingAngle={3}
                  >
                    {activeRecommendedAssets.map((entry) => (
                      <Cell key={entry.id} fill={entry.color} stroke="#0a0a0a" strokeWidth={2} />
                    ))}
                  </Pie>
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const d = payload[0].payload;
                        return (
                          <div className="bg-neutral-950 border border-neutral-800 p-2 rounded text-xs font-mono">
                            <div className="font-semibold text-white">{d.name}</div>
                            <div className="text-emerald-400 font-bold">{d.value}% weight</div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="sm:col-span-7 space-y-1.5 max-h-[170px] overflow-y-auto pr-1">
              {activeRecommendedAssets.map((asset) => (
                <div
                  key={asset.id}
                  className="flex items-center justify-between p-1.5 rounded bg-neutral-950 border border-neutral-800/60 text-xs"
                >
                  <div className="flex items-center gap-2 truncate">
                    <span
                      className="w-2.5 h-2.5 rounded-full shrink-0"
                      style={{ backgroundColor: asset.color }}
                    ></span>
                    <span className="text-neutral-200 truncate font-medium">{asset.name}</span>
                  </div>
                  <div className="flex items-center gap-3 shrink-0 font-mono">
                    <span className="text-neutral-500 text-[10px]">
                      μ: {(asset.expectedReturn * 100).toFixed(1)}%
                    </span>
                    <span className="text-emerald-400 font-bold">{asset.value}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>

      <section className="bg-neutral-900/90 border border-neutral-800 rounded-xl p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-neutral-800 pb-3">
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-200">
              Markowitz Efficient Frontier Curve
            </h2>
            <p className="text-[11px] text-neutral-500">
              30 optimal risk-return portfolios mapped against your current allocation
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 inline-block ring-2 ring-emerald-400/40"></span>
              <span className="text-emerald-300">Recommended Optimal</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-sky-400 inline-block ring-2 ring-sky-400/40"></span>
              <span className="text-sky-300">Current Simulator Allocation</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-neutral-500 inline-block"></span>
              <span className="text-neutral-400">Frontier Points (30)</span>
            </div>
          </div>
        </div>

        <div className="h-[340px] w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 15, right: 25, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
              <XAxis
                type="number"
                dataKey="x"
                name="Volatility (Risk)"
                unit="%"
                stroke="#737373"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#404040' }}
                label={{
                  value: 'Annualized Volatility (Portfolio Risk %)',
                  position: 'bottom',
                  offset: 5,
                  fill: '#737373',
                  fontSize: 11,
                }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="Expected Return"
                unit="%"
                stroke="#737373"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#404040' }}
                label={{
                  value: 'Expected Annual Return (CAGR %)',
                  angle: -90,
                  position: 'insideLeft',
                  fill: '#737373',
                  fontSize: 11,
                }}
              />
              <ZAxis range={[50, 400]} />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-neutral-950 border border-neutral-800 p-3 rounded-lg shadow-2xl text-xs space-y-1 font-mono">
                        <div className="font-semibold text-white border-b border-neutral-800 pb-1">
                          {data.name || `Frontier Point #${data.index + 1}`}
                        </div>
                        <div className="text-emerald-400">Expected Return: {data.y}%</div>
                        <div className="text-amber-400">Volatility (Risk): {data.x}%</div>
                        <div className="text-sky-300">Sharpe Ratio: {data.sharpe?.toFixed(2)}</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter
                name="Efficient Frontier Curve"
                data={frontierChartData}
                fill="#525252"
                line={{ stroke: '#10b981', strokeWidth: 1.5, strokeDasharray: '3 3' }}
              />
              <Scatter
                name="Recommended Portfolio"
                data={recommendedPointData}
                fill="#10b981"
                shape="circle"
              />
              <Scatter
                name="Current Portfolio"
                data={currentPointData}
                fill="#38bdf8"
                shape="diamond"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="p-3.5 rounded-lg bg-neutral-950 border border-neutral-800 flex items-center justify-between gap-4 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="text-neutral-400">Current Simulator:</span>
            <span className="text-white font-bold">{currentPortfolioStats.returnPct}% Return</span>
            <span className="text-neutral-600">|</span>
            <span className="text-neutral-400">{currentPortfolioStats.volPct}% Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-neutral-400">Optimized Recommendation:</span>
            <span className="text-emerald-400 font-bold">
              {optimalResult ? `${(optimalResult.expectedReturn * 100).toFixed(1)}% Return` : '--'}
            </span>
            <span className="text-neutral-600">|</span>
            <span className="text-amber-400">
              {optimalResult ? `${(optimalResult.volatility * 100).toFixed(1)}% Risk` : '--'}
            </span>
          </div>
        </div>
      </section>

      <div className="p-4 rounded-xl bg-neutral-900/60 border border-neutral-800/80 flex items-start gap-3 text-xs text-neutral-400">
        <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <strong className="text-neutral-200">Model Disclosure:</strong> The recommended allocation
          is mathematically optimal based on historical returns, volatilities, and cross-asset
          correlations, and will change as market regimes evolve. Highly concentrated allocations
          are intentionally constrained by your per-asset cap ({maxPerAsset}%) on purpose to prevent
          overfitting and enforce institutional risk diversification.
        </p>
      </div>
    </div>
  );
});

OptimizerTab.displayName = 'OptimizerTab';
