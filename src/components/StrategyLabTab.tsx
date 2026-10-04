/**
 * Strategy Lab Tab Component
 * Evaluates 11 quantitative portfolio strategies, Walk-Forward Out-of-Sample metrics,
 * Historical Block Bootstrap, Consensus Blending, and Financial Goal Solving.
 */
import React, { useState, useEffect, useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import {
  Sparkles,
  Shield,
  Target,
  ArrowRight,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Sliders,
  ChevronDown,
  Layers,
  ArrowUpDown,
  Award,
} from 'lucide-react';
import {
  ALL_ASSET_DEFINITIONS,
  ASSET_MAP,
  OptimizerConstraints,
  StrategyScorecardItem,
  StrategyScorecardResult,
  GoalSolverResult,
  runGoalSolverAsync,
  terminateGoalSolver,
  buildCovariance,
} from '../engine/index.ts';
import { runStrategyLabScorecardAsync } from '../engine/strategyLabService.ts';
import { formatINR, getAssetColor, formatPercent } from '../utils/formatters.ts';
import { weightsToRupeeAllocation, rupeeAllocationToWeights } from '../utils/allocation.ts';

interface TargetCorpusInputProps {
  initialValue: number;
  onCommit: (val: number) => void;
}

const TargetCorpusInput: React.FC<TargetCorpusInputProps> = React.memo(({ initialValue, onCommit }) => {
  const [localValue, setLocalValue] = useState<string>(String(initialValue));

  useEffect(() => {
    setLocalValue(String(initialValue));
  }, [initialValue]);

  const handleCommit = () => {
    const num = Math.max(10000, Number(localValue) || 0);
    setLocalValue(String(num));
    onCommit(num);
  };

  return (
    <div className="relative">
      <span className="absolute left-3 top-2 text-neutral-500 text-xs">₹</span>
      <input
        id="goal-target-corpus"
        type="number"
        min={100000}
        step={500000}
        value={localValue}
        onChange={(e) => setLocalValue(e.target.value)}
        onBlur={handleCommit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            handleCommit();
            (e.target as HTMLInputElement).blur();
          }
        }}
        className="w-full bg-neutral-950 border border-neutral-800 rounded px-2 pl-6 py-2 text-sm font-mono text-white focus:outline-none focus:border-emerald-500"
      />
    </div>
  );
});

export interface StrategyLabTabProps {
  amount?: number;
  totalAmount?: number;
  monthlySip?: number;
  horizon?: number;
  horizonYears?: number;
  mode?: 'lumpsum' | 'sip' | 'both';
  allocation?: Record<string, number>;
  userAllocation?: Record<string, number>;
  userAllocationInr?: Record<string, number>;
  assumptions: Record<string, { cagr: number; volatility: number }>;
  inflationRate: number; // decimal e.g. 0.06
  bondConfig?: any;
  constraints?: OptimizerConstraints;
  maxRisk?: number;
  maxPerAsset?: number;
  haircutSettings?: {
    returnHaircutS?: number;
    categoryPriors?: Record<string, number>;
    rawHistoricalCagrs?: Record<string, number>;
  };
  returnHaircutS?: number;
  categoryPriors?: Record<string, number>;
  rawHistoricalCagrs?: Record<string, number>;
  covarianceMatrix?: number[][];
  correlationMatrix?: number[][];
  assetIds?: string[];
  isFallback?: boolean;
  fallbackFlag?: boolean;
  onApplyAllocationInr?: (newAllocationInr: Record<string, number>) => void;
  onApplyAllocation?: (newAllocationInr: Record<string, number>) => void;
}

export const StrategyLabTab: React.FC<StrategyLabTabProps> = React.memo(({
  totalAmount: initialTotal,
  amount: propAmount,
  monthlySip: initialSip,
  horizonYears: initialYears,
  horizon: propHorizon,
  mode: initialMode,
  allocation: propAllocation,
  userAllocation: propUserAllocation,
  userAllocationInr,
  assumptions,
  inflationRate,
  bondConfig,
  constraints: propConstraints,
  maxRisk: propMaxRisk,
  maxPerAsset: propMaxPerAsset,
  haircutSettings,
  returnHaircutS,
  categoryPriors,
  rawHistoricalCagrs,
  covarianceMatrix: propCovarianceMatrix,
  correlationMatrix: propCorrelationMatrix,
  assetIds: propAssetIds,
  isFallback: propIsFallback,
  fallbackFlag: propFallbackFlag,
  onApplyAllocationInr,
  onApplyAllocation,
}) => {
  const resolvedMonthlySip = initialSip ?? propAmount ?? 25000;
  const resolvedHorizon = initialYears ?? propHorizon ?? 10;
  const resolvedMode = initialMode ?? 'sip';
  const isFallback = propIsFallback ?? propFallbackFlag ?? false;

  const userWeights = useMemo(() => {
    if (propUserAllocation) return propUserAllocation;
    if (propAllocation) return propAllocation;
    if (userAllocationInr) return rupeeAllocationToWeights(userAllocationInr);
    return { nifty_50: 1.0 };
  }, [propUserAllocation, propAllocation, userAllocationInr]);

  const [totalAmount, setTotalAmount] = useState<number>(initialTotal || resolvedMonthlySip * 12 * resolvedHorizon);
  const [monthlySip, setMonthlySip] = useState<number>(resolvedMonthlySip);
  const [horizonYears, setHorizonYears] = useState<number>(resolvedHorizon);
  const defaultMaxRisk = propConstraints?.maxRisk ? propConstraints.maxRisk * 100 : (propMaxRisk ?? 18);
  const defaultMaxPerAsset = propConstraints?.maxPerAsset ? propConstraints.maxPerAsset * 100 : (propMaxPerAsset ?? 35);
  const [maxRisk, setMaxRisk] = useState<number>(defaultMaxRisk);
  const [maxPerAsset, setMaxPerAsset] = useState<number>(defaultMaxPerAsset);

  useEffect(() => {
    if (resolvedMonthlySip) setMonthlySip(resolvedMonthlySip);
  }, [resolvedMonthlySip]);

  useEffect(() => {
    if (resolvedHorizon) setHorizonYears(resolvedHorizon);
  }, [resolvedHorizon]);

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [progressPct, setProgressPct] = useState<number>(0);
  const [scorecardResult, setScorecardResult] = useState<StrategyScorecardResult | null>(null);
  const [scorecardError, setScorecardError] = useState<string | null>(null);
  const [selectedStrategyId, setSelectedStrategyId] = useState<string>('hrp');
  const [metricViewMode, setMetricViewMode] = useState<'bootstrap' | 'parametric'>('bootstrap');
  const [sortColumn, setSortColumn] = useState<string>('rankScore');
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const [appliedFeedbackId, setAppliedFeedbackId] = useState<string | null>(null);

  const [targetCorpus, setTargetCorpus] = useState<number>(10000000); // ₹ 1 Crore
  const [accurateMode, setAccurateMode] = useState<boolean>(false);
  const [goalResult, setGoalResult] = useState<GoalSolverResult | null>(null);
  const [isSolvingGoal, setIsSolvingGoal] = useState<boolean>(false);
  const [goalError, setGoalError] = useState<string | null>(null);

  const [lastSolvedGoalInputs, setLastSolvedGoalInputs] = useState<{
    targetCorpus: number;
    strategyId: string;
    horizonYears: number;
    monthlySip: number;
  } | null>(null);

  const [lastRunInputs, setLastRunInputs] = useState<{
    monthlySip: number;
    horizonYears: number;
    maxRisk: number;
    maxPerAsset: number;
    inflationRate: number;
  } | null>(null);

  const isGoalStale = useMemo(() => {
    if (!goalResult || !lastSolvedGoalInputs) return false;
    return (
      lastSolvedGoalInputs.targetCorpus !== targetCorpus ||
      lastSolvedGoalInputs.strategyId !== selectedStrategyId ||
      lastSolvedGoalInputs.horizonYears !== horizonYears ||
      lastSolvedGoalInputs.monthlySip !== monthlySip
    );
  }, [goalResult, lastSolvedGoalInputs, targetCorpus, selectedStrategyId, horizonYears, monthlySip]);

  const isScorecardStale = useMemo(() => {
    if (!scorecardResult || !lastRunInputs) return false;
    return (
      lastRunInputs.monthlySip !== monthlySip ||
      lastRunInputs.horizonYears !== horizonYears ||
      lastRunInputs.maxRisk !== maxRisk ||
      lastRunInputs.maxPerAsset !== maxPerAsset ||
      lastRunInputs.inflationRate !== inflationRate
    );
  }, [scorecardResult, lastRunInputs, monthlySip, horizonYears, maxRisk, maxPerAsset, inflationRate]);

  const constraints: OptimizerConstraints = useMemo(
    () => ({
      maxRisk: maxRisk / 100,
      maxPerAsset: maxPerAsset / 100,
      minPerAsset: 0.0,
      riskFreeRate: 0.065,
    }),
    [maxRisk, maxPerAsset]
  );

  const handleRunAllStrategies = async () => {
    setIsRunning(true);
    setScorecardError(null);
    setProgressPct(10);
    try {
      const result = await runStrategyLabScorecardAsync(
        {
          amount: monthlySip,
          years: horizonYears,
          mode: 'sip',
          constraints,
          bondConfig,
          inflationRate,
          userAllocation: userWeights,
          seed: 42,
        },
        (pct) => setProgressPct(pct)
      );
      setScorecardResult(result);
      setLastRunInputs({
        monthlySip,
        horizonYears,
        maxRisk,
        maxPerAsset,
        inflationRate,
      });
      if (result.items.length > 0) {
        setSelectedStrategyId(result.topRankedStrategyId || result.items[0].strategyId);
      }
    } catch (err: any) {
      setScorecardError(err?.message || 'Strategy evaluation failed. Please retry.');
    } finally {
      setIsRunning(false);
      setProgressPct(100);
    }
  };

  const handleCalculateGoal = async () => {
    setIsSolvingGoal(true);
    setGoalError(null);
    try {
      const activeStrat =
        selectedStrategyId === 'user_plan'
          ? scorecardResult?.userPlanScorecard
          : selectedStrategyId === 'consensus'
          ? scorecardResult?.consensusScorecard
          : scorecardResult?.items.find((i) => i.strategyId === selectedStrategyId);
      const alloc = activeStrat ? activeStrat.weights : userWeights;

      const activeIds = propAssetIds && propAssetIds.length > 0
        ? propAssetIds
        : ALL_ASSET_DEFINITIONS.filter((a) => a.id !== 'bonds_fd').map((a) => a.id);
      const covMat = propCovarianceMatrix && propCovarianceMatrix.length === activeIds.length
        ? propCovarianceMatrix
        : buildCovariance(activeIds, assumptions, propCorrelationMatrix || []);

      const gRes = await runGoalSolverAsync({
        targetAmount: targetCorpus,
        horizonYears,
        allocation: alloc,
        mode: 'sip',
        monthlySip,
        assetStats: assumptions,
        covarianceMatrix: covMat,
        assetIds: activeIds,
        bondConfig,
        inflationRate,
        nSims: accurateMode ? 5000 : 2000,
        sipPrecision: accurateMode ? 100 : 250,
        accurateMode,
      });
      setGoalResult(gRes);
      setLastSolvedGoalInputs({
        targetCorpus,
        strategyId: selectedStrategyId,
        horizonYears,
        monthlySip,
      });
    } catch (err: any) {
      setGoalError(err?.message || 'Goal solver execution failed.');
    } finally {
      setIsSolvingGoal(false);
    }
  };

  useEffect(() => {
    return () => {
      terminateGoalSolver();
    };
  }, []);

  const handleApplyStrategy = (weights: Record<string, number>, stratId: string) => {
    const inrAlloc = weightsToRupeeAllocation(weights, monthlySip);
    if (onApplyAllocationInr) onApplyAllocationInr(inrAlloc);
    if (onApplyAllocation) onApplyAllocation(inrAlloc);
    setAppliedFeedbackId(stratId);
    setTimeout(() => setAppliedFeedbackId(null), 3000);
  };

  const sortedTableRows = useMemo(() => {
    if (!scorecardResult) return [];
    const list: StrategyScorecardItem[] = [];
    if (scorecardResult.consensusScorecard) {
      list.push(scorecardResult.consensusScorecard);
    }
    list.push(...scorecardResult.items);
    list.sort((a, b) => {
      let valA = (a as any)[sortColumn] ?? 0;
      let valB = (b as any)[sortColumn] ?? 0;
      if (typeof valA === 'string') {
        return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }
      return sortAsc ? valA - valB : valB - valA;
    });
    const finalRows: StrategyScorecardItem[] = [];
    if (scorecardResult.userPlanScorecard) {
      finalRows.push(scorecardResult.userPlanScorecard);
    }
    finalRows.push(...list);
    return finalRows;
  }, [scorecardResult, sortColumn, sortAsc]);

  const bestColumnValues = useMemo(() => {
    if (!scorecardResult) return {};
    const items = scorecardResult.items;
    return {
      inSampleReturn: Math.max(...items.map((i) => i.inSampleReturn)),
      inSampleVol: Math.min(...items.map((i) => i.inSampleVol)),
      oosSharpe: Math.max(...items.map((i) => i.oosSharpe)),
      maxDrawdown: Math.min(...items.map((i) => i.maxDrawdown)),
      bootstrapP10: Math.max(...items.map((i) => i.bootstrapP10)),
      bootstrapP50: Math.max(...items.map((i) => i.bootstrapP50)),
      bootstrapP90: Math.max(...items.map((i) => i.bootstrapP90)),
      parametricP50: Math.max(...items.map((i) => i.parametricP50)),
      probLoss: Math.min(...items.map((i) => i.probLoss)),
      rankScore: Math.max(...items.map((i) => i.rankScore)),
    };
  }, [scorecardResult]);

  const selectedStrategy = useMemo(() => {
    if (!scorecardResult) return null;
    if (selectedStrategyId === 'user_plan') return scorecardResult.userPlanScorecard;
    if (selectedStrategyId === 'consensus') return scorecardResult.consensusScorecard;
    return scorecardResult.items.find((i) => i.strategyId === selectedStrategyId) || scorecardResult.items[0];
  }, [scorecardResult, selectedStrategyId]);

  const fanChartData = useMemo(() => {
    if (!selectedStrategy) return [];
    const data: Array<{ year: number; selectedP10: number; selectedP50: number; selectedP90: number; userP50?: number }> = [];
    for (let y = 1; y <= horizonYears; y++) {
      const frac = y / horizonYears;
      const selP10 = (monthlySip * 12 * y) + (selectedStrategy.bootstrapP10 - monthlySip * 12 * horizonYears) * frac;
      const selP50 = (monthlySip * 12 * y) + (selectedStrategy.bootstrapP50 - monthlySip * 12 * horizonYears) * frac;
      const selP90 = (monthlySip * 12 * y) + (selectedStrategy.bootstrapP90 - monthlySip * 12 * horizonYears) * frac;
      let uP50: number | undefined;
      if (scorecardResult?.userPlanScorecard) {
        uP50 = (monthlySip * 12 * y) + (scorecardResult.userPlanScorecard.bootstrapP50 - monthlySip * 12 * horizonYears) * frac;
      }
      data.push({
        year: y,
        selectedP10: Math.max(0, selP10),
        selectedP50: Math.max(0, selP50),
        selectedP90: Math.max(0, selP90),
        userP50: uP50 ? Math.max(0, uP50) : undefined,
      });
    }
    return data;
  }, [selectedStrategy, scorecardResult, horizonYears, monthlySip]);

  const allocationComparisonData = useMemo(() => {
    if (!scorecardResult) return [];
    const top3 = scorecardResult.items.slice(0, 3);
    const plans: Array<{ name: string; weights: Record<string, number> }> = [];
    if (scorecardResult.userPlanScorecard) {
      plans.push({ name: 'Your Plan', weights: scorecardResult.userPlanScorecard.weights });
    }
    top3.forEach((t) => {
      plans.push({ name: t.strategyName, weights: t.weights });
    });
    const activeAssetIds = new Set<string>();
    plans.forEach((p) => {
      Object.keys(p.weights).forEach((id) => {
        if (p.weights[id] > 0.02) activeAssetIds.add(id);
      });
    });
    return plans.map((p) => {
      const row: any = { planName: p.name };
      activeAssetIds.forEach((id) => {
        row[id] = Number(((p.weights[id] || 0) * 100).toFixed(1));
      });
      return row;
    });
  }, [scorecardResult]);

  const equityCurvesData = useMemo(() => {
    const monthsCount = 48;
    const points: any[] = [];
    for (let m = 0; m <= monthsCount; m++) {
      const pt: any = { month: m === 0 ? 'Start' : `M${m}` };
      const frac = m / 12;
      if (scorecardResult?.userPlanScorecard) {
        pt['Your Plan'] = Number((100 * Math.pow(1 + (scorecardResult.userPlanScorecard.oosCagr || 0.12), frac)).toFixed(1));
      }
      if (scorecardResult?.items) {
        scorecardResult.items.slice(0, 4).forEach((strat) => {
          pt[strat.strategyName] = Number((100 * Math.pow(1 + (strat.oosCagr || 0.14), frac)).toFixed(1));
        });
      }
      points.push(pt);
    }
    return points;
  }, [scorecardResult]);

  return (
    <div className="space-y-6">
      <div className="p-3.5 rounded-xl bg-neutral-900/90 border border-neutral-800 space-y-1.5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-2">
            <Award className="w-4 h-4 text-emerald-400 shrink-0" />
            <span className="font-semibold text-neutral-200">Strategy Lab: Out-of-Sample Quantitative Benchmarking</span>
          </div>
          {isFallback ? (
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-500/15 text-amber-300 border border-amber-500/30">
              Illustrative synthetic data, not real market history
            </span>
          ) : (
            <span className="text-[11px] text-neutral-400 font-mono">10Y Rolling Walk-Forward Backtest</span>
          )}
        </div>
        <p className="text-[11px] text-neutral-400 leading-relaxed">
          *<strong>Mandatory Honesty Note:</strong> Ranking uses out-of-sample walk-forward results because in-sample
          &quot;max return&quot; overfits the 2014–2024 Indian bull market. All forward projections depend strictly on user
          return assumptions and historical distributions.
        </p>
      </div>

      {scorecardResult && (
        <div className="p-4 rounded-xl bg-emerald-950/30 border border-emerald-500/30 text-xs space-y-1">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold uppercase tracking-wider text-[11px]">
            <Sparkles className="w-4 h-4" />
            <span>Quantitative Verdict</span>
          </div>
          <p className="text-neutral-200 text-xs sm:text-sm font-medium leading-relaxed">
            {scorecardResult.verdict}
          </p>
        </div>
      )}

      {scorecardError && (
        <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-500/30 flex items-center gap-2 text-rose-300 text-xs">
          <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{scorecardError}</span>
        </div>
      )}

      <section className="bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 sm:p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-200">
                Strategy Optimization Parameters
              </h2>
              {isScorecardStale && (
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-500/20 text-amber-300 border border-amber-500/40">
                  stale, inputs changed
                </span>
              )}
            </div>
            <span className="text-[11px] text-neutral-500">
              Shared budget: ₹{formatINR(monthlySip, false)}/mo • Horizon: {horizonYears} Years
            </span>
          </div>
          <button
            type="button"
            onClick={handleRunAllStrategies}
            disabled={isRunning}
            className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-neutral-950 font-semibold rounded-lg text-xs flex items-center justify-center gap-2 transition cursor-pointer shadow-md shadow-emerald-500/20 active:scale-[0.99] disabled:opacity-50"
          >
            {isRunning ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Running 11 Strategies ({progressPct}%)...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-3.5 h-3.5" />
                <span>Run All Strategies</span>
              </>
            )}
          </button>
        </div>

        {isRunning && (
          <div className="space-y-1">
            <div className="h-1.5 w-full bg-neutral-950 rounded-full overflow-hidden border border-neutral-800">
              <div
                style={{ width: `${progressPct}%` }}
                className="h-full bg-emerald-500 transition-all duration-300"
              />
            </div>
            <span className="text-[10px] text-neutral-400 font-mono block text-right">
              Computing Walk-Forward Backtests &amp; Bootstrap Paths...
            </span>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-neutral-800/80">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-neutral-400">Max Volatility Ceiling</span>
              <span className="font-mono text-amber-400 font-bold">{maxRisk}% Risk</span>
            </div>
            <input
              type="range"
              min={8}
              max={30}
              step={1}
              value={maxRisk}
              onChange={(e) => setMaxRisk(Number(e.target.value))}
              className="w-full h-1 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-neutral-400">Max Per-Asset Cap</span>
              <span className="font-mono text-emerald-400 font-bold">{maxPerAsset}% Cap</span>
            </div>
            <input
              type="range"
              min={10}
              max={100}
              step={5}
              value={maxPerAsset}
              onChange={(e) => setMaxPerAsset(Number(e.target.value))}
              className="w-full h-1 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            />
          </div>
        </div>
      </section>

      {/* 4. "Your Plan vs Best Plans" Master Comparison Table */}
      <section className="bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 sm:p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-neutral-800 pb-3">
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-200">
              Your Plan vs Best Strategies
            </h2>
            <div className="flex flex-col gap-0.5 mt-0.5">
              <span className="text-[11px] text-neutral-500">
                Ranked by 0.4 × OOS Sharpe + 0.3 × Bootstrap P10 + 0.3 × Max Drawdown
              </span>
              <span className="text-[11px] text-amber-400/90 font-medium">
                {scorecardResult?.note || "Note: In-sample strategies use the user's assumption returns while the walk-forward uses trailing 36-month historical means (two different return views)."}
              </span>
            </div>
          </div>
          <div className="inline-flex rounded-lg p-0.5 bg-neutral-950 border border-neutral-800 text-[11px]">
            <button
              type="button"
              onClick={() => setMetricViewMode('bootstrap')}
              className={`px-3 py-1 rounded-md font-medium transition cursor-pointer ${
                metricViewMode === 'bootstrap'
                  ? 'bg-neutral-800 text-emerald-400 shadow-sm'
                  : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              Historical Bootstrap
            </button>
            <button
              type="button"
              onClick={() => setMetricViewMode('parametric')}
              className={`px-3 py-1 rounded-md font-medium transition cursor-pointer ${
                metricViewMode === 'parametric'
                  ? 'bg-neutral-800 text-emerald-400 shadow-sm'
                  : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              Parametric Monte Carlo
            </button>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-neutral-800 text-[10px] font-mono text-neutral-400 uppercase">
                <th className="py-2.5 px-3">Strategy</th>
                <th
                  onClick={() => {
                    setSortColumn('inSampleReturn');
                    setSortAsc(!sortAsc);
                  }}
                  className="py-2.5 px-3 text-right cursor-pointer hover:text-white"
                >
                  Exp. Return <ArrowUpDown className="inline w-3 h-3 ml-0.5" />
                </th>
                <th
                  onClick={() => {
                    setSortColumn('inSampleVol');
                    setSortAsc(!sortAsc);
                  }}
                  className="py-2.5 px-3 text-right cursor-pointer hover:text-white"
                >
                  Volatility <ArrowUpDown className="inline w-3 h-3 ml-0.5" />
                </th>
                <th
                  onClick={() => {
                    setSortColumn('oosSharpe');
                    setSortAsc(!sortAsc);
                  }}
                  className="py-2.5 px-3 text-right cursor-pointer hover:text-white"
                >
                  OOS Sharpe <ArrowUpDown className="inline w-3 h-3 ml-0.5" />
                </th>
                <th
                  onClick={() => {
                    setSortColumn('maxDrawdown');
                    setSortAsc(!sortAsc);
                  }}
                  className="py-2.5 px-3 text-right cursor-pointer hover:text-white"
                >
                  Max Drawdown <ArrowUpDown className="inline w-3 h-3 ml-0.5" />
                </th>
                <th
                  onClick={() => {
                    setSortColumn(metricViewMode === 'bootstrap' ? 'bootstrapP50' : 'parametricP50');
                    setSortAsc(!sortAsc);
                  }}
                  className="py-2.5 px-3 text-right cursor-pointer hover:text-white"
                >
                  Median (P50) <ArrowUpDown className="inline w-3 h-3 ml-0.5" />
                </th>
                <th className="py-2.5 px-3 text-right">P10 – P90 Band</th>
                <th
                  onClick={() => {
                    setSortColumn('probLoss');
                    setSortAsc(!sortAsc);
                  }}
                  className="py-2.5 px-3 text-right cursor-pointer hover:text-white"
                >
                  Prob. Loss <ArrowUpDown className="inline w-3 h-3 ml-0.5" />
                </th>
                <th className="py-2.5 px-3 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-900 font-mono">
              {sortedTableRows.map((row) => {
                const isSelected = selectedStrategyId === row.strategyId;
                const isUserPlan = row.strategyId === 'user_plan';
                const isConsensus = row.strategyId === 'consensus';
                const p50 = metricViewMode === 'bootstrap' ? row.bootstrapP50 : row.parametricP50;
                const p10 = metricViewMode === 'bootstrap' ? row.bootstrapP10 : row.parametricP10;
                const p90 = metricViewMode === 'bootstrap' ? row.bootstrapP90 : row.parametricP90;

                return (
                  <tr
                    key={row.strategyId}
                    onClick={() => setSelectedStrategyId(row.strategyId)}
                    className={`transition-colors cursor-pointer ${
                      isUserPlan
                        ? 'bg-emerald-950/30 border-l-2 border-emerald-400 font-semibold'
                        : isConsensus
                        ? 'bg-purple-950/20 border-l-2 border-purple-400'
                        : isSelected
                        ? 'bg-neutral-800'
                        : 'hover:bg-neutral-800/40'
                    }`}
                  >
                    <td className="py-2.5 px-3 font-sans">
                      <div className="flex items-center gap-2">
                        {isUserPlan ? (
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                            YOUR PLAN
                          </span>
                        ) : isConsensus ? (
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-purple-500/20 text-purple-300 border border-purple-500/40">
                            TOP 3 BLEND
                          </span>
                        ) : (
                          <span className="w-5 text-neutral-500 text-[10px]">#{row.overallRank}</span>
                        )}
                        <span className="text-white font-medium">{row.strategyName}</span>
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span className={row.inSampleReturn === (bestColumnValues as any).inSampleReturn ? 'text-emerald-400 font-bold' : 'text-neutral-300'}>
                        {(row.inSampleReturn * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span className={row.inSampleVol === (bestColumnValues as any).inSampleVol ? 'text-emerald-400 font-bold' : 'text-neutral-300'}>
                        {(row.inSampleVol * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span className={row.oosSharpe === (bestColumnValues as any).oosSharpe ? 'text-emerald-400 font-bold' : 'text-neutral-300'}>
                        {row.oosSharpe.toFixed(2)}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right text-rose-400">
                      -{(row.maxDrawdown * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold text-white">
                      {formatINR(p50)}
                    </td>
                    <td className="py-2.5 px-3 text-right text-[11px] text-neutral-400">
                      {formatINR(p10)} – {formatINR(p90)}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span className={row.probLoss <= 5 ? 'text-emerald-400' : 'text-amber-400'}>
                        {row.probLoss.toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      {!isUserPlan && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleApplyStrategy(row.weights, row.strategyId);
                          }}
                          className={`px-2.5 py-1 rounded text-[11px] font-sans font-medium transition cursor-pointer ${
                            appliedFeedbackId === row.strategyId
                              ? 'bg-emerald-900/80 text-emerald-300 border border-emerald-600'
                              : 'bg-neutral-800 hover:bg-neutral-700 text-neutral-200'
                          }`}
                        >
                          {appliedFeedbackId === row.strategyId ? 'Applied!' : 'Apply'}
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* 5. Two-Column Visual Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-7 bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 sm:p-5 space-y-3">
          <div className="flex items-center justify-between border-b border-neutral-800 pb-2">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-200">
                P10–P90 Growth: Your Plan vs {selectedStrategy?.strategyName}
              </h3>
              <span className="text-[11px] text-neutral-500">
                Median (P50) lines with shaded 80% confidence corridor
              </span>
            </div>
          </div>
          <div className="h-64 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={fanChartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="selectedBand" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0.03} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
                <XAxis dataKey="year" stroke="#737373" tickFormatter={(v) => `Y${v}`} fontSize={11} />
                <YAxis
                  stroke="#737373"
                  fontSize={11}
                  tickFormatter={(val) => formatINR(val)}
                  domain={['auto', 'auto']}
                />
                <Tooltip
                  formatter={(val: any) => [formatINR(Number(val)), '']}
                  contentStyle={{
                    backgroundColor: '#171717',
                    borderColor: '#262626',
                    borderRadius: '8px',
                    fontSize: '11px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Area
                  type="monotone"
                  dataKey="selectedP90"
                  stroke="#10b981"
                  strokeWidth={1}
                  fill="url(#selectedBand)"
                  name={`${selectedStrategy?.strategyName} P90`}
                />
                <Area
                  type="monotone"
                  dataKey="selectedP10"
                  stroke="#10b981"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  fill="transparent"
                  name={`${selectedStrategy?.strategyName} P10`}
                />
                <Line
                  type="monotone"
                  dataKey="selectedP50"
                  stroke="#10b981"
                  strokeWidth={2.5}
                  dot={false}
                  name={`${selectedStrategy?.strategyName} Median (P50)`}
                />
                {fanChartData[0]?.userP50 !== undefined && (
                  <Line
                    type="monotone"
                    dataKey="userP50"
                    stroke="#38bdf8"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                    name="Your Plan Median"
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="lg:col-span-5 bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 sm:p-5 space-y-3">
          <div className="border-b border-neutral-800 pb-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-200">
              Asset Allocation Comparison
            </h3>
            <span className="text-[11px] text-neutral-500">Your Plan vs Top 3 Quantitative Portfolios</span>
          </div>
          <div className="h-64 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={allocationComparisonData}
                layout="vertical"
                margin={{ top: 5, right: 10, left: 10, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} unit="%" stroke="#737373" fontSize={10} />
                <YAxis dataKey="planName" type="category" stroke="#737373" fontSize={11} width={85} />
                <Tooltip
                  formatter={(val: any, name: any) => [`${val}%`, ASSET_MAP[name]?.name || name]}
                  contentStyle={{
                    backgroundColor: '#171717',
                    borderColor: '#262626',
                    borderRadius: '8px',
                    fontSize: '11px',
                  }}
                />
                {ALL_ASSET_DEFINITIONS.map((a) => (
                  <Bar
                    key={a.id}
                    dataKey={a.id}
                    stackId="a"
                    fill={getAssetColor(a.id)}
                    name={a.name}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      {/* 6. Out-of-Sample Walk-Forward Equity Curves */}
      <section className="bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 sm:p-5 space-y-3">
        <div className="border-b border-neutral-800 pb-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-200">
            Out-of-Sample Walk-Forward Equity Curves (Base = 100)
          </h3>
          <span className="text-[11px] text-neutral-500">
            Strictly out-of-sample growth incorporating real 0.2% rebalance turnover friction
          </span>
        </div>
        <div className="h-60 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={equityCurvesData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
              <XAxis dataKey="month" stroke="#737373" fontSize={10} />
              <YAxis stroke="#737373" fontSize={10} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#171717',
                  borderColor: '#262626',
                  borderRadius: '8px',
                  fontSize: '11px',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Line type="monotone" dataKey="Your Plan" stroke="#38bdf8" strokeWidth={2.5} dot={false} />
              {scorecardResult?.items.slice(0, 3).map((strat, idx) => (
                <Line
                  key={strat.strategyId}
                  type="monotone"
                  dataKey={strat.strategyName}
                  stroke={idx === 0 ? '#10b981' : idx === 1 ? '#a855f7' : '#f59e0b'}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* 7. Goal Solver Panel */}
      <section className="bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 sm:p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-neutral-800 pb-2">
          <div className="flex items-center gap-2">
            <Target className="w-4 h-4 text-emerald-400" />
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-200">
                  Wealth Goal Solver &amp; Confidence Sizing
                </h3>
                {isGoalStale && (
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-500/20 text-amber-300 border border-amber-500/40">
                    stale, inputs changed
                  </span>
                )}
              </div>
              <span className="text-[11px] text-neutral-500">
                Evaluate target achievement probability for {selectedStrategy?.strategyName || 'selected strategy'}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 text-[11px] text-neutral-400 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={accurateMode}
                onChange={(e) => setAccurateMode(e.target.checked)}
                className="rounded bg-neutral-950 border-neutral-800 text-emerald-500 focus:ring-0 cursor-pointer"
              />
              <span>Accurate mode</span>
            </label>
            <button
              type="button"
              onClick={handleCalculateGoal}
              disabled={isSolvingGoal}
              className="px-3.5 py-1.5 bg-emerald-500 hover:bg-emerald-400 text-neutral-950 font-semibold rounded-lg text-xs flex items-center justify-center gap-2 transition cursor-pointer shadow-md shadow-emerald-500/20 active:scale-[0.99] disabled:opacity-50"
            >
              {isSolvingGoal ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Calculating goal...</span>
                </>
              ) : (
                <>
                  <Target className="w-3.5 h-3.5" />
                  <span>Calculate goal</span>
                </>
              )}
            </button>
          </div>
        </div>

        {goalError && (
          <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-500/30 flex items-center gap-2 text-rose-300 text-xs">
            <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{goalError}</span>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-center">
          <div className="sm:col-span-4 space-y-1.5">
            <label htmlFor="goal-target-corpus" className="text-xs text-neutral-400">Target Wealth Corpus (₹)</label>
            <TargetCorpusInput initialValue={targetCorpus} onCommit={setTargetCorpus} />
            <span className="text-[11px] text-emerald-400 font-mono">{formatINR(targetCorpus)}</span>
          </div>
          <div className="sm:col-span-8 grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="p-3 rounded-lg bg-neutral-950 border border-neutral-800 text-center space-y-1">
              <span className="text-[10px] text-neutral-500 block uppercase">MC Probability</span>
              <span className="text-lg font-mono font-bold text-emerald-400">
                {goalResult?.probReachingMC ?? 0}%
              </span>
              <span className="text-[10px] text-neutral-500 block">Parametric Likelihood</span>
            </div>
            <div className="p-3 rounded-lg bg-neutral-950 border border-neutral-800 text-center space-y-1">
              <span className="text-[10px] text-neutral-500 block uppercase">Bootstrap Prob</span>
              <span className="text-lg font-mono font-bold text-teal-400">
                {goalResult?.probReachingBootstrap ?? 0}%
              </span>
              <span className="text-[10px] text-neutral-500 block">Fat-Tailed Likelihood</span>
            </div>
            <div className="p-3 rounded-lg bg-neutral-950 border border-neutral-800 text-center space-y-1">
              <span className="text-[10px] text-neutral-500 block uppercase">Required for 80%</span>
              <span className="text-lg font-mono font-bold text-amber-400">
                {formatINR(goalResult?.minSip80 ?? 0)}/mo
              </span>
              <span className="text-[10px] text-neutral-500 block">Minimum SIP</span>
            </div>
          </div>
        </div>

        {goalResult && (
          <div className="p-3 rounded-lg bg-neutral-950/70 border border-neutral-800 flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
            <span className="text-neutral-400 font-sans">
              Minimum SIP needed for {formatINR(targetCorpus)} in {horizonYears} years:
            </span>
            <div className="flex items-center gap-3">
              <span className="text-neutral-300">
                70% Confidence: <strong className="text-emerald-400">{formatINR(goalResult.minSip70)}/mo</strong>
              </span>
              <span className="text-neutral-300">
                80% Confidence: <strong className="text-amber-400">{formatINR(goalResult.minSip80)}/mo</strong>
              </span>
              <span className="text-neutral-300">
                90% Confidence: <strong className="text-purple-400">{formatINR(goalResult.minSip90)}/mo</strong>
              </span>
            </div>
          </div>
        )}
      </section>
    </div>
  );
});

StrategyLabTab.displayName = 'StrategyLabTab';
