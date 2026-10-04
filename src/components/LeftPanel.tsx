/**
 * Left Panel: Inputs, Rupee Allocation Table, 1-Year Allocation Card, Bond Config, and Return Haircut Assumptions
 * Primary input in Rupees (₹/month), category collapsible groups, and live counters.
 */
import React, { useState, useMemo, useEffect, useRef } from 'react';
import {
  ALL_ASSET_DEFINITIONS,
  ASSET_MAP,
  BondInput,
  PayoutFrequency,
  projectSIP,
  runMonteCarlo,
  buildCovariance,
  isSimulationCancelled,
} from '../engine/index.ts';
import { formatINR, getAssetColor, formatPercent } from '../utils/formatters.ts';
import {
  ASSET_GROUPS,
  rupeeAllocationToWeights,
  weightsToRupeeAllocation,
  computeGroupSubtotals,
} from '../utils/allocation.ts';
import {
  ChevronDown,
  ChevronUp,
  AlertCircle,
  CheckCircle2,
  Sliders,
  DollarSign,
  ShieldAlert,
  RotateCcw,
  Sparkles,
  PieChart as PieIcon,
  ChevronRight,
  TrendingUp,
} from 'lucide-react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';

export type SimulationMode = 'lumpsum' | 'sip' | 'both';

export interface AssetAssumption {
  cagr: number; // in decimal, e.g. 0.14 for 14%
  volatility: number; // in decimal, e.g. 0.20 for 20%
}

interface LeftPanelProps {
  mode: SimulationMode;
  onModeChange: (mode: SimulationMode) => void;
  lumpsumAmount: number;
  onLumpsumAmountChange: (amount: number) => void;
  sipAmount: number;
  onSipAmountChange: (amount: number) => void;
  years: number;
  onYearsChange: (years: number) => void;
  inflationRate: number; // in percent, e.g. 6 for 6%
  onInflationRateChange: (rate: number) => void;
  allocationInr: Record<string, number>; // assetId -> ₹ per month
  onAllocationInrChange: (allocation: Record<string, number>) => void;
  bondConfig: BondInput;
  onBondConfigChange: (config: BondInput) => void;
  assumptions: Record<string, AssetAssumption>;
  onAssumptionsChange: (assumptions: Record<string, AssetAssumption>) => void;
  onResetAssumptions: () => void;
  dataSourceLabel: string;
  assetFallbackStatus?: Record<string, boolean>;
  returnHaircutS: number;
  onReturnHaircutSChange: (s: number) => void;
  categoryPriors: Record<string, number>;
  onCategoryPriorsChange: (priors: Record<string, number>) => void;
  rawHistoricalCagrs: Record<string, number>;
  fatTails: boolean;
  onFatTailsChange: (fatTails: boolean) => void;
  covarianceMatrix?: number[][];
  assetIds?: string[];
}

export const LeftPanel: React.FC<LeftPanelProps> = ({
  mode,
  onModeChange,
  lumpsumAmount,
  onLumpsumAmountChange,
  sipAmount,
  onSipAmountChange,
  years,
  onYearsChange,
  inflationRate,
  onInflationRateChange,
  allocationInr,
  onAllocationInrChange,
  bondConfig,
  onBondConfigChange,
  assumptions,
  onAssumptionsChange,
  onResetAssumptions,
  dataSourceLabel,
  assetFallbackStatus = {},
  returnHaircutS,
  onReturnHaircutSChange,
  categoryPriors,
  onCategoryPriorsChange,
  rawHistoricalCagrs,
  fatTails,
  onFatTailsChange,
  covarianceMatrix,
  assetIds,
}) => {
  const [showOneTimeAmount, setShowOneTimeAmount] = useState(mode === 'lumpsum' || mode === 'both');
  const [showBondDetails, setShowBondDetails] = useState(false);
  const [showAssumptions, setShowAssumptions] = useState(false);
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({
    sectors: true,
  });

  const [fillTargetAsset, setFillTargetAsset] = useState<string>('nifty_50');
  const [cardHorizonYears, setCardHorizonYears] = useState<number>(1);

  const totalAllocatedInr = useMemo(() => {
    return Object.values(allocationInr).reduce((sum, v) => sum + (v > 0 ? v : 0), 0);
  }, [allocationInr]);

  const unallocatedInr = sipAmount - totalAllocatedInr;
  const isAllocatedExact = Math.abs(unallocatedInr) <= 1;
  const isOverAllocated = unallocatedInr < -1;

  const toggleGroup = (groupId: string) => {
    setCollapsedGroups((prev) => ({
      ...prev,
      [groupId]: !prev[groupId],
    }));
  };

  const groupSubtotals = useMemo(() => {
    return computeGroupSubtotals(allocationInr);
  }, [allocationInr]);

  const handleAssetAmountChange = (assetId: string, value: number) => {
    const cleanVal = isNaN(value) || value < 0 ? 0 : Math.round(value);
    onAllocationInrChange({
      ...allocationInr,
      [assetId]: cleanVal,
    });
  };

  const handleDistributeEqually = () => {
    const assetsCount = ALL_ASSET_DEFINITIONS.length;
    const perAsset = Math.floor(sipAmount / assetsCount / 100) * 100;
    const newAlloc: Record<string, number> = {};
    let allocated = 0;
    for (let i = 0; i < assetsCount; i++) {
      const id = ALL_ASSET_DEFINITIONS[i].id;
      newAlloc[id] = perAsset;
      allocated += perAsset;
    }
    newAlloc['nifty_50'] += sipAmount - allocated;
    onAllocationInrChange(newAlloc);
  };

  const handleClearAll = () => {
    const newAlloc: Record<string, number> = {};
    for (const a of ALL_ASSET_DEFINITIONS) newAlloc[a.id] = 0;
    onAllocationInrChange(newAlloc);
  };

  const handleFillRemaining = () => {
    if (unallocatedInr <= 0) return;
    const current = allocationInr[fillTargetAsset] || 0;
    onAllocationInrChange({
      ...allocationInr,
      [fillTargetAsset]: current + unallocatedInr,
    });
  };

  const horizonCardStats = useMemo(() => {
    const weights = rupeeAllocationToWeights(allocationInr);
    const months = cardHorizonYears * 12;
    const totalPrincipal = sipAmount * months;

    let projVal = totalPrincipal;
    try {
      const proj = projectSIP(sipAmount, cardHorizonYears, weights, assumptions, bondConfig);
      projVal = proj.finalValue;
    } catch {
      projVal = totalPrincipal;
    }

    const donutData = ASSET_GROUPS.map((g) => ({
      name: g.name,
      value: (groupSubtotals[g.id]?.totalInr || 0) * cardHorizonYears,
      color: getAssetColor(g.assetIds[0]),
    })).filter((d) => d.value > 0);

    return {
      totalPrincipal,
      projectedValue: projVal,
      donutData,
    };
  }, [allocationInr, sipAmount, cardHorizonYears, assumptions, bondConfig, groupSubtotals]);

  const [percentiles, setPercentiles] = useState<{ p10: number; p50: number; p90: number } | null>(null);
  const [isPercentilesLoading, setIsPercentilesLoading] = useState<boolean>(false);
  const percentilesReqIdRef = useRef<number>(0);

  useEffect(() => {
    const reqId = ++percentilesReqIdRef.current;
    setIsPercentilesLoading(true);

    const timer = setTimeout(async () => {
      const DEBUG_PERF = false;
      if (DEBUG_PERF) performance.mark('leftpanel-pctl-start');

      try {
        const weights = rupeeAllocationToWeights(allocationInr);
        const panelAssetIds = assetIds && assetIds.length > 0 ? assetIds : Object.keys(weights);
        const panelCov =
          covarianceMatrix && covarianceMatrix.length === panelAssetIds.length
            ? covarianceMatrix
            : buildCovariance(panelAssetIds, assumptions, []);

        const mc = await runMonteCarlo(
          {
            allocation: weights,
            years: cardHorizonYears,
            mode: 'sip',
            monthlySip: sipAmount,
            lumpsumAmount: 0,
            assetStats: assumptions,
            covarianceMatrix: panelCov,
            assetIds: panelAssetIds,
            bondConfig,
            nSims: 300,
            inflationRate: inflationRate / 100,
            seed: 42,
          },
          'leftpanel'
        );

        if (percentilesReqIdRef.current !== reqId) return;

        const finalY = mc.percentilesByYear[cardHorizonYears - 1];
        if (finalY) {
          setPercentiles({
            p10: finalY.p10,
            p50: finalY.p50,
            p90: finalY.p90,
          });
        }
      } catch (err: any) {
        if (isSimulationCancelled(err)) return;
      } finally {
        if (percentilesReqIdRef.current === reqId) {
          setIsPercentilesLoading(false);
        }
        if (DEBUG_PERF) {
          performance.mark('leftpanel-pctl-end');
          performance.measure('leftpanel-pctl', 'leftpanel-pctl-start', 'leftpanel-pctl-end');
          const entries = performance.getEntriesByName('leftpanel-pctl');
          if (entries.length > 0) {
            console.log(`[PERF] LeftPanel percentiles: ${entries[entries.length - 1].duration.toFixed(2)}ms`);
          }
        }
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [
    allocationInr,
    sipAmount,
    cardHorizonYears,
    assumptions,
    bondConfig,
    inflationRate,
    covarianceMatrix,
    assetIds,
  ]);

  const displayP10 = percentiles?.p10 ?? horizonCardStats.totalPrincipal;
  const displayP50 = percentiles?.p50 ?? horizonCardStats.projectedValue;
  const displayP90 = percentiles?.p90 ?? horizonCardStats.projectedValue;

  const hasBondHolding = (allocationInr['bonds_fd'] || 0) > 0;

  return (
    <div className="space-y-5">
      {/* 1. Monthly Investment Amount & Capital Controls */}
      <section className="bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 sm:p-5 space-y-4 shadow-sm">
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label htmlFor="primary-monthly-sip" className="text-xs font-semibold text-neutral-200">
              Monthly investment amount (₹)
            </label>
            <span className="text-xs font-mono font-bold text-emerald-400 bg-neutral-950 px-2.5 py-0.5 rounded border border-neutral-800">
              {formatINR(sipAmount)} / month
            </span>
          </div>
          <div className="relative">
            <span className="absolute left-3 top-2.5 text-neutral-500 text-sm font-semibold">₹</span>
            <input
              id="primary-monthly-sip"
              type="number"
              min={0}
              step={1000}
              value={sipAmount || ''}
              onChange={(e) => {
                const val = Math.max(0, Number(e.target.value));
                onSipAmountChange(val);
                if (mode === 'lumpsum') onModeChange('both');
              }}
              className="w-full bg-neutral-950 border border-neutral-800 rounded-lg pl-8 pr-3 py-2 text-sm text-neutral-100 font-mono focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
              placeholder="e.g. 25000"
            />
          </div>
          <div className="flex flex-wrap gap-1.5 pt-1">
            {[10000, 25000, 50000, 100000, 250000].map((preset) => (
              <button
                key={preset}
                type="button"
                onClick={() => {
                  onSipAmountChange(preset);
                  if (totalAllocatedInr > 0) {
                    const weights = rupeeAllocationToWeights(allocationInr);
                    onAllocationInrChange(weightsToRupeeAllocation(weights, preset));
                  }
                }}
                className={`px-2 py-0.5 rounded text-[11px] font-mono transition cursor-pointer border ${
                  sipAmount === preset
                    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                    : 'bg-neutral-950 text-neutral-400 hover:text-neutral-200 border-neutral-800'
                }`}
              >
                {formatINR(preset)}
              </button>
            ))}
          </div>
        </div>

        <div className="pt-2 border-t border-neutral-800/80">
          <button
            type="button"
            onClick={() => {
              const nextState = !showOneTimeAmount;
              setShowOneTimeAmount(nextState);
              if (nextState) {
                if (mode === 'sip') onModeChange('both');
              } else {
                onModeChange('sip');
              }
            }}
            className="flex items-center justify-between w-full text-xs text-neutral-400 hover:text-neutral-200 transition cursor-pointer"
          >
            <div className="flex items-center gap-1.5">
              <span className="font-medium">One-time amount (optional initial lumpsum)</span>
              {lumpsumAmount > 0 && showOneTimeAmount && (
                <span className="text-[10px] font-mono text-emerald-400">({formatINR(lumpsumAmount)})</span>
              )}
            </div>
            {showOneTimeAmount ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
          {showOneTimeAmount && (
            <div className="mt-2.5 p-3 rounded-lg bg-neutral-950/80 border border-neutral-800 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-neutral-400">Initial Lumpsum Capital</span>
                <span className="font-mono text-neutral-200 font-semibold">{formatINR(lumpsumAmount)}</span>
              </div>
              <div className="relative">
                <span className="absolute left-3 top-2 text-neutral-500 text-xs">₹</span>
                <input
                  type="number"
                  min={0}
                  step={25000}
                  value={lumpsumAmount || ''}
                  onChange={(e) => onLumpsumAmountChange(Math.max(0, Number(e.target.value)))}
                  className="w-full bg-neutral-900 border border-neutral-800 rounded px-2.5 pl-7 py-1.5 text-xs text-neutral-200 font-mono focus:outline-none focus:border-emerald-500"
                  placeholder="0"
                />
              </div>
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 gap-3 pt-1 border-t border-neutral-800/80">
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <label htmlFor="simulator-horizon-input" className="text-neutral-400">Horizon</label>
              <span className="font-mono text-emerald-400 text-xs">{years} yrs</span>
            </div>
            <input
              id="simulator-horizon-input"
              type="number"
              min={1}
              max={30}
              value={years}
              onChange={(e) => onYearsChange(Math.min(30, Math.max(1, Number(e.target.value))))}
              className="w-full bg-neutral-950 border border-neutral-800 rounded px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <label htmlFor="simulator-inflation-input" className="text-neutral-400">Inflation</label>
              <span className="font-mono text-amber-400 text-xs">{inflationRate}%</span>
            </div>
            <input
              id="simulator-inflation-input"
              type="number"
              min={0}
              max={15}
              step={0.5}
              value={inflationRate}
              onChange={(e) => onInflationRateChange(Math.max(0, Number(e.target.value)))}
              className="w-full bg-neutral-950 border border-neutral-800 rounded px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-emerald-500"
            />
          </div>
        </div>
      </section>

      {/* 2. Grouped Rupee Allocation Table */}
      <section className="bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 sm:p-5 space-y-4">
        <div className="space-y-2.5 pb-2 border-b border-neutral-800">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-200">
                Monthly Allocation (₹)
              </h2>
              <span className="text-[11px] text-neutral-500">
                Allocate your ₹{formatINR(sipAmount, false)} monthly budget
              </span>
            </div>
            <div
              className={`px-2.5 py-1 rounded-full text-xs font-mono font-semibold flex items-center gap-1.5 border ${
                isAllocatedExact
                  ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                  : isOverAllocated
                  ? 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                  : 'bg-amber-500/15 text-amber-400 border-amber-500/30'
              }`}
            >
              {isAllocatedExact ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>100% Balanced</span>
                </>
              ) : isOverAllocated ? (
                <>
                  <AlertCircle className="w-3.5 h-3.5" />
                  <span>Over by {formatINR(Math.abs(unallocatedInr))}</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-3.5 h-3.5" />
                  <span>{formatINR(unallocatedInr)} Left</span>
                </>
              )}
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-neutral-400">
                Allocated: <strong className="text-white">{formatINR(totalAllocatedInr)}</strong> of {formatINR(sipAmount)}
              </span>
              <span
                className={
                  isAllocatedExact
                    ? 'text-emerald-400 font-semibold'
                    : isOverAllocated
                    ? 'text-rose-400 font-semibold'
                    : 'text-amber-400 font-semibold'
                }
              >
                Unallocated: {formatINR(unallocatedInr)}
              </span>
            </div>
            <div className="h-1.5 w-full bg-neutral-950 rounded-full overflow-hidden flex border border-neutral-800">
              <div
                style={{
                  width: `${Math.min(100, sipAmount > 0 ? (totalAllocatedInr / sipAmount) * 100 : 0)}%`,
                }}
                className={`h-full transition-all duration-300 ${
                  isAllocatedExact ? 'bg-emerald-500' : isOverAllocated ? 'bg-rose-500' : 'bg-amber-500'
                }`}
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-xs">
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={handleDistributeEqually}
                className="px-2 py-1 rounded bg-neutral-950 hover:bg-neutral-800 text-neutral-300 border border-neutral-800 text-[11px] transition cursor-pointer"
              >
                Distribute equally
              </button>
              <button
                type="button"
                onClick={handleClearAll}
                className="px-2 py-1 rounded bg-neutral-950 hover:bg-neutral-800 text-neutral-400 hover:text-neutral-200 border border-neutral-800 text-[11px] transition cursor-pointer"
              >
                Clear all
              </button>
            </div>
            {unallocatedInr > 0 && (
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={handleFillRemaining}
                  className="px-2 py-1 rounded bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 text-[11px] font-medium transition cursor-pointer"
                >
                  Fill +{formatINR(unallocatedInr)} into
                </button>
                <select
                  value={fillTargetAsset}
                  onChange={(e) => setFillTargetAsset(e.target.value)}
                  className="bg-neutral-950 border border-neutral-800 rounded px-1.5 py-1 text-[11px] text-neutral-200 focus:outline-none"
                >
                  {ALL_ASSET_DEFINITIONS.map((a) => (
                    <option key={a.id} value={a.id}>
                      {a.name}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-4">
          {ASSET_GROUPS.map((group) => {
            const isCollapsed = !!collapsedGroups[group.id];
            const subtotal = groupSubtotals[group.id] || { totalInr: 0, percentage: 0 };
            return (
              <div
                key={group.id}
                className="rounded-lg border border-neutral-800/80 bg-neutral-950/40 overflow-hidden"
              >
                <button
                  type="button"
                  onClick={() => toggleGroup(group.id)}
                  className="w-full px-3.5 py-2.5 bg-neutral-900/70 hover:bg-neutral-900 flex items-center justify-between transition cursor-pointer text-left"
                >
                  <div className="flex items-center gap-2">
                    {isCollapsed ? (
                      <ChevronRight className="w-3.5 h-3.5 text-neutral-400" />
                    ) : (
                      <ChevronDown className="w-3.5 h-3.5 text-neutral-400" />
                    )}
                    <span className="text-xs font-semibold text-neutral-200">{group.name}</span>
                    <span className="text-[10px] text-neutral-500 hidden sm:inline">({group.assetIds.length})</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs font-mono">
                    <span className="text-neutral-400 font-semibold">{formatINR(subtotal.totalInr)}</span>
                    <span className="text-[10px] text-neutral-500 bg-neutral-900 px-1.5 py-0.5 rounded border border-neutral-800">
                      {subtotal.percentage}%
                    </span>
                  </div>
                </button>
                {!isCollapsed && (
                  <div className="p-2 sm:p-3 space-y-2.5 divide-y divide-neutral-900">
                    {group.assetIds.map((assetId) => {
                      const asset = ASSET_MAP[assetId];
                      if (!asset) return null;
                      const amount = allocationInr[assetId] || 0;
                      const isZero = amount === 0;
                      const pctOfMonthly = sipAmount > 0 ? (amount / sipAmount) * 100 : 0;
                      const assetColor = getAssetColor(assetId);
                      return (
                        <div
                          key={assetId}
                          className={`pt-2 first:pt-0 transition-opacity ${
                            isZero ? 'opacity-60 hover:opacity-100' : 'opacity-100'
                          }`}
                        >
                          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1.5">
                            <div className="flex items-center gap-2">
                              <span
                                className="w-2.5 h-2.5 rounded-full shrink-0"
                                style={{ backgroundColor: assetColor }}
                              />
                              <div>
                                <span className="text-xs font-medium text-neutral-200 block">
                                  {assetId === 'nifty_50'
                                    ? 'Direct Stocks (Nifty 50 index proxy, not individual stock picks)'
                                    : asset.name}
                                </span>
                                <span className="text-[10px] text-neutral-500 font-mono">
                                  {asset.tickerOrCode} • {(assumptions[assetId]?.cagr * 100 || 12).toFixed(1)}% exp. return
                                </span>
                              </div>
                            </div>
                            <div className="flex items-center gap-2 shrink-0">
                              <span className="text-[11px] font-mono text-neutral-400 bg-neutral-900 px-2 py-1 rounded border border-neutral-800 min-w-[50px] text-right">
                                {pctOfMonthly.toFixed(1)}%
                              </span>
                              <div className="relative w-28">
                                <span className="absolute left-2.5 top-1.5 text-neutral-500 text-xs">₹</span>
                                <input
                                  type="number"
                                  min={0}
                                  step={500}
                                  value={amount || ''}
                                  onChange={(e) => handleAssetAmountChange(assetId, Number(e.target.value))}
                                  className="w-full bg-neutral-900 border border-neutral-800 rounded px-2 pl-6 py-1 text-xs font-mono text-neutral-100 text-right focus:outline-none focus:border-emerald-500"
                                  placeholder="0"
                                />
                              </div>
                            </div>
                          </div>
                          <div className="pt-1.5 flex items-center gap-2">
                            <input
                              type="range"
                              min={0}
                              max={sipAmount || 50000}
                              step={500}
                              value={amount}
                              onChange={(e) => handleAssetAmountChange(assetId, Number(e.target.value))}
                              className="w-full h-1 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* 3. 1-Year Allocation & Horizon Projection Card */}
      <section className="bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 sm:p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
          <div className="flex items-center gap-2">
            <PieIcon className="w-4 h-4 text-emerald-400" />
            <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-200">
              {cardHorizonYears}-Year Allocation Breakdown
            </h2>
          </div>
          <div className="inline-flex rounded-lg p-0.5 bg-neutral-950 border border-neutral-800 text-[11px] font-mono">
            {[1, 3, 5].map((yr) => (
              <button
                key={yr}
                type="button"
                onClick={() => setCardHorizonYears(yr)}
                className={`px-2.5 py-0.5 rounded font-medium transition cursor-pointer ${
                  cardHorizonYears === yr
                    ? 'bg-neutral-800 text-emerald-400 shadow-sm'
                    : 'text-neutral-400 hover:text-neutral-200'
                }`}
              >
                {yr}Y
              </button>
            ))}
          </div>
        </div>

        <div className="p-3 rounded-lg bg-neutral-950 border border-neutral-800 space-y-1">
          <div className="text-xs text-neutral-400">
            Invested after {cardHorizonYears} year{cardHorizonYears > 1 ? 's' : ''}:{' '}
            <strong className="text-white font-mono">{formatINR(horizonCardStats.totalPrincipal)}</strong>{' '}
            <span className="text-[10px] text-neutral-500">(principal only, not including returns)</span>
          </div>
        </div>

        {horizonCardStats.donutData.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 items-center pt-1">
            <div className="sm:col-span-5 h-36">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={horizonCardStats.donutData}
                    cx="50%"
                    cy="50%"
                    innerRadius={36}
                    outerRadius={56}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {horizonCardStats.donutData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(val: any) => [formatINR(Number(val)), 'Period Total']}
                    contentStyle={{
                      backgroundColor: '#171717',
                      borderColor: '#262626',
                      borderRadius: '8px',
                      fontSize: '11px',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="sm:col-span-7 space-y-1.5 text-xs">
              {horizonCardStats.donutData.map((d) => (
                <div key={d.name} className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: d.color }} />
                    <span className="text-neutral-300">{d.name}</span>
                  </div>
                  <span className="font-mono text-neutral-400">{formatINR(d.value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="rounded-lg border border-neutral-800 overflow-hidden text-xs">
          <div className="bg-neutral-950 px-3 py-1.5 grid grid-cols-12 font-medium text-[10px] text-neutral-500 uppercase tracking-wider">
            <span className="col-span-6">Asset</span>
            <span className="col-span-2 text-right">Monthly</span>
            <span className="col-span-2 text-right">{cardHorizonYears}Y Total</span>
            <span className="col-span-2 text-right">Share</span>
          </div>
          <div className="divide-y divide-neutral-900 max-h-44 overflow-y-auto">
            {ALL_ASSET_DEFINITIONS.filter((a) => (allocationInr[a.id] || 0) > 0).map((a) => {
              const mInr = allocationInr[a.id] || 0;
              const periodInr = mInr * cardHorizonYears * 12;
              const share = sipAmount > 0 ? (mInr / sipAmount) * 100 : 0;
              return (
                <div key={a.id} className="px-3 py-1.5 grid grid-cols-12 font-mono items-center">
                  <span className="col-span-6 font-sans text-neutral-300 truncate">{a.name}</span>
                  <span className="col-span-2 text-right text-neutral-400">{formatINR(mInr)}</span>
                  <span className="col-span-2 text-right text-neutral-200">{formatINR(periodInr)}</span>
                  <span className="col-span-2 text-right text-emerald-400">{share.toFixed(1)}%</span>
                </div>
              );
            })}
          </div>
          <div className="bg-neutral-950 px-3 py-2 grid grid-cols-12 font-mono font-bold text-xs border-t border-neutral-800">
            <span className="col-span-6 font-sans text-neutral-200">Grand Total</span>
            <span className="col-span-2 text-right text-neutral-200">{formatINR(totalAllocatedInr)}</span>
            <span className="col-span-2 text-right text-white">
              {formatINR(totalAllocatedInr * cardHorizonYears * 12)}
            </span>
            <span className="col-span-2 text-right text-emerald-400">100.0%</span>
          </div>
        </div>

        <div className="p-3 rounded-lg bg-neutral-950/80 border border-neutral-800 space-y-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-neutral-400 font-medium">
              Projected value after {cardHorizonYears} year{cardHorizonYears > 1 ? 's' : ''} (expected return):
            </span>
            <span className="font-mono text-emerald-400 font-bold">
              {formatINR(horizonCardStats.projectedValue)}
            </span>
          </div>
          <div className={`grid grid-cols-3 gap-2 pt-1 border-t border-neutral-800/80 text-[11px] font-mono text-center ${isPercentilesLoading ? 'opacity-70 transition-opacity' : 'transition-opacity'}`}>
            <div className="p-1.5 rounded bg-neutral-900 border border-neutral-800">
              <span className="text-[10px] text-neutral-500 block">Conservative (P10)</span>
              <span className="text-neutral-300 font-semibold">{formatINR(displayP10)}</span>
            </div>
            <div className="p-1.5 rounded bg-neutral-900 border border-neutral-800">
              <span className="text-[10px] text-neutral-500 block">Median (P50)</span>
              <span className="text-emerald-400 font-semibold">{formatINR(displayP50)}</span>
            </div>
            <div className="p-1.5 rounded bg-neutral-900 border border-neutral-800">
              <span className="text-[10px] text-neutral-500 block">Optimistic (P90)</span>
              <span className="text-teal-300 font-semibold">{formatINR(displayP90)}</span>
            </div>
          </div>
          <span className="text-[10px] text-neutral-500 block text-right italic">
            *Projections based on correlated Monte Carlo simulations and assumed returns.
          </span>
        </div>
      </section>

      {/* 4. Bonds Configuration */}
      {hasBondHolding && (
        <section className="bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 sm:p-5 space-y-3">
          <button
            type="button"
            onClick={() => setShowBondDetails(!showBondDetails)}
            className="flex items-center justify-between w-full text-xs font-semibold uppercase tracking-wider text-neutral-300 cursor-pointer"
          >
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-emerald-400" />
              <span>Bonds &amp; Fixed Income Parameters</span>
            </div>
            {showBondDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
          {showBondDetails && (
            <div className="pt-2 space-y-3 border-t border-neutral-800 text-xs">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label htmlFor="bond-coupon-input" className="text-neutral-400">Coupon Rate (%)</label>
                  <input
                    id="bond-coupon-input"
                    type="number"
                    step={0.1}
                    min={1}
                    max={20}
                    value={(bondConfig.couponRate * 100).toFixed(1)}
                    onChange={(e) =>
                      onBondConfigChange({
                        ...bondConfig,
                        couponRate: Number(e.target.value) / 100,
                      })
                    }
                    className="w-full bg-neutral-950 border border-neutral-800 rounded px-2.5 py-1.5 font-mono text-white"
                  />
                </div>
                <div className="space-y-1">
                  <label htmlFor="bond-tenure-input" className="text-neutral-400">Tenure (Years)</label>
                  <input
                    id="bond-tenure-input"
                    type="number"
                    min={1}
                    max={30}
                    value={bondConfig.tenureYears}
                    onChange={(e) =>
                      onBondConfigChange({
                        ...bondConfig,
                        tenureYears: Number(e.target.value),
                      })
                    }
                    className="w-full bg-neutral-950 border border-neutral-800 rounded px-2.5 py-1.5 font-mono text-white"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label htmlFor="bond-payout-freq" className="text-neutral-400">Payout Frequency</label>
                  <select
                    id="bond-payout-freq"
                    value={bondConfig.payoutFrequency}
                    onChange={(e) =>
                      onBondConfigChange({
                        ...bondConfig,
                        payoutFrequency: e.target.value as PayoutFrequency,
                      })
                    }
                    className="w-full bg-neutral-950 border border-neutral-800 rounded px-2.5 py-1.5 text-neutral-200"
                  >
                    <option value="annual">Annual</option>
                    <option value="semi-annual">Semi-Annual</option>
                    <option value="quarterly">Quarterly</option>
                    <option value="monthly">Monthly</option>
                    <option value="cumulative">Cumulative (At Maturity)</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <label htmlFor="bond-reinvest-rate" className="text-neutral-400">Reinvestment Rate (%)</label>
                  <input
                    id="bond-reinvest-rate"
                    type="number"
                    step={0.1}
                    min={0}
                    max={15}
                    value={((bondConfig.reinvestmentRateAfterMaturity ?? 0.065) * 100).toFixed(1)}
                    onChange={(e) =>
                      onBondConfigChange({
                        ...bondConfig,
                        reinvestmentRateAfterMaturity: Number(e.target.value) / 100,
                      })
                    }
                    className="w-full bg-neutral-950 border border-neutral-800 rounded px-2.5 py-1.5 font-mono text-white"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 pt-1 border-t border-neutral-800/80">
                <div className="space-y-1">
                  <label htmlFor="bond-default-prob" className="text-neutral-400">Default Risk (% / year)</label>
                  <input
                    id="bond-default-prob"
                    type="number"
                    step={0.1}
                    min={0}
                    max={10}
                    value={(bondConfig.creditRisk.defaultProbabilityAnnual * 100).toFixed(1)}
                    onChange={(e) =>
                      onBondConfigChange({
                        ...bondConfig,
                        creditRisk: {
                          ...bondConfig.creditRisk,
                          defaultProbabilityAnnual: Number(e.target.value) / 100,
                        },
                      })
                    }
                    className="w-full bg-neutral-950 border border-neutral-800 rounded px-2.5 py-1.5 font-mono text-white"
                  />
                </div>
                <div className="space-y-1">
                  <label htmlFor="bond-recovery-rate" className="text-neutral-400">Recovery Rate (%)</label>
                  <input
                    id="bond-recovery-rate"
                    type="number"
                    step={5}
                    min={0}
                    max={100}
                    value={(bondConfig.creditRisk.recoveryRate * 100).toFixed(0)}
                    onChange={(e) =>
                      onBondConfigChange({
                        ...bondConfig,
                        creditRisk: {
                          ...bondConfig.creditRisk,
                          recoveryRate: Number(e.target.value) / 100,
                        },
                      })
                    }
                    className="w-full bg-neutral-950 border border-neutral-800 rounded px-2.5 py-1.5 font-mono text-white"
                  />
                </div>
              </div>
            </div>
          )}
        </section>
      )}

      {/* 5. Return Haircut & Assumptions Panel */}
      <section className="bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 sm:p-5 space-y-3">
        <button
          type="button"
          onClick={() => setShowAssumptions(!showAssumptions)}
          className="flex items-center justify-between w-full text-xs font-semibold uppercase tracking-wider text-neutral-300 cursor-pointer"
        >
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-emerald-400" />
            <span>Return Expectations &amp; Assumptions</span>
          </div>
          {showAssumptions ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
        {showAssumptions && (
          <div className="pt-3 space-y-4 border-t border-neutral-800 text-xs">
            <div className="p-3 rounded-lg bg-neutral-950 border border-neutral-800 space-y-2">
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-semibold text-neutral-200 block">Return Expectation Haircut (s)</span>
                  <span className="text-[10px] text-neutral-500">
                    Blends historical 10Y bull CAGR with conservative category priors
                  </span>
                </div>
                <span className="font-mono font-bold text-emerald-400 text-xs">
                  {(returnHaircutS * 100).toFixed(0)}% Prior weight
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={returnHaircutS}
                onChange={(e) => onReturnHaircutSChange(Number(e.target.value))}
                className="w-full h-1.5 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <div className="flex justify-between text-[10px] text-neutral-500 font-mono">
                <span>0% (100% Historical)</span>
                <span>50% (Default Balanced)</span>
                <span>100% (100% Priors)</span>
              </div>
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg bg-neutral-950 border border-neutral-800">
              <div>
                <span className="font-semibold text-neutral-200 block">Stress-Aware Returns (Fat Tails)</span>
                <span className="text-[10px] text-neutral-500">
                  Draws shocks from Student-t(5 df) to model tail crashes
                </span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer shrink-0">
                <input
                  type="checkbox"
                  checked={fatTails}
                  onChange={(e) => onFatTailsChange(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-neutral-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-neutral-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-500"></div>
              </label>
            </div>

            <div className="rounded-lg border border-neutral-800 overflow-hidden">
              <div className="bg-neutral-950 px-3 py-1.5 grid grid-cols-12 font-medium text-[10px] text-neutral-500 uppercase tracking-wider">
                <span className="col-span-5">Asset</span>
                <span className="col-span-2 text-right">Raw 10Y</span>
                <span className="col-span-2 text-right">Prior</span>
                <span className="col-span-3 text-right">Model CAGR</span>
              </div>
              <div className="divide-y divide-neutral-900 max-h-48 overflow-y-auto font-mono text-xs">
                {ALL_ASSET_DEFINITIONS.filter((a) => a.id !== 'bonds_fd').map((a) => {
                  const raw = rawHistoricalCagrs[a.id] ?? 0.12;
                  const prior = categoryPriors[a.id] ?? 0.12;
                  const modelCagr = assumptions[a.id]?.cagr ?? raw;
                  return (
                    <div key={a.id} className="px-3 py-1.5 grid grid-cols-12 items-center">
                      <span className="col-span-5 font-sans text-neutral-300 truncate">{a.name}</span>
                      <span className="col-span-2 text-right text-neutral-500">{(raw * 100).toFixed(1)}%</span>
                      <span className="col-span-2 text-right text-neutral-400">{(prior * 100).toFixed(1)}%</span>
                      <span className="col-span-3 text-right text-emerald-400 font-semibold">
                        {(modelCagr * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="flex items-center justify-between pt-1">
              <span className="text-[10px] text-neutral-500 italic">{dataSourceLabel}</span>
              <button
                type="button"
                onClick={onResetAssumptions}
                className="flex items-center gap-1 text-[11px] text-neutral-400 hover:text-neutral-200 transition cursor-pointer"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Reset to defaults</span>
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
};
