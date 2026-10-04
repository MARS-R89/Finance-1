/**
 * Right Panel: Summary Cards, Fan Chart, Allocation Donut, Asset Gain Contributions,
 * Inflation Toggle, Plain-Language Line, and Persistent Disclaimer.
 */
import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  CartesianGrid,
} from 'recharts';
import {
  TrendingUp,
  Wallet,
  ArrowDownRight,
  ArrowUpRight,
  ShieldCheck,
  Percent,
  Info,
} from 'lucide-react';
import {
  MonteCarloResult,
  DeterministicProjectionResult,
  ASSET_MAP,
} from '../engine/index.ts';
import { formatINR, formatINRFull, getAssetColor, formatPercent } from '../utils/formatters.ts';

interface RightPanelProps {
  monteCarloResult: MonteCarloResult | null;
  projectionResult: {
    lumpsum?: DeterministicProjectionResult;
    sip?: DeterministicProjectionResult;
  } | null;
  mode: 'lumpsum' | 'sip' | 'both';
  allocation: Record<string, number>;
  years: number;
  inflationRate: number; // decimal e.g. 0.06
  isSimulating: boolean;
}

export const RightPanel: React.FC<RightPanelProps> = React.memo(({
  monteCarloResult,
  projectionResult,
  mode,
  allocation,
  years,
  inflationRate,
  isSimulating,
}) => {
  const [isInflationAdjusted, setIsInflationAdjusted] = useState(false);

  const terminalDiscount = isInflationAdjusted ? Math.pow(1 + inflationRate, years) : 1;

  const totalInvestedRaw =
    monteCarloResult?.totalInvested ??
    (projectionResult?.lumpsum?.totalInvested || 0) +
      (projectionResult?.sip?.totalInvested || 0);
  const totalInvestedDisplay = totalInvestedRaw / (isInflationAdjusted ? terminalDiscount : 1);

  const medianRaw = monteCarloResult?.terminalStats.median ?? totalInvestedRaw;
  const medianDisplay = medianRaw / terminalDiscount;

  const deterministicRaw =
    (projectionResult?.lumpsum?.finalValue || 0) +
    (projectionResult?.sip?.finalValue || 0);
  const deterministicDisplay =
    deterministicRaw > 0 ? deterministicRaw / terminalDiscount : medianDisplay;

  const p10Raw =
    monteCarloResult?.percentilesByYear[monteCarloResult.percentilesByYear.length - 1]?.p10 ??
    medianRaw * 0.7;
  const p10Display = p10Raw / terminalDiscount;

  const p90Raw =
    monteCarloResult?.percentilesByYear[monteCarloResult.percentilesByYear.length - 1]?.p90 ??
    medianRaw * 1.5;
  const p90Display = p90Raw / terminalDiscount;

  const probLoss = monteCarloResult?.probabilityOfLoss ?? 0;
  const probBeatInflation = monteCarloResult?.probabilityOfBeatingInflation ?? 85;

  const fanChartData = useMemo(() => {
    const data = (monteCarloResult?.percentilesByYear || []).map((pt) => {
      const yr = pt.year;
      const discount = isInflationAdjusted ? Math.pow(1 + inflationRate, yr) : 1;
      let investedAtYr = 0;
      if (mode === 'lumpsum') {
        investedAtYr = totalInvestedRaw;
      } else if (mode === 'sip') {
        investedAtYr = (totalInvestedRaw / years) * yr;
      } else {
        const lumpPart = projectionResult?.lumpsum?.totalInvested || 0;
        const sipPart = ((projectionResult?.sip?.totalInvested || 0) / years) * yr;
        investedAtYr = lumpPart + sipPart;
      }

      const p10Val = pt.p10 / discount;
      const p50Val = pt.p50 / discount;
      const p90Val = pt.p90 / discount;
      const bandWidth = Math.max(0, p90Val - p10Val);

      return {
        year: `Yr ${yr}`,
        rawYear: yr,
        p10: Math.round(p10Val),
        p50: Math.round(p50Val),
        p90: Math.round(p90Val),
        band: Math.round(bandWidth),
        invested: Math.round(investedAtYr / (isInflationAdjusted ? discount : 1)),
      };
    });

    const initialInvested =
      mode === 'lumpsum'
        ? totalInvestedRaw
        : mode === 'sip'
        ? 0
        : projectionResult?.lumpsum?.totalInvested || 0;

    if (data.length > 0) {
      data.unshift({
        year: 'Yr 0',
        rawYear: 0,
        p10: initialInvested,
        p50: initialInvested,
        p90: initialInvested,
        band: 0,
        invested: initialInvested,
      });
    }

    return data;
  }, [
    monteCarloResult?.percentilesByYear,
    isInflationAdjusted,
    inflationRate,
    mode,
    totalInvestedRaw,
    years,
    projectionResult?.lumpsum?.totalInvested,
    projectionResult?.sip?.totalInvested,
  ]);

  const activeAssets = useMemo(() => {
    const rawSum = Object.values(allocation).reduce((s, v) => s + (v > 0 ? v : 0), 0);
    const isDecimal = rawSum <= 1.5;
    return Object.entries(allocation)
      .filter(([_, weight]) => weight > 0)
      .map(([id, weight]) => ({
        id,
        name: ASSET_MAP[id]?.name || id,
        value: isDecimal ? weight * 100 : weight,
        color: getAssetColor(id),
      }));
  }, [allocation]);

  const contributionData = useMemo(() => {
    const lastSnapshot =
      projectionResult?.lumpsum?.yearByYear?.[years - 1]?.assetValues ||
      projectionResult?.sip?.yearByYear?.[years - 1]?.assetValues ||
      {};
    return activeAssets
      .map((asset) => {
        const endVal = lastSnapshot[asset.id] || (medianRaw * (asset.value / 100));
        const investedInAsset = totalInvestedRaw * (asset.value / 100);
        const gain = Math.max(0, endVal - investedInAsset) / terminalDiscount;
        return {
          name: asset.name,
          id: asset.id,
          gain: Math.round(gain),
          color: asset.color,
        };
      })
      .sort((a, b) => b.gain - a.gain)
      .slice(0, 8);
  }, [
    activeAssets,
    projectionResult?.lumpsum?.yearByYear,
    projectionResult?.sip?.yearByYear,
    years,
    medianRaw,
    totalInvestedRaw,
    terminalDiscount,
  ]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-neutral-900/90 border border-neutral-800 p-3.5 rounded-xl">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
            Valuation Perspective
          </span>
          <span className="text-[11px] text-neutral-500">
            (Inflation Hurdle: {(inflationRate * 100).toFixed(1)}% p.a.)
          </span>
        </div>
        <div className="inline-flex rounded-lg p-0.5 bg-neutral-950 border border-neutral-800 text-xs">
          <button
            onClick={() => setIsInflationAdjusted(false)}
            className={`px-3 py-1 rounded-md font-medium transition cursor-pointer ${
              !isInflationAdjusted
                ? 'bg-neutral-800 text-emerald-400 shadow-sm'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            Nominal Values
          </button>
          <button
            onClick={() => setIsInflationAdjusted(true)}
            className={`px-3 py-1 rounded-md font-medium transition cursor-pointer ${
              isInflationAdjusted
                ? 'bg-neutral-800 text-amber-400 shadow-sm'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            Inflation-Adjusted (Real ₹)
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
        <div className="bg-neutral-900/90 border border-neutral-800 p-3 rounded-xl relative overflow-hidden">
          <div className="flex items-center justify-between text-neutral-400 text-[11px] mb-1">
            <span className="truncate" title="Projection at assumed CAGR">Projection (CAGR)</span>
            <TrendingUp className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          </div>
          <div className="text-base sm:text-lg font-bold font-mono text-emerald-300">
            {formatINR(deterministicDisplay)}
          </div>
          <div className="text-[10px] text-neutral-500 font-mono mt-0.5 truncate">
            Deterministic path
          </div>
        </div>

        <div className="bg-neutral-900/90 border border-neutral-800 p-3 rounded-xl relative overflow-hidden">
          <div className="flex items-center justify-between text-neutral-400 text-[11px] mb-1">
            <span className="truncate" title="Monte Carlo Median">Median (P50)</span>
            <TrendingUp className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          </div>
          <div className="text-base sm:text-lg font-bold font-mono text-emerald-400">
            {formatINR(medianDisplay)}
          </div>
          <div className="text-[10px] text-neutral-500 font-mono mt-0.5 truncate">
            Monte Carlo median
          </div>
        </div>

        <div className="bg-neutral-900/90 border border-neutral-800 p-3 rounded-xl">
          <div className="flex items-center justify-between text-neutral-400 text-[11px] mb-1">
            <span className="truncate">Total Invested</span>
            <Wallet className="w-3.5 h-3.5 text-sky-400 shrink-0" />
          </div>
          <div className="text-base sm:text-lg font-bold font-mono text-white">
            {formatINR(totalInvestedDisplay)}
          </div>
          <div className="text-[10px] text-neutral-500 font-mono mt-0.5 truncate">
            Principal capital
          </div>
        </div>

        <div className="bg-neutral-900/90 border border-neutral-800 p-3 rounded-xl">
          <div className="flex items-center justify-between text-neutral-400 text-[11px] mb-1">
            <span className="truncate">P10 (Worst-Case)</span>
            <ArrowDownRight className="w-3.5 h-3.5 text-rose-400 shrink-0" />
          </div>
          <div className="text-base sm:text-lg font-bold font-mono text-rose-400">
            {formatINR(p10Display)}
          </div>
          <div className="text-[10px] text-neutral-500 font-mono mt-0.5 truncate">
            Bottom 10% market
          </div>
        </div>

        <div className="bg-neutral-900/90 border border-neutral-800 p-3 rounded-xl">
          <div className="flex items-center justify-between text-neutral-400 text-[11px] mb-1">
            <span className="truncate">P90 (Best-Case)</span>
            <ArrowUpRight className="w-3.5 h-3.5 text-teal-400 shrink-0" />
          </div>
          <div className="text-base sm:text-lg font-bold font-mono text-teal-300">
            {formatINR(p90Display)}
          </div>
          <div className="text-[10px] text-neutral-500 font-mono mt-0.5 truncate">
            Top 10% market
          </div>
        </div>

        <div className="col-span-2 sm:col-span-1 bg-neutral-900/90 border border-neutral-800 p-3 rounded-xl">
          <div className="flex items-center justify-between text-neutral-400 text-[11px] mb-1">
            <span className="truncate">Prob of Loss</span>
            <ShieldCheck className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          </div>
          <div className="text-base sm:text-lg font-bold font-mono text-amber-400">
            {probLoss.toFixed(1)}%
          </div>
          <div className="text-[10px] text-neutral-500 font-mono mt-0.5 truncate">
            {probLoss < 3 ? 'Minimal risk' : 'Capital risk'}
          </div>
        </div>
      </div>

      {monteCarloResult?.stressComparison && (
        <div className="p-3.5 rounded-xl bg-neutral-900/90 border border-neutral-800 space-y-2">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1.5">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-neutral-200">Stress-Aware vs Normal Comparison</span>
              <span
                className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${
                  monteCarloResult.stressComparison.isStressAware
                    ? 'bg-purple-500/20 text-purple-300 border-purple-500/40'
                    : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                }`}
              >
                Active: {monteCarloResult.stressComparison.isStressAware ? 'Stress-Aware (Student-t 5 df)' : 'Standard Normal'}
              </span>
            </div>
            <span className="text-[11px] text-neutral-500">Fat tails comparison (P5 &amp; Prob of Loss)</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
            <div
              className={`p-2.5 rounded-lg border text-xs ${
                !monteCarloResult.stressComparison.isStressAware
                  ? 'bg-neutral-950/80 border-emerald-500/40 ring-1 ring-emerald-500/20'
                  : 'bg-neutral-950/40 border-neutral-800'
              }`}
            >
              <div className="text-[11px] font-medium text-neutral-400 mb-1 flex items-center justify-between">
                <span>Standard Normal</span>
                {!monteCarloResult.stressComparison.isStressAware && (
                  <span className="text-[10px] text-emerald-400 font-mono font-bold">ACTIVE</span>
                )}
              </div>
              <div className="flex items-baseline justify-between font-mono">
                <span className="text-neutral-500 text-[11px]">P5 / P10 / P50:</span>
                <span className="text-neutral-200 font-semibold text-[11px]">
                  {formatINR(monteCarloResult.stressComparison.normalP5 / terminalDiscount)} /{' '}
                  {formatINR((monteCarloResult.stressComparison.normalP10 ?? 0) / terminalDiscount)} /{' '}
                  {formatINR((monteCarloResult.stressComparison.normalP50 ?? 0) / terminalDiscount)}
                </span>
              </div>
              <div className="flex items-baseline justify-between font-mono mt-0.5">
                <span className="text-neutral-500 text-[11px]">Probability of Loss:</span>
                <span className="text-amber-400 font-semibold">
                  {monteCarloResult.stressComparison.normalProbLoss.toFixed(1)}%
                </span>
              </div>
            </div>
            <div
              className={`p-2.5 rounded-lg border text-xs ${
                monteCarloResult.stressComparison.isStressAware
                  ? 'bg-neutral-950/80 border-purple-500/40 ring-1 ring-purple-500/20'
                  : 'bg-neutral-950/40 border-neutral-800'
              }`}
            >
              <div className="text-[11px] font-medium text-neutral-400 mb-1 flex items-center justify-between">
                <span>
                  {monteCarloResult.stressComparison.isBearRegimeStress
                    ? 'Bear-Regime Stress Test'
                    : 'Stress-Aware (Student-t 5 df)'}
                </span>
                {monteCarloResult.stressComparison.isStressAware && (
                  <span className="text-[10px] text-purple-400 font-mono font-bold">ACTIVE</span>
                )}
              </div>
              <div className="flex items-baseline justify-between font-mono">
                <span className="text-neutral-500 text-[11px]">P5 / P10 / P50:</span>
                <span className="text-rose-400 font-semibold text-[11px]">
                  {formatINR(monteCarloResult.stressComparison.stressP5 / terminalDiscount)} /{' '}
                  {formatINR((monteCarloResult.stressComparison.stressP10 ?? 0) / terminalDiscount)} /{' '}
                  {formatINR((monteCarloResult.stressComparison.stressP50 ?? 0) / terminalDiscount)}
                </span>
              </div>
              <div className="flex items-baseline justify-between font-mono mt-0.5">
                <span className="text-neutral-500 text-[11px]">Probability of Loss:</span>
                <span className="text-rose-400 font-semibold">
                  {monteCarloResult.stressComparison.stressProbLoss.toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Fan Chart */}
      <section className="bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 sm:p-5 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-300">
              Correlated Monte Carlo Fan Chart (P10 – P90 Cone)
            </h2>
            <p className="text-[11px] text-neutral-500">
              5,000 Geometric Brownian Motion paths with Cholesky cross-asset correlations
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded bg-emerald-500/20 border border-emerald-500/60 inline-block"></span>
              <span className="text-neutral-400 text-[11px]">80% Likelihood Band (P10-P90)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-emerald-400 inline-block"></span>
              <span className="text-neutral-400 text-[11px]">Median</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 border-t border-dashed border-sky-400 inline-block"></span>
              <span className="text-neutral-400 text-[11px]">Invested</span>
            </div>
          </div>
        </div>

        <div className="h-[320px] w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={fanChartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <defs>
                <linearGradient id="fanBandGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.08} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
              <XAxis
                dataKey="year"
                stroke="#737373"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#404040' }}
              />
              <YAxis
                stroke="#737373"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#404040' }}
                tickFormatter={(val) => formatINR(val, true)}
                width={70}
              />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-neutral-950 border border-neutral-800 p-3 rounded-lg shadow-xl text-xs space-y-1.5 font-mono">
                        <div className="font-bold text-neutral-200 border-b border-neutral-800 pb-1">
                          {label} {isInflationAdjusted && '(Real Purchasing Power)'}
                        </div>
                        <div className="flex justify-between gap-4 text-teal-400">
                          <span>P90 (Optimistic):</span>
                          <span>{formatINR(data.p90, true)}</span>
                        </div>
                        <div className="flex justify-between gap-4 text-emerald-400 font-bold">
                          <span>Median (P50):</span>
                          <span>{formatINR(data.p50, true)}</span>
                        </div>
                        <div className="flex justify-between gap-4 text-rose-400">
                          <span>P10 (Pessimistic):</span>
                          <span>{formatINR(data.p10, true)}</span>
                        </div>
                        <div className="flex justify-between gap-4 text-sky-400 border-t border-neutral-800 pt-1">
                          <span>Invested Capital:</span>
                          <span>{formatINR(data.invested, true)}</span>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Area
                type="monotone"
                dataKey="p10"
                stackId="fan"
                stroke="none"
                fill="transparent"
                legendType="none"
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey="band"
                stackId="fan"
                stroke="#10b981"
                strokeWidth={1}
                strokeOpacity={0.6}
                fill="url(#fanBandGradient)"
                name="80% Probability Interval"
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="p50"
                stroke="#10b981"
                strokeWidth={2.5}
                dot={false}
                name="Median Projection"
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="invested"
                stroke="#38bdf8"
                strokeDasharray="4 4"
                strokeWidth={1.8}
                dot={false}
                name="Total Invested"
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="pt-3 border-t border-neutral-800/80 flex items-start gap-2.5 text-xs text-neutral-300">
          <Info className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            In <span className="font-bold text-white">8 out of 10 simulations</span>, your{' '}
            <span className="text-sky-300 font-mono font-medium">{formatINR(totalInvestedDisplay)}</span>{' '}
            portfolio ends between{' '}
            <span className="text-rose-400 font-mono font-medium">{formatINR(p10Display)}</span> and{' '}
            <span className="text-teal-300 font-mono font-medium">{formatINR(p90Display)}</span>{' '}
            (median: <span className="text-emerald-400 font-mono font-semibold">{formatINR(medianDisplay)}</span>).
            There is an{' '}
            <span className="font-semibold text-emerald-300">{probBeatInflation.toFixed(0)}% likelihood</span>{' '}
            of beating the {(inflationRate * 100).toFixed(1)}% annual inflation hurdle.
          </p>
        </div>
      </section>

      {/* Grid: Allocation Donut & Gains Contribution Bar Chart */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-300">
              Portfolio Weight Distribution
            </h2>
            <span className="text-[11px] text-neutral-500 font-mono">
              {activeAssets.length} Active {activeAssets.length === 1 ? 'Asset' : 'Assets'}
            </span>
          </div>
          <div className="h-[210px] w-full flex items-center justify-center">
            {activeAssets.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={activeAssets}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={80}
                    paddingAngle={3}
                    isAnimationActive={false}
                  >
                    {activeAssets.map((entry) => (
                      <Cell key={entry.id} fill={entry.color} stroke="#0a0a0a" strokeWidth={2} />
                    ))}
                  </Pie>
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-neutral-950 border border-neutral-800 p-2 rounded text-xs font-mono">
                            <div className="font-semibold text-white">{data.name}</div>
                            <div className="text-emerald-400">{data.value}% of portfolio</div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-neutral-500 text-xs">No assets allocated</div>
            )}
          </div>
          <div className="flex flex-wrap gap-2 pt-2 border-t border-neutral-800 max-h-[85px] overflow-y-auto">
            {activeAssets.map((asset) => (
              <div key={asset.id} className="flex items-center gap-1.5 text-[11px]">
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: asset.color }}
                ></span>
                <span className="text-neutral-300 truncate max-w-[120px]">{asset.name}</span>
                <span className="text-neutral-500 font-mono">({asset.value}%)</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-neutral-900/90 border border-neutral-800 rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-300">
              Estimated Wealth Contribution
            </h2>
            <span className="text-[11px] text-neutral-500 font-mono">Top Value Drivers</span>
          </div>
          <div className="h-[210px] w-full pt-1">
            {contributionData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={contributionData}
                  layout="vertical"
                  margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" horizontal={false} />
                  <XAxis
                    type="number"
                    stroke="#737373"
                    fontSize={10}
                    tickFormatter={(v) => formatINR(v, true)}
                    tickLine={false}
                    axisLine={{ stroke: '#404040' }}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    stroke="#737373"
                    fontSize={10}
                    tickLine={false}
                    axisLine={{ stroke: '#404040' }}
                    width={90}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-neutral-950 border border-neutral-800 p-2 rounded text-xs font-mono">
                            <div className="font-semibold text-white">{data.name}</div>
                            <div className="text-emerald-400">
                              Estimated Gains: {formatINR(data.gain, true)}
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar dataKey="gain" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                    {contributionData.map((entry) => (
                      <Cell key={entry.id} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-neutral-500 text-xs flex items-center justify-center h-full">
                No contribution data
              </div>
            )}
          </div>
          <div className="pt-2 border-t border-neutral-800 text-[11px] text-neutral-500 flex items-center justify-between">
            <span>Calculated from compounded CAGR &amp; allocation</span>
            <span className="font-mono text-emerald-400 font-medium">
              Net Gain: {formatINR(Math.max(0, medianDisplay - totalInvestedDisplay))}
            </span>
          </div>
        </div>
      </div>

      <footer className="border-t border-neutral-800/80 pt-4 text-center text-xs text-neutral-500">
        <p className="leading-relaxed">
          Simulation based on historical data and assumptions. Not investment advice. Past returns do
          not guarantee future returns.
        </p>
      </footer>
    </div>
  );
});

RightPanel.displayName = 'RightPanel';
