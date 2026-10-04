/**
 * Data & Regimes Tab - Long-Horizon Validation & Diagnostics
 * Features:
 * 1) Nifty 50 Equity Curve with Market Regime Shading (Bear / High Vol / Normal)
 * 2) Rolling 5-Year (60-Month) Line Charts: Return, Volatility, Correlation
 * 3) Asset Parameter Stability Cards ('stable' | 'watch' | 'unstable')
 * 4) Chow Structural Break Test Table & Sup-F Scan
 * 5) Politis-White Optimal Block Length Display
 * 6) 2-State Gaussian HMM Diagnostics & Reliability Flag
 */

import React, { useState, useMemo, useEffect } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  AreaChart,
  Area,
} from 'recharts';
import {
  ShieldAlert,
  Info,
  Activity,
  Layers,
  Calendar,
  Zap,
  TrendingDown,
  TrendingUp,
  Cpu,
  BarChart2,
} from 'lucide-react';
import {
  ALL_VARIABLE_ASSETS,
  FALLBACK_ASSET_HISTORY,
  rollingStats,
  chowTest,
  supSupFScan,
  parameterStability,
  classifyMarketRegimes,
  fitGaussianHMM,
  politisWhiteBlockLength,
  extractReturnsAndDates,
} from '../engine/index.ts';

interface DataRegimesTabProps {
  userAllocation?: Record<string, number>;
}

interface StaticUniverseDiagnostics {
  regimePoints: ReturnType<typeof classifyMarketRegimes>;
  regimeSummary: {
    bear: number;
    highVol: number;
    normal: number;
    total: number;
    bearPct: string;
    highVolPct: string;
    normalPct: string;
  };
  stabilityMap: Record<string, ReturnType<typeof parameterStability>>;
  hmmResult: ReturnType<typeof fitGaussianHMM>;
  equityChartData: Array<{
    date: string;
    price: number;
    drawdownPct: number;
    regime: string;
    bearZone: number | null;
    highVolZone: number | null;
  }>;
}

let cachedUniverseDiagnostics: StaticUniverseDiagnostics | null = null;

export const DataRegimesTab: React.FC<DataRegimesTabProps> = React.memo(() => {
  const [selectedAssetId, setSelectedAssetId] = useState<string>('nifty_bank');
  const [universeDiagnostics, setUniverseDiagnostics] = useState<StaticUniverseDiagnostics | null>(
    () => cachedUniverseDiagnostics
  );

  useEffect(() => {
    if (cachedUniverseDiagnostics) {
      setUniverseDiagnostics(cachedUniverseDiagnostics);
      return;
    }

    const schedule =
      typeof window !== 'undefined' && 'requestIdleCallback' in window
        ? (window as any).requestIdleCallback
        : (cb: () => void) => setTimeout(cb, 0);

    const handle = schedule(() => {
      const niftySeries = FALLBACK_ASSET_HISTORY['nifty_50'].series;
      const regimePoints = classifyMarketRegimes(niftySeries);
      let bear = 0;
      let highVol = 0;
      let normal = 0;
      for (const pt of regimePoints) {
        if (pt.regime === 'bear') bear++;
        else if (pt.regime === 'high_vol') highVol++;
        else normal++;
      }
      const total = regimePoints.length || 1;
      const regimeSummary = {
        bear,
        highVol,
        normal,
        total,
        bearPct: ((bear / total) * 100).toFixed(1),
        highVolPct: ((highVol / total) * 100).toFixed(1),
        normalPct: ((normal / total) * 100).toFixed(1),
      };

      const stabilityMap: Record<string, ReturnType<typeof parameterStability>> = {};
      for (const asset of ALL_VARIABLE_ASSETS) {
        stabilityMap[asset.id] = parameterStability(
          asset.id,
          FALLBACK_ASSET_HISTORY[asset.id]?.series,
          niftySeries
        );
      }

      const hmmResult = fitGaussianHMM(niftySeries, { seed: 123456 });

      const equityChartData = regimePoints.map((pt) => ({
        date: pt.date,
        price: pt.price,
        drawdownPct: Number((pt.drawdown * 100).toFixed(1)),
        regime: pt.regime,
        bearZone: pt.regime === 'bear' ? pt.price : null,
        highVolZone: pt.regime === 'high_vol' ? pt.price : null,
      }));

      cachedUniverseDiagnostics = {
        regimePoints,
        regimeSummary,
        stabilityMap,
        hmmResult,
        equityChartData,
      };
      setUniverseDiagnostics(cachedUniverseDiagnostics);
    });

    return () => {
      if (typeof window !== 'undefined' && 'cancelIdleCallback' in window && typeof handle === 'number') {
        (window as any).cancelIdleCallback(handle);
      } else {
        clearTimeout(handle);
      }
    };
  }, []);

  // 2. Rolling 5-Year Statistics for chosen asset
  const selectedSeries = useMemo(() => {
    return FALLBACK_ASSET_HISTORY[selectedAssetId]?.series || [];
  }, [selectedAssetId]);

  const rollingData = useMemo(() => {
    const raw = rollingStats(selectedSeries, 60, FALLBACK_ASSET_HISTORY['nifty_50'].series);
    return raw.map((pt) => ({
      date: pt.date,
      'Annualized Return (%)': Number((pt.annualizedReturn * 100).toFixed(2)),
      'Annualized Volatility (%)': Number((pt.annualizedVol * 100).toFixed(2)),
      'Nifty 50 Correlation': Number(pt.correlation.toFixed(2)),
    }));
  }, [selectedSeries]);

  // 3. Chow Test & Sup-F Scan
  const breakTestResults = useMemo(() => {
    const assetExtract = extractReturnsAndDates(selectedSeries);
    const mktExtract = extractReturnsAndDates(FALLBACK_ASSET_HISTORY['nifty_50'].series);
    const T = Math.min(assetExtract.returns.length, mktExtract.returns.length);
    const y = assetExtract.returns.slice(assetExtract.returns.length - T);
    const x = mktExtract.returns.slice(mktExtract.returns.length - T);
    const dates = assetExtract.dates.slice(assetExtract.dates.length - T);

    const supScan = supSupFScan(y, x, dates, 0.15);
    return {
      supScan,
      dates,
    };
  }, [selectedSeries]);

  // 6. Politis-White Optimal Block Length for selected asset
  const optimalBlockLength = useMemo(() => {
    const { returns } = extractReturnsAndDates(selectedSeries);
    return politisWhiteBlockLength(returns);
  }, [selectedSeries]);

  if (!universeDiagnostics) {
    return (
      <div className="flex flex-col items-center justify-center p-16 text-center space-y-4 rounded-xl border border-neutral-800 bg-neutral-900/50 my-6">
        <Activity className="w-8 h-8 text-emerald-400 animate-spin" />
        <div className="space-y-1">
          <p className="text-sm font-semibold text-neutral-200">Computing market regimes &amp; diagnostics...</p>
          <p className="text-xs text-neutral-500">Fitting Gaussian HMM and estimating parameter stabilities across historical data.</p>
        </div>
      </div>
    );
  }

  const { regimeSummary, stabilityMap, hmmResult, equityChartData } = universeDiagnostics;
  const selectedAssetStability = stabilityMap[selectedAssetId];

  return (
    <div className="space-y-6 pb-12">
      {/* Prominent Mandatory Diagnostics Note */}
      <div className="rounded-xl border border-amber-500/30 bg-amber-950/20 p-4 flex items-start gap-3 text-amber-200">
        <Info className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div className="text-xs leading-relaxed">
          <span className="font-semibold text-amber-300">Display-Only Diagnostics Notice:</span>{' '}
          Diagnostics describe the past and are indicative only; they do not change your results unless you pick an option.
          Structural break tests and Markov regime classifications provide exploratory context on empirical asset behavior over historical Indian market cycles.
        </div>
      </div>

      {/* Asset Selector & Key Diagnostic Badges */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-neutral-900 border border-neutral-800 p-4 rounded-xl">
        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold text-neutral-300">Selected Asset:</label>
          <select
            value={selectedAssetId}
            onChange={(e) => setSelectedAssetId(e.target.value)}
            className="bg-neutral-950 border border-neutral-700 text-neutral-100 text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-emerald-500 cursor-pointer"
          >
            {ALL_VARIABLE_ASSETS.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name} ({a.category})
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Parameter Stability Status */}
          <div className="flex items-center gap-2 bg-neutral-950 border border-neutral-800 px-3 py-1.5 rounded-lg text-xs">
            <span className="text-neutral-400">Stability Flag:</span>
            {selectedAssetStability?.status === 'stable' && (
              <span className="px-2 py-0.5 rounded-md bg-emerald-500/20 text-emerald-300 font-semibold border border-emerald-500/30">
                Stable
              </span>
            )}
            {selectedAssetStability?.status === 'watch' && (
              <span className="px-2 py-0.5 rounded-md bg-amber-500/20 text-amber-300 font-semibold border border-amber-500/30">
                Watch List
              </span>
            )}
            {selectedAssetStability?.status === 'unstable' && (
              <span className="px-2 py-0.5 rounded-md bg-rose-500/20 text-rose-300 font-semibold border border-rose-500/30">
                Unstable
              </span>
            )}
          </div>

          {/* Politis-White Optimal Block Length */}
          <div className="flex items-center gap-2 bg-neutral-950 border border-neutral-800 px-3 py-1.5 rounded-lg text-xs">
            <span className="text-neutral-400">Optimal Block Length:</span>
            <span className="font-mono font-semibold text-emerald-400">
              {optimalBlockLength} months
            </span>
            <span className="text-[10px] text-neutral-500">(Politis &amp; White 2004)</span>
          </div>
        </div>
      </div>

      {/* Grid: Nifty 50 Equity Curve with Regime Shading */}
      <div className="bg-neutral-900 border border-neutral-800 p-5 rounded-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-400" />
              Nifty 50 Benchmark &amp; Market Regime Classification
            </h3>
            <p className="text-[11px] text-neutral-400 mt-0.5">
              Rule-based classification: Bear (drawdown &gt; 20%), High Vol (rolling 12m vol &gt; 75th percentile), Normal.
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-rose-950/60 border border-rose-500/30 text-rose-300">
              <span className="w-2 h-2 rounded-full bg-rose-500"></span>
              Bear: {regimeSummary.bear} mo ({regimeSummary.bearPct}%)
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-amber-950/60 border border-amber-500/30 text-amber-300">
              <span className="w-2 h-2 rounded-full bg-amber-500"></span>
              High Vol: {regimeSummary.highVol} mo ({regimeSummary.highVolPct}%)
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-950/60 border border-emerald-500/30 text-emerald-300">
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
              Normal: {regimeSummary.normal} mo ({regimeSummary.normalPct}%)
            </span>
          </div>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={equityChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
              <XAxis dataKey="date" stroke="#737373" tick={{ fontSize: 10 }} />
              <YAxis stroke="#737373" tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#262626', fontSize: 12 }}
                formatter={(val: any, name: any) => [
                  name === 'price' ? `₹ ${val}` : val,
                  name === 'price' ? 'Nifty 50 Price' : name,
                ]}
              />
              <Area type="monotone" dataKey="price" stroke="#10b981" fill="#10b981" fillOpacity={0.15} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Grid: Rolling 5-Year Stats Chart for Chosen Asset */}
      <div className="bg-neutral-900 border border-neutral-800 p-5 rounded-xl space-y-4">
        <div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Activity className="w-4 h-4 text-emerald-400" />
            Rolling 5-Year (60-Month) Parameter Trajectory: {FALLBACK_ASSET_HISTORY[selectedAssetId]?.name}
          </h3>
          <p className="text-[11px] text-neutral-400 mt-0.5">
            Evaluates parameter stability over time. Identifies structural drift in annualized return, risk, and beta correlation.
          </p>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rollingData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
              <XAxis dataKey="date" stroke="#737373" tick={{ fontSize: 10 }} />
              <YAxis stroke="#737373" tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#262626', fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
              <Line type="monotone" dataKey="Annualized Return (%)" stroke="#10b981" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="Annualized Volatility (%)" stroke="#f59e0b" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="Nifty 50 Correlation" stroke="#60a5fa" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Split Row: Structural Break Tests & 2-State Gaussian HMM Diagnostics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chow Structural Break Test Table */}
        <div className="bg-neutral-900 border border-neutral-800 p-5 rounded-xl space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Calendar className="w-4 h-4 text-emerald-400" />
              Chow Test for Structural Breaks in Beta
            </h3>
            <span className="text-[10px] text-neutral-400 bg-neutral-950 px-2 py-0.5 rounded border border-neutral-800">
              indicative only (multiple testing, short samples)
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left text-neutral-300">
              <thead className="bg-neutral-950 text-neutral-400 border-b border-neutral-800">
                <tr>
                  <th className="py-2 px-2.5">Historical Event</th>
                  <th className="py-2 px-2.5">Date</th>
                  <th className="py-2 px-2.5 text-right">F-Stat</th>
                  <th className="py-2 px-2.5 text-right">p-Value</th>
                  <th className="py-2 px-2.5 text-right">Pre/Post Beta</th>
                  <th className="py-2 px-2.5 text-center">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800/60 font-mono text-[11px]">
                {breakTestResults.supScan.knownBreakResults.map((item, idx) => {
                  const res = item.result;
                  if (!res) {
                    return (
                      <tr key={idx} className="hover:bg-neutral-800/30">
                        <td className="py-2 px-2.5 font-sans text-neutral-300">{item.label}</td>
                        <td className="py-2 px-2.5 text-neutral-400">{item.date}</td>
                        <td colSpan={4} className="py-2 px-2.5 text-neutral-500 italic text-center">
                          Insufficient history coverage for event
                        </td>
                      </tr>
                    );
                  }
                  return (
                    <tr key={idx} className="hover:bg-neutral-800/30">
                      <td className="py-2 px-2.5 font-sans text-neutral-200">{item.label}</td>
                      <td className="py-2 px-2.5 text-neutral-300">{item.date}</td>
                      <td className="py-2 px-2.5 text-right font-semibold">{res.fStat}</td>
                      <td className="py-2 px-2.5 text-right">
                        <span className={res.isSignificant ? 'text-amber-400 font-bold' : 'text-neutral-400'}>
                          {res.pValue.toFixed(4)}
                        </span>
                      </td>
                      <td className="py-2 px-2.5 text-right text-neutral-400">
                        {res.preBeta} &rarr; {res.postBeta}
                      </td>
                      <td className="py-2 px-2.5 text-center font-sans">
                        {res.isSignificant ? (
                          <span className="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300 text-[10px] font-semibold">
                            Break Detected
                          </span>
                        ) : (
                          <span className="px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-400 text-[10px]">
                            Stable
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}

                {/* Maximum Sup-F Scan Row */}
                <tr className="bg-neutral-950/80 border-t border-neutral-700">
                  <td className="py-2.5 px-2.5 font-sans font-semibold text-emerald-400">
                    Max Sup-F Scan (Global)
                  </td>
                  <td className="py-2.5 px-2.5 text-emerald-300">
                    {breakTestResults.supScan.maxFBreakDate || 'N/A'}
                  </td>
                  <td className="py-2.5 px-2.5 text-right font-bold text-emerald-400">
                    {breakTestResults.supScan.maxF}
                  </td>
                  <td className="py-2.5 px-2.5 text-right font-bold text-emerald-400">
                    {breakTestResults.supScan.pValue.toFixed(4)}
                  </td>
                  <td colSpan={2} className="py-2.5 px-2.5 text-right font-sans text-[10px] text-neutral-400">
                    Scanned across {breakTestResults.supScan.testedDatesCount} candidate dates
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* 2-State Gaussian HMM Diagnostics */}
        <div className="bg-neutral-900 border border-neutral-800 p-5 rounded-xl space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Cpu className="w-4 h-4 text-emerald-400" />
              2-State Gaussian HMM (Nifty 50 Monthly Log Returns)
            </h3>
            {hmmResult.unreliable ? (
              <span className="px-2 py-0.5 rounded-md bg-amber-500/20 text-amber-300 text-[10px] font-semibold border border-amber-500/30">
                Fit Flagged: Unreliable
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded-md bg-emerald-500/20 text-emerald-300 text-[10px] font-semibold border border-emerald-500/30">
                Fit Reliable (&ge; 6mo duration)
              </span>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            {hmmResult.states.map((st) => (
              <div
                key={st.stateIndex}
                className="bg-neutral-950 border border-neutral-800 p-3 rounded-lg space-y-1.5"
              >
                <div className="font-semibold text-neutral-200 flex items-center justify-between">
                  <span>State {st.stateIndex + 1}: {st.name}</span>
                </div>
                <div className="flex justify-between text-neutral-400">
                  <span>Annualized Return:</span>
                  <span className="font-mono text-neutral-200">{(st.annualizedMean * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-neutral-400">
                  <span>Annualized Volatility:</span>
                  <span className="font-mono text-neutral-200">{(st.annualizedVol * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-neutral-400">
                  <span>Average Duration:</span>
                  <span className="font-mono text-emerald-400 font-semibold">{st.averageDurationMonths} months</span>
                </div>
              </div>
            ))}
          </div>

          <div className="bg-neutral-950 border border-neutral-800 p-3 rounded-lg text-xs space-y-2">
            <div className="text-neutral-400 font-semibold">2x2 Transition Probability Matrix:</div>
            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] text-center">
              <div className="bg-neutral-900 p-1.5 rounded border border-neutral-800">
                P(S1 &rarr; S1): <span className="text-emerald-400 font-bold">{hmmResult.transitionMatrix[0][0]}</span>
              </div>
              <div className="bg-neutral-900 p-1.5 rounded border border-neutral-800">
                P(S1 &rarr; S2): <span className="text-neutral-300">{hmmResult.transitionMatrix[0][1]}</span>
              </div>
              <div className="bg-neutral-900 p-1.5 rounded border border-neutral-800">
                P(S2 &rarr; S1): <span className="text-neutral-300">{hmmResult.transitionMatrix[1][0]}</span>
              </div>
              <div className="bg-neutral-900 p-1.5 rounded border border-neutral-800">
                P(S2 &rarr; S2): <span className="text-emerald-400 font-bold">{hmmResult.transitionMatrix[1][1]}</span>
              </div>
            </div>
            {hmmResult.unreliableReason && (
              <div className="text-[11px] text-amber-400/90 pt-1">
                Note: {hmmResult.unreliableReason}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Universe Stability Overview Table */}
      <div className="bg-neutral-900 border border-neutral-800 p-5 rounded-xl space-y-4">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <BarChart2 className="w-4 h-4 text-emerald-400" />
          Full Universe 5-Year Parameter Stability Roster
        </h3>
        <p className="text-[11px] text-neutral-400">
          Documents maximum drift in rolling volatility (&gt; 40% unstable, &gt; 20% watch) and market correlation relative to the full historical baseline.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left text-neutral-300">
            <thead className="bg-neutral-950 text-neutral-400 border-b border-neutral-800 font-medium">
              <tr>
                <th className="py-2 px-3">Asset</th>
                <th className="py-2 px-3">Status</th>
                <th className="py-2 px-3 text-right">Baseline Vol</th>
                <th className="py-2 px-3 text-right">Rolling Vol Range</th>
                <th className="py-2 px-3 text-right">Max Vol Drift</th>
                <th className="py-2 px-3 text-right">Baseline Corr</th>
                <th className="py-2 px-3 text-right">Rolling Corr Range</th>
                <th className="py-2 px-3 text-right">Max Corr Shift</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800/60 font-mono text-[11px]">
              {ALL_VARIABLE_ASSETS.map((asset) => {
                const stab = stabilityMap[asset.id];
                if (!stab) return null;
                return (
                  <tr key={asset.id} className="hover:bg-neutral-800/30">
                    <td className="py-2 px-3 font-sans font-medium text-neutral-200">{asset.name}</td>
                    <td className="py-2 px-3 font-sans">
                      {stab.status === 'stable' && (
                        <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-semibold text-[10px]">
                          Stable
                        </span>
                      )}
                      {stab.status === 'watch' && (
                        <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-semibold text-[10px]">
                          Watch
                        </span>
                      )}
                      {stab.status === 'unstable' && (
                        <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 font-semibold text-[10px]">
                          Unstable
                        </span>
                      )}
                    </td>
                    <td className="py-2 px-3 text-right">{(stab.fullSampleVol * 100).toFixed(1)}%</td>
                    <td className="py-2 px-3 text-right text-neutral-400">
                      {(stab.minRollingVol * 100).toFixed(1)}% &ndash; {(stab.maxRollingVol * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-right font-semibold">
                      {(stab.maxVolRelativeDrift * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-right">{stab.fullSampleCorr.toFixed(2)}</td>
                    <td className="py-2 px-3 text-right text-neutral-400">
                      {stab.minRollingCorr.toFixed(2)} &ndash; {stab.maxRollingCorr.toFixed(2)}
                    </td>
                    <td className="py-2 px-3 text-right font-semibold">
                      {stab.maxCorrAbsoluteDrift.toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
});

DataRegimesTab.displayName = 'DataRegimesTab';
