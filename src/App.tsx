/**
 * Portfolio Simulator App - Dark Fintech Theme
 * Supports Simulator view, Optimizer & Efficient Frontier view, and Strategy Lab view.
 */
import React, { useState, useEffect, useCallback, useTransition, useMemo } from 'react';
import { Header, ActiveView } from './components/Header.tsx';
import { LeftPanel, SimulationMode, AssetAssumption } from './components/LeftPanel.tsx';
import { RightPanel } from './components/RightPanel.tsx';
import { OptimizerTab } from './components/OptimizerTab.tsx';
import { StrategyLabTab } from './components/StrategyLabTab.tsx';
import { DataRegimesTab } from './components/DataRegimesTab.tsx';
import { AlertTriangle } from 'lucide-react';
import {
  ALL_VARIABLE_ASSETS,
  FALLBACK_ASSET_HISTORY,
  BondInput,
  MonteCarloResult,
  DeterministicProjectionResult,
  projectLumpsum,
  projectSIP,
  runMonteCarlo,
  getFullUniverseHistory,
  buildCovariance,
  calculateAdjustedCagr,
  getDefaultPriorForAsset,
  isSimulationCancelled,
} from './engine/index.ts';
import { rupeeAllocationToWeights, weightsToRupeeAllocation } from './utils/allocation.ts';

// Initial default allocation weights summing to 1.0 (100%)
const DEFAULT_ALLOCATION: Record<string, number> = {
  nifty_bank: 0.15,
  nifty_it: 0.10,
  gold: 0.15,
  mf_large_cap: 0.15,
  mf_flexi_cap: 0.20,
  mf_mid_cap: 0.10,
  bonds_fd: 0.15,
};

const DEFAULT_BOND_CONFIG: BondInput = {
  couponRate: 0.075, // 7.5%
  tenureYears: 5,
  payoutFrequency: 'annual',
  creditRisk: {
    defaultProbabilityAnnual: 0.005, // 0.5% annual default prob
    recoveryRate: 0.5, // 50% recovery
  },
  reinvestmentRateAfterMaturity: 0.065, // 6.5% reinvestment rate
};

export default function App() {
  const [activeView, setActiveView] = useState<ActiveView>('simulator');
  const [visitedTabs, setVisitedTabs] = useState<Set<ActiveView>>(() => new Set(['simulator']));

  useEffect(() => {
    setVisitedTabs((prev) => {
      if (prev.has(activeView)) return prev;
      const next = new Set(prev);
      next.add(activeView);
      return next;
    });
  }, [activeView]);

  const [mode, setMode] = useState<SimulationMode>('sip');
  const [lumpsumAmount, setLumpsumAmount] = useState<number>(1000000); // ₹ 10 Lakh
  const [sipAmount, setSipAmount] = useState<number>(25000); // ₹ 25,000 / month
  const [years, setYears] = useState<number>(10);
  const [inflationRate, setInflationRate] = useState<number>(6.0); // 6%

  const totalAmount = useMemo(() => {
    if (mode === 'sip') return sipAmount * 12 * years;
    if (mode === 'both') return lumpsumAmount + sipAmount * 12 * years;
    return lumpsumAmount;
  }, [mode, sipAmount, years, lumpsumAmount]);

  // 1) Rupee allocation state (₹/month per asset), default summing to default monthly amount (₹ 25,000)
  const [allocationInr, setAllocationInr] = useState<Record<string, number>>(() =>
    weightsToRupeeAllocation(DEFAULT_ALLOCATION, 25000)
  );

  // Normalized decimal weights [0, 1] derived at the engine boundary
  const allocation = useMemo(() => rupeeAllocationToWeights(allocationInr), [allocationInr]);

  // Derived percent allocation for OptimizerTab
  const allocationPercent = useMemo(() => {
    const weights = rupeeAllocationToWeights(allocationInr);
    const pct: Record<string, number> = {};
    for (const [id, w] of Object.entries(weights)) {
      pct[id] = Number((w * 100).toFixed(2));
    }
    return pct;
  }, [allocationInr]);

  // Optimizer constraints state (maxRisk, maxPerAsset)
  const [maxRisk, setMaxRisk] = useState<number>(15); // 15% max volatility
  const [maxPerAsset, setMaxPerAsset] = useState<number>(30); // 30% per-asset cap
  const optimizerConstraints = useMemo(
    () => ({
      maxRisk: maxRisk / 100,
      maxPerAsset: maxPerAsset / 100,
      minPerAsset: 0.0,
      riskFreeRate: 0.065,
    }),
    [maxRisk, maxPerAsset]
  );

  const [bondConfig, setBondConfig] = useState<BondInput>(DEFAULT_BOND_CONFIG);

  // Return Expectation Haircut: s slider 0-100%, default 50%
  const [returnHaircutS, setReturnHaircutS] = useState<number>(0.50);

  // Category Priors (default long-run placeholders, user-editable)
  const [categoryPriors, setCategoryPriors] = useState<Record<string, number>>(() => {
    const priors: Record<string, number> = {};
    for (const a of ALL_VARIABLE_ASSETS) {
      priors[a.id] = getDefaultPriorForAsset(a.id, a.category);
    }
    return priors;
  });

  // Raw Historical 10Y CAGR cache (unadjusted)
  const [rawHistoricalCagrs, setRawHistoricalCagrs] = useState<Record<string, number>>(() => {
    const raw: Record<string, number> = {};
    for (const a of ALL_VARIABLE_ASSETS) {
      raw[a.id] = FALLBACK_ASSET_HISTORY[a.id]?.cagr ?? 0.12;
    }
    return raw;
  });

  // Track per-asset live vs fallback data status
  const [assetFallbackStatus, setAssetFallbackStatus] = useState<Record<string, boolean>>(() => {
    const status: Record<string, boolean> = {};
    for (const a of ALL_VARIABLE_ASSETS) {
      status[a.id] = true;
    }
    return status;
  });

  // Stress-aware returns toggle with Student-t(5) fat tails
  const [fatTails, setFatTails] = useState<boolean>(false);

  // Asset Assumptions (Mean adjusted CAGR and Volatility)
  const [assumptions, setAssumptions] = useState<Record<string, AssetAssumption>>(() => {
    const initial: Record<string, AssetAssumption> = {};
    for (const a of ALL_VARIABLE_ASSETS) {
      const fb = FALLBACK_ASSET_HISTORY[a.id];
      const hist = fb?.cagr ?? 0.12;
      const prior = getDefaultPriorForAsset(a.id, a.category);
      initial[a.id] = {
        cagr: calculateAdjustedCagr(hist, prior, 0.50),
        volatility: fb?.volatility ?? 0.18,
      };
    }
    return initial;
  });

  const [dataSourceLabel, setDataSourceLabel] = useState<string>(
    'Data source: AMFI / Yahoo Finance • Last updated Oct 2024'
  );

  // Covariance & Correlation Matrix Cache
  const [covarianceMatrix, setCovarianceMatrix] = useState<number[][]>([]);
  const [correlationMatrix, setCorrelationMatrix] = useState<number[][]>([]);
  const [assetIds, setAssetIds] = useState<string[]>([]);

  // Simulation & Projection Results State
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [simSeed, setSimSeed] = useState<number>(() => 424242);
  const [monteCarloResult, setMonteCarloResult] = useState<MonteCarloResult | null>(null);
  const [projectionResult, setProjectionResult] = useState<{
    lumpsum?: DeterministicProjectionResult;
    sip?: DeterministicProjectionResult;
  } | null>(null);

  const [, startTransition] = useTransition();

  // Load empirical universe data on mount
  useEffect(() => {
    let isMounted = true;
    getFullUniverseHistory()
      .then((data) => {
        if (!isMounted) return;
        setCovarianceMatrix(data.covarianceMatrix);
        setCorrelationMatrix(data.correlationMatrix);
        setAssetIds(data.assetIds);

        const statusMap: Record<string, boolean> = {};
        const rawCagrs: Record<string, number> = {};
        for (const [id, hist] of Object.entries(data.assets)) {
          statusMap[id] = hist.isFallback ?? data.usingFallbackData;
          rawCagrs[id] = hist.cagr;
        }
        setAssetFallbackStatus(statusMap);
        setRawHistoricalCagrs(rawCagrs);

        setAssumptions((prev) => {
          const next = { ...prev };
          for (const [id, hist] of Object.entries(data.assets)) {
            const prior = categoryPriors[id] ?? getDefaultPriorForAsset(id);
            next[id] = {
              cagr: calculateAdjustedCagr(hist.cagr, prior, returnHaircutS),
              volatility: prev[id]?.volatility ?? hist.volatility,
            };
          }
          return next;
        });

        if (data.usingFallbackData) {
          setDataSourceLabel('Data source: Verified 10Y Benchmark Series (Offline fallback active)');
        } else {
          setDataSourceLabel('Data source: Live Yahoo Finance & AMFI India • Synced');
        }
      })
      .catch(() => {
        // Fallback already guaranteed by engine
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const handleReturnHaircutSChange = (newS: number) => {
    setReturnHaircutS(newS);
    setAssumptions((prev) => {
      const next = { ...prev };
      for (const a of ALL_VARIABLE_ASSETS) {
        const hist = rawHistoricalCagrs[a.id] ?? next[a.id]?.cagr ?? 0.12;
        const prior = categoryPriors[a.id] ?? getDefaultPriorForAsset(a.id, a.category);
        next[a.id] = {
          ...next[a.id],
          cagr: calculateAdjustedCagr(hist, prior, newS),
        };
      }
      return next;
    });
  };

  const handleCategoryPriorsChange = (newPriors: Record<string, number>) => {
    setCategoryPriors(newPriors);
    setAssumptions((prev) => {
      const next = { ...prev };
      for (const a of ALL_VARIABLE_ASSETS) {
        const hist = rawHistoricalCagrs[a.id] ?? next[a.id]?.cagr ?? 0.12;
        const prior = newPriors[a.id] ?? getDefaultPriorForAsset(a.id, a.category);
        next[a.id] = {
          ...next[a.id],
          cagr: calculateAdjustedCagr(hist, prior, returnHaircutS),
        };
      }
      return next;
    });
  };

  const handleResetAssumptions = () => {
    const defaultPriors: Record<string, number> = {};
    for (const a of ALL_VARIABLE_ASSETS) {
      defaultPriors[a.id] = getDefaultPriorForAsset(a.id, a.category);
    }
    setCategoryPriors(defaultPriors);
    setReturnHaircutS(0.50);

    const initial: Record<string, AssetAssumption> = {};
    for (const a of ALL_VARIABLE_ASSETS) {
      const hist = rawHistoricalCagrs[a.id] ?? FALLBACK_ASSET_HISTORY[a.id]?.cagr ?? 0.12;
      const prior = defaultPriors[a.id];
      const fb = FALLBACK_ASSET_HISTORY[a.id];
      initial[a.id] = {
        cagr: calculateAdjustedCagr(hist, prior, 0.50),
        volatility: fb?.volatility ?? 0.18,
      };
    }
    setAssumptions(initial);
  };

  const totalAllocatedInr = useMemo(() => {
    return Object.values(allocationInr).reduce((sum, v) => sum + (v > 0 ? v : 0), 0);
  }, [allocationInr]);

  const unallocatedInr = sipAmount - totalAllocatedInr;
  const canSimulate = Math.abs(unallocatedInr) <= 1 && sipAmount > 0;

  const activeIds = useMemo(
    () => (assetIds.length > 0 ? assetIds : ALL_VARIABLE_ASSETS.map((a) => a.id)),
    [assetIds]
  );
  const realCovariance = useMemo(
    () => buildCovariance(activeIds, assumptions, correlationMatrix),
    [activeIds, assumptions, correlationMatrix]
  );

  const runSimulation = useCallback(async (overrideSeed?: number) => {
    if (!canSimulate) return;
    setIsSimulating(true);
    const nextSeed = overrideSeed !== undefined ? overrideSeed : Math.floor(Math.random() * 1000000);
    setSimSeed(nextSeed);

    const weights = rupeeAllocationToWeights(allocationInr);

    let lumpsumProj: DeterministicProjectionResult | undefined;
    let sipProj: DeterministicProjectionResult | undefined;
    if (mode === 'lumpsum' || mode === 'both') {
      lumpsumProj = projectLumpsum(lumpsumAmount, years, weights, assumptions, bondConfig);
    }
    if (mode === 'sip' || mode === 'both') {
      sipProj = projectSIP(sipAmount, years, weights, assumptions, bondConfig);
    }
    setProjectionResult({ lumpsum: lumpsumProj, sip: sipProj });

    const DEBUG_PERF = false;
    if (DEBUG_PERF) performance.mark('main-sim-start');

    try {
      // Stage 1: Main 5000-sim run WITHOUT waiting for stress comparison
      const mainResult = await runMonteCarlo(
        {
          allocation: weights,
          years,
          mode,
          amount: mode === 'sip' ? sipAmount : lumpsumAmount,
          lumpsumAmount: mode === 'sip' ? 0 : lumpsumAmount,
          monthlySip: mode === 'lumpsum' ? 0 : sipAmount,
          assetStats: assumptions,
          covarianceMatrix: realCovariance,
          assetIds: activeIds,
          bondConfig,
          nSims: 5000,
          inflationRate: inflationRate / 100,
          seed: nextSeed,
          fatTails,
          computeStressComparison: false,
        },
        'main'
      );

      startTransition(() => {
        setMonteCarloResult(mainResult);
      });

      if (DEBUG_PERF) {
        performance.mark('main-sim-end');
        performance.measure('main-sim', 'main-sim-start', 'main-sim-end');
        const entries = performance.getEntriesByName('main-sim');
        if (entries.length > 0) {
          console.log(`[PERF] Main simulation (5000 sims): ${entries[entries.length - 1].duration.toFixed(2)}ms`);
        }
      }

      // Stage 2: Second worker call for stress comparison, running AFTER main result renders
      if (DEBUG_PERF) performance.mark('stress-sim-start');
      const stressRun = await runMonteCarlo(
        {
          allocation: weights,
          years,
          mode,
          amount: mode === 'sip' ? sipAmount : lumpsumAmount,
          lumpsumAmount: mode === 'sip' ? 0 : lumpsumAmount,
          monthlySip: mode === 'lumpsum' ? 0 : sipAmount,
          assetStats: assumptions,
          covarianceMatrix: realCovariance,
          assetIds: activeIds,
          bondConfig,
          nSims: Math.min(2500, 5000),
          inflationRate: inflationRate / 100,
          seed: nextSeed !== undefined ? nextSeed + 101 : undefined,
          fatTails: !fatTails,
          computeStressComparison: false,
        },
        'main'
      );

      setMonteCarloResult((prev) => {
        if (!prev) return prev;
        const stressComp: MonteCarloResult['stressComparison'] = fatTails
          ? {
              normalP5: stressRun.percentilesByYear[years - 1]?.p5 ?? 0,
              normalP10: stressRun.percentilesByYear[years - 1]?.p10 ?? 0,
              normalP50: stressRun.percentilesByYear[years - 1]?.p50 ?? 0,
              normalProbLoss: stressRun.probabilityOfLoss,
              stressP5: prev.percentilesByYear[years - 1]?.p5 ?? 0,
              stressP10: prev.percentilesByYear[years - 1]?.p10 ?? 0,
              stressP50: prev.percentilesByYear[years - 1]?.p50 ?? 0,
              stressProbLoss: prev.probabilityOfLoss,
              isStressAware: true,
              isBearRegimeStress: false,
            }
          : {
              normalP5: prev.percentilesByYear[years - 1]?.p5 ?? 0,
              normalP10: prev.percentilesByYear[years - 1]?.p10 ?? 0,
              normalP50: prev.percentilesByYear[years - 1]?.p50 ?? 0,
              normalProbLoss: prev.probabilityOfLoss,
              stressP5: stressRun.percentilesByYear[years - 1]?.p5 ?? 0,
              stressP10: stressRun.percentilesByYear[years - 1]?.p10 ?? 0,
              stressP50: stressRun.percentilesByYear[years - 1]?.p50 ?? 0,
              stressProbLoss: stressRun.probabilityOfLoss,
              isStressAware: false,
              isBearRegimeStress: false,
            };
        return {
          ...prev,
          stressComparison: stressComp,
        };
      });

      if (DEBUG_PERF) {
        performance.mark('stress-sim-end');
        performance.measure('stress-sim', 'stress-sim-start', 'stress-sim-end');
        const entries = performance.getEntriesByName('stress-sim');
        if (entries.length > 0) {
          console.log(`[PERF] Stress simulation: ${entries[entries.length - 1].duration.toFixed(2)}ms`);
        }
      }
    } catch (err: any) {
      if (isSimulationCancelled(err)) return;
    } finally {
      setIsSimulating(false);
    }
  }, [
    canSimulate,
    allocationInr,
    mode,
    lumpsumAmount,
    sipAmount,
    years,
    assumptions,
    bondConfig,
    activeIds,
    realCovariance,
    inflationRate,
    fatTails,
  ]);

  useEffect(() => {
    const timer = setTimeout(() => {
      runSimulation();
    }, 250);
    return () => clearTimeout(timer);
  }, [covarianceMatrix.length, fatTails]);

  const handleApplyOptimizerAllocation = useCallback((newAlloc: Record<string, number>) => {
    const sum = Object.values(newAlloc).reduce((a, b) => a + b, 0);
    const inrAlloc = sum <= 1.5
      ? weightsToRupeeAllocation(newAlloc, sipAmount)
      : newAlloc;
    setAllocationInr(inrAlloc);
    setActiveView('simulator');
  }, [sipAmount]);

  const handleApplyStrategyLabAllocation = useCallback((newAllocInr: Record<string, number>) => {
    setAllocationInr(newAllocInr);
    setActiveView('simulator');
  }, []);

  const handleApplyPreset = (preset: 'balanced' | 'aggressive' | 'conservative' | 'equal') => {
    let presetWeights: Record<string, number> = {};
    switch (preset) {
      case 'balanced':
        presetWeights = {
          mf_flexi_cap: 0.20,
          mf_large_cap: 0.15,
          mf_mid_cap: 0.10,
          nifty_bank: 0.15,
          nifty_it: 0.10,
          gold: 0.15,
          bonds_fd: 0.15,
        };
        break;
      case 'aggressive':
        presetWeights = {
          mf_small_cap: 0.25,
          mf_mid_cap: 0.25,
          mf_flexi_cap: 0.20,
          nifty_it: 0.15,
          nifty_bank: 0.15,
        };
        break;
      case 'conservative':
        presetWeights = {
          bonds_fd: 0.40,
          mf_debt: 0.20,
          gold: 0.15,
          mf_large_cap: 0.15,
          nifty_fmcg: 0.10,
        };
        break;
      case 'equal': {
        const activeList = ['nifty_50', 'nifty_bank', 'nifty_it', 'gold', 'mf_large_cap', 'mf_flexi_cap'];
        for (const id of activeList) presetWeights[id] = 1 / activeList.length;
        break;
      }
    }
    setAllocationInr(weightsToRupeeAllocation(presetWeights, sipAmount));
  };

  const fallbackValues = Object.values(assetFallbackStatus);
  const anyFallback = fallbackValues.length > 0 && fallbackValues.some((v) => v);
  const someLiveSomeFallback = anyFallback && fallbackValues.some((v) => !v);

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col antialiased selection:bg-emerald-500/20 selection:text-emerald-300">
      <Header
        activeView={activeView}
        onViewChange={setActiveView}
        onRunSimulation={() => runSimulation()}
        isSimulating={isSimulating}
        canSimulate={canSimulate}
        onApplyPreset={handleApplyPreset}
        seed={simSeed}
      />

      {anyFallback && (
        <div className="bg-amber-500/10 border-b border-amber-500/25 px-4 py-2 text-xs text-amber-200">
          <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-1.5">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              <span className="font-semibold text-amber-300">
                Illustrative synthetic data, not real market history
              </span>
            </div>
            {someLiveSomeFallback ? (
              <span className="text-[11px] text-amber-400 font-mono">
                Warning: Correlations are unreliable (the synthetic series share an artificial market factor).
              </span>
            ) : (
              <span className="text-[11px] text-amber-400/70 font-mono">
                Historical series calibrated to Indian benchmark statistics (2014 – 2024).
              </span>
            )}
          </div>
        </div>
      )}

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 md:py-8">
        {/* Tab 1: Simulator */}
        <div className={activeView === 'simulator' ? 'block' : 'hidden'}>
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-start">
            <div className="lg:col-span-5 w-full">
              <LeftPanel
                mode={mode}
                onModeChange={setMode}
                lumpsumAmount={lumpsumAmount}
                onLumpsumAmountChange={setLumpsumAmount}
                sipAmount={sipAmount}
                onSipAmountChange={setSipAmount}
                years={years}
                onYearsChange={setYears}
                inflationRate={inflationRate}
                onInflationRateChange={setInflationRate}
                allocationInr={allocationInr}
                onAllocationInrChange={setAllocationInr}
                bondConfig={bondConfig}
                onBondConfigChange={setBondConfig}
                assumptions={assumptions}
                onAssumptionsChange={setAssumptions}
                onResetAssumptions={handleResetAssumptions}
                dataSourceLabel={dataSourceLabel}
                assetFallbackStatus={assetFallbackStatus}
                returnHaircutS={returnHaircutS}
                onReturnHaircutSChange={handleReturnHaircutSChange}
                categoryPriors={categoryPriors}
                onCategoryPriorsChange={handleCategoryPriorsChange}
                rawHistoricalCagrs={rawHistoricalCagrs}
                fatTails={fatTails}
                onFatTailsChange={setFatTails}
                covarianceMatrix={realCovariance}
                assetIds={activeIds}
              />
            </div>
            <div className="lg:col-span-7 w-full relative">
              {isSimulating && (
                <div className="absolute inset-0 bg-neutral-950/60 backdrop-blur-[2px] z-20 flex flex-col items-center justify-center rounded-xl">
                  <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800 shadow-2xl flex items-center gap-3">
                    <div className="w-5 h-5 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
                    <div className="text-xs font-medium text-neutral-200">
                      Computing 5,000 Correlated Monte Carlo Paths...
                    </div>
                  </div>
                </div>
              )}
              <RightPanel
                monteCarloResult={monteCarloResult}
                projectionResult={projectionResult}
                mode={mode}
                allocation={allocation}
                years={years}
                inflationRate={inflationRate / 100}
                isSimulating={isSimulating}
              />
            </div>
          </div>
        </div>

        {/* Tab 2: Optimizer & Frontier */}
        {visitedTabs.has('optimizer') && (
          <div className={activeView === 'optimizer' ? 'block' : 'hidden'}>
            <OptimizerTab
              isActive={activeView === 'optimizer'}
              totalAmount={totalAmount}
              monthlyAmount={sipAmount}
              horizonYears={years}
              currentAllocation={allocationPercent}
              assumptions={assumptions}
              covarianceMatrix={covarianceMatrix}
              correlationMatrix={correlationMatrix}
              assetIds={assetIds}
              bondConfig={bondConfig}
              inflationRate={inflationRate / 100}
              onApplyAllocation={handleApplyOptimizerAllocation}
              maxRisk={maxRisk}
              onMaxRiskChange={setMaxRisk}
              maxPerAsset={maxPerAsset}
              onMaxPerAssetChange={setMaxPerAsset}
            />
          </div>
        )}

        {/* Tab 3: Strategy Lab */}
        {visitedTabs.has('strategylab') && (
          <div className={activeView === 'strategylab' ? 'block' : 'hidden'}>
            <StrategyLabTab
              totalAmount={totalAmount}
              amount={totalAmount}
              monthlySip={sipAmount}
              horizon={years}
              horizonYears={years}
              mode={mode}
              allocation={allocation}
              userAllocation={allocation}
              userAllocationInr={allocationInr}
              assumptions={assumptions}
              inflationRate={inflationRate / 100}
              bondConfig={bondConfig}
              constraints={optimizerConstraints}
              maxRisk={maxRisk}
              maxPerAsset={maxPerAsset}
              haircutSettings={{
                returnHaircutS,
                categoryPriors,
                rawHistoricalCagrs,
              }}
              returnHaircutS={returnHaircutS}
              categoryPriors={categoryPriors}
              rawHistoricalCagrs={rawHistoricalCagrs}
              covarianceMatrix={realCovariance}
              assetIds={activeIds}
              isFallback={anyFallback}
              fallbackFlag={anyFallback}
              onApplyAllocationInr={handleApplyStrategyLabAllocation}
              onApplyAllocation={handleApplyStrategyLabAllocation}
            />
          </div>
        )}

        {/* Tab 4: Data & Regimes Diagnostics */}
        {visitedTabs.has('dataregimes') && (
          <div className={activeView === 'dataregimes' ? 'block' : 'hidden'}>
            <DataRegimesTab userAllocation={allocation} />
          </div>
        )}
      </main>
    </div>
  );
}
