/**
 * Render Check Smoke Test
 * Mounts App and all three tabs using react-dom/server, asserting that each tab renders without throwing.
 */
import React from 'react';
import { renderToString } from 'react-dom/server';
import App from '../src/App.tsx';
import { Header } from '../src/components/Header.tsx';
import { LeftPanel } from '../src/components/LeftPanel.tsx';
import { RightPanel } from '../src/components/RightPanel.tsx';
import { OptimizerTab } from '../src/components/OptimizerTab.tsx';
import { StrategyLabTab } from '../src/components/StrategyLabTab.tsx';
import { rupeeAllocationToWeights, weightsToRupeeAllocation } from '../src/utils/allocation.ts';
import { ALL_VARIABLE_ASSETS, FALLBACK_ASSET_HISTORY } from '../src/engine/index.ts';

if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as any;
}

console.log('--- Starting Smoke Test: Rendering App and all tabs ---');

console.log('1. Rendering full App component...');
const appHtml = renderToString(<App />);
console.log(`App rendered successfully (${appHtml.length} bytes of HTML).`);

const checks = [
  { name: 'Header brand title', needle: 'Portfolio Simulator' },
  { name: 'Simulator tab nav button', needle: 'Simulator' },
  { name: 'Optimizer tab nav button', needle: 'Optimizer &amp; Frontier' },
  { name: 'Strategy Lab tab nav button', needle: 'Strategy Lab' },
  { name: 'Simulator view (LeftPanel monthly input)', needle: 'Monthly investment amount' },
];

let allPassed = true;
for (const check of checks) {
  if (appHtml.includes(check.needle)) {
    console.log(`  ✓ Found ${check.name}`);
  } else {
    console.error(`  ✗ Missing ${check.name} ("${check.needle}")`);
    allPassed = false;
  }
}

if (!allPassed) {
  console.error('Smoke test FAILED: Some tab content was not found in App HTML.');
  process.exit(1);
}

console.log('\n2. Testing isolated tab rendering:');
const defaultAllocation: Record<string, number> = {
  nifty_bank: 0.15,
  nifty_it: 0.10,
  gold: 0.15,
  mf_large_cap: 0.15,
  mf_flexi_cap: 0.20,
  mf_mid_cap: 0.10,
  bonds_fd: 0.15,
};
const defaultAllocationInr = weightsToRupeeAllocation(defaultAllocation, 25000);
const defaultWeights = rupeeAllocationToWeights(defaultAllocationInr);
const defaultAllocationPercent: Record<string, number> = {};
for (const [id, w] of Object.entries(defaultWeights)) {
  defaultAllocationPercent[id] = w * 100;
}
const defaultAssumptions: Record<string, { cagr: number; volatility: number }> = {};
for (const a of ALL_VARIABLE_ASSETS) {
  const fb = FALLBACK_ASSET_HISTORY[a.id];
  defaultAssumptions[a.id] = {
    cagr: fb?.cagr ?? 0.12,
    volatility: fb?.volatility ?? 0.18,
  };
}
const bondConfig = {
  couponRate: 0.075,
  tenureYears: 5,
  payoutFrequency: 'annual' as const,
  creditRisk: { defaultProbabilityAnnual: 0.005, recoveryRate: 0.5 },
  reinvestmentRateAfterMaturity: 0.065,
};

renderToString(
  <Header
    activeView="simulator"
    onViewChange={() => {}}
    onRunSimulation={() => {}}
    isSimulating={false}
    canSimulate={true}
    onApplyPreset={() => {}}
  />
);

const leftPanelHtml = renderToString(
  <LeftPanel
    mode="sip"
    onModeChange={() => {}}
    lumpsumAmount={1000000}
    onLumpsumAmountChange={() => {}}
    sipAmount={25000}
    onSipAmountChange={() => {}}
    years={10}
    onYearsChange={() => {}}
    inflationRate={6.0}
    onInflationRateChange={() => {}}
    allocationInr={defaultAllocationInr}
    onAllocationInrChange={() => {}}
    bondConfig={bondConfig}
    onBondConfigChange={() => {}}
    assumptions={defaultAssumptions}
    onAssumptionsChange={() => {}}
    onResetAssumptions={() => {}}
    dataSourceLabel="Benchmark"
    returnHaircutS={0.5}
    onReturnHaircutSChange={() => {}}
    categoryPriors={{}}
    onCategoryPriorsChange={() => {}}
    rawHistoricalCagrs={{}}
    fatTails={false}
    onFatTailsChange={() => {}}
  />
);
if (!leftPanelHtml.includes('Monthly Allocation')) {
  throw new Error('LeftPanel failed to render monthly allocation');
}

renderToString(
  <RightPanel
    monteCarloResult={null}
    projectionResult={null}
    mode="sip"
    allocation={defaultWeights}
    years={10}
    inflationRate={0.06}
    isSimulating={false}
  />
);

const optHtml = renderToString(
  <OptimizerTab
    totalAmount={3000000}
    monthlyAmount={25000}
    horizonYears={10}
    currentAllocation={defaultAllocationPercent}
    assumptions={defaultAssumptions}
    covarianceMatrix={[[0.04, 0.01], [0.01, 0.03]]}
    correlationMatrix={[[1, 0.2], [0.2, 1]]}
    assetIds={['nifty_50', 'gold']}
    bondConfig={bondConfig}
    inflationRate={0.06}
    onApplyAllocation={() => {}}
  />
);
if (!optHtml.includes('Optimizer Constraints &amp; Targets')) {
  throw new Error('OptimizerTab failed to render');
}

const labHtml = renderToString(
  <StrategyLabTab
    totalAmount={3000000}
    monthlySip={25000}
    horizonYears={10}
    mode="sip"
    allocation={defaultWeights}
    userAllocation={defaultWeights}
    userAllocationInr={defaultAllocationInr}
    assumptions={defaultAssumptions}
    inflationRate={0.06}
    bondConfig={bondConfig}
    constraints={{ maxRisk: 0.18, maxPerAsset: 0.35, minPerAsset: 0, riskFreeRate: 0.065 }}
    maxRisk={18}
    maxPerAsset={35}
    haircutSettings={{ returnHaircutS: 0.5, categoryPriors: {}, rawHistoricalCagrs: {} }}
    isFallback={true}
    onApplyAllocationInr={() => {}}
    onApplyAllocation={() => {}}
  />
);
if (!labHtml.includes('Strategy Lab: Out-of-Sample Quantitative Benchmarking')) {
  throw new Error('StrategyLabTab failed to render');
}

console.log('\n=============================================');
console.log('ALL RENDER & SMOKE CHECKS PASSED SUCCESSFULLY!');
console.log('=============================================');
process.exit(0);
