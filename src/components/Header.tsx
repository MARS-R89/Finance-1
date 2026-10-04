/**
 * Top Navigation Header
 * Features View Switcher (Simulator vs Optimizer vs Strategy Lab) and simulation triggers.
 */
import React from 'react';
import { RefreshCw, Play, Sliders, LineChart, FlaskConical, Activity } from 'lucide-react';

export type ActiveView = 'simulator' | 'optimizer' | 'strategylab' | 'dataregimes';

interface HeaderProps {
  activeView: ActiveView;
  onViewChange: (view: ActiveView) => void;
  onRunSimulation: () => void;
  isSimulating: boolean;
  canSimulate: boolean;
  onApplyPreset: (preset: 'balanced' | 'aggressive' | 'conservative' | 'equal') => void;
  seed?: number;
}

export const Header: React.FC<HeaderProps> = ({
  activeView,
  onViewChange,
  onRunSimulation,
  isSimulating,
  canSimulate,
  onApplyPreset,
  seed,
}) => {
  return (
    <header className="border-b border-neutral-800 bg-neutral-950/95 sticky top-0 z-30 backdrop-blur">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between gap-4">
        {/* Zone 1: Brand title wordmark */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 font-bold text-sm">
            ₹
          </div>
          <div>
            <span className="text-base sm:text-lg font-bold tracking-tight text-white block">
              Portfolio Simulator
            </span>
            <span className="text-[11px] text-neutral-400 block -mt-0.5">
              Correlated Monte Carlo &amp; Markowitz Engine
            </span>
          </div>
        </div>

        {/* Zone 2: Main View Switcher (Simulator / Optimizer / Strategy Lab) */}
        <div className="flex items-center gap-1.5 p-1 bg-neutral-900 border border-neutral-800 rounded-lg">
          <button
            onClick={() => onViewChange('simulator')}
            className={`px-3.5 py-1.5 text-xs font-semibold rounded-md flex items-center gap-1.5 transition cursor-pointer ${
              activeView === 'simulator'
                ? 'bg-neutral-800 text-emerald-400 shadow-sm'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            <LineChart className="w-3.5 h-3.5" />
            <span>Simulator</span>
          </button>
          <button
            onClick={() => onViewChange('optimizer')}
            className={`px-3.5 py-1.5 text-xs font-semibold rounded-md flex items-center gap-1.5 transition cursor-pointer ${
              activeView === 'optimizer'
                ? 'bg-neutral-800 text-emerald-400 shadow-sm'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Optimizer &amp; Frontier</span>
          </button>
          <button
            onClick={() => onViewChange('strategylab')}
            className={`px-3.5 py-1.5 text-xs font-semibold rounded-md flex items-center gap-1.5 transition cursor-pointer ${
              activeView === 'strategylab'
                ? 'bg-neutral-800 text-emerald-400 shadow-sm'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            <FlaskConical className="w-3.5 h-3.5" />
            <span>Strategy Lab</span>
          </button>
          <button
            onClick={() => onViewChange('dataregimes')}
            className={`px-3.5 py-1.5 text-xs font-semibold rounded-md flex items-center gap-1.5 transition cursor-pointer ${
              activeView === 'dataregimes'
                ? 'bg-neutral-800 text-emerald-400 shadow-sm'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Data &amp; Regimes</span>
          </button>
        </div>

        {/* Zone 3: Actions & Presets */}
        <div className="flex items-center gap-3">
          {activeView === 'simulator' ? (
            <>
              <div className="hidden lg:flex items-center gap-1.5 text-xs">
                <span className="text-neutral-500 mr-1 text-[11px]">Presets:</span>
                <button
                  onClick={() => onApplyPreset('balanced')}
                  className="px-2 py-1 rounded bg-neutral-900 hover:bg-neutral-800 text-neutral-300 border border-neutral-800 transition cursor-pointer text-[11px]"
                >
                  Balanced
                </button>
                <button
                  onClick={() => onApplyPreset('aggressive')}
                  className="px-2 py-1 rounded bg-neutral-900 hover:bg-neutral-800 text-neutral-300 border border-neutral-800 transition cursor-pointer text-[11px]"
                >
                  Aggressive
                </button>
                <button
                  onClick={() => onApplyPreset('conservative')}
                  className="px-2 py-1 rounded bg-neutral-900 hover:bg-neutral-800 text-neutral-300 border border-neutral-800 transition cursor-pointer text-[11px]"
                >
                  Debt/Gold
                </button>
              </div>
              {seed !== undefined && (
                <span className="text-[10px] font-mono text-neutral-500 bg-neutral-900 border border-neutral-800 px-2 py-1 rounded hidden sm:inline" title="Seeded PRNG value for reproducible simulations">
                  seed: {seed}
                </span>
              )}
              <button
                onClick={onRunSimulation}
                disabled={isSimulating || !canSimulate}
                className={`px-3.5 py-1.5 text-xs font-semibold rounded-lg flex items-center gap-1.5 transition cursor-pointer ${
                  isSimulating
                    ? 'bg-neutral-800 text-neutral-400 cursor-not-allowed'
                    : canSimulate
                    ? 'bg-emerald-500 hover:bg-emerald-400 text-neutral-950 shadow-sm shadow-emerald-500/20 active:scale-[0.98]'
                    : 'bg-neutral-800 text-neutral-500 cursor-not-allowed'
                }`}
              >
                {isSimulating ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Simulating...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>Run Simulation</span>
                  </>
                )}
              </button>
            </>
          ) : activeView === 'optimizer' ? (
            <span className="text-xs text-neutral-400 hidden sm:inline font-mono">
              Markowitz Mean-Variance
            </span>
          ) : (
            <span className="text-xs text-neutral-400 hidden sm:inline font-mono">
              11 Allocation Algorithms &amp; Backtest
            </span>
          )}
        </div>
      </div>
    </header>
  );
};
