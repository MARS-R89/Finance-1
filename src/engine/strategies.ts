/**
 * Portfolio Strategy Lab - Allocation Engine
 * Pure mathematical functions for multi-strategy asset allocation.
 * Strategies:
 * 1. equalWeight: 1/N baseline, capped and re-projected
 * 2. minVariance: Minimum variance portfolio
 * 3. maxSharpe: Maximum Sharpe ratio portfolio
 * 4. maxReturnAtRisk: Maximum return subject to volatility constraint
 * 5. riskParity: Equal Risk Contribution (ERC) via cyclical coordinate descent / projected gradient
 * 6. hierarchicalRiskParity: HRP (Marcos Lopez de Prado 2016)
 * 7. maxDiversification: Maximum Diversification Ratio (Choueifaty & Coignard 2008)
 * 8. blackLitterman: Equilibrium returns + optional subjective views (Black & Litterman 1992)
 * 9. resampledEfficiency: Michaud (1998) resampled frontier
 * 10. cvarOptimizer: Rockafellar-Uryasev CVaR(95) minimization via projected subgradient descent
 * 11. globalSearch: Differential Evolution optimizer cross-check
 */
import { Matrix, inverse } from 'ml-matrix';
import {
  OptimizableAsset,
  OptimizerConstraints,
  OptimizerPoint,
  Strategy,
  StrategyInput,
  StrategyOutput,
  StrategyRegistry,
  BLView,
} from './types.ts';
import { createMulberry32 } from './prng.ts';
import {
  projectOntoBoundedSimplex,
  computePortfolioReturn,
  computePortfolioVariance,
  findMinimumVariancePortfolio,
  maximizeSharpe,
  maximizeReturn,
  estimateMaxEigenvalue,
} from './optimizer.ts';
import { choleskyDecomposition } from './monteCarlo.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Rounds weights to 4 decimal places, distributes the rounding residual
 * so that weights sum strictly to 1.0 (|sum - 1| < 1e-9) and satisfy caps.
 */
export function cleanAndFinalizeWeights(
  weights: Record<string, number> | number[],
  assetIds: string[],
  maxPerAsset = 1.0,
  minPerAsset = 0.0
): Record<string, number> {
  const n = assetIds.length;
  const raw: number[] = Array.isArray(weights)
    ? weights.slice()
    : assetIds.map((id) => (weights as Record<string, number>)[id] ?? 0);

  // 1. Initial 4-decimal rounding
  const rounded = raw.map((w) => Math.round(Math.max(0, w) * 10000) / 10000);
  let sum = rounded.reduce((a, b) => a + b, 0);
  let residual = Math.round((1.0 - sum) * 10000) / 10000;

  // 2. Distribute residual in increments of 0.0001
  const step = residual > 0 ? 0.0001 : -0.0001;
  const stepsCount = Math.round(Math.abs(residual) * 10000);

  // Order indices by weight descending
  const sortedIndices = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => rounded[b] - rounded[a]
  );

  for (let s = 0; s < stepsCount; s++) {
    let placed = false;
    if (residual > 0) {
      // Put on largest weight where adding step doesn't violate cap
      for (const idx of sortedIndices) {
        if (rounded[idx] + step <= maxPerAsset + 1e-6) {
          rounded[idx] = Math.round((rounded[idx] + step) * 10000) / 10000;
          placed = true;
          break;
        }
      }
      if (!placed) {
        rounded[sortedIndices[0]] = Math.round((rounded[sortedIndices[0]] + step) * 10000) / 10000;
      }
    } else {
      // Negative residual: subtract from largest weight that stays >= minPerAsset
      for (const idx of sortedIndices) {
        if (rounded[idx] + step >= minPerAsset - 1e-6) {
          rounded[idx] = Math.round((rounded[idx] + step) * 10000) / 10000;
          placed = true;
          break;
        }
      }
      if (!placed) {
        rounded[sortedIndices[0]] = Math.round((rounded[sortedIndices[0]] + step) * 10000) / 10000;
      }
    }
  }

  const result: Record<string, number> = {};
  for (let i = 0; i < n; i++) {
    result[assetIds[i]] = rounded[i];
  }
  return result;
}

function makeResult(
  strategyId: string,
  strategyName: string,
  weightsArray: number[],
  assets: OptimizableAsset[],
  covMat: Matrix,
  riskFreeRate: number,
  maxPerAsset = 1.0,
  minPerAsset = 0.0
): StrategyOutput {
  const assetIds = assets.map((a) => a.id);
  const weightsMap = cleanAndFinalizeWeights(weightsArray, assetIds, maxPerAsset, minPerAsset);
  const cleanArray = assetIds.map((id) => weightsMap[id]);
  const returns = assets.map((a) => a.expectedReturn);
  const ret = computePortfolioReturn(cleanArray, returns);
  const vol = Math.sqrt(computePortfolioVariance(cleanArray, covMat));
  const sharpe = (ret - riskFreeRate) / Math.max(1e-4, vol);

  return {
    strategyId,
    strategyName,
    weights: weightsMap,
    expectedReturn: Number(ret.toFixed(4)),
    volatility: Number(vol.toFixed(4)),
    sharpeRatio: Number(sharpe.toFixed(4)),
  };
}

// ---------------------------------------------------------------------------
// Strategy 1: Equal Weight (1/N)
// ---------------------------------------------------------------------------
export const equalWeightStrategy: Strategy = ({ assets, cov, constraints }) => {
  const n = assets.length;
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const rf = constraints.riskFreeRate ?? 0.065;
  const covMat = new Matrix(cov);

  const raw = new Array(n).fill(1 / n);
  const projected = projectOntoBoundedSimplex(raw, maxPerAsset, minPerAsset);

  return makeResult('equal_weight', 'Equal Weight (1/N)', projected, assets, covMat, rf, maxPerAsset, minPerAsset);
};

// ---------------------------------------------------------------------------
// Strategy 2: Minimum Variance
// ---------------------------------------------------------------------------
export const minVarianceStrategy: Strategy = ({ assets, cov, constraints }) => {
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const rf = constraints.riskFreeRate ?? 0.065;
  const point = findMinimumVariancePortfolio(assets, cov, maxPerAsset, minPerAsset);
  const assetIds = assets.map((a) => a.id);
  const cleanWeights = cleanAndFinalizeWeights(point.weights, assetIds, maxPerAsset, minPerAsset);
  const wArr = assetIds.map((id) => cleanWeights[id]);
  const covMat = new Matrix(cov);
  const ret = computePortfolioReturn(wArr, assets.map((a) => a.expectedReturn));
  const vol = Math.sqrt(computePortfolioVariance(wArr, covMat));
  const sharpe = (ret - rf) / Math.max(1e-4, vol);

  return {
    ...point,
    weights: cleanWeights,
    expectedReturn: Number(ret.toFixed(4)),
    volatility: Number(vol.toFixed(4)),
    sharpeRatio: Number(sharpe.toFixed(4)),
    strategyId: 'min_variance',
    strategyName: 'Minimum Variance',
  };
};

// ---------------------------------------------------------------------------
// Strategy 3: Maximum Sharpe Ratio
// ---------------------------------------------------------------------------
export const maxSharpeStrategy: Strategy = ({ assets, cov, constraints }) => {
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const rf = constraints.riskFreeRate ?? 0.065;
  const point = maximizeSharpe(assets, cov, constraints);
  const assetIds = assets.map((a) => a.id);
  const cleanWeights = cleanAndFinalizeWeights(point.weights, assetIds, maxPerAsset, minPerAsset);
  const wArr = assetIds.map((id) => cleanWeights[id]);
  const covMat = new Matrix(cov);
  const ret = computePortfolioReturn(wArr, assets.map((a) => a.expectedReturn));
  const vol = Math.sqrt(computePortfolioVariance(wArr, covMat));
  const sharpe = (ret - rf) / Math.max(1e-4, vol);

  return {
    ...point,
    weights: cleanWeights,
    expectedReturn: Number(ret.toFixed(4)),
    volatility: Number(vol.toFixed(4)),
    sharpeRatio: Number(sharpe.toFixed(4)),
    strategyId: 'max_sharpe',
    strategyName: 'Maximum Sharpe Ratio',
  };
};

// ---------------------------------------------------------------------------
// Strategy 4: Maximum Return (Bounded Risk)
// ---------------------------------------------------------------------------
export const maxReturnAtRiskStrategy: Strategy = ({ assets, cov, constraints }) => {
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const rf = constraints.riskFreeRate ?? 0.065;
  const point = maximizeReturn(assets, cov, constraints);
  const assetIds = assets.map((a) => a.id);
  const cleanWeights = cleanAndFinalizeWeights(point.weights, assetIds, maxPerAsset, minPerAsset);
  const wArr = assetIds.map((id) => cleanWeights[id]);
  const covMat = new Matrix(cov);
  const ret = computePortfolioReturn(wArr, assets.map((a) => a.expectedReturn));
  const vol = Math.sqrt(computePortfolioVariance(wArr, covMat));
  const sharpe = (ret - rf) / Math.max(1e-4, vol);

  return {
    ...point,
    weights: cleanWeights,
    expectedReturn: Number(ret.toFixed(4)),
    volatility: Number(vol.toFixed(4)),
    sharpeRatio: Number(sharpe.toFixed(4)),
    strategyId: 'max_return',
    strategyName: 'Max Return at Risk',
  };
};

// ---------------------------------------------------------------------------
// Strategy 5: Risk Parity / Equal Risk Contribution (ERC)
// ---------------------------------------------------------------------------
export const riskParityStrategy: Strategy = ({ assets, cov, constraints }) => {
  const n = assets.length;
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const rf = constraints.riskFreeRate ?? 0.065;
  const covMat = new Matrix(cov);

  // Initial positive vector y
  let y = assets.map((a) => 1 / Math.max(1e-4, a.volatility));
  const targetPerAsset = 1 / n;

  // Cyclical coordinate descent on Spinu objective: 1/2 y' Sigma y - (1/N) sum ln(y_i)
  for (let iter = 0; iter < 40; iter++) {
    for (let i = 0; i < n; i++) {
      let b_i = 0;
      for (let j = 0; j < n; j++) {
        if (j !== i) b_i += cov[i][j] * y[j];
      }
      const a_i = cov[i][i];
      const disc = Math.sqrt(b_i * b_i + 4 * a_i * targetPerAsset);
      y[i] = (-b_i + disc) / (2 * a_i);
    }
  }

  // Normalize y
  const ySum = y.reduce((a, b) => a + b, 0);
  let w = y.map((v) => v / ySum);

  w = projectOntoBoundedSimplex(w, maxPerAsset, minPerAsset);

  // Fine-tuning projected gradient descent on sum_i (RC_i - sigma_p / n)^2
  const stepSize = 0.01;
  for (let iter = 0; iter < 100; iter++) {
    const sigmaP2 = computePortfolioVariance(w, covMat);
    const sigmaP = Math.sqrt(sigmaP2);
    const covW = covMat.mmul(Matrix.columnVector(w)).to1DArray();
    const rc = new Array(n);
    const target = sigmaP / n;
    for (let i = 0; i < n; i++) {
      rc[i] = (w[i] * covW[i]) / Math.max(1e-6, sigmaP);
    }
    const grad = new Array(n).fill(0);
    for (let k = 0; k < n; k++) {
      let sumK = 0;
      for (let i = 0; i < n; i++) {
        const dRCi_dwk =
          ((i === k ? covW[i] : 0) + w[i] * cov[i][k]) / sigmaP -
          (w[i] * covW[i] * covW[k]) / (sigmaP * sigmaP2);
        sumK += 2 * (rc[i] - target) * dRCi_dwk;
      }
      grad[k] = sumK;
    }
    const nextV = new Array(n);
    for (let i = 0; i < n; i++) nextV[i] = w[i] - stepSize * grad[i];
    const nextW = projectOntoBoundedSimplex(nextV, maxPerAsset, minPerAsset);
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(nextW[i] - w[i]);
    w = nextW;
    if (diff < 1e-6) break;
  }

  return makeResult('risk_parity', 'Risk Parity (ERC)', w, assets, covMat, rf, maxPerAsset, minPerAsset);
};

// ---------------------------------------------------------------------------
// Strategy 6: Hierarchical Risk Parity (HRP)
// ---------------------------------------------------------------------------
interface DendrogramNode {
  left?: DendrogramNode;
  right?: DendrogramNode;
  items: number[];
  distance: number;
}

export const hierarchicalRiskParityStrategy: Strategy = ({ assets, cov, corr, constraints }) => {
  const n = assets.length;
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const rf = constraints.riskFreeRate ?? 0.065;
  const covMat = new Matrix(cov);

  if (n === 1) {
    return makeResult('hrp', 'Hierarchical Risk Parity', [1.0], assets, covMat, rf, maxPerAsset, minPerAsset);
  }

  // 1. Correlation distance matrix: d_ij = sqrt(0.5 * (1 - rho_ij))
  const D: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        D[i][j] = 0;
      } else {
        const rho = corr?.[i]?.[j] ?? (cov[i][j] / Math.sqrt(Math.max(1e-8, cov[i][i] * cov[j][j])));
        D[i][j] = Math.sqrt(Math.max(0, 0.5 * (1 - rho)));
      }
    }
  }

  // 2. Single-linkage clustering
  const clusterDist = (c1: number[], c2: number[]): number => {
    let minDist = Infinity;
    for (const i of c1) {
      for (const j of c2) {
        if (D[i][j] < minDist) minDist = D[i][j];
      }
    }
    return minDist;
  };

  let activeClusters: DendrogramNode[] = Array.from({ length: n }, (_, i) => ({
    items: [i],
    distance: 0,
  }));

  while (activeClusters.length > 1) {
    let bestDist = Infinity;
    let bestI = 0;
    let bestJ = 1;
    for (let i = 0; i < activeClusters.length; i++) {
      for (let j = i + 1; j < activeClusters.length; j++) {
        const dist = clusterDist(activeClusters[i].items, activeClusters[j].items);
        if (dist < bestDist) {
          bestDist = dist;
          bestI = i;
          bestJ = j;
        }
      }
    }

    const mergedNode: DendrogramNode = {
      left: activeClusters[bestI],
      right: activeClusters[bestJ],
      items: [...activeClusters[bestI].items, ...activeClusters[bestJ].items],
      distance: bestDist,
    };

    activeClusters = activeClusters.filter((_, idx) => idx !== bestI && idx !== bestJ);
    activeClusters.push(mergedNode);
  }

  // 3. Quasi-diagonalization: leaf traversal order
  const orderedIndices: number[] = [];
  const traverse = (node: DendrogramNode) => {
    if (!node.left && !node.right) {
      orderedIndices.push(...node.items);
      return;
    }
    if (node.left) traverse(node.left);
    if (node.right) traverse(node.right);
  };
  traverse(activeClusters[0]);

  // 4. Recursive bisection
  const weights = new Array(n).fill(1.0);
  const getClusterVariance = (cluster: number[]): number => {
    if (cluster.length === 1) return cov[cluster[0]][cluster[0]];
    let invSum = 0;
    const invVar: number[] = [];
    for (const idx of cluster) {
      const v = 1 / Math.max(1e-6, cov[idx][idx]);
      invVar.push(v);
      invSum += v;
    }
    const wSub = invVar.map((v) => v / invSum);
    let varSub = 0;
    for (let i = 0; i < cluster.length; i++) {
      for (let j = 0; j < cluster.length; j++) {
        varSub += wSub[i] * wSub[j] * cov[cluster[i]][cluster[j]];
      }
    }
    return Math.max(1e-6, varSub);
  };

  const recurseBisection = (items: number[]) => {
    if (items.length <= 1) return;
    const mid = Math.floor(items.length / 2);
    const left = items.slice(0, mid);
    const right = items.slice(mid);
    const varLeft = getClusterVariance(left);
    const varRight = getClusterVariance(right);
    const alpha = varRight / (varLeft + varRight);
    for (const idx of left) weights[idx] *= alpha;
    for (const idx of right) weights[idx] *= 1 - alpha;
    recurseBisection(left);
    recurseBisection(right);
  };

  recurseBisection(orderedIndices);

  const totalW = weights.reduce((a, b) => a + b, 0);
  const normalized = weights.map((w) => w / totalW);
  const projected = projectOntoBoundedSimplex(normalized, maxPerAsset, minPerAsset);

  return makeResult('hrp', 'Hierarchical Risk Parity (HRP)', projected, assets, covMat, rf, maxPerAsset, minPerAsset);
};

// ---------------------------------------------------------------------------
// Strategy 7: Maximum Diversification
// ---------------------------------------------------------------------------
export const maxDiversificationStrategy: Strategy = ({ assets, cov, constraints, seed }) => {
  const n = assets.length;
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const rf = constraints.riskFreeRate ?? 0.065;
  const covMat = new Matrix(cov);
  const sigmas = assets.map((a) => a.volatility);

  let bestRatio = -Infinity;
  let bestWeights = projectOntoBoundedSimplex(new Array(n).fill(1 / n), maxPerAsset, minPerAsset);
  const rng = createMulberry32(seed ?? 42);

  for (let s = 0; s < 15; s++) {
    let w = new Array(n);
    if (s === 0) {
      w = new Array(n).fill(1 / n);
    } else {
      for (let i = 0; i < n; i++) w[i] = -Math.log(Math.max(1e-9, rng()));
    }
    w = projectOntoBoundedSimplex(w, maxPerAsset, minPerAsset);
    const stepSize = 0.03;
    for (let iter = 0; iter < 120; iter++) {
      let sumWeightedSigma = 0;
      for (let i = 0; i < n; i++) sumWeightedSigma += w[i] * sigmas[i];
      const pVar = computePortfolioVariance(w, covMat);
      const covW = covMat.mmul(Matrix.columnVector(w)).to1DArray();
      const grad = new Array(n);
      for (let i = 0; i < n; i++) {
        grad[i] = sigmas[i] / Math.max(1e-6, sumWeightedSigma) - covW[i] / Math.max(1e-6, pVar);
      }
      const nextV = new Array(n);
      for (let i = 0; i < n; i++) nextV[i] = w[i] + stepSize * grad[i];
      const nextW = projectOntoBoundedSimplex(nextV, maxPerAsset, minPerAsset);
      let diff = 0;
      for (let i = 0; i < n; i++) diff += Math.abs(nextW[i] - w[i]);
      w = nextW;
      if (diff < 1e-6) break;
    }
    let wSigma = 0;
    for (let i = 0; i < n; i++) wSigma += w[i] * sigmas[i];
    const wVol = Math.sqrt(computePortfolioVariance(w, covMat));
    const dr = wSigma / Math.max(1e-6, wVol);
    if (dr > bestRatio) {
      bestRatio = dr;
      bestWeights = w;
    }
  }

  return makeResult('max_diversification', 'Maximum Diversification', bestWeights, assets, covMat, rf, maxPerAsset, minPerAsset);
};

// ---------------------------------------------------------------------------
// Strategy 8: Black-Litterman
// ---------------------------------------------------------------------------
export const blackLittermanStrategy: Strategy = ({ assets, cov, constraints, views }) => {
  const n = assets.length;
  const delta = 2.5; // Risk aversion coefficient
  const tau = 0.05; // Uncertainty scaling parameter
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const rf = constraints.riskFreeRate ?? 0.065;
  const covMat = new Matrix(cov);

  // Equal-weighted market portfolio unless specified
  const wMkt = Matrix.columnVector(new Array(n).fill(1 / n));

  // Equilibrium excess returns: Pi = delta * Sigma * w_mkt
  const piMat = covMat.mmul(wMkt).mul(delta);
  const pi = piMat.to1DArray();
  let posteriorReturns = pi;

  // Process user views if provided
  const activeViews = views && views.length > 0 ? views : [];
  if (activeViews.length > 0) {
    const k = activeViews.length;
    const P = Matrix.zeros(k, n);
    const Q = Matrix.zeros(k, 1);
    const Omega = Matrix.zeros(k, k);

    for (let i = 0; i < k; i++) {
      const v = activeViews[i];
      const assetIdx = assets.findIndex((a) => a.id === v.assetId);
      if (assetIdx >= 0) {
        P.set(i, assetIdx, 1.0);
        Q.set(i, 0, v.expectedReturn);
        const pRow = P.subMatrix(i, i, 0, n - 1);
        const pSigmaP = pRow.mmul(covMat).mmul(pRow.transpose()).get(0, 0);
        const conf = Math.max(0.01, Math.min(0.99, v.confidence));
        const omegaVal = tau * pSigmaP * ((1 - conf) / conf);
        Omega.set(i, i, Math.max(1e-6, omegaVal));
      }
    }

    try {
      const tauSigma = covMat.clone().mul(tau);
      const PtauSigmaPT = P.mmul(tauSigma).mmul(P.transpose());
      const midMat = PtauSigmaPT.add(Omega);
      const midInv = inverse(midMat);
      const PPi = P.mmul(Matrix.columnVector(pi));
      const QminusPPi = Q.sub(PPi);
      const shift = tauSigma.mmul(P.transpose()).mmul(midInv).mmul(QminusPPi);
      const postMat = Matrix.columnVector(pi).add(shift);
      posteriorReturns = postMat.to1DArray();
    } catch {
      posteriorReturns = pi;
    }
  }

  // Mean-variance utility optimization: max w'mu_post - (delta/2) w'Sigma w on bounded simplex
  const maxEig = estimateMaxEigenvalue(covMat, n);
  const stepSize = 1.0 / Math.max(1e-4, delta * maxEig);

  let w = projectOntoBoundedSimplex(new Array(n).fill(1 / n), maxPerAsset, minPerAsset);
  for (let iter = 0; iter < 300; iter++) {
    const covW = covMat.mmul(Matrix.columnVector(w)).to1DArray();
    const candidate = new Array(n);
    for (let i = 0; i < n; i++) {
      candidate[i] = w[i] + stepSize * (posteriorReturns[i] - delta * covW[i]);
    }
    const nextW = projectOntoBoundedSimplex(candidate, maxPerAsset, minPerAsset);
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(nextW[i] - w[i]);
    w = nextW;
    if (diff < 1e-7) break;
  }

  const assetIds = assets.map((a) => a.id);
  const cleanWeights = cleanAndFinalizeWeights(w, assetIds, maxPerAsset, minPerAsset);
  const cleanWArr = assetIds.map((id) => cleanWeights[id]);

  // Requirement 4 Fix: Compute expectedReturn using user's assumption CAGR (w' * assetStats cagr)
  // so strategies are comparable, and report extra fields equilibriumReturn and posteriorReturn
  const assumptionReturns = assets.map((a) => a.expectedReturn);
  const userAssumptionReturn = computePortfolioReturn(cleanWArr, assumptionReturns);
  const vol = Math.sqrt(computePortfolioVariance(cleanWArr, covMat));
  const sharpe = (userAssumptionReturn - rf) / Math.max(1e-4, vol);

  const equilibriumReturn = computePortfolioReturn(cleanWArr, pi) + rf;
  const posteriorReturn = computePortfolioReturn(cleanWArr, posteriorReturns) + rf;

  return {
    strategyId: 'black_litterman',
    strategyName: 'Black-Litterman',
    weights: cleanWeights,
    expectedReturn: Number(userAssumptionReturn.toFixed(4)),
    volatility: Number(vol.toFixed(4)),
    sharpeRatio: Number(sharpe.toFixed(4)),
    equilibriumReturn: Number(equilibriumReturn.toFixed(4)),
    posteriorReturn: Number(posteriorReturn.toFixed(4)),
  };
};

// ---------------------------------------------------------------------------
// Strategy 9: Resampled Efficiency (Michaud 1998)
// ---------------------------------------------------------------------------
export const resampledEfficiencyStrategy: Strategy = ({
  assets,
  cov,
  constraints,
  seed,
  nSamples: propNSamples,
  cachedCholesky,
}) => {
  const n = assets.length;
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const rf = constraints.riskFreeRate ?? 0.065;
  const covMat = new Matrix(cov);
  const nSamples = propNSamples ?? 200;
  const T = 60;
  const L = cachedCholesky ?? choleskyDecomposition(cov);
  const rng = createMulberry32(seed ?? 42);

  const sampleGaussian = (): number => {
    let u1 = rng();
    while (u1 < 1e-12) u1 = rng();
    const u2 = rng();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  };

  const weightAccum = new Array(n).fill(0);
  for (let s = 0; s < nSamples; s++) {
    const sampleReturns: number[][] = [];
    for (let t = 0; t < T; t++) {
      const z = Array.from({ length: n }, () => sampleGaussian());
      const row = new Array(n);
      for (let i = 0; i < n; i++) {
        let shock = 0;
        for (let j = 0; j <= i; j++) shock += L[i][j] * z[j];
        row[i] = assets[i].expectedReturn / 12 + shock / Math.sqrt(12);
      }
      sampleReturns.push(row);
    }

    const sampleMean = new Array(n).fill(0);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < n; i++) sampleMean[i] += sampleReturns[t][i];
    }
    for (let i = 0; i < n; i++) sampleMean[i] = (sampleMean[i] / T) * 12;

    const sampleCov: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let t = 0; t < T; t++) {
          sum += (sampleReturns[t][i] - sampleMean[i] / 12) * (sampleReturns[t][j] - sampleMean[j] / 12);
        }
        sampleCov[i][j] = (sum / (T - 1)) * 12;
        if (i === j) sampleCov[i][j] += 1e-5;
      }
    }
    const sampleCovMat = new Matrix(sampleCov);

    let wDraw = projectOntoBoundedSimplex(new Array(n).fill(1 / n), maxPerAsset, minPerAsset);
    const stepSize = 0.04;
    for (let iter = 0; iter < 40; iter++) {
      const pRet = computePortfolioReturn(wDraw, sampleMean);
      const pVar = Math.max(1e-6, computePortfolioVariance(wDraw, sampleCovMat));
      const pVol = Math.sqrt(pVar);
      const excess = pRet - rf;
      const covW = sampleCovMat.mmul(Matrix.columnVector(wDraw)).to1DArray();
      const grad = new Array(n);
      for (let i = 0; i < n; i++) {
        grad[i] = (sampleMean[i] * pVol - (excess * covW[i]) / pVol) / pVar;
      }
      const nextV = new Array(n);
      for (let i = 0; i < n; i++) nextV[i] = wDraw[i] + stepSize * grad[i];
      wDraw = projectOntoBoundedSimplex(nextV, maxPerAsset, minPerAsset);
    }

    for (let i = 0; i < n; i++) {
      weightAccum[i] += wDraw[i];
    }
  }

  const avgWeights = weightAccum.map((sum) => sum / nSamples);
  const projected = projectOntoBoundedSimplex(avgWeights, maxPerAsset, minPerAsset);

  return makeResult('resampled_efficiency', 'Resampled Efficiency (Michaud)', projected, assets, covMat, rf, maxPerAsset, minPerAsset);
};

// ---------------------------------------------------------------------------
// Strategy 10: CVaR Optimizer (Rockafellar-Uryasev 95%)
// ---------------------------------------------------------------------------
export const cvarOptimizerStrategy: Strategy = ({
  assets,
  cov,
  constraints,
  seed,
  nScenarios: propNScenarios,
  cachedCholesky,
  isWalkForward,
}) => {
  const n = assets.length;
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const rf = constraints.riskFreeRate ?? 0.065;
  const covMat = new Matrix(cov);
  const nScenarios = propNScenarios ?? (isWalkForward ? 500 : 2000);
  const alpha = 0.95;
  const L = cachedCholesky ?? choleskyDecomposition(cov);
  const rng = createMulberry32(seed ?? 42);

  const sampleGaussian = (): number => {
    let u1 = rng();
    while (u1 < 1e-12) u1 = rng();
    const u2 = rng();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  };

  const scenarios: number[][] = [];
  for (let s = 0; s < nScenarios; s++) {
    const z = Array.from({ length: n }, () => sampleGaussian());
    const scenario = new Array(n);
    for (let i = 0; i < n; i++) {
      let shock = 0;
      for (let j = 0; j <= i; j++) shock += L[i][j] * z[j];
      scenario[i] = assets[i].expectedReturn + shock;
    }
    scenarios.push(scenario);
  }

  const tailCount = Math.max(1, Math.floor((1 - alpha) * nScenarios));
  const lossArray = new Float64Array(nScenarios);
  const indices = new Int32Array(nScenarios);
  for (let i = 0; i < nScenarios; i++) indices[i] = i;

  const computeCVaR = (weights: number[]): { cvar: number; subgrad: number[] } => {
    for (let s = 0; s < nScenarios; s++) {
      let r = 0;
      for (let i = 0; i < n; i++) r += weights[i] * scenarios[s][i];
      lossArray[s] = -r;
    }
    indices.sort((a, b) => lossArray[b] - lossArray[a]);
    let cvarVal = 0;
    const subgrad = new Array(n).fill(0);
    for (let k = 0; k < tailCount; k++) {
      const idx = indices[k];
      cvarVal += lossArray[idx];
      const scen = scenarios[idx];
      for (let i = 0; i < n; i++) {
        subgrad[i] += -scen[i];
      }
    }
    cvarVal /= tailCount;
    for (let i = 0; i < n; i++) subgrad[i] /= tailCount;
    return { cvar: cvarVal, subgrad };
  };

  const minVarPt = findMinimumVariancePortfolio(assets, cov, maxPerAsset, minPerAsset);
  const minVarW = assets.map((a) => minVarPt.weights[a.id] || 0);

  const invVolRaw = assets.map((a) => 1 / Math.max(0.01, a.volatility));
  const invVolSum = invVolRaw.reduce((a, b) => a + b, 0);
  const invVolW = projectOntoBoundedSimplex(invVolRaw.map((v) => v / invVolSum), maxPerAsset, minPerAsset);

  const equalW = projectOntoBoundedSimplex(new Array(n).fill(1 / n), maxPerAsset, minPerAsset);

  const candidateStarts = [minVarW, invVolW, equalW];
  let bestW = minVarW.slice();
  let minCVaR = Infinity;
  const maxIters = isWalkForward ? 40 : 80;

  for (const startW of candidateStarts) {
    let w = startW.slice();
    for (let iter = 1; iter <= maxIters; iter++) {
      const { cvar, subgrad } = computeCVaR(w);
      if (cvar < minCVaR) {
        minCVaR = cvar;
        bestW = w.slice();
      }
      const eta = 0.15 / Math.sqrt(iter);
      const candidate = new Array(n);
      for (let i = 0; i < n; i++) candidate[i] = w[i] - eta * subgrad[i];
      w = projectOntoBoundedSimplex(candidate, maxPerAsset, minPerAsset);
    }
  }

  return makeResult('cvar_optimizer', 'CVaR Optimizer (95%)', bestW, assets, covMat, rf, maxPerAsset, minPerAsset);
};

// ---------------------------------------------------------------------------
// Strategy 11: Global Search (Differential Evolution)
// ---------------------------------------------------------------------------
export const globalSearchStrategy: Strategy = ({ assets, cov, constraints, seed, isWalkForward }) => {
  const n = assets.length;
  const maxRisk = constraints.maxRisk ?? 0.18;
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const rf = constraints.riskFreeRate ?? 0.065;
  const covMat = new Matrix(cov);
  const returns = assets.map((a) => a.expectedReturn);
  const popSize = isWalkForward ? 30 : 60;
  const maxGen = isWalkForward ? 40 : 120;
  const F = 0.7;
  const CR = 0.8;
  const rng = createMulberry32(seed ?? 42);

  const evaluate = (vec: number[]): { w: number[]; fitness: number } => {
    const w = projectOntoBoundedSimplex(vec, maxPerAsset, minPerAsset);
    const pRet = computePortfolioReturn(w, returns);
    const pVol = Math.sqrt(computePortfolioVariance(w, covMat));
    let penalty = 0;
    if (pVol > maxRisk) {
      const excess = pVol - maxRisk;
      penalty = 50 * excess * excess + 10 * excess;
    }
    const fitness = pRet - penalty;
    return { w, fitness };
  };

  let pop: number[][] = [];
  let popFitness: number[] = [];
  let bestCandidate = new Array(n).fill(1 / n);
  let bestFitness = -Infinity;

  for (let p = 0; p < popSize; p++) {
    const ind = Array.from({ length: n }, () => rng());
    const { w, fitness } = evaluate(ind);
    pop.push(ind);
    popFitness.push(fitness);
    if (fitness > bestFitness) {
      bestFitness = fitness;
      bestCandidate = w;
    }
  }

  for (let gen = 0; gen < maxGen; gen++) {
    for (let i = 0; i < popSize; i++) {
      let r1 = Math.floor(rng() * popSize);
      while (r1 === i) r1 = Math.floor(rng() * popSize);
      let r2 = Math.floor(rng() * popSize);
      while (r2 === i || r2 === r1) r2 = Math.floor(rng() * popSize);
      let r3 = Math.floor(rng() * popSize);
      while (r3 === i || r3 === r1 || r3 === r2) r3 = Math.floor(rng() * popSize);

      const mutant = new Array(n);
      const fixedDim = Math.floor(rng() * n);
      for (let d = 0; d < n; d++) {
        if (rng() < CR || d === fixedDim) {
          mutant[d] = pop[r1][d] + F * (pop[r2][d] - pop[r3][d]);
        } else {
          mutant[d] = pop[i][d];
        }
      }
      const trialEval = evaluate(mutant);
      if (trialEval.fitness >= popFitness[i]) {
        pop[i] = mutant;
        popFitness[i] = trialEval.fitness;
        if (trialEval.fitness > bestFitness) {
          bestFitness = trialEval.fitness;
          bestCandidate = trialEval.w;
        }
      }
    }
  }

  return makeResult('global_search', 'Global Search (DE)', bestCandidate, assets, covMat, rf, maxPerAsset, minPerAsset);
};

// ---------------------------------------------------------------------------
// Solver Comparison Tool
// ---------------------------------------------------------------------------
export interface SolverComparisonResult {
  gradientResult: OptimizerPoint;
  deResult: OptimizerPoint;
  returnGap: number;
  volGap: number;
  maxRisk: number;
}

export function compareSolvers(
  assets: OptimizableAsset[],
  cov: number[][],
  constraints: OptimizerConstraints,
  seed = 42
): SolverComparisonResult {
  const maxRisk = constraints.maxRisk ?? 0.18;
  const gradientResult = maximizeReturn(assets, cov, constraints);
  const deResult = globalSearchStrategy({
    assets,
    cov,
    corr: cov,
    constraints,
    seed,
  });

  return {
    gradientResult,
    deResult,
    returnGap: Math.abs(gradientResult.expectedReturn - deResult.expectedReturn),
    volGap: Math.abs(gradientResult.volatility - deResult.volatility),
    maxRisk,
  };
}

// ---------------------------------------------------------------------------
// Strategy Registry
// ---------------------------------------------------------------------------
export const STRATEGY_REGISTRY: StrategyRegistry = {
  equal_weight: {
    name: 'Equal Weight (1/N)',
    description: 'Equally balances portfolio weights across all selected assets, capped by constraints.',
    fn: equalWeightStrategy,
  },
  min_variance: {
    name: 'Minimum Variance',
    description: 'Minimizes overall portfolio variance using empirical covariance, focusing on lowest possible volatility.',
    fn: minVarianceStrategy,
  },
  max_sharpe: {
    name: 'Maximum Sharpe Ratio',
    description: 'Tangency portfolio maximizing excess return per unit of standard deviation under asset constraints.',
    fn: maxSharpeStrategy,
  },
  max_return: {
    name: 'Max Return at Risk',
    description: 'Maximizes expected portfolio return strictly bounded by the annual volatility ceiling.',
    fn: maxReturnAtRiskStrategy,
  },
  risk_parity: {
    name: 'Risk Parity (ERC)',
    description: 'Equal Risk Contribution: ensures each asset contributes an equal share of overall portfolio volatility.',
    fn: riskParityStrategy,
  },
  hrp: {
    name: 'Hierarchical Risk Parity (HRP)',
    description: 'Tree clustering and recursive bisection (Lopez de Prado) robust against matrix inversion instability.',
    fn: hierarchicalRiskParityStrategy,
  },
  max_diversification: {
    name: 'Maximum Diversification',
    description: 'Maximizes the Diversification Ratio (Choueifaty), capturing non-overlapping diversification benefits.',
    fn: maxDiversificationStrategy,
  },
  black_litterman: {
    name: 'Black-Litterman',
    description: 'Combines market equilibrium expected returns with investor views using Bayesian shrinkage.',
    fn: blackLittermanStrategy,
  },
  resampled_efficiency: {
    name: 'Resampled Efficiency',
    description: 'Michaud simulated resampling over 200 paths to reduce estimation error and corner solutions.',
    fn: resampledEfficiencyStrategy,
  },
  cvar_optimizer: {
    name: 'CVaR Optimizer (95%)',
    description: 'Directly minimizes Conditional Value-at-Risk at the 95% tail using 2,000 correlated scenarios.',
    fn: cvarOptimizerStrategy,
  },
  global_search: {
    name: 'Global Search (DE)',
    description: 'Differential Evolution stochastic population-based search verifying gradient optimizer solutions.',
    fn: globalSearchStrategy,
  },
};
