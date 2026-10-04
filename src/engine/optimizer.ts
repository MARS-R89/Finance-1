/**
 * Portfolio Optimization Engine (Markowitz Mean-Variance & Sharpe Optimization)
 * Uses ml-matrix for linear algebra and Projected Gradient Descent on the bounded simplex:
 *   w_i >= 0, w_i <= maxPerAsset, sum(w_i) = 1.
 * Guarantees all returned allocations strictly satisfy constraints.
 */
import { Matrix } from 'ml-matrix';
import { OptimizerConstraints, OptimizerPoint } from './types.ts';
import { createMulberry32 } from './prng.ts';

export interface OptimizableAsset {
  id: string;
  expectedReturn: number; // Annualized CAGR / expected return
  volatility: number; // Annualized standard deviation
}

/**
 * Projects an arbitrary vector v in R^N onto the bounded simplex:
 *   S_u = { w in R^N : sum(w) = 1, 0 <= w_i <= maxPerAsset }
 * Uses exact monotonic bisection on the Lagrange multiplier lambda:
 *   w_i(lambda) = min(maxPerAsset, max(0, v_i - lambda))
 */
export function projectOntoBoundedSimplex(
  v: number[],
  maxPerAsset: number,
  minPerAsset = 0
): number[] {
  const n = v.length;
  if (n === 0) return [];
  if (n === 1) return [1.0];

  // Ensure feasibility: n * maxPerAsset >= 1 and n * minPerAsset <= 1
  const effectiveMax = Math.max(1.0 / n, Math.min(1.0, maxPerAsset));
  const effectiveMin = Math.max(0.0, Math.min(minPerAsset, 1.0 / n));

  // The function phi(lambda) = sum( min(effectiveMax, max(effectiveMin, v_i - lambda)) ) is non-increasing.
  let low = Math.min(...v) - effectiveMax - 1;
  let high = Math.max(...v) - effectiveMin + 1;

  for (let iter = 0; iter < 60; iter++) {
    const mid = (low + high) / 2;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const clamped = Math.min(effectiveMax, Math.max(effectiveMin, v[i] - mid));
      sum += clamped;
    }
    if (Math.abs(sum - 1.0) < 1e-12) {
      low = mid;
      high = mid;
      break;
    }
    if (sum > 1.0) {
      // Need smaller values, so increase lambda
      low = mid;
    } else {
      high = mid;
    }
  }

  const optimalLambda = (low + high) / 2;
  const result: number[] = new Array(n);
  let finalSum = 0;
  for (let i = 0; i < n; i++) {
    result[i] = Math.min(effectiveMax, Math.max(effectiveMin, v[i] - optimalLambda));
    finalSum += result[i];
  }

  // Renormalize slightly if tiny floating point drift
  if (Math.abs(finalSum - 1.0) > 1e-9 && finalSum > 0) {
    for (let i = 0; i < n; i++) {
      result[i] /= finalSum;
    }
  }

  return result;
}

/**
 * Computes portfolio variance: w^T * Sigma * w using ml-matrix.
 */
export function computePortfolioVariance(weights: number[], covMatrix: Matrix): number {
  const w = Matrix.columnVector(weights);
  const covW = covMatrix.mmul(w);
  const variance = w.transpose().mmul(covW).get(0, 0);
  return Math.max(1e-12, variance);
}

/**
 * Computes portfolio return: w^T * mu
 */
export function computePortfolioReturn(weights: number[], returns: number[]): number {
  let r = 0;
  for (let i = 0; i < weights.length; i++) {
    r += weights[i] * returns[i];
  }
  return r;
}

/**
 * Computes gradient of portfolio variance: 2 * Sigma * w
 */
export function computeVarianceGradient(weights: number[], covMatrix: Matrix): number[] {
  const w = Matrix.columnVector(weights);
  const grad = covMatrix.mmul(w).mul(2);
  return grad.to1DArray();
}

/**
 * Estimates the maximum eigenvalue of the covariance matrix via power iteration.
 */
export function estimateMaxEigenvalue(covMat: Matrix, n: number): number {
  let v = new Array(n).fill(1 / Math.sqrt(n));
  for (let it = 0; it < 30; it++) {
    const Mv = covMat.mmul(Matrix.columnVector(v)).to1DArray();
    let norm = 0;
    for (let i = 0; i < n; i++) norm += Mv[i] * Mv[i];
    norm = Math.sqrt(norm);
    if (norm < 1e-12) return 1e-4;
    for (let i = 0; i < n; i++) v[i] = Mv[i] / norm;
  }
  const Mv = covMat.mmul(Matrix.columnVector(v)).to1DArray();
  let lambda = 0;
  for (let i = 0; i < n; i++) lambda += v[i] * Mv[i];
  return Math.max(1e-4, lambda);
}

/**
 * Generates deterministic initial weights strictly inside the bounded simplex.
 */
function sampleDeterministicFeasibleWeights(
  n: number,
  maxPerAsset: number,
  seedIndex: number
): number[] {
  if (seedIndex === 0) {
    // Equal weights
    return projectOntoBoundedSimplex(new Array(n).fill(1 / n), maxPerAsset);
  }
  if (seedIndex <= n) {
    // Single asset spike
    const spike = new Array(n).fill(0);
    spike[seedIndex - 1] = 1.0;
    return projectOntoBoundedSimplex(spike, maxPerAsset);
  }
  // Seeded pseudo-random Dirichlet distribution
  const rng = createMulberry32(10000 + seedIndex);
  const unconstrained = new Array(n);
  for (let i = 0; i < n; i++) {
    unconstrained[i] = -Math.log(Math.max(1e-9, rng()));
  }
  return projectOntoBoundedSimplex(unconstrained, maxPerAsset);
}

/**
 * Solves the Minimum Variance Portfolio on the bounded simplex.
 */
export function findMinimumVariancePortfolio(
  assets: OptimizableAsset[],
  covarianceMatrix: number[][],
  maxPerAsset = 1.0,
  minPerAsset = 0.0
): OptimizerPoint {
  const n = assets.length;
  const covMat = new Matrix(covarianceMatrix);
  const returns = assets.map((a) => a.expectedReturn);

  // Initial equal weights projected onto simplex
  let w = projectOntoBoundedSimplex(new Array(n).fill(1 / n), maxPerAsset, minPerAsset);
  const maxEig = estimateMaxEigenvalue(covMat, n);
  const stepSize = 1.0 / (2 * maxEig);

  for (let iter = 0; iter < 400; iter++) {
    const grad = computeVarianceGradient(w, covMat);
    const candidateV = new Array(n);
    for (let i = 0; i < n; i++) {
      candidateV[i] = w[i] - stepSize * grad[i];
    }
    const nextW = projectOntoBoundedSimplex(candidateV, maxPerAsset, minPerAsset);
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(nextW[i] - w[i]);
    w = nextW;
    if (diff < 1e-7) break;
  }

  const ret = computePortfolioReturn(w, returns);
  const vol = Math.sqrt(computePortfolioVariance(w, covMat));
  const sharpe = (ret - 0.065) / Math.max(1e-4, vol);

  const weightsMap: Record<string, number> = {};
  for (let i = 0; i < n; i++) {
    weightsMap[assets[i].id] = Number(w[i].toFixed(4));
  }

  return {
    weights: weightsMap,
    expectedReturn: Number(ret.toFixed(4)),
    volatility: Number(vol.toFixed(4)),
    sharpeRatio: Number(sharpe.toFixed(4)),
  };
}

/**
 * Maximize expected return subject to portfolio volatility <= maxRisk
 * and per-asset cap <= maxPerAsset.
 */
export function maximizeReturn(
  assets: OptimizableAsset[],
  covarianceMatrix: number[][],
  constraints: OptimizerConstraints = {}
): OptimizerPoint {
  const n = assets.length;
  const maxRisk = constraints.maxRisk ?? 0.20;
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const riskFreeRate = constraints.riskFreeRate ?? 0.065;
  const covMat = new Matrix(covarianceMatrix);
  const returns = assets.map((a) => a.expectedReturn);
  const maxEig = estimateMaxEigenvalue(covMat, n);

  // 1. Check if min variance portfolio satisfies maxRisk
  const minVarPoint = findMinimumVariancePortfolio(
    assets,
    covarianceMatrix,
    maxPerAsset,
    minPerAsset
  );

  if (minVarPoint.volatility > maxRisk + 1e-6) {
    // Infeasible: even lowest possible risk exceeds maxRisk. Return minVar with infeasible flag.
    return {
      ...minVarPoint,
      infeasible: true,
      riskLimitBinding: true,
    };
  }

  // 2. Check unconstrained maximum return portfolio (greedy allocation up to maxPerAsset)
  const sortedIndices = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => returns[b] - returns[a]
  );
  const maxRetWeights = new Array(n).fill(0);
  let budget = 1.0;
  for (const idx of sortedIndices) {
    const alloc = Math.min(budget, maxPerAsset);
    maxRetWeights[idx] = alloc;
    budget -= alloc;
    if (budget <= 1e-9) break;
  }
  const cleanMaxRetW = projectOntoBoundedSimplex(maxRetWeights, maxPerAsset, minPerAsset);
  const maxRetVol = Math.sqrt(computePortfolioVariance(cleanMaxRetW, covMat));
  const maxRetVal = computePortfolioReturn(cleanMaxRetW, returns);

  // If the unconstrained max return portfolio already satisfies maxRisk, risk limit is not binding
  if (maxRetVol <= maxRisk + 1e-6) {
    const weightsMap: Record<string, number> = {};
    for (let i = 0; i < n; i++) weightsMap[assets[i].id] = Number(cleanMaxRetW[i].toFixed(4));
    return {
      weights: weightsMap,
      expectedReturn: Number(maxRetVal.toFixed(4)),
      volatility: Number(maxRetVol.toFixed(4)),
      sharpeRatio: Number(((maxRetVal - riskFreeRate) / Math.max(1e-4, maxRetVol)).toFixed(4)),
      infeasible: false,
      riskLimitBinding: false,
    };
  }

  // 3. Subroutine: solves max mu'w - (lambda/2) w'Sigma w on bounded simplex
  const solveLagrangianQP = (lambda: number, warmStartW: number[]): number[] => {
    let w = warmStartW.slice();
    const stepSize = Math.min(1.0, 1.0 / (lambda * maxEig));
    for (let iter = 0; iter < 200; iter++) {
      const covW = covMat.mmul(Matrix.columnVector(w)).to1DArray();
      const candidateV = new Array(n);
      for (let i = 0; i < n; i++) {
        candidateV[i] = w[i] + stepSize * (returns[i] - lambda * covW[i]);
      }
      const nextW = projectOntoBoundedSimplex(candidateV, maxPerAsset, minPerAsset);
      let diff = 0;
      for (let i = 0; i < n; i++) diff += Math.abs(nextW[i] - w[i]);
      w = nextW;
      if (diff < 1e-7) break;
    }
    return w;
  };

  // 4. Bisect lambda so that portfolio volatility approaches maxRisk from below
  let lambdaMin = 1e-4; // low risk aversion -> high vol (typically > maxRisk)
  let lambdaMax = 1e5; // high risk aversion -> approaches minVar (vol <= maxRisk)

  // Start with minVar weights
  const minVarWArr = assets.map((a) => minVarPoint.weights[a.id] || 0);
  const minVarNormW = projectOntoBoundedSimplex(minVarWArr, maxPerAsset, minPerAsset);
  let currentW = minVarNormW.slice();
  let bestFeasibleW = minVarNormW.slice();
  let bestFeasibleVol = minVarPoint.volatility;
  let bestFeasibleRet = minVarPoint.expectedReturn;

  for (let step = 0; step < 40; step++) {
    const lambdaMid = Math.sqrt(lambdaMin * lambdaMax);
    const wMid = solveLagrangianQP(lambdaMid, currentW);
    currentW = wMid;
    const volMid = Math.sqrt(computePortfolioVariance(wMid, covMat));
    const retMid = computePortfolioReturn(wMid, returns);
    if (volMid <= maxRisk + 1e-6) {
      if (retMid >= bestFeasibleRet - 1e-7) {
        bestFeasibleRet = retMid;
        bestFeasibleVol = volMid;
        bestFeasibleW = wMid;
      }
      lambdaMax = lambdaMid;
    } else {
      lambdaMin = lambdaMid;
    }
  }

  // 5. Ensure bestFeasibleW strictly satisfies vol <= maxRisk + 1e-6
  let finalW = bestFeasibleW;
  let finalVol = Math.sqrt(computePortfolioVariance(finalW, covMat));
  if (finalVol > maxRisk) {
    for (let blend = 0.01; blend <= 1.0; blend += 0.01) {
      const blended = finalW.map((fw, i) => (1 - blend) * fw + blend * minVarNormW[i]);
      const bVol = Math.sqrt(computePortfolioVariance(blended, covMat));
      if (bVol <= maxRisk) {
        finalW = blended;
        finalVol = bVol;
        break;
      }
    }
  }
  let finalRet = computePortfolioReturn(finalW, returns);

  // 6. If the Sharpe-optimal portfolio has vol <= maxRisk, ensure maximizeReturn's return is at least as high
  const sharpePortfolio = maximizeSharpe(assets, covarianceMatrix, constraints);
  if (sharpePortfolio.volatility <= maxRisk + 1e-6 && sharpePortfolio.expectedReturn > finalRet) {
    return {
      ...sharpePortfolio,
      infeasible: false,
      riskLimitBinding: true,
    };
  }

  const weightsMap: Record<string, number> = {};
  for (let i = 0; i < n; i++) {
    weightsMap[assets[i].id] = Number(finalW[i].toFixed(4));
  }

  return {
    weights: weightsMap,
    expectedReturn: Number(finalRet.toFixed(4)),
    volatility: Number(finalVol.toFixed(4)),
    sharpeRatio: Number(((finalRet - riskFreeRate) / Math.max(1e-4, finalVol)).toFixed(4)),
    infeasible: false,
    riskLimitBinding: true,
  };
}

/**
 * Maximize Sharpe Ratio under per-asset caps.
 */
export function maximizeSharpe(
  assets: OptimizableAsset[],
  covarianceMatrix: number[][],
  constraints: OptimizerConstraints = {}
): OptimizerPoint {
  const n = assets.length;
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const riskFreeRate = constraints.riskFreeRate ?? 0.065;
  const covMat = new Matrix(covarianceMatrix);
  const returns = assets.map((a) => a.expectedReturn);

  let bestSharpe = -Infinity;
  let bestPoint: OptimizerPoint = {
    weights: {},
    expectedReturn: 0,
    volatility: 0,
    sharpeRatio: 0,
  };

  const numStarts = Math.min(25, n + 10);
  for (let s = 0; s < numStarts; s++) {
    let w = sampleDeterministicFeasibleWeights(n, maxPerAsset, s);
    let stepSize = 0.02;
    for (let iter = 0; iter < 200; iter++) {
      const pRet = computePortfolioReturn(w, returns);
      const pVar = computePortfolioVariance(w, covMat);
      const pVol = Math.sqrt(pVar);
      const excessReturn = pRet - riskFreeRate;
      if (pVol < 1e-8) break;

      const covW = covMat.mmul(Matrix.columnVector(w)).to1DArray();
      const grad: number[] = new Array(n);
      for (let i = 0; i < n; i++) {
        grad[i] = returns[i] / pVol - (excessReturn * covW[i]) / (pVol * pVar);
      }
      const candidateV = new Array(n);
      for (let i = 0; i < n; i++) {
        candidateV[i] = w[i] + stepSize * grad[i];
      }
      const nextW = projectOntoBoundedSimplex(candidateV, maxPerAsset, minPerAsset);
      let diff = 0;
      for (let i = 0; i < n; i++) diff += Math.abs(nextW[i] - w[i]);
      w = nextW;
      if (diff < 1e-6) break;
    }
    const curRet = computePortfolioReturn(w, returns);
    const curVol = Math.sqrt(computePortfolioVariance(w, covMat));
    const curSharpe = (curRet - riskFreeRate) / Math.max(1e-4, curVol);
    if (curSharpe > bestSharpe) {
      bestSharpe = curSharpe;
      const weightsMap: Record<string, number> = {};
      for (let i = 0; i < n; i++) weightsMap[assets[i].id] = Number(w[i].toFixed(4));
      bestPoint = {
        weights: weightsMap,
        expectedReturn: Number(curRet.toFixed(4)),
        volatility: Number(curVol.toFixed(4)),
        sharpeRatio: Number(curSharpe.toFixed(4)),
        infeasible: false,
      };
    }
  }

  return bestPoint;
}

/**
 * Compute Efficient Frontier (default 30 points).
 */
export function computeEfficientFrontier(
  assets: OptimizableAsset[],
  covarianceMatrix: number[][],
  nPoints = 30,
  constraints: OptimizerConstraints = {}
): OptimizerPoint[] {
  const n = assets.length;
  const maxPerAsset = constraints.maxPerAsset ?? 1.0;
  const minPerAsset = constraints.minPerAsset ?? 0.0;
  const riskFreeRate = constraints.riskFreeRate ?? 0.065;
  const covMat = new Matrix(covarianceMatrix);
  const returns = assets.map((a) => a.expectedReturn);

  const minVar = findMinimumVariancePortfolio(assets, covarianceMatrix, maxPerAsset, minPerAsset);

  const sortedAssetIndices = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => returns[b] - returns[a]
  );
  const maxRetWeights = new Array(n).fill(0);
  let remainingBudget = 1.0;
  for (const idx of sortedAssetIndices) {
    const alloc = Math.min(remainingBudget, maxPerAsset);
    maxRetWeights[idx] = alloc;
    remainingBudget -= alloc;
    if (remainingBudget <= 1e-9) break;
  }
  const cleanMaxRetW = projectOntoBoundedSimplex(maxRetWeights, maxPerAsset, minPerAsset);
  const maxRetVal = computePortfolioReturn(cleanMaxRetW, returns);

  const frontier: OptimizerPoint[] = [];
  const minRetVal = minVar.expectedReturn;

  for (let p = 0; p < nPoints; p++) {
    const fraction = p / (nPoints - 1);
    const targetReturn = minRetVal + fraction * (maxRetVal - minRetVal);
    let w = projectOntoBoundedSimplex(
      cleanMaxRetW.map((mw, i) => (1 - fraction) * minVar.weights[assets[i].id] + fraction * mw),
      maxPerAsset,
      minPerAsset
    );
    const lambda = 40.0;
    const stepSize = 0.03;

    for (let iter = 0; iter < 180; iter++) {
      const pRet = computePortfolioReturn(w, returns);
      const retError = pRet - targetReturn;
      const covW = covMat.mmul(Matrix.columnVector(w)).to1DArray();
      const grad: number[] = new Array(n);
      for (let i = 0; i < n; i++) {
        grad[i] = 2 * covW[i] + 2 * lambda * retError * returns[i];
      }
      const candidateV = new Array(n);
      for (let i = 0; i < n; i++) {
        candidateV[i] = w[i] - stepSize * grad[i];
      }
      const nextW = projectOntoBoundedSimplex(candidateV, maxPerAsset, minPerAsset);
      let diff = 0;
      for (let i = 0; i < n; i++) diff += Math.abs(nextW[i] - w[i]);
      w = nextW;
      if (diff < 1e-6) break;
    }

    const curRet = computePortfolioReturn(w, returns);
    const curVol = Math.sqrt(computePortfolioVariance(w, covMat));
    const curSharpe = (curRet - riskFreeRate) / Math.max(1e-4, curVol);

    const weightsMap: Record<string, number> = {};
    for (let i = 0; i < n; i++) weightsMap[assets[i].id] = Number(w[i].toFixed(4));

    frontier.push({
      weights: weightsMap,
      expectedReturn: Number(curRet.toFixed(4)),
      volatility: Number(curVol.toFixed(4)),
      sharpeRatio: Number(curSharpe.toFixed(4)),
      infeasible: false,
    });
  }

  return frontier;
}

/**
 * Extends the optimizable asset universe and covariance matrix with Bonds / Fixed Deposit.
 */
export function extendAssetsAndCovarianceWithBonds(
  assets: OptimizableAsset[],
  covarianceMatrix: number[][],
  bondExpectedReturn: number,
  bondVolatility = 0.005
): {
  assets: OptimizableAsset[];
  covarianceMatrix: number[][];
} {
  const bondAsset: OptimizableAsset = {
    id: 'bonds_fd',
    expectedReturn: bondExpectedReturn,
    volatility: bondVolatility,
  };
  const newAssets = [...assets, bondAsset];
  const n = assets.length;
  const newCov: number[][] = Array.from({ length: n + 1 }, () => new Array(n + 1).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      newCov[i][j] = covarianceMatrix[i][j];
    }
  }

  newCov[n][n] = bondVolatility * bondVolatility;

  return {
    assets: newAssets,
    covarianceMatrix: newCov,
  };
}
