/**
 * Estimation & Matrix Techniques for Long and Unequal History Panels
 * Implements:
 * a) perAssetGeometricMean(panel): annualised geometric mean from asset's own history (refuses if < 36 months)
 * b) emMeanCovariance(panel): Stambaugh-style EM algorithm on monthly log returns with missing values
 * c) pairwiseCovariance(panel) + nearestPSD(matrix): pairwise covariance with eigenvalue clipping
 * d) jamesSteinShrinkMeans(means, nObsPerAsset, cov): positive-part James-Stein shrinkage
 * e) ledoitWolfConstantCorrelation(returnsMatrix): analytic Ledoit-Wolf constant correlation shrinkage
 * f) proxyBackfill(panel, proxyMap, options): regression backfill with residual resampling
 */

import { EigenvalueDecomposition, Matrix, inverse } from 'ml-matrix';
import { UnbalancedPanel } from './types.ts';
import { createMulberry32 } from './prng.ts';

export interface PerAssetMeanResult {
  assetId: string;
  annualizedMean: number | null;
  monthsCount: number;
  reason?: string;
}

/**
 * Computes each asset's annualized geometric mean from its OWN full history.
 * Refuses (returns null with a documented reason) if history is strictly less than 36 months.
 */
export function perAssetGeometricMean(panel: UnbalancedPanel): PerAssetMeanResult[] {
  const { assetIds, matrix } = panel;
  const T = matrix.length;
  const n = assetIds.length;

  const results: PerAssetMeanResult[] = [];

  for (let j = 0; j < n; j++) {
    const id = assetIds[j];
    const validRets: number[] = [];

    for (let t = 0; t < T; t++) {
      const val = matrix[t][j];
      if (!Number.isNaN(val) && Number.isFinite(val)) {
        validRets.push(val);
      }
    }

    const mCount = validRets.length;
    if (mCount < 36) {
      results.push({
        assetId: id,
        annualizedMean: null,
        monthsCount: mCount,
        reason: `Insufficient historical observation months (${mCount} < 36 required threshold).`,
      });
      continue;
    }

    // Cumulative product of gross returns (1 + r)
    let cumulativeGrowth = 1.0;
    for (let t = 0; t < mCount; t++) {
      cumulativeGrowth *= Math.max(1e-6, 1 + validRets[t]);
    }

    const years = mCount / 12;
    const geomMean = Math.pow(cumulativeGrowth, 1 / years) - 1;

    results.push({
      assetId: id,
      annualizedMean: Number(geomMean.toFixed(6)),
      monthsCount: mCount,
    });
  }

  return results;
}

/**
 * EM Algorithm for Multivariate Normal on Monthly Log Returns with Missing Values.
 * This is the Stambaugh-style estimator (Stambaugh 1997 / Little & Rubin 2002):
 * It leverages the longer-history assets to improve the mean and covariance estimates of shorter assets.
 * Max 200 iterations, tolerance 1e-8, small ridge on the diagonal for numerical stability.
 */
export function emMeanCovariance(
  panel: UnbalancedPanel,
  options?: { maxIter?: number; tol?: number; ridge?: number }
): {
  annualizedMeans: number[];
  annualizedCovariance: number[][];
  iterations: number;
  converged: boolean;
} {
  const { assetIds, matrix } = panel;
  const T = matrix.length;
  const N = assetIds.length;
  const maxIter = options?.maxIter ?? 200;
  const tol = options?.tol ?? 1e-8;
  const ridge = options?.ridge ?? 1e-6;

  // Transform available simple returns to log returns: r_log = ln(1 + r)
  const Y: number[][] = Array.from({ length: T }, (_, t) =>
    Array.from({ length: N }, (_, j) => {
      const val = matrix[t][j];
      if (Number.isNaN(val) || !Number.isFinite(val)) return NaN;
      return Math.log(Math.max(1e-6, 1 + val));
    })
  );

  // Initialize means and covariance using available pairwise data
  let mu: number[] = new Array(N).fill(0);
  const counts: number[] = new Array(N).fill(0);

  for (let t = 0; t < T; t++) {
    for (let j = 0; j < N; j++) {
      if (!Number.isNaN(Y[t][j])) {
        mu[j] += Y[t][j];
        counts[j]++;
      }
    }
  }
  for (let j = 0; j < N; j++) {
    mu[j] = counts[j] > 0 ? mu[j] / counts[j] : 0;
  }

  let Sigma: number[][] = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => (i === j ? 0.04 / 12 : 0))
  );

  for (let i = 0; i < N; i++) {
    for (let j = i; j < N; j++) {
      let sum = 0;
      let c = 0;
      for (let t = 0; t < T; t++) {
        if (!Number.isNaN(Y[t][i]) && !Number.isNaN(Y[t][j])) {
          sum += (Y[t][i] - mu[i]) * (Y[t][j] - mu[j]);
          c++;
        }
      }
      const val = c > 1 ? sum / (c - 1) : i === j ? 0.04 / 12 : 0;
      Sigma[i][j] = val;
      Sigma[j][i] = val;
    }
    Sigma[i][i] += ridge;
  }

  let converged = false;
  let iter = 0;

  for (iter = 0; iter < maxIter; iter++) {
    const oldMu = mu.slice();
    const sumY = new Array(N).fill(0);
    const sumYY = Array.from({ length: N }, () => new Array(N).fill(0));

    // E-Step: For each time t, partition into observed (O) and missing (M)
    for (let t = 0; t < T; t++) {
      const obsIdx: number[] = [];
      const missIdx: number[] = [];

      for (let j = 0; j < N; j++) {
        if (Number.isNaN(Y[t][j])) missIdx.push(j);
        else obsIdx.push(j);
      }

      if (missIdx.length === 0) {
        // Complete observation at month t
        for (let i = 0; i < N; i++) {
          sumY[i] += Y[t][i];
          for (let j = 0; j < N; j++) {
            sumYY[i][j] += Y[t][i] * Y[t][j];
          }
        }
        continue;
      }

      if (obsIdx.length === 0) {
        // Entirely missing month
        for (let i = 0; i < N; i++) {
          sumY[i] += mu[i];
          for (let j = 0; j < N; j++) {
            sumYY[i][j] += mu[i] * mu[j] + Sigma[i][j];
          }
        }
        continue;
      }

      // Partition Sigma:
      // Sigma_OO, Sigma_MO, Sigma_OM, Sigma_MM
      const nObs = obsIdx.length;
      const nMiss = missIdx.length;

      const S_OO_data: number[][] = Array.from({ length: nObs }, (_, r) =>
        Array.from({ length: nObs }, (_, c) => {
          const val = Sigma[obsIdx[r]][obsIdx[c]];
          return r === c ? val + ridge : val;
        })
      );
      const S_MO_data: number[][] = Array.from({ length: nMiss }, (_, r) =>
        Array.from({ length: nObs }, (_, c) => Sigma[missIdx[r]][obsIdx[c]])
      );

      const y_O = obsIdx.map((idx) => Y[t][idx] - mu[idx]);
      const mu_M = missIdx.map((idx) => mu[idx]);

      const Mat_OO = new Matrix(S_OO_data);
      const Mat_MO = new Matrix(S_MO_data);
      const Vec_yO = Matrix.columnVector(y_O);

      let condMeanMiss: number[] = [];
      let condCovMiss: number[][] = [];

      try {
        const invOO = inverse(Mat_OO);
        const beta = Mat_MO.mmul(invOO); // (nMiss x nObs)
        const condMeanOffset = beta.mmul(Vec_yO).to1DArray();

        condMeanMiss = missIdx.map((_, idx) => mu_M[idx] + condMeanOffset[idx]);

        // Sigma_MM - S_MO * inv(S_OO) * S_OM
        const condCovMat = new Matrix(
          Array.from({ length: nMiss }, (_, r) =>
            Array.from({ length: nMiss }, (_, c) => Sigma[missIdx[r]][missIdx[c]])
          )
        ).sub(beta.mmul(Mat_MO.transpose()));

        condCovMiss = condCovMat.to2DArray();
      } catch {
        // Fallback if matrix inversion fails
        condMeanMiss = mu_M.slice();
        condCovMiss = Array.from({ length: nMiss }, (_, r) =>
          Array.from({ length: nMiss }, (_, c) => Sigma[missIdx[r]][missIdx[c]])
        );
      }

      // Reconstructed E[Y_t]
      const expY = new Array(N).fill(0);
      for (let r = 0; r < nObs; r++) expY[obsIdx[r]] = Y[t][obsIdx[r]];
      for (let r = 0; r < nMiss; r++) expY[missIdx[r]] = condMeanMiss[r];

      for (let i = 0; i < N; i++) {
        sumY[i] += expY[i];
      }

      // Reconstructed E[Y_t Y_t^T]
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          sumYY[i][j] += expY[i] * expY[j];
        }
      }

      // Add conditional covariance for missing-missing entries
      for (let r = 0; r < nMiss; r++) {
        const i = missIdx[r];
        for (let c = 0; c < nMiss; c++) {
          const j = missIdx[c];
          sumYY[i][j] += condCovMiss[r][c];
        }
      }
    }

    // M-Step: Update mu and Sigma
    const newMu = sumY.map((s) => s / T);
    const newSigma: number[][] = Array.from({ length: N }, (_, i) =>
      Array.from({ length: N }, (_, j) => {
        const covVal = sumYY[i][j] / T - newMu[i] * newMu[j];
        return i === j ? Math.max(1e-7, covVal + ridge) : covVal;
      })
    );

    // Check convergence
    let maxDiff = 0;
    for (let j = 0; j < N; j++) {
      maxDiff = Math.max(maxDiff, Math.abs(newMu[j] - oldMu[j]));
    }

    mu = newMu;
    Sigma = newSigma;

    if (maxDiff < tol) {
      converged = true;
      break;
    }
  }

  // Convert monthly log parameters to annualized simple return parameters
  const annualizedMeans = mu.map((m) => Math.exp(m * 12) - 1);
  const annualizedCovariance = Sigma.map((row) => row.map((v) => v * 12));

  return {
    annualizedMeans,
    annualizedCovariance: nearestPSD(annualizedCovariance),
    iterations: iter + 1,
    converged,
  };
}

/**
 * Pairwise-complete covariance matrix across all available overlapping months.
 */
export function pairwiseCovariance(panel: UnbalancedPanel): number[][] {
  const { assetIds, matrix } = panel;
  const T = matrix.length;
  const N = assetIds.length;

  const cov: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));

  for (let i = 0; i < N; i++) {
    for (let j = i; j < N; j++) {
      let sumProd = 0;
      let sumI = 0;
      let sumJ = 0;
      let count = 0;

      for (let t = 0; t < T; t++) {
        const rI = matrix[t][i];
        const rJ = matrix[t][j];
        if (!Number.isNaN(rI) && !Number.isNaN(rJ)) {
          sumI += rI;
          sumJ += rJ;
          count++;
        }
      }

      if (count < 2) {
        cov[i][j] = i === j ? 0.04 : 0;
        cov[j][i] = cov[i][j];
        continue;
      }

      const meanI = sumI / count;
      const meanJ = sumJ / count;

      for (let t = 0; t < T; t++) {
        const rI = matrix[t][i];
        const rJ = matrix[t][j];
        if (!Number.isNaN(rI) && !Number.isNaN(rJ)) {
          sumProd += (rI - meanI) * (rJ - meanJ);
        }
      }

      const monthlyCov = sumProd / (count - 1);
      const annualCov = monthlyCov * 12;
      cov[i][j] = annualCov;
      cov[j][i] = annualCov;
    }
  }

  return nearestPSD(cov);
}

/**
 * Repairs a symmetric matrix to nearest Positive Semi-Definite (PSD) via Eigenvalue clipping.
 * Floors negative or near-zero eigenvalues to minEigenvalue (default 1e-8).
 */
export function nearestPSD(matrix: number[][], minEigenvalue = 1e-8): number[][] {
  const n = matrix.length;
  if (n <= 1) return matrix.map((row) => row.slice());

  try {
    const mat = new Matrix(matrix);
    // Enforce exact symmetry
    const symMat = mat.add(mat.transpose()).mul(0.5);
    const evd = new EigenvalueDecomposition(symMat);

    const realEval = evd.realEigenvalues;
    const V = evd.eigenvectorMatrix;

    // Diagonal matrix with clipped eigenvalues
    const D = Matrix.zeros(n, n);
    for (let i = 0; i < n; i++) {
      D.set(i, i, Math.max(minEigenvalue, realEval[i]));
    }

    // Reconstruct: V * D * V^T
    const reconstructed = V.mmul(D).mmul(V.transpose());
    const res = reconstructed.to2DArray();

    // Re-verify exact symmetry
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const avg = (res[i][j] + res[j][i]) / 2;
        res[i][j] = avg;
        res[j][i] = avg;
      }
    }
    return res;
  } catch {
    // Fallback: simple diagonal loading
    const res = matrix.map((row) => row.slice());
    for (let i = 0; i < n; i++) {
      res[i][i] = Math.max(minEigenvalue, res[i][i]);
    }
    return res;
  }
}

/**
 * Positive-part James-Stein shrinkage of mean returns toward the grand mean,
 * weighted by each asset's effective sample size.
 */
export function jamesSteinShrinkMeans(
  means: number[],
  nObsPerAsset: number[],
  cov: number[][]
): number[] {
  const p = means.length;
  if (p <= 2) return means.slice();

  // Grand mean weighted by effective sample size
  const totalObs = nObsPerAsset.reduce((a, b) => a + b, 0);
  const grandMean =
    totalObs > 0
      ? means.reduce((sum, m, i) => sum + m * nObsPerAsset[i], 0) / totalObs
      : means.reduce((a, b) => a + b, 0) / p;

  // Average diagonal variance
  const avgVar = cov.reduce((sum, row, i) => sum + row[i], 0) / p;
  const avgN = totalObs / p;
  const sigmaSq = avgVar / Math.max(1, avgN);

  // Sum of squared deviations from grand mean
  const devSqSum = means.reduce((sum, m) => sum + Math.pow(m - grandMean, 2), 0);

  // James-Stein shrinkage factor: c = 1 - ((p - 2) * sigmaSq) / devSqSum
  const rawShrinkage = devSqSum > 1e-12 ? 1 - ((p - 2) * sigmaSq) / devSqSum : 0;
  const c = Math.max(0, Math.min(1, rawShrinkage)); // Positive-part shrinkage

  return means.map((m, i) => {
    // Individual asset effective shrinkage factor scaling with relative observation size
    const relativeObs = avgN > 0 ? nObsPerAsset[i] / avgN : 1.0;
    const assetC = Math.max(0, Math.min(1, 1 - (1 - c) / Math.max(0.2, relativeObs)));
    return Number((grandMean + assetC * (m - grandMean)).toFixed(6));
  });
}

/**
 * Ledoit-Wolf Analytic Shrinkage toward Constant-Correlation Target.
 * Computes optimal shrinkage intensity k in [0, 1].
 */
export function ledoitWolfConstantCorrelation(
  returnsMatrix: number[][]
): {
  shrunkCovariance: number[][];
  shrinkageIntensity: number;
} {
  const T = returnsMatrix.length;
  const N = returnsMatrix[0]?.length ?? 0;

  if (T < 60 || N <= 1) {
    // Fallback: fixed k = 0.2
    return {
      shrunkCovariance: shrinkCovarianceInternal(returnsMatrix, 0.2),
      shrinkageIntensity: 0.2,
    };
  }

  // Sample means and demeaned matrix
  const means = new Array(N).fill(0);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < N; j++) means[j] += returnsMatrix[t][j];
  }
  for (let j = 0; j < N; j++) means[j] /= T;

  const Y = Array.from({ length: T }, (_, t) =>
    Array.from({ length: N }, (_, j) => returnsMatrix[t][j] - means[j])
  );

  // Sample covariance S
  const S: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = i; j < N; j++) {
      let sum = 0;
      for (let t = 0; t < T; t++) sum += Y[t][i] * Y[t][j];
      const val = (sum / (T - 1)) * 12;
      S[i][j] = val;
      S[j][i] = val;
    }
  }

  // Target: Constant Correlation F
  const vols = S.map((row, i) => Math.sqrt(Math.max(1e-8, row[i])));
  let sumR = 0;
  let rCount = 0;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const rho = S[i][j] / (vols[i] * vols[j]);
      sumR += Math.max(-1, Math.min(1, rho));
      rCount++;
    }
  }
  const rBar = rCount > 0 ? sumR / rCount : 0;

  const F: number[][] = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => (i === j ? S[i][i] : rBar * vols[i] * vols[j]))
  );

  // Pi-mat: sum of asymptotic variances of sample covariance entries
  let piHat = 0;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      let sumProd = 0;
      const s_ij_monthly = S[i][j] / 12;
      for (let t = 0; t < T; t++) {
        const dev = Y[t][i] * Y[t][j] - s_ij_monthly;
        sumProd += dev * dev;
      }
      piHat += (sumProd / T) * 144;
    }
  }

  // Gamma-hat: squared Frobenius distance ||S - F||^2
  let gammaHat = 0;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const diff = S[i][j] - F[i][j];
      gammaHat += diff * diff;
    }
  }

  // Analytic intensity kappa = piHat / (T * gammaHat)
  let kappa = gammaHat > 1e-12 ? piHat / (T * gammaHat) : 0.2;
  kappa = Math.max(0, Math.min(1, kappa));

  const shrunkCov: number[][] = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => (1 - kappa) * S[i][j] + kappa * F[i][j])
  );

  return {
    shrunkCovariance: nearestPSD(shrunkCov),
    shrinkageIntensity: Number(kappa.toFixed(4)),
  };
}

function shrinkCovarianceInternal(returnsMatrix: number[][], k = 0.2): number[][] {
  const T = returnsMatrix.length;
  const N = returnsMatrix[0]?.length ?? 0;
  const cov = Array.from({ length: N }, () => new Array(N).fill(0));
  // Compute basic sample cov
  const means = new Array(N).fill(0);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < N; j++) means[j] += returnsMatrix[t][j];
  }
  for (let j = 0; j < N; j++) means[j] /= T;

  for (let i = 0; i < N; i++) {
    for (let j = i; j < N; j++) {
      let sum = 0;
      for (let t = 0; t < T; t++) {
        sum += (returnsMatrix[t][i] - means[i]) * (returnsMatrix[t][j] - means[j]);
      }
      const val = (sum / Math.max(1, T - 1)) * 12;
      cov[i][j] = val;
      cov[j][i] = val;
    }
  }

  const vols = cov.map((row, i) => Math.sqrt(Math.max(1e-8, row[i])));
  let sumCorr = 0;
  let count = 0;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      sumCorr += cov[i][j] / Math.max(1e-8, vols[i] * vols[j]);
      count++;
    }
  }
  const meanCorr = count > 0 ? sumCorr / count : 0;
  const target: number[][] = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => (i === j ? cov[i][i] : meanCorr * vols[i] * vols[j]))
  );

  return Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => (1 - k) * cov[i][j] + k * target[i][j])
  );
}

/**
 * Regression-based Proxy Backfilling with Residual Resampling.
 * Fits y = a + b * x + e on the overlapping window (monthly log returns).
 * Fills missing leading months with a + b * x_t + e_t (seeded residual draw).
 * Refuses if overlap < 24 months.
 */
export function proxyBackfill(
  panel: UnbalancedPanel,
  proxyMap: Record<string, string>,
  options?: { seed?: number; minOverlap?: number }
): {
  backfilledPanel: UnbalancedPanel;
  backfillReport: Record<
    string,
    {
      proxyId: string;
      overlapMonths: number;
      alpha: number;
      beta: number;
      r2: number;
      backfilledMonths: number;
      status: 'success' | 'refused_overlap_too_short' | 'no_proxy_defined';
    }
  >;
} {
  const { dates, assetIds, matrix } = panel;
  const T = matrix.length;
  const N = assetIds.length;
  const minOverlap = options?.minOverlap ?? 24;
  const rng = createMulberry32(options?.seed ?? 999111);

  const report: Record<string, any> = {};
  const backfilledMatrix = matrix.map((row) => row.slice());

  for (let j = 0; j < N; j++) {
    const id = assetIds[j];
    const proxyId = proxyMap[id];

    if (!proxyId) {
      report[id] = { status: 'no_proxy_defined' };
      continue;
    }

    const proxyCol = assetIds.indexOf(proxyId);
    if (proxyCol < 0) {
      report[id] = { status: 'no_proxy_defined' };
      continue;
    }

    // Find overlap
    const overlapY: number[] = [];
    const overlapX: number[] = [];
    const missingIndices: number[] = [];

    for (let t = 0; t < T; t++) {
      const yVal = matrix[t][j];
      const xVal = matrix[t][proxyCol];

      if (Number.isNaN(yVal)) {
        if (!Number.isNaN(xVal)) missingIndices.push(t);
      } else if (!Number.isNaN(xVal)) {
        overlapY.push(Math.log(Math.max(1e-6, 1 + yVal)));
        overlapX.push(Math.log(Math.max(1e-6, 1 + xVal)));
      }
    }

    if (overlapY.length < minOverlap) {
      report[id] = {
        proxyId,
        overlapMonths: overlapY.length,
        status: 'refused_overlap_too_short',
      };
      continue;
    }

    // OLS: y = alpha + beta * x
    const K = overlapY.length;
    const meanX = overlapX.reduce((a, b) => a + b, 0) / K;
    const meanY = overlapY.reduce((a, b) => a + b, 0) / K;

    let covXY = 0;
    let varX = 0;
    let varY = 0;
    for (let t = 0; t < K; t++) {
      const dx = overlapX[t] - meanX;
      const dy = overlapY[t] - meanY;
      covXY += dx * dy;
      varX += dx * dx;
      varY += dy * dy;
    }

    const beta = varX > 1e-12 ? covXY / varX : 1.0;
    const alpha = meanY - beta * meanX;
    const r2 = varY > 1e-12 ? (covXY * covXY) / (varX * varY) : 0;

    // Residuals for empirical resampling
    const residuals: number[] = [];
    for (let t = 0; t < K; t++) {
      residuals.push(overlapY[t] - (alpha + beta * overlapX[t]));
    }

    // Backfill missing months
    let countBackfilled = 0;
    for (const t of missingIndices) {
      const xLog = Math.log(Math.max(1e-6, 1 + matrix[t][proxyCol]));
      const resampledResidual = residuals[Math.floor(rng() * residuals.length)];
      const yPredLog = alpha + beta * xLog + resampledResidual;
      backfilledMatrix[t][j] = Math.exp(yPredLog) - 1;
      countBackfilled++;
    }

    report[id] = {
      proxyId,
      overlapMonths: K,
      alpha: Number(alpha.toFixed(6)),
      beta: Number(beta.toFixed(4)),
      r2: Number(r2.toFixed(4)),
      backfilledMonths: countBackfilled,
      status: 'success',
    };
  }

  return {
    backfilledPanel: {
      dates,
      assetIds,
      matrix: backfilledMatrix,
    },
    backfillReport: report,
  };
}
