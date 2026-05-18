# SPDX-License-Identifier: CECILL-2.1
"""
Frozen NumPy reference for the Phase 20 ``c4a_transfer_metrics_compute``
entry point.

Algorithm parity with the canonical Python utility in
``nirs4all/analysis/transfer_metrics.py`` is preserved for everything that
is rotation- / sign- / random-state-invariant. The two deviations from the
``TransferMetricsComputer`` reference are:

  1. The covariance step uses ``np.cov`` (i.e. ``ddof=1``) exactly as in the
     reference. PCA is computed via eigendecomposition of ``Xc^T Xc`` (with
     sklearn's deterministic ``svd_flip`` sign convention applied on the V
     side) — this matches sklearn's ``PCA(svd_solver='full')`` to within
     machine epsilon and is exactly what the C engine implements.

  2. The subsampling step of ``spread_distance`` uses a SplitMix64-driven
     Fisher-Yates shuffle keyed on the user-supplied ``seed``. This is what
     the C engine implements; the canonical Python utility uses
     ``np.random.RandomState(seed).choice`` which would require
     bit-replicating NumPy's legacy Mersenne-Twister stream — out of scope
     for the chemometrics4all C ABI (we already have a PCG64 RNG of our
     own from Phase 1).

Everything else (centroid distance, CKA on covariance matrices, RV on
covariance matrices, Grassmann via subspace angles, Procrustes superposition
on the first two PCA components, sklearn-style trustworthiness) is the
same arithmetic that the reference performs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Subsample size matches the canonical reference.
_SPREAD_SUBSAMPLE = 100


@dataclass
class TransferMetrics:
    """Container for the 9 transfer-alignment metrics."""

    centroid_distance: float
    cka_similarity: float
    grassmann_distance: float
    rv_coefficient: float
    procrustes_disparity: float
    trustworthiness: float
    spread_distance: float
    evr_source: float
    evr_target: float


# --------------------------------------------------------------------------- #
#  Deterministic Fisher-Yates choice (matches the C engine)                   #
# --------------------------------------------------------------------------- #


def _splitmix64(state: int) -> tuple[int, int]:
    """Step one round of SplitMix64. Returns (new_state, z) where z is the
    produced 64-bit output and new_state is what the next call should use."""
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
    return state, z


def _deterministic_choice(seed: int, n: int, k: int) -> np.ndarray:
    """Partial Fisher-Yates shuffle driven by SplitMix64.

    Returns the first ``k`` entries of a fresh permutation of
    ``np.arange(n)``. Bit-exact with the C engine's
    ``deterministic_choice`` helper for the same ``seed``.
    """
    if k <= 0 or n <= 0:
        return np.empty(0, dtype=np.int32)
    state = seed & 0xFFFFFFFFFFFFFFFF
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    pool = np.arange(n, dtype=np.int32)
    out = np.empty(min(k, n), dtype=np.int32)
    for i in range(min(k, n)):
        state, z = _splitmix64(state)
        rng_range = n - i
        j = i + int(z % rng_range)
        pool[i], pool[j] = pool[j], pool[i]
        out[i] = pool[i]
    return out


# --------------------------------------------------------------------------- #
#  PCA                                                                        #
# --------------------------------------------------------------------------- #


def _pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Center X by column mean, compute the top ``min(n_components, p, n)``
    eigenvectors of ``Xc^T Xc`` via NumPy, project, and return scores Z,
    loadings U, and the explained-variance ratio.

    Sign convention: sklearn's ``svd_flip`` with ``u_based_decision=False``
    — each loading vector's max-abs entry is non-negative.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, p = X.shape
    if n < 1 or p < 1 or n_components < 1:
        raise ValueError("invalid PCA input")
    mean = X.mean(axis=0)
    Xc = X - mean
    # Symmetric eigendecomposition is robust enough for the typical NIR p
    # (and matches what the C engine implements via Jacobi).
    cov = Xc.T @ Xc  # No 1/(n-1) — only the ratio in EVR matters.
    eigs, V = np.linalg.eigh(cov)
    # Sort descending.
    order = np.argsort(-eigs)
    eigs = eigs[order]
    V = V[:, order]
    # Truncate non-positive eigenvalues (rank-deficient case).
    r_max = int(np.sum(eigs > 0))
    r_max = min(r_max, n)
    r = min(n_components, p, n, r_max)
    r = max(r, 0)
    U = V[:, :r].copy()
    Z = Xc @ U if r > 0 else np.zeros((n, 0), dtype=np.float64)
    # Sign flip — make the max-abs entry of each loading column non-negative.
    if r > 0:
        for j in range(r):
            imax = int(np.argmax(np.abs(U[:, j])))
            if U[imax, j] < 0.0:
                U[:, j] *= -1.0
                Z[:, j] *= -1.0
    total = float(np.sum(eigs[eigs > 0]))
    kept = float(np.sum(eigs[:r]))
    evr = (kept / total) if total > 0.0 else 0.0
    return Z, U, evr


# --------------------------------------------------------------------------- #
#  Metric kernels                                                             #
# --------------------------------------------------------------------------- #


def _centroid_distance(Zs: np.ndarray, Zt: np.ndarray) -> float:
    return float(np.linalg.norm(Zs.mean(axis=0) - Zt.mean(axis=0)))


def _cka(Cs: np.ndarray, Ct: np.ndarray) -> float:
    num = np.linalg.norm(Cs @ Ct, "fro") ** 2
    den = np.linalg.norm(Cs @ Cs, "fro") * np.linalg.norm(Ct @ Ct, "fro")
    return float(num / den) if den > 0 else float("nan")


def _rv(Cs: np.ndarray, Ct: np.ndarray) -> float:
    num = np.trace(Cs @ Ct)
    den = np.sqrt(np.trace(Cs @ Cs) * np.trace(Ct @ Ct))
    return float(num / den) if den > 0 else float("nan")


def _grassmann(Us: np.ndarray, Ut: np.ndarray) -> float:
    M = Us.T @ Ut  # (r, r)
    # Singular values of M live in [0, 1] for orthonormal U.
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, 0.0, 1.0)
    theta = np.arccos(sv)
    return float(np.sqrt(np.sum(theta * theta)))


def _procrustes(Zs: np.ndarray, Zt: np.ndarray) -> float:
    """Mirror scipy.spatial.procrustes: take min sample count, up to 2
    components, standardise, return the disparity scalar."""
    n_use = min(Zs.shape[0], Zt.shape[0])
    d = min(2, Zs.shape[1], Zt.shape[1])
    if n_use < 2 or d < 1:
        return float("nan")
    A = Zs[:n_use, :d].astype(np.float64).copy()
    B = Zt[:n_use, :d].astype(np.float64).copy()
    # Center columns.
    A -= A.mean(axis=0)
    B -= B.mean(axis=0)
    # Frobenius normalise.
    fA = np.linalg.norm(A, "fro")
    fB = np.linalg.norm(B, "fro")
    if fA == 0 or fB == 0:
        return float("nan")
    A /= fA
    B /= fB
    # SVD of A^T B → sum of singular values.
    sv = np.linalg.svd(A.T @ B, compute_uv=False)
    s = float(np.sum(sv))
    disparity = 1.0 - s * s
    if disparity < 0.0:
        disparity = 0.0
    return disparity


def _trustworthiness(Zs: np.ndarray, Zt: np.ndarray, k: int) -> float:
    """Replicate sklearn.manifold.trustworthiness with row-wise stable
    argsort (ties broken by ascending index)."""
    n = min(Zs.shape[0], Zt.shape[0])
    if n < 4 or k < 1:
        return float("nan")
    k_eff = min(k, n - 2)
    k_eff = max(k_eff, 2)
    Zs = Zs[:n]
    Zt = Zt[:n]
    # Squared pairwise distances.
    Ds = np.sum((Zs[:, None, :] - Zs[None, :, :]) ** 2, axis=2)
    Dt = np.sum((Zt[:, None, :] - Zt[None, :, :]) ** 2, axis=2)
    # Stable argsort each row.
    penalty = 0.0
    for i in range(n):
        # Indices excluding i. Stable sort by ascending distance, ties broken
        # by ascending index (numpy.argsort kind='stable').
        idx_all = np.arange(n)
        keep = idx_all != i
        idx_keep = idx_all[keep]

        ds_row = Ds[i, idx_keep]
        dt_row = Dt[i, idx_keep]
        src_order = idx_keep[np.argsort(ds_row, kind="stable")]
        tgt_order = idx_keep[np.argsort(dt_row, kind="stable")]

        # ranks_s[v] = position of v in the source ordering.
        ranks_s = np.empty(n, dtype=np.int64)
        ranks_s[i] = -1
        for pos, v in enumerate(src_order):
            ranks_s[v] = pos
        # top-k target.
        Ui = tgt_order[:k_eff]
        for v in Ui:
            rs = ranks_s[v]
            if rs >= k_eff:
                penalty += rs - (k_eff - 1)
    Z = n * k_eff * (2 * n - 3 * k_eff - 1) / 2.0
    if Z <= 0:
        return float("nan")
    return 1.0 - (2.0 / Z) * penalty


def _spread_distance(Zs: np.ndarray, Zt: np.ndarray, seed: int) -> float:
    # Covariance part.
    Cs = np.cov(Zs.T) if Zs.shape[0] >= 2 else np.zeros((Zs.shape[1], Zs.shape[1]))
    Ct = np.cov(Zt.T) if Zt.shape[0] >= 2 else np.zeros((Zt.shape[1], Zt.shape[1]))
    if Cs.ndim == 0:
        Cs = np.array([[float(Cs)]])
        Ct = np.array([[float(Ct)]])
    cov_dist = float(np.linalg.norm(Cs - Ct, "fro"))

    # Subsample-based pairwise — uses SplitMix64-deterministic choice.
    n1 = min(_SPREAD_SUBSAMPLE, Zs.shape[0])
    n2 = min(_SPREAD_SUBSAMPLE, Zt.shape[0])
    idx1 = _deterministic_choice(seed, Zs.shape[0], n1)
    idx2 = _deterministic_choice(seed ^ 0xA5A5A5A5A5A5A5A5, Zt.shape[0], n2)
    Z1s = Zs[idx1]
    Z2s = Zt[idx2]
    diff = Z1s[:, None, :] - Z2s[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    m1 = float(np.mean(np.min(dists, axis=1)))
    m2 = float(np.mean(np.min(dists, axis=0)))
    return cov_dist + 0.5 * (m1 + m2)


# --------------------------------------------------------------------------- #
#  Top-level entry point                                                      #
# --------------------------------------------------------------------------- #


def transfer_metrics(
    X_source: np.ndarray,
    X_target: np.ndarray,
    *,
    n_components: int = 10,
    k_neighbors: int = 10,
    seed: int = 0,
) -> TransferMetrics:
    """Compute the 9-metric transfer-alignment vector.

    Parameters
    ----------
    X_source : array, shape (n_src, p_src)
    X_target : array, shape (n_tgt, p_tgt)
    n_components : int
        PCA truncation rank.
    k_neighbors : int
        Trustworthiness neighbour count.
    seed : int
        Deterministic subsample seed for spread_distance.

    Returns
    -------
    TransferMetrics
    """
    Xs = np.ascontiguousarray(X_source, dtype=np.float64)
    Xt = np.ascontiguousarray(X_target, dtype=np.float64)

    Zs, Us, evr_s = _pca(Xs, n_components)
    Zt, Ut, evr_t = _pca(Xt, n_components)

    r_use = min(Zs.shape[1], Zt.shape[1])
    Zs = Zs[:, :r_use]
    Zt = Zt[:, :r_use]
    Us = Us[:, :r_use]
    Ut = Ut[:, :r_use]

    if r_use == 0:
        nan = float("nan")
        return TransferMetrics(
            centroid_distance=nan,
            cka_similarity=nan,
            grassmann_distance=nan,
            rv_coefficient=nan,
            procrustes_disparity=nan,
            trustworthiness=nan,
            spread_distance=nan,
            evr_source=evr_s,
            evr_target=evr_t,
        )

    # Covariance matrices for CKA / RV.
    if Zs.shape[0] >= 2:
        Cs_full = np.cov(Zs.T)
    else:
        Cs_full = np.zeros((r_use, r_use))
    if Zt.shape[0] >= 2:
        Ct_full = np.cov(Zt.T)
    else:
        Ct_full = np.zeros((r_use, r_use))
    Cs_full = np.atleast_2d(Cs_full)
    Ct_full = np.atleast_2d(Ct_full)

    centroid = _centroid_distance(Zs, Zt)
    cka = _cka(Cs_full, Ct_full)
    rv = _rv(Cs_full, Ct_full)
    grassmann = _grassmann(Us, Ut) if Us.shape[0] == Ut.shape[0] else float("nan")
    procrustes = _procrustes(Zs, Zt)
    n_min = min(Zs.shape[0], Zt.shape[0])
    trust = _trustworthiness(Zs, Zt, k_neighbors) if n_min >= 4 else float("nan")
    spread = _spread_distance(Zs, Zt, seed)

    return TransferMetrics(
        centroid_distance=centroid,
        cka_similarity=cka,
        grassmann_distance=grassmann,
        rv_coefficient=rv,
        procrustes_disparity=procrustes,
        trustworthiness=trust,
        spread_distance=spread,
        evr_source=evr_s,
        evr_target=evr_t,
    )
