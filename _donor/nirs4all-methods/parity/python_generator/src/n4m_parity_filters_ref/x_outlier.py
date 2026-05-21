# SPDX-License-Identifier: CECILL-2.1
"""
X-based outlier filter — frozen sklearn reference (Phase 13).

This is the canonical parity floor for the ``n4m_filter_x_outlier_*``
ABI category. Mirrors ``nirs4all.operators.filters.XOutlierFilter`` for
all six methods, but isolates the implementation here so future nirs4all
releases can change without breaking the nirs4all-methods parity gate.

Six methods:

  * ``mahalanobis``         — empirical covariance (``EmpiricalCovariance``),
                              threshold = ``sqrt(chi2_inv(0.975, p))``.
  * ``robust_mahalanobis``  — ``MinCovDet`` (FAST-MCD ensemble), same
                              chi-squared threshold.
  * ``pca_residual``        — sum-squared residual to the PCA reconstruction
                              with top-k components; threshold = 95th
                              percentile of training Q.
  * ``pca_leverage``        — Hotelling's T² in the PCA score space (sum of
                              ``score_i^2 / variance_i``); threshold = 95th
                              percentile of training T².
  * ``isolation_forest``    — ``IsolationForest`` ensemble; threshold derived
                              from the contamination quantile of training
                              ``score_samples``.
  * ``lof``                 — ``LocalOutlierFactor`` (novelty=True); same
                              contamination-quantile threshold.

The nirs4all-methods C engine produces near-bit-identical masks where the
algorithm is fully deterministic (mahalanobis, pca_residual, pca_leverage),
and PCG64-stable masks for the random-init paths (robust_mahalanobis,
isolation_forest).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


__all__ = ["x_outlier_fit_get_mask"]


def _chi2_inv_975(p: int) -> float:
    """scipy.stats.chi2.ppf(0.975, p) — replicated via scipy for parity."""
    from scipy.stats import chi2
    return float(chi2.ppf(0.975, df=p))


def _fit_mahalanobis(X: np.ndarray, *, robust: bool,
                       n_components: int | None,
                       random_state: int | None,
                       threshold: float | None) -> tuple[np.ndarray, float]:
    n_samples, n_features = X.shape
    if n_components is not None:
        max_components = n_components
    elif robust:
        max_components = 20
    else:
        max_components = 50
    max_stable = max(n_samples // 3, 2)
    effective_max = min(max_components, max_stable)
    if n_features > effective_max:
        n_comp = min(effective_max, n_features)
        pca = PCA(n_components=n_comp)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X
    if robust:
        try:
            est = MinCovDet(random_state=random_state)
            est.fit(X_reduced)
        except ValueError:
            est = EmpiricalCovariance().fit(X_reduced)
    else:
        est = EmpiricalCovariance().fit(X_reduced)
    distances = np.sqrt(est.mahalanobis(X_reduced))
    if threshold is None:
        threshold = float(np.sqrt(_chi2_inv_975(X_reduced.shape[1])))
    return distances, float(threshold)


def _fit_pca_residual(X: np.ndarray, n_components: int | None,
                       threshold: float | None) -> tuple[np.ndarray, float]:
    n_samples, n_features = X.shape
    n_comp = (min(n_components, n_samples, n_features)
              if n_components is not None
              else min(n_samples, n_features, 10))
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    X_rec = pca.inverse_transform(pca.transform(X))
    q = np.sum((X - X_rec) ** 2, axis=1)
    if threshold is None:
        threshold = float(np.percentile(q, 95))
    return q, float(threshold)


def _fit_pca_leverage(X: np.ndarray, n_components: int | None,
                       threshold: float | None) -> tuple[np.ndarray, float]:
    n_samples, n_features = X.shape
    n_comp = (min(n_components, n_samples, n_features)
              if n_components is not None
              else min(n_samples, n_features, 10))
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    scores = pca.transform(X)
    variances = pca.explained_variance_
    t2 = np.sum((scores ** 2) / variances, axis=1)
    if threshold is None:
        threshold = float(np.percentile(t2, 95))
    return t2, float(threshold)


def _fit_iforest(X: np.ndarray, *, contamination: float,
                  random_state: int | None) -> np.ndarray:
    det = IsolationForest(contamination=contamination,
                            random_state=random_state)
    det.fit(X)
    return det.predict(X)


def _fit_lof(X: np.ndarray, *, contamination: float) -> np.ndarray:
    n_samples = X.shape[0]
    k = min(20, n_samples - 1)
    det = LocalOutlierFactor(n_neighbors=k, contamination=contamination,
                               novelty=True)
    det.fit(X)
    return det.predict(X)


def x_outlier_fit_get_mask(X: np.ndarray, *,
                            method: str,
                            threshold: float | None = None,
                            n_components: int | None = None,
                            contamination: float = 0.1,
                            random_state: int | None = None) -> dict[str, Any]:
    """Run the full ``fit + get_mask`` workflow.

    Returns a dict with keys ``mask`` (np.uint8), ``threshold`` (final value),
    and ``distances`` (per-sample score, for diagnostic logging).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    n = X.shape[0]
    if method == "mahalanobis":
        dists, thr = _fit_mahalanobis(X, robust=False,
                                         n_components=n_components,
                                         random_state=random_state,
                                         threshold=threshold)
        keep = dists <= thr
        return {"mask": keep.astype(np.uint8),
                "threshold": thr,
                "distances": dists}
    if method == "robust_mahalanobis":
        dists, thr = _fit_mahalanobis(X, robust=True,
                                         n_components=n_components,
                                         random_state=random_state,
                                         threshold=threshold)
        keep = dists <= thr
        return {"mask": keep.astype(np.uint8),
                "threshold": thr,
                "distances": dists}
    if method == "pca_residual":
        q, thr = _fit_pca_residual(X, n_components, threshold)
        keep = q <= thr
        return {"mask": keep.astype(np.uint8),
                "threshold": thr,
                "distances": q}
    if method == "pca_leverage":
        t2, thr = _fit_pca_leverage(X, n_components, threshold)
        keep = t2 <= thr
        return {"mask": keep.astype(np.uint8),
                "threshold": thr,
                "distances": t2}
    if method == "isolation_forest":
        pred = _fit_iforest(X, contamination=contamination,
                              random_state=random_state)
        keep = (pred == 1).astype(np.uint8)
        return {"mask": keep,
                "threshold": float("nan"),
                "distances": np.zeros(n, dtype=np.float64)}
    if method == "lof":
        pred = _fit_lof(X, contamination=contamination)
        keep = (pred == 1).astype(np.uint8)
        return {"mask": keep,
                "threshold": float("nan"),
                "distances": np.zeros(n, dtype=np.float64)}
    raise ValueError(f"unknown method: {method!r}")
