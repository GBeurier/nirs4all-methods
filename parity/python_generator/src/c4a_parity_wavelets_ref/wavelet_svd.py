# SPDX-License-Identifier: CECILL-2.1
"""WaveletSVD reference: DWT-flatten + Truncated SVD (no centering)."""

from __future__ import annotations

import numpy as np

from .filters import dwt_max_level
from .wavelet_pca import _flatten_matrix, _select_k, _svd_flip_u_based


def wavelet_svd_fit_transform(
    fit_X: np.ndarray,
    transform_X: np.ndarray,
    *,
    family: str,
    mode: str,
    max_level: int,
    n_components: float,
) -> tuple[np.ndarray, int]:
    fit_X = np.ascontiguousarray(fit_X, dtype=np.float64)
    transform_X = np.ascontiguousarray(transform_X, dtype=np.float64)
    if fit_X.ndim != 2 or transform_X.ndim != 2:
        raise ValueError("fit_X and transform_X must be 2-D")
    if transform_X.shape[1] != fit_X.shape[1]:
        raise ValueError("transform_X must share n_features with fit_X")

    m, n = fit_X.shape
    if m < 2:
        raise ValueError("WaveletSVD: fit_X needs m >= 2")

    eff_level = min(max_level, dwt_max_level(n, family))
    F = _flatten_matrix(fit_X, family, mode, eff_level)
    Ft = _flatten_matrix(transform_X, family, mode, eff_level)

    U, S, Vt = np.linalg.svd(F, full_matrices=False)
    U, Vt = _svd_flip_u_based(U, Vt)

    k_full = min(m, F.shape[1])
    sq = S ** 2
    total = float(sq.sum())
    var_ratio = (sq / total) if total > 0.0 else np.zeros_like(sq)
    k = _select_k(float(n_components), k_full, var_ratio)
    components = Vt[:k, :]
    out = Ft @ components.T
    return np.ascontiguousarray(out, dtype=np.float64), k
