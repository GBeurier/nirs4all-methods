# SPDX-License-Identifier: CECILL-2.1
"""WaveletPCA reference: DWT-flatten + PCA on flattened coefficients."""

from __future__ import annotations

import numpy as np

from .dwt import wavedec_1d
from .filters import dwt_max_level, dwt_output_length


def _flatten_row(x: np.ndarray, family: str, mode: str, level: int) -> np.ndarray:
    coeffs = wavedec_1d(x, family, mode, level)
    return np.concatenate(coeffs)


def _flatten_matrix(X: np.ndarray, family: str, mode: str,
                     level: int) -> np.ndarray:
    rows, cols = X.shape
    flat_dim = 0
    cur_len = cols
    for _ in range(level):
        cur_len = dwt_output_length(cur_len, family, mode)
        flat_dim += cur_len
    flat_dim += cur_len  # final approximation
    out = np.zeros((rows, flat_dim), dtype=np.float64)
    for i in range(rows):
        out[i] = _flatten_row(X[i], family, mode, level)
    return out


def _svd_flip_u_based(U: np.ndarray, Vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_abs = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs, np.arange(U.shape[1])])
    signs = np.where(signs == 0.0, 1.0, signs)
    return U * signs[np.newaxis, :], Vt * signs[:, np.newaxis]


def _select_k(param: float, k_full: int, var_ratio: np.ndarray) -> int:
    if param >= 1.0:
        return max(1, min(int(param), k_full))
    cum = np.cumsum(var_ratio)
    idx = int(np.searchsorted(cum, param, side="left"))
    return min(idx + 1, k_full) if idx < k_full else k_full


def wavelet_pca_fit_transform(
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
        raise ValueError("WaveletPCA: fit_X needs m >= 2")

    eff_level = min(max_level, dwt_max_level(n, family))
    F = _flatten_matrix(fit_X, family, mode, eff_level)
    Ft = _flatten_matrix(transform_X, family, mode, eff_level)
    mean = F.mean(axis=0)
    Fc = F - mean

    U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    U, Vt = _svd_flip_u_based(U, Vt)

    k_full = min(m, F.shape[1])
    evar = (S ** 2) / (m - 1)
    total = float(evar.sum())
    var_ratio = (evar / total) if total > 0.0 else np.zeros_like(evar)
    k = _select_k(float(n_components), k_full, var_ratio)
    components = Vt[:k, :]
    out = (Ft - mean) @ components.T
    return np.ascontiguousarray(out, dtype=np.float64), k
