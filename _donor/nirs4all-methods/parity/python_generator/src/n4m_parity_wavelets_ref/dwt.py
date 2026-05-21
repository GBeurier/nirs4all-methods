# SPDX-License-Identifier: CECILL-2.1
"""
Low-level DWT / IDWT / wavedec / waverec implementations.

Pure NumPy, mirrors the n4m C engine (which in turn mirrors PyWavelets
1.6.0 for ``haar / db4 / sym4 / coif1`` and modes
``periodization / symmetric / zero``).

The algorithms used here are:

  - Periodization DWT:
        n_eff = n + (n & 1)  (append last sample for odd n)
        out[k] = sum_{i=0}^{L-1} h[L-1-i] * x_eff[(2*k + i - (L/2 - 1)) % n_eff]
  - Symmetric / Zero DWT:
        extend signal by L-1 on each side (edge-repeat reflection or zeros)
        out[k] = np.convolve(x_ext, h)[L::2][:out_len]

  - Periodization IDWT:
        upsample (zeros at odd positions), convolve with rec filters, sum,
        fold mod 2*m, roll by L/2 - 1, take first n_out samples
  - Symmetric / Zero IDWT:
        upsample, convolve with rec filters, sum, trim ``[L-2 : L-2 + n_out]``
"""

from __future__ import annotations

import numpy as np

from .filters import FILTER_BANKS, dwt_output_length


def _periodization_index(k: int, n_eff: int) -> int:
    r = k % n_eff
    return r if r >= 0 else r + n_eff


def dwt_1d(x: np.ndarray, family: str, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Single-level DWT of a 1-D signal."""
    x = np.asarray(x, dtype=np.float64)
    n = int(x.shape[0])
    if n < 1:
        raise ValueError("dwt requires n >= 1")
    dec_lo, dec_hi, _rec_lo, _rec_hi = FILTER_BANKS[family]
    L = len(dec_lo)
    out_len = dwt_output_length(n, family, mode)
    cA = np.zeros(out_len, dtype=np.float64)
    cD = np.zeros(out_len, dtype=np.float64)
    if mode == "periodization":
        if n & 1:
            x_eff = np.concatenate([x, [x[-1]]])
        else:
            x_eff = x
        n_eff = x_eff.shape[0]
        half_offset = L // 2 - 1
        for k in range(out_len):
            for i in range(L):
                idx = _periodization_index(2 * k + i - half_offset, n_eff)
                v = float(x_eff[idx])
                cA[k] += dec_lo[L - 1 - i] * v
                cD[k] += dec_hi[L - 1 - i] * v
        return cA, cD
    if mode == "symmetric":
        x_ext = np.pad(x, L - 1, mode="symmetric")
    elif mode == "zero":
        x_ext = np.concatenate([np.zeros(L - 1), x, np.zeros(L - 1)])
    else:
        raise ValueError(f"unsupported mode: {mode}")
    cA = np.convolve(x_ext, np.asarray(dec_lo, dtype=np.float64))[L::2][:out_len]
    cD = np.convolve(x_ext, np.asarray(dec_hi, dtype=np.float64))[L::2][:out_len]
    return (np.ascontiguousarray(cA, dtype=np.float64),
            np.ascontiguousarray(cD, dtype=np.float64))


def idwt_1d(cA: np.ndarray, cD: np.ndarray, family: str, mode: str,
             n_out: int) -> np.ndarray:
    """Single-level inverse DWT trimmed to length ``n_out``."""
    cA = np.asarray(cA, dtype=np.float64)
    cD = np.asarray(cD, dtype=np.float64) if cD is not None else np.zeros_like(cA)
    m = int(cA.shape[0])
    _dec_lo, _dec_hi, rec_lo, rec_hi = FILTER_BANKS[family]
    L = len(rec_lo)
    up_cA = np.zeros(2 * m, dtype=np.float64)
    up_cD = np.zeros(2 * m, dtype=np.float64)
    up_cA[::2] = cA
    up_cD[::2] = cD
    full = (np.convolve(up_cA, np.asarray(rec_lo, dtype=np.float64)) +
            np.convolve(up_cD, np.asarray(rec_hi, dtype=np.float64)))
    if mode == "symmetric" or mode == "zero":
        out = full[L - 2:L - 2 + n_out]
        return np.ascontiguousarray(out, dtype=np.float64)
    if mode == "periodization":
        period = 2 * m
        wrapped = np.zeros(period, dtype=np.float64)
        for i in range(full.shape[0]):
            wrapped[i % period] += full[i]
        roll = L // 2 - 1
        wrapped = np.roll(wrapped, -roll)
        return np.ascontiguousarray(wrapped[:n_out], dtype=np.float64)
    raise ValueError(f"unsupported mode: {mode}")


def wavedec_1d(x: np.ndarray, family: str, mode: str, level: int) -> list[np.ndarray]:
    """Multi-level decomposition.  Returns ``[cA_n, cD_n, cD_{n-1}, ..., cD_1]``."""
    coeffs: list[np.ndarray] = []
    cur = np.asarray(x, dtype=np.float64)
    details_reversed: list[np.ndarray] = []
    for _ in range(level):
        cA, cD = dwt_1d(cur, family, mode)
        details_reversed.append(cD)
        cur = cA
    coeffs.append(cur)
    coeffs.extend(reversed(details_reversed))
    return coeffs


def waverec_1d(coeffs: list[np.ndarray], family: str, mode: str,
                n_out: int) -> np.ndarray:
    """Multi-level reconstruction trimmed to length ``n_out``."""
    if not coeffs:
        raise ValueError("waverec_1d: empty coefficient list")
    if len(coeffs) == 1:
        out = np.asarray(coeffs[0], dtype=np.float64)[:n_out]
        return np.ascontiguousarray(out, dtype=np.float64)
    # Compute intermediate approximation lengths from the final n_out.
    level = len(coeffs) - 1
    approx_lens = [n_out]
    for _ in range(level):
        approx_lens.append(dwt_output_length(approx_lens[-1], family, mode))
    cur = np.asarray(coeffs[0], dtype=np.float64)
    for k in range(level, 0, -1):
        det = np.asarray(coeffs[level - k + 1], dtype=np.float64)
        target_len = approx_lens[k - 1]
        cur = idwt_1d(cur, det, family, mode, target_len)
    return np.ascontiguousarray(cur[:n_out], dtype=np.float64)
