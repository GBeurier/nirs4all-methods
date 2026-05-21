#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Validate the frozen NumPy wavelets reference against installed PyWavelets.

Historical once-only validation used PyWavelets 1.6.0. This script confirms
that the pure-NumPy reference under
``n4m_parity_wavelets_ref`` matches ``pywt`` exactly for the supported
(family, mode) pairs.  After this script passes, the frozen reference
becomes the canonical parity floor for nirs4all-methods and pywt is
no longer required in the parity-fixture pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PKG_ROOT))

from n4m_parity_wavelets_ref import (  # noqa: E402
    haar_transform,
    wavelet_denoise_transform,
    wavelet_features_transform,
    wavelet_pca_fit_transform,
    wavelet_svd_fit_transform,
    wavelet_transform,
)
from n4m_parity_wavelets_ref.dwt import (  # noqa: E402
    dwt_1d,
    idwt_1d,
    wavedec_1d,
    waverec_1d,
)


def _seed_signal(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=n).astype(np.float64)


def _validate_dwt_idwt(pywt_mod) -> None:
    for family in ("haar", "db4", "sym4", "coif1"):
        for mode in ("periodization", "symmetric", "zero"):
            for n in (16, 17, 32, 33, 100, 101):
                x = _seed_signal(seed=n * 7 + hash(family + mode) % 1000, n=n)
                cA, cD = dwt_1d(x, family, mode)
                cA_p, cD_p = pywt_mod.dwt(x, family, mode=mode)
                if not np.allclose(cA, cA_p, atol=1e-12) or \
                   not np.allclose(cD, cD_p, atol=1e-12):
                    raise SystemExit(
                        f"DWT mismatch: family={family} mode={mode} n={n}")
                xr_p = pywt_mod.idwt(cA_p, cD_p, family, mode=mode)
                # idwt may return length > n; truncate.
                xr_p_trim = xr_p[:n]
                xr = idwt_1d(cA, cD, family, mode, n)
                if not np.allclose(xr, xr_p_trim, atol=1e-12):
                    raise SystemExit(
                        f"IDWT mismatch: family={family} mode={mode} n={n}")
    print("DWT / IDWT: parity verified (4 families x 3 modes x 6 lengths)")


def _validate_wavedec_waverec(pywt_mod) -> None:
    for family in ("haar", "db4", "sym4", "coif1"):
        for mode in ("periodization", "symmetric", "zero"):
            for n in (32, 100, 200):
                max_lvl = pywt_mod.dwt_max_level(n, family)
                for level in range(1, max_lvl + 1):
                    x = _seed_signal(seed=n * 11 + level + hash(family) % 100, n=n)
                    cs_my = wavedec_1d(x, family, mode, level)
                    cs_p = pywt_mod.wavedec(x, family, mode=mode, level=level)
                    for k, (a, b) in enumerate(zip(cs_my, cs_p)):
                        if not np.allclose(a, b, atol=1e-12):
                            raise SystemExit(
                                f"wavedec mismatch: family={family} mode={mode} "
                                f"n={n} level={level} index={k}")
                    x_my = waverec_1d(cs_my, family, mode, n)
                    x_p = pywt_mod.waverec(cs_p, family, mode=mode)
                    if not np.allclose(x_my, x_p[:n], atol=1e-12):
                        raise SystemExit(
                            f"waverec mismatch: family={family} mode={mode} "
                            f"n={n} level={level}")
    print("wavedec / waverec: parity verified")


def _validate_operators(pywt_mod) -> None:
    rng = np.random.default_rng(20260518)
    X = rng.normal(size=(8, 64)).astype(np.float64)

    # Wavelet operator: cA || cD layout.
    out = wavelet_transform(X, family="db4", mode="periodization")
    for i in range(X.shape[0]):
        cA, cD = pywt_mod.dwt(X[i], "db4", mode="periodization")
        if not np.allclose(out[i, :len(cA)], cA, atol=1e-12):
            raise SystemExit("Wavelet operator cA mismatch")
        if not np.allclose(out[i, len(cA):], cD, atol=1e-12):
            raise SystemExit("Wavelet operator cD mismatch")

    # Haar operator (haar+periodization).
    out_h = haar_transform(X)
    out_w = wavelet_transform(X, family="haar", mode="periodization")
    if not np.allclose(out_h, out_w, atol=1e-12):
        raise SystemExit("Haar shortcut mismatch")

    # WaveletDenoise: with low noise the denoised output should be close to
    # input, with high noise it should differ.  Just check it runs without
    # NaNs and is finite.
    den = wavelet_denoise_transform(
        X, family="db4", mode="periodization", level=3,
        threshold_mode="soft", noise_estimator="median")
    if not np.all(np.isfinite(den)) or den.shape != X.shape:
        raise SystemExit("WaveletDenoise: bad output")

    # WaveletFeatures
    feat = wavelet_features_transform(
        X, family="db4", mode="periodization", max_level=3)
    if feat.shape != (8, 4 * 4):  # level=3 -> 4 bands -> 16 cols
        raise SystemExit(f"WaveletFeatures bad shape {feat.shape}")
    if not np.all(np.isfinite(feat)):
        raise SystemExit("WaveletFeatures: non-finite output")

    # WaveletPCA
    out_pca, k_pca = wavelet_pca_fit_transform(
        X, X, family="db4", mode="periodization", max_level=3,
        n_components=3.0)
    if out_pca.shape != (8, k_pca) or k_pca != 3:
        raise SystemExit(f"WaveletPCA bad shape {out_pca.shape} k={k_pca}")

    # WaveletSVD
    out_svd, k_svd = wavelet_svd_fit_transform(
        X, X, family="db4", mode="periodization", max_level=3,
        n_components=3.0)
    if out_svd.shape != (8, k_svd) or k_svd != 3:
        raise SystemExit(f"WaveletSVD bad shape {out_svd.shape} k={k_svd}")

    print("Operator references: smoke verified")


def main() -> None:
    try:
        import pywt  # type: ignore
    except ImportError:
        print("pywt is not installed; install PyWavelets from "
              "parity/python_generator/requirements-lock.txt to run the "
              "validation.  Skipping (frozen reference assumed valid).")
        return
    print(f"PyWavelets version: {pywt.__version__}")
    _validate_dwt_idwt(pywt)
    _validate_wavedec_waverec(pywt)
    _validate_operators(pywt)
    print("frozen wavelets reference: validation passed.")


if __name__ == "__main__":
    main()
