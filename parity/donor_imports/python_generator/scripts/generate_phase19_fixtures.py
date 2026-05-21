#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 19 parity fixtures.

Three categories:

  * SignalTypeDetector — deterministic enum + confidence + reason string
    produced by `nirs4all.data.signal_type.SignalTypeDetector.detect`.
    Cases include the five canonical signal classes plus a "preprocessed"
    sentinel and an "ambiguous" low-confidence case. Optional wavelength
    axis is exercised on one absorbance case.

  * NIRS regression metrics — 8 closed-form metrics (RMSE, MAE, BIAS, SEP,
    RPD, RPIQ, R², NRMSE) computed against NumPy / pure Python with the
    same evaluation order as `nirs4all.core.metrics._eval_single`.

  * Hotelling T² + Q-residuals — multivariate outlier statistics computed
    against LAPACK SVD (`numpy.linalg.svd`) and scipy.stats.f /
    scipy.stats.norm for the UCL.

Output: four JSON fixture files under ``parity/fixtures/``:

    signal_type_detect_v1.json
    nirs_metrics_v1.json
    hotelling_t2_v1.json
    q_residuals_v1.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from struct import pack
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Encoding helpers (mirror Phase 5).
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray | Iterable[float]) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# SignalTypeDetector — port of nirs4all.data.signal_type.SignalTypeDetector.
# The implementation is reproduced inline so the parity generator stays
# self-contained (does not depend on the nirs4all install). The semantics
# match the canonical Python reference 1:1.
# ---------------------------------------------------------------------------

# C enum integer ordering (must match c_api_phase19.h):
SIG_UNKNOWN               = 0
SIG_ABSORBANCE            = 1
SIG_REFLECTANCE           = 2
SIG_REFLECTANCE_PERCENT   = 3
SIG_TRANSMITTANCE         = 4
SIG_TRANSMITTANCE_PERCENT = 5


def detect_signal_type(spectra: np.ndarray,
                       wavelengths: np.ndarray | None,
                       confidence_threshold: float):
    if spectra.size == 0:
        return SIG_UNKNOWN, 0.0, "Empty data"

    data = spectra.flatten() if spectra.ndim == 1 else spectra
    finite_mask = ~np.isnan(data)
    if not finite_mask.any():
        return SIG_UNKNOWN, 0.0, "All-NaN data"

    min_val  = float(np.nanmin(data))
    max_val  = float(np.nanmax(data))
    mean_val = float(np.nanmean(data))
    # Population std (NumPy default; ddof=0).
    std_val  = float(np.nanstd(data))

    # Preprocessed?
    preproc = False
    if abs(mean_val) < 0.01 and std_val > 0.1:
        preproc = True
    elif abs(std_val - 1.0) < 0.1 and abs(mean_val) < 0.1:
        preproc = True
    elif min_val < -0.5 and max_val < 0.5 and abs(mean_val) < 0.01:
        preproc = True
    if preproc:
        return SIG_UNKNOWN, 0.9, "Data appears preprocessed (centered/normalized)"

    # Scores.
    def s_refl_frac(mn, mx, m):
        s = 0.0
        if mn >= 0.0 and mx <= 1.2:
            s += 0.5
            if 0.1 <= m <= 0.8: s += 0.3
            if mx <= 1.0:       s += 0.2
        return s

    def s_refl_pct(mn, mx, m):
        s = 0.0
        if mn >= 0.0 and 1.5 < mx <= 120.0:
            s += 0.5
            if 10.0 <= m <= 80.0: s += 0.3
            if mx <= 100.0:        s += 0.2
        return s

    def s_trans_frac(mn, mx, m):
        s = 0.0
        if mn >= 0.0 and mx <= 1.2:
            s += 0.4
            if 0.05 <= m <= 0.5: s += 0.2
        return s

    def s_trans_pct(mn, mx, m):
        s = 0.0
        if mn >= 0.0 and 1.5 < mx <= 120.0:
            s += 0.4
            if 5.0 <= m <= 50.0: s += 0.2
        return s

    def s_abs(mn, mx, m):
        s = 0.0
        if mn >= -0.5 and 0.5 <= mx <= 5.0:
            s += 0.4
            if 0.2 <= m <= 2.0: s += 0.3
            if mn >= -0.2:      s += 0.2
            if mx >= 1.0:       s += 0.1
        return s

    scores = {
        SIG_REFLECTANCE:           s_refl_frac(min_val, max_val, mean_val),
        SIG_REFLECTANCE_PERCENT:   s_refl_pct(min_val, max_val, mean_val),
        SIG_TRANSMITTANCE:         s_trans_frac(min_val, max_val, mean_val),
        SIG_TRANSMITTANCE_PERCENT: s_trans_pct(min_val, max_val, mean_val),
        SIG_ABSORBANCE:            s_abs(min_val, max_val, mean_val),
    }

    # Optional water-band cue.
    if wavelengths is not None and spectra.ndim == 2 and \
       len(wavelengths) == spectra.shape[1]:
        bands_nm = [1450.0, 1940.0, 2500.0]
        mean_spec = np.nanmean(spectra, axis=0)
        peak_count = 0
        dip_count = 0
        wl_min, wl_max = float(np.min(wavelengths)), float(np.max(wavelengths))
        for band in bands_nm:
            if wl_min <= band <= wl_max:
                idx = int(np.argmin(np.abs(wavelengths - band)))
                start = max(0, idx - 10)
                end   = min(len(mean_spec), idx + 10)
                local_mean = float(np.nanmean(mean_spec[start:end]))
                band_value = float(mean_spec[idx])
                if band_value > local_mean * 1.05:
                    peak_count += 1
                elif band_value < local_mean * 0.95:
                    dip_count += 1
        if peak_count > dip_count:
            scores[SIG_ABSORBANCE]    += 0.3 * 0.2
            scores[SIG_REFLECTANCE]   += -0.1 * 0.2
            scores[SIG_TRANSMITTANCE] += -0.1 * 0.2
        elif dip_count > peak_count:
            scores[SIG_ABSORBANCE]    += -0.1 * 0.2
            scores[SIG_REFLECTANCE]   +=  0.2 * 0.2
            scores[SIG_TRANSMITTANCE] +=  0.2 * 0.2

    best  = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    confidence = scores[best] / total if total > 0.0 else 0.0
    label = {
        SIG_ABSORBANCE:            "absorbance",
        SIG_REFLECTANCE:           "reflectance",
        SIG_REFLECTANCE_PERCENT:   "reflectance%",
        SIG_TRANSMITTANCE:         "transmittance",
        SIG_TRANSMITTANCE_PERCENT: "transmittance%",
    }[best]
    reason = (f"Range: [{min_val:.3f}, {max_val:.3f}] | "
              f"Mean: {mean_val:.3f} | Detected: {label} | "
              f"Confidence: {confidence:.1%}")
    if confidence < confidence_threshold:
        return SIG_UNKNOWN, confidence, reason
    return best, confidence, reason


# ---------------------------------------------------------------------------
# Synthetic data generators.
# ---------------------------------------------------------------------------

SEED = 20260519


def gen_reflectance(rng: np.random.Generator) -> np.ndarray:
    """A typical reflectance dataset: values in [0.05, 0.95]."""
    X = rng.uniform(0.1, 0.85, size=(8, 60)).astype(np.float64)
    return X


def gen_reflectance_percent(rng: np.random.Generator) -> np.ndarray:
    return gen_reflectance(rng) * 100.0


def gen_transmittance(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.05, 0.45, size=(6, 50)).astype(np.float64)


def gen_transmittance_percent(rng: np.random.Generator) -> np.ndarray:
    return gen_transmittance(rng) * 100.0


def gen_absorbance(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.3, 1.8, size=(8, 64)).astype(np.float64)


def gen_preprocessed_snv(rng: np.random.Generator) -> np.ndarray:
    """Standardised data — mean ~ 0, std ~ 1."""
    return rng.standard_normal(size=(10, 50)).astype(np.float64)


def gen_ambiguous(rng: np.random.Generator) -> np.ndarray:
    """Values straddle multiple classes (e.g. ~[0, 1.4])."""
    return rng.uniform(0.0, 1.4, size=(5, 30)).astype(np.float64)


def gen_absorbance_with_bands(rng: np.random.Generator):
    """Absorbance synthetic spectrum with peaks at NIR water bands."""
    wl = np.linspace(1100.0, 2500.0, 80, dtype=np.float64)
    n = 6
    X = np.zeros((n, len(wl)), dtype=np.float64)
    for i in range(n):
        baseline = 0.5 + 0.0001 * (wl - 1500.0)
        peaks = 0.0
        for centre, amp in [(1450.0, 1.2), (1940.0, 1.5), (2500.0, 1.0)]:
            peaks = peaks + amp * np.exp(-((wl - centre) ** 2) / (2.0 * 40.0 ** 2))
        X[i] = baseline + peaks + 0.01 * rng.standard_normal(len(wl))
    return X, wl


# ---------------------------------------------------------------------------
# Signal type fixture.
# ---------------------------------------------------------------------------

def build_signal_type_cases(rng: np.random.Generator):
    """Hand-picked thresholds that exercise both the "definite" and
    "ambiguous → UNKNOWN" branches. The Python heuristic is fundamentally
    overlap-sensitive (reflectance and transmittance fractions live in the
    same value range), so the score distribution rarely concentrates above
    0.7 on synthetic data; we lower the threshold to 0.3 for the
    overlap-dominated cases and keep 0.7 for the clearly-out-of-range
    classes.
    """
    cases = []

    X = gen_reflectance(rng)
    t, c, r = detect_signal_type(X, None, 0.3)
    cases.append(("reflectance_typical", X, None, 0.3, t, c, r))

    X = gen_reflectance_percent(rng)
    t, c, r = detect_signal_type(X, None, 0.3)
    cases.append(("reflectance_percent", X, None, 0.3, t, c, r))

    X = gen_transmittance(rng)
    t, c, r = detect_signal_type(X, None, 0.3)
    cases.append(("transmittance_low", X, None, 0.3, t, c, r))

    X = gen_transmittance_percent(rng)
    t, c, r = detect_signal_type(X, None, 0.3)
    cases.append(("transmittance_percent", X, None, 0.3, t, c, r))

    X = gen_absorbance(rng)
    t, c, r = detect_signal_type(X, None, 0.3)
    cases.append(("absorbance_typical", X, None, 0.3, t, c, r))

    X = gen_preprocessed_snv(rng)
    t, c, r = detect_signal_type(X, None, 0.7)
    cases.append(("preprocessed_snv", X, None, 0.7, t, c, r))

    # Ambiguous case: very low threshold to exercise the "definite" path
    # on a noisy multi-class score distribution.
    X = gen_ambiguous(rng)
    t, c, r = detect_signal_type(X, None, 0.05)
    cases.append(("ambiguous_low_conf", X, None, 0.05, t, c, r))

    X, wl = gen_absorbance_with_bands(rng)
    t, c, r = detect_signal_type(X, wl, 0.3)
    cases.append(("absorbance_with_water_bands", X, wl, 0.3, t, c, r))

    # Threshold > best confidence so the result downgrades to UNKNOWN.
    X = gen_reflectance(rng)
    t, c, r = detect_signal_type(X, None, 0.95)
    cases.append(("reflectance_forced_unknown", X, None, 0.95, t, c, r))

    return cases


def write_signal_type_fixture(cases, out_path: Path) -> None:
    fixture: dict[str, Any] = {
        "format": "n4m_signal_detect_v1",
        "numpy_version": np.__version__,
        "reference": "nirs4all.data.signal_type.SignalTypeDetector (ported inline)",
        "encoding": "ieee754_binary64_be_hex",
        "cases": [],
    }
    for name, X, wl, threshold, type_out, confidence, reason in cases:
        entry: dict[str, Any] = {
            "name": name,
            "rows": int(X.shape[0]),
            "cols": int(X.shape[1]),
            "input_hex": array_to_hex(X),
            "confidence_threshold": float(threshold),
            "expected_type": int(type_out),
            "expected_confidence_hex": double_to_hex(float(confidence)),
            "expected_reason": reason,
        }
        if wl is not None:
            entry["wavelengths_hex"] = array_to_hex(wl)
            entry["wl_length"] = int(len(wl))
        else:
            entry["wl_length"] = 0
        fixture["cases"].append(entry)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes, "
          f"{len(cases)} cases)")


# ---------------------------------------------------------------------------
# NIRS regression metrics fixture.
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    n = len(y_true)
    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae  = float(np.mean(np.abs(diff)))
    bias = float(np.mean(y_pred - y_true))
    sep  = float(np.std(y_pred - y_true))         # ddof=0
    sd_y = float(np.std(y_true))                  # ddof=0
    rpd  = sd_y / sep if sep != 0.0 else float('inf')
    iqr  = float(np.percentile(y_true, 75) - np.percentile(y_true, 25))
    rpiq = iqr / rmse if rmse != 0.0 else float('inf')
    # R² via SSE / SST.
    y_mean = float(np.mean(y_true))
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - y_mean) ** 2))
    if sst == 0.0:
        r2 = 1.0 if sse == 0.0 else 0.0
    else:
        r2 = 1.0 - sse / sst
    # NRMSE: rmse / mean(y_true).
    nrmse = rmse / y_mean if y_mean != 0.0 else float('inf')
    return {
        "rmse":  rmse,
        "mae":   mae,
        "bias":  bias,
        "sep":   sep,
        "rpd":   rpd,
        "rpiq":  rpiq,
        "r2":    r2,
        "nrmse": nrmse,
    }


def build_metric_cases(rng: np.random.Generator):
    cases = []
    # Typical regression.
    n = 200
    y_true = rng.uniform(0.0, 10.0, size=n).astype(np.float64)
    y_pred = y_true + 0.2 * rng.standard_normal(n).astype(np.float64) + 0.05
    cases.append(("typical_n200", y_true, y_pred))
    # Larger dataset with bigger noise.
    n = 500
    y_true = rng.uniform(20.0, 80.0, size=n).astype(np.float64)
    y_pred = y_true + 1.0 * rng.standard_normal(n).astype(np.float64)
    cases.append(("n500_loud", y_true, y_pred))
    # Small noisy.
    n = 30
    y_true = rng.uniform(-5.0, 5.0, size=n).astype(np.float64)
    y_pred = y_true + 0.4 * rng.standard_normal(n).astype(np.float64)
    cases.append(("small_n30_centred", y_true, y_pred))
    # Perfect.
    n = 100
    y_true = rng.uniform(1.0, 4.0, size=n).astype(np.float64)
    y_pred = y_true.copy()
    cases.append(("perfect_match", y_true, y_pred))
    return cases


def write_metrics_fixture(cases, out_path: Path) -> None:
    fixture: dict[str, Any] = {
        "format": "n4m_metric_v1",
        "numpy_version": np.__version__,
        "reference": "nirs4all.core.metrics (closed-form NumPy reduction)",
        "encoding": "ieee754_binary64_be_hex",
        "cases": [],
    }
    for name, y_true, y_pred in cases:
        m = compute_metrics(y_true, y_pred)
        entry: dict[str, Any] = {
            "name": name,
            "n": int(len(y_true)),
            "y_true_hex": array_to_hex(y_true),
            "y_pred_hex": array_to_hex(y_pred),
            "metrics_hex": {k: double_to_hex(v) for k, v in m.items()},
        }
        fixture["cases"].append(entry)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes, "
          f"{len(cases)} cases)")


# ---------------------------------------------------------------------------
# Hotelling T² + Q-residuals fixture.
# ---------------------------------------------------------------------------

def hotelling_t2_lapack(X: np.ndarray, k: int, alpha: float):
    """Reference T² using numpy.linalg.svd. Returns (per_sample_t2, ucl)."""
    n, p = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    # full_matrices=False -> U: (n, min(n,p)), S: (min(n,p),), Vt: (min(n,p), p)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    # T = U Σ
    T = U[:, :k] * S[:k]
    lambdas = (S[:k] ** 2) / (n - 1)
    t2 = np.zeros(n, dtype=np.float64)
    nonzero = lambdas > 0.0
    if nonzero.any():
        t2 = np.sum((T[:, nonzero] ** 2) / lambdas[nonzero], axis=1)
    # UCL via scipy.stats.f for accuracy.
    try:
        from scipy.stats import f as f_dist
        Fq = float(f_dist.ppf(1.0 - alpha, k, n - k))
    except ImportError:  # pragma: no cover
        # Conservative fallback (won't be hit in CI which pins scipy).
        Fq = 0.0
    ucl = k * (n - 1) * (n + 1) / (n * (n - k)) * Fq
    return t2.astype(np.float64), float(ucl)


def q_residuals_lapack(X: np.ndarray, k: int, alpha: float):
    """Reference Q-residuals via numpy.linalg.svd + Jackson-Mudholkar UCL."""
    n, p = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Reconstruct on first k components.
    rec = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    R = Xc - rec
    q = np.sum(R ** 2, axis=1)

    # Jackson-Mudholkar UCL.
    lambdas = (S ** 2) / (n - 1)
    tail = lambdas[k:]
    theta1 = float(np.sum(tail))
    theta2 = float(np.sum(tail ** 2))
    theta3 = float(np.sum(tail ** 3))
    ucl = 0.0
    if theta1 > 0.0 and theta2 > 0.0:
        h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2 ** 2)
        from scipy.stats import norm
        ca = float(norm.ppf(1.0 - alpha))
        term1 = ca * np.sqrt(2.0 * theta2 * h0 ** 2) / theta1
        term2 = theta2 * h0 * (h0 - 1.0) / (theta1 ** 2)
        base = 1.0 + term1 + term2
        if base > 0.0 and h0 != 0.0:
            ucl = theta1 * (base ** (1.0 / h0))
    return q.astype(np.float64), float(ucl)


def build_t2_q_cases(rng: np.random.Generator):
    """Three datasets with various shapes & noise floors."""
    cases = []
    # Case 1: small, low-rank with structure.
    n, p = 50, 12
    A = rng.standard_normal((n, 3))
    L = rng.standard_normal((3, p))
    X = A @ L + 0.05 * rng.standard_normal((n, p))
    cases.append(("structured_50x12_k3", X.astype(np.float64), 3, 0.05))

    # Case 2: medium PCA, 5 components.
    n, p = 80, 20
    A = rng.standard_normal((n, 5))
    L = rng.standard_normal((5, p))
    X = A @ L + 0.1 * rng.standard_normal((n, p))
    cases.append(("structured_80x20_k5", X.astype(np.float64), 5, 0.05))

    # Case 3: full-rank random Gaussian.
    n, p = 40, 8
    X = rng.standard_normal((n, p))
    cases.append(("random_40x8_k4", X.astype(np.float64), 4, 0.05))

    return cases


def write_t2_fixture(cases, out_path: Path) -> None:
    fixture: dict[str, Any] = {
        "format": "n4m_util_hotelling_t2_v1",
        "numpy_version": np.__version__,
        "reference": "numpy.linalg.svd + scipy.stats.f",
        "encoding": "ieee754_binary64_be_hex",
        "cases": [],
    }
    for name, X, k, alpha in cases:
        t2, ucl = hotelling_t2_lapack(X, k, alpha)
        entry: dict[str, Any] = {
            "name":         name,
            "rows":         int(X.shape[0]),
            "cols":         int(X.shape[1]),
            "n_components": int(k),
            "alpha":        float(alpha),
            "input_hex":    array_to_hex(X),
            "t2_hex":       array_to_hex(t2),
            "ucl_hex":      double_to_hex(ucl),
        }
        fixture["cases"].append(entry)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes, "
          f"{len(cases)} cases)")


def write_q_fixture(cases, out_path: Path) -> None:
    fixture: dict[str, Any] = {
        "format": "n4m_util_q_residuals_v1",
        "numpy_version": np.__version__,
        "reference": "numpy.linalg.svd + Jackson-Mudholkar approximation",
        "encoding": "ieee754_binary64_be_hex",
        "cases": [],
    }
    for name, X, k, alpha in cases:
        q, ucl = q_residuals_lapack(X, k, alpha)
        entry: dict[str, Any] = {
            "name":         name,
            "rows":         int(X.shape[0]),
            "cols":         int(X.shape[1]),
            "n_components": int(k),
            "alpha":        float(alpha),
            "input_hex":    array_to_hex(X),
            "q_hex":        array_to_hex(q),
            "ucl_hex":      double_to_hex(ucl),
        }
        fixture["cases"].append(entry)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes, "
          f"{len(cases)} cases)")


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = REPO_ROOT / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    signal_cases = build_signal_type_cases(rng)
    write_signal_type_fixture(signal_cases,
                              out_dir / "signal_type_detect_v1.json")

    metric_cases = build_metric_cases(rng)
    write_metrics_fixture(metric_cases, out_dir / "nirs_metrics_v1.json")

    t2_q_cases = build_t2_q_cases(rng)
    write_t2_fixture(t2_q_cases, out_dir / "hotelling_t2_v1.json")
    write_q_fixture(t2_q_cases, out_dir / "q_residuals_v1.json")


if __name__ == "__main__":
    main()
