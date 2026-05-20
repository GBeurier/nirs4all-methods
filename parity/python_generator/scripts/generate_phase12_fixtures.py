#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 12 parity fixtures for the Y outlier filter.

One operator, four methods (each emitted as a separate fixture file):

  * iqr        — ``filter_y_outlier_iqr_v1.json``
  * zscore     — ``filter_y_outlier_zscore_v1.json``
  * percentile — ``filter_y_outlier_percentile_v1.json``
  * mad        — ``filter_y_outlier_mad_v1.json``

The reference comes from the frozen ``n4m_parity_filters_ref`` package under
``parity/python_generator/src/``. It has been validated once against
``nirs4all.operators.filters.YOutlierFilter`` and is now the canonical parity
floor (so subsequent nirs4all releases can drift without breaking the
nirs4all-methods parity gates).

Fixture schema (per file) — distinct from the preprocessing fixtures because
filters consume a 1-D `y` and produce a `uint8` mask + a stats record:

  {
    "format": "n4m_filter_y_outlier_<method>_v1",
    "numpy_version": "...",
    "reference": "n4m_parity_filters_ref (frozen against nirs4all 0.8.x)",
    "encoding": "ieee754_binary64_be_hex",
    "n": <int>,                  // length of y
    "y_hex": [ "<16 hex>", ... ], // input y values
    "cases": [
      { "name": "...",
        "params": { ... },
        "expected_mask": [0|1, ...],
        "expected_stats": {
          "n_samples": <int>, "n_kept": <int>, "n_excluded": <int>,
          "exclusion_rate": <hex 16> } },
      ...
    ]
  }

Each fixture file synthesises a different y vector — different distributions
to exercise the bound-computation paths. All fixtures cover the same case
catalog (default + edge cases) for the chosen method.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from struct import pack
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "parity" / "python_generator" / "src"))

from n4m_parity_filters_ref import y_outlier_fit_get_mask  # noqa: E402


# ---------------------------------------------------------------------------
# Encoding helpers.
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# Synthetic y distributions — chosen to exercise different bound paths.
# ---------------------------------------------------------------------------

SEED = 20260518
N = 200


def synth_y_gaussian_with_outliers(seed: int) -> np.ndarray:
    """Mostly Gaussian with a handful of +/- 6 sigma outliers + NaN tails."""
    rng = np.random.default_rng(seed)
    y = rng.normal(loc=10.0, scale=2.0, size=N)
    # Plant outliers at known indices.
    y[5]   = 30.0    # high outlier
    y[20]  = -10.0   # low outlier
    y[50]  = 25.0
    y[123] = -5.0
    # Plant NaNs (must always be excluded).
    y[7]   = np.nan
    y[100] = np.nan
    return y


def synth_y_skewed(seed: int) -> np.ndarray:
    """Right-skewed log-normal — IQR/MAD bound shapes differ noticeably."""
    rng = np.random.default_rng(seed + 1)
    return rng.lognormal(mean=0.0, sigma=0.6, size=N) * 5.0


def synth_y_uniform(seed: int) -> np.ndarray:
    """Bounded uniform — percentile method exercise."""
    rng = np.random.default_rng(seed + 2)
    return rng.uniform(low=-5.0, high=5.0, size=N)


def synth_y_bimodal(seed: int) -> np.ndarray:
    """Bimodal mixture — Z-score and MAD diverge more visibly."""
    rng = np.random.default_rng(seed + 3)
    a = rng.normal(loc=-3.0, scale=0.5, size=N // 2)
    b = rng.normal(loc=+3.0, scale=0.5, size=N - N // 2)
    return np.concatenate([a, b])


# ---------------------------------------------------------------------------
# Case lists per method.
# ---------------------------------------------------------------------------

Case = tuple[str, dict, Callable[[], tuple[np.ndarray, dict]]]


def iqr_cases(y: np.ndarray) -> list[Case]:
    return [
        ("default",
         {"method": "iqr", "threshold": 1.5,
          "lower_percentile": 1.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(y, method="iqr", threshold=1.5)),
        ("strict",
         {"method": "iqr", "threshold": 1.0,
          "lower_percentile": 1.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(y, method="iqr", threshold=1.0)),
        ("lenient",
         {"method": "iqr", "threshold": 3.0,
          "lower_percentile": 1.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(y, method="iqr", threshold=3.0)),
        ("very_strict",
         {"method": "iqr", "threshold": 0.5,
          "lower_percentile": 1.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(y, method="iqr", threshold=0.5)),
    ]


def zscore_cases(y: np.ndarray) -> list[Case]:
    return [
        ("default",
         {"method": "zscore", "threshold": 3.0,
          "lower_percentile": 1.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(y, method="zscore", threshold=3.0)),
        ("strict_2sigma",
         {"method": "zscore", "threshold": 2.0,
          "lower_percentile": 1.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(y, method="zscore", threshold=2.0)),
        ("lenient_4sigma",
         {"method": "zscore", "threshold": 4.0,
          "lower_percentile": 1.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(y, method="zscore", threshold=4.0)),
        ("very_strict_1sigma",
         {"method": "zscore", "threshold": 1.0,
          "lower_percentile": 1.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(y, method="zscore", threshold=1.0)),
    ]


def percentile_cases(y: np.ndarray) -> list[Case]:
    return [
        ("default",
         {"method": "percentile", "threshold": 1.5,
          "lower_percentile": 1.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(
             y, method="percentile", lower_percentile=1.0, upper_percentile=99.0)),
        ("strict_5_95",
         {"method": "percentile", "threshold": 1.5,
          "lower_percentile": 5.0, "upper_percentile": 95.0},
         lambda y=y: y_outlier_fit_get_mask(
             y, method="percentile", lower_percentile=5.0, upper_percentile=95.0)),
        ("loose_0_5_99_5",
         {"method": "percentile", "threshold": 1.5,
          "lower_percentile": 0.5, "upper_percentile": 99.5},
         lambda y=y: y_outlier_fit_get_mask(
             y, method="percentile", lower_percentile=0.5, upper_percentile=99.5)),
        ("asymmetric_10_99",
         {"method": "percentile", "threshold": 1.5,
          "lower_percentile": 10.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(
             y, method="percentile", lower_percentile=10.0, upper_percentile=99.0)),
    ]


def mad_cases(y: np.ndarray) -> list[Case]:
    return [
        ("default",
         {"method": "mad", "threshold": 3.5,
          "lower_percentile": 1.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(y, method="mad", threshold=3.5)),
        ("strict_2_5",
         {"method": "mad", "threshold": 2.5,
          "lower_percentile": 1.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(y, method="mad", threshold=2.5)),
        ("lenient_5",
         {"method": "mad", "threshold": 5.0,
          "lower_percentile": 1.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(y, method="mad", threshold=5.0)),
        ("very_strict_1_5",
         {"method": "mad", "threshold": 1.5,
          "lower_percentile": 1.0, "upper_percentile": 99.0},
         lambda y=y: y_outlier_fit_get_mask(y, method="mad", threshold=1.5)),
    ]


# ---------------------------------------------------------------------------
# Fixture writer.
# ---------------------------------------------------------------------------

def write_fixture(method: str, y: np.ndarray, cases: list[Case],
                  out_dir: Path) -> None:
    fixture: dict[str, Any] = {
        "format": f"n4m_filter_y_outlier_{method}_v1",
        "numpy_version": np.__version__,
        "reference": "n4m_parity_filters_ref (frozen against nirs4all 0.8.x)",
        "encoding": "ieee754_binary64_be_hex",
        "n": int(y.shape[0]),
        "y_hex": array_to_hex(y),
        "cases": [],
    }
    for case_name, params, fn in cases:
        mask, stats = fn()
        fixture["cases"].append({
            "name": case_name,
            "params": params,
            "expected_mask": [int(v) for v in mask.tolist()],
            "expected_stats": {
                "n_samples": int(stats["n_samples"]),
                "n_kept": int(stats["n_kept"]),
                "n_excluded": int(stats["n_excluded"]),
                "exclusion_rate": double_to_hex(stats["exclusion_rate"]),
            },
        })
    out = out_dir / f"filter_y_outlier_{method}_v1.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes, {len(cases)} cases)")


def main() -> None:
    out_dir = REPO_ROOT / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_gauss = synth_y_gaussian_with_outliers(SEED)
    y_skew = synth_y_skewed(SEED)
    y_unif = synth_y_uniform(SEED)
    y_bi = synth_y_bimodal(SEED)
    print(
        f"y_gauss n={y_gauss.size}, range=[{np.nanmin(y_gauss):.3f}, "
        f"{np.nanmax(y_gauss):.3f}], NaNs={np.isnan(y_gauss).sum()}"
    )
    print(
        f"y_skew  n={y_skew.size}, range=[{y_skew.min():.3f}, "
        f"{y_skew.max():.3f}]"
    )
    print(
        f"y_unif  n={y_unif.size}, range=[{y_unif.min():.3f}, "
        f"{y_unif.max():.3f}]"
    )
    print(
        f"y_bi    n={y_bi.size}, range=[{y_bi.min():.3f}, "
        f"{y_bi.max():.3f}]"
    )

    # Choose distinct distributions per method so we cover varied bound
    # geometries in every fixture without inflating the fixture count.
    write_fixture("iqr",        y_gauss, iqr_cases(y_gauss),        out_dir)
    write_fixture("zscore",     y_skew,  zscore_cases(y_skew),      out_dir)
    write_fixture("percentile", y_unif,  percentile_cases(y_unif),  out_dir)
    write_fixture("mad",        y_bi,    mad_cases(y_bi),           out_dir)


if __name__ == "__main__":
    main()
