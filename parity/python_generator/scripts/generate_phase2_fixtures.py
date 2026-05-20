#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 2 parity fixtures for the stateless preprocessing operators.

For each of the seven operators (SNV, LocalSNV, RobustSNV, AreaNormalization,
Normalize, SimpleScale, LogTransform) we pick a small grid of representative
parameter combinations and apply nirs4all's reference implementation to a
deterministic 50 x 200 NIR-shaped synthetic spectrum block. The result is
captured as a JSON fixture that the C++ test harness loads to verify
byte-for-byte equality (1e-12 abs / 1e-13 rel) against the C engine.

The synthetic spectra are generated from a PCG64 stream (seed=20260518) — the
same seed convention as the Phase 1 RNG fixture, so the input arrays are
reproducible from the c4a side too.

Output: ``parity/fixtures/{snv,lsnv,rnv,area_norm,normalize,simple_scale,log_transform}_v1.json``.

Each fixture has the schema:

```json
{
  "format": "c4a_pp_<op>_v1",
  "numpy_version": "1.26.4",
  "nirs4all_version": "0.8.x",
  "encoding": "ieee754_binary64_be_hex",
  "rows": 50,
  "cols": 200,
  "input_hex": ["..."],     # rows * cols big-endian hex doubles, row-major
  "cases": [
    {"name": "...", "params": {...}, "output_hex": ["..."]},
    ...
  ]
}
```
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from struct import pack
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from parity.nirs4all_source import find_nirs4all_root, get_nirs4all_version  # noqa: E402


# ---------------------------------------------------------------------------
# Load nirs4all reference operators directly from source files.
#
# nirs4all's __init__ pulls in heavy optional dependencies; bypass it by
# loading the operator modules in isolation via importlib so the fixture
# generator only depends on numpy + scipy + sklearn + pywavelets.
# ---------------------------------------------------------------------------

NIRS4ALL_ROOT = find_nirs4all_root()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


scalers_mod = _load(NIRS4ALL_ROOT / "operators/transforms/scalers.py",
                    "_n4a_scalers_lite")
nirs_mod = _load(NIRS4ALL_ROOT / "operators/transforms/nirs.py",
                 "_n4a_nirs_lite")


# ---------------------------------------------------------------------------
# Fixture encoding helpers (shared with the Phase 1 RNG fixture).
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    """Encode an IEEE-754 binary64 as a 16-char big-endian hex string."""
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    """Encode a numpy float64 array as a flat list of big-endian hex doubles."""
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# Synthetic NIR-like data generator.
#
# 50 rows × 200 wavelengths of smoothly-varying spectra with:
#   - global offset + low-frequency baseline drift
#   - a couple of Gaussian "absorption" peaks at random centres
#   - additive Gaussian noise
# All values are in the physically-meaningful [-0.5, 2.0] range so the
# transforms exercise both positive and (in the SNV / RSNV centered case)
# negative regions of the spectrum.
# ---------------------------------------------------------------------------

SEED = 20260518
ROWS = 50
COLS = 200


def synthesize_spectra() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    wl = np.linspace(0.0, 1.0, COLS, dtype=np.float64)  # normalised wavelength
    X = np.empty((ROWS, COLS), dtype=np.float64)
    for i in range(ROWS):
        # baseline = a + b*wl + c*wl^2
        a = rng.uniform(0.2, 0.6)
        b = rng.uniform(-0.3, 0.3)
        c = rng.uniform(-0.4, 0.4)
        baseline = a + b * wl + c * wl * wl
        # two Gaussian peaks
        peak1_mu = rng.uniform(0.2, 0.4)
        peak1_sigma = rng.uniform(0.02, 0.06)
        peak1_amp = rng.uniform(0.3, 1.2)
        peak2_mu = rng.uniform(0.6, 0.85)
        peak2_sigma = rng.uniform(0.03, 0.08)
        peak2_amp = rng.uniform(0.2, 0.9)
        peaks = (
            peak1_amp * np.exp(-((wl - peak1_mu) ** 2) / (2.0 * peak1_sigma ** 2))
            + peak2_amp * np.exp(-((wl - peak2_mu) ** 2) / (2.0 * peak2_sigma ** 2))
        )
        noise = rng.normal(0.0, 0.01, size=COLS)
        X[i] = baseline + peaks + noise
    return X


# ---------------------------------------------------------------------------
# Per-operator fixture builders.
# Each builder returns a list of (case_name, params_dict, callable) where
# `callable(X)` produces the reference output array.
# ---------------------------------------------------------------------------

def snv_cases() -> list[tuple[str, dict, Callable[[np.ndarray], np.ndarray]]]:
    return [
        ("default", {"with_mean": True, "with_std": True, "ddof": 0},
         lambda X: scalers_mod.StandardNormalVariate(
             with_mean=True, with_std=True, ddof=0).fit_transform(X.copy())),
        ("ddof1", {"with_mean": True, "with_std": True, "ddof": 1},
         lambda X: scalers_mod.StandardNormalVariate(
             with_mean=True, with_std=True, ddof=1).fit_transform(X.copy())),
        ("center_only", {"with_mean": True, "with_std": False, "ddof": 0},
         lambda X: scalers_mod.StandardNormalVariate(
             with_mean=True, with_std=False).fit_transform(X.copy())),
        ("scale_only", {"with_mean": False, "with_std": True, "ddof": 0},
         lambda X: scalers_mod.StandardNormalVariate(
             with_mean=False, with_std=True, ddof=0).fit_transform(X.copy())),
    ]


def lsnv_cases() -> list[tuple[str, dict, Callable[[np.ndarray], np.ndarray]]]:
    return [
        ("w11_reflect", {"window": 11, "pad_mode": "reflect", "constant_value": 0.0},
         lambda X: scalers_mod.LocalStandardNormalVariate(
             window=11, pad_mode="reflect").fit_transform(X.copy())),
        ("w21_edge", {"window": 21, "pad_mode": "edge", "constant_value": 0.0},
         lambda X: scalers_mod.LocalStandardNormalVariate(
             window=21, pad_mode="edge").fit_transform(X.copy())),
        ("w7_constant_0", {"window": 7, "pad_mode": "constant", "constant_value": 0.0},
         lambda X: scalers_mod.LocalStandardNormalVariate(
             window=7, pad_mode="constant", constant_values=0.0).fit_transform(X.copy())),
        ("w51_reflect", {"window": 51, "pad_mode": "reflect", "constant_value": 0.0},
         lambda X: scalers_mod.LocalStandardNormalVariate(
             window=51, pad_mode="reflect").fit_transform(X.copy())),
    ]


def rnv_cases() -> list[tuple[str, dict, Callable[[np.ndarray], np.ndarray]]]:
    return [
        ("default", {"with_center": True, "with_scale": True, "k": 1.4826},
         lambda X: scalers_mod.RobustStandardNormalVariate(
             with_center=True, with_scale=True, k=1.4826).fit_transform(X.copy())),
        ("center_only", {"with_center": True, "with_scale": False, "k": 1.4826},
         lambda X: scalers_mod.RobustStandardNormalVariate(
             with_center=True, with_scale=False, k=1.4826).fit_transform(X.copy())),
        ("scale_only", {"with_center": False, "with_scale": True, "k": 1.4826},
         lambda X: scalers_mod.RobustStandardNormalVariate(
             with_center=False, with_scale=True, k=1.4826).fit_transform(X.copy())),
        ("k1.0", {"with_center": True, "with_scale": True, "k": 1.0},
         lambda X: scalers_mod.RobustStandardNormalVariate(
             with_center=True, with_scale=True, k=1.0).fit_transform(X.copy())),
    ]


def area_cases() -> list[tuple[str, dict, Callable[[np.ndarray], np.ndarray]]]:
    return [
        ("sum", {"method": "sum"},
         lambda X: nirs_mod.AreaNormalization(method="sum").fit_transform(X.copy())),
        ("abs_sum", {"method": "abs_sum"},
         lambda X: nirs_mod.AreaNormalization(method="abs_sum").fit_transform(X.copy())),
        ("trapz", {"method": "trapz"},
         lambda X: nirs_mod.AreaNormalization(method="trapz").fit_transform(X.copy())),
    ]


def normalize_cases() -> list[tuple[str, dict, Callable[[np.ndarray], np.ndarray]]]:
    return [
        ("linalg_default", {"feature_min": -1.0, "feature_max": 1.0},
         lambda X: scalers_mod.Normalize(feature_range=(-1, 1)).fit_transform(X.copy())),
        ("range_0_1", {"feature_min": 0.0, "feature_max": 1.0},
         lambda X: scalers_mod.Normalize(feature_range=(0.0, 1.0)).fit_transform(X.copy())),
        ("range_neg5_pos5", {"feature_min": -5.0, "feature_max": 5.0},
         lambda X: scalers_mod.Normalize(feature_range=(-5.0, 5.0)).fit_transform(X.copy())),
    ]


def simple_scale_cases() -> list[tuple[str, dict, Callable[[np.ndarray], np.ndarray]]]:
    return [
        ("default", {},
         lambda X: scalers_mod.SimpleScale().fit_transform(X.copy())),
    ]


def log_transform_cases() -> list[tuple[str, dict, Callable[[np.ndarray], np.ndarray]]]:
    return [
        ("natural_auto", {"base": 0.0, "offset": 0.0,
                          "auto_offset": True, "min_value": 1e-8},
         lambda X: nirs_mod.LogTransform(base=np.e, offset=0.0,
                                          auto_offset=True,
                                          min_value=1e-8).fit_transform(X.copy())),
        ("natural_noauto_offset1",
         {"base": 0.0, "offset": 1.0, "auto_offset": False, "min_value": 1e-8},
         lambda X: nirs_mod.LogTransform(base=np.e, offset=1.0,
                                          auto_offset=False,
                                          min_value=1e-8).fit_transform(X.copy())),
        ("log10_auto",
         {"base": 10.0, "offset": 0.0, "auto_offset": True, "min_value": 1e-8},
         lambda X: nirs_mod.LogTransform(base=10.0, offset=0.0,
                                          auto_offset=True,
                                          min_value=1e-8).fit_transform(X.copy())),
        ("log2_offset0.5",
         {"base": 2.0, "offset": 0.5, "auto_offset": True, "min_value": 1e-6},
         lambda X: nirs_mod.LogTransform(base=2.0, offset=0.5,
                                          auto_offset=True,
                                          min_value=1e-6).fit_transform(X.copy())),
    ]


# ---------------------------------------------------------------------------
# Fixture writer.
# ---------------------------------------------------------------------------

def write_fixture(name: str, op_id: str,
                  cases: list[tuple[str, dict, Callable[[np.ndarray], np.ndarray]]],
                  X: np.ndarray, out_dir: Path,
                  nirs4all_version: str) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_pp_{op_id}_v1",
        "numpy_version": np.__version__,
        "nirs4all_version": nirs4all_version,
        "encoding": "ieee754_binary64_be_hex",
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "input_hex": array_to_hex(X),
        "cases": [],
    }
    for case_name, params, fn in cases:
        Y = fn(X)
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        if Y.shape != X.shape:
            raise RuntimeError(
                f"{op_id}/{case_name}: output shape {Y.shape} != input {X.shape}")
        fixture["cases"].append({
            "name": case_name,
            "params": params,
            "output_hex": array_to_hex(Y),
        })
    out = out_dir / f"{name}_v1.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes, {len(cases)} cases)")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    version = get_nirs4all_version(NIRS4ALL_ROOT)

    X = synthesize_spectra()
    print(f"synthesised X with shape {X.shape}, dtype {X.dtype}, "
          f"range [{X.min():.4f}, {X.max():.4f}]")

    write_fixture("snv",           "snv",          snv_cases(),           X, out_dir, version)
    write_fixture("lsnv",          "lsnv",         lsnv_cases(),          X, out_dir, version)
    write_fixture("rnv",           "rnv",          rnv_cases(),           X, out_dir, version)
    write_fixture("area_norm",     "area",         area_cases(),          X, out_dir, version)
    write_fixture("normalize",     "normalize",    normalize_cases(),     X, out_dir, version)
    write_fixture("simple_scale",  "simple_scale", simple_scale_cases(),  X, out_dir, version)
    write_fixture("log_transform", "log",          log_transform_cases(), X, out_dir, version)


if __name__ == "__main__":
    main()
