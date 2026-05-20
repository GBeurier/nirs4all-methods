#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 3 parity fixtures for the stateful preprocessing operators.

The four Phase 3 operators implement the `_fit/_transform` ABI contract:

  * MSC      — Multiplicative Scatter Correction using the conventional
               row-wise reference-spectrum contract. Inverse is checked
               immediately after a forward transform on the same handle.
  * EMSC     — Extended MSC with polynomial wavelength terms (degree=1, 2, 3).
               Not invertible.
  * Baseline — Per-column mean centering (sklearn's StandardScaler
               with_std=False). Invertible.
  * Derivate — np.diff(X, n=order, axis=1) / delta**order. Output column
               count = input - order. `_fit` is a no-op (still required by
               the contract) but the matrix-view shape check enforces a
               consistent column count between fit and transform.

Each fixture exercises the fit/transform split: MSC and EMSC fit on a
training matrix and apply transform on a *different* test matrix to verify
the fitted parameters are correctly reused (the test set is generated from
the same synthesizer with seed+1 to keep the spectra physically realistic
but numerically distinct). Baseline and Derivate use a single matrix since
their fit→transform pair on the same input matches `fit_transform`.

Output: ``parity/fixtures/{msc,emsc,baseline_center,derivate}_v1.json``.

Fixture schema (additive over Phase 2 — adds an optional ``fit_input_hex``
key when fit and transform use different matrices):

```json
{
  "format": "c4a_pp_<op>_v1",
  "numpy_version": "1.26.4",
  "nirs4all_version": "0.8.x",
  "encoding": "ieee754_binary64_be_hex",
  "fit_rows": 50,
  "fit_cols": 200,
  "fit_input_hex": ["..."],     # training matrix (rows*cols big-endian hex)
  "rows": 50,
  "cols": 200,
  "input_hex": ["..."],         # transform input (may equal fit_input)
  "cases": [
    {"name": "...", "params": {...},
     "output_rows": 50, "output_cols": <variable>,
     "output_hex": ["..."]},
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


# ---------------------------------------------------------------------------
# Load nirs4all reference operators directly from source files.
# ---------------------------------------------------------------------------

NIRS4ALL_ROOT = Path("/home/delete/nirs4all/nirs4all/nirs4all")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


nirs_mod = _load(NIRS4ALL_ROOT / "operators/transforms/nirs.py",
                 "_n4a_nirs_phase3")
signal_mod = _load(NIRS4ALL_ROOT / "operators/transforms/signal.py",
                   "_n4a_signal_phase3")


# ---------------------------------------------------------------------------
# Fixture encoding helpers.
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    """Encode an IEEE-754 binary64 as a 16-char big-endian hex string."""
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    """Encode a numpy float64 array as a flat list of big-endian hex doubles."""
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# Synthetic NIR-like spectra (same generator family as Phase 2).
# ---------------------------------------------------------------------------

SEED = 20260518
ROWS = 50
COLS = 200


def synthesize_spectra(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, COLS, dtype=np.float64)
    X = np.empty((ROWS, COLS), dtype=np.float64)
    for i in range(ROWS):
        a = rng.uniform(0.2, 0.6)
        b = rng.uniform(-0.3, 0.3)
        c = rng.uniform(-0.4, 0.4)
        baseline = a + b * wl + c * wl * wl
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
# Reference implementations matching the C engines.
#
# MSC: conventional row-wise reference-spectrum correction, matching
#      prospectr::msc / pls::msc. The C engine stores the row coefficients from
#      the last transform so inverse_transform(transform(X)) remains covered.
# EMSC: nirs4all's ExtendedMultiplicativeScatterCorrection with scale=False.
#       Reference and integer wavelengths are stored at fit time; the
#       polynomial + reference basis is re-fit per row at transform time.
# Baseline: nirs4all's signal.Baseline. mean_ = mean(X_fit, axis=0).
# Derivate: brief specifies np.diff(X, n=order, axis=1) / delta**order. The
#          nirs4all class uses np.gradient(X, delta, axis=0), but Phase 3
#          locks in the np.diff axis=1 variant for the c4a_pp_derivate API
#          (output shape = (rows, cols - order)), reflecting the more common
#          spectroscopic convention. The nirs4all parity is therefore on the
#          algorithm specified in the c4a roadmap, not on
#          nirs4all.Derivate verbatim.
# ---------------------------------------------------------------------------

def ref_msc(fit_X: np.ndarray, transform_X: np.ndarray, **_: Any) -> np.ndarray:
    reference = np.mean(fit_X, axis=0)
    ref_mean = np.mean(reference)
    ref_centered = reference - ref_mean
    denom = float(np.sum(ref_centered * ref_centered))
    if denom <= 0.0:
        raise ValueError("MSC reference spectrum has zero variance")
    out = np.empty_like(transform_X, dtype=np.float64)
    for idx, row in enumerate(transform_X):
        row_mean = np.mean(row)
        slope = float(np.sum((row - row_mean) * ref_centered) / denom)
        intercept = row_mean - slope * ref_mean
        out[idx] = (row - intercept) / slope
    return out


def ref_msc_inverse(fit_X: np.ndarray, transform_X: np.ndarray,
                    **_: Any) -> np.ndarray:
    _ = fit_X
    return np.array(transform_X, copy=True)


def ref_emsc(fit_X: np.ndarray, transform_X: np.ndarray, *,
             degree: int) -> np.ndarray:
    op = nirs_mod.ExtendedMultiplicativeScatterCorrection(
        degree=degree, scale=False)
    op.fit(fit_X.copy())
    return op.transform(transform_X.copy())


def ref_baseline(fit_X: np.ndarray, transform_X: np.ndarray,
                 **_: Any) -> np.ndarray:
    op = signal_mod.Baseline()
    op.fit(fit_X.copy())
    return op.transform(transform_X.copy())


def ref_baseline_inverse(fit_X: np.ndarray, transform_X: np.ndarray,
                         **_: Any) -> np.ndarray:
    op = signal_mod.Baseline()
    op.fit(fit_X.copy())
    return op.inverse_transform(transform_X.copy())


def ref_derivate(fit_X: np.ndarray, transform_X: np.ndarray, *,
                 order: int, delta: float) -> np.ndarray:
    # `fit_X` is required by the contract to validate column counts; we
    # don't actually need it for the computation since the algorithm is
    # stateless mathematically.
    _ = fit_X
    return np.diff(transform_X, n=order, axis=1) / (delta ** order)


# ---------------------------------------------------------------------------
# Per-operator fixture builders.
# ---------------------------------------------------------------------------

def msc_cases(fit_X: np.ndarray, test_X: np.ndarray):
    return [
        ("fit_train_transform_test", {},
         lambda: ref_msc(fit_X, test_X)),
        ("fit_train_inverse_test", {"variant": "inverse"},
         lambda: ref_msc_inverse(fit_X, test_X)),
    ]


def emsc_cases(fit_X: np.ndarray, test_X: np.ndarray):
    return [
        ("degree1", {"degree": 1},
         lambda: ref_emsc(fit_X, test_X, degree=1)),
        ("degree2", {"degree": 2},
         lambda: ref_emsc(fit_X, test_X, degree=2)),
        ("degree3", {"degree": 3},
         lambda: ref_emsc(fit_X, test_X, degree=3)),
    ]


def baseline_cases(fit_X: np.ndarray, test_X: np.ndarray):
    return [
        ("fit_transform_same", {},
         lambda: ref_baseline(fit_X, fit_X)),
        ("fit_train_transform_test", {"variant": "test"},
         lambda: ref_baseline(fit_X, test_X)),
        ("fit_inverse_same", {"variant": "inverse"},
         lambda: ref_baseline_inverse(fit_X, ref_baseline(fit_X, fit_X))),
    ]


def derivate_cases(fit_X: np.ndarray, test_X: np.ndarray):
    # Always use the same matrix for both fit and transform (fit is a no-op
    # for Derivate but we still record fit_input for the test harness).
    return [
        ("order1_delta1", {"order": 1, "delta": 1.0},
         lambda: ref_derivate(test_X, test_X, order=1, delta=1.0)),
        ("order2_delta1", {"order": 2, "delta": 1.0},
         lambda: ref_derivate(test_X, test_X, order=2, delta=1.0)),
        ("order1_delta2", {"order": 1, "delta": 2.0},
         lambda: ref_derivate(test_X, test_X, order=1, delta=2.0)),
        ("order3_delta0p5", {"order": 3, "delta": 0.5},
         lambda: ref_derivate(test_X, test_X, order=3, delta=0.5)),
    ]


# ---------------------------------------------------------------------------
# Fixture writer.
# ---------------------------------------------------------------------------

def write_fixture(name: str, op_id: str,
                  fit_X: np.ndarray, transform_X: np.ndarray,
                  cases: list[tuple[str, dict, Callable[[], np.ndarray]]],
                  out_dir: Path, nirs4all_version: str) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_pp_{op_id}_v1",
        "numpy_version": np.__version__,
        "nirs4all_version": nirs4all_version,
        "encoding": "ieee754_binary64_be_hex",
        "fit_rows": int(fit_X.shape[0]),
        "fit_cols": int(fit_X.shape[1]),
        "fit_input_hex": array_to_hex(fit_X),
        "rows": int(transform_X.shape[0]),
        "cols": int(transform_X.shape[1]),
        "input_hex": array_to_hex(transform_X),
        "cases": [],
    }
    for case_name, params, fn in cases:
        Y = fn()
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        fixture["cases"].append({
            "name": case_name,
            "params": params,
            "output_rows": int(Y.shape[0]),
            "output_cols": int(Y.shape[1]),
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

    n4a_init = NIRS4ALL_ROOT / "__init__.py"
    version = "unknown"
    for line in n4a_init.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("__version__"):
            version = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

    # The training matrix (fit_X) and the test matrix (transform_X) share
    # the same shape but are generated from neighbouring seeds. MSC and
    # EMSC apply their fitted (a, b) / reference parameters to a *different*
    # matrix at transform time to exercise the fit/transform split.
    fit_X       = synthesize_spectra(SEED)
    transform_X = synthesize_spectra(SEED + 1)
    print(f"fit_X shape {fit_X.shape}, range [{fit_X.min():.4f}, {fit_X.max():.4f}]")
    print(f"transform_X shape {transform_X.shape}, range "
          f"[{transform_X.min():.4f}, {transform_X.max():.4f}]")

    write_fixture("msc",             "msc",      fit_X, transform_X,
                  msc_cases(fit_X, transform_X),       out_dir, version)
    write_fixture("emsc",            "emsc",     fit_X, transform_X,
                  emsc_cases(fit_X, transform_X),      out_dir, version)
    write_fixture("baseline_center", "baseline", fit_X, transform_X,
                  baseline_cases(fit_X, transform_X),  out_dir, version)
    write_fixture("derivate",        "derivate", fit_X, transform_X,
                  derivate_cases(fit_X, transform_X),  out_dir, version)


if __name__ == "__main__":
    main()
