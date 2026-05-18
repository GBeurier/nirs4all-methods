#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 10 parity fixtures for the resampling / cropping /
discretization operators.

The five Phase 10 operators are:

  * Resampler              — wavelength-grid interpolation between a fitted
                             source axis and a configured target axis with
                             linear / nearest / cubic methods.
  * CropTransformer        — slice columns ``[start, end)`` from X.
  * ResampleTransformer    — linear resample of every row to a fixed
                             ``num_samples`` target column count on the
                             [0, 1] normalised axis.
  * IntegerKBinsDiscretizer — sklearn KBinsDiscretizer with ``encode='ordinal'``
                             and integer output. Uniform / quantile strategy.
  * RangeDiscretizer       — ``np.digitize(X, bins, right=False)`` over a
                             fixed edge vector.

The fixtures are written to ``parity/fixtures/{resampler, crop, resample,
kbins_disc, range_disc}_v1.json``. Each follows the shared schema documented
in ``cpp/tests/fixture_parser.hpp``.

Note: integer-output operators encode `output_hex` as the float64 cast of
their int32 outputs (so the hex array uses the same encoding as the rest of
the corpus). The C side decodes them via `static_cast<std::int32_t>(want[i])`
in the integer comparator.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from struct import pack
from typing import Any, Callable

import numpy as np
from scipy.interpolate import interp1d


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


resampler_mod = _load(NIRS4ALL_ROOT / "operators/transforms/resampler.py",
                      "_n4a_resampler_phase10")
features_mod  = _load(NIRS4ALL_ROOT / "operators/transforms/features.py",
                      "_n4a_features_phase10")
targets_mod   = _load(NIRS4ALL_ROOT / "operators/transforms/targets.py",
                      "_n4a_targets_phase10")


# ---------------------------------------------------------------------------
# Fixture encoding helpers.
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


def int_array_to_hex(arr: np.ndarray) -> list[str]:
    """Encode an integer array as the float64 cast of its values. The C side
    reads the value back as a double and casts to int32 for comparison; this
    works because every int32 value (|v| < 2**31) is exactly representable
    as a double. """
    float_view = np.asarray(arr, dtype=np.float64)
    return array_to_hex(float_view)


# ---------------------------------------------------------------------------
# Synthetic spectra (same generator family as earlier phases).
# ---------------------------------------------------------------------------

SEED = 20260518
ROWS = 30
COLS = 60


def synthesize_spectra(seed: int, rows: int = ROWS, cols: int = COLS):
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, cols, dtype=np.float64)
    X = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
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
        noise = rng.normal(0.0, 0.01, size=cols)
        X[i] = baseline + peaks + noise
    return X


# ---------------------------------------------------------------------------
# Reference implementations matching the C engines.
# ---------------------------------------------------------------------------

def ref_resampler(fit_X: np.ndarray, transform_X: np.ndarray, *,
                  source_wl: np.ndarray, target_wl: np.ndarray | None,
                  method: str, crop_range: tuple[float, float] | None,
                  fill_value: float | str, bounds_error: bool) -> np.ndarray:
    """Replicate nirs4all's Resampler.{fit, transform} via the same scipy
    backbone. nirs4all's fit accepts the wavelengths as a kwarg; we pass them
    explicitly to keep the test reproducible."""
    op = resampler_mod.Resampler(
        target_wavelengths=target_wl,
        method=method,
        crop_range=crop_range,
        fill_value=fill_value,
        bounds_error=bounds_error,
    )
    op.fit(fit_X, wavelengths=source_wl)
    return op.transform(transform_X)


def ref_crop(X: np.ndarray, *, start: int, end: int | None) -> np.ndarray:
    op = features_mod.CropTransformer(start=start, end=end)
    return op.fit_transform(X)


def ref_resample(X: np.ndarray, *, num_samples: int | None) -> np.ndarray:
    op = features_mod.ResampleTransformer(num_samples=num_samples)
    return op.fit_transform(X)


def ref_kbins_disc(fit_X: np.ndarray, transform_X: np.ndarray, *,
                   n_bins: int, strategy: str) -> np.ndarray:
    op = targets_mod.IntegerKBinsDiscretizer(
        n_bins=n_bins, encode='ordinal', strategy=strategy)
    op.fit(fit_X)
    return op.transform(transform_X)


def ref_range_disc(X: np.ndarray, *, edges: list[float]) -> np.ndarray:
    op = targets_mod.RangeDiscretizer(bins=edges)
    return op.fit_transform(X)


# ---------------------------------------------------------------------------
# Per-operator fixture builders.
# ---------------------------------------------------------------------------

def resampler_cases(fit_X: np.ndarray, test_X: np.ndarray):
    cols = fit_X.shape[1]
    source_wl = np.linspace(1000.0, 2500.0, cols)
    target_wl_dense = np.linspace(1000.0, 2500.0, cols // 2)
    target_wl_sparse = np.linspace(1100.0, 2400.0, cols // 3)
    target_wl_extrap = np.linspace(950.0, 2550.0, cols // 2)
    cases = []
    # Linear, dense target inside source range
    cases.append(("linear_inside", {
        "method": 0, "n_target": len(target_wl_dense),
        "src_min": float(source_wl[0]),
        "src_step": float(source_wl[1] - source_wl[0]),
        "src_n": int(cols),
        "tgt_min": float(target_wl_dense[0]),
        "tgt_step": float(target_wl_dense[1] - target_wl_dense[0]),
        "tgt_n": int(len(target_wl_dense)),
        "use_crop": 0, "crop_min": 0.0, "crop_max": 0.0,
        "fill_value": 0.0, "bounds_error": 0, "extrapolate": 0,
    }, lambda: ref_resampler(fit_X, test_X, source_wl=source_wl,
                              target_wl=target_wl_dense, method='linear',
                              crop_range=None, fill_value=0.0,
                              bounds_error=False)))
    # Nearest interp on a sparse target
    cases.append(("nearest_sparse", {
        "method": 1, "n_target": len(target_wl_sparse),
        "src_min": float(source_wl[0]),
        "src_step": float(source_wl[1] - source_wl[0]),
        "src_n": int(cols),
        "tgt_min": float(target_wl_sparse[0]),
        "tgt_step": float(target_wl_sparse[1] - target_wl_sparse[0]),
        "tgt_n": int(len(target_wl_sparse)),
        "use_crop": 0, "crop_min": 0.0, "crop_max": 0.0,
        "fill_value": 0.0, "bounds_error": 0, "extrapolate": 0,
    }, lambda: ref_resampler(fit_X, test_X, source_wl=source_wl,
                              target_wl=target_wl_sparse, method='nearest',
                              crop_range=None, fill_value=0.0,
                              bounds_error=False)))
    # Cubic on a dense target
    cases.append(("cubic_inside", {
        "method": 2, "n_target": len(target_wl_dense),
        "src_min": float(source_wl[0]),
        "src_step": float(source_wl[1] - source_wl[0]),
        "src_n": int(cols),
        "tgt_min": float(target_wl_dense[0]),
        "tgt_step": float(target_wl_dense[1] - target_wl_dense[0]),
        "tgt_n": int(len(target_wl_dense)),
        "use_crop": 0, "crop_min": 0.0, "crop_max": 0.0,
        "fill_value": 0.0, "bounds_error": 0, "extrapolate": 0,
    }, lambda: ref_resampler(fit_X, test_X, source_wl=source_wl,
                              target_wl=target_wl_dense, method='cubic',
                              crop_range=None, fill_value=0.0,
                              bounds_error=False)))
    # Linear with out-of-range target (extrapolated via fill_value)
    cases.append(("linear_extrap_fill", {
        "method": 0, "n_target": len(target_wl_extrap),
        "src_min": float(source_wl[0]),
        "src_step": float(source_wl[1] - source_wl[0]),
        "src_n": int(cols),
        "tgt_min": float(target_wl_extrap[0]),
        "tgt_step": float(target_wl_extrap[1] - target_wl_extrap[0]),
        "tgt_n": int(len(target_wl_extrap)),
        "use_crop": 0, "crop_min": 0.0, "crop_max": 0.0,
        "fill_value": 0.0, "bounds_error": 0, "extrapolate": 0,
    }, lambda: ref_resampler(fit_X, test_X, source_wl=source_wl,
                              target_wl=target_wl_extrap, method='linear',
                              crop_range=None, fill_value=0.0,
                              bounds_error=False)))
    return cases


def crop_cases(X: np.ndarray):
    cols = X.shape[1]
    cases = []
    cases.append(("middle", {"start": 10, "end": 40},
                  lambda: ref_crop(X, start=10, end=40)))
    cases.append(("end_open", {"start": cols // 2, "end": -1},
                  lambda: ref_crop(X, start=cols // 2, end=None)))
    cases.append(("first_quarter", {"start": 0, "end": cols // 4},
                  lambda: ref_crop(X, start=0, end=cols // 4)))
    return cases


def resample_cases(X: np.ndarray):
    cols = X.shape[1]
    cases = []
    cases.append(("downsample_half", {"num_samples": cols // 2},
                  lambda: ref_resample(X, num_samples=cols // 2)))
    cases.append(("upsample_double", {"num_samples": cols * 2},
                  lambda: ref_resample(X, num_samples=cols * 2)))
    cases.append(("downsample_third", {"num_samples": cols // 3},
                  lambda: ref_resample(X, num_samples=cols // 3)))
    return cases


def kbins_disc_cases(fit_X: np.ndarray, transform_X: np.ndarray):
    # IntegerKBinsDiscretizer operates on a single column (target values),
    # so we feed it a single-column slice.
    fit_y = fit_X[:, :1]
    test_y = transform_X[:, :1]
    cases = []
    cases.append(("uniform_3", {"n_bins": 3, "strategy": 0},
                  lambda: ref_kbins_disc(fit_y, test_y, n_bins=3,
                                          strategy='uniform')))
    cases.append(("uniform_5", {"n_bins": 5, "strategy": 0},
                  lambda: ref_kbins_disc(fit_y, test_y, n_bins=5,
                                          strategy='uniform')))
    cases.append(("quantile_4", {"n_bins": 4, "strategy": 1},
                  lambda: ref_kbins_disc(fit_y, test_y, n_bins=4,
                                          strategy='quantile')))
    return (fit_y, test_y, cases)


def range_disc_cases(X: np.ndarray):
    y = X[:, :1]
    cases = []
    edges_a = [0.4, 0.6, 0.8]
    cases.append(("three_edges", {"edges_csv": ",".join(str(e) for e in edges_a)},
                  lambda: ref_range_disc(y, edges=edges_a)))
    edges_b = [0.3, 0.5, 0.7, 0.9, 1.1]
    cases.append(("five_edges", {"edges_csv": ",".join(str(e) for e in edges_b)},
                  lambda: ref_range_disc(y, edges=edges_b)))
    return (y, cases)


# ---------------------------------------------------------------------------
# Fixture writers.
# ---------------------------------------------------------------------------

def write_fixture_stateless_float(name: str, op_id: str,
                                   transform_X: np.ndarray,
                                   cases: list[tuple[str, dict, Callable]],
                                   out_dir: Path, nirs4all_version: str) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_pp_{op_id}_v1",
        "numpy_version": np.__version__,
        "nirs4all_version": nirs4all_version,
        "encoding": "ieee754_binary64_be_hex",
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


def write_fixture_stateless_int(name: str, op_id: str,
                                 transform_X: np.ndarray,
                                 cases: list[tuple[str, dict, Callable]],
                                 out_dir: Path, nirs4all_version: str) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_pp_{op_id}_v1",
        "numpy_version": np.__version__,
        "nirs4all_version": nirs4all_version,
        "encoding": "ieee754_binary64_be_hex",
        "rows": int(transform_X.shape[0]),
        "cols": int(transform_X.shape[1]),
        "input_hex": array_to_hex(transform_X),
        "cases": [],
    }
    for case_name, params, fn in cases:
        Y = fn()
        # Y is int32 (rows, 1); store as float-cast hex.
        Y = np.ascontiguousarray(Y, dtype=np.int32)
        fixture["cases"].append({
            "name": case_name,
            "params": params,
            "output_rows": int(Y.shape[0]),
            "output_cols": int(Y.shape[1]),
            "output_hex": int_array_to_hex(Y),
        })
    out = out_dir / f"{name}_v1.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes, {len(cases)} cases)")


def write_fixture_stateful_float(name: str, op_id: str,
                                  fit_X: np.ndarray, transform_X: np.ndarray,
                                  cases: list[tuple[str, dict, Callable]],
                                  out_dir: Path,
                                  nirs4all_version: str) -> None:
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


def write_fixture_stateful_int(name: str, op_id: str,
                                fit_X: np.ndarray, transform_X: np.ndarray,
                                cases: list[tuple[str, dict, Callable]],
                                out_dir: Path,
                                nirs4all_version: str) -> None:
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
        Y = np.ascontiguousarray(Y, dtype=np.int32)
        fixture["cases"].append({
            "name": case_name,
            "params": params,
            "output_rows": int(Y.shape[0]),
            "output_cols": int(Y.shape[1]),
            "output_hex": int_array_to_hex(Y),
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

    fit_X       = synthesize_spectra(SEED)
    transform_X = synthesize_spectra(SEED + 1)
    print(f"fit_X shape {fit_X.shape}, range [{fit_X.min():.4f}, {fit_X.max():.4f}]")
    print(f"transform_X shape {transform_X.shape}, range "
          f"[{transform_X.min():.4f}, {transform_X.max():.4f}]")

    # Resampler — stateful.
    write_fixture_stateful_float(
        "resampler", "resampler",
        fit_X, transform_X,
        resampler_cases(fit_X, transform_X),
        out_dir, version)
    # CropTransformer — stateless float.
    write_fixture_stateless_float(
        "crop", "crop",
        transform_X,
        crop_cases(transform_X),
        out_dir, version)
    # ResampleTransformer — stateless float.
    write_fixture_stateless_float(
        "resample", "resample",
        transform_X,
        resample_cases(transform_X),
        out_dir, version)
    # IntegerKBinsDiscretizer — stateful int.
    fit_y, test_y, kb_cases = kbins_disc_cases(fit_X, transform_X)
    write_fixture_stateful_int(
        "kbins_disc", "kbins_disc",
        fit_y, test_y,
        kb_cases, out_dir, version)
    # RangeDiscretizer — stateless int.
    test_y_for_range, rd_cases = range_disc_cases(transform_X)
    write_fixture_stateless_int(
        "range_disc", "range_disc",
        test_y_for_range,
        rd_cases, out_dir, version)


if __name__ == "__main__":
    main()
