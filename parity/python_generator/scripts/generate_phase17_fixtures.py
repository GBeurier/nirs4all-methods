#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Phase 17 parity fixture generator for the 10 mixup / physical /
environmental augmenters.

Each augmenter exposes a stochastic `_apply` step. The fixture pins a
single deterministic case per augmenter — wherever the reference output is
fully determined by the input (no rng draws), we capture the float64
result. For augmenters that rely on PCG64 draws we still record metadata
and parameters but the C harness exercises smoke + lifecycle only.

The deterministic cases captured by this generator are:

  * scatter_sim: a_low=a_high=0.5, b_low=b_high=1.0 -> out = X + 0.5
  * batch_effect: variation_scope=1 with all stds=0 -> out = X * 1 + 0 = X
  * instrument_broaden: fixed FWHM (no RNG); reference output is the
    scipy.ndimage.gaussian_filter1d(X, sigma_pts) with sigma_pts =
    fwhm / (2 sqrt(2 log 2)) / wl_step.
  * temperature: zero temperature_delta -> out = X (early-skip path).
  * dead_band: probability=0 -> out = X.

Output: parity/fixtures/aug_phase17_v1.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from struct import pack
from typing import Any, Callable

import numpy as np
from scipy.ndimage import gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from parity.nirs4all_source import find_nirs4all_root, get_nirs4all_version  # noqa: E402

NIRS4ALL_ROOT = find_nirs4all_root()


def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


SEED = 20260518
ROWS = 8
COLS = 32


def synthesize_spectra() -> tuple[np.ndarray, np.ndarray]:
    """Synth small (rows, cols) block + a wavelength axis spanning 1300-1900 nm."""
    rng = np.random.default_rng(SEED)
    wavelengths = np.linspace(1300.0, 1900.0, COLS, dtype=np.float64)
    X = np.empty((ROWS, COLS), dtype=np.float64)
    for i in range(ROWS):
        a = rng.uniform(0.3, 0.7)
        b = rng.uniform(-0.1, 0.1)
        baseline = a + b * np.linspace(0.0, 1.0, COLS)
        peak_mu = rng.uniform(1400.0, 1800.0)
        peak_amp = rng.uniform(0.2, 0.6)
        peak_sigma = rng.uniform(20.0, 50.0)
        peak = peak_amp * np.exp(-0.5 * ((wavelengths - peak_mu) / peak_sigma) ** 2)
        noise = rng.normal(0.0, 0.005, COLS)
        X[i] = baseline + peak + noise
    return X, wavelengths


# Deterministic cases — no rng draws affect the output value.

def scatter_sim_constant(X: np.ndarray) -> np.ndarray:
    """a_low=a_high=0.5, b_low=b_high=1.0 -> out = X + 0.5."""
    return X + 0.5


def batch_effect_zero(X: np.ndarray) -> np.ndarray:
    """All stds zero, batch-mode -> gain=1, offset=0 -> X."""
    return X.copy()


def instrument_broaden_fixed(X: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Fixed FWHM=5, scipy.ndimage.gaussian_filter1d row-wise."""
    wl_step = np.median(np.diff(wavelengths))
    k = 2.0 * np.sqrt(2.0 * np.log(2.0))
    sigma_pts = 5.0 / k / wl_step
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        out[i] = gaussian_filter1d(X[i], sigma_pts)
    return out


def temperature_zero(X: np.ndarray) -> np.ndarray:
    """delta_t < 0.01 -> early-return X."""
    return X.copy()


def dead_band_zero_prob(X: np.ndarray) -> np.ndarray:
    """probability=0 -> always copy X unchanged."""
    return X.copy()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    version = get_nirs4all_version(NIRS4ALL_ROOT)

    X, wavelengths = synthesize_spectra()
    print(f"Synthesized X shape={X.shape}, wavelengths[{wavelengths[0]:.0f}, {wavelengths[-1]:.0f}]")

    cases: list[dict[str, Any]] = [
        {
            "name": "scatter_sim_constant",
            "augmenter": "ScatterSimulationMSC",
            "params": {"a_low": 0.5, "a_high": 0.5, "b_low": 1.0, "b_high": 1.0},
            "deterministic": True,
            "output_hex": array_to_hex(scatter_sim_constant(X)),
        },
        {
            "name": "batch_effect_zero_batch",
            "augmenter": "BatchEffectAugmenter",
            "params": {"offset_std": 0.0, "slope_std": 0.0, "gain_std": 0.0,
                       "variation_scope": 1},
            "deterministic": True,
            "output_hex": array_to_hex(batch_effect_zero(X)),
        },
        {
            "name": "instrument_broaden_fixed_fwhm5",
            "augmenter": "InstrumentalBroadeningAugmenter",
            "params": {"fwhm": 5.0, "use_fwhm_range": 0,
                       "variation_scope": 0},
            "deterministic": True,
            "output_hex": array_to_hex(instrument_broaden_fixed(X, wavelengths)),
        },
        {
            "name": "temperature_zero_delta",
            "augmenter": "TemperatureAugmenter",
            "params": {"temperature_delta": 0.0,
                       "enable_shift": 1, "enable_intensity": 1,
                       "enable_broadening": 1, "region_specific": 1},
            "deterministic": True,
            "output_hex": array_to_hex(temperature_zero(X)),
        },
        {
            "name": "dead_band_zero_probability",
            "augmenter": "DeadBandAugmenter",
            "params": {"n_bands": 1, "width_low": 5, "width_high": 10,
                       "noise_std": 0.05, "probability": 0.0,
                       "variation_scope": 0},
            "deterministic": True,
            "output_hex": array_to_hex(dead_band_zero_prob(X)),
        },
    ]

    # Stochastic operators — record metadata only (no output values).
    # The C tests rely on smoke + determinism (re-seed produces same output)
    # for these.
    stochastic_cases = [
        {"name": "mixup_alpha02",      "augmenter": "MixupAugmenter",      "params": {"alpha": 0.2}},
        {"name": "local_mixup",        "augmenter": "LocalMixupAugmenter", "params": {"alpha": 0.2, "k_neighbors": 5}},
        {"name": "particle_size",      "augmenter": "ParticleSizeAugmenter",
         "params": {"mean_size_um": 50.0, "size_variation_um": 15.0,
                    "reference_size_um": 50.0, "wavelength_exponent": 1.5,
                    "size_effect_strength": 0.1,
                    "include_path_length": 1, "path_length_sensitivity": 0.5}},
        {"name": "emsc_distort",       "augmenter": "EMSCDistortionAugmenter",
         "params": {"mult_low": 0.9, "mult_high": 1.1,
                    "add_low": -0.05, "add_high": 0.05,
                    "polynomial_order": 2, "polynomial_strength": 0.02,
                    "correlation": 0.3}},
        {"name": "moisture",           "augmenter": "MoistureAugmenter",
         "params": {"water_activity_delta": 0.1,
                    "reference_water_activity": 0.5,
                    "free_water_fraction": 0.3, "bound_water_shift": 25.0,
                    "moisture_content": 0.10,
                    "enable_shift": 1, "enable_intensity": 1}},
    ]
    for case in stochastic_cases:
        cases.append({**case, "deterministic": False, "output_hex": []})

    fixture: dict[str, Any] = {
        "format": "c4a_aug_phase17_v1",
        "numpy_version": np.__version__,
        "nirs4all_version": version,
        "encoding": "ieee754_binary64_be_hex",
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "input_hex": array_to_hex(X),
        "wavelengths_hex": array_to_hex(wavelengths),
        "cases": cases,
    }
    out_path = out_dir / "aug_phase17_v1.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes, "
          f"{sum(1 for c in cases if c['deterministic'])} deterministic, "
          f"{sum(1 for c in cases if not c['deterministic'])} stochastic)")


if __name__ == "__main__":
    main()
