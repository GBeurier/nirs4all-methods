#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 16 parity fixtures for the wavelength + spectral augmenters.

Ten augmenters; one fixture file each. The reference comes from the frozen
``c4a_parity_aug_spectral_ref`` package under ``parity/python_generator/src/``.

Each fixture exercises one operator with a small set of parameter
combinations. The input matrix is a (n_samples, n_features) NIR-like
synthetic spectrum block, the random stream is locked by a per-case
``seed`` parameter (the C engine calls ``c4a_rng_pcg64_set_seed`` before
each apply).

Fixture schema (consumed by the existing ``fixture_parser.hpp``):

  {
    "format": "c4a_aug_<NAME>_v1",
    "numpy_version": "...",
    "reference": "c4a_parity_aug_spectral_ref",
    "encoding": "ieee754_binary64_be_hex",
    "rows": <int>, "cols": <int>,
    "input_hex": [ "<16 hex>", ... ],
    "cases": [
      { "name": "...",
        "params": { "seed": <int>, ...op-specific... },
        "output_rows": <int>, "output_cols": <int>,
        "output_hex": [ "<16 hex>", ... ] },
      ...
    ]
  }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from struct import pack
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "parity" / "python_generator" / "src"))

from c4a_parity_aug_spectral_ref import (  # noqa: E402
    band_mask,
    band_perturb,
    channel_dropout,
    gauss_jitter,
    local_clip,
    local_warp,
    magnitude_warp,
    unsharp_mask,
    wavelength_shift,
    wavelength_stretch,
)


SEED = 20260518


def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


def synthesize_nir_block(n_samples: int, n_features: int,
                          seed: int) -> np.ndarray:
    """Generate a smooth NIR-like (n_samples, n_features) block."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, n_features, dtype=np.float64)
    X = np.empty((n_samples, n_features), dtype=np.float64)
    for i in range(n_samples):
        a = rng.uniform(0.4, 0.8)
        b = rng.uniform(-0.3, 0.4)
        c = rng.uniform(-0.2, 0.3)
        baseline = a + b * wl + c * wl * wl
        n_peaks = int(rng.integers(2, 5))
        peaks = np.zeros(n_features, dtype=np.float64)
        for _ in range(n_peaks):
            mu = rng.uniform(0.1, 0.95)
            sigma = rng.uniform(0.02, 0.08)
            amp = rng.uniform(0.2, 1.0)
            peaks += amp * np.exp(-((wl - mu) ** 2) / (2.0 * sigma ** 2))
        noise = rng.normal(0.0, 0.005, size=n_features)
        X[i] = baseline + peaks + noise
    return X


def write_fixture(name: str, X: np.ndarray, cases: list[tuple],
                   out_dir: Path) -> None:
    """`cases` is a list of (case_name, params_dict, np.ndarray) tuples."""
    fixture: dict[str, Any] = {
        "format": f"c4a_aug_{name}_v1",
        "numpy_version": np.__version__,
        "reference": "c4a_parity_aug_spectral_ref",
        "encoding": "ieee754_binary64_be_hex",
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "input_hex": array_to_hex(X),
        "cases": [],
    }
    for case_name, params, expected in cases:
        fixture["cases"].append({
            "name": case_name,
            "params": params,
            "output_rows": int(expected.shape[0]),
            "output_cols": int(expected.shape[1]),
            "output_hex": array_to_hex(expected),
        })
    out_path = out_dir / f"aug_{name}_v1.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out_path} "
          f"({out_path.stat().st_size:,} bytes, {len(cases)} cases)")


def main() -> None:
    out_dir = REPO_ROOT / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Small block — 12 samples × 40 features. The augmenters scale linearly
    # in rows so the fixtures stay compact while still touching the
    # vectorised paths.
    X = synthesize_nir_block(n_samples=12, n_features=40, seed=SEED)
    print(f"synthetic block: shape {X.shape}, range "
          f"[{X.min():.4f}, {X.max():.4f}]")

    # -------- WavelengthShift --------------------------------------------
    cases_ws: list[tuple[str, dict, np.ndarray]] = []
    for seed, lo, hi in [(101, -2.0, 2.0), (202, -1.0, 1.0)]:
        rng = np.random.default_rng(seed)
        out = wavelength_shift(X, rng, shift_lo=lo, shift_hi=hi)
        cases_ws.append((f"seed{seed}_range{lo}_{hi}",
                          {"seed": seed, "shift_lo": lo, "shift_hi": hi},
                          out))
    write_fixture("wavelength_shift", X, cases_ws, out_dir)

    # -------- WavelengthStretch ------------------------------------------
    cases_wst: list[tuple[str, dict, np.ndarray]] = []
    for seed, lo, hi in [(103, 0.98, 1.02), (204, 0.95, 1.05)]:
        rng = np.random.default_rng(seed)
        out = wavelength_stretch(X, rng, stretch_lo=lo, stretch_hi=hi)
        cases_wst.append((f"seed{seed}_range{lo}_{hi}",
                           {"seed": seed, "stretch_lo": lo,
                            "stretch_hi": hi},
                           out))
    write_fixture("wavelength_stretch", X, cases_wst, out_dir)

    # -------- LocalWavelengthWarp ----------------------------------------
    cases_lw: list[tuple[str, dict, np.ndarray]] = []
    for seed, k, ms in [(105, 5, 1.0), (206, 7, 0.5)]:
        rng = np.random.default_rng(seed)
        out = local_warp(X, rng, n_control_points=k, max_shift=ms)
        cases_lw.append((f"seed{seed}_k{k}_max{ms}",
                          {"seed": seed, "n_control_points": k,
                           "max_shift": ms},
                          out))
    write_fixture("local_warp", X, cases_lw, out_dir)

    # -------- BandPerturbation --------------------------------------------
    cases_bp: list[tuple[str, dict, np.ndarray]] = []
    for seed, nb in [(107, 3), (208, 5)]:
        rng = np.random.default_rng(seed)
        out = band_perturb(X, rng, n_bands=nb,
                            bw_lo=5, bw_hi=15,
                            gain_lo=0.9, gain_hi=1.1,
                            offset_lo=-0.01, offset_hi=0.01)
        cases_bp.append((f"seed{seed}_nb{nb}",
                          {"seed": seed, "n_bands": nb,
                           "bw_lo": 5, "bw_hi": 15,
                           "gain_lo": 0.9, "gain_hi": 1.1,
                           "offset_lo": -0.01, "offset_hi": 0.01},
                          out))
    write_fixture("band_perturb", X, cases_bp, out_dir)

    # -------- BandMasking -------------------------------------------------
    cases_bm: list[tuple[str, dict, np.ndarray]] = []
    for seed, nb_lo, nb_hi, mode in [(109, 1, 3, "zero"),
                                      (210, 2, 4, "interp")]:
        rng = np.random.default_rng(seed)
        out = band_mask(X, rng, n_bands_lo=nb_lo, n_bands_hi=nb_hi,
                         bw_lo=5, bw_hi=15, mode=mode)
        cases_bm.append((f"seed{seed}_{mode}_{nb_lo}_{nb_hi}",
                          {"seed": seed, "n_bands_lo": nb_lo,
                           "n_bands_hi": nb_hi, "bw_lo": 5, "bw_hi": 15,
                           "mode": mode},
                          out))
    write_fixture("band_mask", X, cases_bm, out_dir)

    # -------- ChannelDropout ----------------------------------------------
    cases_cd: list[tuple[str, dict, np.ndarray]] = []
    for seed, p, mode in [(111, 0.05, "zero"), (212, 0.1, "interp")]:
        rng = np.random.default_rng(seed)
        out = channel_dropout(X, rng, dropout_prob=p, mode=mode)
        cases_cd.append((f"seed{seed}_{mode}_p{p}",
                          {"seed": seed, "dropout_prob": p, "mode": mode},
                          out))
    write_fixture("channel_dropout", X, cases_cd, out_dir)

    # -------- GaussianSmoothingJitter -------------------------------------
    cases_gj: list[tuple[str, dict, np.ndarray]] = []
    for seed, lo, hi, w in [(113, 0.5, 1.5, 9), (214, 1.0, 2.5, 13)]:
        rng = np.random.default_rng(seed)
        out = gauss_jitter(X, rng, sigma_lo=lo, sigma_hi=hi, kernel_width=w)
        cases_gj.append((f"seed{seed}_sig{lo}_{hi}_w{w}",
                          {"seed": seed, "sigma_lo": lo, "sigma_hi": hi,
                           "kernel_width": w},
                          out))
    write_fixture("gauss_jitter", X, cases_gj, out_dir)

    # -------- UnsharpSpectralMask -----------------------------------------
    cases_um: list[tuple[str, dict, np.ndarray]] = []
    for seed, lo, hi, sigma, w in [(115, 0.1, 0.5, 1.0, 11),
                                    (216, 0.2, 0.7, 1.5, 13)]:
        rng = np.random.default_rng(seed)
        out = unsharp_mask(X, rng, amount_lo=lo, amount_hi=hi,
                            sigma=sigma, kernel_width=w)
        cases_um.append((f"seed{seed}_amt{lo}_{hi}_sig{sigma}_w{w}",
                          {"seed": seed, "amount_lo": lo, "amount_hi": hi,
                           "sigma": sigma, "kernel_width": w},
                          out))
    write_fixture("unsharp_mask", X, cases_um, out_dir)

    # -------- SmoothMagnitudeWarp -----------------------------------------
    cases_mw: list[tuple[str, dict, np.ndarray]] = []
    for seed, k, lo, hi in [(117, 5, 0.9, 1.1), (218, 7, 0.85, 1.15)]:
        rng = np.random.default_rng(seed)
        out = magnitude_warp(X, rng, n_control_points=k,
                              gain_lo=lo, gain_hi=hi)
        cases_mw.append((f"seed{seed}_k{k}_g{lo}_{hi}",
                          {"seed": seed, "n_control_points": k,
                           "gain_lo": lo, "gain_hi": hi},
                          out))
    write_fixture("magnitude_warp", X, cases_mw, out_dir)

    # -------- LocalClipping ----------------------------------------------
    cases_lc: list[tuple[str, dict, np.ndarray]] = []
    for seed, nr, lo, hi in [(119, 1, 5, 15), (220, 2, 8, 20)]:
        rng = np.random.default_rng(seed)
        out = local_clip(X, rng, n_regions=nr, width_lo=lo, width_hi=hi)
        cases_lc.append((f"seed{seed}_nr{nr}_w{lo}_{hi}",
                          {"seed": seed, "n_regions": nr,
                           "width_lo": lo, "width_hi": hi},
                          out))
    write_fixture("local_clip", X, cases_lc, out_dir)


if __name__ == "__main__":
    main()
