#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 18 parity fixtures for the augmenter slice.

Twelve augmenters share the universal ``c4a_aug_*`` ABI shape locked in
``roadmap/phase-15-18-augmenters-abi-contract.md``. Bit-exact NumPy parity
of the Phase 18 stochastic operators requires a complete in-engine port of
``np.random.Generator.uniform``, ``choice(replace=False)`` and friends on
top of the PCG64 stream we already match in Phase 1. That port is gated
behind v2 of the augmenter ABI.

For Phase 18 we ship **shape + finite-value fixtures**: synthetic input
matrices, the documented operator parameters per case, and the expected
output **shape** (rows × cols), so the C++ harness can run a smoke test
that asserts shape + finite output + determinism rather than per-cell
parity. The fixture format is the existing ``c4a_aug_*_v1`` family — the
``output_hex`` block holds the *input* unchanged so consumers see a
well-formed fixture and the parity harness's "output_shape" path is
exercised.

The two v2-deferred operators (Spline_X_Simplification,
Spline_Curve_Simplification) are listed in DEFERRALS.md and not exercised
here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from struct import pack
from typing import Any

import numpy as np


def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


SEED = 20260518
ROWS = 8
COLS = 32


def synth_spectra(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    wl = np.linspace(1000.0, 1700.0, COLS, dtype=np.float64)
    X = np.empty((ROWS, COLS), dtype=np.float64)
    for i in range(ROWS):
        a = 0.3 + 0.05 * float(i)
        b = 0.001 * float(i)
        cmu = 1200.0 + 50.0 * float(i)
        cs = 80.0
        amp = 0.8
        peak = amp * np.exp(-0.5 * ((wl - cmu) / cs) ** 2)
        noise = rng.normal(0.0, 0.005, COLS)
        X[i] = a + b * (wl - 1000.0) + peak + noise
    return X, wl


def write_smoke_fixture(name: str, op_id: str, X: np.ndarray,
                         wl: np.ndarray | None,
                         cases: list[tuple[str, dict, np.ndarray]],
                         out_dir: Path, nirs4all_version: str) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_aug_{op_id}_v1",
        "numpy_version": np.__version__,
        "nirs4all_version": nirs4all_version,
        "encoding": "ieee754_binary64_be_hex",
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "input_hex": array_to_hex(X),
        "cases": [],
    }
    if wl is not None:
        fixture["wavelengths_hex"] = array_to_hex(wl)
    for case_name, params, Y in cases:
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
    X, wl = synth_spectra(SEED)

    # All operators ship a single shape-fixture case that echoes the
    # input. The C++ harness only consumes the input + shape per case;
    # the parity check is shape + finite-value.
    operators = [
        ("aug_detector_rolloff",
         {"detector_model": 4, "effect_strength": 1.0,
          "noise_amplification": 0.02, "include_baseline_distortion": 1}),
        ("aug_stray_light",
         {"stray_light_fraction": 0.001, "edge_enhancement": 2.0,
          "edge_width": 0.1, "include_peak_truncation": 1}),
        ("aug_edge_curve",
         {"curvature_strength": 0.02, "curvature_type": 1,
          "asymmetry": 0.0, "edge_focus": 0.7}),
        ("aug_truncated_peak",
         {"peak_probability": 0.5,
          "amplitude_min": 0.01, "amplitude_max": 0.1,
          "width_min": 50.0, "width_max": 200.0,
          "left_edge": 1, "right_edge": 1}),
        ("aug_edge_artifacts",
         {"enabled_flags": 0xF, "overall_strength": 1.0,
          "detector_model": 4}),
        ("aug_spline_smooth", {}),
        ("aug_spline_x_perturb",
         {"spline_degree": 3, "perturbation_density": 0.05,
          "perturbation_range_min": -0.1, "perturbation_range_max": 0.1}),
        ("aug_spline_y_perturb",
         {"spline_points": -1, "perturbation_intensity": 0.005}),
        ("aug_rotate_translate", {"p_range": 2.0, "y_factor": 3.0}),
        ("aug_random_x_op",
         {"op_kind": 0, "operator_range_min": 0.97,
          "operator_range_max": 1.03}),
    ]
    # The two v2-deferred operators (spline_x_simplify, spline_curve_simplify)
    # are documented as NOT_IMPLEMENTED stubs in DEFERRALS.md and intentionally
    # not in this fixture set.
    needs_wl = {
        "aug_detector_rolloff", "aug_stray_light", "aug_edge_curve",
        "aug_truncated_peak", "aug_edge_artifacts",
    }
    n4a_init = Path("/home/delete/nirs4all/nirs4all/nirs4all/__init__.py")
    version = "unknown"
    if n4a_init.exists():
        for line in n4a_init.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("__version__"):
                version = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    for op_id, params in operators:
        cases = [("default", params, X.copy())]
        wl_for = wl if op_id in needs_wl else None
        write_smoke_fixture(op_id, op_id, X, wl_for, cases, out_dir, version)


if __name__ == "__main__":
    main()
