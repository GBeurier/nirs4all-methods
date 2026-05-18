#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 8 parity fixtures for the orthogonalization operators
(OSC, EPO).

Both operators implement the `_fit / _transform` ABI contract:

  * OSC — Direct OSC. fit(X, y) with univariate target y; transform(X)
          replays the learned orthogonal-component deflation.
  * EPO — External Parameter Orthogonalization. fit(X, d) with univariate
          external parameter d (length n_samples).  transform(X) returns
          the input matrix unchanged (the `d = d_mean` contract), since d
          is not available at transform time.  The fit_transform path is
          covered separately by a `_apply_with_d` smoke test in the c4a
          test suite.

Fixture schema extension (additive over Phase 3): adds optional
`fit_y_hex` (OSC) and `fit_d_hex` (EPO) hex arrays of length `fit_rows`
to encode the supervisor at fit time.  No change to the parser shape
itself — the JSON keys are read directly by the orthogonalization
test file.

Output: parity/fixtures/{osc, epo}_v1.json.
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

from c4a_parity_orthog_ref import (  # noqa: E402
    epo as epo_ref,
    epo_fit_transform,
    osc as osc_ref,
    osc_fit_transform,
)


# ---------------------------------------------------------------------------
# Encoding helpers (shared with the Phase 3 / 5a generators).
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# Synthetic NIR-like spectra + targets / external parameters.
# ---------------------------------------------------------------------------

SEED = 20260518
ROWS = 40
COLS = 50


def synthesize_dataset(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, d).  X has structure correlated with y and a
    distinct structured component correlated with the external parameter
    d so OSC and EPO both have something to learn."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, COLS, dtype=np.float64)
    # Latent factors
    y = rng.uniform(0.0, 1.0, size=ROWS)
    d = rng.uniform(-1.0, 1.0, size=ROWS)
    # Spectra: y-correlated peak + d-correlated baseline shift + noise.
    y_peak_mu = 0.35
    y_peak_sig = 0.08
    y_loading = np.exp(-((wl - y_peak_mu) ** 2) / (2.0 * y_peak_sig ** 2))
    d_loading = 0.4 + 0.6 * wl  # smooth linear shape
    interference_loading = 1.5 * np.exp(-((wl - 0.65) ** 2) / (2.0 * 0.05 ** 2))
    interference_score = rng.normal(0.0, 1.0, size=ROWS)
    X = (
        np.outer(y, y_loading)
        + np.outer(d, d_loading)
        + np.outer(interference_score, interference_loading)
        + 0.5  # offset so the spectra sit in the positive half-plane
        + rng.normal(0.0, 0.02, size=(ROWS, COLS))
    )
    return X, y, d


# ---------------------------------------------------------------------------
# Case builders.
# ---------------------------------------------------------------------------

def osc_cases(X_fit: np.ndarray, y_fit: np.ndarray,
              X_test: np.ndarray) -> list[tuple[str, dict, Callable[[], np.ndarray]]]:
    return [
        ("n1_scale_true",
         {"n_components": 1, "scale": True, "variant": "transform"},
         lambda: osc_ref(X_fit, y_fit, X_test, n_components=1, scale=True)),
        ("n2_scale_true",
         {"n_components": 2, "scale": True, "variant": "transform"},
         lambda: osc_ref(X_fit, y_fit, X_test, n_components=2, scale=True)),
        ("n1_scale_false",
         {"n_components": 1, "scale": False, "variant": "transform"},
         lambda: osc_ref(X_fit, y_fit, X_test, n_components=1, scale=False)),
        ("n2_fit_transform",
         {"n_components": 2, "scale": True, "variant": "fit_transform"},
         lambda: osc_fit_transform(X_fit, y_fit, n_components=2,
                                    scale=True)["X_transformed"]),
    ]


def epo_cases(X_fit: np.ndarray, d_fit: np.ndarray,
              X_test: np.ndarray) -> list[tuple[str, dict, Callable[[], np.ndarray]]]:
    return [
        # _transform (no d) — pass-through contract.
        ("scale_true_transform",
         {"scale": True, "variant": "transform"},
         lambda: epo_ref(X_fit, d_fit, X_test, scale=True)),
        ("scale_false_transform",
         {"scale": False, "variant": "transform"},
         lambda: epo_ref(X_fit, d_fit, X_test, scale=False)),
        # fit_transform — uses d_fit at training time, exercises the
        # learned projection.
        ("scale_true_fit_transform",
         {"scale": True, "variant": "fit_transform"},
         lambda: epo_fit_transform(X_fit, d_fit, scale=True)["X_filtered"]),
        ("scale_false_fit_transform",
         {"scale": False, "variant": "fit_transform"},
         lambda: epo_fit_transform(X_fit, d_fit, scale=False)["X_filtered"]),
    ]


# ---------------------------------------------------------------------------
# Fixture writer.
# ---------------------------------------------------------------------------

def write_fixture(op_id: str,
                   fit_X: np.ndarray, fit_aux: np.ndarray, fit_aux_key: str,
                   test_X: np.ndarray,
                   cases: list[tuple[str, dict, Callable[[], np.ndarray]]],
                   out_dir: Path) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_pp_{op_id}_v1",
        "numpy_version": np.__version__,
        "reference": "c4a_parity_orthog_ref (frozen against nirs4all==0.8.x)",
        "encoding": "ieee754_binary64_be_hex",
        "fit_rows": int(fit_X.shape[0]),
        "fit_cols": int(fit_X.shape[1]),
        "fit_input_hex": array_to_hex(fit_X),
        fit_aux_key: array_to_hex(fit_aux),
        "rows": int(test_X.shape[0]),
        "cols": int(test_X.shape[1]),
        "input_hex": array_to_hex(test_X),
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
    out = out_dir / f"{op_id}_v1.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=None)
        f.write("\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes, {len(cases)} cases)")


def main() -> None:
    out_dir = REPO_ROOT / "parity" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Two distinct (X, y, d) datasets — one for fit (training), one for
    # transform (held-out). Same generator family, neighbouring seeds.
    fit_X, fit_y, fit_d = synthesize_dataset(SEED)
    test_X, _test_y, _test_d = synthesize_dataset(SEED + 1)
    print(f"fit_X shape {fit_X.shape}, range [{fit_X.min():.4f}, {fit_X.max():.4f}]")
    print(f"fit_y range [{fit_y.min():.4f}, {fit_y.max():.4f}]")
    print(f"fit_d range [{fit_d.min():.4f}, {fit_d.max():.4f}]")

    write_fixture("osc", fit_X, fit_y, "fit_y_hex", test_X,
                  osc_cases(fit_X, fit_y, test_X), out_dir)
    write_fixture("epo", fit_X, fit_d, "fit_d_hex", test_X,
                  epo_cases(fit_X, fit_d, test_X), out_dir)


if __name__ == "__main__":
    main()
