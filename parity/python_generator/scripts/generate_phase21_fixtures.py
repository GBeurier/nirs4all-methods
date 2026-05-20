#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Generate Phase 21 parity fixtures for the FCKStaticTransformer operator.

The FCK static bank takes (alphas, scales, kernel_size) and produces L =
n_orders * n_scales convolutional kernels. For each input row of length p
the operator emits L convolved bands of length p, concatenated along axis 1
into shape (n_samples, L * p).

The canonical Python reference is
``nirs4all.operators.transforms.fck_static._build_kernel``. Convolution uses
``scipy.ndimage.convolve1d(X, kernel, axis=1, mode='nearest')`` so the
resulting fixture is bit-identical when the C engine implements the same
recipe.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from struct import pack
from typing import Any, Callable

import numpy as np
import scipy.ndimage as ndimage

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from parity.nirs4all_source import find_nirs4all_root, get_nirs4all_version  # noqa: E402


NIRS4ALL_ROOT = find_nirs4all_root()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pull in _build_kernel from fck_static.py without importing the full
# nirs4all package (which would drag in TF / Torch / etc.).
fck_static_mod = _load(NIRS4ALL_ROOT / "operators/transforms/fck_static.py",
                       "_n4a_fck_static_phase21")
build_kernel = fck_static_mod._build_kernel


# ---------------------------------------------------------------------------
# Encoding helpers.
# ---------------------------------------------------------------------------

def double_to_hex(value: float) -> str:
    return pack(">d", float(value)).hex()


def array_to_hex(arr: np.ndarray) -> list[str]:
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel(order="C")
    return [double_to_hex(float(v)) for v in flat]


# ---------------------------------------------------------------------------
# Synthetic NIR-like spectra (same generator family as Phase 4).
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
# Reference implementation.
# ---------------------------------------------------------------------------

def ref_fck_static(X: np.ndarray, *, kernel_size: int,
                    alphas: list[float], sigmas: list[float]) -> np.ndarray:
    """Apply the FCK static bank: Cartesian product alphas x scales, alpha slowest.

    Output shape: (n_samples, L * n_features), L = len(alphas) * len(sigmas),
    bands laid out as alpha-then-sigma, then convolution along axis=1 with
    scipy.ndimage.convolve1d in mode='nearest'.

    The fixture key remains ``sigmas`` for backwards compatibility with the
    Python binding's historical constructor alias; the values are FCK scales.
    """
    n_samples, n_features = X.shape
    n_orders = len(alphas)
    n_scales = len(sigmas)
    L = n_orders * n_scales
    out = np.empty((n_samples, L, n_features), dtype=np.float64)
    for a, alpha in enumerate(alphas):
        for s, scale in enumerate(sigmas):
            kernel = build_kernel(
                float(alpha),
                float(scale),
                int(kernel_size),
                3.0,
            )
            band = ndimage.convolve1d(X, kernel, axis=1, mode='nearest')
            out[:, a * n_scales + s, :] = band
    return out.reshape(n_samples, L * n_features)


# ---------------------------------------------------------------------------
# Case lists.
# ---------------------------------------------------------------------------

def fck_static_cases(X: np.ndarray):
    cases: list[tuple[str, dict, Callable[[], np.ndarray]]] = []

    # Case 1: small bank, K=11, smoother + first-deriv-like.
    p = {"kernel_size": 11, "alphas": [0.0, 1.0], "sigmas": [2.0]}
    cases.append(("K11_a01_s2",
                   p,
                   lambda X=X, p=p: ref_fck_static(X, **p)))

    # Case 2: K=15, two alphas x two scales (4 bands).
    p = {"kernel_size": 15, "alphas": [0.5, 1.5], "sigmas": [1.5, 3.0]}
    cases.append(("K15_a05_15_s15_30",
                   p,
                   lambda X=X, p=p: ref_fck_static(X, **p)))

    # Case 3: K=21, single alpha (pure Gaussian), three scales.
    p = {"kernel_size": 21, "alphas": [0.0], "sigmas": [1.0, 2.5, 5.0]}
    cases.append(("K21_smooth_s1_25_5",
                   p,
                   lambda X=X, p=p: ref_fck_static(X, **p)))

    # Case 4: K=13, 4 alphas x 2 scales (8 bands) — the "default-ish" bank.
    p = {"kernel_size": 13, "alphas": [0.0, 0.5, 1.0, 2.0],
         "sigmas": [2.0, 4.0]}
    cases.append(("K13_a0_05_1_2_s2_4",
                   p,
                   lambda X=X, p=p: ref_fck_static(X, **p)))

    # Case 5: alpha below 0.1 exercises the pure Gaussian branch with a
    # non-zero alpha value.
    p = {"kernel_size": 11, "alphas": [0.05], "sigmas": [3.0]}
    cases.append(("K11_a005_s3",
                   p,
                   lambda X=X, p=p: ref_fck_static(X, **p)))

    return cases


# ---------------------------------------------------------------------------
# Fixture writer.
# ---------------------------------------------------------------------------

def write_fixture(name: str, op_id: str, X: np.ndarray,
                  cases: list[tuple[str, dict, Callable[[], np.ndarray]]],
                  out_dir: Path, nirs4all_version: str) -> None:
    fixture: dict[str, Any] = {
        "format": f"c4a_pp_{op_id}_v1",
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "nirs4all_version": nirs4all_version,
        "encoding": "ieee754_binary64_be_hex",
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "input_hex": array_to_hex(X),
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

    version = get_nirs4all_version(NIRS4ALL_ROOT)

    X = synthesize_spectra(SEED)
    print(f"X shape {X.shape}, range [{X.min():.4f}, {X.max():.4f}]")

    write_fixture("fck_static", "fck_static", X, fck_static_cases(X),
                   out_dir, version)


if __name__ == "__main__":
    main()
