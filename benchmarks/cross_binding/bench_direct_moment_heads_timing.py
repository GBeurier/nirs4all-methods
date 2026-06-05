#!/usr/bin/env python3
"""Timing smoke benchmark for direct native moment/linear heads.

This covers the direct reusable heads that are not preprocessing screens:
Ridge, PCR, CPPLS, continuum regression and ECR. It times both the ABI-close
function and the sklearn-style wrapper fit+predict path, then records the
train-prediction replay error.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

import n4m
from n4m.sklearn import (
    NativeContinuumRegressionRegressor,
    NativeCPPLSRegressor,
    NativeECRRegressor,
    NativePCRRegressor,
    NativeRidgeRegressor,
)


def make_dataset(n_samples: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (
        0.90 * X[:, 0]
        - 0.45 * X[:, min(5, n_features - 1)]
        + 0.20 * X[:, min(17, n_features - 1)]
        + 0.04 * rng.standard_normal(n_samples)
    )
    return X, y


def median_ms(fn, repeats: int) -> tuple[float, object]:
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(times)), result


def direct_specs():
    return (
        (
            "ridge",
            n4m.ridge,
            NativeRidgeRegressor,
            {"alpha": 0.1, "scale_x": False},
        ),
        (
            "pcr",
            n4m.pcr,
            NativePCRRegressor,
            {"n_components": 3, "scale_x": False},
        ),
        (
            "cppls",
            n4m.cppls,
            NativeCPPLSRegressor,
            {"gamma": 0.5, "n_components": 3},
        ),
        (
            "continuum_regression",
            n4m.continuum_regression,
            NativeContinuumRegressionRegressor,
            {"tau": 0.5, "n_components": 3},
        ),
        (
            "ecr",
            n4m.ecr,
            NativeECRRegressor,
            {"alpha": 0.5, "n_components": 3},
        ),
    )


def function_row(method, n_samples, n_features, elapsed_ms, result, kwargs):
    return {
        "backend": "native_function",
        "method": method,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components": kwargs.get("n_components", ""),
        "alpha": kwargs.get("alpha", ""),
        "gamma": kwargs.get("gamma", ""),
        "tau": kwargs.get("tau", ""),
        "rmse": float(result["rmse"]),
        "replay_max_abs_error": 0.0,
        "elapsed_ms_median": elapsed_ms,
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }


def wrapper_row(method, n_samples, n_features, elapsed_ms, model, native_result, kwargs, X):
    pred = np.asarray(model.predict(X), dtype=np.float64).reshape(
        native_result["predictions"].shape
    )
    replay_error = float(np.max(np.abs(pred - native_result["predictions"])))
    return {
        "backend": "sklearn_fit_predict",
        "method": method,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components": kwargs.get("n_components", ""),
        "alpha": kwargs.get("alpha", ""),
        "gamma": kwargs.get("gamma", ""),
        "tau": kwargs.get("tau", ""),
        "rmse": float(model.rmse_),
        "replay_max_abs_error": replay_error,
        "elapsed_ms_median": elapsed_ms,
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/direct_moment_heads_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    shapes = [(48, 24), (96, 64), (160, 128)]
    rows = []
    for shape_id, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=8400 + shape_id)
        for method, fn, Estimator, kwargs in direct_specs():
            elapsed, result = median_ms(lambda: fn(X, y, **kwargs), args.repeats)
            rows.append(
                function_row(method, n_samples, n_features, elapsed, result, kwargs)
            )

            def fit_predict():
                model = Estimator(**kwargs).fit(X, y)
                model.predict(X)
                return model

            elapsed, model = median_ms(fit_predict, args.repeats)
            rows.append(
                wrapper_row(
                    method,
                    n_samples,
                    n_features,
                    elapsed,
                    model,
                    result,
                    kwargs,
                    X,
                )
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
