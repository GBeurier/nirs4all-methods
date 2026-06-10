#!/usr/bin/env python3
"""Timing smoke benchmark for direct native moment/linear heads.

This covers the direct reusable heads that are not preprocessing screens:
Ridge, PLS, PCR, CPPLS, weighted PLS, robust PLS, ridge-augmented PLS,
continuum regression and ECR. Each head is timed as both an ABI-close function
and a sklearn-style wrapper fit+predict path.
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
    NativePLSRegressor,
    NativeRidgeRegressor,
    NativeRidgePLSRegressor,
    NativeRobustPLSRegressor,
    NativeWeightedPLSRegressor,
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


PLS_ROUTE_FIELDS = (
    "n_pls_moment_cv_fits",
    "n_pls_moment_host_cv_fits",
    "n_pls_moment_cuda_device_cv_fits",
    "n_pls_moment_cuda_parallel_fold_batches",
    "n_pls_moment_cuda_parallel_fold_jobs",
    "n_pls_moment_cuda_many_batched_batches",
    "n_pls_moment_cuda_many_batched_jobs",
    "n_pls_materialized_cv_fits",
    "n_pls_moment_final_fits",
    "n_pls_moment_host_final_fits",
    "n_pls_moment_cuda_device_final_fits",
    "n_pls_materialized_final_fits",
)


def direct_specs(args):
    pls_kwargs = {
        "n_components": 3,
        "cv": 4,
        "scale_x": False,
        "cuda_pls_parallel_folds": bool(args.cuda_pls_parallel_folds),
        "cuda_pls_min_device_features": args.cuda_pls_min_device_features,
        "cuda_pls_many_batched": bool(args.cuda_pls_many_batched),
    }
    return (
        (
            "ridge",
            n4m.ridge,
            NativeRidgeRegressor,
            {"alpha": 0.1, "scale_x": False},
            True,
        ),
        (
            "pls",
            n4m.pls,
            NativePLSRegressor,
            pls_kwargs,
            True,
        ),
        (
            "pcr",
            n4m.pcr,
            NativePCRRegressor,
            {"n_components": 3, "scale_x": False},
            True,
        ),
        (
            "cppls",
            n4m.cppls,
            NativeCPPLSRegressor,
            {"gamma": 0.5, "n_components": 3},
            True,
        ),
        (
            "weighted_pls",
            n4m.weighted_pls,
            NativeWeightedPLSRegressor,
            {"sample_weights": None, "n_components": 3, "scale_x": False},
            True,
        ),
        (
            "robust_pls",
            n4m.robust_pls,
            NativeRobustPLSRegressor,
            {
                "huber_k": 1.345,
                "max_irls_iter": 3,
                "n_components": 3,
                "scale_x": False,
            },
            True,
        ),
        (
            "ridge_pls",
            n4m.ridge_pls,
            NativeRidgePLSRegressor,
            {"ridge_lambda": 0.1, "n_components": 3, "scale_x": False},
            True,
        ),
        (
            "continuum_regression",
            n4m.continuum_regression,
            NativeContinuumRegressionRegressor,
            {"tau": 0.5, "n_components": 3},
            True,
        ),
        (
            "ecr",
            n4m.ecr,
            NativeECRRegressor,
            {"alpha": 0.5, "n_components": 3},
            True,
        ),
    )


def route_fields(result, kwargs):
    fields = {
        "cuda_pls_parallel_folds": kwargs.get("cuda_pls_parallel_folds", ""),
        "cuda_pls_min_device_features": kwargs.get("cuda_pls_min_device_features", ""),
        "cuda_pls_many_batched": kwargs.get("cuda_pls_many_batched", ""),
    }
    for key in PLS_ROUTE_FIELDS:
        fields[key] = int(float(result.get(key, 0.0)))
    return fields


def function_row(
    method,
    n_samples,
    n_features,
    elapsed_ms,
    result,
    kwargs,
    has_wrapper: bool = False,
):
    row = {
        "backend": "native_function",
        "method": method,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components": kwargs.get("n_components", ""),
        "alpha": kwargs.get("alpha", ""),
        "gamma": kwargs.get("gamma", ""),
        "tau": kwargs.get("tau", ""),
        "huber_k": kwargs.get("huber_k", ""),
        "max_irls_iter": kwargs.get("max_irls_iter", ""),
        "ridge_lambda": kwargs.get("ridge_lambda", ""),
        "surface_status": (
            "function_and_sklearn_replay" if has_wrapper else "function_only"
        ),
        "rmse": float(result["rmse"]),
        "replay_max_abs_error": 0.0,
        "elapsed_ms_median": elapsed_ms,
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }
    row.update(route_fields(result, kwargs))
    return row


def wrapper_row(method, n_samples, n_features, elapsed_ms, model, native_result, kwargs, X):
    pred = np.asarray(model.predict(X), dtype=np.float64).reshape(
        native_result["predictions"].shape
    )
    replay_error = float(np.max(np.abs(pred - native_result["predictions"])))
    row = {
        "backend": "sklearn_fit_predict",
        "method": method,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components": kwargs.get("n_components", ""),
        "alpha": kwargs.get("alpha", ""),
        "gamma": kwargs.get("gamma", ""),
        "tau": kwargs.get("tau", ""),
        "huber_k": kwargs.get("huber_k", ""),
        "max_irls_iter": kwargs.get("max_irls_iter", ""),
        "ridge_lambda": kwargs.get("ridge_lambda", ""),
        "surface_status": "function_and_sklearn_replay",
        "rmse": float(model.rmse_),
        "replay_max_abs_error": replay_error,
        "elapsed_ms_median": elapsed_ms,
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }
    row.update(route_fields(model.result_, kwargs))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/direct_moment_heads_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--cuda-pls-parallel-folds",
        action="store_true",
        help="Enable bounded CUDA-stream exact PLS fold scheduling for n4m.pls.",
    )
    parser.add_argument(
        "--cuda-pls-min-device-features",
        type=int,
        default=None,
        help="Override the minimum feature count for PLS device execution.",
    )
    parser.add_argument(
        "--cuda-pls-many-batched",
        action="store_true",
        help="Enable the experimental many-batched CUDA PLS route for n4m.pls.",
    )
    args = parser.parse_args()

    shapes = [(48, 24), (96, 64), (160, 128)]
    rows = []
    for shape_id, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=8400 + shape_id)
        weights = np.linspace(0.5, 1.5, X.shape[0], dtype=np.float64)
        for method, fn, Estimator, kwargs, has_wrapper in direct_specs(args):
            if kwargs.get("sample_weights") is None and method == "weighted_pls":
                kwargs = dict(kwargs)
                kwargs["sample_weights"] = weights
            elapsed, result = median_ms(lambda: fn(X, y, **kwargs), args.repeats)
            rows.append(
                function_row(
                    method,
                    n_samples,
                    n_features,
                    elapsed,
                    result,
                    kwargs,
                    has_wrapper,
                )
            )
            if not has_wrapper:
                continue

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
