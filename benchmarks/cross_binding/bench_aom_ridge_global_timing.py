#!/usr/bin/env python3
"""Timing smoke benchmark for n4m.aom_ridge_global."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

import n4m
from n4m.sklearn import NativeAOMRidgeGlobalRegressor


OPERATORS = (
    "identity",
    ("finite_difference", [1]),
    ("savgol_smooth", [5, 2]),
)


def make_dataset(n_samples: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (
        0.60 * X[:, 0]
        - 0.25 * X[:, min(4, n_features - 1)]
        + 0.15 * X[:, min(15, n_features - 1)]
        + 0.05 * rng.standard_normal(n_samples)
    )
    return X, y


def balanced_folds(n_samples: int, cv: int) -> np.ndarray:
    return np.arange(n_samples, dtype=np.int32) % int(cv)


def median_ms(fn, repeats: int) -> tuple[float, object]:
    timings = []
    result = None
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        result = fn()
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(timings)), result


def run_function(X, y, folds, cv, lambdas, moment_policy):
    return n4m.aom_ridge_global(
        X,
        y,
        operators=OPERATORS,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        scale_x=False,
        moment_policy=moment_policy,
    )


def run_wrapper(X, y, folds, cv, lambdas, moment_policy):
    return NativeAOMRidgeGlobalRegressor(
        operators=OPERATORS,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        scale_x=False,
        moment_policy=moment_policy,
    ).fit(X, y)


def row(backend, n_samples, n_features, cv, moment_policy, elapsed_ms, result, replay_error):
    scores = np.asarray(result["candidate_scores"], dtype=np.float64)
    return {
        "backend": backend,
        "n_samples": n_samples,
        "n_features": n_features,
        "cv": cv,
        "moment_policy": moment_policy,
        "n_operators": int(result["n_operators"]),
        "n_candidates": int(result["n_candidates"]),
        "selected_operator_index": int(result["selected_operator_index"]),
        "selected_operator_kind": int(result["selected_operator_kind"]),
        "selected_alpha": result["selected_param"],
        "selected_cv_rmse": result["selected_cv_rmse"],
        "best_cv_rmse": float(np.min(scores[:, 4])),
        "ridge_backend": result["ridge_backend"],
        "n_ridge_moment_cv_fits": int(result["n_ridge_moment_cv_fits"]),
        "n_ridge_moment_eigen_path_preparations": int(
            result.get("n_ridge_moment_eigen_path_preparations", 0)
        ),
        "n_ridge_moment_eigen_path_cv_fits": int(
            result.get("n_ridge_moment_eigen_path_cv_fits", 0)
        ),
        "n_ridge_moment_direct_cv_fits": int(
            result.get("n_ridge_moment_direct_cv_fits", 0)
        ),
        "n_ridge_dual_materialized_cv_fits": int(result["n_ridge_dual_materialized_cv_fits"]),
        "prediction_replay_max_abs_error": replay_error,
        "elapsed_ms_median": elapsed_ms,
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/aom_ridge_global_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cv", type=int, default=4)
    parser.add_argument("--moment-policy", default="auto")
    parser.add_argument(
        "--mode",
        default="native",
        choices=("native", "wrapper", "both"),
        help="Benchmark ABI-close function, sklearn wrapper, or both",
    )
    args = parser.parse_args()

    shapes = [(32, 32), (64, 64), (96, 128)]
    lambdas = np.asarray([1e-3, 1e-1, 10.0], dtype=np.float64)
    rows = []
    for i, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=7500 + i)
        folds = balanced_folds(n_samples, args.cv)
        if args.mode in {"native", "both"}:
            elapsed, result = median_ms(
                lambda: run_function(
                    X,
                    y,
                    folds,
                    args.cv,
                    lambdas,
                    args.moment_policy,
                ),
                args.repeats,
            )
            replay = float(np.max(np.abs(
                X @ result["input_coefficients"]
                + result["intercept"]
                - result["predictions"]
            )))
            rows.append(row(
                "native_aom_ridge_global",
                n_samples,
                n_features,
                args.cv,
                args.moment_policy,
                elapsed,
                result,
                replay,
            ))
        if args.mode in {"wrapper", "both"}:
            elapsed, model = median_ms(
                lambda: run_wrapper(
                    X,
                    y,
                    folds,
                    args.cv,
                    lambdas,
                    args.moment_policy,
                ),
                args.repeats,
            )
            pred = np.asarray(model.predict(X), dtype=np.float64).reshape(-1, 1)
            replay = float(np.max(np.abs(pred - model.predictions_)))
            rows.append(row(
                "native_aom_ridge_global_sklearn",
                n_samples,
                n_features,
                args.cv,
                args.moment_policy,
                elapsed,
                model.result_,
                replay,
            ))

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
