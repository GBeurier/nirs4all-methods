#!/usr/bin/env python3
"""Timing smoke benchmark for n4m.aom_chain_ridge_pls."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

import n4m
from n4m.sklearn import NativeAOMChainRidgePLSRegressor


CHAINS = (
    (("identity", ()),),
    (("savgol_smooth", (5, 2)), ("finite_difference", (1,))),
)


def make_dataset(n_samples: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (
        0.70 * X[:, 0]
        - 0.22 * X[:, min(4, n_features - 1)]
        + 0.16 * X[:, min(12, n_features - 1)]
        + 0.04 * rng.standard_normal(n_samples)
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


def run_function(X, y, folds, cv, components, ridge_lambdas):
    return n4m.aom_chain_ridge_pls(
        X,
        y,
        chains=CHAINS,
        pls_components=components,
        ridge_lambdas=ridge_lambdas,
        cv=cv,
        fold_ids=folds,
    )


def run_wrapper(X, y, folds, cv, components, ridge_lambdas):
    return NativeAOMChainRidgePLSRegressor(
        chains=CHAINS,
        pls_components=components,
        ridge_lambdas=ridge_lambdas,
        cv=cv,
        fold_ids=folds,
    ).fit(X, y)


def row(backend, n_samples, n_features, cv, elapsed_ms, result, replay_error):
    scores = np.asarray(result["candidate_scores"], dtype=np.float64)
    return {
        "backend": backend,
        "n_samples": n_samples,
        "n_features": n_features,
        "cv": cv,
        "elapsed_ms": elapsed_ms,
        "n_chains": int(result["n_chains"]),
        "n_candidates": int(result["n_candidates"]),
        "selected_chain_id": int(result["selected_chain_id"]),
        "selected_n_components": int(result["n_components"]),
        "selected_ridge_lambda": float(result["ridge_lambda"]),
        "selected_cv_rmse": float(result["selected_cv_rmse"]),
        "min_candidate_rmse": float(np.min(scores[:, 4])),
        "replay_max_abs_error": replay_error,
        "prediction_replay_max_abs_error": replay_error,
        "selection_mode": result["selection_mode"],
        "ridge_pls_backend": result["ridge_pls_backend"],
        "n_ridge_pls_fit_calls": int(result.get("n_ridge_pls_fit_calls", 0)),
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/aom_chain_ridge_pls_timing.csv",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cv", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=("native", "wrapper", "both"),
        default="both",
    )
    args = parser.parse_args()

    shapes = [(32, 32), (64, 64), (96, 128)]
    components = np.asarray([1, 2], dtype=np.int32)
    ridge_lambdas = np.asarray([0.0, 0.1], dtype=np.float64)
    rows = []
    for i, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=7910 + i)
        folds = balanced_folds(n_samples, args.cv)
        if args.mode in {"native", "both"}:
            elapsed, result = median_ms(
                lambda: run_function(
                    X,
                    y,
                    folds,
                    args.cv,
                    components,
                    ridge_lambdas,
                ),
                args.repeats,
            )
            replay = float(np.max(np.abs(
                X @ result["input_coefficients"]
                + result["intercept"]
                - result["predictions"]
            )))
            rows.append(row(
                "native_aom_chain_ridge_pls",
                n_samples,
                n_features,
                args.cv,
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
                    components,
                    ridge_lambdas,
                ),
                args.repeats,
            )
            pred = np.asarray(model.predict(X), dtype=np.float64).reshape(-1, 1)
            replay = float(np.max(np.abs(pred - model.predictions_)))
            rows.append(row(
                "native_aom_chain_ridge_pls_sklearn",
                n_samples,
                n_features,
                args.cv,
                elapsed,
                model.result_,
                replay,
            ))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
