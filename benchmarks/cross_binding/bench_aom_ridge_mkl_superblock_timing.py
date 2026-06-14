#!/usr/bin/env python3
"""Timing smoke benchmark for aom_ridge_mkl_superblock."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

import n4m
from n4m.compose.aom_superblock import AOMRidgeMKLSuperblockRegressor
from n4m.compose.aom_superblock import aom_ridge_mkl_superblock


OPERATORS = (
    "identity",
    ("finite_difference", [1]),
    ("savgol_smooth", [5, 2]),
    ("savgol_derivative", [5, 2, 1]),
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


def run_function(X, y, folds, cv, alphas, mkl_top_k, block_scaling):
    return aom_ridge_mkl_superblock(
        X,
        y,
        operators=OPERATORS,
        alphas=alphas,
        cv=cv,
        fold_ids=folds,
        mkl_top_k=mkl_top_k,
        block_scaling=block_scaling,
    )


def run_wrapper(X, y, folds, cv, alphas, mkl_top_k, block_scaling):
    return AOMRidgeMKLSuperblockRegressor(
        operators=OPERATORS,
        alphas=alphas,
        cv=cv,
        fold_ids=folds,
        mkl_top_k=mkl_top_k,
        block_scaling=block_scaling,
    ).fit(X, y)


def row(backend, n_samples, n_features, cv, block_scaling, elapsed_ms, result, replay_error):
    scores = np.asarray(result["candidate_scores"], dtype=np.float64)
    weights = np.asarray(result["mkl_weights"], dtype=np.float64).reshape(-1)
    selected = np.asarray(result["selected_operator_indices"], dtype=np.int32).reshape(-1)
    return {
        "backend": backend,
        "n_samples": n_samples,
        "n_features": n_features,
        "cv": cv,
        "block_scaling": block_scaling,
        "n_operators": int(result["n_operators"]),
        "operator_kinds": " ".join(str(int(v)) for v in result["operator_kinds"]),
        "selected_operator_indices": " ".join(str(int(v)) for v in selected),
        "n_mkl_active_operators": int(result["n_mkl_active_operators"]),
        "mkl_weight_sum": float(np.sum(weights)),
        "mkl_top_k": int(result["mkl_top_k"]),
        "n_features_superblock": int(result["n_features_superblock"]),
        "n_candidates": int(result["n_candidates"]),
        "selected_alpha": result["selected_alpha"],
        "selected_cv_rmse": result["selected_cv_rmse"],
        "best_cv_rmse": float(np.min(scores[:, 2])),
        "mkl_mode": result["mkl_mode"],
        "ridge_backend": result["ridge_backend"],
        "prediction_replay_max_abs_error": replay_error,
        "elapsed_ms_median": elapsed_ms,
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/aom_ridge_mkl_superblock_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cv", type=int, default=4)
    parser.add_argument(
        "--mode",
        default="native",
        choices=("native", "wrapper", "both"),
        help="Benchmark ABI-close function, sklearn wrapper, or both",
    )
    parser.add_argument("--mkl-top-k", type=int, default=2)
    parser.add_argument(
        "--block-scaling",
        default="none",
        choices=("rms", "none"),
        help="Superblock scaling policy after KTA weighting",
    )
    args = parser.parse_args()

    shapes = [(32, 32), (64, 64), (96, 128)]
    alphas = np.asarray([0.01, 0.1], dtype=np.float64)
    rows = []
    for i, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=7820 + i)
        folds = balanced_folds(n_samples, args.cv)
        if args.mode in {"native", "both"}:
            elapsed, result = median_ms(
                lambda: run_function(
                    X,
                    y,
                    folds,
                    args.cv,
                    alphas,
                    args.mkl_top_k,
                    args.block_scaling,
                ),
                args.repeats,
            )
            replay = float(np.max(np.abs(
                X @ result["input_coefficients"]
                + result["intercept"]
                - result["predictions"]
            )))
            rows.append(row(
                "native_aom_ridge_mkl_superblock",
                n_samples,
                n_features,
                args.cv,
                args.block_scaling,
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
                    alphas,
                    args.mkl_top_k,
                    args.block_scaling,
                ),
                args.repeats,
            )
            pred = np.asarray(model.predict(X), dtype=np.float64).reshape(-1, 1)
            replay = float(np.max(np.abs(pred - model.predictions_)))
            rows.append(row(
                "native_aom_ridge_mkl_superblock_sklearn",
                n_samples,
                n_features,
                args.cv,
                args.block_scaling,
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
