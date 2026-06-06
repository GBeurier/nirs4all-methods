#!/usr/bin/env python3
"""Timing smoke benchmark for the native moment Ridge sweep.

This compares ``n4m.sweep_run`` against the intentionally simple materialized
baseline that fits ``n4m.ridge`` for every (fold, lambda) pair and scores held
out rows in Python. It is a product smoke timing, not an oracle campaign.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

import n4m


def make_dataset(n_samples: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    signal = (
        1.15 * X[:, 0]
        - 0.55 * X[:, min(7, n_features - 1)]
        + 0.25 * X[:, min(23, n_features - 1)]
    )
    y = signal + 0.05 * rng.standard_normal(n_samples)
    return X, y


def balanced_folds(n_samples: int, cv: int) -> np.ndarray:
    return np.arange(n_samples, dtype=np.int32) % int(cv)


def median_ms(fn, repeats: int) -> tuple[float, object]:
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(times)), result


def rmse(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_hat).reshape(-1) - y.reshape(-1)) ** 2)))


def run_sweep_ridge(X, y, folds, lambdas, cv, *, score_only=False):
    return n4m.sweep_run(
        X,
        y,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        scale_x=False,
        score_only=score_only,
    )


def run_sweep_pls(
    X,
    y,
    folds,
    components,
    cv,
    *,
    score_only=False,
    cuda_pls_parallel_folds=False,
    cuda_pls_min_device_features=None,
    cuda_pls_many_batched=False,
):
    return n4m.sweep_run(
        X,
        y,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=[0.1],
        pls_components=components,
        heads=("pls",),
        scale_x=False,
        score_only=score_only,
        cuda_pls_parallel_folds=cuda_pls_parallel_folds,
        cuda_pls_min_device_features=cuda_pls_min_device_features,
        cuda_pls_many_batched=cuda_pls_many_batched,
    )


def run_materialized_cv(X, y, folds, lambdas, cv):
    candidate_scores = []
    best_score = float("inf")
    best_lambda = None
    best_oof = None

    for lambda_id, alpha in enumerate(lambdas):
        oof = np.empty((X.shape[0], 1), dtype=np.float64)
        for fold in range(cv):
            valid = folds == fold
            train = ~valid
            fit = n4m.ridge(X[train], y[train], alpha=float(alpha), scale_x=False)
            beta = fit["coefficients"]
            intercept = fit["intercept"]
            oof[valid] = X[valid] @ beta + intercept
        score = rmse(oof[:, 0], y)
        candidate_scores.append((lambda_id, float(alpha), score))
        if score < best_score:
            best_score = score
            best_lambda = float(alpha)
            best_oof = oof

    final = n4m.ridge(X, y, alpha=float(best_lambda), scale_x=False)
    return {
        "candidate_scores": candidate_scores,
        "oof_predictions": best_oof,
        "predictions": final["predictions"],
        "selected_cv_rmse": best_score,
        "selected_param": best_lambda,
    }


def row(
    backend,
    n_samples,
    n_features,
    cv,
    lambdas,
    elapsed_ms,
    result,
    *,
    cuda_pls_parallel_folds=False,
    cuda_pls_min_device_features=None,
    cuda_pls_many_batched=False,
):
    return {
        "backend": backend,
        "cuda_pls_parallel_folds": bool(cuda_pls_parallel_folds),
        "cuda_pls_min_device_features": cuda_pls_min_device_features,
        "cuda_pls_many_batched": bool(cuda_pls_many_batched),
        "n_samples": n_samples,
        "n_features": n_features,
        "cv": cv,
        "n_candidates": len(lambdas),
        "n_ridge_moment_candidates": int(
            result.get("n_ridge_moment_candidates", 0.0)
        ),
        "n_ridge_dual_materialized_candidates": int(
            result.get("n_ridge_dual_materialized_candidates", 0.0)
        ),
        "n_ridge_moment_cv_fits": int(
            result.get("n_ridge_moment_cv_fits", 0.0)
        ),
        "n_ridge_moment_eigen_path_preparations": int(
            result.get("n_ridge_moment_eigen_path_preparations", 0.0)
        ),
        "n_ridge_moment_eigen_path_cv_fits": int(
            result.get("n_ridge_moment_eigen_path_cv_fits", 0.0)
        ),
        "n_ridge_moment_direct_cv_fits": int(
            result.get("n_ridge_moment_direct_cv_fits", 0.0)
        ),
        "n_ridge_dual_materialized_cv_fits": int(
            result.get("n_ridge_dual_materialized_cv_fits", 0.0)
        ),
        "n_ridge_dual_cross_cv_fits": int(
            result.get("n_ridge_dual_cross_cv_fits", 0.0)
        ),
        "n_ridge_moment_score_batch_calls": int(
            result.get("n_ridge_moment_score_batch_calls", 0.0)
        ),
        "n_ridge_moment_score_batch_jobs": int(
            result.get("n_ridge_moment_score_batch_jobs", 0.0)
        ),
        "n_ridge_moment_final_fits": int(
            result.get("n_ridge_moment_final_fits", 0.0)
        ),
        "n_ridge_dual_materialized_final_fits": int(
            result.get("n_ridge_dual_materialized_final_fits", 0.0)
        ),
        "n_pls_moment_candidates": int(
            result.get("n_pls_moment_candidates", 0.0)
        ),
        "n_pls_moment_cv_fits": int(
            result.get("n_pls_moment_cv_fits", 0.0)
        ),
        "n_pls_moment_host_cv_fits": int(
            result.get("n_pls_moment_host_cv_fits", 0.0)
        ),
        "n_pls_moment_cuda_device_cv_fits": int(
            result.get("n_pls_moment_cuda_device_cv_fits", 0.0)
        ),
        "n_pls_moment_cuda_parallel_fold_batches": int(
            result.get("n_pls_moment_cuda_parallel_fold_batches", 0.0)
        ),
        "n_pls_moment_cuda_parallel_fold_jobs": int(
            result.get("n_pls_moment_cuda_parallel_fold_jobs", 0.0)
        ),
        "n_pls_moment_cuda_many_batched_batches": int(
            result.get("n_pls_moment_cuda_many_batched_batches", 0.0)
        ),
        "n_pls_moment_cuda_many_batched_jobs": int(
            result.get("n_pls_moment_cuda_many_batched_jobs", 0.0)
        ),
        "n_pls_moment_score_batch_calls": int(
            result.get("n_pls_moment_score_batch_calls", 0.0)
        ),
        "n_pls_moment_score_batch_jobs": int(
            result.get("n_pls_moment_score_batch_jobs", 0.0)
        ),
        "n_pls_materialized_cv_fits": int(
            result.get("n_pls_materialized_cv_fits", 0.0)
        ),
        "n_pls_moment_final_fits": int(
            result.get("n_pls_moment_final_fits", 0.0)
        ),
        "n_pls_moment_host_final_fits": int(
            result.get("n_pls_moment_host_final_fits", 0.0)
        ),
        "n_pls_moment_cuda_device_final_fits": int(
            result.get("n_pls_moment_cuda_device_final_fits", 0.0)
        ),
        "n_pls_materialized_final_fits": int(
            result.get("n_pls_materialized_final_fits", 0.0)
        ),
        "selected_param": result["selected_param"],
        "selected_cv_rmse": result["selected_cv_rmse"],
        "elapsed_ms_median": elapsed_ms,
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/moment_sweep_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="Only time n4m.sweep_run; useful for alternate CUDA builds.",
    )
    parser.add_argument(
        "--cuda-pls-parallel-folds",
        action="store_true",
        help="Enable bounded stream-parallel CUDA scheduling for exact PLS1 moment fold jobs",
    )
    parser.add_argument(
        "--cuda-pls-min-device-features",
        type=int,
        default=None,
        help="Override the PLS1 moment CUDA device-route feature threshold",
    )
    parser.add_argument(
        "--cuda-pls-many-batched",
        action="store_true",
        help="Enable the experimental tiled/strided-batched CUDA route for many PLS1 moment jobs",
    )
    args = parser.parse_args()

    # Include one tall cell so Ridge uses the exact moment path instead of the
    # wide dual/materialized route; this keeps the lambda-path scorer covered.
    shapes = [(64, 64), (128, 128), (192, 256), (96, 48)]
    lambdas = np.asarray([0.001, 0.01, 0.1, 1.0, 10.0], dtype=np.float64)
    components = np.asarray([1, 2, 4], dtype=np.int32)
    rows = []
    for i, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=4300 + i)
        folds = balanced_folds(n_samples, args.cv)

        elapsed, result = median_ms(
            lambda: run_sweep_ridge(X, y, folds, lambdas, args.cv),
            args.repeats,
        )
        rows.append(
            row(
                "native_sweep_ridge",
                n_samples,
                n_features,
                args.cv,
                lambdas,
                elapsed,
                result,
                cuda_pls_parallel_folds=args.cuda_pls_parallel_folds,
                cuda_pls_min_device_features=args.cuda_pls_min_device_features,
                cuda_pls_many_batched=args.cuda_pls_many_batched,
            )
        )

        elapsed, result = median_ms(
            lambda: run_sweep_ridge(
                X, y, folds, lambdas, args.cv, score_only=True
            ),
            args.repeats,
        )
        rows.append(
            row(
                "native_sweep_ridge_score_only",
                n_samples,
                n_features,
                args.cv,
                lambdas,
                elapsed,
                result,
                cuda_pls_parallel_folds=args.cuda_pls_parallel_folds,
                cuda_pls_min_device_features=args.cuda_pls_min_device_features,
                cuda_pls_many_batched=args.cuda_pls_many_batched,
            )
        )

        elapsed, result = median_ms(
            lambda: run_sweep_pls(
                X,
                y,
                folds,
                components,
                args.cv,
                cuda_pls_parallel_folds=args.cuda_pls_parallel_folds,
                cuda_pls_min_device_features=args.cuda_pls_min_device_features,
                cuda_pls_many_batched=args.cuda_pls_many_batched,
            ),
            args.repeats,
        )
        rows.append(
            row(
                "native_sweep_pls",
                n_samples,
                n_features,
                args.cv,
                components,
                elapsed,
                result,
                cuda_pls_parallel_folds=args.cuda_pls_parallel_folds,
                cuda_pls_min_device_features=args.cuda_pls_min_device_features,
                cuda_pls_many_batched=args.cuda_pls_many_batched,
            )
        )

        elapsed, result = median_ms(
            lambda: run_sweep_pls(
                X,
                y,
                folds,
                components,
                args.cv,
                score_only=True,
                cuda_pls_parallel_folds=args.cuda_pls_parallel_folds,
                cuda_pls_min_device_features=args.cuda_pls_min_device_features,
                cuda_pls_many_batched=args.cuda_pls_many_batched,
            ),
            args.repeats,
        )
        rows.append(
            row(
                "native_sweep_pls_score_only",
                n_samples,
                n_features,
                args.cv,
                components,
                elapsed,
                result,
                cuda_pls_parallel_folds=args.cuda_pls_parallel_folds,
                cuda_pls_min_device_features=args.cuda_pls_min_device_features,
                cuda_pls_many_batched=args.cuda_pls_many_batched,
            )
        )

        if args.native_only:
            continue
        elapsed, result = median_ms(
            lambda: run_materialized_cv(X, y, folds, lambdas, args.cv),
            args.repeats,
        )
        rows.append(
            row(
                "materialized_cv",
                n_samples,
                n_features,
                args.cv,
                lambdas,
                elapsed,
                result,
                cuda_pls_parallel_folds=args.cuda_pls_parallel_folds,
                cuda_pls_min_device_features=args.cuda_pls_min_device_features,
                cuda_pls_many_batched=args.cuda_pls_many_batched,
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
