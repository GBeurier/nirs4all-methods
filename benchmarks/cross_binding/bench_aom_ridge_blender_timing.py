#!/usr/bin/env python3
"""Timing smoke benchmark for n4m.aom_ridge_blender."""

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
    y = (
        0.70 * X[:, 0]
        - 0.35 * X[:, min(4, n_features - 1)]
        + 0.25 * X[:, min(13, n_features - 1)]
        + 0.05 * rng.standard_normal(n_samples)
    )
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


def run_blender(X, y, folds, cv, profile, lambdas, regularizer):
    return n4m.aom_ridge_blender(
        X,
        y,
        profile=profile,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=lambdas,
        regularizer=regularizer,
        scale_x=False,
    )


def row(profile, n_samples, n_features, cv, elapsed_ms, result):
    weights = np.asarray(result["weights"], dtype=np.float64).reshape(-1)
    scores = np.asarray(result["candidate_scores"], dtype=np.float64)
    return {
        "backend": "native_aom_ridge_blender",
        "profile": profile,
        "n_samples": n_samples,
        "n_features": n_features,
        "cv": cv,
        "n_chains": int(result["n_chains"]),
        "n_candidates": int(result["n_candidates"]),
        "n_nonzero_weights": int(np.count_nonzero(weights > 1e-8)),
        "regularizer": result["regularizer"],
        "selected_chain_id": int(result["selected_chain_id"]),
        "selected_param": result["selected_param"],
        "selected_cv_rmse": result["selected_cv_rmse"],
        "best_single_cv_rmse": float(np.min(scores[:, 3])),
        "blend_oof_rmse": result["blend_oof_rmse"],
        "elapsed_ms_median": elapsed_ms,
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/aom_ridge_blender_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--profile", default="compact", choices=("compact", "wide"))
    parser.add_argument("--regularizer", type=float, default=0.01)
    args = parser.parse_args()

    shapes = [(32, 64), (64, 128), (96, 256)]
    lambdas = np.asarray([1e-4, 1e-2, 1.0, 100.0], dtype=np.float64)
    rows = []
    for i, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=7200 + i)
        folds = balanced_folds(n_samples, args.cv)
        elapsed, result = median_ms(
            lambda: run_blender(
                X,
                y,
                folds,
                args.cv,
                args.profile,
                lambdas,
                args.regularizer,
            ),
            args.repeats,
        )
        rows.append(row(args.profile, n_samples, n_features, args.cv, elapsed, result))

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
