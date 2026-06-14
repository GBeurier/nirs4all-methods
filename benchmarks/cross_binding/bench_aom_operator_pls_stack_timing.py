#!/usr/bin/env python3
"""Timing smoke benchmark for n4m.aom_operator_pls_stack."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

import n4m
from n4m._impl import NativeAOMOperatorPLSStackRegressor


def make_dataset(n_samples: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (
        0.65 * X[:, min(1, n_features - 1)]
        - 0.30 * X[:, min(7, n_features - 1)]
        + 0.18 * X[:, min(19, n_features - 1)]
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


def run_stack(X, y, folds, cv, profile, components, alphas, std_penalty, gap_penalty):
    return n4m.aom_operator_pls_stack(
        X,
        y,
        profile=profile,
        cv=cv,
        fold_ids=folds,
        components=components,
        alphas=alphas,
        std_penalty=std_penalty,
        gap_penalty=gap_penalty,
        scale_x=False,
    )


def run_stack_wrapper(
    X, y, folds, cv, profile, components, alphas, std_penalty, gap_penalty
):
    return NativeAOMOperatorPLSStackRegressor(
        profile=profile,
        cv=cv,
        fold_ids=folds,
        components=components,
        alphas=alphas,
        std_penalty=std_penalty,
        gap_penalty=gap_penalty,
        scale_x=False,
    ).fit(X, y)


def result_from_model(model):
    return model.result_


def row(backend, profile, n_samples, n_features, cv, elapsed_ms, result, replay_error):
    scores = np.asarray(result["candidate_scores"], dtype=np.float64)
    return {
        "backend": backend,
        "profile": profile,
        "n_samples": n_samples,
        "n_features": n_features,
        "cv": cv,
        "n_operators": int(result["n_operators"]),
        "n_specs": int(result["n_specs"]),
        "n_operator_features": int(result["n_operator_features"]),
        "selected_components": int(result["selected_components"]),
        "selected_alpha": result["selected_alpha"],
        "selected_oof_rmse": result["selected_oof_rmse"],
        "best_single_criterion": float(np.min(scores[:, 6])),
        "n_pls_stack_cv_fits": int(result.get("n_pls_stack_cv_fits", 0)),
        "n_pls_stack_final_fits": int(result.get("n_pls_stack_final_fits", 0)),
        "n_ridge_stack_cv_fits": int(result.get("n_ridge_stack_cv_fits", 0)),
        "n_ridge_stack_final_fits": int(result.get("n_ridge_stack_final_fits", 0)),
        "n_operator_pls_stack_fit_calls": int(
            result.get("n_operator_pls_stack_fit_calls", 0)
        ),
        "n_operator_pls_stack_pls_fit_calls": int(
            result.get("n_operator_pls_stack_pls_fit_calls", 0)
        ),
        "n_operator_pls_stack_ridge_fit_calls": int(
            result.get("n_operator_pls_stack_ridge_fit_calls", 0)
        ),
        "prediction_replay_max_abs_error": replay_error,
        "elapsed_ms_median": elapsed_ms,
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/aom_operator_pls_stack_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cv", type=int, default=4)
    parser.add_argument("--profile", default="compact", choices=("compact", "wide"))
    parser.add_argument("--std-penalty", type=float, default=0.0)
    parser.add_argument("--gap-penalty", type=float, default=0.0)
    parser.add_argument(
        "--mode",
        default="native",
        choices=("native", "wrapper", "both"),
        help="Benchmark ABI-close function, sklearn wrapper, or both",
    )
    args = parser.parse_args()

    shapes = [(32, 64), (48, 96), (64, 128)]
    components = np.asarray([1, 2], dtype=np.int32)
    alphas = np.asarray([0.01, 1.0], dtype=np.float64)
    rows = []
    for i, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=8200 + i)
        folds = balanced_folds(n_samples, args.cv)
        if args.mode in {"native", "both"}:
            elapsed, result = median_ms(
                lambda: run_stack(
                    X,
                    y,
                    folds,
                    args.cv,
                    args.profile,
                    components,
                    alphas,
                    args.std_penalty,
                    args.gap_penalty,
                ),
                args.repeats,
            )
            rows.append(row(
                "native_aom_operator_pls_stack",
                args.profile,
                n_samples,
                n_features,
                args.cv,
                elapsed,
                result,
                0.0,
            ))
        if args.mode in {"wrapper", "both"}:
            elapsed, model = median_ms(
                lambda: run_stack_wrapper(
                    X,
                    y,
                    folds,
                    args.cv,
                    args.profile,
                    components,
                    alphas,
                    args.std_penalty,
                    args.gap_penalty,
                ),
                args.repeats,
            )
            pred = np.asarray(model.predict(X), dtype=np.float64).reshape(-1, 1)
            replay_error = float(np.max(np.abs(pred - model.predictions_)))
            rows.append(row(
                "native_aom_operator_pls_stack_sklearn",
                args.profile,
                n_samples,
                n_features,
                args.cv,
                elapsed,
                result_from_model(model),
                replay_error,
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
