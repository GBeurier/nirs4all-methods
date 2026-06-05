#!/usr/bin/env python3
"""Timing smoke benchmark for reusable native AOM-PLS and POP-PLS selectors.

This measures the ABI-close functions and sklearn-style wrappers added for the
historical AOM/POP selectors. It also verifies that the exposed input-space
linear state replays native predictions.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

import n4m
from n4m.sklearn import NativeAOMPLSRegressor, NativePOPPLSRegressor


def make_dataset(n_samples: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (
        0.75 * X[:, 0]
        - 0.35 * X[:, min(4, n_features - 1)]
        + 0.20 * X[:, min(15, n_features - 1)]
        + 0.04 * rng.standard_normal(n_samples)
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


def replay_error(X: np.ndarray, result: dict) -> float:
    replay = X @ result["input_coefficients"] + result["intercept"]
    return float(np.max(np.abs(replay - result["predictions"])))


def row(method, surface, n_samples, n_features, cv, elapsed_ms, result, replay_max_abs):
    base = {
        "method": method,
        "surface": surface,
        "n_samples": n_samples,
        "n_features": n_features,
        "cv": cv,
        "n_operators": int(result["n_operators"]),
        "max_components": int(result["max_components"]),
        "selected_n_components": int(result["selected_n_components"]),
        "best_score": float(result["best_score"]),
        "replay_max_abs": replay_max_abs,
        "elapsed_ms_median": elapsed_ms,
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }
    if "selected_operator_index" in result:
        base["selected_operator_index"] = int(result["selected_operator_index"])
    else:
        selected = np.asarray(result["selected_operator_indices"], dtype=np.int32)
        base["selected_operator_index"] = int(selected[0]) if selected.size else -1
    return base


def run_function(fn, X, y, folds, cv, max_components):
    return fn(
        X,
        y,
        max_components=max_components,
        cv=cv,
        fold_ids=folds,
        scale_x=False,
    )


def run_estimator(cls, X, y, folds, cv, max_components):
    model = cls(
        max_components=max_components,
        cv=cv,
        fold_ids=folds,
        scale_x=False,
    ).fit(X, y)
    predictions = np.asarray(model.predict(X), dtype=np.float64).reshape(-1, 1)
    np.testing.assert_allclose(
        predictions,
        model.result_["predictions"],
        rtol=1e-10,
        atol=1e-10,
    )
    return model.result_


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/aom_selector_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cv", type=int, default=4)
    parser.add_argument("--max-components", type=int, default=2)
    args = parser.parse_args()

    shapes = [(24, 32), (36, 64), (48, 96)]
    runners = [
        (
            "aom_pls",
            "function",
            lambda X, y, f: run_function(
                n4m.aom_pls, X, y, f, args.cv, args.max_components
            ),
        ),
        (
            "pop_pls",
            "function",
            lambda X, y, f: run_function(
                n4m.pop_pls, X, y, f, args.cv, args.max_components
            ),
        ),
        (
            "aom_pls",
            "sklearn",
            lambda X, y, f: run_estimator(
                NativeAOMPLSRegressor, X, y, f, args.cv, args.max_components
            ),
        ),
        (
            "pop_pls",
            "sklearn",
            lambda X, y, f: run_estimator(
                NativePOPPLSRegressor, X, y, f, args.cv, args.max_components
            ),
        ),
    ]

    rows = []
    for i, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=9100 + i)
        folds = balanced_folds(n_samples, args.cv)
        for method, surface, runner in runners:
            elapsed, result = median_ms(lambda: runner(X, y, folds), args.repeats)
            err = replay_error(X, result)
            if err > 1e-9:
                raise AssertionError(f"{method}/{surface} replay error too large: {err}")
            rows.append(
                row(
                    method,
                    surface,
                    n_samples,
                    n_features,
                    args.cv,
                    elapsed,
                    result,
                    err,
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
