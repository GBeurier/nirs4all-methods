#!/usr/bin/env python3
"""Timing smoke benchmark for NativeMomentStackRegressor."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

import n4m
from n4m.sklearn import NativeMomentStackRegressor


def make_dataset(n_samples: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (
        0.85 * X[:, 0]
        - 0.35 * X[:, min(4, n_features - 1)]
        + 0.15 * X[:, min(11, n_features - 1)]
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


def stack_specs():
    return (
        (
            "ridge_pcr_pls",
            {
                "base_models": ("ridge", "pcr", "pls"),
                "cv": 3,
                "inner_cv": 3,
                "n_components": 2,
                "scale_x": False,
            },
        ),
        (
            "full_moment_stack",
            {
                "cv": 3,
                "inner_cv": 3,
                "n_components": 2,
                "scale_x": False,
            },
        ),
    )


def row(label, n_samples, n_features, elapsed_ms, model):
    return {
        "label": label,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_base_models": len(model.base_model_names_),
        "base_models": "|".join(model.base_model_names_),
        "oof_rmse": float(model.oof_rmse_),
        "train_rmse": float(model.rmse_),
        "elapsed_ms_median": elapsed_ms,
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/moment_stack_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    shapes = [(48, 24), (96, 64)]
    rows = []
    for shape_id, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=9100 + shape_id)
        for label, kwargs in stack_specs():
            elapsed, model = median_ms(
                lambda kwargs=kwargs: NativeMomentStackRegressor(**kwargs).fit(X, y),
                args.repeats,
            )
            rows.append(row(label, n_samples, n_features, elapsed, model))

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
