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


def row(profile, n_samples, n_features, cv, elapsed_ms, result):
    scores = np.asarray(result["candidate_scores"], dtype=np.float64)
    return {
        "backend": "native_aom_operator_pls_stack",
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
    args = parser.parse_args()

    shapes = [(32, 64), (48, 96), (64, 128)]
    components = np.asarray([1, 2], dtype=np.int32)
    alphas = np.asarray([0.01, 1.0], dtype=np.float64)
    rows = []
    for i, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=8200 + i)
        folds = balanced_folds(n_samples, args.cv)
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
