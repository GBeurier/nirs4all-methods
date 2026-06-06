#!/usr/bin/env python3
"""Timing smoke for the PLS cross-validation reference ABI.

``n4m.pls_cross_validate`` is the stable public hook reserved for the future
grouped/fused PLS grinder. The current implementation delegates to the PLS
branch of ``n4m.sweep_run``; this benchmark keeps that reference surface
measurable and records route counters for CPU/CUDA smokes.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

import n4m


PLS_ROUTE_FIELDS = (
    "n_pls_moment_candidates",
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


def make_dataset(n_samples: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (
        0.85 * X[:, 0]
        - 0.30 * X[:, min(6, n_features - 1)]
        + 0.18 * X[:, min(19, n_features - 1)]
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


def parse_shape_list(text: str) -> tuple[tuple[int, int], ...]:
    shapes = []
    for part in text.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError("shapes must use NxP entries, e.g. 48x24,96x64")
        n_text, p_text = item.split("x", 1)
        n_samples = int(n_text)
        n_features = int(p_text)
        if n_samples < 4 or n_features < 1:
            raise ValueError("shape entries must be positive and n_samples >= 4")
        shapes.append((n_samples, n_features))
    if not shapes:
        raise ValueError("at least one shape must be provided")
    return tuple(shapes)


def parse_component_grid(text: str) -> tuple[int, ...]:
    grid = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not grid:
        raise ValueError("component-grid must contain at least one component")
    if any(value < 1 for value in grid):
        raise ValueError("component-grid values must be positive")
    return grid


def run_cross_validate(
    X,
    y,
    folds,
    components,
    cv,
    *,
    score_only,
    cuda_pls_parallel_folds,
    cuda_pls_min_device_features,
    cuda_pls_many_batched,
):
    return n4m.pls_cross_validate(
        X,
        y,
        cv=cv,
        fold_ids=folds,
        component_grid=components,
        scale_x=False,
        score_only=score_only,
        cuda_pls_parallel_folds=cuda_pls_parallel_folds,
        cuda_pls_min_device_features=cuda_pls_min_device_features,
        cuda_pls_many_batched=cuda_pls_many_batched,
    )


def run_sweep_reference(
    X,
    y,
    folds,
    components,
    cv,
    *,
    score_only,
    cuda_pls_parallel_folds,
    cuda_pls_min_device_features,
    cuda_pls_many_batched,
):
    return n4m.sweep_run(
        X,
        y,
        cv=cv,
        fold_ids=folds,
        ridge_lambdas=(),
        pls_components=components,
        heads=("pls",),
        scale_x=False,
        score_only=score_only,
        cuda_pls_parallel_folds=cuda_pls_parallel_folds,
        cuda_pls_min_device_features=cuda_pls_min_device_features,
        cuda_pls_many_batched=cuda_pls_many_batched,
    )


def max_abs_delta(left, right) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        return float("inf")
    if left_arr.size == 0:
        return 0.0
    return float(np.max(np.abs(left_arr - right_arr)))


def row(
    backend,
    n_samples,
    n_features,
    cv,
    components,
    elapsed_ms,
    result,
    reference,
    *,
    score_only,
    cuda_pls_parallel_folds,
    cuda_pls_min_device_features,
    cuda_pls_many_batched,
):
    out = {
        "backend": backend,
        "score_only": bool(score_only),
        "cuda_pls_parallel_folds": bool(cuda_pls_parallel_folds),
        "cuda_pls_min_device_features": (
            "" if cuda_pls_min_device_features is None else int(cuda_pls_min_device_features)
        ),
        "cuda_pls_many_batched": bool(cuda_pls_many_batched),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "cv": int(cv),
        "component_grid": "|".join(str(value) for value in components),
        "n_component_grid": int(len(components)),
        "selected_param": float(result["selected_param"]),
        "selected_cv_rmse": float(result["selected_cv_rmse"]),
        "candidate_score_max_abs_delta": max_abs_delta(
            result["candidate_scores"],
            reference["candidate_scores"],
        ),
        "oof_prediction_max_abs_delta": max_abs_delta(
            result.get("oof_predictions", np.empty((0, 0))),
            reference.get("oof_predictions", np.empty((0, 0))),
        ),
        "prediction_max_abs_delta": max_abs_delta(
            result.get("predictions", np.empty((0, 0))),
            reference.get("predictions", np.empty((0, 0))),
        ),
        "elapsed_ms_median": float(elapsed_ms),
        "library_path": n4m.library_path(),
        "abi": ".".join(str(value) for value in n4m.abi_version()),
    }
    for key in PLS_ROUTE_FIELDS:
        out[key] = int(float(result.get(key, 0.0)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/pls_cross_validate_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cv", type=int, default=4)
    parser.add_argument(
        "--shapes",
        default="48x24,96x64,160x128",
        help="Comma-separated synthetic shapes as NxP entries",
    )
    parser.add_argument("--component-grid", default="1,2,3")
    parser.add_argument("--cuda-pls-parallel-folds", action="store_true")
    parser.add_argument("--cuda-pls-min-device-features", type=int, default=None)
    parser.add_argument("--cuda-pls-many-batched", action="store_true")
    args = parser.parse_args()

    shapes = parse_shape_list(args.shapes)
    components = parse_component_grid(args.component_grid)
    rows = []
    for shape_id, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=9300 + shape_id)
        folds = balanced_folds(n_samples, args.cv)
        for score_only in (False, True):
            backend = (
                "pls_cross_validate_score_only"
                if score_only
                else "pls_cross_validate"
            )
            kwargs = {
                "score_only": bool(score_only),
                "cuda_pls_parallel_folds": bool(args.cuda_pls_parallel_folds),
                "cuda_pls_min_device_features": args.cuda_pls_min_device_features,
                "cuda_pls_many_batched": bool(args.cuda_pls_many_batched),
            }
            elapsed, result = median_ms(
                lambda kwargs=kwargs: run_cross_validate(
                    X, y, folds, components, args.cv, **kwargs
                ),
                args.repeats,
            )
            reference = run_sweep_reference(X, y, folds, components, args.cv, **kwargs)
            rows.append(
                row(
                    backend,
                    n_samples,
                    n_features,
                    args.cv,
                    components,
                    elapsed,
                    result,
                    reference,
                    **kwargs,
                )
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
