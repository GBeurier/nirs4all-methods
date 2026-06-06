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


def parse_base_models(text: str | None):
    if text is None:
        return None
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("base-models must contain at least one entry")
    return values


def row(label, n_samples, n_features, elapsed_ms, model):
    diagnostics = model.get_diagnostics()
    return {
        "label": label,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_base_models": len(model.base_model_names_),
        "base_models": "|".join(model.base_model_names_),
        "oof_rmse": float(model.oof_rmse_),
        "train_rmse": float(model.rmse_),
        "elapsed_ms_median": elapsed_ms,
        "n_base_oof_pls_moment_cv_fits": int(
            diagnostics["n_base_oof_pls_moment_cv_fits"]
        ),
        "n_base_oof_pls_moment_host_cv_fits": int(
            diagnostics["n_base_oof_pls_moment_host_cv_fits"]
        ),
        "n_base_oof_pls_moment_cuda_device_cv_fits": int(
            diagnostics["n_base_oof_pls_moment_cuda_device_cv_fits"]
        ),
        "n_base_oof_pls_moment_cuda_parallel_fold_batches": int(
            diagnostics["n_base_oof_pls_moment_cuda_parallel_fold_batches"]
        ),
        "n_base_oof_pls_moment_cuda_parallel_fold_jobs": int(
            diagnostics["n_base_oof_pls_moment_cuda_parallel_fold_jobs"]
        ),
        "n_base_final_pls_moment_cv_fits": int(
            diagnostics["n_base_final_pls_moment_cv_fits"]
        ),
        "n_base_final_pls_moment_host_cv_fits": int(
            diagnostics["n_base_final_pls_moment_host_cv_fits"]
        ),
        "n_base_final_pls_moment_cuda_device_cv_fits": int(
            diagnostics["n_base_final_pls_moment_cuda_device_cv_fits"]
        ),
        "n_base_final_pls_moment_cuda_parallel_fold_batches": int(
            diagnostics["n_base_final_pls_moment_cuda_parallel_fold_batches"]
        ),
        "n_base_final_pls_moment_cuda_parallel_fold_jobs": int(
            diagnostics["n_base_final_pls_moment_cuda_parallel_fold_jobs"]
        ),
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
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--inner-cv", type=int, default=3)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument(
        "--shapes",
        default="48x24,96x64",
        help="Comma-separated synthetic shapes as NxP entries",
    )
    parser.add_argument(
        "--base-models",
        default=None,
        help="Optional comma-separated base model list; when set, runs one custom row",
    )
    parser.add_argument("--cuda-pls-parallel-folds", action="store_true")
    parser.add_argument("--cuda-pls-min-device-features", type=int, default=None)
    parser.add_argument("--cuda-pls-many-batched", action="store_true")
    args = parser.parse_args()

    shapes = parse_shape_list(args.shapes)
    custom_base_models = parse_base_models(args.base_models)
    if custom_base_models is None:
        specs = stack_specs()
    else:
        specs = (
            (
                "custom_" + "_".join(custom_base_models),
                {
                    "base_models": custom_base_models,
                    "cv": args.cv,
                    "inner_cv": args.inner_cv,
                    "n_components": args.n_components,
                    "scale_x": False,
                },
            ),
        )
    rows = []
    for shape_id, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=9100 + shape_id)
        for label, kwargs in specs:
            run_kwargs = dict(kwargs)
            run_kwargs.setdefault("cv", args.cv)
            run_kwargs.setdefault("inner_cv", args.inner_cv)
            run_kwargs.setdefault("n_components", args.n_components)
            run_kwargs["cuda_pls_parallel_folds"] = bool(args.cuda_pls_parallel_folds)
            run_kwargs["cuda_pls_min_device_features"] = (
                args.cuda_pls_min_device_features
            )
            run_kwargs["cuda_pls_many_batched"] = bool(args.cuda_pls_many_batched)
            elapsed, model = median_ms(
                lambda kwargs=run_kwargs: NativeMomentStackRegressor(**kwargs).fit(
                    X, y
                ),
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
