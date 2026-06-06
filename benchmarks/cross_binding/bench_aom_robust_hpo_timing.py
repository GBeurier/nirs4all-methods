#!/usr/bin/env python3
"""Timing smoke benchmark for AOM robust-HPO screens.

The native backend exercises ``n4m_aom_robust_hpo_fit`` through
``n4m.aom_robust_hpo``. The sklearn backend exercises the reusable native
``NativeAOMRobustHPORegressor`` wrapper and checks that it replays the native
fitted predictions. This is a small timing bench, not an oracle-quality accuracy
campaign.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

import n4m
from n4m import NativeAOMRobustHPORegressor


def make_dataset(n_samples: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (
        1.2 * X[:, min(3, n_features - 1)]
        - 0.45 * X[:, min(17, n_features - 1)]
        + 0.15 * np.sin(X[:, min(5, n_features - 1)])
        + 0.03 * rng.standard_normal(n_samples)
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


def run_native(X, y, profile: str, cv: int, repeats: int):
    return median_ms(
        lambda: n4m.aom_robust_hpo(
            X, y, profile=profile, cv=cv, heads=("ridge", "pls")
        ),
        repeats,
    )


def run_sklearn(X, y, profile: str, cv: int, repeats: int):
    def fit_once():
        model = NativeAOMRobustHPORegressor(
            profile=profile,
            cv=cv,
            heads=("ridge", "pls"),
        )
        model.fit(X, y)
        return model

    return median_ms(fit_once, repeats)


def native_row(profile, n_samples, n_features, cv, elapsed_ms, result):
    return {
        "backend": "native_abi",
        "profile": profile,
        "n_samples": n_samples,
        "n_features": n_features,
        "cv": cv,
        "n_candidates": int(result["n_candidates"]),
        "selected_chain_id": int(result["selected_chain_id"]),
        "selected_head_id": int(result["selected_head_id"]),
        "selected_param": result["selected_param"],
        "selected_cv_rmse": result["selected_cv_rmse"],
        "prediction_replay_max_abs_error": "",
        "elapsed_ms_median": elapsed_ms,
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }


def sklearn_row(profile, n_samples, n_features, cv, elapsed_ms, X, model):
    expected = np.asarray(model.result_["predictions"], dtype=np.float64)
    pred = np.asarray(model.predict(X), dtype=np.float64)
    if pred.ndim == 1 and expected.ndim == 2 and expected.shape[1] == 1:
        expected = expected.ravel()
    replay_error = float(np.max(np.abs(pred - expected)))
    diagnostics = model.get_diagnostics()
    return {
        "backend": "native_sklearn",
        "profile": profile,
        "n_samples": n_samples,
        "n_features": n_features,
        "cv": cv,
        "n_candidates": int(diagnostics["n_candidates"]),
        "selected_chain_id": int(diagnostics["selected_chain_id"]),
        "selected_head_id": int(model.selected_head_id_),
        "selected_param": float(diagnostics["selected_param"]),
        "selected_cv_rmse": float(diagnostics["selected_cv_rmse"]),
        "prediction_replay_max_abs_error": replay_error,
        "elapsed_ms_median": elapsed_ms,
        "library_path": n4m.library_path(),
        "abi": ".".join(str(v) for v in n4m.abi_version()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/aom_robust_hpo_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--include-wide", action="store_true")
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="Only time the native C ABI path; useful for alternate lib builds.",
    )
    args = parser.parse_args()

    shapes = [(32, 64), (64, 128), (96, 256)]
    profiles = ["compact"] + (["wide"] if args.include_wide else [])
    rows = []
    for profile in profiles:
        for i, (n_samples, n_features) in enumerate(shapes):
            X, y = make_dataset(n_samples, n_features, seed=2026 + i)
            elapsed, result = run_native(X, y, profile, args.cv, args.repeats)
            rows.append(
                native_row(profile, n_samples, n_features, args.cv, elapsed, result)
            )
            if args.native_only:
                continue
            elapsed, model = run_sklearn(X, y, profile, args.cv, args.repeats)
            rows.append(
                sklearn_row(
                    profile,
                    n_samples,
                    n_features,
                    args.cv,
                    elapsed,
                    X,
                    model,
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
