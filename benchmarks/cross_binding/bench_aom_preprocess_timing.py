#!/usr/bin/env python3
"""Timing smoke benchmark for the native AOM preprocessing primitive.

The reusable primitive has a strict-linear production contract in
``nirs4all-methods``: direct single-operator identity, polynomial detrending,
Savitzky-Golay, Norris-Williams, finite-difference, Gaussian, Whittaker and FCK
preprocessing in hard/soft gating modes. Strict chains and model scoring are
covered by the AOM sweep and campaign helpers.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

import n4m


OPERATOR_SPECS = (
    {
        "label": "identity",
        "operator": "identity",
        "kind": 0,
        "checks_input_replay": True,
    },
    {
        "label": "detrend_poly_1",
        "operator": ("detrend_poly", [1]),
        "kind": 7,
        "checks_input_replay": False,
    },
    {
        "label": "savgol_smooth_5_2",
        "operator": ("savgol_smooth", [5, 2]),
        "kind": 8,
        "checks_input_replay": False,
    },
    {
        "label": "savgol_derivative_5_2_1",
        "operator": ("savgol_derivative", [5, 2, 1]),
        "kind": 9,
        "checks_input_replay": False,
    },
    {
        "label": "norris_williams_5_5_1",
        "operator": ("norris_williams", [5, 5, 1]),
        "kind": 10,
        "checks_input_replay": False,
    },
    {
        "label": "finite_difference_1",
        "operator": ("finite_difference", [1]),
        "kind": 15,
        "checks_input_replay": False,
    },
    {
        "label": "gaussian_1",
        "operator": ("gaussian", [1.0]),
        "kind": 18,
        "checks_input_replay": False,
    },
    {
        "label": "whittaker_100",
        "operator": ("whittaker", [100.0]),
        "kind": 16,
        "checks_input_replay": False,
    },
    {
        "label": "fck_1",
        "operator": ("fck", [1.0]),
        "kind": 17,
        "checks_input_replay": False,
    },
)


def make_dataset(n_samples: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = X[:, 0] - 0.2 * X[:, min(3, n_features - 1)]
    return X, y


def median_ms(fn, repeats: int) -> tuple[float, object]:
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(times)), result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/cross_binding/aom_preprocess_timing.csv",
        help="CSV output path",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    shapes = [(48, 24), (96, 64), (160, 128)]
    rows = []
    for shape_id, (n_samples, n_features) in enumerate(shapes):
        X, y = make_dataset(n_samples, n_features, seed=9300 + shape_id)
        for mode in ("soft", "hard"):
            for spec in OPERATOR_SPECS:
                elapsed, result = median_ms(
                    lambda spec=spec: n4m.aom_preprocess(
                        X,
                        y,
                        operators=[spec["operator"]],
                        gating_mode=mode,
                    ),
                    args.repeats,
                )

                transformed = np.asarray(result["transformed"], dtype=float)
                operator_outputs = np.asarray(result["operator_outputs"], dtype=float)
                if operator_outputs.size != transformed.size:
                    raise AssertionError(
                        "single-operator output size mismatch: "
                        f"{operator_outputs.shape} vs {transformed.shape}"
                    )
                operator_view = operator_outputs.reshape(transformed.shape)
                replay_error = float(np.max(np.abs(transformed - operator_view)))
                if replay_error > 1e-12:
                    raise AssertionError(
                        "aom_preprocess single-operator replay error too large: "
                        f"{spec['label']} {mode} {replay_error}"
                    )

                input_replay_error = float(np.max(np.abs(transformed - X)))
                if spec["checks_input_replay"] and input_replay_error > 1e-12:
                    raise AssertionError(
                        "aom_preprocess identity input replay error too large: "
                        f"{input_replay_error}"
                    )

                weights = np.asarray(result["weights"], dtype=float)
                weight_sum = float(np.sum(weights))
                if abs(weight_sum - 1.0) > 1e-12:
                    raise AssertionError(
                        "aom_preprocess single-operator weights should sum to 1, "
                        f"got {weight_sum}"
                    )
                operator_kinds = np.asarray(result["operator_kinds"], dtype=int).ravel()
                operator_kind = int(operator_kinds[0])
                if operator_kind != spec["kind"]:
                    raise AssertionError(
                        f"aom_preprocess {spec['label']} kind mismatch: "
                        f"{operator_kind} != {spec['kind']}"
                    )
                rows.append(
                    {
                        "method": "aom_preprocess",
                        "operator": spec["label"],
                        "gating_mode": mode,
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "n_operators": int(result["n_operators"]),
                        "mode_id": int(result["mode"]),
                        "operator_kind": operator_kind,
                        "weight_shape": "x".join(str(v) for v in weights.shape),
                        "weight_sum": weight_sum,
                        "replay_max_abs_error": replay_error,
                        "input_replay_max_abs_error": input_replay_error,
                        "elapsed_ms_median": elapsed,
                        "library_path": n4m.library_path(),
                        "abi": ".".join(str(v) for v in n4m.abi_version()),
                    }
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
