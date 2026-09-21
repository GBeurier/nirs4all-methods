#!/usr/bin/env python3
"""Generic AOM calibration/stack smoke benchmark; emits data, no manuscript claims."""

import argparse
import json
import time

import numpy as np
from n4m.ensemble import LinearRidgeStackRegressor
from n4m.model_selection import AOMPLSRegressor, AOMRidgeRegressor, FastAOMPLSRegressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--features", type=int, default=100)
    args = parser.parse_args()
    rng = np.random.default_rng(17)
    X = rng.normal(size=(args.samples, args.features))
    y = X[:, 0] + rng.normal(size=len(X)) * 0.1
    V = rng.normal(size=(20, args.features))
    results = []
    for cls in (
        AOMPLSRegressor,
        AOMRidgeRegressor,
        FastAOMPLSRegressor,
        LinearRidgeStackRegressor,
    ):
        start = time.perf_counter()
        model = cls().fit(X, y)
        row = dict(method=cls.__name__, fit_s=time.perf_counter() - start)
        prediction = model.predict(V)
        if cls is LinearRidgeStackRegressor:
            export = model.export_affine()
            row.update(
                max_prediction_difference=float(
                    np.max(np.abs(prediction - model.predict_uncompressed(V)))
                ),
                numeric_bytes=export.numeric_bytes,
            )
        results.append(row)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
