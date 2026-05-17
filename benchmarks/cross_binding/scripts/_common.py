"""Helpers shared by all Python benchmark scripts."""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    """Return (X, y, n_components, n_runs)."""
    if len(sys.argv) != 4:
        print("usage: SCRIPT csv_path n_components n_runs", file=sys.stderr)
        sys.exit(2)
    csv = Path(sys.argv[1])
    nc = int(sys.argv[2])
    runs = int(sys.argv[3])
    arr = np.loadtxt(csv, delimiter=",", skiprows=1, dtype=np.float64)
    X = np.ascontiguousarray(arr[:, :-1])
    y = arr[:, -1].astype(np.float64)
    return X, y, nc, runs


def time_runs(fit_predict, n_runs: int) -> dict:
    """Run `fit_predict()` n_runs times, return a stats dict.

    `fit_predict` should perform one fit (+ optionally one in-sample
    predict) and return nothing. The first run is treated as a warmup
    and excluded from the timing when n_runs >= 3."""
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fit_predict()
        samples.append((time.perf_counter() - t0) * 1000.0)
    if n_runs >= 3:
        # discard the warmup
        timed = samples[1:]
    else:
        timed = samples
    return {
        "ok": True,
        "n_runs": len(timed),
        "median_ms": statistics.median(timed),
        "min_ms": min(timed),
        "max_ms": max(timed),
    }


def emit(rec: dict) -> None:
    print(json.dumps(rec))
