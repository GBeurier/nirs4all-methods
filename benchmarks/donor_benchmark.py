#!/usr/bin/env python3
"""Donor-side benchmark — runs the 87 donor methods via libn4m + Python ctypes.

Complement to benchmarks/cross_binding/orchestrator.py which is PLS-centric
and only knows the 73 pls4all methods. This script handles the
preprocessing / augmentation / splitters / filters / utilities catalogued
in `catalog/methods.yaml` whose backend is libn4m (built by
cpp/src/n4m_targets.cmake).

Output: docs/_static/donor-bench-data.csv with one row per (method,
binding, dataset_size, threads). The unified bench-data.json will
merge this with the pls4all orchestrator's output in a later M8
session.

Usage:
    # Build libn4m first:
    cmake --build --preset dev-release -j 4

    # Then:
    N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.0.0 \\
        python3 benchmarks/donor_benchmark.py --sizes 30x30 100x50 \\
                                              --out-csv /tmp/donor.csv

Requires:
    - libn4m.so built (cpp/src/n4m_targets.cmake target)
    - bindings/python_n4m/src/n4m/ in sys.path
"""

from __future__ import annotations
import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Wire the n4m Python binding so n4m.* imports resolve
sys.path.insert(0, str(REPO / "bindings" / "python_n4m" / "src"))

# Default libn4m path if N4M_LIB_PATH isn't set
_default_lib = REPO / "build" / "dev-release" / "cpp" / "src" / "libn4m.so.1.0.0"
if "N4M_LIB_PATH" not in os.environ and _default_lib.exists():
    os.environ["N4M_LIB_PATH"] = str(_default_lib)


def _seeded_xy(n_samples: int, n_features: int, seed: int = 1234567890):
    """Deterministic synthetic spectroscopic dataset."""
    import numpy as np
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_samples, n_features))
    # Add some baseline drift and noise to make preprocessing meaningful
    wavelength = np.linspace(0, 1, n_features)
    drift = 5 * np.sin(2 * np.pi * wavelength) + 0.1 * wavelength**2
    base = base + drift[None, :]
    # Per-sample scale
    scale = 1 + 0.3 * rng.standard_normal((n_samples, 1))
    base = base * scale
    y = rng.standard_normal((n_samples,))
    return base, y


def _measure(callable_, n_runs: int = 5) -> dict:
    """Time `callable_()` n_runs times and return percentile statistics."""
    timings: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            callable_()
            ok = True
            err = ""
        except Exception as e:
            ok = False
            err = type(e).__name__ + ": " + str(e)[:100]
            break
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)  # ms
    if not timings:
        return {"ok": False, "err": err, "median_ms": float("nan"),
                "min_ms": float("nan"), "max_ms": float("nan"), "n_runs": 0}
    return {
        "ok": True,
        "err": "",
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "n_runs": len(timings),
    }


# ----- Method catalogue --------------------------------------------------
# Each entry: (method_id, category, fn_name, kwargs_factory, needs_xy)
#   fn_name is the donor Python binding function name (n4m.python.<fn_name>)
#   kwargs_factory(n_samples, n_features) → dict of kwargs to pass
#   needs_xy=True for splitters that need both X and y
DONOR_METHODS: list[tuple] = [
    # ----- preprocessing.scatter
    ("preprocessing.scatter.snv",                "preprocessing", "snv",                  lambda n, p: {}, False),
    ("preprocessing.scatter.msc",                "preprocessing", "msc",                  lambda n, p: {}, False),
    ("preprocessing.scatter.emsc",               "preprocessing", "emsc",                 lambda n, p: {}, False),
    ("preprocessing.scatter.robust_snv",         "preprocessing", "rnv",                  lambda n, p: {}, False),
    ("preprocessing.scatter.local_snv",          "preprocessing", "lsnv",                 lambda n, p: {}, False),
    ("preprocessing.scatter.area_normalization", "preprocessing", "area_normalization",   lambda n, p: {}, False),
    # ----- preprocessing.derivatives
    ("preprocessing.derivatives.savitzky_golay", "preprocessing", "savitzky_golay",       lambda n, p: {"window_length": 5, "polyorder": 2}, False),
    ("preprocessing.derivatives.first_derivative", "preprocessing", "first_derivative",   lambda n, p: {}, False),
    ("preprocessing.derivatives.second_derivative","preprocessing", "second_derivative",  lambda n, p: {}, False),
    ("preprocessing.derivatives.norris_williams",  "preprocessing", "norris_williams",    lambda n, p: {"segment": 5, "gap": 5, "derivative_order": 1}, False),
    # ----- preprocessing.baselines
    ("preprocessing.baselines.detrend",          "preprocessing", "detrend",              lambda n, p: {"polyorder": 1}, False),
    ("preprocessing.baselines.asls",             "preprocessing", "asls",                 lambda n, p: {"lam": 1e5, "p": 0.01}, False),
    ("preprocessing.baselines.airpls",           "preprocessing", "airpls",               lambda n, p: {"lam": 1e5, "max_iter": 15}, False),
    ("preprocessing.baselines.arpls",            "preprocessing", "arpls",                lambda n, p: {"lam": 1e5, "max_iter": 50}, False),
    # ----- preprocessing.smoothing
    ("preprocessing.smoothing.gaussian",         "preprocessing", "gaussian",             lambda n, p: {"sigma": 2.0}, False),
    # ----- preprocessing.scaling
    ("preprocessing.scaling.normalize",          "preprocessing", "normalize",            lambda n, p: {}, False),
    ("preprocessing.scaling.simple_scale",       "preprocessing", "simple_scale",         lambda n, p: {}, False),
    # ----- preprocessing.signal_conversion
    ("preprocessing.signal_conversion.to_absorbance",        "preprocessing", "to_absorbance",        lambda n, p: {}, False),
    ("preprocessing.signal_conversion.from_absorbance",      "preprocessing", "from_absorbance",      lambda n, p: {}, False),
    ("preprocessing.signal_conversion.kubelka_munk",         "preprocessing", "kubelka_munk",         lambda n, p: {}, False),
    # ----- preprocessing.wavelets
    ("preprocessing.wavelets.haar",              "preprocessing", "haar",                 lambda n, p: {}, False),
    # ----- preprocessing.orthogonalization
    ("preprocessing.orthogonalization.osc",      "preprocessing", "osc",                  lambda n, p: {"n_components": 1}, True),
    ("preprocessing.orthogonalization.epo",      "preprocessing", "epo",                  lambda n, p: {}, True),
    # ----- splitters
    ("splitters.kennard_stone",                  "splitters",     "kennard_stone",        lambda n, p: {"test_size": 0.25}, False),
    ("splitters.spxy",                           "splitters",     "spxy",                 lambda n, p: {"test_size": 0.25}, True),
    ("splitters.kbins_stratified",               "splitters",     "kbins_stratified",     lambda n, p: {"n_bins": 5, "test_size": 0.25}, "y_only"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", nargs="+", default=["30x30", "100x50"],
                    help="Dataset sizes (NxP)")
    ap.add_argument("--n-runs", type=int, default=5, help="Timed runs per method")
    ap.add_argument("--out-csv", default=str(REPO / "docs/_static/donor-bench-data.csv"),
                    help="CSV output path")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="Subset of method IDs to run (default: all)")
    args = ap.parse_args(argv)

    # Late import so missing libn4m fails with a clear message.
    try:
        from n4m import python as n4m_py
    except Exception as e:
        print(f"FAIL: could not load donor Python binding: {e}")
        print(f"      N4M_LIB_PATH={os.environ.get('N4M_LIB_PATH', '<unset>')}")
        return 1

    methods = DONOR_METHODS
    if args.methods:
        wanted = set(args.methods)
        methods = [m for m in DONOR_METHODS if m[0] in wanted]
        if not methods:
            print(f"No methods matched: {args.methods}")
            return 1

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method_id", "category", "n_samples", "n_features",
            "ok", "median_ms", "min_ms", "max_ms", "n_runs", "err",
        ])

        for size in args.sizes:
            try:
                n, p = (int(x) for x in size.lower().split("x"))
            except ValueError:
                print(f"Skipping invalid size {size!r}")
                continue
            X, y = _seeded_xy(n, p)

            print(f"\n=== Dataset {n}×{p} ===")
            for method_id, category, fn_name, kwargs_factory, needs_xy in methods:
                fn = getattr(n4m_py, fn_name, None)
                if fn is None:
                    print(f"  {method_id:50s} ❌ no n4m.python.{fn_name}")
                    w.writerow([method_id, category, n, p, False, "", "", "", 0,
                                f"missing binding fn {fn_name}"])
                    continue
                kwargs = kwargs_factory(n, p)
                if needs_xy == "y_only":
                    call = lambda f=fn, y=y, k=kwargs: f(y, **k)
                elif needs_xy:
                    call = lambda f=fn, X=X, y=y, k=kwargs: f(X, y, **k)
                else:
                    call = lambda f=fn, X=X, k=kwargs: f(X, **k)
                stats = _measure(call, n_runs=args.n_runs)
                if stats["ok"]:
                    print(f"  {method_id:50s} ✓  median={stats['median_ms']:7.3f}ms"
                          f"  range=[{stats['min_ms']:.3f}, {stats['max_ms']:.3f}]")
                else:
                    print(f"  {method_id:50s} ❌ {stats['err']}")
                w.writerow([
                    method_id, category, n, p,
                    stats["ok"], stats.get("median_ms"), stats.get("min_ms"),
                    stats.get("max_ms"), stats.get("n_runs"), stats.get("err"),
                ])

    print(f"\nCSV → {out_path.relative_to(REPO) if out_path.is_relative_to(REPO) else out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
