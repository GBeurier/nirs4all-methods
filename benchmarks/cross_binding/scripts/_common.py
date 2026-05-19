"""Helpers shared by all Python benchmark scripts.

The new (parity + timing + versions) cell convention:

  python script.py <algo> <n> <p> <n_components> <max_total_runs> \
      <seed_base> <predictions_path>

Each script:
  - loads / regenerates the deterministic dataset per run (seed_base+i)
  - runs the adaptive timing protocol used by the benchmark dashboard
  - saves the LAST run's predictions as a .npy file under
    <predictions_path> so the orchestrator can compute parity post-hoc
  - emits a JSON record on the LAST stdout line with timing + versions

R / MATLAB equivalents follow the same JSON contract (see
bench_r_tier1.R / bench_matlab_tier1.m for the R/Octave mirror).
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Argument parsing — new 7-arg contract
# ----------------------------------------------------------------------

def parse_args():
    """Return (algo, csv_dir, n, p, n_components, max_total_runs, seed_base, predictions_path).

    All backends read the orchestrator-generated CSV files
    `<csv_dir>/data_<n>x<p>_seed<seed>.csv` so they train on the *exact*
    same input bits. Cross-language parity then makes sense: any diff
    in predictions is attributable to the implementation, not to the
    fact that NumPy / R / Octave use different RNGs from the same seed.
    """
    if len(sys.argv) != 9:
        print("usage: SCRIPT algo csv_dir n p n_components max_total_runs "
               "seed_base predictions_path",
               file=sys.stderr)
        sys.exit(2)
    algo = sys.argv[1]
    csv_dir = Path(sys.argv[2])
    n = int(sys.argv[3])
    p = int(sys.argv[4])
    nc = int(sys.argv[5])
    runs = int(sys.argv[6])
    seed_base = int(sys.argv[7])
    pred_path = Path(sys.argv[8])
    return algo, csv_dir, n, p, nc, runs, seed_base, pred_path


def load_dataset(csv_dir: Path, n: int, p: int, seed: int):
    """Load the orchestrator-cached deterministic dataset for (n, p, seed).
    Returns (X, y) with X shape (n, p) and y shape (n,)."""
    csv = csv_dir / f"data_{n}x{p}_seed{seed}.csv"
    arr = np.loadtxt(csv, delimiter=",", skiprows=1, dtype=np.float64)
    X = np.ascontiguousarray(arr[:, :-1])
    y = arr[:, -1].astype(np.float64)
    return X, y


# ----------------------------------------------------------------------
# Adaptive timed-run loop with per-run seed
# ----------------------------------------------------------------------

WARMUP_ABORT_MS = 5 * 60 * 1000.0
PROBE_ABORT_MS = 60 * 1000.0
DEFAULT_MAX_RUNS = 40


def adaptive_target(probe_ms: float, max_runs: int) -> tuple[int, str, str]:
    """Return (target total executions, statistic, decision) from run #2."""
    max_runs = max(2, int(max_runs))
    if probe_ms > PROBE_ABORT_MS:
        return 2, "single", "probe_gt_60s"
    if probe_ms > 30_000.0:
        return min(max_runs, 2), "single", "probe_gt_30s"
    if probe_ms > 5_000.0:
        return min(max_runs, 3), "mean", "probe_gt_5s"
    if probe_ms > 1_000.0:
        return min(max_runs, 10), "median", "probe_gt_1s"
    if probe_ms > 100.0:
        return min(max_runs, 20), "median", "probe_gt_100ms"
    return min(max_runs, DEFAULT_MAX_RUNS), "median", "probe_le_100ms"


def _reported_value(samples: list[float], statistic: str) -> float:
    if statistic == "mean":
        return statistics.fmean(samples)
    return statistics.median(samples)


def _stats(samples: list[float], *, statistic: str, warmup_ms: float,
           decision: str, max_runs: int, total_runs: int,
           warmup_included: bool = False) -> dict:
    if len(samples) == 1:
        statistic = "single"
    reported = _reported_value(samples, statistic)
    return {
        "ok": True,
        "n_runs": len(samples),
        # Compatibility: legacy renderers read median_ms as the score.
        # For adaptive-v1 this is the reported score, which may be a mean
        # for slow cells. New renderers should prefer reported_ms.
        "median_ms": reported,
        "reported_ms": reported,
        "sample_median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "warmup_ms": warmup_ms,
        "warmup_included": warmup_included,
        "timing_statistic": statistic,
        "timing_decision": decision,
        "max_runs": max(1, int(max_runs)),
        "total_runs": max(1, int(total_runs)),
    }


def time_runs_seeded(fit_predict_seeded, n_runs: int, seed_base: int):
    """Run the adaptive timing protocol.

    Protocol:
      1. Run one warm-up with `seed_base` and time it.
      2. If the warm-up takes more than 5 min, report that single time.
      3. Otherwise run one timed probe with `seed_base`.
      4. If run #2 takes more than 30 s, report that single second-run
         time (the warm-up is discarded).
      5. Otherwise choose the total execution count from run #2:
         >5 s -> 3 runs / mean of runs #2..#3; >1 s -> 10 runs /
         median of runs #2..#10; >0.1 s -> 20 runs / median; <=0.1 s
         -> up to 40 runs / median.

    Returns:
      (stats_dict, last_predictions_array)
      - stats_dict: ok / n_runs / reported_ms / timing_statistic /
        median_ms / mean_ms / min_ms / max_ms
      - last_predictions_array: the predictions array from the LAST
        timed run (seed_base + n_runs - 1). The caller persists it to
        the predictions file for post-hoc parity computation.

      `fit_predict_seeded(seed)` must return a numpy.ndarray of predictions
      (any shape, will be flattened for parity comparison).
    """
    max_runs = max(1, int(n_runs))

    t0 = time.perf_counter()
    warmup_preds = fit_predict_seeded(seed_base)
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    if warmup_ms > WARMUP_ABORT_MS:
        return _stats([warmup_ms], statistic="single", warmup_ms=warmup_ms,
                      decision="warmup_gt_5min", max_runs=max_runs,
                      total_runs=1, warmup_included=True), warmup_preds

    if max_runs < 2:
        return _stats([warmup_ms], statistic="single", warmup_ms=warmup_ms,
                      decision="max_runs_1_warmup_only",
                      max_runs=max_runs, total_runs=1,
                      warmup_included=True), warmup_preds

    samples = []
    t0 = time.perf_counter()
    last_preds = fit_predict_seeded(seed_base)
    probe_ms = (time.perf_counter() - t0) * 1000.0
    samples.append(probe_ms)

    target_total, statistic, decision = adaptive_target(probe_ms, max_runs)
    target_samples = max(1, target_total - 1)
    for i in range(1, target_samples):
        seed = seed_base + i
        t0 = time.perf_counter()
        last_preds = fit_predict_seeded(seed)
        samples.append((time.perf_counter() - t0) * 1000.0)

    return _stats(samples, statistic=statistic, warmup_ms=warmup_ms,
                  decision=decision, max_runs=max_runs,
                  total_runs=1 + len(samples)), last_preds


# ----------------------------------------------------------------------
# Version metadata
# ----------------------------------------------------------------------

def _safe_version(import_name: str, attr: str = "__version__") -> str:
    """Try to import and return the version attribute; return 'MISSING' on
    failure. Used in version capture without burdening the script with
    explicit try/except per module."""
    try:
        m = __import__(import_name)
        return str(getattr(m, attr, "unknown"))
    except Exception:
        return "MISSING"


def _blas_info() -> str:
    """Best-effort BLAS detection. threadpoolctl gives the most accurate
    info; fallback to env-var hint."""
    try:
        import threadpoolctl  # type: ignore
        info = threadpoolctl.threadpool_info()
        if not info:
            return "unknown"
        parts = []
        for entry in info:
            v = entry.get("version") or "?"
            api = entry.get("user_api") or entry.get("internal_api") or "?"
            n = entry.get("num_threads") or "?"
            name = entry.get("prefix") or entry.get("filepath", "?")
            parts.append(f"{name} {v} {api}={n}")
        return "; ".join(parts)
    except Exception:
        return f"env OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '?')}"


def collect_versions(language: str, **extras) -> dict:
    """Return a dict capturing language version, BLAS info, plus
    extra named lib versions passed in `extras` as `{"name": "1.2.3"}`."""
    base = {
        "language": f"{language} {platform.python_version()}"
            if language == "Python"
            else f"{language} (host {platform.system()} "
                  f"{platform.machine()})",
        "blas": _blas_info(),
    }
    base.update(extras)
    return base


# ----------------------------------------------------------------------
# JSON emission with parity-predictions persistence
# ----------------------------------------------------------------------

def emit_record(stats: dict, predictions, predictions_path: Path,
                 versions: dict | None = None) -> None:
    """Persist the predictions vector to disk and print the JSON record
    on stdout (as the LAST line, so the orchestrator picks it up via
    `out.strip().splitlines()[-1]`).
    """
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(predictions, dtype=np.float64).ravel()
    np.save(predictions_path, arr, allow_pickle=False)
    rec = dict(stats)
    rec["predictions_path"] = str(predictions_path)
    rec["versions"] = versions or {}
    print(json.dumps(rec))


# ----------------------------------------------------------------------
# Convenience: numpy version helper for scripts that need it
# ----------------------------------------------------------------------

def numpy_version() -> str:
    return _safe_version("numpy")


# The legacy `dispatch_pls4all_fit` hand-coded cascade was retired:
# bench_cpp.py and bench_python_tier1.py now route through
# `MethodSpec.pls4all_fn` in `benchmarks/parity_timing/registry.py`, which
# covers the full canonical method catalog. See bench_registry_common.py.
