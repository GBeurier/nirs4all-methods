"""Helpers shared by all Python benchmark scripts.

The new (parity + timing + versions) cell convention:

  python script.py <algo> <n> <p> <n_components> <n_runs> \
      <seed_base> <predictions_path>

Each script:
  - loads / regenerates the deterministic dataset per run (seed_base+i)
  - runs `fit_predict(seed)` n_runs times, discarding the first as warmup
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
    """Return (algo, csv_dir, n, p, n_components, n_runs, seed_base, predictions_path).

    All backends read the orchestrator-generated CSV files
    `<csv_dir>/data_<n>x<p>_seed<seed>.csv` so they train on the *exact*
    same input bits. Cross-language parity then makes sense: any diff
    in predictions is attributable to the implementation, not to the
    fact that NumPy / R / Octave use different RNGs from the same seed.
    """
    if len(sys.argv) != 9:
        print("usage: SCRIPT algo csv_dir n p n_components n_runs "
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
# Timed-run loop with per-run seed
# ----------------------------------------------------------------------

def time_runs_seeded(fit_predict_seeded, n_runs: int, seed_base: int):
    """Run `fit_predict_seeded(seed)` for seed in [base, base+1, …, base+n_runs-1].

    Returns:
      (stats_dict, last_predictions_array)
      - stats_dict: ok / n_runs / median_ms / min_ms / max_ms (warmup
        discarded when n_runs >= 3)
      - last_predictions_array: the predictions array from the LAST
        timed run (seed_base + n_runs - 1). The caller persists it to
        the predictions file for post-hoc parity computation.

    `fit_predict_seeded(seed)` must return a numpy.ndarray of predictions
    (any shape, will be flattened for parity comparison).
    """
    samples = []
    last_preds = None
    for i in range(n_runs):
        seed = seed_base + i
        t0 = time.perf_counter()
        last_preds = fit_predict_seeded(seed)
        samples.append((time.perf_counter() - t0) * 1000.0)
    if n_runs >= 3:
        timed = samples[1:]
    else:
        timed = samples
    return {
        "ok": True,
        "n_runs": len(timed),
        "median_ms": statistics.median(timed),
        "min_ms": min(timed),
        "max_ms": max(timed),
    }, last_preds


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


# ----------------------------------------------------------------------
# Shared pls4all dispatcher (used by bench_cpp + bench_python_tier1)
# ----------------------------------------------------------------------

def dispatch_pls4all_fit(algo: str, ctx, cfg, X, y2d, n: int, p: int,
                          seed: int):
    """Call the right _methods.<algo>_fit with algorithm-specific params.

    Returns the MethodResult. Both bench_cpp.py and bench_python_tier1.py
    route through here so their predictions stay bit-identical, which
    keeps the `cpp` reference column meaningful across the entire matrix.
    """
    from pls4all import _methods  # local import: heavy
    import numpy as np

    if algo == "pls":
        return _methods.sparse_simpls_fit(ctx, cfg, X, y2d, sparsity_lambda=0.0)
    if algo == "sparse_simpls":
        return _methods.sparse_simpls_fit(ctx, cfg, X, y2d, sparsity_lambda=0.05)
    if algo == "cppls":
        return _methods.cppls_fit(ctx, cfg, X, y2d, gamma=0.5)
    if algo == "ecr":
        return _methods.ecr_fit(ctx, cfg, X, y2d, alpha=0.5)
    if algo == "mir_pls":
        return _methods.mir_pls_fit(ctx, cfg, X, y2d)
    if algo == "ridge_pls":
        return _methods.ridge_pls_fit(ctx, cfg, X, y2d, ridge_lambda=1.0)
    if algo == "robust_pls":
        return _methods.robust_pls_fit(ctx, cfg, X, y2d, huber_k=1.345,
                                         max_irls_iter=20)
    if algo == "missing_aware_nipals":
        return _methods.missing_aware_nipals_fit(ctx, cfg, X, y2d)
    if algo == "continuum_regression":
        return _methods.continuum_regression_fit(ctx, cfg, X, y2d, tau=0.5)
    if algo == "kernel_pls":
        return _methods.kernel_pls_fit(ctx, cfg, X, y2d, kernel_type=1,
                                         gamma=0.0, coef0=1.0, degree=3)
    if algo == "o2pls":
        return _methods.o2pls_fit(ctx, cfg, X, y2d, n_predictive=cfg.n_components,
                                    n_x_orthogonal=1, n_y_orthogonal=1)
    if algo == "bagging_pls":
        return _methods.bagging_pls_fit(ctx, cfg, X, y2d, n_estimators=50, seed=0)
    if algo == "boosting_pls":
        return _methods.boosting_pls_fit(ctx, cfg, X, y2d, n_estimators=50,
                                           learning_rate=0.1)
    if algo == "random_subspace_pls":
        return _methods.random_subspace_pls_fit(ctx, cfg, X, y2d,
                                                  n_estimators=50,
                                                  features_per_subspace=10,
                                                  seed=0)
    if algo == "fused_sparse_pls":
        return _methods.fused_sparse_pls_fit(ctx, cfg, X, y2d, l1_lambda=0.05,
                                               fusion_lambda=0.05)
    if algo == "group_sparse_pls":
        ga = np.zeros(p, dtype=np.int32)
        ga[::2] = 1
        return _methods.group_sparse_pls_fit(ctx, cfg, X, y2d,
                                                group_assignment=ga,
                                                group_lambda=0.05)
    if algo == "lw_pls":
        return _methods.lw_pls_fit(ctx, cfg, X, y2d, n_neighbors=30)
    if algo == "pls_glm":
        return _methods.pls_glm_fit(ctx, cfg, X, y2d, poisson=False)
    if algo == "weighted_pls":
        w = np.ones(n, dtype=np.float64)
        return _methods.weighted_pls_fit(ctx, cfg, X, y2d, sample_weights=w)
    if algo in ("pls_qda", "pls_lda", "pls_logistic", "sparse_pls_da"):
        labels = np.zeros(n, dtype=np.int32)
        # Binary class from y > 0 (deterministic given seed-controlled data).
        labels[(y2d.ravel() > 0)] = 1
        if algo == "pls_qda":
            return _methods.pls_qda_fit(ctx, cfg, X, y_labels=labels)
        if algo == "pls_lda":
            return _methods.pls_lda_fit(ctx, cfg, X, y_labels=labels, n_classes=2)
        if algo == "pls_logistic":
            return _methods.pls_logistic_fit(ctx, cfg, X, y_labels=labels, n_classes=2)
        return _methods.sparse_pls_da_fit(ctx, cfg, X, y_labels=labels)
    if algo == "pls_cox":
        ev = np.ones(n, dtype=np.int32)
        st = np.abs(y2d.ravel()) + 0.5
        return _methods.pls_cox_fit(ctx, cfg, X, survival_times=st,
                                      event_indicators=ev)
    if algo == "pds":
        rng = np.random.default_rng(seed)
        Xt = X + 0.01 * rng.standard_normal(X.shape)
        return _methods.pds_fit(ctx, X, Xt, window_half_width=2)
    if algo == "ds":
        rng = np.random.default_rng(seed)
        Xt = X + 0.01 * rng.standard_normal(X.shape)
        return _methods.ds_fit(ctx, X, Xt)
    if algo == "gpr_pls":
        return _methods.gpr_pls_fit(ctx, cfg, X, y2d, length_scale=1.0,
                                      noise_level=1e-4, seed=0)
    # Generic fallback (works only for algos whose C signature is exactly
    # (ctx, cfg, X, y) — most don't, but if a new algo follows this
    # convention it works without touching the dispatcher).
    fit_fn = getattr(_methods, f"{algo}_fit", None)
    if fit_fn is None:
        raise RuntimeError(f"no _methods.{algo}_fit available")
    return fit_fn(ctx, cfg, X, y2d)
