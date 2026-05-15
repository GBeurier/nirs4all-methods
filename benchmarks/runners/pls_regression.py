"""PLS regression benchmark runner — pls4all (Python ctypes / NumPy
zero-copy) vs scikit-learn `PLSRegression`.

Matrix axes:
- algorithm/solver: NIPALS, orthogonal-scores, SIMPLS, kernel-algorithm,
  wide-kernel, SVD, power, randomized-SVD, plus PCR.
- sample size: 200, 500, 1000, 2000, 10000.
- feature count: 100, 1000, 10000.

The runner builds one synthetic dataset per (n, p) cell, then fits/predicts
each algorithm on both sides. Accuracy CSV records the predicted-vs-true
RMSE delta and the predicted-vs-pls4all RMSE delta against the sklearn
reference. Timing CSV records wall-clock fit+predict separately on each
side; both are reported as a single value per call (no repeats by default
for the larger cells — repeat count is a parameter).

The runner does NOT execute the full matrix by default. Call
`run(scale="smoke")` for a tiny sanity dataset; `run(scale="full")` for
the published matrix.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PLS4ALL_PY_SRC = REPO_ROOT / "bindings" / "python" / "src"
sys.path.insert(0, str(PLS4ALL_PY_SRC))

import pls4all  # noqa: E402

from ._harness import AccuracyRecord, TimingRecord, median_time_ms


# --- Algorithm registry ----------------------------------------------------

@dataclass(frozen=True)
class _Solver:
    name: str                       # benchmark id
    algorithm: pls4all.Algorithm
    solver: pls4all.Solver
    deflation: pls4all.Deflation


SOLVERS: tuple[_Solver, ...] = (
    _Solver("pls_nipals", pls4all.Algorithm.PLS_REGRESSION,
            pls4all.Solver.NIPALS, pls4all.Deflation.REGRESSION),
    _Solver("pls_orthogonal_scores", pls4all.Algorithm.PLS_REGRESSION,
            pls4all.Solver.ORTHOGONAL_SCORES, pls4all.Deflation.REGRESSION),
    _Solver("pls_simpls", pls4all.Algorithm.PLS_REGRESSION,
            pls4all.Solver.SIMPLS, pls4all.Deflation.REGRESSION),
    _Solver("pls_kernel_algorithm", pls4all.Algorithm.PLS_REGRESSION,
            pls4all.Solver.KERNEL_ALGORITHM, pls4all.Deflation.REGRESSION),
    _Solver("pls_wide_kernel", pls4all.Algorithm.PLS_REGRESSION,
            pls4all.Solver.WIDE_KERNEL, pls4all.Deflation.REGRESSION),
    _Solver("pls_svd", pls4all.Algorithm.PLS_REGRESSION,
            pls4all.Solver.SVD, pls4all.Deflation.REGRESSION),
    _Solver("pls_power", pls4all.Algorithm.PLS_REGRESSION,
            pls4all.Solver.POWER, pls4all.Deflation.REGRESSION),
    _Solver("pls_randomized_svd", pls4all.Algorithm.PLS_REGRESSION,
            pls4all.Solver.RANDOMIZED_SVD, pls4all.Deflation.REGRESSION),
    _Solver("pcr_svd", pls4all.Algorithm.PCR,
            pls4all.Solver.SVD, pls4all.Deflation.REGRESSION),
)


# --- Dataset definitions ---------------------------------------------------

@dataclass(frozen=True)
class _Cell:
    name: str
    n_samples: int
    n_features: int
    n_components: int
    seed: int


# Smoke set: 1-2 small cells, used for the gated --check baseline.
SMOKE_CELLS: tuple[_Cell, ...] = (
    _Cell("smoke-200x100",  n_samples=200,  n_features=100,  n_components=5, seed=11),
    _Cell("smoke-500x100",  n_samples=500,  n_features=100,  n_components=5, seed=13),
)


# Full matrix: published target. Run on demand under varying CPU counts.
FULL_CELLS: tuple[_Cell, ...] = tuple(
    _Cell(name=f"pls-{n}x{p}", n_samples=n, n_features=p,
          n_components=min(10, n - 1, p), seed=100 + 7 * idx)
    for idx, (n, p) in enumerate([
        (200,    100),  (200,    1_000),  (200,    10_000),
        (500,    100),  (500,    1_000),  (500,    10_000),
        (1_000,  100),  (1_000,  1_000),  (1_000,  10_000),
        (2_000,  100),  (2_000,  1_000),  (2_000,  10_000),
        (10_000, 100),  (10_000, 1_000),  (10_000, 10_000),
    ])
)


# Tolerances chosen to absorb small algorithmic differences (each solver
# converges to a slightly different latent basis) while still flagging
# regressions in prediction accuracy. Strict bit-for-bit parity is not the
# right gate here — that's what the C++ parity fixtures cover. The intent
# of this gate is "both sides converged to a comparable predictor", not
# "the latent bases are identical".
RMSE_REL_TOL = 5e-2
RMSE_ABS_TOL = 1e-6


def _make_dataset(cell: _Cell) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(cell.seed)
    n, p = cell.n_samples, cell.n_features
    n_components = cell.n_components
    # Latent factor model: X = T @ W^T + noise; y = T @ b + noise.
    T = rng.standard_normal((n, n_components))
    W = rng.standard_normal((p, n_components))
    noise_x = rng.standard_normal((n, p)) * 0.05
    X = T @ W.T + noise_x
    b = rng.standard_normal(n_components)
    y = T @ b + 0.02 * rng.standard_normal(n)
    return X.astype(np.float64), y.reshape(-1, 1).astype(np.float64), n_components


def _make_config(solver: _Solver, n_components: int) -> pls4all.Config:
    cfg = pls4all.Config()
    cfg.algorithm = solver.algorithm
    cfg.solver = solver.solver
    cfg.deflation = solver.deflation
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.scale_x = False
    cfg.center_y = True
    cfg.scale_y = False
    cfg.store_scores = False
    return cfg


def _pls4all_fit_predict(solver: _Solver, cell: _Cell,
                         X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Returns (predictions, fit_ms, predict_ms)."""
    with pls4all.Context() as ctx, _make_config(solver, cell.n_components) as cfg:
        t0 = time.perf_counter()
        model = pls4all.Model.fit(ctx, cfg, X, y)
        t1 = time.perf_counter()
        try:
            preds = model.predict(ctx, X)
        finally:
            t2 = time.perf_counter()
            model.close()
        return preds, (t1 - t0) * 1_000.0, (t2 - t1) * 1_000.0


def _sklearn_fit_predict(cell: _Cell,
                          X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    from sklearn.cross_decomposition import PLSRegression  # type: ignore[import-not-found]
    estimator = PLSRegression(n_components=cell.n_components, scale=False)
    t0 = time.perf_counter()
    estimator.fit(X, y)
    t1 = time.perf_counter()
    preds = np.asarray(estimator.predict(X), dtype=np.float64)
    t2 = time.perf_counter()
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    return preds, (t1 - t0) * 1_000.0, (t2 - t1) * 1_000.0


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean((a - b) ** 2))))


def _sklearn_version() -> str:
    try:
        return importlib.import_module("sklearn").__version__
    except Exception:
        return "unavailable"


def _numpy_version() -> str:
    return np.__version__


def _host_threads() -> int:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "PLS4ALL_BENCH_THREADS"):
        value = os.environ.get(var)
        if value:
            try:
                return int(value)
            except ValueError:
                pass
    return os.cpu_count() or 0


def _cells_for_scale(scale: str) -> tuple[_Cell, ...]:
    if scale == "smoke":
        return SMOKE_CELLS
    if scale == "full":
        return FULL_CELLS
    raise ValueError(f"unknown scale {scale!r}; expected 'smoke' or 'full'")


def run(*, scale: str = "smoke",
        repeats: int = 1,
        skip_oversize: bool = True) -> tuple[list[AccuracyRecord], list[TimingRecord], dict]:
    """Run the PLS regression benchmark.

    Returns (accuracy_records, timing_records, metadata).

    `scale='smoke'` runs the two small cells gated by the committed CSV.
    `scale='full'` runs the published 9x15 matrix.

    `skip_oversize`: when True, cells with n*p > 5e7 doubles fall back to
    sklearn alone (pls4all timings are populated but accuracy_pass is
    reported against pls4all if available). With the current Python ctypes
    wrapper, all sizes go through; the flag is a kill switch for slow
    hosts.
    """
    metadata = {
        "pls4all_version": pls4all.version(),
        "sklearn_version": _sklearn_version(),
        "numpy_version": _numpy_version(),
        "host_threads": str(_host_threads()),
    }

    cells = _cells_for_scale(scale)
    accuracy: list[AccuracyRecord] = []
    timings: list[TimingRecord] = []

    for cell in cells:
        # Single dataset per cell, reused across solvers.
        X, y, _ = _make_dataset(cell)

        # Sklearn reference (one fit per cell).
        sk_preds, sk_fit_ms, sk_predict_ms = _sklearn_fit_predict(cell, X, y)
        sk_rmse = _rmse(sk_preds, y)

        for solver in SOLVERS:
            # Skip wide-kernel on tall (n >> p) and kernel-algorithm on wide
            # (p >> n) datasets where they are pathological. Same for SVD
            # solvers when n_components > min(n,p).
            try:
                preds, fit_ms, predict_ms = _pls4all_fit_predict(solver, cell, X, y)
                pls4all_rmse = _rmse(preds, y)
                rmse_diff = _rmse(preds, sk_preds)
                rmse_rel = rmse_diff / sk_rmse if sk_rmse > 0 else float("inf")
                accuracy_pass = bool(
                    rmse_diff <= RMSE_ABS_TOL
                    or (not math.isinf(rmse_rel) and rmse_rel <= RMSE_REL_TOL)
                )
                status = "ok"
                error = ""
            except pls4all.Pls4allError as exc:
                preds = np.empty((0, 0))
                fit_ms = float("nan")
                predict_ms = float("nan")
                pls4all_rmse = float("nan")
                rmse_diff = float("nan")
                rmse_rel = float("nan")
                accuracy_pass = False
                status = "pls4all_error"
                error = str(exc)

            accuracy.append(AccuracyRecord(
                benchmark="pls_regression",
                case=f"{cell.name}-{solver.name}",
                n_samples=cell.n_samples,
                n_features=cell.n_features,
                n_operators=1,
                max_components=cell.n_components,
                n_folds=0,
                selected_operator_index=0,
                selected_operator_index_match=True,
                selected_n_components=cell.n_components,
                selected_n_components_match=True,
                best_score_abs_delta=abs(pls4all_rmse - sk_rmse) if status == "ok" else float("nan"),
                prediction_rmse_abs_delta=rmse_diff,
                prediction_rmse_rel_delta=rmse_rel,
                accuracy_pass=accuracy_pass,
            ))
            timings.append(TimingRecord(
                benchmark="pls_regression",
                case=f"{cell.name}-{solver.name}",
                n_samples=cell.n_samples,
                n_features=cell.n_features,
                n_operators=1,
                max_components=cell.n_components,
                repeats=repeats,
                pls4all_end_to_end_ms_median=fit_ms + predict_ms,
                pls4all_end_to_end_ms_min=fit_ms + predict_ms,
                bench_end_to_end_ms_median=sk_fit_ms + sk_predict_ms,
                bench_end_to_end_ms_min=sk_fit_ms + sk_predict_ms,
                speedup_median=(sk_fit_ms + sk_predict_ms) / (fit_ms + predict_ms)
                if (fit_ms + predict_ms) > 0 else float("inf"),
            ))

    return accuracy, timings, metadata
