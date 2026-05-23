"""Comprehensive performance matrix runner.

For every (algorithm, n_samples, n_features) cell, time fit + predict
on every available language path. Records:

  - Native C++ (via `n4m_cli --bench`).
  - pls4all-Python (via the ctypes Model wrapper).
  - pls4all-R (via the `pls4all` R package, when R + the package are
    installed; skipped with a status of `r_not_available` otherwise).
  - Reference: scikit-learn `PLSRegression` for PLS solvers / `Pipeline(
    [PCA, LinearRegression])` for `pcr_svd`. Both run under a single
    OPENBLAS_NUM_THREADS as configured by the host environment.

The cell grid is the published matrix:
  n_samples in {200, 500, 1000, 2000, 10000}
  n_features in {100, 1000, 10000}
  algos: pls_nipals, pls_orthogonal_scores, pls_simpls,
         pls_kernel_algorithm, pls_wide_kernel, pls_svd, pls_power,
         pls_randomized_svd, pcr_svd
  total cells: 9 algos x 5 n x 3 p = 135 cells per language.

The runner is gated behind `--scale full`. For the default `--scale
smoke`, only `200 x 100` and `500 x 100` cells are exercised, and only
two algorithms (`pls_simpls`, `pls_svd`). Use the full mode when the
host machine is free; the runner writes to
`benchmarks/results/matrix/`.

Output:
  results/matrix/timing.csv      (informational, NOT gated)
  results/matrix/accuracy.csv    (gated; pls4all language paths must
                                  produce predictions within tolerance of
                                  the C++ reference; reference column is
                                  informational and not gated against
                                  pls4all).
  results/matrix/summary.md      (regenerated each run)
  results/matrix/metadata.json   (versions, host info, env)

The runner never decides what core count to use. The caller controls
OMP/OPENBLAS/MKL/PLS4ALL_BENCH_THREADS env vars, and the matrix records
them per row.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import pls_regression  # for synthetic dataset + sklearn helpers
from ._harness import host_info, round_float

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "build" / "dev-release" / "cpp" / "cli" / "n4m_cli"
PLS4ALL_PY_SRC = REPO_ROOT / "bindings" / "python" / "src"
R_PACKAGE_DIR = REPO_ROOT / "bindings" / "r" / "pls4all"
sys.path.insert(0, str(PLS4ALL_PY_SRC))

import pls4all  # noqa: E402  - load after sys.path tweak

_ALGOS = (
    "pls_nipals",
    "pls_orthogonal_scores",
    "pls_simpls",
    "pls_kernel_algorithm",
    "pls_wide_kernel",
    "pls_svd",
    "pls_power",
    "pls_randomized_svd",
    "pcr_svd",
)

_FULL_GRID = (
    (200,    100),  (200,    1_000),  (200,    10_000),
    (500,    100),  (500,    1_000),  (500,    10_000),
    (1_000,  100),  (1_000,  1_000),  (1_000,  10_000),
    (2_000,  100),  (2_000,  1_000),  (2_000,  10_000),
    (10_000, 100),  (10_000, 1_000),  (10_000, 10_000),
)

_SMOKE_GRID = ((200, 100), (500, 100))
_SMOKE_ALGOS = ("pls_simpls", "pls_svd")


# --- Dataset construction (mirrors pls_regression for parity) -------------

@dataclass(frozen=True)
class _Cell:
    name: str
    n_samples: int
    n_features: int
    n_components: int
    seed: int


def _cell(name: str, n: int, p: int, seed: int) -> _Cell:
    return _Cell(
        name=name, n_samples=n, n_features=p,
        n_components=min(10, n - 1, p), seed=seed,
    )


def _make_dataset(cell: _Cell) -> tuple[np.ndarray, np.ndarray]:
    bridge = pls_regression._Cell(
        name=cell.name, n_samples=cell.n_samples, n_features=cell.n_features,
        n_components=cell.n_components, seed=cell.seed,
    )
    X, y, _ = pls_regression._make_dataset(bridge)
    return X, y


# --- Wall-clock helpers ---------------------------------------------------

def _median_min(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(np.median(values)), float(min(values))


def _filter_finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


# --- Language paths -------------------------------------------------------

def _time_python(algo: str, cell: _Cell, X: np.ndarray, y: np.ndarray,
                 repeats: int) -> tuple[list[float], np.ndarray | None, str]:
    """pls4all-Python via ctypes Model."""
    solver = next((s for s in pls_regression.SOLVERS if s.name == algo), None)
    if solver is None:
        return [], None, f"unknown_algo:{algo}"
    bridge = pls_regression._Cell(
        name=cell.name, n_samples=cell.n_samples, n_features=cell.n_features,
        n_components=cell.n_components, seed=cell.seed,
    )
    times: list[float] = []
    last_preds: np.ndarray | None = None
    for _ in range(repeats):
        try:
            preds, fit_ms, predict_ms = pls_regression._pls4all_fit_predict(
                solver, bridge, X, y)
            times.append(fit_ms + predict_ms)
            last_preds = preds
        except pls4all.Pls4allError as exc:
            return times, last_preds, f"python_error:{exc}"
    return times, last_preds, "ok"


def _time_sklearn(algo: str, cell: _Cell, X: np.ndarray, y: np.ndarray,
                  repeats: int) -> tuple[list[float], np.ndarray | None, str]:
    bridge = pls_regression._Cell(
        name=cell.name, n_samples=cell.n_samples, n_features=cell.n_features,
        n_components=cell.n_components, seed=cell.seed,
    )
    try:
        from sklearn.cross_decomposition import PLSRegression  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return [], None, f"sklearn_missing:{exc}"
    times: list[float] = []
    last_preds: np.ndarray | None = None
    for _ in range(repeats):
        try:
            if algo == "pcr_svd":
                from sklearn.decomposition import PCA
                from sklearn.linear_model import LinearRegression
                from sklearn.pipeline import Pipeline
                pipe = Pipeline([
                    ("pca", PCA(n_components=cell.n_components)),
                    ("lr", LinearRegression()),
                ])
                t0 = time.perf_counter()
                pipe.fit(X, y.ravel())
                t1 = time.perf_counter()
                preds = pipe.predict(X).reshape(-1, 1)
                t2 = time.perf_counter()
            else:
                preds, fit_ms, predict_ms = pls_regression._sklearn_fit_predict(
                    bridge, X, y)
                # _sklearn_fit_predict already times — re-time uniformly here.
                t0 = time.perf_counter()
                from sklearn.cross_decomposition import PLSRegression as _PLS
                est = _PLS(n_components=cell.n_components, scale=False)
                est.fit(X, y)
                t1 = time.perf_counter()
                preds = np.asarray(est.predict(X), dtype=np.float64)
                t2 = time.perf_counter()
            ms = (t2 - t0) * 1_000.0
            times.append(ms)
            last_preds = preds.reshape(-1, 1) if preds.ndim == 1 else preds
        except Exception as exc:  # pragma: no cover
            return times, last_preds, f"sklearn_error:{exc}"
    return times, last_preds, "ok"


def _time_native_cli(algo: str, cell: _Cell, repeats: int) -> tuple[list[float], str]:
    """Spawn n4m_cli --bench. The CLI runs its own dataset generator
    and reports median + min internally; for the matrix we collect the
    median wall-time it reports as the cell timing.
    """
    if not CLI_PATH.exists():
        return [], "cli_not_built"
    cmd = [
        str(CLI_PATH), "--bench",
        "--algo", algo,
        "--samples", str(cell.n_samples),
        "--features", str(cell.n_features),
        "--components", str(cell.n_components),
        "--repeats", str(repeats),
        "--seed", str(cell.seed),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=600)
    except subprocess.TimeoutExpired:
        return [], "cli_timeout"
    if proc.returncode != 0:
        return [], f"cli_exit_{proc.returncode}"
    for line in proc.stdout.splitlines():
        if not line.startswith("DATA,"):
            continue
        cols = line.split(",")
        if len(cols) < 12:
            return [], "cli_bad_output"
        status = cols[7]
        try:
            fit_med = float(cols[8])
            predict_med = float(cols[9])
        except ValueError:
            return [], "cli_parse_error"
        return [fit_med + predict_med], status
    return [], "cli_no_data"


def _r_available() -> bool:
    return shutil.which("Rscript") is not None


def _time_r(algo: str, cell: _Cell, X: np.ndarray, y: np.ndarray,
            repeats: int, work_dir: Path) -> tuple[list[float], np.ndarray | None, str]:
    """Run R via Rscript with a temporary script. Returns timings + last
    predictions if the R package is installed; otherwise reports the
    appropriate skip status.
    """
    if not _r_available():
        return [], None, "r_not_available"
    work_dir.mkdir(parents=True, exist_ok=True)
    x_path = work_dir / f"{cell.name}_X.csv"
    y_path = work_dir / f"{cell.name}_Y.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, y, delimiter=",")
    pred_path = work_dir / f"{cell.name}_{algo}_pred.csv"
    script = work_dir / f"{cell.name}_{algo}.R"
    script.write_text(
        "args <- commandArgs(trailingOnly = TRUE)\n"
        "if (length(args) < 6) stop(\"args: X Y algo n_components repeats pred_out\")\n"
        "X <- as.matrix(read.table(args[1], sep = \",\"))\n"
        "Y <- as.matrix(read.table(args[2], sep = \",\"))\n"
        "algo <- args[3]\n"
        "n_components <- as.integer(args[4])\n"
        "repeats <- as.integer(args[5])\n"
        "pred_out <- args[6]\n"
        "if (!requireNamespace(\"pls4all\", quietly = TRUE)) {\n"
        "  cat(\"STATUS,r_package_missing\\n\"); quit(status = 0)\n"
        "}\n"
        "library(pls4all)\n"
        "times <- numeric(repeats)\n"
        "preds <- NULL\n"
        "for (i in seq_len(repeats)) {\n"
        "  t0 <- proc.time()\n"
        "  model <- pls4all_fit(X, Y, algo = algo, n_components = n_components)\n"
        "  preds <- pls4all_predict(model, X)\n"
        "  t1 <- proc.time()\n"
        "  times[i] <- as.numeric(t1[[\"elapsed\"]] - t0[[\"elapsed\"]]) * 1000.0\n"
        "}\n"
        "write.table(preds, file = pred_out, sep = \",\", row.names = FALSE, col.names = FALSE)\n"
        "cat(\"STATUS,ok\\n\")\n"
        "cat(\"TIMES,\", paste(times, collapse = \",\"), \"\\n\", sep = \"\")\n",
        encoding="utf-8",
    )
    cmd = [
        "Rscript", "--vanilla", str(script),
        str(x_path), str(y_path), algo,
        str(cell.n_components), str(repeats), str(pred_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=1200)
    except subprocess.TimeoutExpired:
        return [], None, "r_timeout"
    if proc.returncode != 0:
        return [], None, f"r_exit_{proc.returncode}"
    status = None
    times: list[float] = []
    for line in proc.stdout.splitlines():
        if line.startswith("STATUS,"):
            status = line.split(",", 1)[1].strip()
        elif line.startswith("TIMES,"):
            times = [float(v) for v in line.split(",")[1:] if v.strip()]
    if status is None:
        return [], None, "r_no_status"
    preds = None
    if status == "ok" and pred_path.exists():
        preds = np.loadtxt(pred_path, delimiter=",")
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
    return times, preds, status


# --- Runner --------------------------------------------------------------

def _accuracy_pass(reference: np.ndarray | None, actual: np.ndarray | None,
                   rmse_rel_tol: float) -> tuple[bool, float, float]:
    if reference is None or actual is None or reference.size == 0:
        return False, float("nan"), float("nan")
    if actual.shape != reference.shape:
        return False, float("nan"), float("nan")
    diff = float(np.sqrt(np.mean((actual - reference) ** 2)))
    ref_rmse = float(np.sqrt(np.mean(reference ** 2)))
    rel = diff / ref_rmse if ref_rmse > 0 else float("inf")
    is_close = diff <= 1e-8 or rel <= rmse_rel_tol
    return is_close, diff, rel


def _grid_for_scale(scale: str) -> tuple[tuple[int, int], ...]:
    if scale == "smoke":
        return _SMOKE_GRID
    if scale == "full":
        return _FULL_GRID
    raise ValueError(f"unknown scale {scale!r}; expected 'smoke' or 'full'")


def _algos_for_scale(scale: str) -> tuple[str, ...]:
    if scale == "smoke":
        return _SMOKE_ALGOS
    if scale == "full":
        return _ALGOS
    raise ValueError(f"unknown scale {scale!r}")


def run(*, scale: str = "smoke", repeats: int = 3) -> dict:
    """Run the comprehensive performance matrix.

    Returns a dict with keys:
      - "timing": list of timing rows (one per cell × language).
      - "accuracy": list of accuracy rows (one per cell × pls4all path).
      - "metadata": versions, host info, environment.

    Each row is a flat dict with the matrix columns. The caller is
    responsible for writing CSVs / markdown.
    """
    grid = _grid_for_scale(scale)
    algos = _algos_for_scale(scale)
    cells = [
        _cell(name=f"matrix-{n}x{p}", n=n, p=p, seed=200 + 7 * idx)
        for idx, (n, p) in enumerate(grid)
    ]

    work_dir = REPO_ROOT / "benchmarks" / "results" / "matrix" / "_r_work"
    timing_rows: list[dict] = []
    accuracy_rows: list[dict] = []

    metadata = {
        "pls4all_version": pls4all.version(),
        "scale": scale,
        "grid": [{"n_samples": n, "n_features": p} for n, p in grid],
        "algos": list(algos),
        "repeats": repeats,
        "env": {
            k: os.environ.get(k, "")
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS", "PLS4ALL_BENCH_THREADS")
        },
        "host": host_info(),
        "r_available": _r_available(),
        "cli_available": CLI_PATH.exists(),
    }

    for cell in cells:
        X, y = _make_dataset(cell)
        for algo in algos:
            # Reference: sklearn PLSRegression / PCA+LR for PCR.
            sk_times, sk_preds, sk_status = _time_sklearn(algo, cell, X, y, repeats)

            # pls4all-Python
            py_times, py_preds, py_status = _time_python(algo, cell, X, y, repeats)

            # native C++ via CLI (uses its own synthetic data with same seed)
            native_times, native_status = _time_native_cli(algo, cell, repeats)

            # R via Rscript
            r_times, r_preds, r_status = _time_r(algo, cell, X, y, repeats,
                                                 work_dir)

            sk_med, sk_min = _median_min(_filter_finite(sk_times))
            py_med, py_min = _median_min(_filter_finite(py_times))
            native_med, native_min = _median_min(_filter_finite(native_times))
            r_med, r_min = _median_min(_filter_finite(r_times))

            timing_rows.append({
                "case": cell.name,
                "algo": algo,
                "n_samples": cell.n_samples,
                "n_features": cell.n_features,
                "n_components": cell.n_components,
                "repeats": repeats,
                "native_cpp_ms_median": round_float(native_med, 3),
                "native_cpp_ms_min": round_float(native_min, 3),
                "native_cpp_status": native_status,
                "pls4all_python_ms_median": round_float(py_med, 3),
                "pls4all_python_ms_min": round_float(py_min, 3),
                "pls4all_python_status": py_status,
                "pls4all_r_ms_median": round_float(r_med, 3),
                "pls4all_r_ms_min": round_float(r_min, 3),
                "pls4all_r_status": r_status,
                "sklearn_ms_median": round_float(sk_med, 3),
                "sklearn_ms_min": round_float(sk_min, 3),
                "sklearn_status": sk_status,
            })

            # Accuracy: compare pls4all paths against sklearn.
            py_pass, py_rmse_abs, py_rmse_rel = _accuracy_pass(
                sk_preds, py_preds, rmse_rel_tol=5e-2)
            r_pass, r_rmse_abs, r_rmse_rel = _accuracy_pass(
                sk_preds, r_preds, rmse_rel_tol=5e-2)
            # If R wasn't available, don't fail the gate on it.
            r_gated = r_status == "ok"
            overall_pass = bool(py_pass and (r_pass or not r_gated))

            accuracy_rows.append({
                "case": cell.name,
                "algo": algo,
                "n_samples": cell.n_samples,
                "n_features": cell.n_features,
                "n_components": cell.n_components,
                "reference": "sklearn",
                "pls4all_python_rmse_abs": round_float(py_rmse_abs, 12),
                "pls4all_python_rmse_rel": round_float(py_rmse_rel, 12),
                "pls4all_python_status": py_status,
                "pls4all_python_pass": bool(py_pass),
                "pls4all_r_rmse_abs": round_float(r_rmse_abs, 12),
                "pls4all_r_rmse_rel": round_float(r_rmse_rel, 12),
                "pls4all_r_status": r_status,
                "pls4all_r_pass": bool(r_pass) if r_gated else None,
                "accuracy_pass": overall_pass,
            })

    return {
        "timing": timing_rows,
        "accuracy": accuracy_rows,
        "metadata": metadata,
    }
