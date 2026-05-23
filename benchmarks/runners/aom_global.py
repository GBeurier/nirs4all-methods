"""AOM-PLS global selection benchmark.

Compares the public C ABI surface (`n4m_aom_global_select`) against the local
`nirs4all/bench/AOM_v0/aompls` oracle on small synthetic datasets. Identical
validation folds are passed to both sides so the comparison is exact.

Numerical outputs (selected operator index, selected component count, best
score, prediction RMSE) are gated against tight tolerances. Timing outputs
are recorded separately and are NOT gated.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ._harness import AccuracyRecord, TimingRecord, median_time_ms


REPO_ROOT = Path(__file__).resolve().parents[2]
PLS4ALL_PY_SRC = REPO_ROOT / "bindings" / "python" / "src"
# Default lookup: assume the host layout has `pls4all/` and `nirs4all/` as
# sibling directories under one parent. `nirs4all/bench/AOM_v0/aompls/` is
# what we need on sys.path for the bench oracle.
DEFAULT_BENCH_DIR = REPO_ROOT.parent

sys.path.insert(0, str(PLS4ALL_PY_SRC))

import pls4all  # noqa: E402  — ctypes wrapper for libn4m


BENCH_OPERATOR_NAMES = ["identity", "detrend_d1"]
BENCH_OPERATOR_KIND_IDS = {
    "identity": int(pls4all.OperatorKind.IDENTITY),
    "detrend_d1": int(pls4all.OperatorKind.DETREND_POLY),
}
BENCH_OPERATOR_PARAMS = {
    "identity": [],
    "detrend_d1": [1.0],
}

# --- Tight numerical-equivalence gates ---
RMSE_ABS_TOL = 1e-8
RMSE_REL_TOL = 1e-8
BEST_SCORE_ABS_TOL = 1e-8
COEFFICIENT_L2_TOL = 1e-8


@dataclass(frozen=True)
class _Case:
    name: str
    n_samples: int
    n_features: int
    seed: int
    max_components: int
    cv: int
    random_state: int
    kind: str = "smooth_monotonic"


# Cases match the fixture scale documented in the design review.
CASES: tuple[_Case, ...] = (
    _Case(name="aom-global-9x6-ncomp3-cv3",
          n_samples=9, n_features=6, seed=49,
          max_components=3, cv=3, random_state=7),
    _Case(name="aom-global-12x8-ncomp4-cv3",
          n_samples=12, n_features=8, seed=53,
          max_components=4, cv=3, random_state=11),
    _Case(name="aom-global-16x10-ncomp4-cv4",
          n_samples=16, n_features=10, seed=61,
          max_components=4, cv=4, random_state=13),
    # Detrend-favouring case: each row has a large per-row linear baseline
    # that has no useful signal; the residual y depends on a high-frequency
    # cosine. Identity PLS sees mostly the baseline, while detrend exposes
    # the cosine before fitting.
    _Case(name="aom-global-detrend-favoured-14x9",
          n_samples=14, n_features=9, seed=71,
          max_components=4, cv=4, random_state=17,
          kind="detrend_favoured"),
)


def _ensure_bench_on_path() -> str:
    env_dir = os.environ.get("PLS4ALL_AOM_BENCH_DIR")
    candidate_dirs = [Path(env_dir)] if env_dir else []
    candidate_dirs.append(DEFAULT_BENCH_DIR)
    for candidate in candidate_dirs:
        aompls_dir = candidate / "nirs4all" / "bench" / "AOM_v0"
        if (aompls_dir / "aompls").is_dir():
            sys.path.insert(0, str(aompls_dir))
            return str(candidate)
    raise FileNotFoundError(
        "Could not locate nirs4all/bench/AOM_v0/aompls. "
        "Set PLS4ALL_AOM_BENCH_DIR to the parent directory of "
        "nirs4all/bench/AOM_v0."
    )


def _make_dataset(case: _Case) -> tuple[np.ndarray, np.ndarray]:
    """Build a small structured spectroscopy-like dataset.

    `kind="smooth_monotonic"` favours identity (X and y both trend linearly
    with the sample index). `kind="detrend_favoured"` adds a strong per-row
    linear baseline in X whose slope correlates with the sample index but
    not with y; y depends only on the post-detrend cosine residual. In that
    case the bench oracle and pls4all should both pick `detrend_d1`.
    """
    rng = np.random.default_rng(case.seed)
    n, p = case.n_samples, case.n_features
    feature_axis = np.linspace(0.0, 1.0, p)
    sample_axis = np.linspace(0.0, 1.0, n)
    if case.kind == "detrend_favoured":
        signal = np.cos(2.0 * np.pi * 1.5 * feature_axis[None, :]
                        + 0.4 * sample_axis[:, None])
        baseline_slope = 1.5 + 3.0 * sample_axis  # row-dependent slope
        baseline = baseline_slope[:, None] * feature_axis[None, :]
        X = baseline + 0.6 * signal + 0.02 * rng.standard_normal((n, p))
        # y depends only on the residual after detrend: integrate cosine.
        y_values = (np.cos(0.4 * sample_axis) + 0.05 * rng.standard_normal(n))
        y = y_values.reshape(n, 1)
    else:
        column_baseline = 1.0 + 0.5 * feature_axis
        sample_scale = 1.0 + 0.6 * sample_axis
        X = (sample_scale[:, None] * column_baseline[None, :]
             + 0.05 * rng.standard_normal((n, p)))
        y_trend = 0.4 + 0.7 * sample_axis
        y_noise = 0.02 * rng.standard_normal(n)
        y = (y_trend + y_noise).reshape(n, 1)
    return X.astype(np.float64), y.astype(np.float64)


def _kfold_indices(n_samples: int, cv: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Produce shuffled k-fold (train, test) index pairs for `cv` folds.

    Identical fold indices are passed to both pls4all and the bench oracle,
    so the only requirement is that the splitter be deterministic for a
    given (n_samples, cv, random_state). The implementation matches
    sklearn's `KFold(shuffle=True, random_state=...)` partitioning logic
    (one contiguous slice per fold over a shuffled index array) but uses
    NumPy directly so the benchmark does not depend on sklearn.
    """
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    permutation = indices.copy()
    rng.shuffle(permutation)
    fold_sizes = np.full(cv, n_samples // cv, dtype=np.int64)
    fold_sizes[: n_samples % cv] += 1
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    current = 0
    for fold_size in fold_sizes:
        stop = current + fold_size
        test_idx = np.sort(permutation[current:stop])
        mask = np.ones(n_samples, dtype=bool)
        mask[test_idx] = False
        train_idx = np.sort(indices[mask])
        splits.append((train_idx, test_idx))
        current = stop
    return splits


def _build_pls4all_artifacts(case: _Case,
                             folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[
        pls4all.OperatorBank, pls4all.ValidationPlan]:
    bank = pls4all.OperatorBank()
    for name in BENCH_OPERATOR_NAMES:
        bank.add(BENCH_OPERATOR_KIND_IDS[name], BENCH_OPERATOR_PARAMS[name])
    plan = pls4all.ValidationPlan()
    plan.n_samples = case.n_samples
    for train_idx, test_idx in folds:
        plan.add_fold(train_idx.tolist(), test_idx.tolist())
    return bank, plan


def _pls4all_run(case: _Case,
                 X: np.ndarray,
                 y: np.ndarray,
                 folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict:
    with pls4all.Context() as ctx, pls4all.Config() as cfg:
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.center_x = True
        cfg.scale_x = False
        cfg.center_y = True
        cfg.scale_y = False
        cfg.store_scores = False
        bank, plan = _build_pls4all_artifacts(case, folds)
        try:
            result = pls4all.aom_global_select(
                ctx, cfg, bank,
                X.reshape(-1).tolist(), y.reshape(-1).tolist(),
                plan, case.max_components,
                x_rows=case.n_samples, x_cols=case.n_features,
                y_rows=case.n_samples, y_cols=1,
            )
        finally:
            bank.close()
            plan.close()
        try:
            preds, pred_rows, pred_cols = result.predictions
            preds_arr = np.asarray(preds, dtype=np.float64).reshape(pred_rows, pred_cols)
            return {
                "selected_operator_index": int(result.selected_operator_index),
                "selected_n_components": int(result.selected_n_components),
                "best_score": float(result.best_score),
                "predictions": preds_arr,
                "operator_kinds": list(result.operator_kinds),
                "n_operators": int(result.n_operators),
            }
        finally:
            result.close()


def _bench_run(case: _Case,
               X: np.ndarray,
               y: np.ndarray,
               folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict:
    # bench imports are made lazily once the local oracle path is on sys.path.
    from aompls.estimators import AOMPLSRegressor  # type: ignore[import-not-found]
    from aompls.operators import (  # type: ignore[import-not-found]
        DetrendProjectionOperator,
        IdentityOperator,
    )

    # Build the bench-AOM_v0 operator bank to match the fixtures exactly.
    bank = [
        IdentityOperator(p=X.shape[1]),
        DetrendProjectionOperator(degree=1, p=X.shape[1]),
    ]

    # Pass explicit folds to AOM-PLS so both sides see the same CV plan.
    class _ExplicitFolds:
        def __init__(self, folds_):
            self._folds = list(folds_)

        def get_n_splits(self, *args, **kwargs):
            return len(self._folds)

        def split(self, X, y=None, groups=None):
            for train, test in self._folds:
                yield train, test

    splitter = _ExplicitFolds(folds)
    model = AOMPLSRegressor(
        n_components="auto",
        max_components=case.max_components,
        engine="simpls_materialized",
        selection="global",
        criterion="cv",
        operator_bank=bank,
        orthogonalization="transformed",
        center=True,
        scale=False,
        cv=case.cv,
        random_state=case.random_state,
        one_se_rule=False,
        cv_splitter=splitter,
    ).fit(X, y.ravel())
    preds = np.asarray(model.predict(X), dtype=np.float64)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    return {
        "selected_operator_index": int(model.selected_operator_indices_[0]),
        "selected_n_components": int(model.n_components_),
        "best_score": float(model.diagnostics_.extras["best_score"]),
        "predictions": preds,
        "coef": np.asarray(model.coef_, dtype=np.float64),
    }


def _rmse(predictions: np.ndarray, y: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean((predictions - y) ** 2))))


def _bench_version() -> str:
    try:
        from aompls import __version__ as bench_version  # type: ignore[import-not-found]
        return str(bench_version)
    except Exception:
        return "unversioned"


def run(*, repeats: int = 5) -> tuple[list[AccuracyRecord], list[TimingRecord], dict]:
    """Run the AOM-PLS global benchmark across `CASES`.

    Returns (accuracy_records, timing_records, metadata). Raises on any error.
    """
    _ensure_bench_on_path()
    metadata = {
        "pls4all_version": pls4all.version(),
        "bench_version": _bench_version(),
    }

    accuracy: list[AccuracyRecord] = []
    timings: list[TimingRecord] = []
    for case in CASES:
        X, y = _make_dataset(case)
        folds = _kfold_indices(case.n_samples, case.cv, case.random_state)

        pls4all_result = _pls4all_run(case, X, y, folds)
        bench_result = _bench_run(case, X, y, folds)

        pred_diff = pls4all_result["predictions"] - bench_result["predictions"]
        rmse_abs = float(np.sqrt(np.mean(pred_diff ** 2)))
        bench_rmse = _rmse(bench_result["predictions"], y)
        rmse_rel = rmse_abs / bench_rmse if bench_rmse > 0 else float("inf")
        best_score_abs = abs(pls4all_result["best_score"] - bench_result["best_score"])
        operator_match = (
            pls4all_result["selected_operator_index"]
            == bench_result["selected_operator_index"]
        )
        ncomp_match = (
            pls4all_result["selected_n_components"]
            == bench_result["selected_n_components"]
        )

        rmse_within_tol = (
            rmse_abs <= RMSE_ABS_TOL
            or (not math.isinf(rmse_rel) and rmse_rel <= RMSE_REL_TOL)
        )
        accuracy_pass = bool(
            rmse_within_tol
            and best_score_abs <= BEST_SCORE_ABS_TOL
            and operator_match
            and ncomp_match
        )
        accuracy.append(AccuracyRecord(
            benchmark="aom_global",
            case=case.name,
            n_samples=case.n_samples,
            n_features=case.n_features,
            n_operators=len(BENCH_OPERATOR_NAMES),
            max_components=case.max_components,
            n_folds=case.cv,
            selected_operator_index=pls4all_result["selected_operator_index"],
            selected_operator_index_match=operator_match,
            selected_n_components=pls4all_result["selected_n_components"],
            selected_n_components_match=ncomp_match,
            best_score_abs_delta=best_score_abs,
            prediction_rmse_abs_delta=rmse_abs,
            prediction_rmse_rel_delta=rmse_rel,
            accuracy_pass=accuracy_pass,
        ))

        pls4all_median, pls4all_min = median_time_ms(
            lambda: _pls4all_run(case, X, y, folds), repeats=repeats)
        bench_median, bench_min = median_time_ms(
            lambda: _bench_run(case, X, y, folds), repeats=repeats)
        speedup = bench_median / pls4all_median if pls4all_median > 0 else float("inf")
        timings.append(TimingRecord(
            benchmark="aom_global",
            case=case.name,
            n_samples=case.n_samples,
            n_features=case.n_features,
            n_operators=len(BENCH_OPERATOR_NAMES),
            max_components=case.max_components,
            repeats=repeats,
            pls4all_end_to_end_ms_median=pls4all_median,
            pls4all_end_to_end_ms_min=pls4all_min,
            bench_end_to_end_ms_median=bench_median,
            bench_end_to_end_ms_min=bench_min,
            speedup_median=speedup,
        ))

    return accuracy, timings, metadata
