"""sklearn-side diagnostic functions for fitted pls4all models.

These wrappers are intentionally **module-level functions**, not
`BaseEstimator` subclasses: they consume an already-fitted model (or a
matrix of fold-RMSEs) and produce a scalar / vector / matrix output.
They don't have `fit/predict` semantics that fit cleanly into
`Pipeline`.

Functions:
    * :func:`t2_score(model, X)` — Hotelling's T² per row
    * :func:`q_score(model, X)` — Q residual / SPE per row
    * :func:`dmodx_score(model, X)` — Distance-to-Model-X per row
    * :func:`pls_monitoring(model, X_ref, X_mon, alpha)` — T²/Q + alarms
    * :func:`approximate_press(X, y, max_components)` — Eastment-Krzanowski
      PRESS-style component-selection helper
    * :func:`one_se_rule(fold_rmse_matrix, max_components, n_folds)` —
      pick the most parsimonious component count within 1 SE of best
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import _methods
from .._config import Config
from .._context import Context
from .._model import Model
from .._types import Algorithm, Deflation, Solver


def _basic_cfg(n_components: int) -> Config:
    cfg = Config()
    cfg.algorithm = Algorithm.PLS_REGRESSION
    cfg.solver = Solver.SIMPLS
    cfg.deflation = Deflation.REGRESSION
    cfg.n_components = int(n_components)
    cfg.center_x = True
    cfg.scale_x = False
    cfg.center_y = True
    cfg.scale_y = False
    cfg.store_scores = True
    return cfg


def _t2_q_dmodx(model, X, key: str) -> np.ndarray:
    """Compute T², Q or DModX via the C diagnostics kernel."""
    import pls4all
    if not isinstance(model, Model):
        raise TypeError(
            "diagnostic functions require a pls4all.Model instance "
            f"(got {type(model).__name__})")
    ctx = Context()
    X_arr = np.ascontiguousarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={X_arr.ndim}")
    res = _methods.pls_diagnostics_compute(ctx, model, X_arr, X_reference=X_arr)
    return np.asarray(res.matrix(key), dtype=np.float64).ravel()


def t2_score(model, X) -> np.ndarray:
    """Hotelling T² per row of X under a fitted pls4all Model."""
    return _t2_q_dmodx(model, X, "t2")


def q_score(model, X) -> np.ndarray:
    """Q residual / squared prediction error per row of X."""
    return _t2_q_dmodx(model, X, "q")


def dmodx_score(model, X) -> np.ndarray:
    """Distance-to-Model-X per row of X."""
    return _t2_q_dmodx(model, X, "dmodx")


def pls_monitoring(model, X_reference, X_monitor, *, alpha: float = 0.95):
    """Run process monitoring on a fitted model: returns T²/Q time
    series + alarm flags for X_monitor against the X_reference baseline."""
    if not isinstance(model, Model):
        raise TypeError(
            f"pls_monitoring needs a pls4all.Model (got {type(model).__name__})")
    ctx = Context()
    Xr = np.ascontiguousarray(X_reference, dtype=np.float64)
    Xm = np.ascontiguousarray(X_monitor, dtype=np.float64)
    res = _methods.pls_monitoring_run(
        ctx, model, X_reference=Xr, X_monitor=Xm, alpha=float(alpha))
    return {
        "t2": np.asarray(res.matrix("t2"), dtype=np.float64).ravel(),
        "q": np.asarray(res.matrix("q"), dtype=np.float64).ravel(),
        "t2_alarm": np.asarray(res.matrix("t2_alarm"), dtype=np.float64).ravel(),
        "q_alarm": np.asarray(res.matrix("q_alarm"), dtype=np.float64).ravel(),
    }


def approximate_press(X, y, *, max_components: int) -> np.ndarray:
    """Eastment-Krzanowski-style approximate PRESS vector.

    Returns a length-(max_components) vector of approximate PRESS
    values; pick the component count via ``np.argmin``.
    """
    ctx = Context()
    cfg = _basic_cfg(int(max_components))
    try:
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        res = _methods.approximate_press_compute(
            ctx, cfg, X_arr, y_arr, max_components=int(max_components))
    finally:
        cfg.close()
    return np.asarray(res.matrix("press"), dtype=np.float64).ravel()


def one_se_rule(fold_rmse_matrix, *, max_components: int,
                n_folds: int) -> int:
    """Pick the most parsimonious n_components within 1 SE of the best
    CV-RMSE, given a (max_components, n_folds) RMSE matrix."""
    ctx = Context()
    cfg = _basic_cfg(int(max_components))
    try:
        rmse = np.ascontiguousarray(fold_rmse_matrix, dtype=np.float64)
        if rmse.ndim != 2:
            raise ValueError(
                f"fold_rmse_matrix must be 2-D (max_components, n_folds); "
                f"got ndim={rmse.ndim}")
        res = _methods.one_se_rule_compute(
            ctx, cfg, rmse,
            max_components=int(max_components),
            n_folds=int(n_folds))
    finally:
        cfg.close()
    return int(res.scalar("selected"))


__all__ = [
    "t2_score",
    "q_score",
    "dmodx_score",
    "pls_monitoring",
    "approximate_press",
    "one_se_rule",
]
