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

import numpy as np

from .. import _methods
from .._aom import OperatorBank, OperatorKind
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


def _t2_q_dmodx(model, X, key: str, X_reference=None) -> np.ndarray:
    """Compute T², Q or DModX via the C diagnostics kernel."""
    if not isinstance(model, Model):
        raise TypeError(
            "diagnostic functions require a pls4all.Model instance "
            f"(got {type(model).__name__})")
    ctx = Context()
    X_arr = np.ascontiguousarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={X_arr.ndim}")
    res = _methods.pls_diagnostics_compute(
        ctx, model, X_arr, X_reference=X_reference)
    try:
        return np.asarray(res.matrix(key), dtype=np.float64).ravel()
    finally:
        res.close()


def t2_score(model, X, X_reference=None) -> np.ndarray:
    """Hotelling T² per row of X under a fitted pls4all Model."""
    return _t2_q_dmodx(model, X, "t2", X_reference=X_reference)


def q_score(model, X, X_reference=None) -> np.ndarray:
    """Q residual / squared prediction error per row of X."""
    return _t2_q_dmodx(model, X, "q", X_reference=X_reference)


def dmodx_score(model, X, X_reference=None) -> np.ndarray:
    """Distance-to-Model-X per row of X."""
    return _t2_q_dmodx(model, X, "dmodx", X_reference=X_reference)


def pls_monitoring(model, X_reference, X_monitor, *, alpha: float = 0.95):
    """Run process monitoring on a fitted model: returns T²/Q time
    series + alarm flags for X_monitor against the X_reference baseline.

    ``alpha`` follows the sklearn / chemometrics convention (0.95 ⇒
    upper 5 % tail flagged as anomalous). Internally the C kernel uses
    ``1 - alpha`` as the alarm threshold (see
    ``cpp/src/core/pls_monitoring.cpp::pls_monitoring_run``), so the
    conversion happens here.
    """
    if not isinstance(model, Model):
        raise TypeError(
            f"pls_monitoring needs a pls4all.Model (got {type(model).__name__})")
    ctx = Context()
    Xr = np.ascontiguousarray(X_reference, dtype=np.float64)
    Xm = np.ascontiguousarray(X_monitor, dtype=np.float64)
    # C-side expects the tail probability (1 - alpha).
    res = _methods.pls_monitoring_run(
        ctx, model, X_reference=Xr, X_monitor=Xm,
        alpha=float(1.0 - alpha))
    # Result schema (see cpp/src/c_api/c_api_method_result.cpp:1062):
    # double matrices: "t2", "q"
    # int32 vectors: "t2_alarms", "q_alarms", "any_alarms"
    try:
        return {
            "t2": np.asarray(res.matrix("t2"), dtype=np.float64).ravel(),
            "q": np.asarray(res.matrix("q"), dtype=np.float64).ravel(),
            "t2_alarms": np.asarray(
                res.vector_int("t2_alarms"), dtype=np.int32),
            "q_alarms": np.asarray(
                res.vector_int("q_alarms"), dtype=np.int32),
            "any_alarms": np.asarray(
                res.vector_int("any_alarms"), dtype=np.int32),
        }
    finally:
        res.close()


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
    try:
        return np.asarray(
            res.matrix("press_per_component"), dtype=np.float64).ravel()
    finally:
        res.close()


def one_se_rule(fold_rmse_matrix, *, max_components: int,
                n_folds: int, return_curve: bool = False):
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
            ctx, rmse,
            max_components=int(max_components),
            n_folds=int(n_folds))
    finally:
        cfg.close()
    try:
        selected = int(res.vector_int("one_se_n_components")[0])
        if return_curve:
            return {
                "selected": selected,
                "mean_rmse_per_component": np.asarray(
                    res.matrix("mean_rmse_per_component"),
                    dtype=np.float64).ravel(),
            }
        return selected
    finally:
        res.close()


def aom_preprocess(X, y=None, *, n_operators: int = 3,
                   gating_mode: int = 0) -> np.ndarray:
    """Run the canonical AOM preprocessing bank and return transformed X."""
    import ctypes
    from .._ffi import lib

    if not hasattr(lib.n4m_gating_strategy_create, "restype") or \
            lib.n4m_gating_strategy_create.restype is None:
        lib.n4m_gating_strategy_create.restype = ctypes.c_int
        lib.n4m_gating_strategy_create.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        lib.n4m_gating_strategy_destroy.restype = None
        lib.n4m_gating_strategy_destroy.argtypes = [ctypes.c_void_p]

    bank = OperatorBank()
    for kind in [
        OperatorKind.IDENTITY,
        OperatorKind.CENTER,
        OperatorKind.PARETO_SCALE,
        OperatorKind.AUTOSCALE,
        OperatorKind.SNV,
    ][:int(n_operators)]:
        bank.add(kind)

    gate_handle = ctypes.c_void_p(0)
    status = lib.n4m_gating_strategy_create(
        ctypes.byref(gate_handle), int(gating_mode))
    if status != 0 or not gate_handle:
        bank.close()
        raise RuntimeError(
            f"n4m_gating_strategy_create failed (status={status})")
    ctx = Context()
    try:
        Y = None if y is None else np.ascontiguousarray(y, dtype=np.float64)
        if Y is not None and Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        res = _methods.aom_preprocess_fit(
            ctx, bank, gate_handle,
            np.ascontiguousarray(X, dtype=np.float64), Y)
        try:
            return np.asarray(res.matrix("transformed"), dtype=np.float64)
        finally:
            res.close()
    finally:
        lib.n4m_gating_strategy_destroy(gate_handle)
        bank.close()


def _split_blocks(X: np.ndarray, n_blocks: int) -> list[np.ndarray]:
    nb = max(1, min(int(n_blocks), int(X.shape[1])))
    base = X.shape[1] // nb
    blocks = []
    cursor = 0
    for block in range(nb):
        width = base if block < nb - 1 else X.shape[1] - cursor
        blocks.append(np.ascontiguousarray(X[:, cursor:cursor + width],
                                           dtype=np.float64))
        cursor += width
    return blocks


def on_pls(X, *, n_blocks: int, n_joint: int,
           n_unique_per_block) -> dict[str, np.ndarray]:
    """Run ON-PLS decomposition and return exported loading/score matrices."""
    ctx = Context()
    cfg = Config()
    cfg.algorithm = Algorithm.PLS_REGRESSION
    cfg.solver = Solver.NIPALS
    cfg.deflation = Deflation.REGRESSION
    cfg.center_x = True
    cfg.center_y = True
    blocks = _split_blocks(np.ascontiguousarray(X, dtype=np.float64), n_blocks)
    try:
        res = _methods.on_pls_fit(
            ctx, cfg, blocks, int(n_joint),
            np.ascontiguousarray(n_unique_per_block, dtype=np.int32))
    finally:
        cfg.close()
    try:
        out = {}
        for block in range(len(blocks)):
            for prefix in ("joint_loadings", "unique_loadings",
                            "joint_scores", "block_reconstruction"):
                key = f"{prefix}_{block}"
                try:
                    out[key] = np.asarray(res.matrix(key), dtype=np.float64)
                except Exception:
                    pass
        return out
    finally:
        res.close()


__all__ = [
    "t2_score",
    "q_score",
    "dmodx_score",
    "pls_monitoring",
    "approximate_press",
    "one_se_rule",
    "aom_preprocess",
    "on_pls",
]
