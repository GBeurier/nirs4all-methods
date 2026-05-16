"""Shared persistence + predict path for MethodResult-based sklearn
regressors.

Many pls4all `*_fit` C entry points (sparse_simpls, di_pls, cppls,
ecr, mb_pls, mir_pls, n_pls, sparse_pls_da, …) return a
:class:`MethodResult` rather than a re-fittable :class:`Model` handle.
The MethodResult carries the regression coefficients plus the X/Y
mean/scale used at fit time, which is everything we need to compute
predictions on new X via plain NumPy linear algebra:

    Y_pred = (X - x_mean_) / x_scale_ @ coef_.T * y_scale_ + y_mean_

This mixin captures that pattern so every concrete subclass only has to
implement ``_fit_method_result(ctx, cfg, X, y)`` and (optionally)
override ``_extract_state`` for non-standard result keys.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .._context import Context
from ._base import _check_X_p4a, _check_X_y_p4a


class _MethodResultRegressor(BaseEstimator, RegressorMixin):
    """Base for regressors backed by a MethodResult (not a Model)."""

    _result_keys: dict[str, str] = {
        "coef": "coefficients",
        "x_mean": "x_mean",
        "y_mean": "y_mean",
        "x_scale": None,    # set to "x_scale" in subclasses that scaled X
        "y_scale": None,    # set to "y_scale" in subclasses that scaled Y
    }

    # --- Subclass-provided ----------------------------------------------

    def _fit_method_result(self, ctx: Context, X: np.ndarray,
                            y: np.ndarray):
        """Return a MethodResult from the appropriate `_methods.X_fit`."""
        raise NotImplementedError

    # --- Common state extraction ----------------------------------------

    def _extract_state(self, result) -> None:
        """Pull coef + means + scales from the MethodResult and store as
        numpy arrays. Subclasses with non-standard result schemas may
        override."""
        keys = self._result_keys
        coef = np.asarray(result.matrix(keys["coef"]), dtype=np.float64)
        # MethodResult stores coefficients (p, q); sklearn uses (q, p).
        self.coef_ = coef.T.copy()
        if self.coef_.shape[0] == 1:
            # Single-target convenience: 1-D coef
            self.coef_ = self.coef_.reshape(self.coef_.shape[1])
        x_mean = np.asarray(result.matrix(keys["x_mean"]),
                              dtype=np.float64).ravel()
        y_mean = np.asarray(result.matrix(keys["y_mean"]),
                              dtype=np.float64).ravel()
        self.x_mean_ = x_mean
        self.y_mean_ = y_mean
        self.x_scale_ = None
        self.y_scale_ = None
        if keys.get("x_scale"):
            try:
                self.x_scale_ = np.asarray(
                    result.matrix(keys["x_scale"]),
                    dtype=np.float64).ravel()
            except Exception:
                self.x_scale_ = None
        if keys.get("y_scale"):
            try:
                self.y_scale_ = np.asarray(
                    result.matrix(keys["y_scale"]),
                    dtype=np.float64).ravel()
            except Exception:
                self.y_scale_ = None
        # intercept_ = y_mean - x_mean @ coef.T (sklearn convention)
        if self.coef_.ndim == 1:
            self.intercept_ = float(self.y_mean_[0]
                                       - x_mean @ self.coef_)
        else:
            self.intercept_ = self.y_mean_ - x_mean @ self.coef_.T

    # --- Sklearn estimator surface --------------------------------------

    def fit(self, X: Any, y: Any) -> "_MethodResultRegressor":
        X_arr, y_arr, self._y_ndim_ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        try:
            result = self._fit_method_result(ctx, X_arr, y_arr)
        finally:
            pass
        self._extract_state(result)
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self)
        X_arr = _check_X_p4a(self, X)
        if self.x_scale_ is None:
            Xc = X_arr - self.x_mean_
        else:
            Xc = (X_arr - self.x_mean_) / self.x_scale_
        if self.coef_.ndim == 1:
            preds = Xc @ self.coef_ + self.y_mean_[0]
        else:
            preds = Xc @ self.coef_.T + self.y_mean_
        if self.y_scale_ is not None:
            preds = preds * self.y_scale_ + (
                self.y_mean_ - self.y_scale_ * self.y_mean_)
        # 1-D y in => 1-D preds out (sklearn convention).
        if getattr(self, "_y_ndim_", 2) == 1:
            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds.ravel()
        return preds
