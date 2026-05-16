"""Shared persistence + predict path for MethodResult-based sklearn
regressors.

Many pls4all `*_fit` C entry points (sparse_simpls, di_pls, cppls,
ecr, mb_pls, mir_pls, n_pls, sparse_pls_da, …) return a
:class:`MethodResult` rather than a re-fittable :class:`Model` handle.
The MethodResult carries the regression coefficients plus the X/Y
mean used at fit time, which is everything we need to compute
predictions on new X.

The C ABI convention (confirmed against
``cpp/src/core/model.cpp::fill_prediction``) is that ``coefficients``
are already in the ORIGINAL X/Y space — ``x_scale`` was folded in at
fit time and must NOT be re-applied at predict time. Exact formula:

    Y_pred = (X - x_mean_) @ coef_.T + y_mean_

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
from ._base import _check_X_p4a, _validate_X_y_no_mutate


class _MethodResultRegressor(BaseEstimator, RegressorMixin):
    """Base for regressors backed by a MethodResult (not a Model)."""

    # Coefficient + preprocessing-mean keys read from the MethodResult.
    # We only read x_mean and y_mean: per the C ABI contract, coefficients
    # are already in the original X/Y space, so x_scale / y_scale are not
    # needed for prediction.
    _result_keys: dict[str, str] = {
        "coef": "coefficients",
        "x_mean": "x_mean",
        "y_mean": "y_mean",
    }

    # --- Subclass-provided ----------------------------------------------

    def _fit_method_result(self, ctx: Context, X: np.ndarray,
                            y: np.ndarray):
        """Return a MethodResult from the appropriate `_methods.X_fit`."""
        raise NotImplementedError

    # --- Common state extraction ----------------------------------------

    def _extract_state(self, result) -> None:
        """Pull coef + means from the MethodResult and store as numpy
        arrays. Subclasses with non-standard result schemas (e.g.
        MBPLSRegression, which exposes intercept directly) may
        override."""
        keys = self._result_keys
        coef = np.asarray(result.matrix(keys["coef"]), dtype=np.float64)
        # MethodResult stores coefficients (p, q); sklearn uses (q, p) or
        # (p,) when q == 1.
        coef_T = coef.T.copy()
        if coef_T.shape[0] == 1:
            coef_T = coef_T.reshape(coef_T.shape[1])
        x_mean = np.asarray(
            result.matrix(keys["x_mean"]), dtype=np.float64).ravel()
        y_mean = np.asarray(
            result.matrix(keys["y_mean"]), dtype=np.float64).ravel()
        # Compute sklearn-style intercept BEFORE binding so a partial
        # failure here leaves the previous fit intact.
        if coef_T.ndim == 1:
            intercept = float(y_mean[0] - x_mean @ coef_T)
        else:
            intercept = y_mean - x_mean @ coef_T.T
        # Commit everything together.
        self.coef_ = coef_T
        self.x_mean_ = x_mean
        self.y_mean_ = y_mean
        self.intercept_ = intercept

    # --- Sklearn estimator surface --------------------------------------

    def fit(self, X: Any, y: Any) -> "_MethodResultRegressor":
        # Validate inputs WITHOUT mutating fitted state — a failed fit
        # must leave the prior (or unfitted) state untouched.
        X_arr, y_arr, y_ndim_orig = _validate_X_y_no_mutate(X, y)
        ctx = Context()
        result = self._fit_method_result(ctx, X_arr, y_arr)
        # Commit fitted state only after every read from `result` succeeded.
        self._extract_state(result)
        self.n_features_in_ = int(X_arr.shape[1])
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self._y_ndim_ = y_ndim_orig
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self)
        X_arr = _check_X_p4a(self, X)
        # C ABI convention: coefficients are in original X/Y space; do NOT
        # divide by x_scale. See cpp/src/core/model.cpp::fill_prediction.
        Xc = X_arr - self.x_mean_
        if self.coef_.ndim == 1:
            preds = Xc @ self.coef_ + self.y_mean_[0]
        else:
            preds = Xc @ self.coef_.T + self.y_mean_
        # 1-D y in => 1-D preds out (sklearn convention).
        if getattr(self, "_y_ndim_", 2) == 1:
            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds.ravel()
        return preds
