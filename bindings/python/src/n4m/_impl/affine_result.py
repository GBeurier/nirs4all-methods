# SPDX-License-Identifier: CECILL-2.1
"""Shared sklearn prediction path for native affine MethodResult regressors."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from n4m._errors import check
from n4m._ffi import lib
from n4m._matrix import numpy_to_view

from .compat import BaseEstimator, RegressorMixin


class AffineMethodResultRegressor(RegressorMixin, BaseEstimator):
    """Fit through a native MethodResult and predict from its exported state.

    Subclasses supply ``_fit_native`` and method-specific argument validation.
    Native coefficients are in original X/Y units with shape (features, targets).
    Fitted models are stored as native N4MM bytes so prediction, export, pickle,
    and process boundaries all use the same C ABI model contract.
    """

    def _fit_native(self, X: Any, y: Any) -> dict[str, np.ndarray | float | bytes]:
        raise NotImplementedError

    def fit(self, X: Any, y: Any) -> AffineMethodResultRegressor:
        y_ndim = np.asarray(y).ndim
        result = self._fit_native(X, y)
        coef = np.asarray(result["coefficients"], dtype=np.float64)
        x_mean = np.asarray(result["x_mean"], dtype=np.float64).ravel()
        y_mean = np.asarray(result["y_mean"], dtype=np.float64).ravel()
        intercept = np.asarray(result["intercept"], dtype=np.float64).ravel()
        bundle = result["model_bundle"]
        if coef.ndim != 2 or coef.shape != (x_mean.size, y_mean.size):
            raise RuntimeError("native method returned incompatible affine state")
        if intercept.shape != y_mean.shape or not isinstance(bundle, bytes):
            raise RuntimeError("native method returned incompatible model state")
        self.coef_ = coef[:, 0].copy() if coef.shape[1] == 1 else coef.T.copy()
        self.x_mean_ = x_mean
        self.y_mean_ = y_mean
        self.intercept_ = float(intercept[0]) if y_ndim == 1 else intercept
        self._bundle_ = bundle
        self.n_features_in_ = x_mean.size
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self._y_ndim_ = y_ndim
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError(f"{type(self).__name__} must be fitted before predict")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features")
        if not np.isfinite(X_arr).all():
            raise ValueError("X must be finite")
        X_contiguous = np.ascontiguousarray(X_arr)
        predicted = np.empty((X_contiguous.shape[0], self.y_mean_.size), dtype=np.float64)
        ctx = ctypes.c_void_p()
        model = ctypes.c_void_p()
        bundle = ctypes.create_string_buffer(self._bundle_)
        try:
            check(lib.n4m_context_create(ctypes.byref(ctx)), "n4m_context_create")
            check(lib.n4m_model_import_from_buffer(
                ctx, bundle, len(self._bundle_), ctypes.byref(model)),
                "n4m_model_import_from_buffer")
            x_view = numpy_to_view(X_contiguous)
            out_view = numpy_to_view(predicted)
            check(lib.n4m_model_predict(
                ctx, model, ctypes.byref(x_view), ctypes.byref(out_view)),
                "n4m_model_predict")
        finally:
            if model.value:
                lib.n4m_model_destroy(model)
            if ctx.value:
                lib.n4m_context_destroy(ctx)
        return predicted.ravel() if self._y_ndim_ == 1 else predicted

    def export_n4mm(self) -> bytes:
        """Return the native predict-only N4MM model payload."""
        if not hasattr(self, "_bundle_"):
            raise RuntimeError(f"{type(self).__name__} must be fitted before export")
        return self._bundle_


__all__ = ["AffineMethodResultRegressor"]
