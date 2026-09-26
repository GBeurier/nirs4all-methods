# SPDX-License-Identifier: CECILL-2.1
"""Sklearn-compatible domain-invariant PLS over the native C ABI."""

from __future__ import annotations

import numpy as np

from . import native
from .compat import BaseEstimator, RegressorMixin


class DIPLS(RegressorMixin, BaseEstimator):
    """Fit DI-PLS using labeled source and unlabeled target spectra.

    Supply ``X_target`` explicitly to every ``fit`` call. The target cohort
    affects the fitted weights; it is *not* an independent validation set.
    Cross-validation callers must define the target cohort for each fold and
    avoid using held-out source samples as target-domain training data.

    The fitted predictor stores only native coefficients and source means,
    not the target cohort. This is a reusable affine predictor, not a
    cross-language trained-pipeline recipe for reproducing the fit.
    """

    def __init__(self, n_components: int = 2, *, di_lambda: float = 1.0) -> None:
        self.n_components = n_components
        self.di_lambda = di_lambda

    def fit(self, X, y, *, X_target):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("X_source must be a 2-D matrix")
        if y_arr.ndim not in (1, 2) or y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("y_source must have one response per source row")
        result = native.di_pls(
            X,
            y,
            X_target=X_target,
            n_components=self.n_components,
            di_lambda=self.di_lambda,
        )
        coefficients = np.asarray(result["coefficients"], dtype=np.float64)
        x_mean = np.asarray(result["x_mean"], dtype=np.float64).reshape(-1)
        y_mean = np.asarray(result["y_mean"], dtype=np.float64).reshape(-1)
        if coefficients.shape != (X_arr.shape[1], 1):
            raise ValueError("native DI-PLS returned incompatible coefficients")
        if x_mean.size != X_arr.shape[1] or y_mean.size != 1:
            raise ValueError("native DI-PLS returned incompatible centering state")
        # Commit only after the ABI result and every shape have been checked.
        self.coef_ = coefficients[:, 0].copy()
        self.x_mean_ = x_mean.copy()
        self.y_mean_ = float(y_mean[0])
        self.intercept_ = self.y_mean_ - float(self.x_mean_ @ self.coef_)
        self.n_features_in_ = X_arr.shape[1]
        self._y_was_1d_ = y_arr.ndim == 1
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("DIPLS.predict called before fit")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count than during fit")
        if not np.isfinite(X_arr).all():
            raise ValueError("X must contain only finite values")
        if (
            hasattr(self, "feature_names_in_")
            and hasattr(X, "columns")
            and tuple(X.columns) != tuple(self.feature_names_in_)
        ):
            raise ValueError("X feature names and order differ from fit")
        predictions = X_arr @ self.coef_ + self.intercept_
        return predictions if self._y_was_1d_ else predictions.reshape(-1, 1)


__all__ = ["DIPLS"]
