"""sklearn-compatible PLS-DA classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from .._context import Context
from .._model import Model, ModelArrayKind
from .._types import Algorithm, Deflation
from ._base import (
    _Pls4allModelEstimator,
    _check_X_p4a,
    _config_from_params,
    _resolve_solver,
)
from sklearn.utils.validation import check_X_y


class PLSDAClassifier(_Pls4allModelEstimator, BaseEstimator, ClassifierMixin):
    """Partial Least Squares Discriminant Analysis (PLS-DA).

    Fits PLS regression on one-hot-encoded class indicators (sklearn
    convention) and decodes the highest column as the predicted class.
    Backed by `Algorithm.PLS_DA` in the C ABI, so the fitted state
    survives `pickle.dumps` bit-exactly via the `.n4a` bundle.

    Currently single-output multiclass only. Multi-label classification
    is not supported.

    Parameters
    ----------
    n_components : int, default=2
    solver : str, default='simpls'
    center_x, scale_x : bool, default=True
    tol : float, default=1e-6
    max_iter : int, default=500
    """

    _algorithm = Algorithm.PLS_DA

    def __init__(self,
                  n_components: int = 2,
                  *,
                  solver: str = "simpls",
                  center_x: bool = True,
                  scale_x: bool = True,
                  tol: float = 1e-6,
                  max_iter: int = 500) -> None:
        self.n_components = n_components
        self.solver = solver
        self.center_x = center_x
        self.scale_x = scale_x
        self.tol = tol
        self.max_iter = max_iter

    def _make_config(self):
        return _config_from_params(
            n_components=self.n_components,
            algorithm=self._algorithm,
            solver=_resolve_solver(self.solver),
            deflation=Deflation.REGRESSION,
            center_x=self.center_x,
            scale_x=self.scale_x,
            # Always y-center for PLS-DA so the dummy means cancel.
            center_y=True,
            scale_y=False,
            tol=self.tol,
            max_iter=self.max_iter,
            store_scores=False,
        )

    @staticmethod
    def _one_hot(y_int: np.ndarray, n_classes: int) -> np.ndarray:
        out = np.zeros((y_int.shape[0], n_classes), dtype=np.float64)
        out[np.arange(y_int.shape[0]), y_int] = 1.0
        return out

    def fit(self, X: Any, y: Any) -> "PLSDAClassifier":
        # Validate inputs WITHOUT mutating fitted state, so a failed
        # fit leaves the estimator in its prior (or unfitted) state.
        X_chk, y_chk = check_X_y(
            X, y,
            dtype=np.float64, ensure_2d=True,
            ensure_all_finite=True, multi_output=False, y_numeric=False,
        )
        y_arr = np.asarray(y_chk).ravel()
        encoder = LabelEncoder()
        y_int = encoder.fit_transform(y_arr)
        classes = np.asarray(encoder.classes_)
        if classes.size < 2:
            raise ValueError(
                f"PLSDAClassifier requires at least 2 classes; got "
                f"{classes.size}"
            )
        X_arr = np.ascontiguousarray(X_chk, dtype=np.float64)
        Y_dummy = self._one_hot(y_int, classes.size)
        ctx = Context()
        cfg = self._make_config()
        try:
            model = Model.fit(ctx, cfg, X_arr, Y_dummy)
        finally:
            cfg.close()
        # All checks passed; now commit fitted state.
        self._drop_handle()
        self._label_encoder_ = encoder
        self.classes_ = classes
        self.n_features_in_ = int(X_arr.shape[1])
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.coef_ = model.get_array(ctx, ModelArrayKind.COEFFICIENTS).T
        self._store_fitted_model(ctx, model)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self)
        X_arr = _check_X_p4a(self, X)
        ctx, model = self._ensure_model_handle()
        return model.predict(ctx, X_arr)

    def predict(self, X: Any) -> np.ndarray:
        scores = self.decision_function(X)
        if scores.ndim == 1:
            # Binary one-column dummy edge case — collapse via 0.5 cut.
            int_pred = (scores >= 0.5).astype(np.int64)
        else:
            int_pred = np.argmax(scores, axis=1)
        return self._label_encoder_.inverse_transform(int_pred)


class OPLSDAClassifier(PLSDAClassifier):
    """Orthogonal PLS Discriminant Analysis (Trygg & Wold 2002).

    Same one-hot-Y + argmax-decode contract as :class:`PLSDAClassifier`,
    but uses ``Algorithm.OPLS_DA + Deflation.ORTHOGONAL`` for the
    underlying decomposition. Predictive vs orthogonal components are
    separated; classification boundary lives in the predictive subspace.
    """

    _algorithm = Algorithm.OPLS_DA

    def __init__(self,
                  n_components: int = 2,
                  *,
                  solver: str = "nipals",
                  center_x: bool = True,
                  scale_x: bool = True,
                  tol: float = 1e-6,
                  max_iter: int = 500) -> None:
        super().__init__(
            n_components=n_components,
            solver=solver,
            center_x=center_x,
            scale_x=scale_x,
            tol=tol,
            max_iter=max_iter,
        )

    def _make_config(self):
        cfg = super()._make_config()
        cfg.deflation = Deflation.ORTHOGONAL
        return cfg
