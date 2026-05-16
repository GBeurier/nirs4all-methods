"""sklearn-compatible PLS / OPLS regression wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .._context import Context
from .._model import Model, ModelArrayKind
from .._types import Algorithm, Deflation
from ._base import (
    _Pls4allModelEstimator,
    _as_2d_float64,
    _check_X_p4a,
    _check_X_y_p4a,
    _config_from_params,
    _resolve_solver,
)


class _PlsRegressorBase(_Pls4allModelEstimator, BaseEstimator, RegressorMixin):
    """Shared `fit`/`predict`/`transform` plumbing for Model-based PLS
    regressors. Concrete subclasses override `_algorithm` / `_default_solver`
    and may extend ``_fitted_attributes``."""

    _algorithm: Algorithm
    _default_solver: str
    _deflation: Deflation = Deflation.REGRESSION

    # Hyperparameters mirrored from the C `Config`. Init signature is
    # explicit on each subclass so `get_params` returns the documented
    # name set (sklearn rebuilds get_params from __init__'s signature).

    def _make_config(self):
        return _config_from_params(
            n_components=self.n_components,
            algorithm=self._algorithm,
            solver=_resolve_solver(self.solver),
            deflation=self._deflation,
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=self.center_y,
            scale_y=self.scale_y,
            tol=self.tol,
            max_iter=self.max_iter,
            store_scores=self.store_scores,
        )

    def fit(self, X: Any, y: Any) -> "_PlsRegressorBase":
        X_arr, y_arr, self._y_ndim_ = _check_X_y_p4a(self, X, y)
        self._drop_handle()
        ctx = Context()
        cfg = self._make_config()
        try:
            model = Model.fit(ctx, cfg, X_arr, y_arr)
        finally:
            cfg.close()
        # sklearn-compatible fitted attributes (trailing underscore).
        self.coef_ = model.get_array(ctx, ModelArrayKind.COEFFICIENTS).T
        self.x_mean_ = model.get_array(
            ctx, ModelArrayKind.X_MEAN).ravel()
        self.y_mean_ = model.get_array(
            ctx, ModelArrayKind.Y_MEAN).ravel()
        if self.scale_x:
            self.x_std_ = model.get_array(
                ctx, ModelArrayKind.X_SCALE).ravel()
        if self.scale_y:
            self.y_std_ = model.get_array(
                ctx, ModelArrayKind.Y_SCALE).ravel()
        # sklearn naming: intercept_ is (n_targets,)
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_.T
        if self.store_scores:
            self.x_scores_ = model.get_array(
                ctx, ModelArrayKind.SCORES_T)
        self._store_fitted_model(ctx, model)
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self)
        X_arr = _check_X_p4a(self, X)
        ctx, model = self._ensure_model_handle()
        preds = model.predict(ctx, X_arr)
        # sklearn convention: 1-D y in → 1-D predictions out.
        if getattr(self, "_y_ndim_", 2) == 1 and preds.shape[1] == 1:
            preds = preds.ravel()
        return preds

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self)
        X_arr = _check_X_p4a(self, X)
        ctx, model = self._ensure_model_handle()
        return model.transform(ctx, X_arr)


class PLSRegression(_PlsRegressorBase):
    """Partial Least Squares regression backed by `pls4all`'s C core.

    Drop-in replacement for `sklearn.cross_decomposition.PLSRegression`
    with two distinguishing knobs:

    * `solver` selects the inner algorithm (NIPALS, SIMPLS, SVD, …)
      directly — sklearn only exposes 'nipals' / 'svd'.
    * Round-trip via `pickle.dumps` is bit-exact, backed by the C ABI
      `.n4a` bundle (`p4a_model_export_to_buffer`).

    Parameters
    ----------
    n_components : int, default=2
        Number of latent components.
    solver : str, default='nipals'
        One of 'nipals', 'simpls', 'orthogonal-scores', 'kernel',
        'wide-kernel', 'svd', 'power', 'randomized-svd'.
    center_x, scale_x : bool, default=True
        Standardize X columns to zero mean / unit variance.
    center_y : bool, default=True
        Center y to zero mean.
    scale_y : bool, default=False
        Standardize y columns to unit variance.
    tol : float, default=1e-6
        Convergence tolerance for iterative solvers.
    max_iter : int, default=500
        Max NIPALS iterations.
    store_scores : bool, default=False
        Keep the latent score matrices (`x_scores_`) after fit.
    """

    _algorithm = Algorithm.PLS_REGRESSION
    _default_solver = "nipals"

    def __init__(self,
                  n_components: int = 2,
                  *,
                  solver: str = "nipals",
                  center_x: bool = True,
                  scale_x: bool = True,
                  center_y: bool = True,
                  scale_y: bool = False,
                  tol: float = 1e-6,
                  max_iter: int = 500,
                  store_scores: bool = False) -> None:
        self.n_components = n_components
        self.solver = solver
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y
        self.tol = tol
        self.max_iter = max_iter
        self.store_scores = store_scores


class OPLSRegression(_PlsRegressorBase):
    """Orthogonal PLS regression (Trygg & Wold 2002).

    Same surface as `PLSRegression`; uses the OPLS solver path under
    the hood, which separates predictive vs orthogonal latent
    components. The first score column is predictive; later columns
    are orthogonal corrections.
    """

    _algorithm = Algorithm.OPLS
    _default_solver = "nipals"
    _deflation = Deflation.ORTHOGONAL

    def __init__(self,
                  n_components: int = 2,
                  *,
                  solver: str = "nipals",
                  center_x: bool = True,
                  scale_x: bool = True,
                  center_y: bool = True,
                  scale_y: bool = False,
                  tol: float = 1e-6,
                  max_iter: int = 500,
                  store_scores: bool = False) -> None:
        self.n_components = n_components
        self.solver = solver
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y
        self.tol = tol
        self.max_iter = max_iter
        self.store_scores = store_scores


class PCR(_PlsRegressorBase):
    """Principal Components Regression — fits a least-squares regression
    on the SVD of X.

    PCR ≡ ``Algorithm.PCR`` + ``Solver.SVD``. Functionally similar to
    sklearn's ``Pipeline([PCA(k), LinearRegression()])`` but with the
    advantage that pls4all centers / scales in lock-step with the
    inverse transform, so coefficient inspection matches the original
    feature space.
    """

    _algorithm = Algorithm.PCR
    _default_solver = "svd"

    def __init__(self,
                  n_components: int = 2,
                  *,
                  center_x: bool = True,
                  scale_x: bool = True,
                  center_y: bool = True,
                  scale_y: bool = False,
                  tol: float = 1e-6,
                  max_iter: int = 500,
                  store_scores: bool = False) -> None:
        self.n_components = n_components
        # PCR is always SVD-solved — no other solver makes sense; we keep
        # the attribute so the base class can pass it through unchanged.
        self.solver = "svd"
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y
        self.tol = tol
        self.max_iter = max_iter
        self.store_scores = store_scores


class SparsePLSRegression(_PlsRegressorBase):
    """Sparse PLS regression with soft-thresholded weights (Chun & Keles
    2010-style L1 weight shrinkage during SIMPLS).

    ``sparsity_lambda`` controls the soft-threshold magnitude applied to
    each weight vector after the SIMPLS deflation; ``lambda=0`` recovers
    standard SIMPLS. Backed by ``Algorithm.SPARSE_PLS``.
    """

    _algorithm = Algorithm.SPARSE_PLS
    _default_solver = "simpls"

    def __init__(self,
                  n_components: int = 2,
                  *,
                  sparsity_lambda: float = 0.05,
                  solver: str = "simpls",
                  center_x: bool = True,
                  scale_x: bool = True,
                  center_y: bool = True,
                  scale_y: bool = False,
                  tol: float = 1e-6,
                  max_iter: int = 500,
                  store_scores: bool = False) -> None:
        self.n_components = n_components
        self.sparsity_lambda = sparsity_lambda
        self.solver = solver
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y
        self.tol = tol
        self.max_iter = max_iter
        self.store_scores = store_scores

    def _make_config(self):
        cfg = super()._make_config()
        cfg.sparsity_lambda = float(self.sparsity_lambda)
        return cfg


class PLSCanonical(_PlsRegressorBase):
    """PLS Canonical (NIPALS- or SVD-deflated) — symmetric two-block PLS
    where the latent rotations of X and Y are extracted jointly.

    Same surface as :class:`PLSRegression` but with
    ``Deflation.CANONICAL``. Number of components is bounded above by
    ``min(n_features, n_targets)``.
    """

    _algorithm = Algorithm.PLS_CANONICAL
    _default_solver = "nipals"
    _deflation = Deflation.CANONICAL

    def __init__(self,
                  n_components: int = 2,
                  *,
                  solver: str = "nipals",
                  center_x: bool = True,
                  scale_x: bool = True,
                  center_y: bool = True,
                  scale_y: bool = True,
                  tol: float = 1e-6,
                  max_iter: int = 500,
                  store_scores: bool = False) -> None:
        self.n_components = n_components
        self.solver = solver
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y
        self.tol = tol
        self.max_iter = max_iter
        self.store_scores = store_scores


class PLSSVD(_PlsRegressorBase):
    """PLS via direct SVD of the cross-covariance matrix
    (``Algorithm.PLS_SVD``).

    Same shape contract as :class:`PLSCanonical` (k ≤ min(p, q)) but
    derives the latent rotations from a single SVD of ``X^T Y`` without
    NIPALS-style deflation.
    """

    _algorithm = Algorithm.PLS_SVD
    _default_solver = "svd"
    _deflation = Deflation.CANONICAL

    def __init__(self,
                  n_components: int = 2,
                  *,
                  center_x: bool = True,
                  scale_x: bool = True,
                  center_y: bool = True,
                  scale_y: bool = True,
                  tol: float = 1e-6,
                  max_iter: int = 500,
                  store_scores: bool = False) -> None:
        self.n_components = n_components
        self.solver = "svd"
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y
        self.tol = tol
        self.max_iter = max_iter
        self.store_scores = store_scores
