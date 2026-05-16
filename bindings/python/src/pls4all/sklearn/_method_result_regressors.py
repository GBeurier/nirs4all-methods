"""MethodResult-based regressors: sparse SIMPLS, CPPLS, ECR, DI-PLS,
MIR-PLS, MB-PLS, N-PLS, and the sparse PLS-DA classifier.

These wrap C-side `*_fit` functions that return a :class:`MethodResult`
carrying the regression coefficients and X/Y preprocessing stats. The
common predict-on-new-X path is provided by
:class:`_MethodResultRegressor`; subclasses only declare which tier-1
function to call and (if non-default) which result keys to read.

Methods deliberately NOT wrapped here:

* ``weighted_pls_fit`` / ``robust_pls_fit`` / ``ridge_pls_fit`` /
  ``continuum_regression_fit`` — current C ABI exposes only
  predictions in the MethodResult, not coefficients, so we cannot
  predict on new X without a refit.
* ``recursive_pls_run`` / ``so_pls_fit`` / ``rosa_fit`` / ``lw_pls_fit``
  — return predictions only; locally-fitted models with no global
  coefficient export.
* ``bagging_pls_fit`` / ``boosting_pls_fit`` /
  ``random_subspace_pls_fit`` — ensembles without exposed per-member
  state; deferred until the ABI grows an export hook.
* ``group_sparse_pls_fit`` / ``fused_sparse_pls_fit`` /
  ``on_pls_fit`` / ``o2pls_fit`` — decomposition only; no
  predict-on-new-X path yet.
* ``missing_aware_nipals_fit`` / ``kernel_pls_fit`` — special preds
  paths (kernel needs alpha + kernel matrix); deferred to dedicated
  wrappers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import _methods
from .._context import Context
from ._method_result import _MethodResultRegressor


class SparseSimplsRegression(_MethodResultRegressor):
    """Sparse SIMPLS with soft-thresholded weights (Chun & Keles 2010).

    Same numerical kernel as :class:`SparsePLSRegression`, but wraps the
    dedicated ``sparse_simpls_fit`` C entry-point, which exposes weights
    + predictions alongside coefficients for downstream inspection.
    """

    def __init__(self, n_components: int = 2,
                  *, sparsity_lambda: float = 0.05) -> None:
        self.n_components = n_components
        self.sparsity_lambda = sparsity_lambda

    def _fit_method_result(self, ctx: Context, X, y):
        with _ManagedConfig(int(self.n_components)) as cfg:
            return _methods.sparse_simpls_fit(
                ctx, cfg, X, y,
                sparsity_lambda=float(self.sparsity_lambda))


class CPPLSRegression(_MethodResultRegressor):
    """Canonical Powered PLS (Indahl 2005)."""

    def __init__(self, n_components: int = 2, *, gamma: float = 0.5) -> None:
        self.n_components = n_components
        self.gamma = gamma

    def _fit_method_result(self, ctx, X, y):
        with _ManagedConfig(int(self.n_components)) as cfg:
            return _methods.cppls_fit(
                ctx, cfg, X, y, gamma=float(self.gamma))


class ECRegression(_MethodResultRegressor):
    """Elastic Component Regression (Liu 2013) — interpolates PCR (α=0)
    and PLS (α=1)."""

    def __init__(self, n_components: int = 2, *, alpha: float = 0.5) -> None:
        self.n_components = n_components
        self.alpha = alpha

    def _fit_method_result(self, ctx, X, y):
        with _ManagedConfig(int(self.n_components)) as cfg:
            return _methods.ecr_fit(
                ctx, cfg, X, y, alpha=float(self.alpha))


class DIPLSRegression(_MethodResultRegressor):
    """Domain-invariant PLS (Nikzad-Langerodi 2018).

    `fit` requires the target-domain X via the ``X_target`` keyword.
    Sklearn pattern: ``DIPLSRegression(...).fit(X_source, y_source,
    X_target=X_t)``.
    """

    def __init__(self, n_components: int = 2,
                  *, di_lambda: float = 1.0) -> None:
        self.n_components = n_components
        self.di_lambda = di_lambda

    def fit(self, X, y, X_target=None):
        if X_target is None:
            raise ValueError(
                "DIPLSRegression.fit requires a `X_target` keyword "
                "(target-domain feature matrix)"
            )
        # Defer to base; we need to plumb X_target through.
        self._X_target_ = np.ascontiguousarray(X_target, dtype=np.float64)
        return super().fit(X, y)

    def _fit_method_result(self, ctx, X, y):
        with _ManagedConfig(int(self.n_components)) as cfg:
            return _methods.di_pls_fit(
                ctx, cfg, X, y, self._X_target_,
                di_lambda=float(self.di_lambda))


class MIRPLSRegression(_MethodResultRegressor):
    """Multivariate Inverse Regression PLS (Sjöblom 1998).

    Coefficients are derived from the inverse regression form and then
    pseudo-inverted back; predicts via the standard X @ coef + intercept
    pattern.
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def _fit_method_result(self, ctx, X, y):
        with _ManagedConfig(int(self.n_components)) as cfg:
            return _methods.mir_pls_fit(ctx, cfg, X, y)


class MBPLSRegression(_MethodResultRegressor):
    """Multi-block PLS (Westerhuis 1998).

    Concatenates per-block X (block-balanced via x_scale_) and fits a
    SIMPLS regression. Block boundaries declared via ``block_sizes``.
    """

    # mb_pls_fit doesn't expose y_mean; it materializes intercept_ directly.
    # Override _extract_state to honour that.

    def __init__(self, n_components: int = 2, *, block_sizes=None) -> None:
        self.n_components = n_components
        self.block_sizes = block_sizes

    def fit(self, X, y):
        if not self.block_sizes:
            raise ValueError(
                "MBPLSRegression requires `block_sizes=[k1, k2, ...]` "
                "summing to X.shape[1]"
            )
        return super().fit(X, y)

    def _fit_method_result(self, ctx, X, y):
        block_sizes = np.asarray(self.block_sizes, dtype=np.int64).ravel()
        if int(block_sizes.sum()) != X.shape[1]:
            raise ValueError(
                f"block_sizes sum ({int(block_sizes.sum())}) must equal "
                f"n_features ({X.shape[1]})"
            )
        from .._types import Solver
        with _ManagedConfig(int(self.n_components)) as cfg:
            cfg.solver = Solver.NIPALS  # mb_pls_fit expects NIPALS
            return _methods.mb_pls_fit(ctx, cfg, X, y, block_sizes)

    def _extract_state(self, result) -> None:
        coef = np.asarray(result.matrix("coefficients"), dtype=np.float64)
        coef_T = coef.T.copy()
        if coef_T.shape[0] == 1:
            coef_T = coef_T.reshape(coef_T.shape[1])
        x_mean = np.asarray(
            result.matrix("x_mean"), dtype=np.float64).ravel()
        x_scale = np.asarray(
            result.matrix("x_scale"), dtype=np.float64).ravel()
        intercept_arr = np.asarray(
            result.matrix("intercept"), dtype=np.float64).ravel()
        block_weights = np.asarray(
            result.matrix("block_weights"), dtype=np.float64).ravel()
        # Commit only after every read succeeded.
        self.coef_ = coef_T
        self.x_mean_ = x_mean
        self.x_scale_ = x_scale
        self.intercept_ = (
            float(intercept_arr[0]) if intercept_arr.size == 1
            else intercept_arr
        )
        self.block_weights_ = block_weights

    def predict(self, X):
        from sklearn.utils.validation import check_is_fitted
        from ._base import _check_X_p4a
        check_is_fitted(self)
        X_arr = _check_X_p4a(self, X)
        # MB-PLS stores coefficients in the ORIGINAL (un-centered,
        # un-scaled) X space and a separate intercept that folds in
        # y_mean - x_mean @ coef. So predict is X @ coef.T + intercept
        # (no centering / scaling at predict time).
        if self.coef_.ndim == 1:
            preds = X_arr @ self.coef_ + self.intercept_
        else:
            preds = X_arr @ self.coef_.T + self.intercept_
        if (getattr(self, "_y_ndim_", 2) == 1 and preds.ndim == 2
                and preds.shape[1] == 1):
            preds = preds.ravel()
        return preds


# ---- helper ----------------------------------------------------------------

class _ManagedConfig:
    """Tiny context manager so subclasses can build a Config with the
    standard pls4all-regression defaults and have it freed deterministically
    when the `_fit_method_result` block exits."""

    def __init__(self, n_components: int) -> None:
        self._n_components = int(n_components)
        self._cfg = None

    def __enter__(self):
        from .._config import Config
        from .._types import Algorithm, Deflation, Solver
        cfg = Config()
        cfg.algorithm = Algorithm.PLS_REGRESSION
        cfg.solver = Solver.SIMPLS
        cfg.deflation = Deflation.REGRESSION
        cfg.n_components = self._n_components
        cfg.center_x = True
        cfg.scale_x = False
        cfg.center_y = True
        cfg.scale_y = False
        cfg.tol = 1e-6
        cfg.max_iter = 500
        self._cfg = cfg
        return cfg

    def __exit__(self, *exc):
        if self._cfg is not None:
            self._cfg.close()
            self._cfg = None


def _make_basic_config(n_components: int):
    """Legacy direct-use helper. Prefer ``_ManagedConfig(n_components)``
    inside a ``with`` block so the Config is freed deterministically."""
    return _ManagedConfig(n_components).__enter__()


__all__ = [
    "SparseSimplsRegression",
    "CPPLSRegression",
    "ECRegression",
    "DIPLSRegression",
    "MIRPLSRegression",
    "MBPLSRegression",
]
