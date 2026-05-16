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


class NPLSRegression(_MethodResultRegressor):
    """N-PLS (3-way tensor) regression (Bro 1996).

    Input ``X`` is supplied as a flattened (n, j*k) row-major matrix; the
    fit signature requires the ``mode_j`` and ``mode_k`` extents so the C
    kernel knows how to refold. Predict-on-new-X uses the standard
    coefficient path.
    """

    def __init__(self, n_components: int = 2,
                  *, mode_j: int, mode_k: int) -> None:
        self.n_components = n_components
        self.mode_j = mode_j
        self.mode_k = mode_k

    def _fit_method_result(self, ctx, X, y):
        with _ManagedConfig(int(self.n_components)) as cfg:
            return _methods.n_pls_fit(
                ctx, cfg,
                X_flat=X,
                mode_j=int(self.mode_j),
                mode_k=int(self.mode_k),
                Y=y)


class O2PLSRegression(_MethodResultRegressor):
    """O2-PLS (bi-directional OPLS, Trygg & Wold 2003)."""

    def __init__(self, n_predictive: int = 2,
                  *, n_x_orthogonal: int = 1,
                  n_y_orthogonal: int = 1) -> None:
        self.n_predictive = n_predictive
        self.n_x_orthogonal = n_x_orthogonal
        self.n_y_orthogonal = n_y_orthogonal

    @property
    def n_components(self) -> int:  # for _ManagedConfig
        return int(self.n_predictive)

    def _fit_method_result(self, ctx, X, y):
        with _ManagedConfig(int(self.n_predictive)) as cfg:
            return _methods.o2pls_fit(
                ctx, cfg, X, y,
                n_predictive=int(self.n_predictive),
                n_x_orthogonal=int(self.n_x_orthogonal),
                n_y_orthogonal=int(self.n_y_orthogonal))


class KernelPLSRegression(_MethodResultRegressor):
    """Non-linear kernel PLS (Rosipal & Trejo 2001).

    Predicts via the kernel between new X and the training X:

        Y_pred = K(X_new, X_train) @ alpha + y_mean

    We must keep the training matrix around for the kernel. The
    `kernel_type` enum follows the C ABI:
    ``0=linear, 1=RBF, 2=polynomial, 3=sigmoid``.
    """

    def __init__(self, n_components: int = 2,
                  *, kernel_type: int = 1,
                  gamma: float = 0.0,
                  coef0: float = 1.0,
                  degree: int = 3) -> None:
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

    def fit(self, X, y):
        from ._base import _validate_X_y_no_mutate
        X_arr, y_arr, y_ndim = _validate_X_y_no_mutate(X, y)
        ctx = Context()
        with _ManagedConfig(int(self.n_components)) as cfg:
            res = _methods.kernel_pls_fit(
                ctx, cfg, X_arr, y_arr,
                kernel_type=int(self.kernel_type),
                gamma=float(self.gamma),
                coef0=float(self.coef0),
                degree=int(self.degree))
        self._extract_state_kernel(res)
        self._X_train_ = X_arr.copy()
        self.n_features_in_ = int(X_arr.shape[1])
        self._y_ndim_ = y_ndim
        return self

    def _extract_state_kernel(self, result) -> None:
        self.alpha_ = np.asarray(
            result.matrix("alpha"), dtype=np.float64)
        self.y_mean_ = np.asarray(
            result.matrix("y_mean"), dtype=np.float64).ravel()
        # Stub coef_ for sklearn introspection.
        self.coef_ = np.zeros(1)

    def _kernel(self, X1, X2):
        if self.kernel_type == 0:  # linear
            return X1 @ X2.T
        if self.kernel_type == 1:  # RBF
            gamma = self.gamma if self.gamma > 0 else 1.0 / X1.shape[1]
            sq = (np.sum(X1**2, axis=1)[:, None]
                    + np.sum(X2**2, axis=1)[None, :]
                    - 2.0 * X1 @ X2.T)
            return np.exp(-gamma * sq)
        if self.kernel_type == 2:  # polynomial
            gamma = self.gamma if self.gamma > 0 else 1.0 / X1.shape[1]
            return (gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        if self.kernel_type == 3:  # sigmoid
            gamma = self.gamma if self.gamma > 0 else 1.0 / X1.shape[1]
            return np.tanh(gamma * (X1 @ X2.T) + self.coef0)
        raise ValueError(f"unknown kernel_type {self.kernel_type}")

    def predict(self, X):
        from sklearn.utils.validation import check_is_fitted
        from ._base import _check_X_p4a
        check_is_fitted(self, ["alpha_", "_X_train_"])
        X_arr = _check_X_p4a(self, X)
        K = self._kernel(X_arr, self._X_train_)
        if self.alpha_.ndim == 1:
            preds = K @ self.alpha_ + self.y_mean_[0]
        else:
            preds = K @ self.alpha_ + self.y_mean_
        if (getattr(self, "_y_ndim_", 2) == 1 and preds.ndim == 2
                and preds.shape[1] == 1):
            preds = preds.ravel()
        return preds


class PLSGLMRegressor(_MethodResultRegressor):
    """PLS + Generalised Linear Model head (Bastien 2005).

    Currently supports Gaussian (default) and Poisson families via the
    `poisson` flag. The C result carries `coefficients` and a scalar
    `intercept`; predict uses `X @ coef + intercept` then applies the
    link inverse implicitly (the C side returns the response-scale
    coefficients).
    """

    def __init__(self, n_components: int = 2, *, poisson: bool = False) -> None:
        self.n_components = n_components
        self.poisson = poisson

    def _fit_method_result(self, ctx, X, y):
        with _ManagedConfig(int(self.n_components)) as cfg:
            return _methods.pls_glm_fit(
                ctx, cfg, X, y, poisson=bool(self.poisson))

    def _extract_state(self, result) -> None:
        # pls_glm_fit returns `intercept` directly (not `y_mean`); reuse
        # the same custom-extract pattern as MBPLSRegression.
        coef = np.asarray(result.matrix("coefficients"), dtype=np.float64)
        coef_T = coef.T.copy()
        if coef_T.shape[0] == 1:
            coef_T = coef_T.reshape(coef_T.shape[1])
        intercept_arr = np.asarray(
            result.matrix("intercept"), dtype=np.float64).ravel()
        self.coef_ = coef_T
        # GLM has no x_mean / y_mean in the result; the centering is
        # already folded into the intercept.
        self.x_mean_ = np.zeros(coef_T.shape[-1] if coef_T.ndim > 1
                                  else coef_T.shape[0])
        self.y_mean_ = np.zeros(1, dtype=np.float64)
        self.intercept_ = (
            float(intercept_arr[0]) if intercept_arr.size == 1
            else intercept_arr
        )

    def predict(self, X):
        from sklearn.utils.validation import check_is_fitted
        from ._base import _check_X_p4a
        check_is_fitted(self)
        X_arr = _check_X_p4a(self, X)
        if self.coef_.ndim == 1:
            preds = X_arr @ self.coef_ + self.intercept_
        else:
            preds = X_arr @ self.coef_.T + self.intercept_
        if (getattr(self, "_y_ndim_", 2) == 1 and preds.ndim == 2
                and preds.shape[1] == 1):
            preds = preds.ravel()
        return preds


__all__ = [
    "SparseSimplsRegression",
    "CPPLSRegression",
    "ECRegression",
    "DIPLSRegression",
    "MIRPLSRegression",
    "MBPLSRegression",
    "NPLSRegression",
    "O2PLSRegression",
    "PLSGLMRegressor",
    "KernelPLSRegression",
]
