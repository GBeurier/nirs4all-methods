"""sklearn-compatible wrappers for methods whose C MethodResult exposes
only in-sample predictions (no regression coefficients or per-feature
preprocessing stats).

The C ABI for these kernels intentionally captures only what was
computed during the fit (often a moving window, an ensemble vote, a
local refit, or a non-linear basis without a global coefficient
matrix). Predict-on-arbitrary-new-X would require either:

* a coefficient export in the C ABI (planned follow-up), or
* re-fitting on (X_train, y_train) + X_new at predict time (which the
  user can do explicitly via tier 1).

Until then we expose these methods as **fit-only sklearn estimators**:

* ``fit(X, y)`` runs the C kernel, stores ``self.predictions_`` (1-D
  or 2-D, matching the y shape on the way in).
* ``predict(X)`` returns ``self.predictions_`` IFF ``X is X_train``
  (identity match — same numpy array). Otherwise raises an
  informative :class:`NotImplementedError` explaining the limitation
  and pointing at the tier-1 entry point.

The fit-only contract makes these classes usable in:

* end-of-pipeline reporting (e.g. ``fit(X_train, y).predictions_``),
* exploration / training-set diagnostics,
* sklearn ``cross_val_score`` with ``cv="prefit"`` workflows that
  refit per fold via the wrapper itself.

It is NOT a drop-in for ``Pipeline(... → predict(X_new))``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .. import _methods
from .._context import Context
from .._types import Algorithm, Deflation, Solver
from ._base import _validate_X_y_no_mutate


class _InSampleOnlyRegressor(BaseEstimator, RegressorMixin):
    """Base for fit-only PLS-style regressors.

    Subclasses override :meth:`_run_fit` to call the tier-1 fit function
    and return a :class:`MethodResult` whose ``predictions`` matrix
    holds the in-sample fit.
    """

    predictions_: np.ndarray | None = None
    _X_train_ref: int = 0   # id() of the training X — for identity-check predict.

    def _run_fit(self, ctx: Context, X: np.ndarray,
                  y: np.ndarray) -> Any:
        raise NotImplementedError

    def fit(self, X, y):
        X_arr, y_arr, y_ndim_orig = _validate_X_y_no_mutate(X, y)
        ctx = Context()
        result = self._run_fit(ctx, X_arr, y_arr)
        # Standardise predictions to numpy (n, q) then optionally ravel.
        preds = np.asarray(result.matrix("predictions"), dtype=np.float64)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if y_ndim_orig == 1 and preds.shape[1] == 1:
            preds = preds.ravel()
        self.predictions_ = preds
        self.n_features_in_ = int(X_arr.shape[1])
        self._y_ndim_ = y_ndim_orig
        # Track the identity of the training array so predict can verify.
        self._X_train_ref = id(X)
        return self

    def predict(self, X):
        check_is_fitted(self)
        if id(X) != self._X_train_ref:
            cls = type(self).__name__
            raise NotImplementedError(
                f"{cls} is an in-sample-only estimator: its C kernel "
                f"does not export regression coefficients, so predict-on-"
                f"new-X requires refitting. Pass the same X used in "
                f"fit() to retrieve the stored predictions, or call "
                f"pls4all._methods directly to refit on the union of "
                f"training and new X."
            )
        return self.predictions_


def _basic_cfg(n_components: int) -> Any:
    """Standard PLS regression Config used by every in-sample wrapper."""
    from .._config import Config
    cfg = Config()
    cfg.algorithm = Algorithm.PLS_REGRESSION
    cfg.solver = Solver.SIMPLS
    cfg.deflation = Deflation.REGRESSION
    cfg.n_components = int(n_components)
    cfg.center_x = True
    cfg.scale_x = False
    cfg.center_y = True
    cfg.scale_y = False
    return cfg


# ----------------------------------------------------------------------
# Regressors with extra fit-time parameters
# ----------------------------------------------------------------------

class WeightedPLSRegression(_InSampleOnlyRegressor):
    """Sample-weighted PLS (sqrt(w)-prescaled SIMPLS)."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            raise ValueError(
                "WeightedPLSRegression.fit requires `sample_weight`")
        self._sample_weight = np.ascontiguousarray(
            sample_weight, dtype=np.float64).ravel()
        return super().fit(X, y)

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            return _methods.weighted_pls_fit(
                ctx, cfg, X, y, sample_weights=self._sample_weight)
        finally:
            cfg.close()


class RobustPLSRegression(_InSampleOnlyRegressor):
    """Robust PLS via Huber IRLS over weighted SIMPLS."""

    def __init__(self, n_components: int = 2,
                  *, huber_k: float = 1.345,
                  max_irls_iter: int = 20) -> None:
        self.n_components = n_components
        self.huber_k = huber_k
        self.max_irls_iter = max_irls_iter

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            return _methods.robust_pls_fit(
                ctx, cfg, X, y,
                huber_k=float(self.huber_k),
                max_irls_iter=int(self.max_irls_iter))
        finally:
            cfg.close()


class RidgePLSRegression(_InSampleOnlyRegressor):
    """L2-augmented PLS regression."""

    def __init__(self, n_components: int = 2,
                  *, ridge_lambda: float = 1.0) -> None:
        self.n_components = n_components
        self.ridge_lambda = ridge_lambda

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            return _methods.ridge_pls_fit(
                ctx, cfg, X, y, ridge_lambda=float(self.ridge_lambda))
        finally:
            cfg.close()


class ContinuumRegression(_InSampleOnlyRegressor):
    """Continuum regression τ ∈ [0, 1] interpolates PLS (1) / OLS (0)."""

    def __init__(self, n_components: int = 2, *, tau: float = 0.5) -> None:
        self.n_components = n_components
        self.tau = tau

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            return _methods.continuum_regression_fit(
                ctx, cfg, X, y, tau=float(self.tau))
        finally:
            cfg.close()


class RecursivePLSRegression(_InSampleOnlyRegressor):
    """Moving-window recursive PLS."""

    def __init__(self, n_components: int = 2,
                  *, window_size: int = 50) -> None:
        self.n_components = n_components
        self.window_size = window_size

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            return _methods.recursive_pls_run(
                ctx, cfg, X, y, window_size=int(self.window_size))
        finally:
            cfg.close()


class LWPLSRegression(_InSampleOnlyRegressor):
    """Locally-weighted PLS (Næs & Centner 1998)."""

    def __init__(self, n_components: int = 2,
                  *, n_neighbors: int = 30) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            return _methods.lw_pls_fit(
                ctx, cfg, X, y, n_neighbors=int(self.n_neighbors))
        finally:
            cfg.close()


class MissingAwareNipalsRegression(_InSampleOnlyRegressor):
    """Nelson 1996 missing-aware NIPALS PLS."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            return _methods.missing_aware_nipals_fit(ctx, cfg, X, y)
        finally:
            cfg.close()


class GroupSparsePLSRegression(_InSampleOnlyRegressor):
    """Group-sparse PLS — L1 across pre-declared feature groups."""

    def __init__(self, n_components: int = 2,
                  *, group_assignment=None,
                  group_lambda: float = 0.05) -> None:
        self.n_components = n_components
        self.group_assignment = group_assignment
        self.group_lambda = group_lambda

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            ga = np.ascontiguousarray(
                self.group_assignment, dtype=np.int32).ravel()
            return _methods.group_sparse_pls_fit(
                ctx, cfg, X, y, group_assignment=ga,
                group_lambda=float(self.group_lambda))
        finally:
            cfg.close()


class FusedSparsePLSRegression(_InSampleOnlyRegressor):
    """Fused-sparse PLS — L1 + adjacent-coef smoothing."""

    def __init__(self, n_components: int = 2,
                  *, l1_lambda: float = 0.05,
                  fusion_lambda: float = 0.05) -> None:
        self.n_components = n_components
        self.l1_lambda = l1_lambda
        self.fusion_lambda = fusion_lambda

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            return _methods.fused_sparse_pls_fit(
                ctx, cfg, X, y,
                l1_lambda=float(self.l1_lambda),
                fusion_lambda=float(self.fusion_lambda))
        finally:
            cfg.close()


# ----------------------------------------------------------------------
# Ensemble regressors (need n_estimators)
# ----------------------------------------------------------------------

class BaggingPLSRegression(_InSampleOnlyRegressor):
    """Bagged PLS (Breiman 1996)."""

    def __init__(self, n_components: int = 2,
                  *, n_estimators: int = 50, seed: int = 0) -> None:
        self.n_components = n_components
        self.n_estimators = n_estimators
        self.seed = seed

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            return _methods.bagging_pls_fit(
                ctx, cfg, X, y,
                n_estimators=int(self.n_estimators),
                seed=int(self.seed))
        finally:
            cfg.close()


class BoostingPLSRegression(_InSampleOnlyRegressor):
    """Boosted PLS regression."""

    def __init__(self, n_components: int = 2,
                  *, n_estimators: int = 50,
                  learning_rate: float = 0.1) -> None:
        self.n_components = n_components
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            return _methods.boosting_pls_fit(
                ctx, cfg, X, y,
                n_estimators=int(self.n_estimators),
                learning_rate=float(self.learning_rate))
        finally:
            cfg.close()


class RandomSubspacePLSRegression(_InSampleOnlyRegressor):
    """Random-subspace PLS — Ho 1998."""

    def __init__(self, n_components: int = 2,
                  *, n_estimators: int = 50,
                  features_per_subspace: int = 10,
                  seed: int = 0) -> None:
        self.n_components = n_components
        self.n_estimators = n_estimators
        self.features_per_subspace = features_per_subspace
        self.seed = seed

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            return _methods.random_subspace_pls_fit(
                ctx, cfg, X, y,
                n_estimators=int(self.n_estimators),
                features_per_subspace=int(self.features_per_subspace),
                seed=int(self.seed))
        finally:
            cfg.close()


# ----------------------------------------------------------------------
# Multi-block regressors (need X_blocks; X is the concatenated matrix)
# ----------------------------------------------------------------------

class SOPLSRegression(_InSampleOnlyRegressor):
    """Sequential & Orthogonalised multi-block PLS (Næs et al. 2011)."""

    def __init__(self, *, n_components_per_block=None,
                  block_sizes=None) -> None:
        self.n_components_per_block = n_components_per_block
        self.block_sizes = block_sizes

    @property
    def n_components(self) -> int:
        return int(sum(self.n_components_per_block or [2]))

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            blocks = _split_blocks(X, self.block_sizes)
            comps = np.ascontiguousarray(
                self.n_components_per_block, dtype=np.int32)
            return _methods.so_pls_fit(
                ctx, cfg, X_blocks=blocks, Y=y,
                n_components_per_block=comps)
        finally:
            cfg.close()


class ROSARegression(_InSampleOnlyRegressor):
    """Response-Oriented Sequential Alternation (Liland & Næs 2016)."""

    def __init__(self, n_components: int = 2,
                  *, block_sizes=None) -> None:
        self.n_components = n_components
        self.block_sizes = block_sizes

    def _run_fit(self, ctx, X, y):
        cfg = _basic_cfg(self.n_components)
        try:
            blocks = _split_blocks(X, self.block_sizes)
            return _methods.rosa_fit(
                ctx, cfg, X_blocks=blocks, Y=y,
                n_components=int(self.n_components))
        finally:
            cfg.close()


def _split_blocks(X, block_sizes):
    """Split a concatenated row-major (n, sum(block_sizes)) matrix into
    a list of (n, k_i) blocks, contiguous in memory."""
    if not block_sizes:
        raise ValueError("block_sizes must be provided (list of ints)")
    sizes = list(block_sizes)
    if int(sum(sizes)) != X.shape[1]:
        raise ValueError(
            f"sum(block_sizes)={sum(sizes)} must equal X.shape[1]={X.shape[1]}")
    out = []
    cursor = 0
    for k in sizes:
        block = np.ascontiguousarray(X[:, cursor:cursor + int(k)], dtype=np.float64)
        out.append(block)
        cursor += int(k)
    return out


__all__ = [
    "WeightedPLSRegression",
    "RobustPLSRegression",
    "RidgePLSRegression",
    "ContinuumRegression",
    "RecursivePLSRegression",
    "LWPLSRegression",
    "MissingAwareNipalsRegression",
    "GroupSparsePLSRegression",
    "FusedSparsePLSRegression",
    "BaggingPLSRegression",
    "BoostingPLSRegression",
    "RandomSubspacePLSRegression",
    "SOPLSRegression",
    "ROSARegression",
]
# NOTE: ONPLSRegression deliberately NOT exposed: the on_pls_fit
# MethodResult is decomposition-only (joint/unique loadings + scores)
# and carries no `predictions` matrix, so the in-sample-only contract
# cannot be honored. Tier-1 access via pls4all.on_pls_fit remains.
