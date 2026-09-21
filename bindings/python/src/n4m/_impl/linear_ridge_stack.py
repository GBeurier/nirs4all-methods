# SPDX-License-Identifier: CECILL-2.1
"""Ridge stacking orchestration over the native sweep and affine compression ABI."""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass

import numpy as np

from .._errors import check
from .._ffi import lib
from .._matrix import as_f64_2d, numpy_to_view
from .._types import MatrixView
from .._validation import _selector_fold_ids
from .compat import BaseEstimator, RegressorMixin
from .native import aom_chain_sweep_run


@dataclass
class AffinePredictor:
    """Deployment-only coefficients and intercept; no base models or fit data."""

    coef_: np.ndarray
    intercept_: np.ndarray

    def predict(self, X):
        X = as_f64_2d(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError("X feature count differs from coefficients")
        result = X @ self.coef_ + self.intercept_
        return result[:, 0] if result.shape[1] == 1 else result

    @property
    def numeric_bytes(self):
        return self.coef_.nbytes + self.intercept_.nbytes


def compress_linear_stack(
    base_coefficients, base_intercepts, meta_weights, meta_intercept
):
    """Compress a fully affine stack, including arbitrary base/meta intercepts.

    Base columns and meta-weight rows must be in the same order. Coefficients
    must already include all fixed linear/affine preprocessing. Never pass a
    SNV/MSC predictor as an affine function of raw spectra.
    """
    base = as_f64_2d(base_coefficients)
    bias = np.ascontiguousarray(base_intercepts, dtype=np.float64).reshape(1, -1)
    weights = np.asarray(meta_weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[:, None]
    weights = as_f64_2d(weights)
    meta = np.ascontiguousarray(meta_intercept, dtype=np.float64).reshape(1, -1)
    coef = np.empty((base.shape[0], weights.shape[1]))
    intercept = np.empty((1, weights.shape[1]))
    arrays = [base, bias, weights, meta, coef, intercept]
    views = [numpy_to_view(a) for a in arrays]
    fn = lib.n4m_ensemble_linear_stack_compress
    fn.restype = ct.c_int
    fn.argtypes = [ct.POINTER(MatrixView)] * 6
    check(fn(*(ct.byref(v) for v in views)), "linear_stack_compress")
    return AffinePredictor(coef, intercept[0])


class LinearRidgeStackRegressor(RegressorMixin, BaseEstimator):
    """One native Ridge per strict-linear view, OOF Ridge meta-head, affine export.

    Each OOF base prediction comes from an independent inner-CV selection on
    its outer training rows. Meta alpha is selected on the OOF design; its CV
    score is a tuning criterion, not an independent performance estimate.
    Evaluate accuracy on an untouched external test set. Compression changes
    deployment only: all base/OOF/meta training is still performed.
    """

    def __init__(
        self,
        *,
        operators=None,
        alphas=None,
        meta_alphas=None,
        cv=3,
        fold_ids=None,
        inner_cv=3,
    ):
        self.operators = operators
        self.alphas = alphas
        self.meta_alphas = meta_alphas
        self.cv = cv
        self.fold_ids = fold_ids
        self.inner_cv = inner_cv

    def fit(self, X, y):
        X = as_f64_2d(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(y) != len(X):
            raise ValueError("X and y have incompatible lengths")
        from ..model_selection.aom_calibration import BANK

        operators = (
            [descriptor for _, descriptor in BANK]
            if self.operators is None
            else list(self.operators)
        )
        if not operators:
            raise ValueError("operators must be nonempty")

        def grid(value):
            centered = value - value.mean(axis=0)
            base = max(float(np.sum(centered * centered)) / len(value), 1e-12)
            return base * np.logspace(-6, 6, 50)

        alphas = grid(X) if self.alphas is None else self.alphas
        folds = _selector_fold_ids(len(X), self.cv, self.fold_ids)
        oof = np.empty((len(X), len(operators)))
        results = []
        inner_choices = []

        def sweep(x, y, op, grid, cv, fold_ids=None):
            # Near-square/wide folds benefit from factoring min(n_train, p).
            # Tall data retain the existing sufficient-statistics sweep.
            # This choice uses dimensions only, never validation/test outcomes.
            if x.shape[1] > len(x) // 2:
                from ..model_selection.aom_calibration import _fit_single_view_ridge

                return _fit_single_view_ridge(x, y, op, grid, cv, fold_ids)
            return aom_chain_sweep_run(
                x,
                y,
                [[op]],
                heads=("ridge",),
                ridge_lambdas=grid,
                cv=cv,
                fold_ids=fold_ids,
                center_x=True,
                scale_x=False,
                center_y=True,
                scale_y=False,
            )

        # Grid construction is configuration preparation, shared across views.
        fold_grids = [
            grid(X[folds != f]) if self.alphas is None else self.alphas
            for f in range(self.cv)
        ]
        for j, operator in enumerate(operators):
            choices = []
            for fold in range(self.cv):
                train, valid = folds != fold, folds == fold
                if not np.any(valid) or train.sum() < self.inner_cv:
                    raise ValueError(
                        "each fold needs validation rows and enough rows for inner CV"
                    )
                fitted = sweep(
                    X[train], y[train], operator, fold_grids[fold], self.inner_cv
                )
                oof[valid, j] = (
                    X[valid] @ fitted["input_coefficients"] + fitted["intercept"]
                ).ravel()
                choices.append(float(fitted["selected_param"]))
            results.append(sweep(X, y, operator, alphas, self.cv, folds))
            inner_choices.append(choices)
        meta_alphas = grid(oof) if self.meta_alphas is None else self.meta_alphas
        meta = sweep(oof, y, "identity", meta_alphas, self.cv, folds)
        self.base_coefficients_ = np.column_stack(
            [r["input_coefficients"].ravel() for r in results]
        )
        self.base_intercepts_ = np.array([r["intercept"].item() for r in results])
        self.meta_weights_ = meta["input_coefficients"].ravel().copy()
        self.meta_intercept_ = float(meta["intercept"].item())
        self.compressed_ = compress_linear_stack(
            self.base_coefficients_,
            self.base_intercepts_,
            self.meta_weights_,
            [self.meta_intercept_],
        )
        self.coef_ = self.compressed_.coef_[:, 0].copy()
        self.intercept_ = float(self.compressed_.intercept_[0])
        self.n_features_in_ = X.shape[1]
        self.selected_base_alphas_ = [float(r["selected_param"]) for r in results]
        self.selected_meta_alpha_ = float(meta["selected_param"])
        self.inner_base_alphas_ = inner_choices
        self.oof_predictions_ = oof
        return self

    def predict(self, X):
        if not hasattr(self, "compressed_"):
            raise RuntimeError("predict called before fit")
        return self.compressed_.predict(X)

    def predict_uncompressed(self, X):
        """Audit the retained base models plus meta-model without compression."""
        if not hasattr(self, "compressed_"):
            raise RuntimeError("predict called before fit")
        X = as_f64_2d(X)
        views = np.column_stack(
            [
                X @ self.base_coefficients_[:, j] + self.base_intercepts_[j]
                for j in range(self.base_coefficients_.shape[1])
            ]
        )
        return views @ self.meta_weights_ + self.meta_intercept_

    def export_affine(self):
        """Independent deployment object holding only one coefficient vector and bias."""
        if not hasattr(self, "compressed_"):
            raise RuntimeError("export called before fit")
        return AffinePredictor(
            self.compressed_.coef_.copy(), self.compressed_.intercept_.copy()
        )
