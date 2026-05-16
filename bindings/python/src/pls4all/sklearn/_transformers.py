"""sklearn TransformerMixin wrappers for calibration-transfer methods.

Calibration transfer fits a standardization transformation from a
source-instrument spectrum to a target-instrument spectrum, then
applies the transformation to map any new source spectrum to the
target instrument's response space.

This module wraps:

* PDS — Piecewise Direct Standardization (Wang 1991)
* DS  — Direct Standardization
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .. import _methods
from .._context import Context
from ._base import _check_X_p4a


class _CalibrationTransferBase(TransformerMixin, BaseEstimator):
    """Shared `fit` / `transform` plumbing.

    The C ABI returns a (p × p) ``transformation`` matrix and, for DS,
    a (p,) ``bias`` vector. Transform applies ``X_corrected = X @ T +
    bias`` so source spectra are mapped onto the target instrument.
    """

    transformation_: np.ndarray | None = None
    bias_: np.ndarray | None = None

    def fit(self, X, y=None, X_target=None):
        if X_target is None:
            raise ValueError(
                f"{type(self).__name__}.fit requires `X_target` (the "
                f"target-instrument spectra to align onto)")
        X_src = np.ascontiguousarray(X, dtype=np.float64)
        X_tgt = np.ascontiguousarray(X_target, dtype=np.float64)
        if X_src.shape != X_tgt.shape:
            raise ValueError(
                f"X_source ({X_src.shape}) and X_target ({X_tgt.shape}) "
                f"must have identical shape")
        ctx = Context()
        result = self._call_fit(ctx, X_src, X_tgt)
        self.transformation_ = np.asarray(
            result.matrix("transformation"), dtype=np.float64)
        try:
            self.bias_ = np.asarray(
                result.matrix("bias"), dtype=np.float64).ravel()
        except Exception:
            self.bias_ = None
        self.n_features_in_ = int(X_src.shape[1])
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_arr = _check_X_p4a(self, X)
        # The two C ABI calibration kernels store `transformation` with
        # opposite conventions; subclasses know which side to apply.
        X_corr = self._apply_transform(X_arr)
        if self.bias_ is not None:
            X_corr = X_corr + self.bias_
        return X_corr

    def _apply_transform(self, X_arr):
        raise NotImplementedError

    def _call_fit(self, ctx, X_src, X_tgt):
        raise NotImplementedError


class PDSTransformer(_CalibrationTransferBase):
    """Piecewise Direct Standardization (Wang 1991).

    Each target-instrument wavelength is regressed against a window of
    source-instrument wavelengths centred on it. ``window_half_width``
    controls the per-wavelength fitting window; ``0`` reduces to DS.
    """

    def __init__(self, window_half_width: int = 5) -> None:
        self.window_half_width = window_half_width

    def _call_fit(self, ctx, X_src, X_tgt):
        return _methods.pds_fit(
            ctx, X_src, X_tgt,
            window_half_width=int(self.window_half_width))

    def _apply_transform(self, X_arr):
        # PDS stores `transformation` as p_target × p_source; left-multiply.
        return X_arr @ self.transformation_.T


class DSTransformer(_CalibrationTransferBase):
    """Direct Standardization — full-band cross-instrument regression."""

    def __init__(self) -> None:
        pass  # no hyperparameters

    def _call_fit(self, ctx, X_src, X_tgt):
        return _methods.ds_fit(ctx, X_src, X_tgt)

    def _apply_transform(self, X_arr):
        # DS stores `transformation` as p_source × p_target; no transpose.
        # See c_api_method_result.cpp:1879 — predict is
        # `X_source @ transformation + bias`.
        return X_arr @ self.transformation_


__all__ = ["PDSTransformer", "DSTransformer"]
