# SPDX-License-Identifier: CECILL-2.1
"""Thin sklearn-compatible facades for the native spectral encoders."""

from __future__ import annotations

import ctypes as ct

import numpy as np

from ._errors import check
from ._ffi import lib
from ._impl.compat import BaseEstimator, TransformerMixin
from ._matrix import as_f64_2d, numpy_to_view
from ._types import MatrixView


def _function(name, types, result=ct.c_int):
    fn = getattr(lib, "n4m_decomposition_spectral_" + name)
    fn.argtypes, fn.restype = types, result
    return fn


class _SpectralEncoder(TransformerMixin, BaseEstimator):
    def __getstate__(self):
        if getattr(self, "_handle", None):
            raise TypeError(
                "A fitted spectral encoder owns a process-local native handle; "
                "copying and pickling are unsupported. Use sklearn.base.clone "
                "to create an unfitted encoder."
            )
        return self.__dict__.copy()

    def fit(self, X, y=None):
        X = as_f64_2d(X)
        handle = ct.c_void_p()
        create = _function(
            "create",
            [
                ct.c_int32,
                ct.c_int32,
                ct.c_int32,
                ct.c_double,
                ct.c_int32,
                ct.c_int32,
                ct.c_int32,
                ct.c_double,
                ct.POINTER(ct.c_void_p),
            ],
        )
        check(create(*self._native_parameters(), ct.byref(handle)), "spectral_create")
        try:
            check(
                _function("fit", [ct.c_void_p, MatrixView])(handle, numpy_to_view(X)),
                "spectral_fit",
            )
            count = ct.c_int64()
            check(
                _function("output_cols", [ct.c_void_p, ct.POINTER(ct.c_int64)])(
                    handle, ct.byref(count)
                ),
                "spectral_output_cols",
            )
        except BaseException:
            _function("destroy", [ct.c_void_p], None)(handle)
            raise
        self.close()
        self._handle = handle
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = count.value
        return self

    def close(self):
        handle = getattr(self, "_handle", None)
        if handle:
            _function("destroy", [ct.c_void_p], None)(handle)
            self._handle = None

    def __del__(self):
        self.close()

    def transform(self, X):
        if not getattr(self, "_handle", None):
            raise RuntimeError("transform called before fit or after close")
        X = as_f64_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X feature count differs from fit")
        out = np.empty((len(X), self.n_features_out_), dtype=np.float64)
        if len(X):
            check(
                _function("transform", [ct.c_void_p, MatrixView, MatrixView])(
                    self._handle, numpy_to_view(X), numpy_to_view(out)
                ),
                "spectral_transform",
            )
        return out


class LVSE(_SpectralEncoder):
    """Local variance subspaces fitted exclusively on training spectra.

    Rank per block is capped at min(n_samples, block_width) - 1, matching the
    NIRS research implementation. A final singleton block is skipped. With
    overlap, the last full window is anchored at the final channel. Fits use
    all supplied rows and an exact Gram SVD; no implicit subsampling occurs.
    """

    def __init__(self, width=64, rank=4, overlap=0.0, standardize=True, snv=False):
        self.width = width
        self.rank = rank
        self.overlap = overlap
        self.standardize = standardize
        self.snv = snv

    def _native_parameters(self):
        return (
            0,
            self.width,
            self.rank,
            self.overlap,
            int(self.standardize),
            int(self.snv),
            60,
            1e-3,
        )

    def export_linear_operator(self):
        """Return (A, b) such that transform(X) = X @ A.T + b.

        SNV is row-dependent and cannot be exported as one affine operator.
        """
        if not getattr(self, "_handle", None):
            raise RuntimeError("export called before fit or after close")
        if self.snv:
            raise ValueError("LVSE with SNV has no raw-input affine operator")
        operator = np.empty((self.n_features_out_, self.n_features_in_))
        offset = np.empty((1, self.n_features_out_))
        check(
            _function("export_affine", [ct.c_void_p, MatrixView, MatrixView])(
                self._handle, numpy_to_view(operator), numpy_to_view(offset)
            ),
            "spectral_export_affine",
        )
        return operator, offset.ravel()


class GCU(_SpectralEncoder):
    """Global nonnegative factors with training minima and global scaling.

    Native NNDSVDa initialization and cyclic coordinate descent. Query spectra
    are clipped at the training minima; the fitted basis stays fixed.
    """

    def __init__(self, rank=16, max_iter=60, tol=1e-3):
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol

    def _native_parameters(self):
        return (1, 64, self.rank, 0.0, 1, 0, self.max_iter, self.tol)
