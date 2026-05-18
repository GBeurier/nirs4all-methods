# SPDX-License-Identifier: CECILL-2.1
"""Base classes for sklearn-style operator wrappers.

Each subclass declares a class-level ``_C_PREFIX`` constant (e.g. ``"c4a_pp_snv"``)
that the mixin uses to dispatch ``_create``, ``_transform``, ``_destroy``,
``_fit`` and ``_is_fitted`` calls on the FFI handle.
"""
from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from .._errors import check
from .._ffi import lib
from .._matrix import as_f64_2d, empty_like_f64, numpy_to_view


class _HandleEstimator:
    """Lifecycle scaffolding for an operator backed by a libc4a handle.

    Subclasses set ``_C_PREFIX`` and override :meth:`_create_handle` to invoke
    the corresponding ``c4a_*_create`` function with the constructor params.
    """

    _C_PREFIX: str = ""

    def __init__(self) -> None:
        self._handle: ctypes.c_void_p | None = None
        self._fitted: bool = False

    # -- subclass hooks -----------------------------------------------------

    def _create_handle(self) -> ctypes.c_void_p:
        raise NotImplementedError

    # -- lifecycle ---------------------------------------------------------

    def _ensure_handle(self) -> ctypes.c_void_p:
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        return self._handle

    def _destroy(self) -> None:
        if self._handle is not None and self._handle.value:
            destroy = getattr(lib, f"{self._C_PREFIX}_destroy")
            destroy(self._handle)
        self._handle = None
        self._fitted = False

    def __del__(self) -> None:
        try:
            self._destroy()
        except Exception:
            pass

    # -- sklearn-style helpers --------------------------------------------

    def __sklearn_is_fitted__(self) -> bool:  # noqa: D401 - sklearn protocol
        """Return ``True`` if the operator has been fitted."""
        return self._fitted

    def _call_transform(self, X: np.ndarray, out_shape=None) -> np.ndarray:
        """Run ``c4a_<prefix>_transform`` on a 2-D F64 array.

        ``out_shape`` defaults to ``X.shape`` for shape-preserving operators.
        """
        if out_shape is None:
            out_shape = X.shape
        handle = self._ensure_handle()
        out = empty_like_f64(out_shape)
        x_view = numpy_to_view(X)
        out_view = numpy_to_view(out)
        transform = getattr(lib, f"{self._C_PREFIX}_transform")
        status = transform(handle, x_view, out_view)
        check(status, f"{self._C_PREFIX}_transform")
        return out


class StatelessOperator(_HandleEstimator):
    """Wrapper for stateless operators (no ``fit`` needed)."""

    def fit(self, X, y=None):  # noqa: D401 - sklearn protocol
        """No-op: stateless operators don't need fitting."""
        X = as_f64_2d(X)
        self._ensure_handle()
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        X = as_f64_2d(X)
        return self._call_transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class StatefulOperator(_HandleEstimator):
    """Wrapper for stateful operators with a ``c4a_*_fit`` entry point."""

    def fit(self, X, y=None):  # noqa: D401 - sklearn protocol
        X = as_f64_2d(X)
        handle = self._ensure_handle()
        fit = getattr(lib, f"{self._C_PREFIX}_fit")
        x_view = numpy_to_view(X)
        status = fit(handle, x_view)
        check(status, f"{self._C_PREFIX}_fit")
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before transform")
        X = as_f64_2d(X)
        return self._call_transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


__all__ = ["_HandleEstimator", "StatefulOperator", "StatelessOperator"]
