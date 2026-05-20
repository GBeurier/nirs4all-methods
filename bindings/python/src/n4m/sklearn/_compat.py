# SPDX-License-Identifier: CECILL-2.1
"""Optional scikit-learn base classes.

The Python wheel only requires NumPy, but when scikit-learn is installed the
``n4m.sklearn`` surface should be recognisable by sklearn tools.
"""
from __future__ import annotations

try:  # pragma: no cover - exercised only when scikit-learn is installed
    from sklearn.base import BaseEstimator, TransformerMixin
except Exception:  # pragma: no cover - keep the core binding dependency-light

    class BaseEstimator:
        """Small fallback with the get/set params contract used by sklearn."""

        def get_params(self, deep: bool = True) -> dict:
            return {
                key: value
                for key, value in vars(self).items()
                if not key.endswith("_") and not key.startswith("_")
            }

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

    class TransformerMixin:
        """Fallback transformer mixin."""

        def fit_transform(self, X, y=None, **fit_params):
            if y is None:
                return self.fit(X, **fit_params).transform(X)
            return self.fit(X, y, **fit_params).transform(X)


__all__ = ["BaseEstimator", "TransformerMixin"]
