# SPDX-License-Identifier: CECILL-2.1
"""Sparse regression estimators backed by the libn4m C ABI."""

from __future__ import annotations

import ctypes
from numbers import Real
from typing import Any

import numpy as np

from n4m._impl import native as _native
from n4m._impl.affine_result import AffineMethodResultRegressor


def _groups_i32(group_assignment: Any, n_features: int) -> np.ndarray:
    groups = np.asarray(group_assignment, dtype=object)
    if groups.ndim != 1 or groups.size != n_features:
        raise ValueError("group_assignment must contain one ID per X feature")
    if any(
        isinstance(group, (bool, np.bool_))
        or not isinstance(group, (int, np.integer))
        or group < 0
        or group > np.iinfo(np.int32).max
        for group in groups
    ):
        raise ValueError("group_assignment IDs must be nonnegative int32 integers")
    return np.ascontiguousarray(groups, dtype=np.int32)


def _nonnegative_penalty(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("group_lambda must be a finite nonnegative number")  # noqa: TRY004
    try:
        penalty = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("group_lambda must be a finite nonnegative number") from exc
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("group_lambda must be a finite nonnegative number")
    return penalty


def group_sparse_pls(
    X: Any,
    y: Any,
    *,
    n_components: int = 2,
    group_assignment: Any,
    group_lambda: float = 0.05,
) -> dict[str, np.ndarray | float | bytes]:
    """Return native GroupSparsePLS coefficients and fit means."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or not all(X_arr.shape) or not np.isfinite(X_arr).all():
        raise ValueError("X must be a nonempty finite 2-D matrix")
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim not in (1, 2) or y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("y must have one row per X sample")
    if y_arr.size == 0 or not np.isfinite(y_arr).all():
        raise ValueError("y must be nonempty and finite")
    if (
        isinstance(n_components, (bool, np.bool_))
        or not isinstance(n_components, (int, np.integer))
        or not 1 <= n_components <= np.iinfo(np.int32).max
    ):
        raise ValueError("n_components must be a positive int32 integer")
    groups = _groups_i32(group_assignment, X_arr.shape[1])
    penalty = _nonnegative_penalty(group_lambda)
    return _native._fit_method_result(
        "n4m_estimators_group_sparse_pls_fit",
        X_arr,
        y_arr,
        groups.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(groups.size),
        ctypes.c_double(penalty),
        matrices=("coefficients", "x_mean", "y_mean", "predictions"),
        scalars=(),
        n_components=int(n_components),
        solver=1,  # N4M_SOLVER_SIMPLS
        center_x=True,
        scale_x=False,
        center_y=True,
        scale_y=False,
        as_model=True,
    )


class GroupSparsePLS(AffineMethodResultRegressor):
    """Group-sparse SIMPLS with held-out prediction from native coefficients.

    ``group_assignment`` contains one nonnegative int32 ID per input feature.
    IDs may be nonconsecutive. ``group_lambda`` applies group shrinkage after
    native SIMPLS fitting.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        group_assignment: Any = None,
        group_lambda: float = 0.05,
    ) -> None:
        self.n_components = n_components
        self.group_assignment = group_assignment
        self.group_lambda = group_lambda

    def _fit_native(self, X: Any, y: Any) -> dict[str, np.ndarray | float | bytes]:
        return group_sparse_pls(
            X,
            y,
            n_components=self.n_components,
            group_assignment=self.group_assignment,
            group_lambda=self.group_lambda,
        )


__all__ = ["GroupSparsePLS", "group_sparse_pls"]
