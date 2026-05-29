# SPDX-License-Identifier: CECILL-2.1
"""Train / test splitter wrappers.

Each splitter exposes a single :meth:`split` method that returns a pair of
1-D :class:`numpy.ndarray` (``int64``) carrying the train and test indices.

The libn4m side allocates the index arrays via :c:struct:`n4m_split_result_t`;
the binding copies them into NumPy arrays and releases the C-side memory via
``n4m_split_result_destroy`` before returning.
"""
from __future__ import annotations

import ctypes
from typing import Tuple

import numpy as np

from .._errors import check
from .._ffi import lib
from .._matrix import as_f64_2d, numpy_to_view
from .._types import SplitResult
from ._compat import BaseEstimator

# Strategy enums (mirror n4m_split_kbins_strategy_t).
_KBINS_STRATEGIES = {"uniform": 0, "quantile": 1, 0: 0, 1: 1}
# C ABI use_y enum (cpp/src/core/splitters/spxy_fold.h): 0 = pure Kennard-Stone
# (X only), 1 = SPXY (X + euclidean Y), 2 = hamming Y. SPXY MUST add the Y
# distance, so "euclidean" maps to 1 — the old map sent it to 0, which silently
# ran X-only Kennard-Stone and dropped y entirely (a real bug: the split then
# ignored the target). The C engine's only joint metric is euclidean; nirs4all's
# SPXY y_metric is a scipy cdist metric with no Mahalanobis kernel here, so both
# string aliases route to the euclidean-Y code. Use "x_only" for pure KS.
_SPXY_Y_METRICS = {"x_only": 0, "euclidean": 1, "mahalanobis": 1, 0: 0, 1: 1, 2: 2}
_SPXY_GROUP_AGGREGATIONS = {"mean": 0, "median": 1, 0: 0, 1: 1}


def _copy_split_result(result: SplitResult) -> Tuple[np.ndarray, np.ndarray]:
    """Copy ``n4m_split_result_t`` index arrays into NumPy and free the C side."""
    train = np.fromiter(
        (result.train_idx[i] for i in range(result.n_train)),
        dtype=np.int64, count=result.n_train,
    )
    test = np.fromiter(
        (result.test_idx[i] for i in range(result.n_test)),
        dtype=np.int64, count=result.n_test,
    )
    lib.n4m_split_result_destroy(ctypes.byref(result))
    return train, test


def _as_i64_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


class _SplitHandle(BaseEstimator):
    """Small lifecycle helper for opaque splitter handles."""

    _C_PREFIX = ""

    def __init__(self) -> None:
        self._handle: ctypes.c_void_p | None = None

    def _create_handle(self):
        raise NotImplementedError

    def _ensure_handle(self):
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        return self._handle

    def __del__(self):
        if self._handle is not None and self._handle.value:
            getattr(lib, f"{self._C_PREFIX}_destroy")(self._handle)
            self._handle = None


class KennardStoneSplitter(_SplitHandle):
    """Kennard-Stone train/test split.

    Picks the most diverse samples for the training set, in descending order
    of pairwise Euclidean distance.
    """

    def __init__(self, test_size: float = 0.25):
        super().__init__()
        self.test_size = float(test_size)
        self._C_PREFIX = "n4m_split_kennard_stone"

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_split_kennard_stone_create(
                ctypes.byref(h), ctypes.c_double(self.test_size)
            ),
            "n4m_split_kennard_stone_create",
        )
        return h

    def split(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X = as_f64_2d(X)
        self._ensure_handle()
        x_view = numpy_to_view(X)
        result = SplitResult()
        check(
            lib.n4m_split_kennard_stone_split(self._handle, x_view, ctypes.byref(result)),
            "n4m_split_kennard_stone_split",
        )
        return _copy_split_result(result)


class SPXYSplitter(_SplitHandle):
    """SPXY (Sample set Partitioning based on X and Y) train/test split."""

    def __init__(self, test_size: float = 0.25):
        super().__init__()
        self.test_size = float(test_size)
        self._C_PREFIX = "n4m_split_spxy"

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_split_spxy_create(ctypes.byref(h), ctypes.c_double(self.test_size)),
            "n4m_split_spxy_create",
        )
        return h

    def split(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        X = as_f64_2d(X)
        y = as_f64_2d(y)
        self._ensure_handle()
        x_view = numpy_to_view(X)
        y_view = numpy_to_view(y)
        result = SplitResult()
        check(
            lib.n4m_split_spxy_split(self._handle, x_view, y_view, ctypes.byref(result)),
            "n4m_split_spxy_split",
        )
        return _copy_split_result(result)


class SPXYFoldSplitter(_SplitHandle):
    """SPXY k-fold splitter over paired ``X`` and ``y`` matrices."""

    _C_PREFIX = "n4m_split_spxy_fold"

    def __init__(self, n_splits: int = 5, y_metric: str | int = "mahalanobis"):
        super().__init__()
        self.n_splits_value = int(n_splits)
        try:
            self.y_metric = _SPXY_Y_METRICS[y_metric]
        except KeyError as exc:
            raise ValueError(f"Unknown SPXY y_metric: {y_metric!r}") from exc

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_split_spxy_fold_create(
                ctypes.byref(h), ctypes.c_int32(self.n_splits_value),
                ctypes.c_int32(self.y_metric),
            ),
            "n4m_split_spxy_fold_create",
        )
        return h

    def get_n_splits(self) -> int:
        out = ctypes.c_int32()
        check(
            lib.n4m_split_spxy_fold_n_splits(self._ensure_handle(), ctypes.byref(out)),
            "n4m_split_spxy_fold_n_splits",
        )
        return int(out.value)

    def split_fold(self, X, y, fold_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        X = as_f64_2d(X)
        # SPXY now uses the Y distance (y_metric -> use_y=1), so y must be a
        # column (n, 1): as_f64_2d promotes a 1-D y to (1, n), which mismatches
        # X's n rows. A 2-D y (multi-target) is passed through unchanged.
        y = as_f64_2d(np.reshape(y, (-1, 1)) if np.ndim(y) == 1 else y)
        result = SplitResult()
        check(
            lib.n4m_split_spxy_fold_split_fold(
                self._ensure_handle(),
                numpy_to_view(X),
                numpy_to_view(y),
                ctypes.c_int32(fold_idx),
                ctypes.byref(result),
            ),
            "n4m_split_spxy_fold_split_fold",
        )
        return _copy_split_result(result)

    def split(self, X, y):
        for fold_idx in range(self.get_n_splits()):
            yield self.split_fold(X, y, fold_idx)


class SPXYGroupFoldSplitter(_SplitHandle):
    """Group-aware SPXY k-fold splitter."""

    _C_PREFIX = "n4m_split_spxy_g_fold"

    def __init__(
        self,
        n_splits: int = 5,
        y_metric: str | int = "mahalanobis",
        aggregation: str | int = "mean",
    ):
        super().__init__()
        self.n_splits_value = int(n_splits)
        try:
            self.y_metric = _SPXY_Y_METRICS[y_metric]
            self.aggregation = _SPXY_GROUP_AGGREGATIONS[aggregation]
        except KeyError as exc:
            raise ValueError(f"Unknown SPXY group parameter: {exc.args[0]!r}") from exc

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_split_spxy_g_fold_create(
                ctypes.byref(h),
                ctypes.c_int32(self.n_splits_value),
                ctypes.c_int32(self.y_metric),
                ctypes.c_int32(self.aggregation),
            ),
            "n4m_split_spxy_g_fold_create",
        )
        return h

    def get_n_splits(self) -> int:
        out = ctypes.c_int32()
        check(
            lib.n4m_split_spxy_g_fold_n_splits(self._ensure_handle(), ctypes.byref(out)),
            "n4m_split_spxy_g_fold_n_splits",
        )
        return int(out.value)

    def split_fold(self, X, y, groups, fold_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        X = as_f64_2d(X)
        # SPXY uses the Y distance (use_y=1): y must be a column (n, 1); a 1-D y
        # promoted to (1, n) by as_f64_2d would mismatch X's n rows.
        y = as_f64_2d(np.reshape(y, (-1, 1)) if np.ndim(y) == 1 else y)
        groups_arr = _as_i64_1d(groups, "groups")
        result = SplitResult()
        check(
            lib.n4m_split_spxy_g_fold_split_fold(
                self._ensure_handle(),
                numpy_to_view(X),
                numpy_to_view(y),
                groups_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ctypes.c_int64(groups_arr.size),
                ctypes.c_int32(fold_idx),
                ctypes.byref(result),
            ),
            "n4m_split_spxy_g_fold_split_fold",
        )
        return _copy_split_result(result)

    def split(self, X, y, groups):
        for fold_idx in range(self.get_n_splits()):
            yield self.split_fold(X, y, groups, fold_idx)


class KMeansSplitter(_SplitHandle):
    """K-means++ diversity splitter."""

    _C_PREFIX = "n4m_split_kmeans"

    def __init__(self, test_size: float = 0.25, seed: int = 0, max_iter: int = 100):
        super().__init__()
        self.test_size = float(test_size)
        self.seed = int(seed)
        self.max_iter = int(max_iter)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_split_kmeans_create(
                ctypes.byref(h), ctypes.c_double(self.test_size),
                ctypes.c_uint64(self.seed), ctypes.c_int32(self.max_iter)
            ),
            "n4m_split_kmeans_create",
        )
        return h

    def split(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X = as_f64_2d(X)
        result = SplitResult()
        check(
            lib.n4m_split_kmeans_split(
                self._ensure_handle(), numpy_to_view(X), ctypes.byref(result)
            ),
            "n4m_split_kmeans_split",
        )
        return _copy_split_result(result)


class KBinsStratifiedSplitter(_SplitHandle):
    """Stratified split using K equal-width or quantile bins of ``y``."""

    def __init__(self, test_size: float = 0.25, seed: int = 0,
                 n_bins: int = 5, strategy: str | int = "uniform"):
        super().__init__()
        self.test_size = float(test_size)
        self.seed = int(seed)
        self.n_bins = int(n_bins)
        try:
            self.strategy = _KBINS_STRATEGIES[strategy]
        except KeyError as exc:
            raise ValueError(f"Unknown KBins strategy: {strategy!r}") from exc
        self._C_PREFIX = "n4m_split_kbins_stratified"

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_split_kbins_stratified_create(
                ctypes.byref(h),
                ctypes.c_double(self.test_size),
                ctypes.c_uint64(self.seed),
                ctypes.c_int32(self.n_bins),
                ctypes.c_int32(self.strategy),
            ),
            "n4m_split_kbins_stratified_create",
        )
        return h

    def split(self, y) -> Tuple[np.ndarray, np.ndarray]:
        y = as_f64_2d(y)
        self._ensure_handle()
        y_view = numpy_to_view(y)
        result = SplitResult()
        check(
            lib.n4m_split_kbins_stratified_split(
                self._handle, y_view, ctypes.byref(result)
            ),
            "n4m_split_kbins_stratified_split",
        )
        return _copy_split_result(result)

class BinnedStratifiedGroupKFoldSplitter(_SplitHandle):
    """Stratified group k-fold splitter after binning continuous ``y``."""

    _C_PREFIX = "n4m_split_binned_strat_group_kfold"

    def __init__(
        self,
        n_splits: int = 5,
        n_bins: int = 5,
        strategy: str | int = "uniform",
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.n_splits_value = int(n_splits)
        self.n_bins = int(n_bins)
        try:
            self.strategy = _KBINS_STRATEGIES[strategy]
        except KeyError as exc:
            raise ValueError(f"Unknown KBins strategy: {strategy!r}") from exc
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_split_binned_strat_group_kfold_create(
                ctypes.byref(h),
                ctypes.c_int32(self.n_splits_value),
                ctypes.c_int32(self.n_bins),
                ctypes.c_int32(self.strategy),
                ctypes.c_int32(1 if self.shuffle else 0),
                ctypes.c_uint64(self.seed),
            ),
            "n4m_split_binned_strat_group_kfold_create",
        )
        return h

    def get_n_splits(self) -> int:
        out = ctypes.c_int32()
        check(
            lib.n4m_split_binned_strat_group_kfold_n_splits(
                self._ensure_handle(), ctypes.byref(out)
            ),
            "n4m_split_binned_strat_group_kfold_n_splits",
        )
        return int(out.value)

    def split_fold(self, y, groups, fold_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        y = as_f64_2d(y)
        groups_arr = _as_i64_1d(groups, "groups")
        result = SplitResult()
        check(
            lib.n4m_split_binned_strat_group_kfold_split_fold(
                self._ensure_handle(),
                numpy_to_view(y),
                groups_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ctypes.c_int64(groups_arr.size),
                ctypes.c_int32(fold_idx),
                ctypes.byref(result),
            ),
            "n4m_split_binned_strat_group_kfold_split_fold",
        )
        return _copy_split_result(result)

    def split(self, y, groups):
        for fold_idx in range(self.get_n_splits()):
            yield self.split_fold(y, groups, fold_idx)


class SystematicCircularSplitter(_SplitHandle):
    """Systematic circular split over sorted or ordered targets."""

    _C_PREFIX = "n4m_split_systematic_circular"

    def __init__(self, test_size: float = 0.25, seed: int = 0):
        super().__init__()
        self.test_size = float(test_size)
        self.seed = int(seed)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_split_systematic_circular_create(
                ctypes.byref(h), ctypes.c_double(self.test_size),
                ctypes.c_uint64(self.seed)
            ),
            "n4m_split_systematic_circular_create",
        )
        return h

    def split(self, y) -> Tuple[np.ndarray, np.ndarray]:
        y = as_f64_2d(y)
        result = SplitResult()
        check(
            lib.n4m_split_systematic_circular_split(
                self._ensure_handle(), numpy_to_view(y), ctypes.byref(result)
            ),
            "n4m_split_systematic_circular_split",
        )
        return _copy_split_result(result)


class SPlitSplitter(_SplitHandle):
    """SPlit data-twinning splitter."""

    _C_PREFIX = "n4m_split_split_splitter"

    def __init__(self, test_size: float = 0.25, seed: int = 0):
        super().__init__()
        self.test_size = float(test_size)
        self.seed = int(seed)

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.n4m_split_split_splitter_create(
                ctypes.byref(h), ctypes.c_double(self.test_size),
                ctypes.c_uint64(self.seed)
            ),
            "n4m_split_split_splitter_create",
        )
        return h

    def split(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X = as_f64_2d(X)
        result = SplitResult()
        check(
            lib.n4m_split_split_splitter_split(
                self._ensure_handle(), numpy_to_view(X), ctypes.byref(result)
            ),
            "n4m_split_split_splitter_split",
        )
        return _copy_split_result(result)


SplitSplitter = SPlitSplitter


__all__ = [
    "BinnedStratifiedGroupKFoldSplitter",
    "KBinsStratifiedSplitter",
    "KennardStoneSplitter",
    "KMeansSplitter",
    "SPXYFoldSplitter",
    "SPXYGroupFoldSplitter",
    "SPXYSplitter",
    "SPlitSplitter",
    "SplitSplitter",
    "SystematicCircularSplitter",
]
