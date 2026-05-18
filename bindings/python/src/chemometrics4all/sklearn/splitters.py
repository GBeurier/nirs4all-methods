# SPDX-License-Identifier: CECILL-2.1
"""Train / test splitter wrappers.

Each splitter exposes a single :meth:`split` method that returns a pair of
1-D :class:`numpy.ndarray` (``int64``) carrying the train and test indices.

The libc4a side allocates the index arrays via :c:struct:`c4a_split_result_t`;
the binding copies them into NumPy arrays and releases the C-side memory via
``c4a_split_result_destroy`` before returning.
"""
from __future__ import annotations

import ctypes
from typing import Tuple

import numpy as np

from .._errors import check
from .._ffi import lib
from .._matrix import as_f64_2d, numpy_to_view
from .._types import SplitResult

# Strategy enums (mirror c4a_split_kbins_strategy_t).
_KBINS_STRATEGIES = {"uniform": 0, "quantile": 1, 0: 0, 1: 1}


def _copy_split_result(result: SplitResult) -> Tuple[np.ndarray, np.ndarray]:
    """Copy ``c4a_split_result_t`` index arrays into NumPy and free the C side."""
    train = np.fromiter(
        (result.train_idx[i] for i in range(result.n_train)),
        dtype=np.int64, count=result.n_train,
    )
    test = np.fromiter(
        (result.test_idx[i] for i in range(result.n_test)),
        dtype=np.int64, count=result.n_test,
    )
    lib.c4a_split_result_destroy(ctypes.byref(result))
    return train, test


class KennardStoneSplitter:
    """Kennard-Stone train/test split.

    Picks the most diverse samples for the training set, in descending order
    of pairwise Euclidean distance.
    """

    def __init__(self, test_size: float = 0.25):
        self.test_size = float(test_size)
        self._handle: ctypes.c_void_p | None = None

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_split_kennard_stone_create(
                ctypes.byref(h), ctypes.c_double(self.test_size)
            ),
            "c4a_split_kennard_stone_create",
        )
        return h

    def split(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X = as_f64_2d(X)
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        x_view = numpy_to_view(X)
        result = SplitResult()
        check(
            lib.c4a_split_kennard_stone_split(self._handle, x_view, ctypes.byref(result)),
            "c4a_split_kennard_stone_split",
        )
        return _copy_split_result(result)

    def __del__(self):
        if self._handle is not None and self._handle.value:
            lib.c4a_split_kennard_stone_destroy(self._handle)
            self._handle = None


class SPXYSplitter:
    """SPXY (Sample set Partitioning based on X and Y) train/test split."""

    def __init__(self, test_size: float = 0.25):
        self.test_size = float(test_size)
        self._handle: ctypes.c_void_p | None = None

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_split_spxy_create(ctypes.byref(h), ctypes.c_double(self.test_size)),
            "c4a_split_spxy_create",
        )
        return h

    def split(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        X = as_f64_2d(X)
        y = as_f64_2d(y)
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        x_view = numpy_to_view(X)
        y_view = numpy_to_view(y)
        result = SplitResult()
        check(
            lib.c4a_split_spxy_split(self._handle, x_view, y_view, ctypes.byref(result)),
            "c4a_split_spxy_split",
        )
        return _copy_split_result(result)

    def __del__(self):
        if self._handle is not None and self._handle.value:
            lib.c4a_split_spxy_destroy(self._handle)
            self._handle = None


class KBinsStratifiedSplitter:
    """Stratified split using K equal-width or quantile bins of ``y``."""

    def __init__(self, test_size: float = 0.25, seed: int = 0,
                 n_bins: int = 5, strategy: str | int = "uniform"):
        self.test_size = float(test_size)
        self.seed = int(seed)
        self.n_bins = int(n_bins)
        try:
            self.strategy = _KBINS_STRATEGIES[strategy]
        except KeyError as exc:
            raise ValueError(f"Unknown KBins strategy: {strategy!r}") from exc
        self._handle: ctypes.c_void_p | None = None

    def _create_handle(self):
        h = ctypes.c_void_p()
        check(
            lib.c4a_split_kbins_stratified_create(
                ctypes.byref(h),
                ctypes.c_double(self.test_size),
                ctypes.c_uint64(self.seed),
                ctypes.c_int32(self.n_bins),
                ctypes.c_int32(self.strategy),
            ),
            "c4a_split_kbins_stratified_create",
        )
        return h

    def split(self, y) -> Tuple[np.ndarray, np.ndarray]:
        y = as_f64_2d(y)
        if self._handle is None or not self._handle.value:
            self._handle = self._create_handle()
        y_view = numpy_to_view(y)
        result = SplitResult()
        check(
            lib.c4a_split_kbins_stratified_split(
                self._handle, y_view, ctypes.byref(result)
            ),
            "c4a_split_kbins_stratified_split",
        )
        return _copy_split_result(result)

    def __del__(self):
        if self._handle is not None and self._handle.value:
            lib.c4a_split_kbins_stratified_destroy(self._handle)
            self._handle = None


__all__ = ["KBinsStratifiedSplitter", "KennardStoneSplitter", "SPXYSplitter"]
