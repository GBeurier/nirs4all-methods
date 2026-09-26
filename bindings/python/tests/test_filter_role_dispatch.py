# SPDX-License-Identifier: CECILL-2.1
"""The closed C role handles reuse the existing numerical filter kernels."""

import ctypes

import numpy as np
import pytest
from n4m._ffi import lib
from n4m._matrix import numpy_to_view
from n4m._types import FilterStats, Status
from n4m.feature_selection.filter import CorrelationFilter, VarianceFilter
from n4m.outlier_detection import SpectralQualityFilter, YOutlierFilter


def _view(x):
    return numpy_to_view(np.ascontiguousarray(x, dtype=np.float64))


def _i(values):
    return (ctypes.c_int64 * len(values))(*values)


def _d(values):
    return (ctypes.c_double * len(values))(*values)


def test_row_mask_native_role_matches_named_python_oracle_on_holdout():
    x = np.arange(30, dtype=np.float64).reshape(10, 3) + np.array([1., 3., 6.])
    y = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 100.])[:, None]
    xv, yv = _view(x), _view(y)
    handle = ctypes.c_void_p()
    assert lib.n4m_sample_filter_create(0, _i([0]), 1, _d([1.5, 5., 95.]),
                                         3, 0, ctypes.byref(handle)) == Status.OK
    try:
        mask = np.zeros(3, dtype=np.uint8)
        stats = FilterStats()
        hold_x = _view(x[7:])
        hold_y = _view(y[7:])
        assert lib.n4m_sample_filter_apply(
            handle, ctypes.byref(hold_x), ctypes.byref(hold_y),
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(stats)) == Status.ERR_NOT_FITTED
        assert lib.n4m_sample_filter_fit(
            handle, ctypes.byref(xv), ctypes.byref(yv)) == Status.OK
        assert lib.n4m_sample_filter_apply(
            handle, ctypes.byref(hold_x), ctypes.byref(hold_y),
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(stats)) == Status.OK
        oracle, _ = YOutlierFilter(method="iqr", threshold=1.5,
                                   lower_percentile=5., upper_percentile=95.).fit(y).apply(y[7:])
        np.testing.assert_array_equal(mask, oracle)
        np.testing.assert_array_equal(mask, [1, 1, 0])
        assert stats.n_samples == 3 and stats.n_kept == 2
        wrong_y = _view(np.ones((3, 2)))
        assert lib.n4m_sample_filter_apply(
            handle, ctypes.byref(hold_x), ctypes.byref(wrong_y),
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(stats)) == Status.ERR_SHAPE_MISMATCH
    finally:
        lib.n4m_sample_filter_destroy(handle)


@pytest.mark.parametrize("kind,oracle", [
    (0, VarianceFilter(threshold=0.01)),
    (1, CorrelationFilter(threshold=0.01)),
])
def test_feature_filter_native_indices_and_holdout(kind, oracle):
    x = np.array([[1., 4., 2.], [2., 4., 3.], [3., 4., 4.],
                  [4., 4., 5.], [5., 4., 6.]])
    y = np.arange(1., 6.)[:, None]
    held = np.array([[6., 9., 7.], [7., 9., 8.]])
    xv, yv, hv = _view(x), _view(y), _view(held)
    handle = ctypes.c_void_p()
    assert lib.n4m_feature_filter_create(kind, 0.01, -1,
                                          ctypes.byref(handle)) == Status.OK
    try:
        count = ctypes.c_int64()
        assert lib.n4m_feature_filter_selected_indices(
            handle, None, 0, ctypes.byref(count)) == Status.ERR_NOT_FITTED
        yy = ctypes.byref(yv) if kind else None
        assert lib.n4m_feature_filter_fit(handle, ctypes.byref(xv), yy) == Status.OK
        assert lib.n4m_feature_filter_selected_indices(
            handle, None, 0, ctypes.byref(count)) == Status.OK
        indices = (ctypes.c_int64 * count.value)()
        assert lib.n4m_feature_filter_selected_indices(
            handle, indices, count.value, ctypes.byref(count)) == Status.OK
        assert list(indices) == [0, 2]
        output = np.empty((2, count.value), dtype=np.float64)
        ov = _view(output)
        assert lib.n4m_feature_filter_transform(
            handle, ctypes.byref(hv), ctypes.byref(ov)) == Status.OK
        expected = oracle.fit(x, y.ravel() if kind else None).transform(held)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(output, held[:, [0, 2]])
        bad_y = _view(np.ones((5, 2)))
        if kind:
            assert lib.n4m_feature_filter_fit(
                handle, ctypes.byref(xv), ctypes.byref(bad_y)) == Status.ERR_SHAPE_MISMATCH
    finally:
        lib.n4m_feature_filter_destroy(handle)


def test_quality_role_same_mask_as_direct_kernel():
    x = np.array([[1., 2., 3.], [0., 0., 0.], [4., 5., 6.]])
    xv = _view(x)
    h = ctypes.c_void_p()
    assert lib.n4m_sample_filter_create(3, _i([0, 0, 1]), 3,
                                         _d([0.1, 0.5, 1e-8, 0., 0.]), 5,
                                         0, ctypes.byref(h)) == Status.OK
    try:
        assert lib.n4m_sample_filter_fit(h, ctypes.byref(xv), None) == Status.OK
        mask = np.zeros(3, dtype=np.uint8)
        stats = FilterStats()
        assert lib.n4m_sample_filter_apply(
            h, ctypes.byref(xv), None,
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(stats)) == Status.OK
        expected, _ = SpectralQualityFilter().fit().apply(x)
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(mask, [1, 0, 1])
    finally:
        lib.n4m_sample_filter_destroy(h)
