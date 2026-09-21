# SPDX-License-Identifier: CECILL-2.1
"""Native AOM calibrations with explicit branch, bank and search contracts.

All model calculations (including fold-local SNV/MSC, covariance screening,
Ridge and SIMPLS paths and coefficient folding) go through the C ABI. This
module constructs candidate descriptors and adapts arrays/results only.
Historical experimental methods keep their existing, separate entry points.
"""

from __future__ import annotations

import ctypes as ct
from itertools import permutations, product

import numpy as np

from .._errors import check
from .._ffi import lib
from .._impl.compat import BaseEstimator, RegressorMixin
from .._impl.native import _create_aom_operator_bank
from .._matrix import as_f64_2d, numpy_to_view
from .._types import MatrixView
from .._validation import _selector_fold_ids

# Versioned discrete bank: widths are sample counts, zero padding, no FD.
BANK_ID = "strict10-gaussian-v1"
BANK = (
    ("identity", ("identity", ())),
    ("sg_smooth_w11_p2", ("savgol_smooth", (11, 2))),
    ("sg_smooth_w21_p3", ("savgol_smooth", (21, 3))),
    ("sg_d1_w11_p2", ("savgol_derivative", (11, 2, 1))),
    ("sg_d1_w21_p3", ("savgol_derivative", (21, 3, 1))),
    ("sg_d2_w11_p2", ("savgol_derivative", (11, 2, 2))),
    ("detrend_d1", ("detrend_poly", (1,))),
    ("detrend_d2", ("detrend_poly", (2,))),
    ("gauss_d0_s1", ("gaussian", (1.0, 4.0, 0.0))),
    ("gauss_d0_s2", ("gaussian", (2.0, 4.0, 0.0))),
)


def strict_chain_bank(max_depth: int = 1):
    """Return the 10/62/206 typed chain universe in signature order.

    Each role (smoother, derivative, detrend) occurs at most once. All orders
    of distinct roles are permitted; identity is a separate singleton.
    """
    if max_depth not in (1, 2, 3):
        raise ValueError("max_depth must be 1, 2 or 3")
    roles = ((1, 2, 8, 9), (3, 4, 5), (6, 7))
    chains = [(0,)]
    for depth in range(1, max_depth + 1):
        for order in permutations(range(3), depth):
            chains.extend(product(*(roles[r] for r in order)))
    chains.sort(key=lambda chain: (len(chain), ">".join(BANK[i][0] for i in chain)))
    return tuple(tuple(BANK[i] for i in chain) for chain in chains)


def _ptr(array, ctype):
    return array.ctypes.data_as(ct.POINTER(ctype))


class _Calibration(RegressorMixin, BaseEstimator):
    _head = "pls"
    _fast = False
    _bank_id = BANK_ID
    _ridge_absolute_head = 1

    def __init__(
        self,
        *,
        max_components: int = 25,
        alphas=None,
        cv: int = 3,
        fold_ids=None,
        branches=("raw", "snv", "msc"),
        max_depth: int = 1,
        rank: int = 200,
    ):
        self.max_components = max_components
        self.alphas = alphas
        self.cv = cv
        self.fold_ids = fold_ids
        self.branches = branches
        self.max_depth = max_depth
        self.rank = rank

    def _candidate_chains(self):
        return strict_chain_bank(self.max_depth)

    def fit(self, X, y):
        X = as_f64_2d(X)
        y = np.asarray(y, dtype=np.float64)
        self._y_was_1d = y.ndim == 1
        if y.ndim == 1:
            y = y[:, None]
        if y.shape != (X.shape[0], 1):
            raise ValueError("AOM calibration requires one target per sample")
        y = np.ascontiguousarray(y)
        if not self.branches or len(set(self.branches)) != len(self.branches):
            raise ValueError("branches must be nonempty and unique")
        try:
            branches = np.array(
                [{"raw": 0, "snv": 1, "msc": 2}[v] for v in self.branches],
                dtype=np.int32,
            )
        except KeyError as error:
            raise ValueError("branches must be raw, snv or msc") from error
        if self.max_components < 1 or self.rank < 1:
            raise ValueError("max_components and rank must be positive")
        folds = _selector_fold_ids(len(X), self.cv, self.fold_ids)
        chains = self._candidate_chains()
        ops = [descriptor for chain in chains for _, descriptor in chain]
        offsets = np.asarray([0, *np.cumsum([len(c) for c in chains])], dtype=np.int32)
        if self._head == "ridge":
            alphas = np.ascontiguousarray(
                np.logspace(-6, 6, 50) if self.alphas is None else self.alphas,
                dtype=np.float64,
            )
            if alphas.ndim != 1 or not len(alphas):
                raise ValueError("alphas must be a nonempty 1D sequence")
            head = 2 if self.alphas is None else self._ridge_absolute_head
            count = len(alphas)
        else:
            alphas = np.empty(0, dtype=np.float64)
            head, count = 0, int(self.max_components)
        p = X.shape[1]
        coef, state = np.empty(p), np.empty(2 * p + 2)
        scores = np.empty(count if self._fast else len(branches) * len(chains) * count)
        selected = np.empty(3, dtype=np.int32)
        ctx, bank = ct.c_void_p(), ct.c_void_p()
        fit = lib.n4m_model_selection_aom_calibration_fit
        fit.restype = ct.c_int
        fit.argtypes = [
            ct.c_void_p,
            ct.POINTER(MatrixView),
            ct.POINTER(MatrixView),
            ct.c_void_p,
            ct.POINTER(ct.c_int32),
            ct.c_int32,
            ct.POINTER(ct.c_int32),
            ct.c_int32,
            ct.POINTER(ct.c_int32),
            ct.c_int32,
            ct.c_int32,
            ct.c_int32,
            ct.c_int32,
            ct.POINTER(ct.c_double),
            ct.c_int32,
            ct.POINTER(ct.c_double),
            ct.c_int64,
            ct.POINTER(ct.c_double),
            ct.c_int64,
            ct.POINTER(ct.c_double),
            ct.c_int64,
            ct.POINTER(ct.c_int32),
        ]
        try:
            check(lib.n4m_context_create(ct.byref(ctx)), "n4m_context_create")
            bank = _create_aom_operator_bank(ops)
            xv, yv = numpy_to_view(X), numpy_to_view(y)
            check(
                fit(
                    ctx,
                    ct.byref(xv),
                    ct.byref(yv),
                    bank,
                    _ptr(offsets, ct.c_int32),
                    len(chains),
                    _ptr(branches, ct.c_int32),
                    len(branches),
                    _ptr(folds, ct.c_int32),
                    self.cv,
                    head,
                    int(self._fast),
                    self.rank,
                    _ptr(alphas, ct.c_double),
                    count,
                    _ptr(coef, ct.c_double),
                    len(coef),
                    _ptr(state, ct.c_double),
                    len(state),
                    _ptr(scores, ct.c_double),
                    len(scores),
                    _ptr(selected, ct.c_int32),
                ),
                "aom_calibration_fit",
            )
        finally:
            if bank.value:
                lib.n4m_operator_bank_destroy(bank)
            if ctx.value:
                lib.n4m_context_destroy(ctx)
        b, c, a = map(int, selected)
        self.coef_, self.intercept_, self._state = coef, float(state[2 * p]), state
        self.selected_branch_ = tuple(self.branches)[b]
        self._branch_kind = int(branches[b])
        self.selected_chain_ = ">".join(name for name, _ in chains[c])
        self.selected_chain_index_ = c
        self.selected_parameter_index_ = a
        self.selected_n_components_ = a + 1 if self._head == "pls" else None
        self.selected_alpha_ = (
            float(alphas[a] * state[-1]) if self._head == "ridge" else None
        )
        self.cv_scores_ = (
            scores if self._fast else scores.reshape(len(branches), len(chains), count)
        )
        self.n_features_in_ = p
        self.bank_id_ = self._bank_id
        self.protocol_id_ = (
            f"{'fast' if self._fast else 'global'}-{self._head}-branch-cv-v1"
        )
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise RuntimeError("predict called before fit")
        X = as_f64_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X feature count differs from fit")
        output = np.empty((len(X), 1))
        if not len(X):
            return output.ravel() if self._y_was_1d else output
        fn = lib.n4m_model_selection_aom_calibration_predict
        fn.restype = ct.c_int
        fn.argtypes = [
            ct.POINTER(MatrixView),
            ct.c_int32,
            ct.POINTER(ct.c_double),
            ct.c_int64,
            ct.POINTER(ct.c_double),
            ct.c_int64,
            ct.POINTER(MatrixView),
        ]
        xv, ov = numpy_to_view(X), numpy_to_view(output)
        check(
            fn(
                ct.byref(xv),
                self._branch_kind,
                _ptr(self.coef_, ct.c_double),
                len(self.coef_),
                _ptr(self._state, ct.c_double),
                len(self._state),
                ct.byref(ov),
            ),
            "aom_calibration_predict",
        )
        return output.ravel() if self._y_was_1d else output

    def get_diagnostics(self):
        if not hasattr(self, "coef_"):
            raise RuntimeError("get_diagnostics called before fit")
        return {
            "protocol": self.protocol_id_,
            "bank": self.bank_id_,
            "backend": "native-cpu",
            "branch": self.selected_branch_,
            "chain": self.selected_chain_,
            "n_components": self.selected_n_components_,
            "alpha": self.selected_alpha_,
            "affine_raw_input": self.selected_branch_ == "raw",
            "rank": self.rank,
            "svd": "exact-leading-r-through-gram" if self._fast else None,
        }


class AOMPLSRegressor(_Calibration):
    """Global branch/operator/prefix selection with one SIMPLS path per candidate."""


class AOMRidgeRegressor(_Calibration):
    """Global Ridge selection; default alpha grid is raw-training trace-relative."""

    _head = "ridge"


class FastAOMPLSRegressor(_Calibration):
    """Fold-local covariance leader screening followed by one SIMPLS path."""

    _fast = True


class FastAOMRidgeRegressor(AOMRidgeRegressor):
    """Experimental Ridge extension of leader screening; separate from PLS evidence."""

    _fast = True


class _SingleViewRidge(AOMRidgeRegressor):
    """Internal stack adapter to the native smaller-Gram calibration path."""

    _bank_id = "explicit-single-view"
    _ridge_absolute_head = 3

    def _candidate_chains(self):
        return (((str(self._operator), self._operator),),)


def _fit_single_view_ridge(X, y, operator, alphas, cv, fold_ids=None):
    model = _SingleViewRidge(branches=("raw",), alphas=alphas, cv=cv, fold_ids=fold_ids)
    model._operator = operator
    model.fit(X, y)
    return {
        "input_coefficients": model.coef_[:, None],
        "intercept": np.array([[model.intercept_]]),
        "selected_param": model.selected_alpha_,
    }


__all__ = [
    "BANK_ID",
    "AOMPLSRegressor",
    "AOMRidgeRegressor",
    "FastAOMPLSRegressor",
    "FastAOMRidgeRegressor",
    "strict_chain_bank",
]
