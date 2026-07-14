# SPDX-License-Identifier: CECILL-2.1
"""Owning Python façade for native validation plans."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable

import numpy as np

from ._errors import check
from ._ffi import lib


def _selector_fold_ids(n_samples: int, cv: int, fold_ids) -> np.ndarray:
    """Normalize the canonical n4m fold-id syntax.

    This is shared by the historical AOM/selection wrappers and the public
    :class:`ValidationPlan`; it remains the single Python authority for fold-id
    validation and default deterministic fold assignment.
    """

    if fold_ids is None:
        if cv < 2 or cv > n_samples:
            raise ValueError(
                "cv must be in [2, n_samples] when fold_ids are not provided"
            )
        return np.ascontiguousarray(
            [(i * int(cv)) // int(n_samples) for i in range(n_samples)],
            dtype=np.int32,
        )
    raw = np.asarray(fold_ids)
    if raw.dtype.kind not in {"i", "u"}:
        raise TypeError("fold_ids must contain integers")
    flat = raw.reshape(-1)
    if flat.size and (np.any(flat < 0) or np.any(flat > np.iinfo(np.int32).max)):
        raise ValueError("fold_ids must be in [0, INT32_MAX]")
    folds = np.ascontiguousarray(flat, dtype=np.int32)
    if folds.size != n_samples:
        raise ValueError("fold_ids length must match X.shape[0]")
    inferred_cv = int(folds.max()) + 1 if folds.size else 0
    actual_cv = int(cv) if cv and int(cv) > 0 else inferred_cv
    if actual_cv < 2 or actual_cv > n_samples:
        raise ValueError("inferred/provided cv must be in [2, n_samples]")
    if np.any(folds >= actual_cv):
        raise ValueError("fold_ids contain id >= cv")
    return folds


def _create_validation_plan_from_fold_ids(
    fold_ids: np.ndarray, cv: int
) -> ctypes.c_void_p:
    """Create an owning native handle from already-normalized fold ids."""

    n_samples = int(fold_ids.size)
    plan = ctypes.c_void_p()
    try:
        check(
            lib.n4m_validation_plan_create(ctypes.byref(plan)),
            "n4m_validation_plan_create",
        )
        check(
            lib.n4m_validation_plan_set_n_samples(plan, ctypes.c_int64(n_samples)),
            "n4m_validation_plan_set_n_samples",
        )
        all_idx = np.arange(n_samples, dtype=np.int64)
        for fold in range(int(cv)):
            test = np.ascontiguousarray(all_idx[fold_ids == fold], dtype=np.int64)
            train = np.ascontiguousarray(all_idx[fold_ids != fold], dtype=np.int64)
            if test.size == 0 or train.size == 0:
                raise ValueError(
                    "every fold must contain test rows and leave train rows"
                )
            check(
                lib.n4m_validation_plan_add_fold(
                    plan,
                    train.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    ctypes.c_int64(train.size),
                    test.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    ctypes.c_int64(test.size),
                ),
                "n4m_validation_plan_add_fold",
            )
    except BaseException:
        if plan.value:
            lib.n4m_validation_plan_destroy(plan)
        raise
    return plan


def _indices(values, *, name: str, n_samples: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if raw.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain integer sample indices")
    indices = np.ascontiguousarray(raw, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= n_samples):
        raise ValueError(f"{name} contains an index outside [0, n_samples)")
    return indices


class ValidationPlan:
    """Owning native validation plan used by :func:`finetune_estimator`.

    Build a plan with :meth:`from_fold_ids` for the common case, or
    :meth:`from_splits` for explicit train/test indices. Native calls copy every
    index array; the input sequences may be released immediately. The plan is a
    context manager and must not be used after :meth:`close`.
    """

    def __init__(self, n_samples: int) -> None:
        if isinstance(n_samples, bool) or not isinstance(n_samples, int):
            raise TypeError("n_samples must be an int")
        if n_samples < 2 or n_samples > 2**63 - 1:
            raise ValueError("n_samples must be in [2, INT64_MAX]")
        handle = ctypes.c_void_p()
        check(
            lib.n4m_validation_plan_create(ctypes.byref(handle)),
            "n4m_validation_plan_create",
        )
        try:
            check(
                lib.n4m_validation_plan_set_n_samples(
                    handle, ctypes.c_int64(n_samples)
                ),
                "n4m_validation_plan_set_n_samples",
            )
        except BaseException:
            lib.n4m_validation_plan_destroy(handle)
            raise
        self._handle = handle
        self.n_samples = n_samples
        self.n_folds = 0

    @classmethod
    def from_fold_ids(cls, fold_ids, *, cv: int | None = None) -> "ValidationPlan":
        """Create a plan from one zero-based test-fold id per sample.

        When ``cv`` is omitted it is inferred as ``max(fold_ids) + 1``. Every
        fold must contain test rows and leave at least one training row.
        """

        if cv is not None and (isinstance(cv, bool) or not isinstance(cv, int)):
            raise TypeError("cv must be an int or None")
        raw = np.asarray(fold_ids)
        if raw.ndim != 1:
            raise ValueError("fold_ids must be a 1-D sequence")
        if raw.dtype.kind not in {"i", "u"}:
            raise TypeError("fold_ids must contain integers")
        n_samples = int(raw.size)
        if cv is not None and (cv < 2 or cv > n_samples):
            raise ValueError("cv must be in [2, n_samples] when provided")
        requested_cv = 0 if cv is None else cv
        folds = _selector_fold_ids(n_samples, requested_cv, raw)
        actual_cv = requested_cv if requested_cv > 0 else int(folds.max()) + 1
        handle = _create_validation_plan_from_fold_ids(folds, actual_cv)
        plan = cls.__new__(cls)
        plan._handle = handle
        plan.n_samples = n_samples
        plan.n_folds = actual_cv
        return plan

    @classmethod
    def from_splits(
        cls,
        n_samples: int,
        splits: Iterable[tuple[object, object]],
    ) -> "ValidationPlan":
        """Create a plan from ordered ``(train_indices, test_indices)`` pairs."""

        plan = cls(n_samples)
        try:
            count = 0
            for train, test in splits:
                plan.add_fold(train, test)
                count += 1
            if count == 0:
                raise ValueError("splits must contain at least one fold")
            return plan
        except BaseException:
            plan.close()
            raise

    def _require_open(self) -> None:
        if not getattr(self, "_handle", None) or not self._handle.value:
            raise RuntimeError("ValidationPlan is closed")

    def add_fold(self, train_indices, test_indices) -> "ValidationPlan":
        """Append one fold; native preflight later enforces global coverage."""

        self._require_open()
        train = _indices(train_indices, name="train_indices", n_samples=self.n_samples)
        test = _indices(test_indices, name="test_indices", n_samples=self.n_samples)
        check(
            lib.n4m_validation_plan_add_fold(
                self._handle,
                train.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ctypes.c_int64(train.size),
                test.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ctypes.c_int64(test.size),
            ),
            "n4m_validation_plan_add_fold",
        )
        self.n_folds += 1
        return self

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle and handle.value:
            lib.n4m_validation_plan_destroy(handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "ValidationPlan":
        self._require_open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __copy__(self):
        raise TypeError("ValidationPlan is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("ValidationPlan is not copyable")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["ValidationPlan"]
