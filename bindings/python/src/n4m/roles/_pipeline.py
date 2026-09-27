# SPDX-License-Identifier: CECILL-2.1
"""scikit-learn facade over the native role pipeline (ABI 2.14).

A :class:`RolePipeline` is a trained linear recipe of catalog estimators:
sample filters (training rows only), transformers / selectors, then one
regressor or classifier. Recipe validation, fit-input routing, the
feature-name check and the per-step N4ME states are native
(``n4m_role_pipeline_*``); this module only translates Python objects.
Classifier label names stay here, as in :class:`NativeClassifier`.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from .._errors import N4MError, check
from .._ffi import lib
from .._matrix import as_f64_2d, numpy_to_view
from .._types import Status
from ._base import _Context, _fit_inputs, method_class

_ROLE_NAMES = {
    1 << 0: "transformer",
    1 << 1: "regressor",
    1 << 2: "classifier",
    1 << 3: "selector",
    1 << 4: "sample_filter",
}
_ROLE_CLASSIFIER = 1 << 2
_EXPORT_ALLOW_TRAINING_ROWS = 1


class _StepInfoV1(ctypes.Structure):
    """``n4m_role_pipeline_step_info_v1_t`` (``n4m/estimator.h``)."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("method_index", ctypes.c_int32),
        ("method_id", ctypes.c_char_p),
        ("role", ctypes.c_uint32),
        ("state_index", ctypes.c_int32),
        ("contains_training_rows", ctypes.c_int32),
        ("n_features_in", ctypes.c_int64),
        ("n_features_out", ctypes.c_int64),
    ]


def _parse_step(step: Any) -> tuple[str, dict[str, Any]]:
    """``(method_id, params)`` of a step token.

    Accepted: ``"<method id>"`` or ``"n4m:<method id>"``, ``(method_id, params)``
    and ``{"class": "n4m:<method id>", "params": {...}}``.
    """
    if isinstance(step, str):
        return step.removeprefix("n4m:"), {}
    if isinstance(step, dict):
        return _parse_step(step["class"])[0], dict(step.get("params") or {})
    method_id, params = step
    return _parse_step(method_id)[0], dict(params or {})


def _label_ids(y) -> tuple[np.ndarray, np.ndarray | None]:
    """Class ids for the core, and the label names when they are not integers."""
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError(f"class labels must be 1-D, got shape {labels.shape}")
    if labels.dtype.kind in "iu":
        return np.ascontiguousarray(labels, dtype=np.int64), None
    names, codes = np.unique(labels, return_inverse=True)
    return np.ascontiguousarray(codes, dtype=np.int64), names


def _column_names(X) -> list[str] | None:
    return [str(c) for c in X.columns] if hasattr(X, "columns") else None


class RolePipeline(BaseEstimator):
    """Native trained recipe of role steps, portable as N4ME states.

    Args:
        steps: Step tokens: method ids (``"models.pls.cppls"`` or
            ``"n4m:models.pls.cppls"``), ``(method_id, params)`` pairs or
            ``{"class": "n4m:<id>", "params": {...}}`` recipe tokens.

    ``fit`` takes the targets of the final step: responses for a regressor,
    class labels for a classifier (non-terminal steps that require ``y`` then
    receive the class ids). DataFrame column names are stored and checked at
    every later call; arrays are positional.
    """

    def __init__(self, steps=()):
        self.steps = steps

    @classmethod
    def from_steps(cls, steps) -> RolePipeline:
        """An unfitted pipeline of ``steps`` (validated natively)."""
        pipeline = cls(list(steps))
        pipeline._release(pipeline._create())
        return pipeline

    @classmethod
    def from_states(
        cls,
        steps,
        states,
        feature_names=None,
        class_names=None,
    ) -> RolePipeline:
        """A fitted pipeline rebuilt from one N4ME state per stateful step.

        The native import refuses states that contradict ``steps`` (count,
        method, parameters, role, widths). ``class_names`` restores the label
        names of a classifier trained on non-integer labels.
        """
        pipeline = cls(list(steps))
        handle = pipeline._create()
        try:
            payloads = [bytes(s[1] if isinstance(s, tuple) else s) for s in states]
            buffers = [(ctypes.c_ubyte * len(p)).from_buffer_copy(p) for p in payloads]
            with _Context() as ctx:
                pipeline._set_feature_names(ctx, handle, feature_names)
                ctx.check(
                    lib.n4m_role_pipeline_import_states(
                        ctx.handle,
                        handle,
                        len(buffers),
                        (ctypes.c_void_p * len(buffers))(
                            *(ctypes.addressof(b) for b in buffers)
                        ),
                        (ctypes.c_size_t * len(buffers))(*(len(p) for p in payloads)),
                    ),
                    "RolePipeline.from_states",
                )
        except BaseException:
            pipeline._release(handle)
            raise
        pipeline._publish(
            handle,
            None if feature_names is None else [str(n) for n in feature_names],
            None if class_names is None else np.asarray(class_names),
            None,
        )
        return pipeline

    # -- native handle --------------------------------------------------------

    def _create(self) -> ctypes.c_void_p:
        parsed = [_parse_step(step) for step in self.steps]
        handle = ctypes.c_void_p()
        with _Context() as ctx:
            params = [
                method_class(method_id)(**values)._native_params(ctx)
                for method_id, values in parsed
            ]
            try:
                ids = (ctypes.c_char_p * len(parsed))(*(m.encode() for m, _ in parsed))
                ctx.check(
                    lib.n4m_role_pipeline_create(
                        ctx.handle,
                        len(parsed),
                        ids,
                        (ctypes.c_void_p * len(parsed))(*params),
                        ctypes.byref(handle),
                    ),
                    "RolePipeline",
                )
            finally:
                for p in params:
                    lib.n4m_params_destroy(p)
        return handle

    @staticmethod
    def _set_feature_names(ctx: _Context, handle, names) -> None:
        if names is None:
            return
        encoded = [str(n).encode() for n in names]
        ctx.check(
            lib.n4m_role_pipeline_set_feature_names(
                ctx.handle,
                handle,
                (ctypes.c_char_p * len(encoded))(*encoded),
                len(encoded),
            ),
            "feature names",
        )

    @staticmethod
    def _release(handle) -> None:
        if handle is not None and handle.value is not None:
            lib.n4m_role_pipeline_destroy(handle)

    def _publish(self, handle, names, label_names, y_1d) -> None:
        """Replaces the fitted state only once the native call succeeded."""
        self._release(self.__dict__.get("_handle_"))
        self._handle_ = handle
        if names is None:
            self.__dict__.pop("feature_names_in_", None)
        else:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        self._label_names_ = label_names
        n = ctypes.c_int64()
        check(
            lib.n4m_role_pipeline_n_features_in(handle, ctypes.byref(n)),
            "n_features_in",
        )
        self.n_features_in_ = int(n.value)
        check(lib.n4m_role_pipeline_n_outputs(handle, ctypes.byref(n)), "n_outputs")
        # One output column comes back 1-D unless the fit target was a matrix.
        self._y_1d_ = n.value == 1 if y_1d is None else y_1d

    def _handle(self) -> ctypes.c_void_p:
        handle = self.__dict__.get("_handle_")
        if handle is None:
            raise N4MError(Status.ERR_NOT_FITTED, "RolePipeline is not fitted")
        return handle

    def __del__(self) -> None:
        self._release(self.__dict__.get("_handle_"))

    # -- fit --------------------------------------------------------------------

    def fit(self, X, y=None, **inputs) -> RolePipeline:
        """Fit every step natively.

        Extra keyword inputs (``sample_weight``, ``groups``, ``feature_groups``,
        ``blocks``, ``axis``, ``X_target``, ``fold_ids``) reach the steps whose
        manifest declares them; the core refuses those no step uses.
        """
        handle = self._create()
        try:
            classifier = self._terminal_role(handle) == _ROLE_CLASSIFIER
            labels = label_names = None
            if classifier and y is not None:
                labels, label_names = _label_ids(y)
                y = None
            y_1d = y is not None and np.ndim(y) == 1
            names = _column_names(X)
            keep: list[Any] = []
            fit_inputs = _fit_inputs(keep, X, y, labels=labels, **inputs)
            with _Context() as ctx:
                self._set_feature_names(ctx, handle, names)
                ctx.check(
                    lib.n4m_role_pipeline_fit(
                        ctx.handle, handle, ctypes.addressof(fit_inputs)
                    ),
                    "RolePipeline.fit",
                )
            del keep
        except BaseException:
            self._release(handle)
            raise
        self._publish(handle, names, label_names, y_1d if not classifier else None)
        return self

    @staticmethod
    def _terminal_role(handle) -> int:
        n = ctypes.c_int32()
        check(lib.n4m_role_pipeline_n_steps(handle, ctypes.byref(n)), "n_steps")
        return RolePipeline._step_info(handle, n.value - 1).role

    @staticmethod
    def _step_info(handle, step: int) -> _StepInfoV1:
        info = _StepInfoV1()
        info.struct_size = ctypes.sizeof(_StepInfoV1)
        check(
            lib.n4m_role_pipeline_step_info_v1(handle, step, ctypes.byref(info)),
            "step_info",
        )
        return info

    # -- operations -------------------------------------------------------------

    def _rows(self, ctx: _Context, X) -> np.ndarray:
        """X as float64 rows, after the native width and feature-name check."""
        names = _column_names(X)
        X_arr = as_f64_2d(X)
        encoded = (
            None
            if names is None
            else (ctypes.c_char_p * len(names))(*(n.encode() for n in names))
        )
        ctx.check(
            lib.n4m_role_pipeline_check_features(
                ctx.handle, self._handle(), X_arr.shape[1], encoded
            ),
            "RolePipeline: input columns",
        )
        return X_arr

    def _matrix(self, symbol: str, width_symbol: str, X) -> np.ndarray:
        handle = self._handle()
        cols = ctypes.c_int64()
        check(getattr(lib, width_symbol)(handle, ctypes.byref(cols)), width_symbol)
        with _Context() as ctx:
            X_arr = self._rows(ctx, X)
            out = np.empty((X_arr.shape[0], cols.value), dtype=np.float64)
            X_view, out_view = numpy_to_view(X_arr), numpy_to_view(out)
            ctx.check(
                getattr(lib, symbol)(
                    ctx.handle, handle, ctypes.byref(X_view), ctypes.byref(out_view)
                ),
                symbol,
            )
        return out

    def transform(self, X) -> np.ndarray:
        """Rows after the transformers and selectors (the final step's input)."""
        return self._matrix(
            "n4m_role_pipeline_transform", "n4m_role_pipeline_transform_cols", X
        )

    def predict(self, X) -> np.ndarray:
        """Predictions of a regressor, or class labels of a classifier."""
        handle = self._handle()
        if self._terminal_role(handle) == _ROLE_CLASSIFIER:
            with _Context() as ctx:
                X_arr = self._rows(ctx, X)
                ids = np.empty(X_arr.shape[0], dtype=np.int64)
                X_view = numpy_to_view(X_arr)
                ctx.check(
                    lib.n4m_role_pipeline_predict_labels(
                        ctx.handle,
                        handle,
                        ctypes.byref(X_view),
                        ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                        ids.size,
                    ),
                    "n4m_role_pipeline_predict_labels",
                )
            return ids if self._label_names_ is None else self._label_names_[ids]
        out = self._matrix(
            "n4m_role_pipeline_predict", "n4m_role_pipeline_n_outputs", X
        )
        return out.ravel() if self._y_1d_ and out.shape[1] == 1 else out

    def decision_function(self, X) -> np.ndarray:
        """Class scores of the final classifier, one column per class."""
        return self._matrix(
            "n4m_role_pipeline_decision_function", "n4m_role_pipeline_n_outputs", X
        )

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, for final classifiers that define them."""
        return self._matrix(
            "n4m_role_pipeline_predict_proba", "n4m_role_pipeline_n_outputs", X
        )

    @property
    def classes_(self) -> np.ndarray:
        handle = self._handle()
        count = ctypes.c_int64()
        check(
            lib.n4m_role_pipeline_classes(handle, None, 0, ctypes.byref(count)),
            "classes",
        )
        ids = np.empty(count.value, dtype=np.int64)
        check(
            lib.n4m_role_pipeline_classes(
                handle,
                ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                count,
                ctypes.byref(count),
            ),
            "classes",
        )
        return ids if self._label_names_ is None else self._label_names_[ids]

    # -- introspection and states -------------------------------------------------

    @property
    def steps_info_(self) -> list[dict[str, Any]]:
        """Per step: method id, role played, state index (-1: filter), widths, training rows."""
        handle = self._handle()
        n = ctypes.c_int32()
        check(lib.n4m_role_pipeline_n_steps(handle, ctypes.byref(n)), "n_steps")
        out = []
        for step in range(n.value):
            info = self._step_info(handle, step)
            out.append(
                {
                    "method_id": info.method_id.decode(),
                    "role": _ROLE_NAMES[info.role],
                    "state_index": info.state_index,
                    "contains_training_rows": bool(info.contains_training_rows),
                    "n_features_in": info.n_features_in,
                    "n_features_out": info.n_features_out,
                }
            )
        return out

    def export_states(
        self, *, allow_training_rows: bool = False
    ) -> list[tuple[str, bytes, bool]]:
        """``(method_id, N4ME bytes, contains_training_rows)`` per stateful step.

        A state that embeds training rows is refused unless
        ``allow_training_rows`` is set.
        """
        handle = self._handle()
        flags = _EXPORT_ALLOW_TRAINING_ROWS if allow_training_rows else 0
        states = []
        n = ctypes.c_int32()
        check(lib.n4m_role_pipeline_n_steps(handle, ctypes.byref(n)), "n_steps")
        with _Context() as ctx:
            for step in range(n.value):
                info = self._step_info(handle, step)
                if info.state_index < 0:
                    continue  # sample filters are train-only
                size = ctypes.c_size_t()
                ctx.check(
                    lib.n4m_role_pipeline_export_state_size(
                        ctx.handle, handle, info.state_index, flags, ctypes.byref(size)
                    ),
                    "export_states",
                )
                buf = (ctypes.c_ubyte * size.value)()
                written = ctypes.c_size_t()
                ctx.check(
                    lib.n4m_role_pipeline_export_state_to_buffer(
                        ctx.handle,
                        handle,
                        info.state_index,
                        flags,
                        ctypes.addressof(buf),
                        size,
                        ctypes.byref(written),
                    ),
                    "export_states",
                )
                states.append(
                    (
                        info.method_id.decode(),
                        bytes(buf)[: written.value],
                        bool(info.contains_training_rows),
                    )
                )
        return states

    def __getstate__(self) -> dict[str, Any]:
        state = {k: v for k, v in self.__dict__.items() if k != "_handle_"}
        if self.__dict__.get("_handle_") is not None:
            state["_n4me_states_"] = [
                s[1] for s in self.export_states(allow_training_rows=True)
            ]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        payloads = state.pop("_n4me_states_", None)
        self.__dict__.update(state)
        self._handle_ = None
        if payloads is not None:
            names = state.get("feature_names_in_")
            restored = RolePipeline.from_states(
                self.steps,
                payloads,
                None if names is None else list(names),
                state.get("_label_names_"),
            )
            self._handle_, restored._handle_ = restored._handle_, None
