# SPDX-License-Identifier: CECILL-2.1
"""scikit-learn facade over the generic native roles (ABI 2.13).

Every class in :mod:`n4m.roles` is a thin, generated subclass of a role base:
estimators (:class:`NativeEstimator` roles) with a fitted N4ME state, and
procedures (:class:`NativeSplitter`, :class:`NativeAugmenter`,
:class:`NativeProcedure`) that run once. Parameters, defaults, required
inputs and all numerics live in libn4m; this module only translates Python
objects to ``n4m_estimator_*`` / ``n4m_procedure_run`` calls.
"""

from __future__ import annotations

import ctypes
import math
import numbers
from typing import Any, ClassVar, Self

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.metaestimators import available_if

from .._errors import N4MError, check
from .._ffi import lib
from .._matrix import as_f64_2d, numpy_to_view
from .._types import FitInputsV1, MethodInfoV1, Status

CAP_TRANSFORM = 1 << 0
CAP_PREDICT = 1 << 1
CAP_PREDICT_PROBA = 1 << 2

_REGISTRY: dict[str, type[NativeMethod]] = {}


class _Context:
    """One short-lived native context whose error text enriches failures."""

    def __enter__(self) -> Self:
        self.handle = ctypes.c_void_p()
        check(lib.n4m_context_create(ctypes.byref(self.handle)), "n4m_context_create")
        return self

    def __exit__(self, *exc) -> None:
        lib.n4m_context_destroy(self.handle)

    def check(self, status: int, where: str) -> None:
        if status != Status.OK:
            detail = lib.n4m_context_last_error(self.handle)
            text = ctypes.string_at(detail).decode() if detail else ""
            raise N4MError(status, f"{where}: {text}" if text else where)


# Order of n4m_fit_input_t (n4m/estimator.h); X_target is the target domain.
_FIT_INPUT_NAMES = (
    "y",
    "labels",
    "sample_weight",
    "groups",
    "feature_groups",
    "blocks",
    "axis",
    "X_target",
    "fold_ids",
)


def method_class(method_id: str) -> type[NativeMethod]:
    """The :mod:`n4m.roles` class of a catalog method id."""
    try:
        return _REGISTRY[method_id]
    except KeyError:
        raise ValueError(f"no n4m role class for {method_id!r}") from None


def manifest() -> dict[str, Any]:
    """The native manifest: every method's roles, node kinds, inputs and typed parameters."""
    import json

    size = ctypes.c_size_t()
    check(lib.n4m_method_manifest_json(None, 0, ctypes.byref(size)), "manifest")
    buf = ctypes.create_string_buffer(size.value)
    check(lib.n4m_method_manifest_json(buf, size, ctypes.byref(size)), "manifest")
    return json.loads(buf.raw[: size.value])


def method_info(method_id: str) -> MethodInfoV1:
    """Native manifest entry of ``method_id``."""
    index = ctypes.c_int32()
    check(
        lib.n4m_method_find(method_id.encode(), ctypes.byref(index)),
        f"unknown method {method_id!r}",
    )
    info = MethodInfoV1()
    info.struct_size = ctypes.sizeof(MethodInfoV1)
    check(lib.n4m_method_info_v1(index, ctypes.addressof(info)), "n4m_method_info_v1")
    return info


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _check_int64(values: np.ndarray, name: str) -> None:
    """Refuses integers outside int64: the cast would wrap them silently."""
    if values.size == 0:
        return
    lo, hi = values.min(), values.max()
    if values.dtype.kind == "f":
        outside = hi >= 2.0**63 or lo < -(2.0**63)
    else:
        outside = int(hi) > _INT64_MAX or int(lo) < _INT64_MIN
    if outside:
        raise ValueError(f"{name} must fit int64; got values in [{lo}, {hi}]")


def _vector(values, name: str, length: int, what: str, dtype) -> np.ndarray:
    """1-D array of exactly ``length`` entries; never reshaped or broadcast."""
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape[0] != length:
        raise ValueError(
            f"{name} must be a 1-D array of length {length} ({what}); got shape {arr.shape}"
        )
    if dtype is np.int64:
        whole = arr.dtype.kind in "iub" or (
            arr.dtype.kind == "f" and np.all(np.isfinite(arr) & (arr == np.round(arr)))
        )
        if not whole:
            raise ValueError(f"{name} must contain integers")
        _check_int64(arr, name)
    return np.ascontiguousarray(arr, dtype=dtype)


def _targets(y, n_rows: int) -> np.ndarray:
    """``y`` as an (n_rows, n_targets) matrix: a 1-D target of length n_rows
    becomes one column, a matrix must have n_rows rows. Nothing else is
    reshaped: a transposed or flattened target is refused."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == n_rows:
        arr = arr.reshape(n_rows, 1)
    elif arr.ndim != 2 or arr.shape[0] != n_rows or arr.shape[1] == 0:
        raise ValueError(
            f"y must have shape ({n_rows},) or ({n_rows}, n_targets), one row per row of X; "
            f"got shape {arr.shape}"
        )
    return np.ascontiguousarray(arr)


def _label_array(y) -> np.ndarray:
    """Labels as an array. A Python list is read element by element: NumPy would
    turn integers beyond int64 into floats, merging distinct labels."""
    return (
        np.asarray(y, dtype=object) if isinstance(y, (list, tuple)) else np.asarray(y)
    )


def _class_labels(y) -> np.ndarray:
    """Class labels as a 1-D array (a one-column matrix is accepted)."""
    labels = _label_array(y)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        raise ValueError(
            f"class labels must be a 1-D array, one per row of X; got shape {labels.shape}"
        )
    return labels


def _check_exact_float(values: list[Any], name: str) -> None:
    """Numbers that must stay exact as float64 (JSON, JS and R numbers are doubles).

    An integer beyond ±2^53 would be rounded, merging distinct labels, so it
    is refused.
    """
    for v in values:
        if isinstance(v, numbers.Integral) and abs(int(v)) > 2**53:
            raise ValueError(
                f"{name} must be exactly representable as float64; {v} is beyond ±2^53"
            )


def _object_labels(labels: np.ndarray) -> np.ndarray:
    """Labels held as Python objects, as strings, integers or floats."""
    values = labels.tolist()
    numeric = all(
        isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))
        for v in values
    )
    # Among strings, NaN is how pandas marks a missing label.
    if any(
        v is None or (not numeric and isinstance(v, float) and math.isnan(v))
        for v in values
    ):
        raise ValueError("class labels must not be missing (None or NaN)")
    if all(isinstance(v, str) for v in values):
        return np.asarray(values, dtype=str)
    if not numeric:
        raise ValueError(
            "class labels must be all integers, all numbers or all strings "
            "(no booleans, no mix)"
        )
    if all(isinstance(v, numbers.Integral) for v in values):
        _check_int64(np.asarray(values, dtype=object), "class labels")
        return np.asarray(values, dtype=np.int64)
    _check_exact_float(values, "class labels mixing integers and fractions")
    return np.asarray(values, dtype=np.float64)


def _encode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Class ids for the core and the label table they index (shared label contract).

    Integer labels are the ids themselves (no table) and must fit int64: they
    are never wrapped. Strings and non-integer numbers become the table of
    sorted unique labels, the ids their positions. Missing, non-finite and
    boolean labels are refused.
    """
    if labels.dtype.kind == "O":
        labels = _object_labels(labels)
    kind = labels.dtype.kind
    if kind in "iu":
        _check_int64(labels, "class labels")
        return np.ascontiguousarray(labels, dtype=np.int64), None
    if kind == "f" and not np.all(np.isfinite(labels)):
        raise ValueError("class labels must be finite (no NaN or infinity)")
    if kind not in "fU":
        raise ValueError(
            f"class labels must be integers, finite numbers or strings; got dtype {labels.dtype}"
        )
    names, codes = np.unique(labels, return_inverse=True)
    return np.ascontiguousarray(codes, dtype=np.int64), names


def _fit_inputs(
    keep: list[Any],
    X,
    y=None,
    *,
    labels=None,
    sample_weight=None,
    groups=None,
    feature_groups=None,
    blocks=None,
    axis=None,
    X_target=None,
    fold_ids=None,
) -> FitInputsV1:
    """``n4m_fit_inputs_v1_t`` over the given data; ``keep`` holds the buffers.

    Per-row inputs (y, labels, sample_weight, groups, fold_ids) must have one
    entry per row of X and per-column inputs (feature_groups, axis) one per
    column: they are checked here, before any reshaping, and again natively.
    """
    X_arr = as_f64_2d(X)
    X_view = numpy_to_view(X_arr)
    keep += [X_arr, X_view]
    n_rows, n_cols = X_arr.shape
    inputs = FitInputsV1()
    inputs.struct_size = ctypes.sizeof(FitInputsV1)
    inputs.X = ctypes.addressof(X_view)
    if labels is not None:
        labels = _vector(labels, "labels", n_rows, "one per row of X", np.int64)
        keep.append(labels)
        inputs.labels, inputs.n_labels = labels.ctypes.data, labels.size
    if y is not None:
        y_arr = _targets(y, n_rows)
        y_view = numpy_to_view(y_arr)
        keep += [y_arr, y_view]
        inputs.Y = ctypes.addressof(y_view)
    if sample_weight is not None:
        w = _vector(
            sample_weight, "sample_weight", n_rows, "one per row of X", np.float64
        )
        keep.append(w)
        inputs.sample_weight, inputs.n_sample_weight = w.ctypes.data, w.size
    for name, value, length, what, ptr_field, len_field in (
        ("groups", groups, n_rows, "one per row of X", "groups", "n_groups"),
        (
            "feature_groups",
            feature_groups,
            n_cols,
            "one per column of X",
            "feature_groups",
            "n_feature_groups",
        ),
        ("fold_ids", fold_ids, n_rows, "one per row of X", "fold_ids", "n_fold_ids"),
    ):
        if value is not None:
            arr = _vector(value, name, length, what, np.int64)
            keep.append(arr)
            setattr(inputs, ptr_field, arr.ctypes.data)
            setattr(inputs, len_field, arr.size)
    if blocks is not None:
        sizes = _vector(blocks, "blocks", np.size(blocks), "block sizes", np.int64)
        keep.append(sizes)
        inputs.block_sizes, inputs.n_blocks = sizes.ctypes.data, sizes.size
    if axis is not None:
        a = _vector(axis, "axis", n_cols, "one per column of X", np.float64)
        keep.append(a)
        inputs.axis, inputs.n_axis = a.ctypes.data, a.size
    if X_target is not None:
        t_arr = as_f64_2d(X_target)
        t_view = numpy_to_view(t_arr)
        keep += [t_arr, t_view]
        inputs.X_target = ctypes.addressof(t_view)
    return inputs


class NativeMethod(BaseEstimator):
    """Parameters and manifest of one catalog method (estimator or procedure).

    Subclasses declare ``_method_id`` and ``_param_types`` (parameter name to
    manifest type) and an explicit ``__init__`` so scikit-learn can clone them.
    """

    _method_id: ClassVar[str] = ""
    _param_types: ClassVar[dict[str, str]] = {}
    _enum_choices: ClassVar[dict[str, tuple[str, ...]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls._method_id:
            _REGISTRY[cls._method_id] = cls
        # Generated classes are public as n4m.roles.<Name>: that path is the
        # stable operator token pipelines serialize.
        if cls.__module__ == "n4m.roles._generated":
            cls.__module__ = "n4m.roles"

    @classmethod
    def input_requirements(cls) -> dict[str, str]:
        """Fit input name -> "required" / "optional" / "none", from the manifest."""
        levels = ("none", "optional", "required")
        info = method_info(cls._method_id)
        return {name: levels[info.inputs[i]] for i, name in enumerate(_FIT_INPUT_NAMES)}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = self.input_requirements()["y"] == "required"
        return tags

    # -- construction -------------------------------------------------------

    def _native_params(self, ctx: _Context) -> ctypes.c_void_p:
        index = ctypes.c_int32()
        check(
            lib.n4m_method_find(self._method_id.encode(), ctypes.byref(index)),
            self._method_id,
        )
        params = ctypes.c_void_p()
        ctx.check(
            lib.n4m_params_create(ctx.handle, index, ctypes.byref(params)),
            "n4m_params_create",
        )
        try:
            for name, kind in self._param_types.items():
                value = getattr(self, name)
                if value is None:
                    continue
                key = name.encode()
                if kind == "int":
                    status = lib.n4m_params_set_int(
                        params, key, ctypes.c_int64(int(value))
                    )
                elif kind == "double":
                    status = lib.n4m_params_set_double(
                        params, key, ctypes.c_double(float(value))
                    )
                elif kind == "bool":
                    status = lib.n4m_params_set_bool(
                        params, key, ctypes.c_int32(1 if value else 0)
                    )
                elif kind == "enum":
                    status = lib.n4m_params_set_enum(params, key, str(value).encode())
                elif kind == "int_array":
                    arr = np.ascontiguousarray(value, dtype=np.int64).reshape(-1)
                    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
                    status = lib.n4m_params_set_int_array(
                        params, key, ptr, ctypes.c_int64(arr.size)
                    )
                elif kind == "double_array":
                    arr = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
                    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                    status = lib.n4m_params_set_double_array(
                        params, key, ptr, ctypes.c_int64(arr.size)
                    )
                else:
                    raise TypeError(f"unsupported parameter type {kind!r}")
                if status != Status.OK:
                    raise ValueError(
                        f"{type(self).__name__}: invalid value {value!r} for parameter {name!r}"
                    )
            ctx.check(
                lib.n4m_params_validate(ctx.handle, params), "n4m_params_validate"
            )
        except BaseException:
            lib.n4m_params_destroy(params)
            raise
        return params


class NativeEstimator(NativeMethod):
    """Shared life cycle of the generated :mod:`n4m.roles` estimators.

    It owns parameters, fitting and the N4ME state only. Operations belong to
    typed role interfaces (:class:`NativeRegressor`, :class:`NativeTransformer`,
    ...); a generated class inherits exactly the roles its native method
    declares, so a pure regressor has no ``transform`` and vice versa.
    Optional fit inputs (``feature_groups``, ``blocks``, ``X_target``,
    ``sample_weight``, ``groups``, ``axis``, ``fold_ids``) are fit keywords, as
    they are data, not hyperparameters.
    """

    def fit(
        self,
        X,
        y=None,
        *,
        sample_weight=None,
        groups=None,
        feature_groups=None,
        blocks=None,
        axis=None,
        X_target=None,
        fold_ids=None,
    ):
        """Fit the native estimator; unused inputs are refused by the core.

        The fitted state (native handle, class names, output shape, feature
        names) is replaced only when the fit succeeds: a failed refit leaves
        the previously fitted estimator unchanged and usable.
        """
        labels = label_names = None
        if y is not None and isinstance(self, NativeClassifier):
            (labels, label_names), y = _encode_labels(_class_labels(y)), None
        # A single prediction column comes back 1-D unless y was a one-column
        # matrix (a survival (time, event) response gives one risk column).
        y_1d = y is not None and not (np.ndim(y) == 2 and np.shape(y)[1] == 1)
        keep: list[Any] = []
        inputs = _fit_inputs(
            keep,
            X,
            y,
            labels=labels,
            sample_weight=sample_weight,
            groups=groups,
            feature_groups=feature_groups,
            blocks=blocks,
            axis=axis,
            X_target=X_target,
            fold_ids=fold_ids,
        )
        with _Context() as ctx:
            params = self._native_params(ctx)
            handle = ctypes.c_void_p()
            try:
                ctx.check(
                    lib.n4m_estimator_create(
                        ctx.handle,
                        self._method_id.encode(),
                        params,
                        ctypes.byref(handle),
                    ),
                    "n4m_estimator_create",
                )
            finally:
                lib.n4m_params_destroy(params)
            status = lib.n4m_estimator_fit(ctx.handle, handle, ctypes.addressof(inputs))
            if status != Status.OK:
                lib.n4m_estimator_destroy(handle)
                ctx.check(status, f"{type(self).__name__}.fit")
        del keep
        self._set_handle(handle)
        self._y_1d_ = y_1d
        if isinstance(self, NativeClassifier):
            self._label_names_ = label_names
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            self.__dict__.pop("feature_names_in_", None)
        return self

    def _set_handle(self, handle: ctypes.c_void_p) -> None:
        self._release()
        self._handle_ = handle
        index = ctypes.c_int32()
        caps = ctypes.c_uint64()
        check(
            lib.n4m_estimator_info(handle, ctypes.byref(index), ctypes.byref(caps)),
            "n4m_estimator_info",
        )
        n_in = ctypes.c_int64()
        n_out = ctypes.c_int64()
        check(
            lib.n4m_estimator_n_features_in(handle, ctypes.byref(n_in)),
            "n4m_estimator_n_features_in",
        )
        check(
            lib.n4m_estimator_n_outputs(handle, ctypes.byref(n_out)),
            "n4m_estimator_n_outputs",
        )
        self.capabilities_ = int(caps.value)
        self.n_features_in_ = int(n_in.value)
        self.n_outputs_ = int(n_out.value)

    # -- inference ------------------------------------------------------------

    def _handle(self) -> ctypes.c_void_p:
        handle = getattr(self, "_handle_", None)
        if handle is None:
            raise N4MError(
                Status.ERR_NOT_FITTED, f"{type(self).__name__} is not fitted"
            )
        return handle

    def _matrix_call(self, symbol: str, X, n_cols: int) -> np.ndarray:
        handle = self._handle()
        X_arr = as_f64_2d(X)
        out = np.empty((X_arr.shape[0], n_cols), dtype=np.float64)
        X_view = numpy_to_view(X_arr)
        out_view = numpy_to_view(out)
        with _Context() as ctx:
            ctx.check(
                getattr(lib, symbol)(
                    ctx.handle, handle, ctypes.byref(X_view), ctypes.byref(out_view)
                ),
                symbol,
            )
        return out

    def _n_outputs(self) -> int:
        n = ctypes.c_int64()
        check(
            lib.n4m_estimator_n_outputs(self._handle(), ctypes.byref(n)),
            "n4m_estimator_n_outputs",
        )
        return int(n.value)

    # -- N4ME fitted state ----------------------------------------------------

    @property
    def contains_training_rows_(self) -> bool:
        """True when the fitted state embeds training rows (kernel PLS,
        GPR-PLS, LW-PLS, ...): exporting it shares training data."""
        out = ctypes.c_int32()
        check(
            lib.n4m_estimator_contains_training_rows(self._handle(), ctypes.byref(out)),
            "n4m_estimator_contains_training_rows",
        )
        return bool(out.value)

    def to_n4me(self, *, allow_training_rows: bool = False) -> bytes:
        """Portable fitted state readable by every n4m binding.

        A state that embeds training rows (``contains_training_rows_``) is
        refused unless ``allow_training_rows=True``: sharing the export shares
        those rows. Pickling is an in-process checkpoint of the live object and
        keeps them, like the object does.
        """
        handle = self._handle()
        flags = ctypes.c_uint32(1 if allow_training_rows else 0)
        size = ctypes.c_size_t()
        with _Context() as ctx:
            ctx.check(
                lib.n4m_estimator_export_size(
                    ctx.handle, handle, flags, ctypes.byref(size)
                ),
                "export",
            )
            buf = (ctypes.c_ubyte * size.value)()
            written = ctypes.c_size_t()
            ctx.check(
                lib.n4m_estimator_export_to_buffer(
                    ctx.handle,
                    handle,
                    flags,
                    ctypes.addressof(buf),
                    size,
                    ctypes.byref(written),
                ),
                "export",
            )
        return bytes(buf)[: written.value]

    @staticmethod
    def from_n4me(payload: bytes) -> NativeEstimator:
        """Rebuild a fitted estimator (of the class registered for its method)."""
        data = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        handle = ctypes.c_void_p()
        with _Context() as ctx:
            ctx.check(
                lib.n4m_estimator_import_from_buffer(
                    ctx.handle,
                    ctypes.addressof(data),
                    len(payload),
                    ctypes.byref(handle),
                ),
                "import",
            )
        try:
            index = ctypes.c_int32()
            caps = ctypes.c_uint64()
            check(
                lib.n4m_estimator_info(handle, ctypes.byref(index), ctypes.byref(caps)),
                "n4m_estimator_info",
            )
            info = MethodInfoV1()
            info.struct_size = ctypes.sizeof(MethodInfoV1)
            check(
                lib.n4m_method_info_v1(index, ctypes.addressof(info)),
                "n4m_method_info_v1",
            )
            cls = _REGISTRY[info.method_id.decode()]
            est = cls.__new__(cls)
            est.__dict__.update(_native_param_values(handle, cls))
        except BaseException:
            lib.n4m_estimator_destroy(handle)
            raise
        est._set_handle(handle)
        # N4ME does not record the caller's target shape: single-output
        # estimators predict 1-D arrays, as after fitting on a 1-D y.
        est._y_1d_ = est.n_outputs_ == 1
        return est

    def __getstate__(self) -> dict[str, Any]:
        state = {k: v for k, v in self.__dict__.items() if k != "_handle_"}
        if getattr(self, "_handle_", None) is not None:
            state["_n4me_"] = self.to_n4me(allow_training_rows=True)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        payload = state.pop("_n4me_", None)
        self.__dict__.update(state)
        self._handle_ = None
        if payload is not None:
            restored = NativeEstimator.from_n4me(payload)
            handle, restored._handle_ = restored._handle_, None
            self._set_handle(handle)

    def _release(self) -> None:
        handle = self.__dict__.get("_handle_")
        if handle is not None:
            lib.n4m_estimator_destroy(handle)
            self._handle_ = None

    def __del__(self) -> None:
        self._release()


class NativeRegressor(RegressorMixin, NativeEstimator):
    """Regressor role: Data[n, p] + Target[n, q] -> Prediction[n, q]."""

    def predict(self, X) -> np.ndarray:
        """Native out-of-sample prediction."""
        out = self._matrix_call("n4m_estimator_predict", X, self._n_outputs())
        return (
            out.ravel() if getattr(self, "_y_1d_", False) and out.shape[1] == 1 else out
        )


class NativeTransformer(TransformerMixin, NativeEstimator):
    """Transformer role: Data[n, p] (+ Target) -> Data[n, k]."""

    def transform(self, X) -> np.ndarray:
        """Native out-of-sample transform (for PLS-like methods, latent scores)."""
        cols = ctypes.c_int64()
        check(
            lib.n4m_estimator_transform_cols(self._handle(), ctypes.byref(cols)),
            "n4m_estimator_transform_cols",
        )
        return self._matrix_call("n4m_estimator_transform", X, int(cols.value))


def _native_param_values(
    handle: ctypes.c_void_p, cls: type[NativeEstimator]
) -> dict[str, Any]:
    """Resolved parameter values stored in a fitted native estimator."""
    params = ctypes.c_void_p()
    with _Context() as ctx:
        ctx.check(
            lib.n4m_estimator_get_params(ctx.handle, handle, ctypes.byref(params)),
            "get_params",
        )
    try:
        values: dict[str, Any] = {}
        for name, kind in cls._param_types.items():
            key = name.encode()
            count = ctypes.c_int64()
            if kind in ("double", "double_array"):
                check(
                    lib.n4m_params_get_double(
                        params, key, None, 0, ctypes.byref(count)
                    ),
                    name,
                )
                buf = (ctypes.c_double * max(count.value, 1))()
                check(
                    lib.n4m_params_get_double(
                        params, key, buf, count, ctypes.byref(count)
                    ),
                    name,
                )
                vals: list[Any] = list(buf)[: count.value]
            else:
                check(
                    lib.n4m_params_get_int(params, key, None, 0, ctypes.byref(count)),
                    name,
                )
                buf_i = (ctypes.c_int64 * max(count.value, 1))()
                check(
                    lib.n4m_params_get_int(
                        params, key, buf_i, count, ctypes.byref(count)
                    ),
                    name,
                )
                vals = list(buf_i)[: count.value]
            if kind == "bool":
                values[name] = bool(vals[0])
            elif kind == "enum":
                values[name] = cls._enum_choices[name][vals[0]]
            elif kind.endswith("_array"):
                values[name] = tuple(vals)  # as the generated array defaults
            elif kind == "double" and math.isnan(vals[0]):
                values[name] = None  # NaN marks an unused optional value
            else:
                values[name] = vals[0]
        return values
    finally:
        lib.n4m_params_destroy(params)


class NativeSampleFilter(NativeEstimator):
    """Sample-filter role: Data[n, p] (+ Target) -> keep mask[n], train only.

    ``get_mask`` follows the nirs4all ``SampleFilter`` contract (True keeps
    the row). Filters on the target read ``y`` at fit and in ``get_mask``;
    the others ignore it.
    """

    def get_mask(self, X, y=None) -> np.ndarray:
        """Boolean keep mask of the rows of ``X``."""
        handle = self._handle()
        X_arr = as_f64_2d(X)
        X_view = numpy_to_view(X_arr)
        y_ref = None
        if y is not None:
            y_arr = _targets(y, X_arr.shape[0])
            y_view = numpy_to_view(y_arr)
            y_ref = ctypes.byref(y_view)
        out = np.empty(X_arr.shape[0], dtype=np.uint8)
        with _Context() as ctx:
            ctx.check(
                lib.n4m_estimator_apply_mask(
                    ctx.handle,
                    handle,
                    ctypes.byref(X_view),
                    y_ref,
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    ctypes.c_int64(out.size),
                ),
                "n4m_estimator_apply_mask",
            )
        return out.astype(bool)


class NativeSelector(SelectorMixin, NativeEstimator):
    """Selector role: Data[n, p] (+ Target) -> Data[n, k] with k selected input columns.

    ``selected_indices_`` keeps the native selection order (rank or pick
    order); ``transform`` returns the selected columns in ascending input
    order, as every n4m binding does.
    """

    @property
    def selected_indices_(self) -> np.ndarray:
        handle = self._handle()
        count = ctypes.c_int64()
        check(
            lib.n4m_estimator_selected_indices(handle, None, 0, ctypes.byref(count)),
            "selected_indices",
        )
        out = np.empty(count.value, dtype=np.int64)
        check(
            lib.n4m_estimator_selected_indices(
                handle,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                count,
                ctypes.byref(count),
            ),
            "selected_indices",
        )
        return out

    def _get_support_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def transform(self, X) -> np.ndarray:
        """Selected columns, computed natively."""
        cols = ctypes.c_int64()
        check(
            lib.n4m_estimator_transform_cols(self._handle(), ctypes.byref(cols)),
            "n4m_estimator_transform_cols",
        )
        return self._matrix_call("n4m_estimator_transform", X, int(cols.value))


class NativeClassifier(ClassifierMixin, NativeEstimator):
    """Classifier role: Data[n, p] + Labels[n] -> labels, decision scores, probabilities.

    The core works on integer class ids; labels of another type (strings,
    ...) are encoded here and ``classes_`` restores them. ``predict_proba``
    exists only for methods that define probabilities.
    """

    @property
    def classes_(self) -> np.ndarray:
        ids = self._native_classes()
        names = getattr(self, "_label_names_", None)
        return ids if names is None else names[ids]

    def _native_classes(self) -> np.ndarray:
        handle = self._handle()
        count = ctypes.c_int64()
        check(
            lib.n4m_estimator_classes(handle, None, 0, ctypes.byref(count)), "classes"
        )
        out = np.empty(count.value, dtype=np.int64)
        check(
            lib.n4m_estimator_classes(
                handle,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                count,
                ctypes.byref(count),
            ),
            "classes",
        )
        return out

    def predict(self, X) -> np.ndarray:
        """Native class labels."""
        handle = self._handle()
        X_arr = as_f64_2d(X)
        out = np.empty(X_arr.shape[0], dtype=np.int64)
        X_view = numpy_to_view(X_arr)
        with _Context() as ctx:
            ctx.check(
                lib.n4m_estimator_predict_labels(
                    ctx.handle,
                    handle,
                    ctypes.byref(X_view),
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    ctypes.c_int64(out.size),
                ),
                "n4m_estimator_predict_labels",
            )
        names = getattr(self, "_label_names_", None)
        return out if names is None else names[out]

    def decision_function(self, X) -> np.ndarray:
        """Method-defined class scores (for example log posteriors up to a constant)."""
        return self._matrix_call(
            "n4m_estimator_decision_function", X, self._n_outputs()
        )

    @available_if(
        lambda self: bool(method_info(self._method_id).capabilities & CAP_PREDICT_PROBA)
    )
    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, for methods that define them."""
        return self._matrix_call("n4m_estimator_predict_proba", X, self._n_outputs())


class _ProcedureBase(NativeMethod):
    """A catalog procedure: one native run, no fitted state."""

    def _call(self, read, X, y=None, **inputs):
        """Run the procedure and hand the native result to ``read``."""
        keep: list[Any] = []
        fit_inputs = _fit_inputs(keep, X, y, **inputs)
        index = ctypes.c_int32()
        check(
            lib.n4m_method_find(self._method_id.encode(), ctypes.byref(index)),
            self._method_id,
        )
        result = ctypes.c_void_p()
        with _Context() as ctx:
            params = self._native_params(ctx)
            try:
                ctx.check(
                    lib.n4m_procedure_run(
                        ctx.handle,
                        index,
                        params,
                        ctypes.addressof(fit_inputs),
                        ctypes.byref(result),
                    ),
                    type(self).__name__,
                )
            finally:
                lib.n4m_params_destroy(params)
        del keep
        try:
            return read(result)
        finally:
            lib.n4m_method_result_destroy(result)


def _result_entry(result: ctypes.c_void_p, name: bytes, kind: int):
    if kind == 0:
        data = ctypes.POINTER(ctypes.c_double)()
        rows, cols = ctypes.c_int64(), ctypes.c_int64()
        check(
            lib.n4m_method_result_get_double_matrix(
                result, name, ctypes.byref(data), ctypes.byref(rows), ctypes.byref(cols)
            ),
            name.decode(),
        )
        if rows.value * cols.value == 0:
            return np.empty((rows.value, cols.value))
        return (
            np.ctypeslib.as_array(data, (rows.value * cols.value,))
            .reshape(rows.value, cols.value)
            .copy()
        )
    if kind == 3:
        value = ctypes.c_double()
        check(
            lib.n4m_method_result_get_scalar(result, name, ctypes.byref(value)),
            name.decode(),
        )
        return value.value
    if kind == 1:
        data32 = ctypes.POINTER(ctypes.c_int32)()
        n32 = ctypes.c_int32()
        check(
            lib.n4m_method_result_get_int_vector(
                result, name, ctypes.byref(data32), ctypes.byref(n32)
            ),
            name.decode(),
        )
        return (
            np.ctypeslib.as_array(data32, (n32.value,)).copy()
            if n32.value
            else np.empty(0, np.int32)
        )
    data64 = ctypes.POINTER(ctypes.c_int64)()
    n64 = ctypes.c_int64()
    check(
        lib.n4m_method_result_get_int64_vector(
            result, name, ctypes.byref(data64), ctypes.byref(n64)
        ),
        name.decode(),
    )
    return (
        np.ctypeslib.as_array(data64, (n64.value,)).copy()
        if n64.value
        else np.empty(0, np.int64)
    )


def _result_dict(result: ctypes.c_void_p) -> dict[str, Any]:
    count = ctypes.c_int32()
    check(lib.n4m_method_result_entry_count(result, ctypes.byref(count)), "entry_count")
    out: dict[str, Any] = {}
    for i in range(count.value):
        name = ctypes.c_char_p()
        kind = ctypes.c_int32()
        check(
            lib.n4m_method_result_entry(
                result, i, ctypes.byref(name), ctypes.byref(kind)
            ),
            "entry",
        )
        out[name.value.decode()] = _result_entry(result, name.value, kind.value)
    return out


class NativeProcedure(_ProcedureBase):
    """Generic procedure (diagnostics, utilities): inputs -> named outputs."""

    def run(self, X, y=None, **inputs) -> dict[str, Any]:
        """Named outputs of the native function (arrays or floats)."""
        return self._call(_result_dict, X, y, **inputs)


class NativeSplitter(_ProcedureBase):
    """Splitter role: Data[n, p] (+ Target, groups) -> folds of row indices.

    A scikit-learn cross-validator: ``split`` yields ``(train, test)`` index
    arrays, so an instance can be passed as ``cv=``.
    """

    def _folds(self, X, y=None, groups=None) -> list[tuple[np.ndarray, np.ndarray]]:
        inputs = {} if groups is None else {"groups": groups}

        def read(result):
            n = ctypes.c_int32()
            check(lib.n4m_method_result_get_n_folds(result, ctypes.byref(n)), "n_folds")
            folds = []
            for k in range(n.value):
                tr, te = (
                    ctypes.POINTER(ctypes.c_int64)(),
                    ctypes.POINTER(ctypes.c_int64)(),
                )
                ntr, nte = ctypes.c_int64(), ctypes.c_int64()
                check(
                    lib.n4m_method_result_get_fold(
                        result,
                        k,
                        ctypes.byref(tr),
                        ctypes.byref(ntr),
                        ctypes.byref(te),
                        ctypes.byref(nte),
                    ),
                    "fold",
                )
                folds.append(
                    tuple(
                        np.ctypeslib.as_array(p, (m.value,)).copy()
                        if m.value
                        else np.empty(0, np.int64)
                        for p, m in ((tr, ntr), (te, nte))
                    )
                )
            return folds

        return self._call(read, X, y, **inputs)

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` zero-based row indices for each fold."""
        yield from self._folds(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of folds for this data (the native splitter decides)."""
        if X is None:
            raise ValueError(f"{type(self).__name__}.get_n_splits needs X")
        return len(self._folds(X, y, groups))


class NativeAugmenter(_ProcedureBase):
    """Augmenter role: Data[n, p] (+ Target) -> augmented Data[n, p], train only.

    Methods that mix rows (mixup) require ``y`` and return it mixed with the
    same draw, row for row.
    """

    def augment(self, X, y=None, *, axis=None):
        """Augmented rows, or ``(X, y)`` augmented for the methods that mix targets.

        Seeds are parameters, so a run is reproducible. The mixed targets keep
        the dimensionality of ``y``.
        """
        inputs = {} if axis is None else {"axis": axis}

        def read(result):
            out = _result_dict(result)
            if "Y" not in out:
                return out["X"]
            return out["X"], out["Y"].reshape(np.shape(y))

        return self._call(read, X, y, **inputs)


__all__ = [
    "NativeAugmenter",
    "NativeClassifier",
    "NativeEstimator",
    "NativeMethod",
    "NativeProcedure",
    "NativeRegressor",
    "NativeSampleFilter",
    "NativeSelector",
    "NativeSplitter",
    "NativeTransformer",
    "manifest",
    "method_class",
    "method_info",
]
