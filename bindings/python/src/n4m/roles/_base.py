# SPDX-License-Identifier: CECILL-2.1
"""scikit-learn facade over the generic native estimator roles (ABI 2.13).

Every class in :mod:`n4m.roles` is a thin, generated subclass of
:class:`NativeEstimator`: parameters, defaults, required fit inputs, fitting,
prediction, transformation and the N4ME fitted state all live in libn4m.
This module only translates Python objects to ``n4m_estimator_*`` calls.
"""

from __future__ import annotations

import ctypes
from typing import Any, ClassVar, Self

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.feature_selection import SelectorMixin

from .._errors import N4MError, check
from .._ffi import lib
from .._matrix import as_f64_2d, numpy_to_view
from .._types import FitInputsV1, MethodInfoV1, Status

CAP_TRANSFORM = 1 << 0
CAP_PREDICT = 1 << 1

_REGISTRY: dict[str, type[NativeEstimator]] = {}


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


def _as_int64(values, name: str) -> np.ndarray:
    arr = np.ascontiguousarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


class NativeEstimator(BaseEstimator):
    """Shared life cycle of the generated :mod:`n4m.roles` estimators.

    It owns parameters, fitting and the N4ME state only. Operations belong to
    typed role interfaces (:class:`NativeRegressor`, :class:`NativeTransformer`,
    ...); a generated class inherits exactly the roles its native method
    declares, so a pure regressor has no ``transform`` and vice versa.

    Subclasses declare ``_method_id`` and ``_param_types`` (parameter name to
    manifest type) and an explicit ``__init__`` so scikit-learn can clone them.
    Optional fit inputs (``feature_groups``, ``blocks``, ``X_target``,
    ``sample_weight``, ``groups``, ``axis``, ``fold_ids``) are fit keywords, as
    they are data, not hyperparameters.
    """

    _method_id: ClassVar[str] = ""
    _param_types: ClassVar[dict[str, str]] = {}
    _enum_choices: ClassVar[dict[str, tuple[str, ...]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls._method_id:
            _REGISTRY[cls._method_id] = cls

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

    # -- fit ----------------------------------------------------------------

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
        """Fit the native estimator; unused inputs are refused by the core."""
        X_arr = as_f64_2d(X)
        keep: list[Any] = [X_arr]
        inputs = FitInputsV1()
        inputs.struct_size = ctypes.sizeof(FitInputsV1)
        X_view = numpy_to_view(X_arr)
        inputs.X = ctypes.addressof(X_view)
        self._y_1d_ = False
        if y is not None:
            y_arr = np.asarray(y, dtype=np.float64)
            self._y_1d_ = y_arr.ndim == 1
            y_arr = np.ascontiguousarray(y_arr.reshape(X_arr.shape[0], -1))
            y_view = numpy_to_view(y_arr)
            keep += [y_arr, y_view]
            inputs.Y = ctypes.addressof(y_view)
        if sample_weight is not None:
            w = np.ascontiguousarray(sample_weight, dtype=np.float64).reshape(-1)
            keep.append(w)
            inputs.sample_weight, inputs.n_sample_weight = w.ctypes.data, w.size
        for name, value, ptr_field, len_field in (
            ("groups", groups, "groups", "n_groups"),
            ("feature_groups", feature_groups, "feature_groups", "n_feature_groups"),
            ("blocks", blocks, "block_sizes", "n_blocks"),
            ("fold_ids", fold_ids, "fold_ids", "n_fold_ids"),
        ):
            if value is not None:
                arr = _as_int64(value, name)
                keep.append(arr)
                setattr(inputs, ptr_field, arr.ctypes.data)
                setattr(inputs, len_field, arr.size)
        if axis is not None:
            a = np.ascontiguousarray(axis, dtype=np.float64).reshape(-1)
            keep.append(a)
            inputs.axis, inputs.n_axis = a.ctypes.data, a.size
        if X_target is not None:
            t_arr = as_f64_2d(X_target)
            t_view = numpy_to_view(t_arr)
            keep += [t_arr, t_view]
            inputs.X_target = ctypes.addressof(t_view)

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
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
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

    def to_n4me(self, *, allow_training_rows: bool = False) -> bytes:
        """Portable fitted state readable by every n4m binding."""
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
                values[name] = vals
            else:
                values[name] = vals[0]
        return values
    finally:
        lib.n4m_params_destroy(params)


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


__all__ = [
    "NativeEstimator",
    "NativeRegressor",
    "NativeSelector",
    "NativeTransformer",
    "method_info",
]
